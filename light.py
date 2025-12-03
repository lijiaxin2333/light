#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结合亮度/宫格判定 + 光流相似度的方式，批量裁剪视频开头闪烁帧。

依赖:
    pip install opencv-python numpy

批处理用法:
    python light.py --source-dir ./videos --output-dir ./output \
        --window-seconds 0.5 --similarity-threshold 0.3

默认遍历 source-dir 下支持的格式（.mp4/.mov/.mkv/.avi），并将处理后文件以
「原名_fix.扩展名」写入 output 目录；可通过 --max-trim-seconds 限制最多
删除的时长（默认 1 秒），避免过度裁剪。除了光流相似度检测，还会基于
亮度占比与强竖/横分割线（2/4 宫格）来判定前景是否为合成图层。
"""

from __future__ import annotations

import argparse
import math
import pathlib
import shutil
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np


@dataclass
class FrameFeature:
    whiteness: float
    vertical_score: float
    horizontal_score: float
    vertical_peak: float
    horizontal_peak: float
    histogram: Optional[np.ndarray]


def compute_whiteness_ratio(frame: np.ndarray, threshold: int = 235) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray >= threshold))


def compute_grid_scores(frame: np.ndarray) -> Tuple[float, float, float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    edges = edges / 255.0

    h, w = edges.shape
    band = max(1, min(h, w) // 40)
    vertical_positions = [w // 2, w // 3, 2 * w // 3]
    horizontal_positions = [h // 2, h // 3, 2 * h // 3]

    def band_mean(column: bool, positions: List[int]) -> float:
        values = []
        for pos in positions:
            if column:
                left = max(0, pos - band)
                right = min(w, pos + band)
                if right > left:
                    values.append(float(np.mean(edges[:, left:right])))
            else:
                top = max(0, pos - band)
                bottom = min(h, pos + band)
                if bottom > top:
                    values.append(float(np.mean(edges[top:bottom, :])))
        return float(np.mean(values)) if values else 0.0

    vertical_score = band_mean(True, vertical_positions)
    horizontal_score = band_mean(False, horizontal_positions)

    vertical_profile = np.mean(edges, axis=0)
    horizontal_profile = np.mean(edges, axis=1)
    vertical_peak = float(np.max(vertical_profile)) if vertical_profile.size else 0.0
    horizontal_peak = float(np.max(horizontal_profile)) if horizontal_profile.size else 0.0
    return vertical_score, horizontal_score, vertical_peak, horizontal_peak


def compute_color_histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def analyze_video(
    video_path: pathlib.Path,
    analysis_seconds: float,
) -> Tuple[List[float], List[FrameFeature], Optional[np.ndarray], int, float]:
    """遍历视频，收集光流均值及前段帧的视觉特征，并生成参考直方图。"""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    magnitudes: List[float] = []
    features: List[FrameFeature] = []
    total_frames = 0
    prev_gray = None
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    analysis_limit = max(1, math.ceil(max(analysis_seconds, 0.0) * fps))
    tail_hist: Optional[np.ndarray] = None
    tail_count = 0

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(float(np.mean(mag)))
        prev_gray = gray

        hist = compute_color_histogram(frame)
        if len(features) < analysis_limit:
            whiteness = compute_whiteness_ratio(frame)
            vertical_score, horizontal_score, vertical_peak, horizontal_peak = compute_grid_scores(frame)
            features.append(
                FrameFeature(
                    whiteness=whiteness,
                    vertical_score=vertical_score,
                    horizontal_score=horizontal_score,
                    vertical_peak=vertical_peak,
                    horizontal_peak=horizontal_peak,
                    histogram=hist,
                )
            )
        else:
            if tail_hist is None:
                tail_hist = np.zeros_like(hist)
            tail_hist += hist
            tail_count += 1

    capture.release()
    reference_hist = None
    if tail_count > 0 and tail_hist is not None:
        reference_hist = tail_hist / float(tail_count)
    elif features:
        stacked = np.stack([f.histogram for f in features if f.histogram is not None])
        reference_hist = np.mean(stacked, axis=0) if stacked.size else None
    return magnitudes, features, reference_hist, total_frames, fps


def decide_trim_frames_heuristic(
    magnitudes: List[float],
    features: List[FrameFeature],
    reference_hist: Optional[np.ndarray],
    similarity_threshold: float,
    white_threshold: float,
    grid_threshold: float,
    grid_split_threshold: float,
    hist_threshold: float,
    flow_delta_ratio: float,
) -> int:
    """综合宫格/亮度/光流差异判断需要裁剪的前缀帧数。"""
    if not magnitudes or not features:
        return 0

    analysis_frames = min(len(features), len(magnitudes))
    baseline_region = magnitudes[analysis_frames:] or magnitudes
    baseline = float(np.median(baseline_region)) if baseline_region else 0.0
    eps = 1e-6
    ref_hist = reference_hist
    ref_norm = None
    if ref_hist is not None:
        ref_norm = np.linalg.norm(ref_hist) + eps

    trim_frames = 0
    for idx in range(analysis_frames):
        feature = features[idx]
        if idx >= len(magnitudes):
            break

        whiteness_flag = feature.whiteness >= white_threshold
        grid_flag = max(feature.vertical_score, feature.horizontal_score) >= grid_threshold
        grid_split_flag = (
            feature.vertical_peak >= grid_split_threshold or feature.horizontal_peak >= grid_split_threshold
        )

        similarity_flag = False
        next_idx = min(idx + 1, len(magnitudes) - 1)
        if next_idx > idx:
            m1 = magnitudes[idx]
            m2 = magnitudes[next_idx]
            similarity = min(m1, m2) / (max(m1, m2) + eps)
            similarity_flag = similarity < similarity_threshold

        hist_flag = False
        if ref_hist is not None and ref_norm is not None and feature.histogram is not None:
            hist_norm = np.linalg.norm(feature.histogram) + eps
            corr = float(np.dot(feature.histogram, ref_hist) / (hist_norm * ref_norm))
            hist_flag = corr < hist_threshold

        flow_delta_flag = False
        if baseline > 0:
            flow_delta_flag = abs(magnitudes[idx] - baseline) / (baseline + eps) >= flow_delta_ratio

        if (
            whiteness_flag
            or grid_flag
            or grid_split_flag
            or similarity_flag
            or flow_delta_flag
            or hist_flag
        ):
            trim_frames = idx + 1
            continue

        break

    return trim_frames


def export_video_without_prefix(
    src: pathlib.Path, dst: pathlib.Path, trim_frames: int, codec: str
) -> None:
    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        raise RuntimeError(f"无法重新打开视频: {src}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if trim_frames >= total_frames:
        raise RuntimeError("裁剪帧数超过或等于总帧数，放弃导出。")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))

    idx = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        if idx >= trim_frames:
            writer.write(frame)
        idx += 1

    capture.release()
    writer.release()


SUPPORTED_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}


def iter_video_files(source_dir: pathlib.Path, allowed_suffixes: Iterable[str]) -> Iterable[pathlib.Path]:
    allowed = {suffix.lower() for suffix in allowed_suffixes}
    for child in sorted(source_dir.iterdir()):
        if child.is_file() and child.suffix.lower() in allowed:
            yield child


def process_single_video(
    input_path: pathlib.Path,
    output_dir: pathlib.Path,
    window_seconds: float,
    similarity_threshold: float,
    white_threshold: float,
    grid_threshold: float,
    grid_split_threshold: float,
    hist_threshold: float,
    flow_delta_ratio: float,
    codec: str,
    dry_run: bool,
    max_trim_seconds: float,
) -> None:
    start_time = time.perf_counter()
    print(f"\n==== 处理 {input_path.name} (开始: {time.strftime('%Y-%m-%d %H:%M:%S')}) ====")
    magnitudes, features, reference_hist, frame_count, fps = analyze_video(
        input_path,
        analysis_seconds=window_seconds,
    )
    trim_frames = decide_trim_frames_heuristic(
        magnitudes=magnitudes,
        features=features,
        reference_hist=reference_hist,
        similarity_threshold=similarity_threshold,
        white_threshold=white_threshold,
        grid_threshold=grid_threshold,
        grid_split_threshold=grid_split_threshold,
        hist_threshold=hist_threshold,
        flow_delta_ratio=flow_delta_ratio,
    )
    max_trim_frames = max(0, math.floor(fps * max(0.0, max_trim_seconds)))
    if max_trim_frames:
        trim_frames = min(trim_frames, max_trim_frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{input_path.stem}_fix{input_path.suffix}"
    output_path = output_dir / output_name

    print(f"总帧数: {frame_count}")
    if magnitudes:
        preview_flow = min(int(math.ceil(fps * window_seconds)), len(magnitudes))
        print(f"前{preview_flow}个光流均值: {magnitudes[:preview_flow]}")
    if features:
        preview_features = min(int(math.ceil(fps * window_seconds)), len(features))
        whiteness_info = [round(f.whiteness, 3) for f in features[:preview_features]]
        grid_info = [
            (
                round(f.vertical_score, 3),
                round(f.horizontal_score, 3),
                round(f.vertical_peak, 3),
                round(f.horizontal_peak, 3),
            )
            for f in features[:preview_features]
        ]
        print(f"前{preview_features}帧白色占比: {whiteness_info}")
        print(f"前{preview_features}帧宫格评分 (局部/全局竖横): {grid_info}")

    if trim_frames == 0:
        print("未检测到明显闪帧。")
        if dry_run:
            return
        if input_path.resolve() == output_path.resolve():
            print("输入输出相同且无需裁剪，跳过写入。")
            return
        shutil.copy2(input_path, output_path)
        elapsed = time.perf_counter() - start_time
        print(f"直接复制原视频到 {output_path} (用时 {elapsed:.2f}s)")
        return

    print(f"检测到闪帧，计划裁掉前 {trim_frames} 帧。")
    if dry_run:
        print("Dry-run 模式，未写入新文件。")
        return

    export_video_without_prefix(
        input_path,
        output_path,
        trim_frames=trim_frames,
        codec=codec,
    )
    elapsed = time.perf_counter() - start_time
    print(f"输出完成: {output_path} (裁剪了 {trim_frames} 帧, 用时 {elapsed:.2f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量光流检测闪烁帧并裁剪。")
    parser.add_argument(
        "--source-dir",
        default=".",
        help="待处理视频所在目录 (默认当前目录)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="保存处理后视频的目录 (默认 ./output)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1,
        help="仅在前多少秒内检测与比较 (默认0.5秒)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.25,
        help="与后续相邻帧的光流相似度低于该阈值即视为异常，范围0-1 (默认0.3)",
    )
    parser.add_argument(
        "--white-threshold",
        type=float,
        default=0.45,
        help="亮度超过该占比视为叠图/白底，默认0.45",
    )
    parser.add_argument(
        "--grid-threshold",
        type=float,
        default=0.09,
        help="中央竖/横带边缘占比超过该值时判定为宫格，默认0.09",
    )
    parser.add_argument(
        "--grid-split-threshold",
        type=float,
        default=0.12,
        help="若检测到明显分割线（2/4宫格），超过该阈值直接裁剪，默认0.12",
    )
    parser.add_argument(
        "--hist-threshold",
        type=float,
        default=0.5,
        help="与稳态色彩直方图的相关性低于该值即视为异常，默认0.5",
    )
    parser.add_argument(
        "--flow-delta-ratio",
        type=float,
        default=0.4,
        help="与稳态光流差异超过 baseline*ratio 视为异常，默认0.4",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="写出视频的 fourcc，默认 mp4v (适合 .mp4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出分析结果，不写入新文件",
    )
    parser.add_argument(
        "--max-trim-seconds",
        type=float,
        default=0.8,
        help="单个视频最多可裁剪的秒数",
    )
    parser.add_argument(
        "--ext",
        default=",".join(sorted(SUPPORTED_SUFFIXES)),
        help="要处理的扩展名（以逗号分隔，需带点），默认 .avi,.mkv,.mov,.mp4",
    )
    return parser.parse_args()


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    source_dir = pathlib.Path(args.source_dir).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()

    exts = {ext.strip().lower() for ext in args.ext.split(",") if ext.strip()}
    if exts:
        supported_exts = exts
    else:
        supported_exts = SUPPORTED_SUFFIXES

    if not source_dir.exists():
        raise RuntimeError(f"目录不存在: {source_dir}")

    files = list(iter_video_files(source_dir, supported_exts))

    if not files:
        print(f"目录 {source_dir} 未找到匹配的文件 (支持: {', '.join(sorted(supported_exts))})")
        return

    print(f"共发现 {len(files)} 个视频，将逐一处理，输出目录: {output_dir}")
    for path in files:
        process_single_video(
            input_path=path,
            output_dir=output_dir,
            window_seconds=args.window_seconds,
            similarity_threshold=args.similarity_threshold,
            white_threshold=args.white_threshold,
            grid_threshold=args.grid_threshold,
            grid_split_threshold=args.grid_split_threshold,
            hist_threshold=args.hist_threshold,
            flow_delta_ratio=args.flow_delta_ratio,
            codec=args.codec,
            dry_run=args.dry_run,
            max_trim_seconds=args.max_trim_seconds,
        )
    total_elapsed = time.perf_counter() - total_start
    print(f"\n全部处理完成，总用时 {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()

