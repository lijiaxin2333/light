#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测并裁剪视频开头的宫格帧（四宫格/两宫格）。

只检测以下两种情况：
- 四宫格：画面中心有十字分割线（竖线+横线同时存在）
- 两宫格：画面中心只有一条分割线（竖线或横线）

其他情况一律不处理。

依赖:
    pip install opencv-python numpy
    brew install ffmpeg

用法:
    python light.py --source-dir ./videos --output-dir ./output
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np


@dataclass
class GridDetectionResult:
    """宫格检测结果"""
    is_grid: bool  # 是否是宫格帧
    vertical_score: float  # 竖线分数 (0-1)
    horizontal_score: float  # 横线分数 (0-1)
    grid_type: str  # "none", "two_h", "two_v", "four"


def detect_center_split_line(frame: np.ndarray, threshold: float = 0.6) -> GridDetectionResult:
    """
    检测画面中心是否有分割线。

    策略：
    1. 缩放至固定尺寸，分析中央竖/横带贯穿程度
    2. 若背景接近纯黑/纯白，先提取非背景区域，再判断线条是否存在
    3. 若背景非纯色，则退回到边缘检测

    threshold: 分割线需要贯穿的比例，默认0.6 表示贯穿 60% 以上
    """

    DETECT_SIZE = (320, 180)
    BG_RATIO_THRESHOLD = 0.7  # 背景判定（黑/白占比达到 70%）
    BG_TOLERANCE = 12
    MIN_CONTENT_RATIO = 0.05
    BAND_OCCUPANCY_THRESHOLD = 0.08  # 中央带中允许的前景占比

    small = cv2.resize(frame, DETECT_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    center_x = w // 2
    center_y = h // 2
    band_width = max(3, w // 30)
    band_height = max(3, h // 30)
    left = center_x - band_width // 2
    right = center_x + band_width // 2 + 1
    top = center_y - band_height // 2
    bottom = center_y + band_height // 2 + 1

    def compute_band_background_ratio(mask_band: np.ndarray, axis: int) -> float:
        """
        计算中央带有多少比例是“背景”(即缺少内容)。
        axis = 0 表示纵向统计（对行求平均）；axis = 1 表示横向统计（对列求平均）
        """
        if axis == 0:  # vertical band -> 按行统计
            occupancy = np.mean(mask_band, axis=1)
        else:  # horizontal band -> 按列统计
            occupancy = np.mean(mask_band, axis=0)
        return float(np.mean(occupancy <= BAND_OCCUPANCY_THRESHOLD))

    # 1) 判断是否为纯色背景
    white_ratio = np.mean(gray >= 255 - BG_TOLERANCE)
    black_ratio = np.mean(gray <= BG_TOLERANCE)
    background_value: Optional[int] = None
    if white_ratio >= BG_RATIO_THRESHOLD:
        background_value = 255
    elif black_ratio >= BG_RATIO_THRESHOLD:
        background_value = 0

    if background_value is not None:
        # 纯黑 / 纯白背景：先提取非背景区域
        mask = (np.abs(gray.astype(np.int16) - background_value) > BG_TOLERANCE).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_bool = mask > 0
        content_ratio = float(np.mean(mask_bool))
        if content_ratio < MIN_CONTENT_RATIO:
            # 几乎全是背景，不认为是宫格
            vertical_score = 0.0
            horizontal_score = 0.0
        else:
            vertical_band = mask_bool[:, left:right]
            horizontal_band = mask_bool[top:bottom, :]
            vertical_score = compute_band_background_ratio(vertical_band, axis=0)
            horizontal_score = compute_band_background_ratio(horizontal_band, axis=1)
    else:
        # 2) 普通背景：使用边缘检测
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 80, 200)

        vertical_band = edges[:, left:right]
        row_has_edge = np.any(vertical_band > 0, axis=1)
        vertical_score = float(np.mean(row_has_edge))

        horizontal_band = edges[top:bottom, :]
        col_has_edge = np.any(horizontal_band > 0, axis=0)
        horizontal_score = float(np.mean(col_has_edge))

    has_vertical = vertical_score >= threshold
    has_horizontal = horizontal_score >= threshold

    if has_vertical and has_horizontal:
        grid_type = "four"
        is_grid = True
    elif has_vertical:
        grid_type = "two_v"
        is_grid = True
    elif has_horizontal:
        grid_type = "two_h"
        is_grid = True
    else:
        grid_type = "none"
        is_grid = False

    return GridDetectionResult(
        is_grid=is_grid,
        vertical_score=vertical_score,
        horizontal_score=horizontal_score,
        grid_type=grid_type,
    )


def analyze_video_for_grid(
    video_path: pathlib.Path,
    max_check_seconds: float,
    grid_threshold: float,
) -> Tuple[int, int, float, List[GridDetectionResult]]:
    """
    分析视频前若干秒，检测宫格帧。
    
    返回: (需要裁剪的帧数, 总帧数, FPS, 每帧的检测结果)
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    max_check_frames = min(total_frames, max(1, math.ceil(max_check_seconds * fps)))
    
    results: List[GridDetectionResult] = []
    frame_idx = 0
    last_grid_idx = -1  # 最后一个宫格帧的索引
    
    while frame_idx < max_check_frames:
        ret, frame = capture.read()
        if not ret:
            break
        
        result = detect_center_split_line(frame, threshold=grid_threshold)
        results.append(result)
        
        if result.is_grid:
            last_grid_idx = frame_idx
        
        frame_idx += 1
    
    capture.release()
    
    # 裁剪策略：删除从开头到最后一个宫格帧（包含该帧）
    trim_frames = last_grid_idx + 1 if last_grid_idx >= 0 else 0
    
    return trim_frames, total_frames, fps, results


def export_video_without_prefix(
    video_path: pathlib.Path,
    output_path: pathlib.Path,
    trim_frames: int,
    codec: str,
) -> Tuple[float, float, int]:
    """导出视频（跳过前 trim_frames 帧），保留原始音轨。"""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    original_duration = total_frames / fps if fps > 0 else 0.0

    if trim_frames >= total_frames:
        capture.release()
        raise RuntimeError("裁剪帧数超过或等于总帧数，放弃导出。")

    start_seconds = trim_frames / fps if fps > 0 else 0.0
    output_duration = max(0.0, original_duration - start_seconds)

    temp_dir = tempfile.TemporaryDirectory(prefix="grid_trim_")
    temp_dir_path = pathlib.Path(temp_dir.name)
    temp_video_path = temp_dir_path / "video_only.mp4"
    temp_audio_path = temp_dir_path / "audio_track.m4a"

    # 写出视频（无音频）
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
    if trim_frames > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, trim_frames)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        writer.write(frame)

    capture.release()
    writer.release()

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("  警告: 未找到 ffmpeg，无法保留音频，输出将是无声视频。")
        shutil.copy2(temp_video_path, output_path)
        temp_dir.cleanup()
        return original_duration, output_duration, trim_frames

    def extract_audio(reencode: bool) -> bool:
        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            str(video_path),
        ]
        if start_seconds > 0:
            cmd += ["-ss", f"{start_seconds:.6f}"]
        cmd += ["-vn"]
        if reencode:
            cmd += [
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
            ]
        else:
            cmd += [
                "-c:a",
                "copy",
            ]
        cmd += [
            "-avoid_negative_ts",
            "1",
            str(temp_audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(
                f"  警告: 音轨提取失败({ '重编码' if reencode else '直接复制' }), "
                "尝试其它策略。"
            )
            return False
        return True

    audio_ok = extract_audio(reencode=True)
    if not audio_ok:
        audio_ok = extract_audio(reencode=False)
    if not audio_ok:
        print("  警告: 音轨提取失败，输出将是无声视频。")
        shutil.copy2(temp_video_path, output_path)
        temp_dir.cleanup()
        return original_duration, output_duration, trim_frames

    # 将音轨塞回视频
    mux_cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(temp_video_path),
        "-i",
        str(temp_audio_path),
        "-c",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(output_path),
    ]
    mux_result = subprocess.run(mux_cmd, capture_output=True)
    if mux_result.returncode != 0:
        print("  警告: 合并音轨失败，输出将是无声视频。")
        shutil.copy2(temp_video_path, output_path)
        temp_dir.cleanup()
        return original_duration, output_duration, trim_frames

    temp_dir.cleanup()
    return original_duration, output_duration, trim_frames


SUPPORTED_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}


def iter_video_files(source_dir: pathlib.Path, allowed_suffixes: Iterable[str]) -> Iterable[pathlib.Path]:
    allowed = {suffix.lower() for suffix in allowed_suffixes}
    for child in sorted(source_dir.iterdir()):
        if child.is_file() and child.suffix.lower() in allowed:
            yield child


def process_single_video(
    input_path: pathlib.Path,
    output_dir: pathlib.Path,
    max_check_seconds: float,
    grid_threshold: float,
    max_trim_seconds: float,
    codec: str,
    dry_run: bool,
) -> Optional[Tuple[str, float, float, int, float]]:
    """处理单个视频"""
    start_time = time.perf_counter()
    print(f"\n==== 处理 {input_path.name} ====")
    
    trim_frames, total_frames, fps, results = analyze_video_for_grid(
        input_path,
        max_check_seconds=max_check_seconds,
        grid_threshold=grid_threshold,
    )
    
    # 限制最大裁剪帧数
    max_trim_frames = max(0, math.floor(fps * max(0.0, max_trim_seconds)))
    if max_trim_frames > 0 and trim_frames > max_trim_frames:
        print(f"  裁剪帧数 {trim_frames} 超过上限 {max_trim_frames}，限制为上限值")
        trim_frames = max_trim_frames
    
    original_duration = total_frames / fps if fps > 0 else 0.0
    trim_duration = trim_frames / fps if fps > 0 else 0.0
    
    print(f"  总帧数: {total_frames}, FPS: {fps:.2f}, 时长: {original_duration:.2f}s")
    
    # 打印检测结果
    grid_frames = [i for i, r in enumerate(results) if r.is_grid]
    if grid_frames:
        print(f"  检测到宫格帧: {grid_frames}")
        for i in grid_frames[:5]:  # 只打印前5个
            r = results[i]
            print(f"    帧{i}: v={r.vertical_score:.2f}, h={r.horizontal_score:.2f}, type={r.grid_type}")
    else:
        print(f"  未检测到宫格帧")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{input_path.stem}_fix{input_path.suffix}"
    output_path = output_dir / output_name
    
    if trim_frames == 0:
        if dry_run:
            elapsed = time.perf_counter() - start_time
            print(f"  [Dry-run] 保持原样 (用时 {elapsed:.2f}s)")
            return (input_path.name, original_duration, original_duration, 0, 0.0)
        shutil.copy2(input_path, output_path)
        elapsed = time.perf_counter() - start_time
        print(f"  保持原样，复制到 {output_path} (用时 {elapsed:.2f}s)")
        return (input_path.name, original_duration, original_duration, 0, 0.0)
    
    print(f"  将裁掉前 {trim_frames} 帧 ({trim_duration:.2f}s)")
    
    if dry_run:
        elapsed = time.perf_counter() - start_time
        output_duration = original_duration - trim_duration
        print(f"  [Dry-run] 用时 {elapsed:.2f}s")
        return (input_path.name, original_duration, output_duration, trim_frames, trim_duration)
    
    original_dur, output_dur, actual_trim = export_video_without_prefix(
        input_path, output_path, trim_frames=trim_frames, codec=codec,
    )
    elapsed = time.perf_counter() - start_time
    actual_trim_dur = actual_trim / fps if fps > 0 else 0.0
    print(f"  输出: {output_path} (裁剪 {actual_trim} 帧, 用时 {elapsed:.2f}s)")
    return (input_path.name, original_dur, output_dur, actual_trim, actual_trim_dur)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检测并裁剪视频开头的宫格帧（四宫格/两宫格）")
    parser.add_argument("--source-dir", default=".", help="视频目录 (默认当前目录)")
    parser.add_argument("--output-dir", default="output", help="输出目录 (默认 ./output)")
    parser.add_argument("--max-check-seconds", type=float, default=1.2, help="最多检测开头多少秒 (默认2)")
    parser.add_argument("--grid-threshold", type=float, default=0.5, help="分割线贯穿比例阈值，越高越严格 (默认0.6)")
    parser.add_argument("--max-trim-seconds", type=float, default=1.2, help="最多裁剪多少秒 (默认2)")
    parser.add_argument("--codec", default="mp4v", help="输出编码 (默认 mp4v)")
    parser.add_argument("--dry-run", action="store_true", help="只分析不写入")
    parser.add_argument("--ext", default=",".join(sorted(SUPPORTED_SUFFIXES)), help="文件扩展名")
    return parser.parse_args()


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    source_dir = pathlib.Path(args.source_dir).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()

    exts = {ext.strip().lower() for ext in args.ext.split(",") if ext.strip()} or SUPPORTED_SUFFIXES
    if not source_dir.exists():
        raise RuntimeError(f"目录不存在: {source_dir}")

    files = list(iter_video_files(source_dir, exts))
    if not files:
        print(f"目录 {source_dir} 未找到视频文件")
        return

    print(f"共 {len(files)} 个视频，阈值: {args.grid_threshold}")
    
    stats: List[Tuple[str, float, float, int, float]] = []
    for path in files:
        result = process_single_video(
            input_path=path,
            output_dir=output_dir,
            max_check_seconds=args.max_check_seconds,
            grid_threshold=args.grid_threshold,
            max_trim_seconds=args.max_trim_seconds,
            codec=args.codec,
            dry_run=args.dry_run,
        )
        if result:
            stats.append(result)
    
    total_elapsed = time.perf_counter() - total_start
    print(f"\n完成，总用时 {total_elapsed:.2f}s")
    
    if stats:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "processing_stats.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["文件名", "原始时长(秒)", "处理后时长(秒)", "裁剪帧数", "裁剪时长(秒)"])
            for row in stats:
                writer.writerow([row[0], f"{row[1]:.2f}", f"{row[2]:.2f}", row[3], f"{row[4]:.2f}"])
        print(f"统计: {csv_path}")
        
        trimmed = sum(1 for s in stats if s[3] > 0)
        print(f"\n汇总: 处理 {len(stats)} 个视频, 裁剪 {trimmed} 个, 保持原样 {len(stats) - trimmed} 个")


if __name__ == "__main__":
    main()
