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

用法:
    python light.py --source-dir ./videos --output-dir ./output
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import shutil
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
    
    只检测画面正中央（1/2位置）是否有贯穿的分割线：
    - 竖线：从上到下贯穿画面中央
    - 横线：从左到右贯穿画面中央
    
    threshold: 分割线需要贯穿的比例，默认0.6表示需要贯穿60%以上才算
    """
    # 缩放到统一尺寸
    DETECT_SIZE = (320, 180)
    small = cv2.resize(frame, DETECT_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    # 边缘检测（使用较严格的参数）
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 80, 200)
    
    h, w = edges.shape
    
    # 检测中央竖线（在宽度的1/2位置，检测一个窄带）
    center_x = w // 2
    band_width = max(3, w // 30)  # 窄带宽度
    left = center_x - band_width // 2
    right = center_x + band_width // 2 + 1
    
    vertical_band = edges[:, left:right]
    # 计算每一行是否有边缘点（即该行是否被竖线穿过）
    row_has_edge = np.any(vertical_band > 0, axis=1)
    vertical_score = np.mean(row_has_edge)  # 竖线贯穿的比例
    
    # 检测中央横线（在高度的1/2位置，检测一个窄带）
    center_y = h // 2
    band_height = max(3, h // 30)
    top = center_y - band_height // 2
    bottom = center_y + band_height // 2 + 1
    
    horizontal_band = edges[top:bottom, :]
    # 计算每一列是否有边缘点（即该列是否被横线穿过）
    col_has_edge = np.any(horizontal_band > 0, axis=0)
    horizontal_score = np.mean(col_has_edge)  # 横线贯穿的比例
    
    # 判断：分数需要超过阈值才算有分割线
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
    """导出视频（跳过前trim_frames帧）"""
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

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if trim_frames > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, trim_frames)
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        writer.write(frame)

    capture.release()
    writer.release()
    
    output_frames = total_frames - trim_frames
    output_duration = output_frames / fps if fps > 0 else 0.0
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
    parser.add_argument("--max-check-seconds", type=float, default=2.0, help="最多检测开头多少秒 (默认2)")
    parser.add_argument("--grid-threshold", type=float, default=0.6, help="分割线贯穿比例阈值，越高越严格 (默认0.6)")
    parser.add_argument("--max-trim-seconds", type=float, default=1.0, help="最多裁剪多少秒 (默认1)")
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
