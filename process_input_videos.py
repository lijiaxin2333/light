#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取执行结果 CSV 中的 URL 顺序，批量处理 input 目录的视频并记录状态。

示例:
    python process_input_videos.py \
        --csv-path 执行结果9.csv \
        --input-dir input \
        --output-dir output \
        --grid-threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import light


@dataclass
class VideoEntry:
    index: int
    file_id: str
    filename: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量处理 input 目录视频并记录状态")
    parser.add_argument("--input-dir", type=Path, default=Path("input"), help="待处理视频目录")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="输出目录")
    parser.add_argument("--max-check-seconds", type=float, default=1.2, help="检测开头多少秒")
    parser.add_argument("--grid-threshold", type=float, default=0.5, help="分割线贯穿阈值")
    parser.add_argument("--max-trim-seconds", type=float, default=1.0, help="最多裁剪秒数")
    parser.add_argument("--codec", default="mp4v", help="导出编码")
    parser.add_argument("--dry-run", action="store_true", help="仅分析不写入")
    parser.add_argument("--workers", type=int, default=8, help="并行线程数 (默认 8)")
    return parser.parse_args()


def load_entries(input_dir: Path) -> List[VideoEntry]:
    if not input_dir.exists():
        raise FileNotFoundError(f"未找到 input 目录: {input_dir}")

    files = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"],
        key=lambda p: p.name,
    )
    entries: List[VideoEntry] = []
    for idx, path in enumerate(files, start=1):
        entries.append(
            VideoEntry(
                index=idx,
                file_id=path.stem,
                filename=path.name,
            )
        )
    return entries


def process_videos(entries: Iterable[VideoEntry], args: argparse.Namespace) -> None:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries_list = list(entries)
    total_entries = len(entries_list)
    trimmed_values: List[int] = []
    processed_count = 0

    status_path = output_dir / "status.csv"
    with status_path.open("w", encoding="utf-8", newline="") as status_file:
        writer = csv.writer(status_file)
        writer.writerow(["序号", "原文件名", "裁剪帧数", "裁剪时长(秒)", "状态", "人工校验"])
        status_file.flush()

        status_lock = threading.Lock()
        stats_lock = threading.Lock()

        def append_row(row: List[str]) -> None:
            with status_lock:
                writer.writerow(row)
                status_file.flush()

        def handle_entry(entry: VideoEntry) -> tuple[List[str], Optional[int]]:
            input_path = input_dir / entry.filename
            if not input_path.exists():
                return [
                    str(entry.index),
                    entry.file_id,
                    "0",
                    "0.00",
                    "缺失",
                    "",
                ], None
            try:
                result = light.process_single_video(
                    input_path=input_path,
                    output_dir=output_dir,
                    max_check_seconds=args.max_check_seconds,
                    grid_threshold=args.grid_threshold,
                    max_trim_seconds=args.max_trim_seconds,
                    codec=args.codec,
                    dry_run=args.dry_run,
                )
            except Exception as exc:  # pylint: disable=broad-except
                return [
                    str(entry.index),
                    entry.file_id,
                    "0",
                    "0.00",
                    f"失败: {exc}",
                    "",
                ], None

            if not result:
                return [
                    str(entry.index),
                    entry.file_id,
                    "0",
                    "0.00",
                    "跳过",
                    "",
                ], None

            _, _, _, trimmed_frames, trimmed_seconds = result
            status = "已裁剪" if trimmed_frames > 0 else "保持原样"
            return [
                str(entry.index),
                entry.file_id,
                str(trimmed_frames),
                f"{trimmed_seconds:.2f}",
                status,
                "",
            ], trimmed_frames

        futures = {}
        seen_files = set()

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for entry in entries_list:
                if entry.file_id in seen_files:
                    append_row(
                        [
                            str(entry.index),
                            entry.file_id,
                            "0",
                            "0.00",
                            "重复跳过",
                            "",
                        ]
                    )
                    continue
                seen_files.add(entry.file_id)
                futures[executor.submit(handle_entry, entry)] = entry

            for future in as_completed(futures):
                row, trimmed_frames = future.result()
                append_row(row)
                if trimmed_frames is not None:
                    with stats_lock:
                        processed_count += 1
                        trimmed_values.append(trimmed_frames)

    print(f"状态写入 {status_path}")

    avg = (sum(trimmed_values) / processed_count) if processed_count else 0.0
    avg_path = output_dir / "average_trim.csv"
    with avg_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["视频总数", "成功处理数", "平均裁剪帧数"])
        writer.writerow([total_entries, processed_count, f"{avg:.2f}"])
    print(f"平均统计写入 {avg_path}")


def main() -> None:
    args = parse_args()
    entries = load_entries(args.input_dir.expanduser().resolve())
    if not entries:
        print("input 目录中没有可用视频")
        return
    process_videos(entries, args)


if __name__ == "__main__":
    main()

