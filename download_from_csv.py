#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量下载执行结果 CSV 中的视频到 input 目录。

示例:
    python download_from_csv.py \
        --csv-path 执行结果9.csv \
        --output-dir input \
        --max-workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从执行结果 CSV 批量下载视频")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("执行结果9.csv"),
        help="包含 result 字段的 CSV 路径 (默认 执行结果9.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input"),
        help="视频保存目录 (默认 ./input)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="下载线程数 (默认 8)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="单个请求超时时间 (秒, 默认 30)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若输出文件已存在则跳过下载",
    )
    return parser.parse_args()


JsonType = Union[str, int, float, bool, None, dict, list]


def extract_mp4_url(data: JsonType) -> Optional[str]:
    if isinstance(data, str):
        candidate = data.strip()
        if candidate.lower().endswith(".mp4") and candidate.startswith("http"):
            return candidate
        # 有些链接后带 query，满足 *.mp4?xx
        if ".mp4" in candidate and candidate.startswith("http"):
            idx = candidate.lower().find(".mp4")
            return candidate[: idx + 4]
        return None
    if isinstance(data, dict):
        # 优先常见字段
        video = data.get("video")
        if isinstance(video, dict):
            url = video.get("url")
            if isinstance(url, str) and url.strip():
                return extract_mp4_url(url)
        for key in ("mediaUrl", "url"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                found = extract_mp4_url(val)
                if found:
                    return found
        for value in data.values():
            found = extract_mp4_url(value)
            if found:
                return found
    if isinstance(data, list):
        for item in data:
            found = extract_mp4_url(item)
            if found:
                return found
    return None


def iter_video_urls(csv_path: Path) -> Iterable[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 CSV: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header_skipped = False
        for row in reader:
            if not row:
                continue
            if not header_skipped and row[0].strip().lower() == "result":
                header_skipped = True
                continue
            header_skipped = True
            raw = row[0]
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[WARN] JSON 解析失败，跳过：{raw[:80]}...")
                continue
            url = extract_mp4_url(data)
            if url:
                yield url
            else:
                print(f"[WARN] 找不到视频链接，跳过：{raw[:120]}...")


def url_to_filename(url: str) -> str:
    name = url.rstrip("/").rsplit("/", 1)[-1]
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    safe = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_"))
    return safe or "video"


def download_one(
    url: str,
    output_dir: Path,
    timeout: float,
    skip_existing: bool,
    lock: Optional[threading.Lock] = None,
) -> tuple[str, bool, Optional[str]]:
    file_id = url_to_filename(url)
    dest = output_dir / f"{file_id}.mp4"
    if skip_existing and dest.exists():
        return file_id, True, "存在，跳过"

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:  # pylint: disable=broad-except
        if dest.exists():
            dest.unlink(missing_ok=True)
        return file_id, False, str(exc)

    if lock:
        with lock:
            print(f"[OK] {file_id} <- {url}")

    return file_id, True, None


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = list(dict.fromkeys(iter_video_urls(args.csv_path.expanduser().resolve())))
    if not urls:
        print("未找到任何可下载的 URL")
        return

    print(f"待下载 {len(urls)} 个视频，输出目录：{output_dir}")
    lock = threading.Lock()
    success = 0
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_map = {
            executor.submit(
                download_one,
                url,
                output_dir,
                args.timeout,
                args.skip_existing,
                lock,
            ): url
            for url in urls
        }

        for future in as_completed(future_map):
            file_id, ok, message = future.result()
            if ok:
                success += 1
            else:
                failures.append(f"{file_id}: {message}")
                print(f"[ERR] {file_id}: {message}")

    print(f"成功 {success} / {len(urls)}")
    if failures:
        print("失败列表：")
        for item in failures:
            print(f"  - {item}")


if __name__ == "__main__":
    main()

