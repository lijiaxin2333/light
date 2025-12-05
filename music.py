#!/usr/bin/env python3
"""
基于 PyQt 的 B 站音频下载器：输入链接，按下回车或“下载”按钮即可提取音频。

功能：
- 自动命名（output_1.m4a、output_2.m4a ...），无需手动输入
- “取消”按钮用于终止当前下载任务
- 通过设置 UA/Referer 防止被 B 站拦截

依赖：
    pip install yt-dlp PyQt5
    brew install ffmpeg   # macOS 下若未安装
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Dict, Optional

try:
    from yt_dlp import YoutubeDL
except ImportError as exc:  # pragma: no cover
    print("缺少依赖：请先运行 `pip install yt-dlp` 再使用此脚本。", file=sys.stderr)
    raise


def compute_default_name(directory: pathlib.Path) -> str:
    """
    自动生成 output_x.m4a 文件名。
    """
    directory.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        candidate = directory / f"output_{idx}.m4a"
        if not candidate.exists():
            return candidate.name
        idx += 1


def build_ydl_opts(output_path: pathlib.Path) -> Dict[str, Any]:
    """
    构造 yt-dlp 下载参数，提取音频并保存为指定文件。
    """
    outtmpl = str(output_path.as_posix())
    return {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": False,
        "no_warnings": True,
        "ignoreerrors": False,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.bilibili.com/",
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                "preferredquality": "192",
            },
            {"key": "FFmpegMetadata"},
        ],
    }


def download_audio(url: str, output_path: pathlib.Path, hook=None) -> None:
    """
    下载音频到指定路径；支持 progress hook 和取消。
    """
    ydl_opts = build_ydl_opts(output_path)
    if hook:
        ydl_opts.setdefault("progress_hooks", []).append(hook)
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B 站音频下载器（GUI）")
    parser.add_argument("--gui", action="store_true", help="启动 PyQt 图形界面")
    parser.add_argument("--url", help="B 站视频链接（CLI 模式）")
    parser.add_argument("-o", "--output", help="输出文件名，CLI 模式下有效")
    parser.add_argument("--dir", default=".", help="输出目录（CLI 模式）")
    return parser.parse_args()


def run_cli(url: str, directory: pathlib.Path, filename: Optional[str]) -> None:
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if not filename:
        filename = compute_default_name(directory)
    output_path = directory / filename
    print(f"开始下载音频: {url}")
    print(f"保存到: {output_path}")
    download_audio(url, output_path)
    print("下载完成。")


def launch_gui() -> None:  # pragma: no cover - GUI逻辑
    from PyQt5 import QtCore, QtWidgets

    class DownloadThread(QtCore.QThread):
        progress = QtCore.pyqtSignal(str)
        finished = QtCore.pyqtSignal(bool, str)

        def __init__(self, url: str, output_path: pathlib.Path) -> None:
            super().__init__()
            self.url = url
            self.output_path = output_path
            self._cancelled = False

        def cancel(self) -> None:
            self._cancelled = True

        def progress_hook(self, status: dict) -> None:
            if self._cancelled:
                raise Exception("下载已被取消")
            if status.get("status") == "downloading":
                speed = status.get("speed")
                speed_str = f"{speed/1024:.1f} KB/s" if speed else ""
                eta = status.get("eta")
                eta_str = f" 剩余 {eta:.0f}s" if eta else ""
                self.progress.emit(f"下载中… {speed_str}{eta_str}")
            elif status.get("status") == "finished":
                self.progress.emit("下载完成，开始转码…")

        def run(self) -> None:
            try:
                download_audio(self.url, self.output_path, hook=self.progress_hook)
                self.finished.emit(True, f"已保存: {self.output_path}")
            except Exception as exc:
                self.finished.emit(False, f"失败: {exc}")

    class MainWindow(QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("B站音频下载器")
            self.resize(560, 320)

            self.url_edit = QtWidgets.QLineEdit()
            self.url_edit.setPlaceholderText("https://www.bilibili.com/video/BV...")

            self.dir_edit = QtWidgets.QLineEdit()
            self.dir_edit.setPlaceholderText("选择保存目录")
            self.browse_btn = QtWidgets.QPushButton("浏览…")

            self.status_label = QtWidgets.QLabel("文件名将自动命名，例如 output_1.m4a")

            self.download_btn = QtWidgets.QPushButton("开始下载")
            self.cancel_btn = QtWidgets.QPushButton("取消当前下载")
            self.cancel_btn.setEnabled(False)

            self.log_view = QtWidgets.QTextEdit()
            self.log_view.setReadOnly(True)

            form = QtWidgets.QFormLayout()
            form.addRow("B站链接:", self.url_edit)
            dir_layout = QtWidgets.QHBoxLayout()
            dir_layout.addWidget(self.dir_edit)
            dir_layout.addWidget(self.browse_btn)
            form.addRow("保存目录:", dir_layout)
            form.addRow("命名说明:", self.status_label)

            layout = QtWidgets.QVBoxLayout(self)
            layout.addLayout(form)
            layout.addWidget(self.download_btn)
            layout.addWidget(self.cancel_btn)
            layout.addWidget(self.log_view)

            self.worker: Optional[DownloadThread] = None

            self.browse_btn.clicked.connect(self.choose_directory)
            self.download_btn.clicked.connect(self.start_download)
            self.cancel_btn.clicked.connect(self.cancel_download)
            self.url_edit.returnPressed.connect(self.start_download)

        def choose_directory(self) -> None:
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择保存目录")
            if directory:
                self.dir_edit.setText(directory)

        def start_download(self) -> None:
            url = self.url_edit.text().strip()
            directory = pathlib.Path(self.dir_edit.text()).expanduser()
            if not url:
                self.append_log("请填写 B 站视频链接。")
                return
            if not directory:
                self.append_log("请选择保存目录。")
                return
            directory.mkdir(parents=True, exist_ok=True)
            filename = compute_default_name(directory)
            output_path = directory / filename

            self.append_log(f"开始下载 -> {output_path.name}")
            self.toggle_inputs(False)
            self.worker = DownloadThread(url, output_path)
            self.worker.progress.connect(self.append_log)
            self.worker.finished.connect(self.on_finished)
            self.worker.start()

        def cancel_download(self) -> None:
            if self.worker and self.worker.isRunning():
                self.worker.cancel()
                self.append_log("取消指令已发送…")

        def on_finished(self, success: bool, message: str) -> None:
            self.append_log(message)
            self.toggle_inputs(True)
            self.worker = None

        def toggle_inputs(self, enabled: bool) -> None:
            for widget in (self.url_edit, self.dir_edit, self.browse_btn, self.download_btn):
                widget.setEnabled(enabled)
            self.cancel_btn.setEnabled(not enabled)

        def append_log(self, text: str) -> None:
            self.log_view.append(text)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


def main() -> None:
    args = parse_args()

    if args.gui or (not args.url and not args.output):
        launch_gui()
        return

    if not args.url:
        print("CLI 模式需要提供 --url", file=sys.stderr)
        sys.exit(1)

    directory = pathlib.Path(args.dir)
    run_cli(args.url, directory, args.output)


if __name__ == "__main__":
    main()
