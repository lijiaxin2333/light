#!/usr/bin/env python3
"""
CLI + PyQt GUI：输入 zao ci 分享链接，提取并下载 .mp4。

依赖:
    pip install playwright httpx PyQt5
    python -m playwright install

命令行示例:
    python get_zc_video.py https://.../share/video/xxx output.mp4

GUI:
    python get_zc_video.py --gui
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import TYPE_CHECKING

import httpx
from playwright.sync_api import sync_playwright

if TYPE_CHECKING:  # 仅为类型提示，避免 GUI 依赖在 CLI 模式就报错
    from PyQt5 import QtCore, QtWidgets


def sniff_mp4(source_url: str, timeout: int) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(source_url, wait_until="domcontentloaded", timeout=timeout * 1000)
            response = page.wait_for_event(
                "response",
                lambda resp: resp.ok and resp.url.lower().endswith(".mp4"),
                timeout=timeout * 1000,
            )
            return response.url
        finally:
            browser.close()


def download_file(file_url: str, destination: pathlib.Path) -> None:
    with httpx.Client(follow_redirects=True) as client:
        with client.stream("GET", file_url) as resp:
            resp.raise_for_status()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                for chunk in resp.iter_bytes():
                    handle.write(chunk)


def compute_default_output_name(directory: pathlib.Path, suffix: str = ".mp4") -> str:
    pattern = re.compile(r"output_(\d+)$")
    max_idx = 0
    if directory.exists():
        for child in directory.iterdir():
            if child.is_file():
                match = pattern.match(child.stem)
                if match:
                    max_idx = max(max_idx, int(match.group(1)))
    return f"output_{max_idx + 1}{suffix}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抓取 zao ci 分享链接中的 .mp4")
    parser.add_argument("url", nargs="?", help="分享页链接")
    parser.add_argument("output", nargs="?", help="输出文件路径")
    parser.add_argument("--timeout", type=int, default=30, help="等待秒数 (默认30)")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="启动 PyQt GUI（此模式下忽略其它参数）",
    )
    return parser


def run_cli(url: str, output: str, timeout: int) -> None:
    try:
        mp4_url = sniff_mp4(url, timeout)
        destination = pathlib.Path(output).expanduser().resolve()
        download_file(mp4_url, destination)
        print(f"已保存: {destination} (源地址: {mp4_url})")
    except Exception as exc:
        print(f"失败: {exc}", file=sys.stderr)
        sys.exit(1)


def launch_gui() -> None:
    try:
        from PyQt5 import QtCore, QtWidgets
    except ImportError as exc:  # pragma: no cover - 运行时提示
        raise RuntimeError("请先安装 PyQt5: pip install PyQt5") from exc

    class VideoFetcherWorker(QtCore.QThread):
        progress = QtCore.pyqtSignal(str)
        finished = QtCore.pyqtSignal(bool, str)

        def __init__(self, url: str, destination: pathlib.Path, timeout: int) -> None:
            super().__init__()
            self.url = url
            self.destination = destination
            self.timeout = timeout

        def run(self) -> None:
            try:
                self.progress.emit("开始解析分享页...")
                mp4_url = sniff_mp4(self.url, self.timeout)
                self.progress.emit(f"找到 MP4: {mp4_url}")
                self.progress.emit(f"开始下载 -> {self.destination}")
                download_file(mp4_url, self.destination)
                self.finished.emit(True, f"下载完成: {self.destination}")
            except Exception as exc:
                self.finished.emit(False, f"失败: {exc}")

    class MainWindow(QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("ZaoCi MP4 下载器")
            self.resize(520, 320)

            self.url_edit = QtWidgets.QLineEdit()
            self.url_edit.setPlaceholderText("https://www.zaoci.tv/share/video/...")

            self.dir_edit = QtWidgets.QLineEdit()
            self.dir_edit.setPlaceholderText("选择输出目录")
            self.browse_btn = QtWidgets.QPushButton("浏览...")

            self.output_edit = QtWidgets.QLineEdit()
            self.output_edit.setPlaceholderText("output_x.mp4")

            self.start_btn = QtWidgets.QPushButton("开始下载")
            self.log_view = QtWidgets.QTextEdit()
            self.log_view.setReadOnly(True)

            form = QtWidgets.QFormLayout()
            form.addRow("分享链接:", self.url_edit)

            dir_layout = QtWidgets.QHBoxLayout()
            dir_layout.addWidget(self.dir_edit)
            dir_layout.addWidget(self.browse_btn)
            form.addRow("输出目录:", dir_layout)
            form.addRow("输出文件名:", self.output_edit)

            layout = QtWidgets.QVBoxLayout(self)
            layout.addLayout(form)
            layout.addWidget(self.start_btn)
            layout.addWidget(self.log_view)

            self.worker: VideoFetcherWorker | None = None
            self.last_auto_name = ""

            self.browse_btn.clicked.connect(self.choose_directory)
            self.dir_edit.editingFinished.connect(self.handle_dir_finished)
            self.start_btn.clicked.connect(self.start_download)

        def choose_directory(self) -> None:
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录")
            if directory:
                self.dir_edit.setText(directory)
                self.update_default_output()

        def handle_dir_finished(self) -> None:
            self.update_default_output()

        def update_default_output(self) -> None:
            directory = pathlib.Path(self.dir_edit.text()).expanduser()
            if not directory.exists():
                return
            new_name = compute_default_output_name(directory)
            current = self.output_edit.text().strip()
            if not current or current == self.last_auto_name:
                self.output_edit.setText(new_name)
            self.last_auto_name = new_name

        def start_download(self) -> None:
            url = self.url_edit.text().strip()
            directory = pathlib.Path(self.dir_edit.text()).expanduser()
            filename = self.output_edit.text().strip()

            if not url:
                self.append_log("请填写分享链接。")
                return
            if not directory:
                self.append_log("请选择输出目录。")
                return
            directory.mkdir(parents=True, exist_ok=True)

            if not filename:
                filename = compute_default_output_name(directory)
                self.output_edit.setText(filename)
            if not filename.lower().endswith(".mp4"):
                filename = f"{filename}.mp4"
                self.output_edit.setText(filename)

            destination = directory / filename
            if destination.exists():
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "覆盖确认",
                    f"{destination} 已存在，是否覆盖？",
                )
                if reply != QtWidgets.QMessageBox.Yes:
                    return

            self.toggle_inputs(False)
            self.append_log("任务启动...")
            self.worker = VideoFetcherWorker(url, destination, timeout=30)
            self.worker.progress.connect(self.append_log)
            self.worker.finished.connect(self.on_finished)
            self.worker.finished.connect(lambda *_: self.worker and self.worker.deleteLater())
            self.worker.start()

        def on_finished(self, success: bool, message: str) -> None:
            self.append_log(message)
            if not success and self.dir_edit.text():
                self.append_log("请检查链接或网络后重试。")
            self.toggle_inputs(True)
            self.worker = None

        def toggle_inputs(self, enabled: bool) -> None:
            for widget in (
                self.url_edit,
                self.dir_edit,
                self.output_edit,
                self.browse_btn,
                self.start_btn,
            ):
                widget.setEnabled(enabled)

        def append_log(self, text: str) -> None:
            self.log_view.append(text)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.gui:
        launch_gui()
        return

    if not args.url or not args.output:
        parser.error("缺少 url/output；若需图形界面请使用 --gui")

    run_cli(args.url, args.output, args.timeout)


if __name__ == "__main__":
    main()