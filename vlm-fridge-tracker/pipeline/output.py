"""运行日志捕获：双通道输出（终端 + 缓冲区），结束后保存为 md 文件"""

import os
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Generator


class OutputLogger:
    """双通道输出：同时写终端和缓冲区，运行结束后保存为 md 文件"""

    def __init__(self, original_stdout):
        self.original = original_stdout
        self.buffer: list[str] = []

    def write(self, text: str) -> None:
        self.original.write(text)
        self.buffer.append(text)

    def flush(self) -> None:
        self.original.flush()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        content = "".join(self.buffer)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# 冰箱物品追踪 — 运行记录\n\n")
            f.write(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("```\n")
            f.write(content)
            f.write("```\n")


@contextmanager
def capture_stdout() -> Generator[OutputLogger, None, None]:
    """安全地捕获 stdout，异常时也能恢复"""
    logger = OutputLogger(sys.stdout)
    sys.stdout = logger
    try:
        yield logger
    finally:
        sys.stdout = logger.original
