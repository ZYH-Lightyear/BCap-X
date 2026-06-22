"""RoboMEx 的日志配置:给框架一个简洁、可选落盘的 logger。

库代码只用 :func:`get_logger` 取 ``robomex.*`` 命名空间下的 logger 写 INFO 级日志;
默认不挂任何 handler(对把 robomex 当库用的人保持安静)。入口脚本(``examples/``、
评测脚手架)调用一次 :func:`configure_logging`,把日志同时打到控制台和可选的文件。
"""

from __future__ import annotations

import logging
from pathlib import Path

_ROOT_NAME = "robomex"
_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str = "") -> logging.Logger:
    """返回 ``robomex`` 命名空间下的 logger(``name`` 为子模块名,如 ``"planner"``)。"""

    return logging.getLogger(f"{_ROOT_NAME}.{name}" if name else _ROOT_NAME)


def configure_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """配置 ``robomex`` 顶层 logger:控制台 + 可选文件,统一简洁格式。

    可重复调用:每次调用会先清掉旧 handler,避免重复运行时日志翻倍。返回顶层 logger。
    """

    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(level)
    logger.propagate = False  # 不向 root 冒泡,避免和其他库的 handler 打架

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
