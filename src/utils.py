from __future__ import annotations

import logging
import os
import time
from typing import Any, Mapping, Optional

# Пауза после загрузки до скрина; при значении ≥ этого не советуем отдельно «добавить delay 1500 ms».
CAPTURE_WAIT_TYPING_OK_SEC = 1.5


def stats_capture_wait_seconds(stats: Mapping[str, Any]) -> float:
    try:
        return float(stats.get("capture_wait_seconds", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def setup_logging(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
) -> None:
    """Настраивает корневой логгер один раз (идемпотентно по флагу)."""
    root = logging.getLogger()
    if getattr(root, "_fvt_logging_configured", False):
        return
    logging.basicConfig(level=level, format=fmt)
    setattr(root, "_fvt_logging_configured", True)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ts_ms() -> int:
    return int(time.time() * 1000)


def abs_if_rel(root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(root, path))


def env_token() -> Optional[str]:
    return os.environ.get("FIGMA_ACCESS_TOKEN") or os.environ.get("FIGMA_TOKEN")
