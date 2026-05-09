"""Вспомогательные функции: каталоги, base64, сохранение истории."""

from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from PIL import Image


def pil_to_base64_png(im: Image.Image, max_side: int = 1536) -> str:
    """PNG в base64; при необходимости уменьшает длинную сторону (для Ollama)."""
    img = im.convert("RGB") if im.mode not in ("RGB", "L") else im
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def append_report_history(reports_dir: Path, record: Dict[str, Any]) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"run_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    idx = reports_dir / "history_index.jsonl"
    line = json.dumps({"ts": ts, "file": path.name, "verdict": record.get("verdict")}, ensure_ascii=False)
    with open(idx, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path
