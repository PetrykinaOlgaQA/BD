from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple


def _bbox_tuple(el: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    try:
        x, y, w, h = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def element_bbox(el: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    return _bbox_tuple(el)


def find_element_for_recommendation_line(line: str, elements: List[Any]) -> Optional[Dict[str, Any]]:
    """
    Подбирает блок layout по тексту правки (snippet, класс, тег из «Блок h1…», «.foo» в строке).
    """
    if not line or not isinstance(elements, list):
        return None
    low = line.lower()
    cands = [e for e in elements if isinstance(e, dict) and _bbox_tuple(e)]
    cands.sort(key=lambda e: len(str(e.get("snippet", ""))), reverse=True)
    for el in cands:
        sn = str(el.get("snippet", "")).strip()
        if not sn:
            continue
        sn_l = sn.lower()
        if sn_l in low:
            return el
        for part in re.split(r"[.\#]", sn):
            p = part.strip()
            if not p:
                continue
            if len(p) == 1:
                if re.search(rf"(?i)(?<![a-z0-9#._]){re.escape(p)}(?![a-z0-9])", line):
                    return el
            elif len(p) >= 2 and p.lower() in low:
                return el
    return None


def save_highlight_crop(
    screenshot_path: str,
    bbox: Tuple[int, int, int, int],
    out_path: str,
    *,
    pad: int = 14,
    outline: str = "#ff3355",
    width: int = 3,
) -> bool:
    """Вырезает область вокруг блока и рисует прямоугольник по границе блока (в координатах кропа)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    if not screenshot_path or not os.path.isfile(screenshot_path):
        return False
    x, y, w, h = bbox
    try:
        im = Image.open(screenshot_path).convert("RGBA")
    except OSError:
        return False
    iw, ih = im.size
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(iw, x + w + pad)
    y1 = min(ih, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return False
    crop = im.crop((x0, y0, x1, y1))
    draw = ImageDraw.Draw(crop)
    rx0, ry0 = x - x0, y - y0
    rx1, ry1 = x + w - x0, y + h - y0
    try:
        draw.rectangle([rx0, ry0, rx1, ry1], outline=outline, width=width)
    except TypeError:
        for d in range(width):
            draw.rectangle([rx0 - d, ry0 - d, rx1 + d, ry1 + d], outline=outline)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    crop.convert("RGB").save(out_path, format="PNG", optimize=True)
    return True
