from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage
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


_ZONE_SNIPPET_HINTS: Dict[str, tuple[str, ...]] = {
    "в шапке": ("header", "nav", "menu", "top"),
    "в подвале": ("footer",),
    "у кнопки": ("button", "btn"),
    "у карточки": ("card",),
    "в меню": ("menu", "nav"),
    "у заголовка": ("h1", "h2", "h3", "heading", "title"),
    "у текста": ("p.", "span", "text"),
    "у картинки": ("img", "image", "picture"),
    "в баннере": ("banner",),
}


def find_element_for_bug_item(
    line: str,
    elements: List[Any],
    bug_item: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Элемент для кропа: bbox из item, snippet, зона в тексте или подстрока в layout."""
    if isinstance(bug_item, dict):
        try:
            x, y, w, h = int(bug_item["x"]), int(bug_item["y"]), int(bug_item["w"]), int(bug_item["h"])
            if w > 0 and h > 0:
                return {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "snippet": str(bug_item.get("snippet", "")),
                }
        except (KeyError, TypeError, ValueError):
            pass
        sn = str(bug_item.get("snippet", "")).strip()
        if sn and isinstance(elements, list):
            for el in elements:
                if isinstance(el, dict) and str(el.get("snippet", "")).strip() == sn:
                    return el
    if not line or not isinstance(elements, list):
        return None
    low = line.lower()
    for zone, hints in _ZONE_SNIPPET_HINTS.items():
        if zone not in low:
            continue
        cands = [e for e in elements if isinstance(e, dict) and _bbox_tuple(e)]
        for hint in hints:
            for el in sorted(cands, key=lambda e: -len(str(e.get("snippet", "")))):
                sn = str(el.get("snippet", "")).lower()
                if hint in sn:
                    return el
    return find_element_for_recommendation_line(line, elements)


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


_THUMB_MAX_SIDE = 120
_THUMB_MAX_ASPECT = 2.0
_THUMB_MIN_SIDE = 24


def _clamp_bbox_aspect(
    x: int, y: int, w: int, h: int, *, max_aspect: float = _THUMB_MAX_ASPECT
) -> Tuple[int, int, int, int]:
    """Убирает слишком вытянутые полосы (full-width/header strip) для превью в таблице."""
    if w < 1 or h < 1:
        return x, y, max(w, 1), max(h, 1)
    aspect = w / float(h)
    if aspect <= max_aspect and aspect >= 1.0 / max_aspect:
        return x, y, w, h
    if aspect > max_aspect:
        nw = max(_THUMB_MIN_SIDE, int(h * max_aspect))
        x = x + (w - nw) // 2
        w = nw
    else:
        nh = max(_THUMB_MIN_SIDE, int(w / max_aspect))
        y = y + (h - nh) // 2
        h = nh
    return x, y, w, h


def _fit_thumbnail(im: PILImage.Image, max_side: int = _THUMB_MAX_SIDE) -> PILImage.Image:
    from PIL import Image

    im = im.copy()
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return im


def save_plain_crop(
    screenshot_path: str,
    bbox: Tuple[int, int, int, int],
    out_path: str,
    *,
    pad: int = 8,
    max_thumb_side: int = _THUMB_MAX_SIDE,
) -> bool:
    """Компактный кроп для таблицы (без растягивания на всю ширину страницы)."""
    try:
        from PIL import Image
    except ImportError:
        return False
    if not screenshot_path or not os.path.isfile(screenshot_path):
        return False
    x, y, w, h = _clamp_bbox_aspect(*bbox)
    try:
        im = Image.open(screenshot_path).convert("RGB")
    except OSError:
        return False
    iw, ih = im.size
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(iw, x + w + pad)
    y1 = min(ih, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return False
    crop = _fit_thumbnail(im.crop((x0, y0, x1, y1)), max_thumb_side)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    crop.save(out_path, format="PNG", optimize=True)
    return True


def save_highlight_crop(
    screenshot_path: str,
    bbox: Tuple[int, int, int, int],
    out_path: str,
    *,
    pad: int = 8,
    outline: str = "#ff3355",
    width: int = 2,
    max_thumb_side: int = _THUMB_MAX_SIDE,
) -> bool:
    """Компактный кроп с рамкой блока для колонки «фактический»."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    if not screenshot_path or not os.path.isfile(screenshot_path):
        return False
    x, y, w, h = _clamp_bbox_aspect(*bbox)
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
    thumb = _fit_thumbnail(crop, max_thumb_side)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    thumb.convert("RGB").save(out_path, format="PNG", optimize=True)
    return True
