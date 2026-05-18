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
    "в шапке": ("header", "nav", "menu", "top", "head"),
    "в подвале": ("footer", "foot"),
    "у кнопки": ("button", "btn"),
    "у карточки": ("card",),
    "в меню": ("menu", "nav"),
    "у заголовка": ("h1", "h2", "h3", "heading", "title"),
    "у текста": ("p.", "span", "text"),
    "у картинки": ("img", "image", "picture"),
    "в баннере": ("banner", "hero", "head"),
    "баннер": ("banner", "hero", "head"),
    "кнопк": ("button", "btn"),
    "карточ": ("card",),
    "подвал": ("footer",),
}

_KEYWORD_SNIPPET_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("заказать", ("button", "btn", "order", "заказ")),
    ("корпоративн", ("h1", "h2", "title", "heading", "head")),
    ("баннер", ("banner", "hero", "head")),
    ("карточ", ("card",)),
    ("подвал", ("footer",)),
    ("footer", ("footer",)),
    ("шапк", ("header", "nav", "menu")),
    ("меню", ("menu", "nav")),
    ("заголов", ("h1", "h2", "h3", "title", "heading")),
    ("картин", ("img", "image", "picture")),
)


def image_size(path: str) -> Optional[Tuple[int, int]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except OSError:
        return None


def scale_bbox(
    bbox: Tuple[int, int, int, int],
    from_wh: Tuple[int, int],
    to_wh: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Масштаб bbox из координат viewport/страницы в размер другого скрина."""
    x, y, w, h = bbox
    fw, fh = max(1, int(from_wh[0])), max(1, int(from_wh[1]))
    tw, th = max(1, int(to_wh[0])), max(1, int(to_wh[1]))
    if (fw, fh) == (tw, th):
        return x, y, w, h
    sx, sy = tw / float(fw), th / float(fh)
    return (
        int(round(x * sx)),
        int(round(y * sy)),
        max(1, int(round(w * sx))),
        max(1, int(round(h * sy))),
    )


def _extract_quoted(line: str) -> List[str]:
    found = re.findall(r'["«„]([^"»""]+)["»""]', line or "")
    return [s.strip() for s in found if len(s.strip()) >= 2]


def _pick_by_keywords(low: str, cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for needle, hints in _KEYWORD_SNIPPET_HINTS:
        if needle not in low:
            continue
        for hint in hints:
            for el in sorted(cands, key=lambda e: -len(str(e.get("snippet", "")))):
                sn = str(el.get("snippet", "")).lower()
                inner = str(el.get("innerText", "") or "").lower()
                if hint in sn or hint in inner:
                    return el
    return None


def _pick_zone_fallback(low: str, cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cands:
        return None
    if "сверху" in low or "отступ сверху" in low or "шапк" in low:
        tops = sorted(cands, key=lambda e: (int(e.get("y", 0)), -int(e.get("w", 0))))
        for el in tops[:12]:
            if int(el.get("y", 99999)) < 500:
                return el
        return tops[0]
    if "центр" in low:
        cands_s = [e for e in cands if int(e.get("w", 0)) > 80 and int(e.get("h", 0)) > 24]
        if not cands_s:
            cands_s = cands
        return sorted(
            cands_s,
            key=lambda e: abs(int(e.get("y", 0)) + int(e.get("h", 0)) // 2 - 1200),
        )[0]
    if "подвал" in low or "footer" in low:
        return sorted(cands, key=lambda e: -int(e.get("y", 0)))[0]
    return None


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
    cands = [e for e in elements if isinstance(e, dict) and _bbox_tuple(e)]
    for q in _extract_quoted(line):
        ql = q.lower()
        for el in cands:
            inner = str(el.get("innerText", "") or "").strip().lower()
            if inner and (ql in inner or inner in ql):
                return el
    hit = _pick_by_keywords(low, cands)
    if hit:
        return hit
    for zone, hints in _ZONE_SNIPPET_HINTS.items():
        if zone not in low:
            continue
        for hint in hints:
            for el in sorted(cands, key=lambda e: -len(str(e.get("snippet", "")))):
                sn = str(el.get("snippet", "")).lower()
                if hint in sn:
                    return el
    zone_el = _pick_zone_fallback(low, cands)
    if zone_el:
        return zone_el
    return find_element_for_recommendation_line(line, elements)


def find_element_for_recommendation_line(line: str, elements: List[Any]) -> Optional[Dict[str, Any]]:
    """
    Подбирает блок layout по тексту правки (snippet, класс, тег из «Блок h1…», «.foo» в строке).
    """
    if not line or not isinstance(elements, list):
        return None
    low = line.lower()
    cands = [e for e in elements if isinstance(e, dict) and _bbox_tuple(e)]
    for q in _extract_quoted(line):
        ql = q.lower()
        for el in cands:
            inner = str(el.get("innerText", "") or "").strip().lower()
            if inner and (ql in inner or inner in ql):
                return el
    hit = _pick_by_keywords(low, cands)
    if hit:
        return hit
    cands.sort(key=lambda e: len(str(e.get("snippet", ""))), reverse=True)
    for el in cands:
        inner = str(el.get("innerText", "") or "").strip().lower()
        if inner and len(inner) >= 3 and inner in low:
            return el
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


def refine_bug_table_bbox(
    bbox: Optional[Tuple[int, int, int, int]],
    bug_item: Optional[Dict[str, Any]],
    hotspots: Optional[Dict[str, Any]],
    *,
    ref_wh: Optional[Tuple[int, int]],
    pad: int = 12,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Если bbox — почти весь экран (типичный div.wrapper), подменяем на ячейку diff
    или пересечение с элементом из elements_overlap — чтобы в отчёте было видно отличие.
    """
    if not bbox:
        return None
    x, y, w, h = bbox
    if w < 8 or h < 8:
        return bbox
    iw, ih = (ref_wh or (1280, 720))
    iw, ih = max(1, int(iw)), max(1, int(ih))
    page_area = float(iw * ih)
    box_area = float(w * h)
    tall_strip = h > int(ih * 0.4) and (box_area / page_area) > 0.22
    huge = box_area > 0.28 * page_area
    if not huge and not tall_strip:
        return bbox
    hot = hotspots if isinstance(hotspots, dict) else {}
    sn = ""
    if isinstance(bug_item, dict):
        sn = str(bug_item.get("snippet", "")).strip()
    for ov in hot.get("elements_overlap") or []:
        if not isinstance(ov, dict):
            continue
        try:
            ox, oy, ow, oh = int(ov["x"]), int(ov["y"]), int(ov["w"]), int(ov["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if sn and str(ov.get("snippet", "")).strip() == sn and ow > 8 and oh > 8:
            if ow * oh < box_area * 0.95:
                return ox, oy, ow, oh
    best: Optional[Tuple[int, int, int, int]] = None
    best_score = 0.0
    cells = [c for c in hot.get("grid_cells") or [] if isinstance(c, dict)]
    for cell in cells:
        try:
            cx, cy, cw, ch = int(cell["x"]), int(cell["y"]), int(cell["w"]), int(cell["h"])
            frac = float(cell.get("changed_frac_pct", 0) or 0)
        except (KeyError, TypeError, ValueError):
            continue
        ix0 = max(x, cx)
        iy0 = max(y, cy)
        ix1 = min(x + w, cx + cw)
        iy1 = min(y + h, cy + ch)
        iw_i = ix1 - ix0
        ih_i = iy1 - iy0
        if iw_i < 4 or ih_i < 4:
            continue
        inter_a = float(iw_i * ih_i)
        score = inter_a * max(0.5, frac)
        if score > best_score:
            best_score = score
            best = (ix0, iy0, iw_i, ih_i)
    if best and best_score > 0:
        bx, by, bw, bh = best
        return (
            max(0, bx - pad),
            max(0, by - pad),
            min(iw, bx + bw + pad) - max(0, bx - pad),
            min(ih, by + bh + pad) - max(0, by - pad),
        )
    # Нет пересечения с сеткой — берём самую «горячую» ячейку внутри большого блока (по центру)
    cx_mid, cy_mid = x + w // 2, y + h // 2
    for cell in sorted(cells, key=lambda c: -float(c.get("changed_frac_pct", 0) or 0)):
        try:
            cx, cy, cw, ch = int(cell["x"]), int(cell["y"]), int(cell["w"]), int(cell["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if cx <= cx_mid < cx + cw and cy <= cy_mid < cy + ch and cw * ch < box_area * 0.5:
            return (
                max(0, cx - pad),
                max(0, cy - pad),
                min(iw, cx + cw + pad) - max(0, cx - pad),
                min(ih, cy + ch + pad) - max(0, cy - pad),
            )
    return bbox


_THUMB_MAX_SIDE = 200
_THUMB_MAX_ASPECT = 2.5
_THUMB_MIN_SIDE = 32


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
    pad: int = 12,
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
    pad: int = 12,
    outline: str = "#ff3355",
    width: int = 3,
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
