"""
Ложные «фрагмента нет на макете/странице» из-за вставки блока на сайте между двумя
одинаковыми секциями (на Figma блока нет, страница длиннее — координаты не совпадают).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Пороги (подобраны под 1280×720 и длинные лендинги)
_MIN_SHIFT_PX = 40
_MAX_SHIFT_SEARCH_PX = 720
_SHIFT_STEP_PX = 24
_PATCH_SIG_SIZE = 32
_MATCH_CORR = 0.55
_EXTRA_BLOCK_MIN_H = 48
_DUPLICATE_CORR = 0.52
_MIN_PAGE_EXTRA_PX = 28
_BELOW_MOCKUP_MARGIN_PX = 24


def _patch_vec(rgb: np.ndarray, x: int, y: int, w: int, h: int, size: int = _PATCH_SIG_SIZE) -> Optional[np.ndarray]:
    if rgb is None or rgb.size == 0:
        return None
    mh, mw = rgb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(mw, x + max(w, 4)), min(mh, y + max(h, 4))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    sub = rgb[y0:y1, x0:x1]
    if sub.ndim == 3:
        lum = sub.mean(axis=2).astype(np.float32)
    else:
        lum = sub.astype(np.float32)
    try:
        from PIL import Image

        im = Image.fromarray(lum.clip(0, 255).astype(np.uint8), mode="L")
        im = im.resize((size, size), Image.Resampling.BILINEAR)
        v = np.asarray(im, dtype=np.float32).flatten() / 255.0
        v = v - v.mean()
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else None
    except Exception:
        return None


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b))


def best_vertical_match_dy(
    baseline_rgb: np.ndarray,
    current_rgb: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    max_shift: int = _MAX_SHIFT_SEARCH_PX,
    step: int = _SHIFT_STEP_PX,
) -> Tuple[float, int]:
    """
  Ищет на макете (baseline) патч, похожий на кроп сайта (current) при сдвиге по Y.
  Возвращает (корреляция, dy): dy>0 — контент на сайте соответствует макету ниже.
    """
    sig_c = _patch_vec(current_rgb, x, y, w, h)
    if sig_c is None:
        return 0.0, 0
    best_c, best_dy = 0.0, 0
    mh = baseline_rgb.shape[0]
    for dy in range(-max_shift, max_shift + 1, step):
        by = y + dy
        if by < 0 or by + h > mh:
            continue
        sig_b = _patch_vec(baseline_rgb, x, y, w, h) if dy == 0 else _patch_vec(baseline_rgb, x, by, w, h)
        if sig_b is None:
            continue
        c = _corr(sig_c, sig_b)
        if c > best_c:
            best_c, best_dy = c, dy
    return best_c, best_dy


def _page_heights(baseline_rgb: np.ndarray, current_rgb: np.ndarray) -> Tuple[int, int]:
    return int(baseline_rgb.shape[0]), int(current_rgb.shape[0])


def is_taller_page_shift_zone(
    baseline_rgb: np.ndarray,
    current_rgb: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    missing_on_mockup: bool,
) -> bool:
    """
    Сайт длиннее макета: контент ниже низа Figma — это сдвиг из-за вставки, не «нет на макете».
    """
    if not missing_on_mockup:
        return False
    mh, ch = _page_heights(baseline_rgb, current_rgb)
    if ch <= mh + _MIN_PAGE_EXTRA_PX:
        return False
    cy = y + max(h, 1) // 2
    if cy < mh - _BELOW_MOCKUP_MARGIN_PX:
        return False
    if _content_frac(current_rgb, x, y, w, h) < 0.10:
        return False
    corr, dy = best_vertical_match_dy(baseline_rgb, current_rgb, x, y, w, h)
    if corr >= _MATCH_CORR and dy <= -_MIN_SHIFT_PX:
        return True
    # Низ страницы на сайте, на макете в этой зоне пусто
    if cy >= mh - 8 and _content_frac(baseline_rgb, x, y, w, h) <= _ABSENT_BASE_MAX_FRAC:
        return True
    return False


_ABSENT_BASE_MAX_FRAC = 0.08


def _find_duplicate_bands(
    rgb: np.ndarray,
    *,
    min_band_h: int = 56,
    step: int = 20,
) -> List[Tuple[int, int, np.ndarray]]:
    """Полосы с похожим визуальным паттерном (две одинаковые секции на сайте)."""
    mh, mw = rgb.shape[:2]
    bands: List[Tuple[int, int, np.ndarray]] = []
    y = 0
    while y + min_band_h <= mh:
        sig = _patch_vec(rgb, 0, y, mw, min_band_h)
        if sig is not None and _content_frac(rgb, 0, y, mw, min_band_h) >= 0.12:
            bands.append((y, min_band_h, sig))
        y += step
    return bands


def is_extra_block_between_duplicate_bands(
    baseline_rgb: np.ndarray,
    current_rgb: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> bool:
    """
    Без DOM: на сайте две похожие полосы, между ними зона; на макете в той же Y — пусто/другое.
    """
    if _content_frac(baseline_rgb, x, y, w, h) > _ABSENT_BASE_MAX_FRAC:
        return False
    if _content_frac(current_rgb, x, y, w, h) < 0.10:
        return False
    mh, mw = current_rgb.shape[:2]
    bh = max(_EXTRA_BLOCK_MIN_H, min(h, 96))
    bands = _find_duplicate_bands(current_rgb, min_band_h=bh)
    if len(bands) < 2:
        return False
    cy = y + h // 2
    for i in range(len(bands) - 1):
        y1, h1, sig1 = bands[i]
        y2, h2, sig2 = bands[i + 1]
        if _corr(sig1, sig2) < _DUPLICATE_CORR:
            continue
        gap_top = y1 + h1
        gap_bot = y2
        if gap_bot - gap_top < _EXTRA_BLOCK_MIN_H:
            continue
        if not (gap_top - 12 <= cy <= gap_bot + 12):
            continue
        ca, _ = best_vertical_match_dy(baseline_rgb, current_rgb, 0, y1, mw, h1)
        cb, _ = best_vertical_match_dy(baseline_rgb, current_rgb, 0, y2, mw, h2)
        if ca >= _MATCH_CORR * 0.8 and cb >= _MATCH_CORR * 0.8:
            return True
    return False


def is_structural_shift_false_positive(
    baseline_rgb: np.ndarray,
    current_rgb: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    missing_on_mockup: bool,
    missing_on_page: bool,
) -> bool:
    """
    True = не показывать presence-баг (вставленный блок сдвинул вёрстку, дубликаты секций).
    """
    if not missing_on_mockup and not missing_on_page:
        return False
    if is_taller_page_shift_zone(
        baseline_rgb, current_rgb, x, y, w, h, missing_on_mockup=missing_on_mockup
    ):
        return True
    corr, dy = best_vertical_match_dy(baseline_rgb, current_rgb, x, y, w, h)
    if missing_on_mockup and corr >= _MATCH_CORR and abs(dy) >= _MIN_SHIFT_PX:
        return True
    if missing_on_page and corr >= _MATCH_CORR and abs(dy) >= _MIN_SHIFT_PX:
        return True
    return False


def _layout_boxes(elements: Optional[List[Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(elements, list):
        return out
    for el in elements:
        if not isinstance(el, dict):
            continue
        try:
            x, y, ew, eh = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if ew < 20 or eh < 20:
            continue
        out.append({**el, "x": x, "y": y, "w": ew, "h": eh})
    out.sort(key=lambda e: (int(e["y"]), int(e["x"])))
    return out


def _content_frac(rgb: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    mh, mw = rgb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(mw, x + w), min(mh, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub = rgb[y0:y1, x0:x1]
    lum = sub.mean(axis=2) if sub.ndim == 3 else sub.astype(np.float32)
    return float((lum < 235).mean())


def is_extra_block_between_site_duplicates(
    baseline_rgb: np.ndarray,
    current_rgb: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    layout_elements: Optional[List[Any]],
) -> bool:
    """
    На сайте между двумя похожими блоками вставлен лишний — в этой зоне пусто в Figma.
    """
    base_f = _content_frac(baseline_rgb, x, y, w, h)
    curr_f = _content_frac(current_rgb, x, y, w, h)
    if base_f > _ABSENT_BASE_MAX_FRAC or curr_f < 0.12:
        return False
    boxes = _layout_boxes(layout_elements)
    if len(boxes) < 3:
        return False
    cy = y + h // 2
    above = [b for b in boxes if int(b["y"]) + int(b["h"]) <= cy - 8]
    below = [b for b in boxes if int(b["y"]) >= cy + 8]
    if not above or not below:
        return False
    a, b = above[-1], below[0]
    sig_a = _patch_vec(current_rgb, int(a["x"]), int(a["y"]), int(a["w"]), int(a["h"]))
    sig_b = _patch_vec(current_rgb, int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]))
    if sig_a is None or sig_b is None:
        return False
    if _corr(sig_a, sig_b) < _DUPLICATE_CORR:
        return False
    gap_h = int(b["y"]) - (int(a["y"]) + int(a["h"]))
    if gap_h < _EXTRA_BLOCK_MIN_H:
        return False
    # Оба «соседа» должны где-то находиться на макете (не оба чисто сайтовые артефакты)
    ca, _ = best_vertical_match_dy(
        baseline_rgb, current_rgb, int(a["x"]), int(a["y"]), int(a["w"]), int(a["h"])
    )
    cb, _ = best_vertical_match_dy(
        baseline_rgb, current_rgb, int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
    )
    return ca >= _MATCH_CORR * 0.85 and cb >= _MATCH_CORR * 0.85


def should_suppress_presence_at_bbox(
    baseline_rgb: np.ndarray,
    current_rgb: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    missing_on_mockup: bool,
    missing_on_page: bool,
    layout_elements: Optional[List[Any]] = None,
) -> bool:
    if is_structural_shift_false_positive(
        baseline_rgb, current_rgb, x, y, w, h,
        missing_on_mockup=missing_on_mockup,
        missing_on_page=missing_on_page,
    ):
        return True
    if missing_on_mockup:
        if is_extra_block_between_site_duplicates(
            baseline_rgb, current_rgb, x, y, w, h, layout_elements
        ):
            return True
        if is_extra_block_between_duplicate_bands(baseline_rgb, current_rgb, x, y, w, h):
            return True
    return False


def filter_structural_shift_bug_items(
    items: List[Dict[str, Any]],
    *,
    baseline_rgb: Optional[np.ndarray],
    current_rgb: Optional[np.ndarray],
    layout_elements: Optional[List[Any]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Убирает presence-пункты из-за вставки блока / сдвига. Возвращает (список, сколько убрано)."""
    if baseline_rgb is None or current_rgb is None:
        return items, 0
    from src.fragment_match.similarity import is_presence_bug

    kept: List[Dict[str, Any]] = []
    removed = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text", "")).strip()
        if not is_presence_bug(text):
            kept.append(it)
            continue
        try:
            x, y, w, h = int(it["x"]), int(it["y"]), int(it["w"]), int(it["h"])
        except (KeyError, TypeError, ValueError):
            kept.append(it)
            continue
        low = text.lower()
        on_mockup = "фрагмента нет на макете" in low
        on_page = "фрагмента нет на странице" in low
        if should_suppress_presence_at_bbox(
            baseline_rgb,
            current_rgb,
            x,
            y,
            w,
            h,
            missing_on_mockup=on_mockup,
            missing_on_page=on_page,
            layout_elements=layout_elements,
        ):
            removed += 1
            continue
        kept.append(it)
    return kept, removed


def filter_items_with_paths(
    items: List[Dict[str, Any]],
    baseline_path: Optional[str],
    current_path: Optional[str],
    layout_elements: Optional[List[Any]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Фильтр по путям к PNG (после Ollama или до отчёта)."""
    if not baseline_path or not current_path:
        return items, 0
    try:
        from src.bug_reports import _load_aligned_rgb_pair

        baseline_rgb, current_rgb = _load_aligned_rgb_pair(baseline_path, current_path)
    except Exception:
        return items, 0
    if baseline_rgb is None or current_rgb is None:
        return items, 0
    return filter_structural_shift_bug_items(
        items,
        baseline_rgb=baseline_rgb,
        current_rgb=current_rgb,
        layout_elements=layout_elements,
    )
