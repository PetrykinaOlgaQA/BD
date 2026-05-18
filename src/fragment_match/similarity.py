"""
Сопоставление фрагментов: семантика + контент, без учёта CSS-стилей (Style weight = 0).

Этап 1 — семантика (тег, роль: кнопка/карточка/подвал).
Этап 2 — контент (плотность «непустых» пикселей в кропе макета и сайта).
Этап 3 — визуальная CNN-оценка с малым весом; padding/margin/font не снижают итог.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Веса итоговой метрики (Style не входит в сумму)
W_STRUCTURE = 0.7
W_CONTENT = 0.3
W_STYLE = 0.0
W_VISUAL = 0.12  # доп. сигнал от CNN, не доминирует

DEFAULT_FUZZY_PX = 40

_PRESENCE_MARKERS = ("фрагмента нет на макете", "фрагмента нет на странице")

_STYLE_BUG_MARKERS = (
    "padding",
    "margin",
    "отступ",
    "шрифт",
    "font",
    "line-height",
    "px больше",
    "px меньше",
    "внутренний отступ",
    "размер блока",
    "размер шрифта",
    "размер не совпадает",
    "размер меньше",
    "размер больше",
    "не совпадает с макетом",
    "выровнять",
    "центр",
)

_ROLE_FROM_TEXT = (
    (("кнопк", "button", "btn"), "button"),
    (("карточ", "card"), "card"),
    (("шапк", "header", "nav", "меню"), "header"),
    (("подвал", "footer"), "footer"),
    (("баннер", "banner"), "banner"),
    (("заголов", "h1", "h2", "h3"), "heading"),
    (("картин", "изображ", "img", "picture"), "image"),
    (("текст", "п.", "span"), "text"),
)

_TAG_BUTTON = frozenset({"button", "a"})
_TAG_HEADING = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_TAG_TEXT = frozenset({"p", "span", "label"})
_TAG_IMAGE = frozenset({"img", "picture", "svg"})


def _parse_tag(snippet: str) -> str:
    s = (snippet or "").strip().lower()
    if not s:
        return ""
    return s.split(".")[0].split("#")[0]


def _semantic_role(snippet: str, bug_text: str = "") -> str:
    blob = f"{snippet} {bug_text}".lower()
    for needles, role in _ROLE_FROM_TEXT:
        if any(n in blob for n in needles):
            return role
    tag = _parse_tag(snippet)
    if tag in _TAG_BUTTON:
        return "button"
    if tag in _TAG_HEADING:
        return "heading"
    if tag in _TAG_IMAGE:
        return "image"
    if tag in _TAG_TEXT:
        return "text"
    if tag:
        return "block"
    return "unknown"


def _roles_compatible(a: str, b: str) -> float:
    if a == b and a != "unknown":
        return 1.0
    if a == "unknown" or b == "unknown":
        return 0.55
    pairs = (
        frozenset({"button", "text"}),
        frozenset({"heading", "text"}),
        frozenset({"card", "block"}),
        frozenset({"image", "block"}),
    )
    if frozenset({a, b}) in pairs:
        return 0.82
    return 0.25


def _tags_compatible(ta: str, tb: str) -> float:
    if not ta or not tb:
        return 0.5
    if ta == tb:
        return 1.0
    if ta in _TAG_BUTTON and tb in _TAG_BUTTON:
        return 0.95
    if ta in _TAG_HEADING and tb in _TAG_HEADING:
        return 0.95
    return 0.4 if ta[0] == tb[0] else 0.3


def is_presence_bug(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _PRESENCE_MARKERS)


def is_style_only_bug(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _STYLE_BUG_MARKERS)


def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w * 0.5, y + h * 0.5


def _normalize_bbox(
    bbox: Tuple[int, int, int, int],
    viewport: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    vw, vh = max(1, viewport[0]), max(1, viewport[1])
    x, y, w, h = bbox
    return x / vw, y / vh, w / vw, h / vh


def _bbox_fuzzy_distance(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    """Расстояние между центрами + разница размеров (нормализовано)."""
    cx1, cy1 = _bbox_center(a)
    cx2, cy2 = _bbox_center(b)
    dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    size_pen = abs(a[2] - b[2]) + abs(a[3] - b[3])
    return dist + size_pen * 0.25


def find_layout_element_fuzzy(
    elements: List[Any],
    bbox: Tuple[int, int, int, int],
    *,
    fuzzy_px: int = DEFAULT_FUZZY_PX,
) -> Optional[Dict[str, Any]]:
    """Кандидат по DOM: тот же блок ± fuzzy_px (этап 1)."""
    if not elements:
        return None
    best: Optional[Dict[str, Any]] = None
    best_d = 1e18
    for el in elements:
        if not isinstance(el, dict):
            continue
        try:
            eb = (int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"]))
        except (KeyError, TypeError, ValueError):
            continue
        d = _bbox_fuzzy_distance(bbox, eb)
        if d < best_d:
            best_d = d
            best = el
    if best is None or best_d > fuzzy_px * 2.5:
        return None
    return best


def _box_content_frac(rgb, x0: int, y0: int, w: int, h: int) -> float:
    try:
        import numpy as np
    except ImportError:
        return 0.0
    mh, mw = rgb.shape[:2]
    x1, y1 = min(mw, x0 + w), min(mh, y0 + h)
    x0, y0 = max(0, x0), max(0, y0)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub = rgb[y0:y1, x0:x1]
    lum = sub.mean(axis=2) if sub.ndim == 3 else sub.astype(np.float32)
    return float((lum < 235).mean())


def score_structure(
    bug_item: Dict[str, Any],
    layout_elements: Optional[List[Any]],
    bbox: Tuple[int, int, int, int],
    *,
    fuzzy_px: int = DEFAULT_FUZZY_PX,
) -> float:
    """Этап 1: тег/роль элемента, fuzzy-позиция."""
    snippet = str(bug_item.get("snippet", "")).strip()
    text = str(bug_item.get("text", "")).strip()
    role_bug = _semantic_role(snippet, text)
    tag_bug = _parse_tag(snippet)

    el = find_layout_element_fuzzy(layout_elements or [], bbox, fuzzy_px=fuzzy_px)
    if el is None:
        if role_bug != "unknown":
            return 0.72
        return 0.5

    sn_el = str(el.get("snippet", "")).strip()
    role_el = _semantic_role(sn_el, "")
    tag_el = _parse_tag(sn_el)

    role_s = _roles_compatible(role_bug, role_el)
    tag_s = _tags_compatible(tag_bug, tag_el)

    try:
        eb = (int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"]))
    except (KeyError, TypeError, ValueError):
        eb = bbox
    shift = _bbox_fuzzy_distance(bbox, eb)
    pos_s = 1.0 if shift <= fuzzy_px else max(0.0, 1.0 - (shift - fuzzy_px) / max(fuzzy_px * 2, 1))

    return float(0.45 * role_s + 0.25 * tag_s + 0.30 * pos_s)


def score_content(
    baseline_rgb,
    current_rgb,
    bbox: Tuple[int, int, int, int],
) -> float:
    """Этап 2: оба кропа содержат контент (не «пустой» vs «полный»)."""
    if baseline_rgb is None or current_rgb is None:
        return 0.55
    x, y, w, h = bbox
    bf = _box_content_frac(baseline_rgb, x, y, w, h)
    cf = _box_content_frac(current_rgb, x, y, w, h)
    if bf <= 0.04 and cf <= 0.04:
        return 1.0
    if (bf <= 0.04) != (cf <= 0.04):
        return 0.15
    denom = max(bf, cf, 0.05)
    density_sim = 1.0 - min(1.0, abs(bf - cf) / denom)
    return float(0.35 + 0.65 * density_sim)


def score_content_with_text(
    baseline_rgb,
    current_rgb,
    bbox: Tuple[int, int, int, int],
    layout_el: Optional[Dict[str, Any]],
) -> float:
    """Этап 2: плотность контента в кропе + наличие текста в DOM (без margin/padding)."""
    base = score_content(baseline_rgb, current_rgb, bbox)
    if not layout_el:
        return base
    inner = str(layout_el.get("innerText", "") or layout_el.get("text", "")).strip()
    if len(inner) >= 2 and base >= 0.4:
        return min(1.0, 0.55 * base + 0.45)
    return base


def calculate_similarity(
    *,
    bug_item: Dict[str, Any],
    bbox: Tuple[int, int, int, int],
    layout_elements: Optional[List[Any]] = None,
    baseline_rgb=None,
    current_rgb=None,
    p_visual: Optional[float] = None,
    fuzzy_px: int = DEFAULT_FUZZY_PX,
    viewport: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """
    Итог: Structure×0.7 + Content×0.3 + Visual×0.12 (Style=0, не используется).

    Возвращает компоненты и p_same — вероятность «тот же фрагмент».
    """
    text = str(bug_item.get("text", "")).strip()
    if is_presence_bug(text):
        el = find_layout_element_fuzzy(layout_elements or [], bbox, fuzzy_px=fuzzy_px)
        s_struct = score_structure(bug_item, layout_elements, bbox, fuzzy_px=fuzzy_px)
        s_content = score_content_with_text(baseline_rgb, current_rgb, bbox, el)
        p_sem = W_STRUCTURE * s_struct + W_CONTENT * s_content
        p_same = min(p_sem, 0.35)
        return {
            "p_same": round(p_same, 4),
            "structure": round(s_struct, 4),
            "content": round(s_content, 4),
            "style": 0.0,
            "visual": round(float(p_visual or 0.0), 4),
        }

    el = find_layout_element_fuzzy(layout_elements or [], bbox, fuzzy_px=fuzzy_px)
    s_struct = score_structure(bug_item, layout_elements, bbox, fuzzy_px=fuzzy_px)
    s_content = score_content_with_text(baseline_rgb, current_rgb, bbox, el)
    s_style = 0.0

    p_semantic = W_STRUCTURE * s_struct + W_CONTENT * s_content + W_STYLE * s_style
    p_vis = float(p_visual) if p_visual is not None else 0.5
    p_same = (1.0 - W_VISUAL) * p_semantic + W_VISUAL * p_vis

    # Реальные правки вёрстки (отступы, шрифт) не считаем «тем же фрагментом» для авто-отсечения.
    if is_style_only_bug(text):
        p_same = min(p_same, 0.45)

    if viewport:
        _ = _normalize_bbox(bbox, viewport)

    return {
        "p_same": round(min(1.0, max(0.0, p_same)), 4),
        "structure": round(s_struct, 4),
        "content": round(s_content, 4),
        "style": s_style,
        "visual": round(p_vis, 4),
    }
