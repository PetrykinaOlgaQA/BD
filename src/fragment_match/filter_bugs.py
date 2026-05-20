"""
Фильтрация ложных багов: diff есть, но фрагмент тот же (семантика + контент, не CSS).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.fragment_match.inference import FragmentMatcherHandle, score_bbox_match
from src.fragment_match.similarity import (
    calculate_similarity,
    is_presence_bug,
    is_style_only_bug,
)

_PRESENCE_MARKERS = ("фрагмента нет на макете", "фрагмента нет на странице")


def _bbox_from_item(it: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    try:
        x, y, w, h = int(it["x"]), int(it["y"]), int(it["w"]), int(it["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _skip_fragment_filter(text: str) -> bool:
    if is_presence_bug(text):
        return True
    low = (text or "").lower()
    if ("цифра:" in low or "текстовка:" in low) and "макет" in low:
        return True
    if "эмодзи не совпадает" in low or "эмодзи/иконка:" in low:
        return True
    if "логотип" in low and "котик" in low:
        return True
    if "карточка" in low and "эмодзи" in low:
        return True
    return False


def _load_rgb_pair(baseline_path: str, current_path: str):
    try:
        from src.bug_reports import _load_aligned_rgb_pair

        return _load_aligned_rgb_pair(baseline_path, current_path)
    except Exception:
        return None, None


def _should_filter_bug(
    p_same: float,
    text: str,
    *,
    threshold: float,
    scores: Dict[str, float],
) -> bool:
    """
    Убираем только шаблонные ложные пункты diff («блок не как в макете»),
    когда семантика говорит «тот же фрагмент».

    Отступы, padding, шрифт, размер кнопки — это целевые баги сверки, не фильтруем.
    """
    low = text.lower()
    if p_same < threshold:
        return False
    if is_presence_bug(text) or is_style_only_bug(text):
        return False
    if ("цифра:" in low or "текстовка:" in low) and "макет" in low:
        return False
    if "визуально отличается от макета" in low:
        return p_same >= max(threshold, 0.68)
    if "эмодзи" in low or "иконка отличается" in low:
        if "не совпадает" in low or "→" in low:
            return False
        return p_same >= max(threshold, 0.62)
    if "текст:" in low and "≠" in low:
        return False
    if "вёрстка не как в макете" in low or "блок не как в макете" in low:
        return True
    if "явно не совпадает с figma" in low and scores.get("structure", 0) >= 0.8:
        return True
    # Секционный отчёт без OCR: «сверьте с макетом» при высоком P(same) — ложное срабатывание
    if p_same >= threshold and (
        "сверьте с макетом" in low
        or "ocr:" in low
        or ("на сайте:" in low and "текст" not in low and "≠" not in low)
    ):
        return True
    if p_same >= max(threshold, 0.72) and "вёрстка раздела не совпадает" in low:
        return True
    if p_same >= threshold and "шрифт на сайте" in low:
        return True
    if "margin на сайте" in low or "padding на сайте" in low:
        if p_same >= threshold or scores.get("structure", 0) >= 0.62:
            return True
    if "блок «" in low and "не совпадает с макетом" in low:
        if scores.get("structure", 0) >= 0.6 or len(text) <= 48:
            return True
    return False


def apply_fragment_matcher_to_bug_items(
    items: List[Dict[str, Any]],
    *,
    baseline_path: str,
    current_path: str,
    diff_path: Optional[str],
    matcher: Optional[FragmentMatcherHandle],
    same_threshold: float = 0.55,
    segment_size: int = 64,
    layout_elements: Optional[List[Any]] = None,
    viewport: Optional[Tuple[int, int]] = None,
    fuzzy_px: int = 40,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Убирает ложные срабатывания: P(совпадение) по Structure×0.7 + Content×0.3, Style=0.
    """
    baseline_rgb, current_rgb = _load_rgb_pair(baseline_path, current_path)
    kept: List[Dict[str, Any]] = []
    filtered = 0
    scored = 0

    for it in items:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text", "")).strip()
        bbox = _bbox_from_item(it)
        if not bbox or _skip_fragment_filter(text):
            kept.append(it)
            continue

        p_visual: Optional[float] = None
        if matcher is not None:
            try:
                p_visual = score_bbox_match(
                    matcher,
                    baseline_path=baseline_path,
                    diff_path=diff_path,
                    current_path=current_path,
                    bbox=bbox,
                    segment_size=segment_size,
                )
            except OSError:
                p_visual = None

        try:
            scores = calculate_similarity(
                bug_item=it,
                bbox=bbox,
                layout_elements=layout_elements,
                baseline_rgb=baseline_rgb,
                current_rgb=current_rgb,
                p_visual=p_visual,
                fuzzy_px=fuzzy_px,
                viewport=viewport,
            )
        except Exception:
            kept.append(it)
            continue

        scored += 1
        p_same = float(scores["p_same"])
        row = dict(it)
        row["fragment_match_p_same"] = p_same
        row["fragment_match_structure"] = scores.get("structure")
        row["fragment_match_content"] = scores.get("content")
        row["fragment_match_visual"] = scores.get("visual")

        if _should_filter_bug(p_same, text, threshold=same_threshold, scores=scores):
            filtered += 1
            continue
        kept.append(row)

    meta = {
        "fragment_match_scored": scored,
        "fragment_match_filtered": filtered,
        "fragment_match_threshold": same_threshold,
        "fragment_match_metric": "structure*0.7+content*0.3+visual*0.12",
    }
    return kept, meta
