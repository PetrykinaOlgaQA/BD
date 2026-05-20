"""
Пост-обработка багов comparator: отсечение ложных срабатываний.

Не показываем: padding, сдвиг ≤15px (уже в align), лёгкий цвет иконок, косметику текста.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_IDENTICAL_MIN = 0.72
_LAYOUT_BORDERLINE = (0.48, 0.58)
# Лёгкий цвет иконки при умеренном image_match
_ICON_COLOR_MIN = 0.55


def _bug_types(bugs: List[Dict[str, Any]]) -> set[str]:
    return {str(b.get("type", "")) for b in bugs}


def apply_comparator_rules(
    scores: Dict[str, float],
    bugs: List[Dict[str, Any]],
    *,
    region: Optional[Dict[str, Any]] = None,
    ocr_result: Optional[Dict[str, Any]] = None,
    image_result: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    s = scores
    out = list(bugs)
    ocr_result = ocr_result or {}
    image_result = image_result or {}
    region = region or {}
    snippet = str(region.get("snippet", "")).lower()

    # Косметика текста без смены цифр
    if ocr_result.get("cosmetic_only") and not ocr_result.get("important_numeric_change"):
        out = [b for b in out if b.get("type") != "text_mismatch"]

    # OCR: пропажа / цифры — гарантированный text-баг
    if ocr_result.get("text_missing") or ocr_result.get("important_numeric_change"):
        if not any(b.get("type") == "text_mismatch" for b in out):
            block = snippet.split(".")[-1] if snippet else "блок"
            out.append({
                "type": "text_mismatch",
                "severity": "high",
                "title": "Текст отличается от макета",
                "description": f"[{block}] текст не совпадает с макетом",
                "nn_score": round(float(ocr_result.get("similarity", 0.25)), 3),
                "source": "ocr",
            })

    # Эвристика изображения сильнее NN — не режем image_mismatch
    if not image_result.get("mismatch"):
        # Лёгкий цвет / антиалиасинг иконки
        if any(x in snippet for x in ("emoji", "fact-emoji", "icon")):
            if s["image_match"] >= _ICON_COLOR_MIN and s["overall_similarity"] >= 0.62:
                out = [b for b in out if b.get("type") != "image_mismatch"]

    if not out:
        return out

    min_aspect = min(
        s["text_match"],
        s["layout_match"],
        s["typography_match"],
        s["color_match"],
        s["image_match"],
    )

    # Все аспекты высокие — убираем visual
    if "visual_mismatch" in _bug_types(out):
        if min_aspect >= _IDENTICAL_MIN and s["overall_similarity"] >= 0.50:
            out = [b for b in out if b.get("type") != "visual_mismatch"]
        elif min_aspect >= 0.68 and s["text_match"] >= 0.88:
            out = [b for b in out if b.get("type") != "visual_mismatch"]

    # Пограничный layout при хорошем тексте (после align ≤15px)
    lo, hi = _LAYOUT_BORDERLINE
    if "layout_mismatch" in _bug_types(out):
        if lo <= s["layout_match"] < hi and s["text_match"] >= 0.85:
            out = [b for b in out if b.get("type") != "layout_mismatch"]
        if s["text_match"] >= 0.90 and s["image_match"] >= 0.78:
            out = [b for b in out if b.get("type") != "layout_mismatch"]

    # Шапка: слабый overall без явного контента
    if "visual_mismatch" in _bug_types(out) and any(x in snippet for x in ("logo", "header")):
        if s["text_match"] >= 0.85 and s["image_match"] >= 0.70 and not image_result.get("mismatch"):
            out = [b for b in out if b.get("type") != "visual_mismatch"]

    # Только image — не дублировать visual
    if "visual_mismatch" in _bug_types(out):
        if s["image_match"] < 0.55 and s["image_match"] <= min_aspect + 0.08:
            out = [b for b in out if b.get("type") != "visual_mismatch"]
        elif s["image_match"] < min(s["text_match"], s["layout_match"], s["typography_match"]) - 0.12:
            out = [b for b in out if b.get("type") != "visual_mismatch"]

    return out
