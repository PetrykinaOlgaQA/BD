"""
Гибридная проверка текста: EasyOCR + Levenshtein + эвристики цифр.

Чувствительность: 700M→600M, %, цены, часы.
Устойчивость: косметика без смены цифр, OCR-шум.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

_NUM_TOKEN = re.compile(
    r"\d+(?:[.,]\d+)?(?:\s*[MmКкKk])?\+?%?|"
    r"\d+\s*час(?:ов|а)?",
    re.I,
)
_STAT_M = re.compile(r"\d+\s*M\+", re.I)
_PERCENT = re.compile(r"\d+\s*%")
_PRICE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:руб|₽|р\.?)", re.I)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower().replace("ё", "е"))


def levenshtein_similarity(a: str, b: str) -> float:
    """Нормированное сходство 1 - distance/max_len (через SequenceMatcher — быстрее pure Levenshtein)."""
    a, b = _normalize(a), _normalize(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def extract_numeric_tokens(text: str) -> List[str]:
    t = (text or "").replace(" ", "")
    return _NUM_TOKEN.findall(t)


def has_important_numeric_change(figma_text: str, site_text: str) -> bool:
    """
    Смена значимых цифр: 700M+↔600M+, 95%↔85%, цены, «12 часов»↔«16 часов».
    """
    a, b = figma_text or "", site_text or ""
    if _STAT_M.search(a.replace(" ", "")) and _STAT_M.search(b.replace(" ", "")):
        ma = _STAT_M.findall(a.replace(" ", ""))
        mb = _STAT_M.findall(b.replace(" ", ""))
        if ma != mb:
            return True

    ta = extract_numeric_tokens(a)
    tb = extract_numeric_tokens(b)
    if ta and tb and ta != tb:
        return True
    if (ta and not tb) or (tb and not ta):
        return len(ta) + len(tb) > 0

    # Одна доминирующая цифра в короткой строке (900 ↔ 700)
    da = re.findall(r"\d+", _normalize(a))
    db = re.findall(r"\d+", _normalize(b))
    if da and db and da != db:
        if len(da) <= 3 and len(db) <= 3:
            return True
        if _PERCENT.search(a) or _PERCENT.search(b) or _PRICE.search(a) or _PRICE.search(b):
            return True

    return False


def is_cosmetic_text_change(figma_text: str, site_text: str) -> bool:
    """
    Для снижения ложных срабатываний: «Подробнее»↔«Подробно», OCR-опечатки без смены цифр.
    """
    if has_important_numeric_change(figma_text, site_text):
        return False
    sim = levenshtein_similarity(figma_text, site_text)
    return sim >= 0.88


@lru_cache(maxsize=1)
def _ocr_reader():
    import easyocr

    return easyocr.Reader(["ru", "en"], gpu=False, verbose=False)


def ocr_crop(image_path: str | Path) -> str:
    try:
        reader = _ocr_reader()
        parts = reader.readtext(str(image_path), detail=0, paragraph=True)
        return " ".join(str(p) for p in parts)
    except Exception:
        return ""


def compare_crop_texts(
    figma_path: str | Path,
    site_path: str | Path,
    *,
    site_dom_text: str = "",
) -> Dict[str, Any]:
    """
    OCR + Levenshtein. important_numeric_change → баг даже при среднем NN.
    """
    figma_text = ocr_crop(figma_path)
    site_text = (site_dom_text or "").strip() or ocr_crop(site_path)

    lev_sim = levenshtein_similarity(figma_text, site_text)
    important = has_important_numeric_change(figma_text, site_text)
    cosmetic = is_cosmetic_text_change(figma_text, site_text)
    fig_norm = _normalize(figma_text)
    site_norm = _normalize(site_text)

    # Текст на макете есть, на сайте пусто (OCR)
    text_missing = bool(fig_norm) and len(site_norm) < 2

    # Сильное отличие без косметики
    text_different = (
        not text_missing
        and not cosmetic
        and lev_sim < 0.72
        and (fig_norm != site_norm)
    )

    if text_missing:
        return {
            "similarity": 0.0,
            "levenshtein_similarity": lev_sim,
            "mismatch": True,
            "text_missing": True,
            "text_different": False,
            "important_numeric_change": False,
            "cosmetic_only": False,
            "figma_text": figma_text,
            "site_text": site_text,
        }

    if important:
        return {
            "similarity": min(lev_sim, 0.35),
            "levenshtein_similarity": lev_sim,
            "mismatch": True,
            "important_numeric_change": True,
            "text_missing": False,
            "text_different": True,
            "cosmetic_only": False,
            "figma_text": figma_text,
            "site_text": site_text,
        }

    if cosmetic:
        return {
            "similarity": max(lev_sim, 0.92),
            "levenshtein_similarity": lev_sim,
            "mismatch": False,
            "important_numeric_change": False,
            "text_missing": False,
            "text_different": False,
            "cosmetic_only": True,
            "figma_text": figma_text,
            "site_text": site_text,
        }

    mismatch = lev_sim < 0.80 or text_different
    return {
        "similarity": float(lev_sim),
        "levenshtein_similarity": lev_sim,
        "mismatch": mismatch,
        "important_numeric_change": False,
        "text_missing": False,
        "text_different": text_different,
        "cosmetic_only": False,
        "figma_text": figma_text,
        "site_text": site_text,
    }
