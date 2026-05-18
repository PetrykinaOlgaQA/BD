"""Кэш текста макета (OCR по разделам) для сравнения с DOM сайта без Tesseract на каждом прогоне."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.section_compare import _group_site_sections, _ocr_text, _text_similar  # noqa: PLC0415
from src.typography_compare import _crop_rgb, _normalize_text, _short


def _cache_path(baseline_png: str) -> str:
    return baseline_png + ".text.json"


def load_baseline_text_cache(baseline_png: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(baseline_png)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    try:
        mtime = float(data.get("baseline_mtime", 0))
    except (TypeError, ValueError):
        mtime = 0.0
    try:
        cur = os.path.getmtime(baseline_png)
    except OSError:
        return None
    if abs(cur - mtime) > 1.0:
        return None
    return data


def build_baseline_text_cache(
    baseline_png: str,
    baseline_rgb: np.ndarray,
    layout_elements: Optional[List[Any]],
) -> Dict[str, Any]:
    """OCR по разделам макета; сохраняет JSON рядом с PNG."""
    sections_out: List[Dict[str, str]] = []
    if isinstance(layout_elements, list) and layout_elements:
        for sec in _group_site_sections(layout_elements):
            bb = sec.get("bbox") or {}
            try:
                x, y, w, h = int(bb["x"]), int(bb["y"]), int(bb["w"]), int(bb["h"])
            except (KeyError, TypeError, ValueError):
                continue
            crop = _crop_rgb(baseline_rgb, x, y, w, h, pad=0)
            if crop is None:
                continue
            txt = _ocr_text(crop)
            if txt:
                sections_out.append(
                    {
                        "id": str(sec.get("id", "")),
                        "title": str(sec.get("title", "")),
                        "text": txt,
                    }
                )
    if not sections_out:
        txt = _ocr_text(baseline_rgb)
        if txt:
            sections_out.append({"id": "full", "title": "Страница", "text": txt})
    try:
        mtime = os.path.getmtime(baseline_png)
    except OSError:
        mtime = 0.0
    data = {"baseline_mtime": mtime, "sections": sections_out}
    path = _cache_path(baseline_png)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
    return data


def ensure_baseline_text_cache(
    baseline_png: str,
    baseline_rgb: np.ndarray,
    layout_elements: Optional[List[Any]],
) -> Optional[Dict[str, Any]]:
    cached = load_baseline_text_cache(baseline_png)
    if cached and cached.get("sections"):
        return cached
    built = build_baseline_text_cache(baseline_png, baseline_rgb, layout_elements)
    return built if built.get("sections") else None


def section_text_from_cache(cache: Optional[Dict[str, Any]], section_id: str) -> str:
    if not cache:
        return ""
    for row in cache.get("sections") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("id", "")) == section_id:
            return str(row.get("text", "") or "").strip()
    return ""


def compare_site_to_cached_section(
    cache: Optional[Dict[str, Any]],
    section_id: str,
    site_text: str,
    *,
    title: str,
    min_ratio: float = 0.82,
) -> Optional[str]:
    mock = section_text_from_cache(cache, section_id)
    if not mock or not site_text:
        return None
    if _text_similar(mock, site_text) < min_ratio:
        return (
            f"[{title}] текст: макет «{_short(mock, 50)}» ≠ сайт «{_short(site_text, 50)}»"
        )
    return None
