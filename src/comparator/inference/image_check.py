"""
Визуальные эвристики для иконок/эмодзи в кропе (без тяжёлой NN).

Сравниваем долю «чернил» и центр масс — ловим другую картинку и сильный масштаб.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image


def _ink_mask(rgb: np.ndarray, lum_thresh: float = 232.0) -> np.ndarray:
    if rgb.ndim == 3:
        lum = rgb.astype(np.float32).mean(axis=2)
    else:
        lum = rgb.astype(np.float32)
    return lum < lum_thresh


def _glyph_metrics(rgb: np.ndarray) -> Dict[str, float]:
    """Метрики глифа в кропе: fill_ratio, центр."""
    mask = _ink_mask(rgb)
    if not np.any(mask):
        return {"fill_ratio": 0.0, "cx": 0.5, "cy": 0.5}
    ys, xs = np.where(mask)
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    gw, gh = max(1, x1 - x0), max(1, y1 - y0)
    fill = float(gw * gh) / max(1, w * h)
    cx = float(xs.mean()) / max(1, w - 1)
    cy = float(ys.mean()) / max(1, h - 1)
    return {"fill_ratio": fill, "cx": cx, "cy": cy, "gw": gw, "gh": gh}


def compare_crop_images(
    figma_path: str | Path,
    site_path: str | Path,
) -> Dict[str, Any]:
    """
    Эвристики для image_mismatch:
    - missing: на сайте почти пусто, на макете есть контент
    - size_change: |fill_fig - fill_site| / max > 0.22
    - different: низкое попиксельное сходство после выравнивания центров
    """
    try:
        f = np.array(Image.open(figma_path).convert("RGB"), dtype=np.uint8)
        s = np.array(Image.open(site_path).convert("RGB"), dtype=np.uint8)
    except OSError:
        return {"mismatch": False}

    if f.shape != s.shape:
        s_img = Image.fromarray(s).resize((f.shape[1], f.shape[0]), Image.Resampling.LANCZOS)
        s = np.array(s_img, dtype=np.uint8)

    mf, ms = _glyph_metrics(f), _glyph_metrics(s)
    fill_f, fill_s = mf["fill_ratio"], ms["fill_ratio"]

    # Почти пустой сайт при контенте на макете
    if fill_f >= 0.06 and fill_s < 0.025:
        return {
            "mismatch": True,
            "reason": "missing",
            "fill_figma": fill_f,
            "fill_site": fill_s,
        }

    # Сильное изменение масштаба глифа
    denom = max(fill_f, fill_s, 0.04)
    rel_size = abs(fill_f - fill_s) / denom
    if rel_size >= 0.24 and max(fill_f, fill_s) >= 0.05:
        bigger = "крупнее" if fill_s > fill_f else "мельче"
        return {
            "mismatch": True,
            "reason": "size_change",
            "size_hint": bigger,
            "fill_figma": fill_f,
            "fill_site": fill_s,
            "rel_size_delta": rel_size,
        }

    # Пиксельное сходство (грубо)
    diff = np.abs(f.astype(np.float32) - s.astype(np.float32)).mean() / 255.0
    pixel_sim = 1.0 - diff
    if pixel_sim < 0.82 and max(fill_f, fill_s) >= 0.04:
        return {
            "mismatch": True,
            "reason": "different",
            "pixel_sim": float(pixel_sim),
            "fill_figma": fill_f,
            "fill_site": fill_s,
        }

    return {
        "mismatch": False,
        "pixel_sim": float(pixel_sim),
        "fill_figma": fill_f,
        "fill_site": fill_s,
    }
