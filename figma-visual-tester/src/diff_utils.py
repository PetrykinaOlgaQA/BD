"""
Качественная карта различий Figma vs сайт: выравнивание, blur, контраст, CNN-размер.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


@dataclass
class DiffMapResult:
    """Результат построения diff."""

    diff_display: Image.Image
    """RGB, крупный превью для UI и отчёта."""
    diff_gray_pil: Image.Image
    """Grayscale PIL (после усиления), та же геометрия что diff_display."""
    diff_gray_64: np.ndarray
    """float32 (64,64) в [0,1] для TinyDiffCNN."""
    aligned_figma: Image.Image
    aligned_website: Image.Image
    canvas_size: Tuple[int, int]


def _to_gray_np(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def _letterbox_to_canvas(
    im: Image.Image,
    canvas_w: int,
    canvas_h: int,
    fill: float = 0.92,
) -> np.ndarray:
    """Масштаб с сохранением AR, центрирование на canvas (значения 0..1)."""
    w, h = im.size
    if w <= 0 or h <= 0:
        raise ValueError("Пустое изображение")
    scale = min(canvas_w / w, canvas_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    arr = _to_gray_np(resized)
    out = np.full((canvas_h, canvas_w), fill, dtype=np.float32)
    y0 = (canvas_h - nh) // 2
    x0 = (canvas_w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = arr
    return out


def _pil_gaussian_blur_u8(img: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Размытие через Pillow (без тяжёлых окон NumPy на огромных матрицах)."""
    pil = Image.fromarray(img, mode="L")
    pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(pil, dtype=np.uint8)


def _clahe_uint8(gray_u8: np.ndarray) -> np.ndarray:
    if _HAS_CV2:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray_u8)
    # Fallback: линейное растяжение гистограммы
    lo, hi = float(gray_u8.min()), float(gray_u8.max())
    if hi <= lo + 1e-6:
        return gray_u8
    x = (gray_u8.astype(np.float32) - lo) / (hi - lo)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def create_diff_map(
    figma_pil: Image.Image,
    website_pil: Image.Image,
    target_size: Tuple[int, int] = (64, 64),
    blur_ksize: int = 3,
    contrast_gain: float = 1.35,
) -> DiffMapResult:
    """
    1) Общий canvas = max сторон (AR + padding).
    2) Абсолютная разница по яркости.
    3) Gaussian blur, усиление контраста (CLAHE / gain).
    4) Отдельное уменьшение до target_size для CNN.
    """
    tw, th = target_size
    w1, h1 = figma_pil.size
    w2, h2 = website_pil.size
    cw, ch = max(w1, w2), max(h1, h2)
    cw = max(cw, 1)
    ch = max(ch, 1)

    a = _letterbox_to_canvas(figma_pil, cw, ch, fill=0.94)
    b = _letterbox_to_canvas(website_pil, cw, ch, fill=0.94)
    raw = np.abs(a - b)

    k = max(3, int(blur_ksize) | 1)
    if _HAS_CV2:
        r8 = np.clip(raw * 255.0, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(r8, (k, k), 0)
        enhanced = _clahe_uint8(blurred)
        diff_f = enhanced.astype(np.float32) / 255.0
        diff_f = np.clip(diff_f * contrast_gain, 0.0, 1.0)
    else:
        r8 = np.clip(raw * 255.0, 0, 255).astype(np.uint8)
        blurred = _pil_gaussian_blur_u8(r8, radius=max(0.5, k / 3.0))
        enhanced = _clahe_uint8(blurred)
        diff_f = np.clip(enhanced.astype(np.float32) / 255.0 * contrast_gain, 0.0, 1.0)

    diff_u8 = np.clip(diff_f * 255.0, 0, 255).astype(np.uint8)
    diff_display = Image.fromarray(diff_u8).convert("RGB")
    diff_gray_pil = Image.fromarray(diff_u8, mode="L")

    # CNN: LANCZOS resize to tw x th
    small = diff_gray_pil.resize((tw, th), Image.Resampling.LANCZOS)
    diff_gray_64 = np.asarray(small, dtype=np.float32) / 255.0

    af = Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8), mode="L").convert("RGB")
    wb = Image.fromarray(np.clip(b * 255.0, 0, 255).astype(np.uint8), mode="L").convert("RGB")

    return DiffMapResult(
        diff_display=diff_display,
        diff_gray_pil=diff_gray_pil,
        diff_gray_64=diff_gray_64,
        aligned_figma=af,
        aligned_website=wb,
        canvas_size=(cw, ch),
    )


def diff_map_to_numpy_hw(diff_result: DiffMapResult) -> np.ndarray:
    """Массив (H,W) float для predict_diff / numpy_gray_to_tensor."""
    return np.asarray(diff_result.diff_gray_64, dtype=np.float32)
