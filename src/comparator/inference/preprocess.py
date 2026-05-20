"""
Подготовка кропов Figma / Site перед MultiAspectComparator.

Для снижения ложных срабатываний:
- phase cross-correlation (skimage → OpenCV → перебор), max_shift=15;
- лёгкое Gaussian blur (антиалиасинг, субпиксель).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


def _clamp_shift(dx: float, dy: float, max_shift: int) -> Tuple[int, int]:
    lim = int(max_shift)
    return (
        int(max(-lim, min(lim, round(dx)))),
        int(max(-lim, min(lim, round(dy)))),
    )


def _apply_shift(img: Image.Image, dx: int, dy: int) -> Image.Image:
    if dx == 0 and dy == 0:
        return img
    return img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        fillcolor=(255, 255, 255),
    )


def align_images_skimage(
    figma: Image.Image,
    site: Image.Image,
    max_shift: int = 15,
) -> Tuple[Image.Image, Image.Image, Tuple[int, int]]:
    """phase_cross_correlation (skimage) — эталон: figma, движущееся: site."""
    if site.size != figma.size:
        site = site.resize(figma.size, Image.Resampling.LANCZOS)

    f_gray = np.asarray(figma.convert("L"), dtype=np.float64)
    s_gray = np.asarray(site.convert("L"), dtype=np.float64)

    try:
        from skimage.registration import phase_cross_correlation

        shift, _error, _ = phase_cross_correlation(
            f_gray,
            s_gray,
            upsample_factor=10,
        )
        dy, dx = float(shift[0]), float(shift[1])
        dx_i, dy_i = _clamp_shift(dx, dy, max_shift)
        site = _apply_shift(site, dx_i, dy_i)
        return figma, site, (dx_i, dy_i)
    except Exception:
        return align_images_phase(figma, site, max_shift=max_shift)


def align_images_phase(
    figma: Image.Image,
    site: Image.Image,
    max_shift: int = 15,
) -> Tuple[Image.Image, Image.Image, Tuple[int, int]]:
    """OpenCV phaseCorrelate (fallback)."""
    if site.size != figma.size:
        site = site.resize(figma.size, Image.Resampling.LANCZOS)

    f_gray = np.asarray(figma.convert("L"), dtype=np.float32)
    s_gray = np.asarray(site.convert("L"), dtype=np.float32)

    try:
        import cv2

        (shift_x, shift_y), _response = cv2.phaseCorrelate(f_gray, s_gray)
        dx, dy = _clamp_shift(shift_x, shift_y, max_shift)
        site = _apply_shift(site, dx, dy)
        return figma, site, (dx, dy)
    except Exception:
        return align_images_exhaustive(figma, site, max_shift=max_shift)


def align_images_exhaustive(
    figma: Image.Image,
    site: Image.Image,
    max_shift: int = 15,
) -> Tuple[Image.Image, Image.Image, Tuple[int, int]]:
    """Перебор целочисленного сдвига (последний fallback)."""
    if site.size != figma.size:
        site = site.resize(figma.size, Image.Resampling.LANCZOS)

    f_gray = np.asarray(figma.convert("L"), dtype=np.float32)
    best_dx, best_dy = 0, 0
    best_mae = float("inf")
    lim = int(max_shift)

    for dy in range(-lim, lim + 1):
        for dx in range(-lim, lim + 1):
            shifted = _apply_shift(site, dx, dy)
            s_gray = np.asarray(shifted.convert("L"), dtype=np.float32)
            mae = float(np.mean(np.abs(f_gray - s_gray)))
            if mae < best_mae:
                best_mae = mae
                best_dx, best_dy = dx, dy

    site = _apply_shift(site, best_dx, best_dy)
    return figma, site, (best_dx, best_dy)


def align_images(
    figma: Image.Image,
    site: Image.Image,
    max_shift: int = 15,
    *,
    method: str = "auto",
) -> Tuple[Image.Image, Image.Image, Tuple[int, int]]:
    """
    Выравнивает site относительно figma (компенсация сдвига до max_shift px).
    method: auto | skimage | phase | exhaustive
    """
    if method == "exhaustive":
        return align_images_exhaustive(figma, site, max_shift=max_shift)
    if method == "phase":
        return align_images_phase(figma, site, max_shift=max_shift)
    if method == "skimage":
        return align_images_skimage(figma, site, max_shift=max_shift)

    # auto: skimage → OpenCV → перебор
    try:
        from skimage.registration import phase_cross_correlation  # noqa: F401

        return align_images_skimage(figma, site, max_shift=max_shift)
    except ImportError:
        pass
    try:
        import cv2  # noqa: F401

        return align_images_phase(figma, site, max_shift=max_shift)
    except ImportError:
        return align_images_exhaustive(figma, site, max_shift=max_shift)


def anti_alias_blur(img: Image.Image, radius: float = 0.8) -> Image.Image:
    """Для снижения ложных срабатываний на антиалиасинге и субпиксельном шуме."""
    if radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=float(radius)))


def prepare_crop_pair(
    figma: Image.Image,
    site: Image.Image,
    *,
    max_shift: int = 15,
    blur_radius: float = 0.8,
    align: bool = True,
    align_method: str = "auto",
) -> Tuple[Image.Image, Image.Image, Tuple[int, int]]:
    """Полный preprocessing перед нейросетью."""
    shift = (0, 0)
    if align:
        figma, site, shift = align_images(
            figma, site, max_shift=max_shift, method=align_method
        )
    if blur_radius > 0:
        figma = anti_alias_blur(figma, blur_radius)
        site = anti_alias_blur(site, blur_radius)
    return figma, site, shift
