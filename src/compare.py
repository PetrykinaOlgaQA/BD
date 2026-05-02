from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

# Сравнение «как есть» на 20+ Mpx даёт сотни МБ в float32 и падает на ноутбуке.
_MAX_COMPARE_PIXELS = 8_000_000
_MSE_CHUNK = 2_000_000


@dataclass
class CompareResult:
    mse: float
    changed_ratio: float
    width: int
    height: int
    diff_path: str | None
    changed_ratio_raw: float = 0.0
    changed_ratio_shift: float = 0.0
    tolerance_shift_px: int = 0
    tolerance_speckle_iter: int = 0


def _to_rgb(im: Image.Image) -> Image.Image:
    if im.mode != "RGB":
        return im.convert("RGB")
    return im


def _mse_from_diff_image(diff: Image.Image) -> float:
    """MSE эталона и текущего по карте |a-b| без float32 на весь RGB."""
    d = np.asarray(diff.convert("RGB"), dtype=np.uint8).reshape(-1)
    acc = 0.0
    for i in range(0, d.size, _MSE_CHUNK):
        x = d[i : i + _MSE_CHUNK].astype(np.float64)
        acc += float(np.dot(x, x))
    return (acc / max(1, d.size)) / (255.0**2)


def _shift_min_diff(a_hwc: np.ndarray, b_hwc: np.ndarray, t: int) -> np.ndarray:
    h, w, _ = a_hwc.shape
    b = b_hwc.astype(np.int16)
    if t <= 0:
        d = np.abs(a_hwc.astype(np.int16) - b)
        return np.max(d, axis=2).astype(np.float32)
    ap = np.pad(a_hwc.astype(np.int16), ((t, t), (t, t), (0, 0)), mode="edge")
    min_d = np.full((h, w), np.inf, dtype=np.float32)
    for dy in range(2 * t + 1):
        for dx in range(2 * t + 1):
            sub = ap[dy : dy + h, dx : dx + w, :]
            d = np.max(np.abs(sub - b), axis=2).astype(np.float32)
            min_d = np.minimum(min_d, d)
    return min_d


def _downscale_if_huge(a: Image.Image, b: Image.Image) -> Tuple[Image.Image, Image.Image]:
    w, h = a.size
    if w * h <= _MAX_COMPARE_PIXELS:
        return a, b
    scale = (_MAX_COMPARE_PIXELS / float(w * h)) ** 0.5
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return (
        a.resize((nw, nh), Image.Resampling.LANCZOS),
        b.resize((nw, nh), Image.Resampling.LANCZOS),
    )


def _opening_binary(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask
    im = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    for _ in range(iterations):
        im = im.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    return np.asarray(im, dtype=np.uint8) > 127


def compare_screenshots(
    baseline_path: str,
    current_path: str,
    out_dir: str,
    tag: str = "diff",
    pixel_threshold: int = 30,
    tolerance_shift_px: int = 0,
    tolerance_speckle_iter: int = 0,
) -> CompareResult:
    tolerance_shift_px = max(0, min(5, int(tolerance_shift_px)))
    tolerance_speckle_iter = max(0, min(5, int(tolerance_speckle_iter)))
    os.makedirs(out_dir, exist_ok=True)
    a = _to_rgb(Image.open(baseline_path))
    b = _to_rgb(Image.open(current_path))
    if a.size != b.size:
        b = b.resize(a.size, Image.Resampling.LANCZOS)
    a, b = _downscale_if_huge(a, b)
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    diff = ImageChops.difference(a, b)
    mse = _mse_from_diff_image(diff)
    gray = diff.convert("L")
    g = np.asarray(gray, dtype=np.uint8)
    thr_u8 = int(pixel_threshold)
    changed_raw = float(np.mean(g > thr_u8))
    min_d = _shift_min_diff(a_np, b_np, tolerance_shift_px)
    if tolerance_shift_px <= 0:
        mask_shift = g > thr_u8
        changed_shift = changed_raw
    else:
        mask_shift = min_d > pixel_threshold
        changed_shift = float(np.mean(mask_shift))
    if tolerance_shift_px <= 0 and tolerance_speckle_iter <= 0:
        mask_final = mask_shift
        changed = changed_raw
    else:
        mask_final = _opening_binary(mask_shift, tolerance_speckle_iter)
        changed = float(np.mean(mask_final))
    base = os.path.splitext(os.path.basename(current_path))[0]
    diff_path = os.path.join(out_dir, f"{tag}_{base}.png")
    vis = np.clip(min_d * 2.0, 0, 255).astype(np.uint8)
    heat = Image.fromarray(vis, mode="L")
    overlay = Image.blend(a, Image.merge("RGB", (heat, heat, heat)), 0.35)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, 0, a.size[0] - 1, a.size[1] - 1], outline=(255, 0, 0), width=2)
    overlay.save(diff_path)
    return CompareResult(
        mse=mse,
        changed_ratio=changed,
        width=a.size[0],
        height=a.size[1],
        diff_path=diff_path,
        changed_ratio_raw=changed_raw,
        changed_ratio_shift=changed_shift,
        tolerance_shift_px=tolerance_shift_px,
        tolerance_speckle_iter=tolerance_speckle_iter,
    )


def diff_tensor_gray(diff_path: str, size: Tuple[int, int] = (64, 64)):
    import torch

    im = Image.open(diff_path).convert("L").resize(size, Image.Resampling.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
