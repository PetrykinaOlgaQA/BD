"""
Инференс: [фрагмент макета | diff-помеха | фрагмент сайта] → P(тот же блок).

DOMAIN: image — кропы PNG. Для text/code подставьте свой энкодер и сборку панелей.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np
import torch

FragmentMatcherHandle = Tuple[str, Any]


def _crop_rgb_panel(path: str, bbox: Tuple[int, int, int, int], size: int) -> np.ndarray:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    x, y, w, h = bbox
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(im.width, x + w), min(im.height, y + h)
    if x1 <= x0 or y1 <= y0:
        arr = np.zeros((size, size, 3), dtype=np.float32)
        return arr
    crop = im.crop((x0, y0, x1, y1))
    crop = crop.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(crop, dtype=np.float32) / 255.0


def panels_from_paths(
    baseline_path: str,
    diff_path: str,
    current_path: str,
    bbox: Tuple[int, int, int, int],
    segment_size: int = 64,
) -> torch.Tensor:
    """Одна тройка 1×3×3×H×W."""
    left = _crop_rgb_panel(baseline_path, bbox, segment_size)
    mid = _crop_rgb_panel(diff_path, bbox, segment_size) if diff_path else np.zeros_like(left)
    right = _crop_rgb_panel(current_path, bbox, segment_size)
    stacked = np.stack([left, mid, right], axis=0)
    return torch.from_numpy(stacked).permute(0, 3, 1, 2).unsqueeze(0)


def load_fragment_matcher(path: str | None) -> Tuple[Optional[FragmentMatcherHandle], bool]:
    if not path or not os.path.isfile(path):
        return None, False
    try:
        from src.fragment_match.config import FragmentMatchConfig
        from src.fragment_match.models import build_matcher
    except ImportError:
        return None, False
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    except OSError:
        return None, False

    cfg = FragmentMatchConfig()
    meta = ckpt.get("config") if isinstance(ckpt, dict) else {}
    if isinstance(meta, dict):
        if meta.get("encoder"):
            cfg.model.encoder = str(meta["encoder"])
        if meta.get("embed_dim"):
            cfg.model.embed_dim = int(meta["embed_dim"])
        if meta.get("segment_size"):
            cfg.data.segment_size = int(meta["segment_size"])

    model = build_matcher(cfg)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state)
    except Exception:
        return None, False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    threshold = 0.5
    if isinstance(ckpt, dict) and ckpt.get("decision_threshold") is not None:
        threshold = float(ckpt["decision_threshold"])
    return ("torch", (model, cfg, device, threshold)), True


@torch.no_grad()
def predict_same_prob(handle: FragmentMatcherHandle, panels: torch.Tensor) -> float:
    """P(is_same=1): левый и правый фрагмент совпадают несмотря на помеху посередине."""
    _, (model, _cfg, device, _thr) = handle
    x = panels.to(device)
    logits, _, _, _ = model(x)
    return float(torch.sigmoid(logits).item())


def matcher_decision_threshold(handle: FragmentMatcherHandle) -> float:
    _, (_, _, _, thr) = handle
    return float(thr)


def score_bbox_match(
    handle: FragmentMatcherHandle,
    *,
    baseline_path: str,
    diff_path: Optional[str],
    current_path: str,
    bbox: Tuple[int, int, int, int],
    segment_size: int = 64,
) -> float:
    diff_p = diff_path if diff_path and os.path.isfile(diff_path) else baseline_path
    panels = panels_from_paths(
        baseline_path, diff_p, current_path, bbox, segment_size=segment_size
    )
    return predict_same_prob(handle, panels)
