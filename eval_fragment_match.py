#!/usr/bin/env python3
"""
Оценка matching accuracy и robustness к длине/типу помехи.

  python eval_fragment_match.py --weights weights/fragment_matcher.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.fragment_match.config import FragmentMatchConfig
from src.fragment_match.dataset import FragmentMatchDataset, collate_batch, load_manifest
from src.fragment_match.models import build_matcher
from src.fragment_match.train_loop import (
    calibrate_threshold,
    collect_probs_labels,
    evaluate_matching,
)
from scripts.build_fragment_match_dataset import (
    _compose_triple,
    _load_patch,
    _noise_patch,
    collect_fragment_pool,
)


@torch.no_grad()
def eval_by_noise_type(
    model, cfg: FragmentMatchConfig, device: torch.device, *, threshold: float = 0.5
) -> Dict[str, float]:
    ds = FragmentMatchDataset(
        cfg.data.root,
        cfg.data.manifest_test,
        segment_size=cfg.data.segment_size,
        vit_size=224 if cfg.model.encoder == "vit_hf" else cfg.data.segment_size,
    )
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_batch)
    model.eval()
    by_type: Dict[str, List[float]] = {}
    for batch in loader:
        panels = batch["panels"].to(device)
        y = batch["is_same"]
        logits, _, _, _ = model(panels)
        pred = (torch.sigmoid(logits) >= threshold).cpu()
        for i, nt in enumerate(batch["noise_type"]):
            by_type.setdefault(nt, []).append(float((pred[i] == y[i]).item()))
    return {k: sum(v) / max(1, len(v)) for k, v in by_type.items()}


def eval_robustness_noise_ratio(
    model,
    pool: List,
    device: torch.device,
    panel_size: int,
    ratios: List[float],
    n_per: int = 80,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Точность при фиксированном A|noise|A, ширина помехи меняется (визуально та же, ratio в метаданных)."""
    model.eval()
    rng = random.Random(0)
    out: Dict[str, float] = {}
    for ratio in ratios:
        correct = 0
        for _ in range(n_per):
            left_path, left_id = rng.choice(pool)
            left_im = _load_patch(left_path, panel_size)
            right_im = left_im.copy()
            noise_type = rng.choice(("random", "blur", "stripe"))
            mid = _noise_patch(panel_size, rng, noise_type)
            combo = _compose_triple(left_im, mid, right_im)
            arr = np.asarray(combo.resize((panel_size * 3, panel_size)), dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            pw = panel_size
            panels = torch.stack(
                [
                    t[:, :, :, :pw],
                    t[:, :, :, pw : 2 * pw],
                    t[:, :, :, 2 * pw :],
                ],
                dim=1,
            ).to(device)
            logits, _, _, _ = model(panels)
            pred = torch.sigmoid(logits) >= threshold
            correct += int(pred.item())
        out[f"ratio_{ratio:.2f}"] = correct / n_per
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fragment_match")
    ap.add_argument("--weights", default="weights/fragment_matcher.pt")
    ap.add_argument("--encoder", choices=("cnn", "vit_hf"), default="cnn")
    args = ap.parse_args()

    cfg = FragmentMatchConfig()
    cfg.data.root = args.data
    cfg.model.encoder = args.encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model = build_matcher(cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    saved_thr = float(ckpt.get("decision_threshold", 0.5)) if isinstance(ckpt, dict) else 0.5

    from src.fragment_match.dataset import make_loaders

    _, val_loader, test_loader = make_loaders(
        cfg.data,
        batch_size=64,
        vit_size=224 if cfg.model.encoder == "vit_hf" else cfg.data.segment_size,
        train_augment=False,
    )
    val_probs, val_labels = collect_probs_labels(model, val_loader, device)
    cal = calibrate_threshold(val_probs, val_labels)
    thr = float(cal["threshold"])
    base = evaluate_matching(model, test_loader, device, threshold=thr)
    base_default = evaluate_matching(model, test_loader, device, threshold=0.5)
    by_noise = eval_by_noise_type(model, cfg, device, threshold=thr)
    pool = collect_fragment_pool()
    robust = eval_robustness_noise_ratio(
        model, pool, device, cfg.data.segment_size, [0.2, 0.33, 0.45, 0.6], n_per=60, threshold=thr
    )

    report = {
        "test_accuracy_at_calibrated_threshold": base["accuracy"],
        "test_accuracy_at_0_5": base_default["accuracy"],
        "calibrated_threshold": thr,
        "saved_checkpoint_threshold": saved_thr,
        "val_calibration": cal,
        "accuracy_by_noise_type": by_noise,
        "robustness_same_A_by_noise_ratio": robust,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out_path = os.path.join(args.data, "eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
