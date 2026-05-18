from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import torch

from src.fragment_match.config import FragmentMatchConfig
from src.fragment_match.dataset import make_loaders
from src.fragment_match.losses import combined_loss
from src.fragment_match.models import build_matcher


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _negative_right_emb(right_emb: torch.Tensor) -> torch.Tensor:
    """Для triplet: правый эмбеддинг от другого примера в батче."""
    if right_emb.size(0) < 2:
        return right_emb
    perm = torch.randperm(right_emb.size(0), device=right_emb.device)
    return right_emb[perm]


@torch.no_grad()
def collect_probs_labels(
    model: torch.nn.Module, loader, device: torch.device
) -> tuple[list[float], list[int]]:
    probs: list[float] = []
    labels: list[int] = []
    model.eval()
    for batch in loader:
        panels = batch["panels"].to(device)
        y = batch["is_same"].to(device)
        logits, _, _, _ = model(panels)
        p = torch.sigmoid(logits).cpu().tolist()
        probs.extend(p)
        labels.extend(int(v) for v in y.cpu().tolist())
    return probs, labels


def calibrate_threshold(probs: list[float], labels: list[int]) -> Dict[str, float]:
    """Порог по max F1 на val (шаг 0.02)."""
    if not probs:
        return {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    best_t, best_f1 = 0.5, -1.0
    best_p, best_r = 0.0, 0.0
    for t_i in range(20, 81):
        t = t_i / 100.0
        tp = fp = fn = 0
        for p, y in zip(probs, labels):
            pred = 1 if p >= t else 0
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 1:
                fn += 1
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if f1 > best_f1:
            best_f1, best_t, best_p, best_r = f1, t, prec, rec
    return {
        "threshold": best_t,
        "f1": best_f1,
        "precision": best_p,
        "recall": best_r,
    }


@torch.no_grad()
def evaluate_matching(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs, labels = collect_probs_labels(model, loader, device)
    if not probs:
        return {"accuracy": 0.0, "n": 0.0}
    correct = sum(1 for p, y in zip(probs, labels) if (p >= threshold) == bool(y))
    return {"accuracy": correct / len(probs), "n": float(len(probs))}


def _save_checkpoint(
    path: str,
    model: torch.nn.Module,
    cfg: FragmentMatchConfig,
    *,
    threshold: float,
    val_metrics: Dict[str, float],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "encoder": cfg.model.encoder,
                "embed_dim": cfg.model.embed_dim,
                "segment_size": cfg.data.segment_size,
                "dropout": cfg.model.dropout,
                "middle_attn_penalty": cfg.model.middle_attn_penalty,
            },
            "decision_threshold": threshold,
            "val_metrics": val_metrics,
        },
        path,
    )


def train_fragment_matcher(
    cfg: FragmentMatchConfig,
    *,
    resume_path: Optional[str] = None,
) -> Dict[str, Any]:
    set_seed(cfg.train.seed)
    device = torch.device(
        cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    vit_size = 224 if cfg.model.encoder == "vit_hf" else cfg.data.segment_size
    train_loader, val_loader, test_loader = make_loaders(
        cfg.data,
        cfg.train.batch_size,
        cfg.train.num_workers,
        vit_size=vit_size,
    )
    model = build_matcher(cfg).to(device)
    if resume_path and os.path.isfile(resume_path):
        model.load_state_dict(torch.load(resume_path, map_location=device, weights_only=True))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, cfg.train.epochs),
        eta_min=cfg.train.lr * cfg.train.lr_min_ratio,
    )
    history: Dict[str, Any] = {"epochs": []}
    best_acc = 0.0
    best_threshold = 0.5
    best_val_metrics: Dict[str, float] = {}
    stale_epochs = 0

    for ep in range(cfg.train.epochs):
        model.train()
        sums = {"total": 0.0, "bce": 0.0, "contrastive": 0.0, "triplet": 0.0}
        n_batches = 0
        for batch in train_loader:
            panels = batch["panels"].to(device)
            y = batch["is_same"].to(device)
            logits, left_e, right_e, _ = model(panels, return_embeddings=True)
            neg_r = _negative_right_emb(right_e)
            losses = combined_loss(
                logits,
                left_e,
                right_e,
                y,
                neg_r,
                triplet_margin=cfg.train.triplet_margin,
                contrastive_weight=cfg.train.contrastive_weight,
                bce_weight=cfg.train.bce_weight,
                label_smoothing=cfg.train.label_smoothing,
            )
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in sums:
                sums[k] += float(losses[k].item())
            n_batches += 1

        val_probs, val_labels = collect_probs_labels(model, val_loader, device)
        cal = calibrate_threshold(val_probs, val_labels)
        thr = float(cal["threshold"])
        val_m = evaluate_matching(model, val_loader, device, threshold=thr)
        test_m = evaluate_matching(model, test_loader, device, threshold=thr)
        sched.step()
        row = {
            "epoch": ep + 1,
            "train_loss": sums["total"] / max(1, n_batches),
            "lr": float(opt.param_groups[0]["lr"]),
            "val_accuracy": val_m["accuracy"],
            "test_accuracy": test_m["accuracy"],
            "val_threshold": thr,
            "val_f1": cal["f1"],
        }
        history["epochs"].append(row)
        print(
            f"epoch {ep + 1}/{cfg.train.epochs} "
            f"loss={row['train_loss']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"test_acc={row['test_accuracy']:.4f} thr={thr:.2f} f1={cal['f1']:.3f}"
        )
        if val_m["accuracy"] >= best_acc:
            best_acc = val_m["accuracy"]
            best_threshold = thr
            best_val_metrics = {**val_m, **cal}
            stale_epochs = 0
            _save_checkpoint(
                cfg.train.out_path,
                model,
                cfg,
                threshold=best_threshold,
                val_metrics=best_val_metrics,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.train.early_stop_patience:
                print(f"early stop: val_acc не растёт {cfg.train.early_stop_patience} эпох")
                break

    history["best_val_accuracy"] = best_acc
    history["best_threshold"] = best_threshold
    history["best_val_metrics"] = best_val_metrics
    return history
