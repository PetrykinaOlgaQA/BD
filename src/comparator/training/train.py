"""Обучение MultiAspectComparator."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.comparator.models.multi_aspect import (
    ASPECT_KEYS,
    ComparatorConfig,
    MultiAspectComparator,
    load_comparator,
    save_comparator,
)
from src.comparator.training.dataset import ComparatorDataset
from src.comparator.training.rico_dataset import RicoDataset
from src.comparator.training.metrics import compute_metrics


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _config_from_yaml(raw: dict) -> ComparatorConfig:
    m = raw.get("model", {})
    return ComparatorConfig(
        image_size=int(m.get("input_size", 224)),
        in_channels=int(m.get("num_channels", 4)),
        dropout=float(m.get("dropout", 0.25)),
        pretrained_backbone=bool(m.get("pretrained", True)),
        freeze_backbone=bool(m.get("freeze_backbone", True)),
    )


def _aspect_loss_weights(config: dict) -> Dict[str, float]:
    tr = config.get("training", {})
    w = {k: 1.0 for k in ASPECT_KEYS}
    w["overall_similarity"] = float(tr.get("overall_weight", 2.0))
    w["text_match"] = float(tr.get("text_weight", 2.5))
    return w


def train() -> None:
    root = _project_root()
    config_path = root / "config" / "comparator.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    image_size = int(config["model"].get("input_size", 224))
    paths_cfg = config.get("paths", {})
    train_manifest = paths_cfg.get("manifest_train", "data/comparator/manifest_train.jsonl")
    val_manifest = paths_cfg.get("manifest_val", "data/comparator/manifest_val.jsonl")
    # Опционально: manifest_train_list: [rico, synthetic] вместо merged
    if paths_cfg.get("manifest_train_list"):
        train_manifest = paths_cfg["manifest_train_list"]
    if paths_cfg.get("manifest_val_list"):
        val_manifest = paths_cfg["manifest_val_list"]

    use_rico_ds = bool(config.get("training", {}).get("use_rico_dataset", False))
    ds_cls = RicoDataset if use_rico_ds else ComparatorDataset
    train_ds = ds_cls(train_manifest, root=root, image_size=image_size)
    val_ds = ds_cls(val_manifest, root=root, image_size=image_size)

    num_workers = 0 if sys.platform == "win32" else 4
    batch_size = int(config["training"]["batch_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model_cfg = _config_from_yaml(config)
    model = MultiAspectComparator(model_cfg).to(device)

    resume = str(config["training"].get("resume_from", "") or "").strip()
    start_epoch = 0
    if resume:
        rpath = root / resume if not Path(resume).is_absolute() else Path(resume)
        if rpath.is_file():
            model, meta = load_comparator(str(rpath), device, strict=False)
            start_epoch = int(meta.get("epoch", 0) or 0)
            print(f"Resume: {rpath.name}, epoch={start_epoch}")

    aspect_w = _aspect_loss_weights(config)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs - start_epoch))
    criterion = nn.BCELoss()

    weights_dir = root / config["paths"]["weights_dir"]
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_path = root / config["paths"]["best_model"]
    last_path = root / config["paths"]["last_model"]
    patience = int(config["training"]["patience"])

    best_f1 = 0.0
    patience_counter = 0

    print("Начало обучения...\n")

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{start_epoch + epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds_dict = model(x)
            preds = torch.cat([preds_dict[k] for k in ASPECT_KEYS], dim=1)

            loss = torch.tensor(0.0, device=device)
            for i, key in enumerate(ASPECT_KEYS):
                loss = loss + criterion(preds[:, i : i + 1], y[:, i : i + 1]) * aspect_w[key]

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                preds_dict = model(x)
                preds = torch.cat([preds_dict[k] for k in ASPECT_KEYS], dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.numpy())

        val_preds = np.vstack(all_preds)
        val_targets = np.vstack(all_targets)
        metrics = compute_metrics(val_preds, val_targets)

        avg_f1 = metrics["mean_f1"]
        text_f1 = metrics.get("text_match/f1", 0)
        print(
            f"Epoch {epoch:2d} | Loss: {train_loss / len(train_loader):.4f} | "
            f"F1: {avg_f1:.4f} | text F1: {text_f1:.4f} | MAE: {metrics['mean_mae']:.4f}"
        )

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            save_comparator(
                str(best_path),
                model,
                optimizer=optimizer,
                epoch=epoch,
                extra={"mean_f1": best_f1, "metrics": metrics},
            )
            patience_counter = 0
            print(f"   -> best F1={best_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping!")
            break

    save_comparator(str(last_path), model, optimizer=optimizer, epoch=epoch, extra={"mean_f1": best_f1})
    print(f"\nГотово. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    train()
