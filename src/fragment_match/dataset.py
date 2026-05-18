"""
DataLoader для троек [A | помеха | B].

DOMAIN: image — читает PNG с тремя панелями в ряд.
Для text: замените __getitem__ на загрузку трёх текстовых полей из JSONL.
Для code/table: токенизация трёх ячеек/фрагментов кода.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.fragment_match.config import DataConfig


def load_manifest(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class FragmentMatchDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        manifest_name: str,
        *,
        segment_size: int = 64,
        panel_width: int = 64,
        modality: str = "image",
        vit_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.data_root = data_root
        self.modality = modality
        self.segment_size = segment_size
        self.panel_width = panel_width
        self.vit_size = vit_size
        self.augment = augment
        manifest_path = os.path.join(data_root, manifest_name)
        self.rows = load_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image_panels(self, row: Dict[str, Any]) -> torch.Tensor:
        rel = row.get("image") or row.get("path")
        path = rel if os.path.isabs(str(rel)) else os.path.join(self.data_root, str(rel))
        im = Image.open(path).convert("RGB")
        w, h = im.size
        pw = w // 3
        panels = []
        for i in range(3):
            crop = im.crop((i * pw, 0, (i + 1) * pw, h))
            if self.vit_size and self.vit_size != self.segment_size:
                crop = crop.resize((self.vit_size, self.vit_size), Image.Resampling.BILINEAR)
            else:
                crop = crop.resize((self.segment_size, self.segment_size), Image.Resampling.BILINEAR)
            arr = np.asarray(crop, dtype=np.float32) / 255.0
            panels.append(torch.from_numpy(arr).permute(2, 0, 1))
        stacked = torch.stack(panels, dim=0)
        if self.augment:
            stacked = self._augment_panels(stacked)
        return stacked

    def _augment_panels(self, panels: torch.Tensor) -> torch.Tensor:
        import random as _rnd

        if _rnd.random() < 0.5:
            panels = torch.flip(panels, dims=[-1])
        if _rnd.random() < 0.35:
            delta = (_rnd.random() - 0.5) * 0.12
            panels = (panels + delta).clamp(0.0, 1.0)
        return panels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        if self.modality != "image":
            raise NotImplementedError(
                "Для text/code/table реализуйте загрузку в dataset.py (см. DOMAIN)."
            )
        panels = self._load_image_panels(row)
        is_same = int(row.get("is_same", 0))
        return {
            "panels": panels,
            "is_same": torch.tensor(is_same, dtype=torch.float32),
            "noise_type": str(row.get("noise_type", "")),
            "noise_ratio": float(row.get("noise_ratio", 0.33)),
            "left_id": str(row.get("left_id", "")),
            "right_id": str(row.get("right_id", "")),
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "panels": torch.stack([b["panels"] for b in batch], dim=0),
        "is_same": torch.stack([b["is_same"] for b in batch], dim=0),
        "noise_type": [b["noise_type"] for b in batch],
        "noise_ratio": torch.tensor([b["noise_ratio"] for b in batch], dtype=torch.float32),
    }


def make_loaders(
    cfg: DataConfig,
    batch_size: int,
    num_workers: int = 0,
    vit_size: int = 224,
    *,
    train_augment: bool = True,
):
    from torch.utils.data import DataLoader

    kw = dict(
        segment_size=cfg.segment_size,
        panel_width=cfg.panel_width,
        modality=cfg.modality,
        vit_size=vit_size if cfg.modality == "image" else 0,
    )
    tr = FragmentMatchDataset(cfg.root, cfg.manifest_train, **kw, augment=train_augment)
    va = FragmentMatchDataset(cfg.root, cfg.manifest_val, **kw, augment=False)
    te = FragmentMatchDataset(cfg.root, cfg.manifest_test, **kw, augment=False)
    train_loader = DataLoader(
        tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        va, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch
    )
    test_loader = DataLoader(
        te, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch
    )
    return train_loader, val_loader, test_loader
