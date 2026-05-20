"""
PyTorch Dataset для пар Rico / merged manifest.

Совместим с MultiAspectComparator: 4-канальный вход, 6 лейблов.
При обучении align=False (сдвиги уже в сохранённых site.png).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.comparator.models.multi_aspect import ASPECT_KEYS, build_four_channel_input


def load_manifest(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class RicoDataset(Dataset):
    """
    Датасет из manifest.jsonl (Rico, synthetic или merged).

    Пример:
        ds = RicoDataset("data/comparator/manifest_train_rico.jsonl", root=project_root)
        x, y = ds[0]   # x: 4×224×224, y: 6 floats
    """

    def __init__(
        self,
        manifest: Union[str, Path, List[Union[str, Path]]],
        *,
        root: Optional[Path] = None,
        image_size: int = 224,
        align_at_load: bool = False,
        align_max_shift: int = 15,
        blur_radius: float = 0.0,
    ) -> None:
        self.root = Path(root) if root else Path.cwd()
        self.image_size = image_size
        self.align_at_load = align_at_load
        self.align_max_shift = align_max_shift
        self.blur_radius = blur_radius
        self.records: List[Dict] = []

        paths = manifest if isinstance(manifest, list) else [manifest]
        for p in paths:
            mp = Path(p)
            if not mp.is_absolute():
                mp = self.root / mp
            if mp.is_file():
                self.records.extend(load_manifest(mp))
            else:
                print(f"RicoDataset: нет файла {mp}")

        if not self.records:
            raise FileNotFoundError(f"Пустой датасет: {paths}")

        # Краткая сводка по aug_type
        from collections import Counter
        aug_cnt = Counter(r.get("aug_type", "?") for r in self.records)
        print(f"RicoDataset: {len(self.records)} пар | aug: {dict(aug_cnt.most_common(6))}")

    def __len__(self) -> int:
        return len(self.records)

    def _path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def _load_rgb(self, rel: str) -> torch.Tensor:
        img = Image.open(self._path(rel)).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def build_input(self, figma_rel: str, site_rel: str) -> torch.Tensor:
        if self.align_at_load:
            from src.comparator.inference.preprocess import prepare_crop_pair

            f = Image.open(self._path(figma_rel)).convert("RGB")
            s = Image.open(self._path(site_rel)).convert("RGB")
            if f.size != (self.image_size, self.image_size):
                f = f.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            if s.size != (self.image_size, self.image_size):
                s = s.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            f, s, _ = prepare_crop_pair(
                f, s,
                max_shift=self.align_max_shift,
                blur_radius=self.blur_radius,
                align=True,
            )
            ft = torch.from_numpy(np.array(f, dtype=np.float32) / 255.0).permute(2, 0, 1)
            st = torch.from_numpy(np.array(s, dtype=np.float32) / 255.0).permute(2, 0, 1)
        else:
            ft = self._load_rgb(figma_rel)
            st = self._load_rgb(site_rel)

        return build_four_channel_input(ft.unsqueeze(0), st.unsqueeze(0)).squeeze(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        x = self.build_input(rec["figma"], rec["site"])
        y = torch.tensor(
            [float(rec["labels"][k]) for k in ASPECT_KEYS],
            dtype=torch.float32,
        )
        return x, y

    @classmethod
    def from_split(
        cls,
        split: str = "train",
        *,
        root: Optional[Path] = None,
        use_merged: bool = True,
    ) -> "RicoDataset":
        """
        split: train | val | test
        use_merged: manifest_{split}.jsonl (merged) иначе manifest_{split}_rico.jsonl
        """
        root = root or Path.cwd()
        name = f"manifest_{split}.jsonl" if use_merged else f"manifest_{split}_rico.jsonl"
        return cls(root / "data/comparator" / name, root=root)
