"""PyTorch Dataset для MultiAspectComparator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.comparator.models.multi_aspect import ASPECT_KEYS, build_four_channel_input


def four_channel_from_paths(
    figma_path: str | Path,
    site_path: str | Path,
    *,
    root: Optional[Path] = None,
    image_size: int = 224,
    align_max_shift: int = 15,
    align: bool = True,
    blur_radius: float = 0.8,
    align_method: str = "auto",
    return_shift: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[int, int]]:
    """4-канальный тензор C×H×W для inference (без manifest)."""
    root = Path(root) if root else Path.cwd()
    loader = ComparatorDataset.__new__(ComparatorDataset)
    loader.root = root
    loader.image_size = image_size
    tensor, shift = loader.build_four_channel_input(
        str(figma_path),
        str(site_path),
        align=align,
        align_max_shift=align_max_shift,
        blur_radius=blur_radius,
        align_method=align_method,
        return_shift=True,
    )
    if return_shift:
        return tensor, shift
    return tensor


def _load_manifest_records(manifest_path: Path) -> List[Dict]:
    records: List[Dict] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class ComparatorDataset(Dataset):
    """Датасет пар (figma, site) с метками по 6 аспектам."""

    def __init__(
        self,
        manifest_path: str | Path | List[str | Path],
        *,
        root: Optional[Path] = None,
        image_size: int = 224,
    ) -> None:
        self.root = Path(root) if root else Path.cwd()
        self.image_size = image_size
        self.records: List[Dict] = []

        paths = manifest_path if isinstance(manifest_path, list) else [manifest_path]
        for mp in paths:
            manifest = Path(mp)
            if not manifest.is_absolute():
                manifest = self.root / manifest
            if manifest.is_file():
                self.records.extend(_load_manifest_records(manifest))
            else:
                print(f"  предупреждение: нет файла {manifest}")

        print(f"Загружено {len(self.records)} пар из {len(paths)} manifest(s)")

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root / p

    def _load_rgb_tensor(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # 3, H, W

    def build_four_channel_input(
        self,
        figma_path: str,
        site_path: str,
        *,
        align: bool = True,
        align_max_shift: int = 15,
        blur_radius: float = 0.8,
        align_method: str = "auto",
        return_shift: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[int, int]]:
        """4 канала через общую функцию модели (BT.601 + diff + mean)."""
        from src.comparator.inference.preprocess import prepare_crop_pair

        figma_pil = Image.open(self._resolve(figma_path)).convert("RGB")
        site_pil = Image.open(self._resolve(site_path)).convert("RGB")
        if figma_pil.size != (self.image_size, self.image_size):
            figma_pil = figma_pil.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        if site_pil.size != (self.image_size, self.image_size):
            site_pil = site_pil.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        figma_pil, site_pil, shift = prepare_crop_pair(
            figma_pil,
            site_pil,
            max_shift=align_max_shift,
            blur_radius=blur_radius if align else 0.0,
            align=align,
            align_method=align_method,
        )

        figma = torch.from_numpy(np.array(figma_pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
        site = torch.from_numpy(np.array(site_pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
        x4 = build_four_channel_input(figma.unsqueeze(0), site.unsqueeze(0))
        out = x4.squeeze(0)
        if return_shift:
            return out, shift
        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        x = self.build_four_channel_input(record["figma"], record["site"], align=False)
        labels = [float(record["labels"][key]) for key in ASPECT_KEYS]
        y = torch.tensor(labels, dtype=torch.float32)
        return x, y
