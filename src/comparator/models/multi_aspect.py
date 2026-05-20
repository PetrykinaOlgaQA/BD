"""
MultiAspectComparator — одна модель для сравнения кропа Figma и кропа сайта.

Вход (4 канала, 224×224):
  1. яркость макета (Figma)
  2. яркость сайта (Site)
  3. карта отличий |Figma − Site|
  4. средняя яркость (контекст освещённости)

Backbone: MobileNetV3-Small (ImageNet), перед ним адаптер 4→3 канала.
Выход: 6 вероятностей сходства в [0, 1] — overall + 5 аспектов.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

# Порядок выходов фиксирован — используется в dataset, train и отчёте
ASPECT_KEYS: Tuple[str, ...] = (
    "overall_similarity",
    "text_match",
    "image_match",
    "layout_match",
    "typography_match",
    "color_match",
)

TensorLike = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def _to_nchw_rgb(x: torch.Tensor) -> torch.Tensor:
    """Приводит тензор к виду B×3×H×W, значения ожидаются в [0, 1]."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    if x.shape[1] != 3:
        raise ValueError(f"ожидается 1 или 3 канала, получено {x.shape}")
    return x.clamp(0.0, 1.0)


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    """BT.601 яркость, форма B×1×H×W."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def build_four_channel_input(
    figma_rgb: torch.Tensor,
    site_rgb: torch.Tensor,
) -> torch.Tensor:
    """
    Собирает 4-канальный тензор для модели.

    Args:
        figma_rgb: B×3×H×W или 3×H×W, значения [0, 1]
        site_rgb:  B×3×H×W или 3×H×W, значения [0, 1]

    Returns:
        B×4×H×W — [figma_luma, site_luma, diff, mean_luma]
    """
    figma = _to_nchw_rgb(figma_rgb)
    site = _to_nchw_rgb(site_rgb)
    if figma.shape[-2:] != site.shape[-2:]:
        site = torch.nn.functional.interpolate(
            site,
            size=figma.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    figma_l = _luminance(figma)
    site_l = _luminance(site)
    diff = (figma - site).abs().mean(dim=1, keepdim=True)
    mean_l = 0.5 * (figma_l + site_l)
    return torch.cat([figma_l, site_l, diff, mean_l], dim=1)


@dataclass
class ComparatorConfig:
    """Гиперпараметры архитектуры (сохраняются в checkpoint)."""

    image_size: int = 224
    in_channels: int = 4
    trunk_dim: int = 256
    dropout: float = 0.2
    pretrained_backbone: bool = True
    freeze_backbone: bool = True


class MultiAspectComparator(nn.Module):
    """
    Сравнивает пару изображений (макет Figma vs скрин сайта) по нескольким аспектам.

    Пример:
        model = MultiAspectComparator()
        x4 = build_four_channel_input(figma, site)
        preds = model(x4)  # dict с 6 тензорами формы B×1
    """

    def __init__(self, cfg: Optional[ComparatorConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ComparatorConfig()

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if self.cfg.pretrained_backbone else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Адаптер 4→3: не трогаем первый Conv backbone, проще и стабильнее на CPU
        self.stem_adapter = nn.Sequential(
            nn.Conv2d(self.cfg.in_channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.backbone = backbone.features
        self.pool = backbone.avgpool
        feat_dim = 576  # mobilenet_v3_small

        if self.cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, self.cfg.trunk_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.cfg.dropout),
        )
        self.heads = nn.ModuleDict(
            {key: nn.Linear(self.cfg.trunk_dim, 1) for key in ASPECT_KEYS}
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        figma_rgb: Optional[torch.Tensor] = None,
        site_rgb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: B×4×H×W — готовый 4-канальный вход, либо
            figma_rgb + site_rgb — тогда x собирается автоматически

        Returns:
            словарь {aspect_key: B×1 tensor в [0,1] после sigmoid}
        """
        if figma_rgb is not None and site_rgb is not None:
            x = build_four_channel_input(figma_rgb, site_rgb)
        if x.ndim != 4 or x.shape[1] != self.cfg.in_channels:
            raise ValueError(
                f"ожидается B×{self.cfg.in_channels}×H×W, получено {tuple(x.shape)}"
            )

        h = self.stem_adapter(x)
        h = self.backbone(h)
        h = self.pool(h)
        h = torch.flatten(h, 1)
        h = self.trunk(h)

        out: Dict[str, torch.Tensor] = {}
        for key in ASPECT_KEYS:
            out[key] = torch.sigmoid(self.heads[key](h))
        return out

    def predict_dict(self, x: torch.Tensor) -> Dict[str, float]:
        """Инференс для одного примера (batch=1) → float по каждому аспекту."""
        self.eval()
        with torch.no_grad():
            preds = self.forward(x)
        return {k: float(v.view(-1)[0].item()) for k, v in preds.items()}

    def unfreeze_backbone_last_block(self, n_blocks: int = 2) -> None:
        """Разморозить последние n блоков features (тонкий fine-tune)."""
        blocks = list(self.backbone.children())
        for layer in blocks[-n_blocks:]:
            for p in layer.parameters():
                p.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """Возвращает (обучаемых, всего) параметров — удобно для слайда диплома."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_comparator(
    path: str,
    model: MultiAspectComparator,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, object] = {
        "model_state": model.state_dict(),
        "config": model.cfg.__dict__,
        "aspect_keys": list(ASPECT_KEYS),
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_comparator(
    path: str,
    device: Optional[torch.device] = None,
    *,
    strict: bool = True,
) -> Tuple[MultiAspectComparator, Dict[str, object]]:
    """
    Загружает веса из checkpoint.

    Returns:
        (model, meta) — meta содержит epoch, metrics и т.д.
    """
    dev = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    cfg_dict = ckpt.get("config") if isinstance(ckpt, dict) else {}
    cfg = ComparatorConfig(**cfg_dict) if isinstance(cfg_dict, dict) else ComparatorConfig()
    model = MultiAspectComparator(cfg)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=strict)
    model.to(dev)
    model.eval()
    meta = ckpt if isinstance(ckpt, dict) else {}
    return model, meta


def _self_test() -> None:
    """Быстрая проверка форм (python -m src.comparator.models.multi_aspect)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiAspectComparator().to(device)
    tr, tot = count_trainable_parameters(model)
    print(f"MultiAspectComparator: trainable={tr:,} / total={tot:,}")

    figma = torch.rand(2, 3, 224, 224, device=device)
    site = torch.rand(2, 3, 224, 224, device=device)
    x4 = build_four_channel_input(figma, site)
    assert x4.shape == (2, 4, 224, 224), x4.shape

    preds = model(x4)
    for k in ASPECT_KEYS:
        assert k in preds, k
        assert preds[k].shape == (2, 1), (k, preds[k].shape)

    one = model.predict_dict(x4[:1])
    print("sample preds:", {k: round(v, 3) for k, v in one.items()})


if __name__ == "__main__":
    _self_test()
