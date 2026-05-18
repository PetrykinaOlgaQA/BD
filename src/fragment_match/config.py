"""
Конфигурация обучения сопоставления фрагментов.

Домен по умолчанию — изображения (кропы UI / diff). Для текста/кода/таблиц
замените encoder в models.py и загрузчик в dataset.py (см. комментарии DOMAIN).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    root: str = "data/fragment_match"
    manifest_train: str = "manifest_train.jsonl"
    manifest_val: str = "manifest_val.jsonl"
    manifest_test: str = "manifest_test.jsonl"
    segment_size: int = 64
    panel_width: int = 64
    # Для изображений: 3 панели в ряд; для текста — max_tokens на сегмент
    modality: str = "image"


@dataclass
class ModelConfig:
    # cnn | vit_hf — vit_hf требует transformers и больше VRAM
    encoder: str = "cnn"
    hf_model_name: str = "google/vit-base-patch16-224"
    embed_dim: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.15
    # Штраф внимания к средней панели (помеха)
    middle_attn_penalty: float = 0.72


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    triplet_margin: float = 0.35
    contrastive_weight: float = 0.5
    bce_weight: float = 1.0
    label_smoothing: float = 0.05
    early_stop_patience: int = 6
    lr_min_ratio: float = 0.05
    num_workers: int = 0
    seed: int = 42
    out_path: str = "weights/fragment_matcher.pt"


@dataclass
class FragmentMatchConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: Optional[str] = None
