"""
Модель: три сегмента [A | помеха | B], общий энкодер, Transformer с подавлением средней панели.

DOMAIN: image — RGB/gray панели 64×64.
Для text/code/table замените SegmentEncoder на свой эмбеддер (BERT, CodeBERT, …)
и подайте input_ids вместо pixel_values в forward().
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fragment_match.config import FragmentMatchConfig


class SegmentEncoderCNN(nn.Module):
    """Лёгкий CNN для кропов UI / diff (DOMAIN: image)."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.proj(h)


class SegmentEncoderViTHF(nn.Module):
    """Энкодер на HuggingFace ViT (DOMAIN: image). Требует transformers."""

    def __init__(self, model_name: str, out_dim: int) -> None:
        super().__init__()
        from transformers import ViTModel

        self.vit = ViTModel.from_pretrained(model_name)
        hid = self.vit.config.hidden_size
        self.proj = nn.Linear(hid, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B×3×H×W, ожидается 224 для base ViT — ресайз снаружи в dataset
        out = self.vit(pixel_values=x).pooler_output
        return self.proj(out)


class InterferenceAwareMatcher(nn.Module):
    """
    Энкодирует левый A, помеху и правый B; сравнивает A_left с A_right.
    Transformer между тремя токенами + маска, ослабляющая влияние среднего токена.
    """

    def __init__(self, cfg: FragmentMatchConfig) -> None:
        super().__init__()
        m = cfg.model
        self.embed_dim = m.embed_dim
        if m.encoder == "vit_hf":
            self.encoder = SegmentEncoderViTHF(m.hf_model_name, m.embed_dim)
            self._needs_224 = True
        else:
            self.encoder = SegmentEncoderCNN(m.embed_dim)
            self._needs_224 = False

        self.seg_type_emb = nn.Embedding(3, m.embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=m.embed_dim,
            nhead=m.transformer_heads,
            dim_feedforward=m.embed_dim * 4,
            dropout=m.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=m.transformer_layers,
            enable_nested_tensor=False,
        )
        # Ослабление эмбеддинга средней панели (помеха), без float-mask в attention
        # (в eval() additive mask в PyTorch Transformer даёт NaN).
        self.middle_scale = max(0.0, 1.0 - float(m.middle_attn_penalty))
        self.head = nn.Sequential(
            nn.Linear(m.embed_dim * 2, m.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(m.dropout),
            nn.Linear(m.embed_dim, 1),
        )

    def _encode_panels(self, panels: torch.Tensor) -> torch.Tensor:
        """panels: B×3×C×H×W → B×3×D"""
        b, n, c, h, w = panels.shape
        flat = panels.view(b * n, c, h, w)
        emb = self.encoder(flat).view(b, n, -1)
        return emb

    def _attenuate_middle_token(self, tok: torch.Tensor) -> torch.Tensor:
        """Снижает вклад помехи до self-attention (DOMAIN: средний сегмент)."""
        if self.middle_scale >= 1.0:
            return tok
        out = tok.clone()
        out[:, 1, :] = out[:, 1, :] * self.middle_scale
        return out

    def forward(
        self,
        panels: torch.Tensor,
        *,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        panels: B×3×3×H×W — [левый A, помеха, правый A/B]
        Returns: logits (B,), left_emb (B,D), right_emb (B,D), fused (B,D) optional
        """
        tok = self._encode_panels(panels)
        type_ids = torch.arange(3, device=panels.device).unsqueeze(0).expand(panels.size(0), -1)
        tok = tok + self.seg_type_emb(type_ids)
        tok = self._attenuate_middle_token(tok)
        fused_seq = self.transformer(tok)

        left = F.normalize(fused_seq[:, 0, :], dim=-1)
        right = F.normalize(fused_seq[:, 2, :], dim=-1)
        cat = torch.cat([left, right], dim=-1)
        logits = self.head(cat).squeeze(-1)
        if return_embeddings:
            return logits, left, right, fused_seq[:, 0, :]
        return logits, left, right, None


def build_matcher(cfg: FragmentMatchConfig) -> InterferenceAwareMatcher:
    return InterferenceAwareMatcher(cfg)
