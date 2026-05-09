"""TinyDiffCNN: классификация карты diff (pass vs fail)."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class TinyDiffCNN(nn.Module):
    """Компактный CNN по одному каналу 64×64 (после нормализации [0,1])."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(96, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model(path: str | os.PathLike[str] | None, device: torch.device | None = None) -> Tuple[TinyDiffCNN, bool]:
    """
    Загружает веса. Если файла нет — возвращает модель в eval с случайными весами (loaded=False).
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = TinyDiffCNN()
    p = str(path) if path else ""
    if not p or not os.path.isfile(p):
        m.to(dev)
        m.eval()
        return m, False
    try:
        state = torch.load(p, map_location=dev, weights_only=True)
    except TypeError:
        state = torch.load(p, map_location=dev)
    m.load_state_dict(state)
    m.to(dev)
    m.eval()
    return m, True


@torch.no_grad()
def predict_diff(model: TinyDiffCNN, diff_tensor: torch.Tensor, device: torch.device | None = None) -> dict:
    """
    diff_tensor: форма (1, 1, H, W), значения ~ [0, 1].
    Возвращает logits, softmax, prob_fail (класс 1 = «похоже на баг»).
    """
    dev = device or next(model.parameters()).device
    x = diff_tensor.to(dev, dtype=torch.float32)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    p_fail = float(prob[1].item())
    p_pass = float(prob[0].item())
    return {
        "logits": logits.cpu().numpy().tolist(),
        "prob_pass": p_pass,
        "prob_fail": p_fail,
        "verdict_cnn": "FAIL" if p_fail >= 0.5 else "PASS",
    }


def numpy_gray_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(H, W) float32 [0,1] -> (1,1,H,W)."""
    if arr.ndim != 2:
        raise ValueError("Ожидается 2D grayscale")
    t = torch.from_numpy(arr.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    return t
