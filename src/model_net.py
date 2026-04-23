from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn as nn


class TinyDiffCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_classifier(path: str | None, device: torch.device) -> Tuple[TinyDiffCNN | None, bool]:
    if not path or not os.path.isfile(path):
        return None, False
    m = TinyDiffCNN()
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    m.load_state_dict(state)
    m.to(device)
    m.eval()
    return m, True


@torch.no_grad()
def predict_fail_prob(model: TinyDiffCNN, x: torch.Tensor, device: torch.device) -> float:
    logits = model(x.to(device))
    prob = torch.softmax(logits, dim=1)[0, 1].item()
    return float(prob)
