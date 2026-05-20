"""Метрики обучения MultiAspectComparator."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from src.comparator.models.multi_aspect import ASPECT_KEYS


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """MAE, accuracy и F1 по каждому аспекту (бинаризация порог 0.5)."""
    metrics: Dict[str, float] = {}

    for i, aspect in enumerate(ASPECT_KEYS):
        p = preds[:, i]
        t = targets[:, i]

        metrics[f"{aspect}/mae"] = float(mean_absolute_error(t, p))

        p_bin = (p > 0.5).astype(int)
        t_bin = (t > 0.5).astype(int)

        metrics[f"{aspect}/acc"] = float(accuracy_score(t_bin, p_bin))
        metrics[f"{aspect}/f1"] = float(f1_score(t_bin, p_bin, zero_division=0))

    metrics["mean_mae"] = float(np.mean([metrics[f"{k}/mae"] for k in ASPECT_KEYS]))
    metrics["mean_f1"] = float(np.mean([metrics[f"{k}/f1"] for k in ASPECT_KEYS]))

    return metrics
