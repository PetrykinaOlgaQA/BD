from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def bce_same_loss(
    logits: torch.Tensor,
    is_same: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    y = is_same.float()
    if label_smoothing > 0:
        y = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits, y)


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.35,
) -> torch.Tensor:
    """anchor≈positive (тот же A), anchor далеко от negative (другой B)."""
    d_pos = F.pairwise_distance(anchor, positive, p=2)
    d_neg = F.pairwise_distance(anchor, negative, p=2)
    return F.relu(d_pos - d_neg + margin).mean()


def contrastive_pair_loss(
    left: torch.Tensor,
    right: torch.Tensor,
    is_same: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    dist = F.pairwise_distance(left, right, p=2)
    same = is_same.float()
    loss_same = same * dist.pow(2)
    loss_diff = (1.0 - same) * F.relu(margin - dist).pow(2)
    return (loss_same + loss_diff).mean()


def combined_loss(
    logits: torch.Tensor,
    left_emb: torch.Tensor,
    right_emb: torch.Tensor,
    is_same: torch.Tensor,
    neg_right_emb: torch.Tensor,
    *,
    triplet_margin: float,
    contrastive_weight: float,
    bce_weight: float,
    label_smoothing: float = 0.0,
) -> dict[str, Any]:
    l_bce = bce_same_loss(logits, is_same, label_smoothing=label_smoothing)
    l_ctr = contrastive_pair_loss(left_emb, right_emb, is_same, margin=triplet_margin)
    pos = is_same.bool()
    if pos.any():
        l_tri = triplet_loss(
            left_emb[pos], right_emb[pos], neg_right_emb[pos], margin=triplet_margin
        )
    else:
        l_tri = logits.new_zeros(())
    total = bce_weight * l_bce + contrastive_weight * (l_ctr + l_tri)
    return {"total": total, "bce": l_bce, "contrastive": l_ctr, "triplet": l_tri}
