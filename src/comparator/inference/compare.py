"""Инференс MultiAspectComparator на паре кропов Figma / Site."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import yaml

from src.comparator.models.multi_aspect import ASPECT_KEYS, load_comparator
from src.comparator.training.dataset import four_channel_from_paths


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def calibrate_scores(
    scores: Dict[str, float],
    align_shift: Tuple[int, int] = (0, 0),
    *,
    max_compensated_shift: int = 15,
) -> Dict[str, float]:
    """
    Пост-калибровка после align+blur.

  - Компенсированный сдвиг ≤15px: поднимаем layout (меньше ложных layout-багов).
  - Контентный overall: text и image важнее layout/typography для вердикта.
    """
    out = dict(scores)
    dx, dy = int(align_shift[0]), int(align_shift[1])
    shift_px = abs(dx) + abs(dy)

    if 0 < shift_px <= max_compensated_shift:
        boost = min(0.14, 0.025 * shift_px)
        out["layout_match"] = min(1.0, out["layout_match"] + boost)
        out["overall_similarity"] = min(1.0, out["overall_similarity"] + boost * 0.25)

    # Контентно-взвешенный overall (для порога PASS/FAIL)
    content = (
        0.42 * out["text_match"]
        + 0.28 * out["image_match"]
        + 0.18 * out["layout_match"]
        + 0.12 * out["typography_match"]
    )
    out["overall_similarity"] = min(1.0, 0.55 * out["overall_similarity"] + 0.45 * content)
    out["content_similarity"] = round(content, 4)
    return out


class ComparatorInference:
    def __init__(
        self,
        weights_path: str | Path = "weights/multi_aspect_comparator_best.pt",
        *,
        config_path: str | Path = "config/comparator.yaml",
        device: Optional[torch.device] = None,
        root: Optional[Path] = None,
        align: bool = True,
        align_max_shift: int = 15,
        blur_radius: float = 0.8,
    ) -> None:
        self.root = root or _project_root()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg_file = Path(config_path)
        if not cfg_file.is_absolute():
            cfg_file = self.root / cfg_file
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        wpath = Path(weights_path)
        if not wpath.is_absolute():
            wpath = self.root / wpath

        self.model, self._meta = load_comparator(str(wpath), self.device)
        self.image_size = int(self.config.get("model", {}).get("input_size", 224))
        inf_cfg = self.config.get("inference", {})
        self.align = bool(inf_cfg.get("align", align))
        self.align_max_shift = int(inf_cfg.get("align_max_shift", align_max_shift))
        self.blur_radius = float(inf_cfg.get("blur_radius", blur_radius))
        self.pass_threshold = float(inf_cfg.get("pass_threshold", 0.68))
        self.align_method = str(inf_cfg.get("align_method", "auto"))
        self._last_align_shift: Tuple[int, int] = (0, 0)
        print(f"MultiAspectComparator загружена: {wpath.name} ({self.device})")

    @torch.no_grad()
    def predict(self, figma_path: str | Path, site_path: str | Path) -> Dict[str, float]:
        tensor, shift = four_channel_from_paths(
            figma_path,
            site_path,
            root=self.root,
            image_size=self.image_size,
            align=self.align,
            align_max_shift=self.align_max_shift,
            blur_radius=self.blur_radius,
            align_method=self.align_method,
            return_shift=True,
        )
        self._last_align_shift = shift
        tensor = tensor.unsqueeze(0).to(self.device)
        outputs = self.model(tensor)
        raw = {key: float(outputs[key].view(-1)[0].item()) for key in ASPECT_KEYS}
        return calibrate_scores(raw, shift, max_compensated_shift=self.align_max_shift)

    def predict_pair(
        self,
        figma_path: str | Path,
        site_path: str | Path,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        thr = float(threshold if threshold is not None else self.pass_threshold)
        scores = self.predict(figma_path, site_path)
        overall = scores["overall_similarity"]
        return {
            "scores": scores,
            "verdict": "PASS" if overall >= thr else "FAIL",
            "confidence": overall,
            "recommended_threshold": thr,
            "align_shift": list(self._last_align_shift),
        }


def compare_fragments(
    figma_path: Union[str, Path],
    site_path: Union[str, Path],
    *,
    weights_path: str | Path = "weights/multi_aspect_comparator_best.pt",
    threshold: float = 0.68,
    **kwargs: Any,
) -> Dict[str, Any]:
    infer = ComparatorInference(weights_path=weights_path, **kwargs)
    return infer.predict_pair(figma_path, site_path, threshold=threshold)
