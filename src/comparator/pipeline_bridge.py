"""
Мост к основному QA-пайплайну (src/pipeline.py, src/report.py).

Пример:
    from src.comparator.pipeline_bridge import comparator_bugs_for_crops
    bugs = comparator_bugs_for_crops("path/figma.png", "path/site.png")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def comparator_bugs_for_crops(
    figma_crop: str | Path,
    site_crop: str | Path,
    *,
    threshold: float = 0.68,
    weights_path: str = "weights/multi_aspect_comparator_best.pt",
    root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """NN + правила для одной пары кропов (для вставки в основной отчёт)."""
    from src.comparator.inference.compare import ComparatorInference
    from src.comparator.inference.merge_report import merge_nn_with_rules

    project_root = root or Path(__file__).resolve().parents[2]
    infer = ComparatorInference(weights_path=weights_path, root=project_root)
    nn_result = infer.predict_pair(figma_crop, site_crop, threshold=threshold)
    return merge_nn_with_rules(nn_result, dom_data=None, ocr_result=None)
