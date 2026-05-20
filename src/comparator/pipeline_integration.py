"""
Интеграция MultiAspectComparator в основной QA-пайплайн (сайт vs Figma).

Для каждого «горячего» региона diff строятся кропы макета и страницы,
нейросеть оценивает 6 аспектов, результат попадает в bug_items для HTML-отчёта.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.comparator.inference.merge_report import merge_nn_with_rules
from src.comparator.inference.text_check import compare_crop_texts

_ASPECT_TO_TITLE = {
    "visual_mismatch": "Визуальное несоответствие",
    "text_mismatch": "Несоответствие текста",
    "typography": "Типографика",
    "color_mismatch": "Цвет",
    "layout_mismatch": "Вёрстка",
    "image_mismatch": "Изображение",
}


def _collect_regions(
    hotspots: Dict[str, Any],
    layout_elements: Optional[List[Any]],
    *,
    max_regions: int,
) -> List[Dict[str, Any]]:
    """Регионы с bbox для прогона comparator (приоритет — DOM-элементы с diff)."""
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []

    def _add(reg: Dict[str, Any]) -> None:
        try:
            key = f"{int(reg['x'])}:{int(reg['y'])}:{int(reg['w'])}:{int(reg['h'])}"
        except (KeyError, TypeError, ValueError):
            return
        if key in seen:
            return
        seen.add(key)
        out.append(reg)

    if isinstance(hotspots, dict):
        for el in hotspots.get("elements_by_overlap") or []:
            if isinstance(el, dict) and el.get("w") and el.get("h"):
                _add(el)
        for z in hotspots.get("grid_zones") or []:
            if isinstance(z, dict) and z.get("w") and z.get("h"):
                _add({**z, "snippet": z.get("snippet", f"зона {z.get('col')}×{z.get('row')}")})

    if len(out) < max_regions and isinstance(layout_elements, list):
        for el in layout_elements[: max_regions * 2]:
            if isinstance(el, dict) and el.get("w") and el.get("h"):
                _add(el)

    return out[:max_regions]


def _save_crop_224(
    image_path: str,
    bbox: Tuple[int, int, int, int],
    out_path: str,
    *,
    pad: int = 8,
) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False
    if not os.path.isfile(image_path):
        return False
    x, y, w, h = bbox
    try:
        im = Image.open(image_path).convert("RGB")
    except OSError:
        return False
    iw, ih = im.size
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(iw, x + w + pad)
    y1 = min(ih, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return False
    crop = im.crop((x0, y0, x1, y1))
    crop = crop.resize((224, 224), Image.Resampling.LANCZOS)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    crop.save(out_path, format="PNG", optimize=True)
    return True


def _bugs_to_items(
    bugs: List[Dict[str, Any]],
    region: Dict[str, Any],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    snippet = str(region.get("snippet", "")).strip()
    try:
        x, y, w, h = int(region["x"]), int(region["y"]), int(region["w"]), int(region["h"])
    except (KeyError, TypeError, ValueError):
        x = y = w = h = 0

    for bug in bugs:
        title = bug.get("title") or _ASPECT_TO_TITLE.get(bug.get("type", ""), "Отличие")
        desc = str(bug.get("description", "")).strip()
        score = bug.get("nn_score", bug.get("nn_overall"))
        text = f"[NN] {title}"
        if desc:
            text += f" — {desc}"
        if score is not None:
            text += f" ({float(score):.2f})"
        if snippet:
            text += f": {snippet}"

        row: Dict[str, Any] = {
            "text": text,
            "source": "comparator",
            "comparator_type": bug.get("type"),
        }
        if w > 0 and h > 0:
            row.update({"x": x, "y": y, "w": w, "h": h})
        if snippet:
            row["snippet"] = snippet
        items.append(row)
    return items


def augment_bug_items_with_comparator(
    bug_items: List[Dict[str, Any]],
    *,
    baseline_path: str,
    current_path: str,
    hotspots: Dict[str, Any],
    layout_elements: Optional[List[Any]],
    project_root: str,
    weights_path: str = "weights/multi_aspect_comparator_best.pt",
    pass_threshold: float = 0.68,
    max_regions: int = 10,
    crops_dir: Optional[str] = None,
    stats_sink: Optional[Dict[str, Any]] = None,
    log: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Дополняет bug_items предсказаниями MultiAspectComparator по регионам diff.
    """
    root = Path(project_root)
    wpath = Path(weights_path)
    if not wpath.is_absolute():
        wpath = root / wpath
    if not wpath.is_file():
        if log:
            log(f"Comparator: веса не найдены ({wpath}), пропуск")
        return bug_items

    regions = _collect_regions(hotspots, layout_elements, max_regions=max_regions)
    if not regions:
        if log:
            log("Comparator: нет регионов для анализа")
        return bug_items

    from src.comparator.inference.compare import ComparatorInference

    if log:
        log(f"Comparator: анализ до {len(regions)} регионов…")

    infer = ComparatorInference(weights_path=wpath, root=root)
    tmp_base = crops_dir or tempfile.mkdtemp(prefix="comparator_crops_")
    os.makedirs(tmp_base, exist_ok=True)

    nn_items: List[Dict[str, Any]] = []
    region_scores: List[Dict[str, Any]] = []

    for i, reg in enumerate(regions):
        try:
            bbox = (int(reg["x"]), int(reg["y"]), int(reg["w"]), int(reg["h"]))
        except (KeyError, TypeError, ValueError):
            continue
        figma_crop = os.path.join(tmp_base, f"r{i:02d}_figma.png")
        site_crop = os.path.join(tmp_base, f"r{i:02d}_site.png")
        if not _save_crop_224(baseline_path, bbox, figma_crop):
            continue
        if not _save_crop_224(current_path, bbox, site_crop):
            continue

        scores = infer.predict(figma_crop, site_crop)
        nn_result = {
            "scores": scores,
            "verdict": "PASS" if scores["overall_similarity"] >= pass_threshold else "FAIL",
        }
        snippet = str(reg.get("snippet", ""))
        inner = str(reg.get("innerText", ""))
        need_ocr = "stats" in snippet.lower() or any(
            c.isdigit() for c in inner
        ) or "M+" in inner or "m+" in inner.lower()
        ocr_result = None
        if need_ocr:
            ocr_result = compare_crop_texts(
                figma_crop, site_crop, site_dom_text=inner
            )
        bugs = merge_nn_with_rules(
            nn_result,
            dom_data=None,
            ocr_result=ocr_result,
            region=reg,
            figma_path=figma_crop,
            site_path=site_crop,
        )
        region_scores.append({"region": reg, "scores": scores, "bugs": len(bugs)})
        if bugs:
            worst = min(bugs, key=lambda b: float(b.get("nn_score", 1.0) or 1.0))
            nn_items.extend(_bugs_to_items([worst], reg))

    if stats_sink is not None:
        stats_sink["comparator_used"] = True
        stats_sink["comparator_regions"] = region_scores
        if region_scores:
            avg_overall = sum(r["scores"]["overall_similarity"] for r in region_scores) / len(
                region_scores
            )
            stats_sink["comparator_mean_overall"] = round(avg_overall, 4)

    if log:
        log(f"Comparator: добавлено {len(nn_items)} пунктов из NN")

    if not nn_items:
        return bug_items

    existing_texts = {str(it.get("text", "")).strip().lower() for it in bug_items}
    merged = list(bug_items)
    for it in nn_items:
        t = str(it.get("text", "")).strip().lower()
        if t and t not in existing_texts:
            merged.append(it)
            existing_texts.add(t)
    return merged
