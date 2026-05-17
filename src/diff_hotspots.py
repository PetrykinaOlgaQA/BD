from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.compare import build_change_mask


def _grid_zones(
    mask: np.ndarray,
    *,
    cols: int = 14,
    rows: int = 10,
    top_k: int = 8,
    min_frac: float = 0.012,
) -> List[Dict[str, Any]]:
    """Топ ячеек сетки по доле изменённых пикселей (без OpenCV)."""
    h, w = mask.shape
    cols = max(4, min(32, int(cols)))
    rows = max(4, min(24, int(rows)))
    cell_h = max(1, h // rows)
    cell_w = max(1, w // cols)
    scored: List[tuple[float, int, int, int, int, int, int]] = []
    for ri in range(rows):
        for ci in range(cols):
            y0, x0 = ri * cell_h, ci * cell_w
            y1, x1 = min(h, y0 + cell_h), min(w, x0 + cell_w)
            sub = mask[y0:y1, x0:x1]
            if sub.size == 0:
                continue
            frac = float(sub.mean())
            scored.append((frac, x0, y0, x1 - x0, y1 - y0, ci + 1, ri + 1))
    scored.sort(key=lambda t: -t[0])
    out: List[Dict[str, Any]] = []
    for frac, x0, y0, cw, ch, c1, r1 in scored:
        if frac < min_frac:
            continue
        out.append(
            {
                "col": c1,
                "row": r1,
                "x": x0,
                "y": y0,
                "w": cw,
                "h": ch,
                "changed_frac_pct": round(frac * 100, 2),
            }
        )
        if len(out) >= top_k:
            break
    return out


def _elements_by_overlap(
    mask: np.ndarray,
    elements: Optional[List[Any]],
    *,
    top_k: int = 12,
    min_frac: float = 0.02,
) -> List[Dict[str, Any]]:
    if not isinstance(elements, list):
        return []
    h, w = mask.shape
    scored: List[tuple[float, str, int, int, int, int]] = []
    for el in elements:
        if not isinstance(el, dict):
            continue
        try:
            x, y, ew, eh = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
        except (KeyError, TypeError, ValueError):
            continue
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + ew), min(h, y + eh)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = mask[y0:y1, x0:x1]
        frac = float(sub.mean())
        if frac < min_frac:
            continue
        sn = str(el.get("snippet", "?")).strip() or "?"
        scored.append((frac, sn, x0, y0, x1 - x0, y1 - y0))
    scored.sort(key=lambda t: -t[0])
    out: List[Dict[str, Any]] = []
    for frac, sn, x0, y0, ww, hh in scored[:top_k]:
        out.append(
            {
                "snippet": sn,
                "x": x0,
                "y": y0,
                "w": ww,
                "h": hh,
                "changed_frac_in_box_pct": round(frac * 100, 2),
            }
        )
    return out


def analyze_diff_for_qa(
    baseline_path: str,
    current_path: str,
    layout_elements: Optional[List[Any]],
    *,
    pixel_threshold: int,
    tolerance_shift_px: int,
    tolerance_speckle_iter: int,
) -> Dict[str, Any]:
    """
    Детерминированный разбор маски изменений: сетка + пересечение с блоками layout.
    Не заменяет VLM, но даёт конкретные «якоря» по координатам и селекторам.
    """
    mask, (mw, mh) = build_change_mask(
        baseline_path,
        current_path,
        pixel_threshold=pixel_threshold,
        tolerance_shift_px=tolerance_shift_px,
        tolerance_speckle_iter=tolerance_speckle_iter,
    )
    total_changed = float(mask.mean()) * 100.0
    grid = _grid_zones(mask, cols=14, rows=10, top_k=8)
    el_ov = _elements_by_overlap(mask, layout_elements, top_k=12)
    return {
        "mask_size": [int(mw), int(mh)],
        "changed_pixels_pct": round(total_changed, 3),
        "grid_cells": grid,
        "elements_overlap": el_ov,
    }


def diff_hotspots_to_task_lines(hot: Dict[str, Any], *, max_lines: int = 14) -> List[str]:
    """Строки для fallback/HTML: только по блокам, без «полос» и координат сетки."""
    lines: List[str] = []
    for e in hot.get("elements_overlap") or []:
        if not isinstance(e, dict):
            continue
        sn = e.get("snippet", "?")
        lines.append(
            f"Блок {sn} на скрине явно не совпадает с Figma — сравни с макетом размер текста и отступы (в первую очередь сверху/снизу) именно у этого элемента."
        )
        if len(lines) >= max_lines:
            return lines
    if not lines:
        lines.append(
            "По данным layout нет пересечений блоков с сильным diff — открой JSON elements и diff-картинку и выпиши несовпадения по классам и видимому тексту."
        )
    return lines[:max_lines]
