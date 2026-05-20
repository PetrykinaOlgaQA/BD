"""
Гибридный отчёт: MultiAspectComparator + EasyOCR + эвристики изображения.

Принцип: лучше пропустить мелочь, чем 10 ложных.
Приоритет сигналов: OCR (цифры/текст) → image heuristics → NN scores.
"""
from __future__ import annotations

import html as html_module
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Пороги (согласованы с config/comparator.yaml) ---
TEXT_NN_STRONG = 0.62
TEXT_NN_WEAK = 0.72
OCR_SIM_STRONG = 0.78
IMAGE_NN_STRONG = 0.58
OVERALL_STRONG = 0.55
LAYOUT_STRONG = 0.48
# layout ниже 0.55 при хорошем тексте — часто шум после align
LAYOUT_BORDERLINE = (0.48, 0.58)

_SEV = {"high": 2, "medium": 1, "low": 0}


def _block_label(region: Optional[Dict[str, Any]]) -> str:
    if not region:
        return "блок"
    sn = str(region.get("snippet", "")).strip()
    if sn:
        return sn.split(".")[-1] or sn
    return "блок"


def _merge_same_type(bugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Один пункт на тип внутри блока (берём худший score)."""
    buckets: Dict[str, Dict[str, Any]] = {}
    for b in bugs:
        t = str(b.get("type", "other"))
        sc = float(b.get("nn_score", 1.0) or 1.0)
        prev = buckets.get(t)
        if prev is None or sc < float(prev.get("nn_score", 1.0) or 1.0):
            buckets[t] = dict(b)
    order = ("text_mismatch", "image_mismatch", "visual_mismatch", "layout_mismatch", "typography", "color_mismatch")
    out: List[Dict[str, Any]] = []
    for key in order:
        if key in buckets:
            out.append(buckets[key])
    for t, b in buckets.items():
        if t not in order:
            out.append(b)
    return out


def _text_bug(
    scores: Dict[str, float],
    ocr: Dict[str, Any],
    block: str,
) -> Optional[Dict[str, Any]]:
    ocr_sim = float(ocr.get("similarity", 1.0) or 1.0)
    fig_t = str(ocr.get("figma_text", "")).strip()
    site_t = str(ocr.get("site_text", "")).strip()

    if ocr.get("text_missing"):
        return {
            "type": "text_mismatch",
            "severity": "high",
            "title": "Текст отличается от макета",
            "description": f"[{block}] текст отсутствует на сайте (на макете «{_short(fig_t)}»)",
            "nn_score": round(min(scores["text_match"], ocr_sim), 3),
            "source": "ocr",
        }

    if ocr.get("important_numeric_change"):
        return {
            "type": "text_mismatch",
            "severity": "high",
            "title": "Текст отличается от макета",
            "description": f"[{block}] цифры: макет «{_short(fig_t)}» → сайт «{_short(site_t)}»",
            "nn_score": round(min(scores["text_match"], ocr_sim), 3),
            "source": "ocr",
        }

    if ocr.get("text_different"):
        return {
            "type": "text_mismatch",
            "severity": "high" if ocr_sim < 0.55 else "medium",
            "title": "Текст отличается от макета",
            "description": f"[{block}] макет «{_short(fig_t)}» → сайт «{_short(site_t)}»",
            "nn_score": round(min(scores["text_match"], ocr_sim), 3),
            "source": "ocr+nn",
        }

    if scores["text_match"] < TEXT_NN_STRONG:
        return {
            "type": "text_mismatch",
            "severity": "high",
            "title": "Текст отличается от макета",
            "description": f"[{block}] надпись не совпадает с макетом",
            "nn_score": round(scores["text_match"], 3),
            "source": "nn",
        }

    if scores["text_match"] < TEXT_NN_WEAK and ocr_sim < OCR_SIM_STRONG and not ocr.get("cosmetic_only"):
        return {
            "type": "text_mismatch",
            "severity": "medium",
            "title": "Текст отличается от макета",
            "description": f"[{block}] текст заметно отличается",
            "nn_score": round(min(scores["text_match"], ocr_sim), 3),
            "source": "hybrid",
        }
    return None


def _image_bug(
    scores: Dict[str, float],
    img: Dict[str, Any],
    ocr: Dict[str, Any],
    block: str,
) -> Optional[Dict[str, Any]]:
    if ocr.get("cosmetic_only"):
        return None

    reason = str(img.get("reason", ""))
    if img.get("mismatch"):
        if reason == "missing":
            return {
                "type": "image_mismatch",
                "severity": "high",
                "title": "Изображение отличается",
                "description": f"[{block}] иконка/картинка отсутствует на сайте",
                "nn_score": round(min(scores["image_match"], 0.35), 3),
                "source": "heuristic",
            }
        if reason == "size_change":
            hint = img.get("size_hint", "другой размер")
            return {
                "type": "image_mismatch",
                "severity": "medium",
                "title": "Изображение отличается",
                "description": f"[{block}] иконка на сайте {hint}, чем в макете",
                "nn_score": round(scores["image_match"], 3),
                "source": "heuristic",
            }
        return {
            "type": "image_mismatch",
            "severity": "medium",
            "title": "Изображение отличается",
            "description": f"[{block}] эмодзи/иконка не как в макете",
            "nn_score": round(scores["image_match"], 3),
            "source": "heuristic",
        }

    if scores["image_match"] < IMAGE_NN_STRONG:
        return {
            "type": "image_mismatch",
            "severity": "medium",
            "title": "Изображение отличается",
            "description": f"[{block}] графический элемент отличается",
            "nn_score": round(scores["image_match"], 3),
            "source": "nn",
        }
    return None


def _visual_bug(scores: Dict[str, float], block: str) -> Optional[Dict[str, Any]]:
    if scores["overall_similarity"] >= OVERALL_STRONG:
        return None
    min_a = min(scores["text_match"], scores["image_match"], scores["layout_match"])
    if min_a >= 0.68:
        return None
    return {
        "type": "visual_mismatch",
        "severity": "high" if scores["overall_similarity"] < 0.42 else "medium",
        "title": "Визуально отличается",
        "description": f"[{block}] блок заметно не как в макете",
        "nn_score": round(scores["overall_similarity"], 3),
        "source": "nn",
    }


def _layout_bug(scores: Dict[str, float], block: str) -> Optional[Dict[str, Any]]:
    lo, hi = LAYOUT_BORDERLINE
    if scores["layout_match"] >= hi:
        return None
    if lo <= scores["layout_match"] < hi and scores["text_match"] >= 0.85:
        return None
    if scores["layout_match"] >= LAYOUT_STRONG:
        return None
    if scores["text_match"] >= 0.88 and scores["image_match"] >= 0.75:
        return None
    return {
        "type": "layout_mismatch",
        "severity": "medium",
        "title": "Вёрстка",
        "description": f"[{block}] сдвиг или размер блока (после выравнивания)",
        "nn_score": round(scores["layout_match"], 3),
        "source": "nn",
    }


def _short(s: str, n: int = 36) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def get_short_bug_description(
    scores: Dict[str, float],
    ocr_data: Optional[Dict[str, Any]] = None,
    image_data: Optional[Dict[str, Any]] = None,
    *,
    region: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Собрать баги для одного кропа (до группировки по блокам)."""
    ocr = ocr_data or {}
    img = image_data or {}
    block = _block_label(region)
    bugs: List[Dict[str, Any]] = []

    tb = _text_bug(scores, ocr, block)
    if tb:
        bugs.append(tb)

    ib = _image_bug(scores, img, ocr, block)
    if ib:
        bugs.append(ib)

    vb = _visual_bug(scores, block)
    if vb and not any(b["type"] == "text_mismatch" for b in bugs):
        bugs.append(vb)

    lb = _layout_bug(scores, block)
    if lb:
        bugs.append(lb)

    return _merge_same_type(bugs)


def merge_bugs_by_block(bugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Несколько типов в одном блоке → одна строка с перечислением."""
    by_block: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for b in bugs:
        desc = str(b.get("description", ""))
        key = desc.split("]")[0] + "]" if desc.startswith("[") and "]" in desc else desc[:40]
        by_block[key].append(b)

    out: List[Dict[str, Any]] = []
    for _, group in by_block.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        parts = [str(g.get("description", "")).split("]", 1)[-1].strip() for g in group]
        parts = [p for p in parts if p]
        worst = min(group, key=lambda g: float(g.get("nn_score", 1.0) or 1.0))
        merged = dict(worst)
        merged["description"] = str(worst.get("description", "")).split("]")[0] + "] " + "; ".join(parts[:3])
        merged["count"] = len(group)
        out.append(merged)
    out.sort(key=lambda b: (_SEV.get(str(b.get("severity", "low")), 0), float(b.get("nn_score", 1.0) or 1.0)))
    return out


def merge_nn_with_rules(
    nn_result: Dict[str, Any],
    dom_data: Optional[Dict[str, Any]] = None,
    ocr_result: Optional[Dict[str, Any]] = None,
    *,
    region: Optional[Dict[str, Any]] = None,
    figma_path: Optional[str | Path] = None,
    site_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    from src.comparator.inference.rules import apply_comparator_rules

    scores = nn_result.get("scores", nn_result)
    image_data: Dict[str, Any] = {}
    if figma_path and site_path:
        try:
            from src.comparator.inference.image_check import compare_crop_images
            image_data = compare_crop_images(figma_path, site_path)
        except Exception:
            image_data = {}

    bugs = get_short_bug_description(scores, ocr_result, image_data, region=region)
    bugs = apply_comparator_rules(scores, bugs, region=region, ocr_result=ocr_result, image_result=image_data)
    return merge_bugs_by_block(bugs)


def generate_html_report(
    figma_full: str | Path,
    site_full: str | Path,
    bugs: List[Dict[str, Any]],
    *,
    output_path: str | Path = "reports/comparator_final.html",
    nn_result: Optional[Dict[str, Any]] = None,
    verdict: Optional[str] = None,
) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _src(p: str | Path) -> str:
        p = Path(p).resolve()
        try:
            return p.relative_to(out.parent.resolve()).as_posix()
        except ValueError:
            return p.as_uri()

    stamp = datetime.now().strftime("%d.%m.%Y %H:%M")
    scores_html = ""
    if nn_result and "scores" in nn_result:
        rows = "".join(
            f"<tr><td>{html_module.escape(k)}</td><td>{v:.4f}</td></tr>"
            for k, v in nn_result["scores"].items()
        )
        scores_html = (
            f"<h2>Оценки NN</h2><p>Вердикт: {html_module.escape(str(verdict or nn_result.get('verdict', '—')))}</p>"
            f"<table><tr><th>Аспект</th><th>Score</th></tr>{rows}</table>"
        )

    body = [
        "<!DOCTYPE html><html lang='ru'><head><meta charset='UTF-8'>",
        "<title>Comparator</title>",
        "<style>body{font-family:Segoe UI,sans-serif;margin:20px}",
        ".bug{border:1px solid #ddd;padding:12px;margin:10px 0;border-radius:6px}",
        ".high{border-left:4px solid #c62828}.medium{border-left:4px solid #ef6c00}</style>",
        "</head><body>",
        f"<h1>Отчёт MultiAspectComparator</h1><p>{stamp}</p>",
        f"<img src='{_src(figma_full)}' width='240'> <img src='{_src(site_full)}' width='240'>",
        scores_html,
        f"<h2>Багов: {len(bugs)}</h2>",
    ]
    for bug in bugs:
        sev = bug.get("severity", "medium")
        body.append(
            f"<div class='bug {sev}'>"
            f"<b>{html_module.escape(str(bug.get('title', '')))}</b><br>"
            f"{html_module.escape(str(bug.get('description', '')))}"
            f"<br><small>score={bug.get('nn_score', '—')}</small></div>"
        )
    body.append("</body></html>")
    page = "".join(body)
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"bugs": bugs, "nn_result": nn_result}, f, ensure_ascii=False, indent=2)
    return str(out)
