#!/usr/bin/env python3
"""Добавляет в manifest_train кропы из баг-репорта с разметкой по фидбеку."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.comparator.models.multi_aspect import ASPECT_KEYS

ROOT = Path(__file__).resolve().parents[1]
BUG_TABLE = ROOT / "reports" / "bug_table_20260519_205734"
OUT_DIR = ROOT / "data" / "comparator" / "feedback"
MANIFEST = ROOT / "data" / "comparator" / "manifest_train.jsonl"

# index в bug_table (exp_N / act_N) -> labels (целевые scores)
FEEDBACK: dict[int, dict[str, float]] = {
    0: {  # баг 1: размер контейнера
        "overall_similarity": 0.52,
        "layout_match": 0.38,
        "typography_match": 0.55,
        "text_match": 0.90,
        "image_match": 0.92,
        "color_match": 0.88,
    },
    2: {  # баг 3 header — ложный layout
        "overall_similarity": 0.88,
        "layout_match": 0.90,
        "typography_match": 0.92,
        "text_match": 0.94,
        "image_match": 0.86,
        "color_match": 0.90,
    },
    3: {  # баг 4 logo — ложный
        "overall_similarity": 0.90,
        "layout_match": 0.88,
        "typography_match": 0.92,
        "text_match": 0.95,
        "image_match": 0.84,
        "color_match": 0.88,
    },
    4: {  # баг 5 — только картинка (emoji)
        "overall_similarity": 0.58,
        "image_match": 0.42,
        "text_match": 0.88,
        "layout_match": 0.85,
        "typography_match": 0.90,
        "color_match": 0.92,
    },
    7: {  # баг 8 — картинка
        "overall_similarity": 0.60,
        "image_match": 0.45,
        "text_match": 0.90,
        "layout_match": 0.82,
        "typography_match": 0.88,
        "color_match": 0.94,
    },
    8: {  # баг 9
        "overall_similarity": 0.58,
        "image_match": 0.44,
        "text_match": 0.88,
        "layout_match": 0.80,
        "typography_match": 0.86,
        "color_match": 0.93,
    },
    11: {  # баг 12 stats — идентичны
        "overall_similarity": 0.92,
        "text_match": 0.94,
        "layout_match": 0.95,
        "typography_match": 0.93,
        "image_match": 0.96,
        "color_match": 0.95,
    },
    12: {  # баг 13 stats-grid — 700M vs 600M
        "overall_similarity": 0.58,
        "text_match": 0.48,
        "layout_match": 0.94,
        "typography_match": 0.90,
        "image_match": 0.92,
        "color_match": 0.90,
    },
}


def _ones() -> dict[str, float]:
    return {k: 1.0 for k in ASPECT_KEYS}


def main() -> None:
    if not BUG_TABLE.is_dir():
        raise SystemExit(f"Нет папки {BUG_TABLE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    added = 0
    lines: list[str] = []

    for idx, labels in FEEDBACK.items():
        exp = BUG_TABLE / f"exp_{idx}.png"
        act = BUG_TABLE / f"act_{idx}.png"
        if not exp.is_file() or not act.is_file():
            print(f"пропуск {idx}: нет кропов")
            continue
        prefix = f"feedback_{idx:02d}"
        figma_name = f"{prefix}_figma.png"
        site_name = f"{prefix}_site.png"
        shutil.copy2(exp, OUT_DIR / figma_name)
        shutil.copy2(act, OUT_DIR / site_name)
        record = {
            "figma": f"data/comparator/feedback/{figma_name}",
            "site": f"data/comparator/feedback/{site_name}",
            "labels": {**_ones(), **labels},
            "aug_type": "koto_feedback",
        }
        lines.append(json.dumps(record, ensure_ascii=False))
        added += 1

    with open(MANIFEST, "a", encoding="utf-8") as mf:
        for line in lines:
            mf.write(line + "\n")

    print(f"Добавлено {added} пар в {MANIFEST}")


if __name__ == "__main__":
    main()
