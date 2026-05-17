"""
Сбор датасета для TinyDiffCNN из прогонов пайплайна (shots/diffs, reports).

pass — changed_ratio_pct <= порога; fail — выше порога (или явная метка в labels.jsonl).

  python scripts/build_train_dataset.py
  python scripts/build_train_dataset.py --fail-threshold 0.5 --max-per-class 200
  python train.py --data data/train --epochs 35
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_json_sidecars(reports_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isdir(reports_dir):
        return rows
    for name in os.listdir(reports_dir):
        if not name.endswith("_last.json") and not name.endswith(".json"):
            continue
        path = os.path.join(reports_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                rows.append(data)
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def _find_diff_png(shots_dir: str) -> List[str]:
    diffs: List[str] = []
    d = os.path.join(shots_dir, "diffs")
    if not os.path.isdir(d):
        return diffs
    for name in os.listdir(d):
        if name.lower().endswith(".png") and "diff" in name.lower():
            diffs.append(os.path.join(d, name))
    diffs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return diffs


def _to_64_gray(src: str, dst: str, size: int) -> None:
    im = Image.open(src).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    im.save(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Датасет pass/fail из карт diff прогонов")
    ap.add_argument("--shots", default=os.path.join(ROOT, "shots"), help="Папка скринов")
    ap.add_argument("--reports", default=os.path.join(ROOT, "reports"), help="Папка отчётов")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "train"), help="data/train/pass|fail")
    ap.add_argument("--fail-threshold-pct", type=float, default=0.5, help="Выше — класс fail")
    ap.add_argument("--max-per-class", type=int, default=300)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--labels-out", default=os.path.join(ROOT, "data", "bugs", "labels.jsonl"))
    args = ap.parse_args()

    pass_dir = os.path.join(args.out, "pass")
    fail_dir = os.path.join(args.out, "fail")
    for d in (pass_dir, fail_dir):
        os.makedirs(d, exist_ok=True)

    meta_by_diff: Dict[str, float] = {}
    for row in _load_json_sidecars(args.reports):
        diff = row.get("diff_path") or row.get("diff")
        cr = row.get("changed_ratio_pct")
        if diff and cr is not None:
            try:
                meta_by_diff[os.path.normpath(str(diff))] = float(cr)
            except (TypeError, ValueError):
                pass

    diff_files = _find_diff_png(args.shots)
    labels_path = args.labels_out
    os.makedirs(os.path.dirname(labels_path) or ".", exist_ok=True)
    n_pass = n_fail = 0
    label_rows: List[str] = []

    for src in diff_files:
        norm = os.path.normpath(src)
        cr = meta_by_diff.get(norm)
        if cr is None:
            continue
        is_fail = cr > float(args.fail_threshold_pct)
        if is_fail and n_fail >= args.max_per_class:
            continue
        if not is_fail and n_pass >= args.max_per_class:
            continue
        sub = "fail" if is_fail else "pass"
        base = re.sub(r"[^\w.-]+", "_", os.path.basename(src))[:80]
        dst = os.path.join(args.out, sub, base)
        if os.path.isfile(dst):
            continue
        _to_64_gray(src, dst, args.size)
        if is_fail:
            n_fail += 1
        else:
            n_pass += 1
        label_rows.append(
            json.dumps(
                {"path": dst, "label": sub, "changed_ratio_pct": cr, "source_diff": norm},
                ensure_ascii=False,
            )
        )

    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_rows) + ("\n" if label_rows else ""))

    print(f"pass: {n_pass}  fail: {n_fail}  → {args.out}")
    print(f"метки: {labels_path}")
    if n_pass + n_fail < 4:
        print("Мало файлов. Сначала прогоните сверку или: python scripts/bootstrap_train_dataset.py")
    else:
        print("Доп. UI-баги: python scripts/import_external_to_train.py --limit 80")
        print("Обучение: python train.py --data data/train --epochs 35 --out weights/diff_cnn.pt")


if __name__ == "__main__":
    main()
