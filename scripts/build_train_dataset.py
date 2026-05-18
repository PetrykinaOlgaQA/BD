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

import numpy as np

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_RUNS_DIFF_RE = re.compile(r"^Diff:\s*(.+)$", re.MULTILINE)
_RUNS_CR_RE = re.compile(r"Changed pixels \(итог\):\s*([\d.]+)%")
_RUNS_GEMMA_BUG_RE = re.compile(r"^-\s+(.+)$", re.MULTILINE)


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


def _parse_runs_txt(reports_dir: str) -> Dict[str, float]:
    """Карта diff → changed_ratio_pct из reports/runs_*.txt (все прогоны, не только last)."""
    meta: Dict[str, float] = {}
    if not os.path.isdir(reports_dir):
        return meta
    for name in os.listdir(reports_dir):
        if not name.startswith("runs_") or not name.endswith(".txt"):
            continue
        path = os.path.join(reports_dir, name)
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            continue
        blocks = re.split(r"\n={10,}\n", text)
        for block in blocks:
            dm = _RUNS_DIFF_RE.search(block)
            cm = _RUNS_CR_RE.search(block)
            if not dm or not cm:
                continue
            try:
                meta[os.path.normpath(dm.group(1).strip())] = float(cm.group(1))
            except ValueError:
                continue
    return meta


def _estimate_changed_ratio_pct(diff_path: str, size: int = 64) -> float:
    """Оценка доли изменённых пикселей по карте diff (если нет метаданных прогона)."""
    try:
        im = Image.open(diff_path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
        return float((arr > 30).mean() * 100.0)
    except OSError:
        return 0.0


def _extract_bug_samples_from_runs(reports_dir: str, out_path: str) -> int:
    """Черновики формулировок из Gemma-блоков runs_*.txt → data/bugs/samples_from_runs.jsonl."""
    rows: List[str] = []
    seen: set[str] = set()
    if not os.path.isdir(reports_dir):
        return 0
    for name in os.listdir(reports_dir):
        if not name.startswith("runs_") or not name.endswith(".txt"):
            continue
        try:
            text = open(os.path.join(reports_dir, name), encoding="utf-8").read()
        except OSError:
            continue
        for block in re.split(r"\n={10,}\n", text):
            diff_m = _RUNS_DIFF_RE.search(block)
            if not diff_m:
                continue
            diff = os.path.normpath(diff_m.group(1).strip())
            for line in _RUNS_GEMMA_BUG_RE.findall(block):
                s = line.strip()
                if len(s) < 12 or s in seen:
                    continue
                if s.startswith("[") or "ollama" in s.lower():
                    continue
                seen.add(s)
                rows.append(
                    json.dumps({"text": s, "source_diff": diff}, ensure_ascii=False)
                )
    if not rows:
        return 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    mode = "a" if os.path.isfile(out_path) else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        if mode == "a" and os.path.getsize(out_path) > 0:
            f.write("\n")
        f.write("\n".join(rows) + "\n")
    return len(rows)


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
    ap.add_argument(
        "--samples-out",
        default=os.path.join(ROOT, "data", "bugs", "samples_from_runs.jsonl"),
        help="Тексты багов из runs_*.txt",
    )
    ap.add_argument(
        "--infer-missing",
        action="store_true",
        help="Оценить changed_ratio по PNG, если нет записи в отчётах",
    )
    ap.add_argument("--append-labels", action="store_true", help="Дописать labels.jsonl, не перезаписывать")
    args = ap.parse_args()

    pass_dir = os.path.join(args.out, "pass")
    fail_dir = os.path.join(args.out, "fail")
    for d in (pass_dir, fail_dir):
        os.makedirs(d, exist_ok=True)

    meta_by_diff: Dict[str, float] = {}
    meta_by_diff.update(_parse_runs_txt(args.reports))
    for row in _load_json_sidecars(args.reports):
        diff = row.get("diff_path") or row.get("diff")
        cr = row.get("changed_ratio_pct")
        if diff and cr is not None:
            try:
                meta_by_diff[os.path.normpath(str(diff))] = float(cr)
            except (TypeError, ValueError):
                pass

    n_samples = _extract_bug_samples_from_runs(args.reports, args.samples_out)

    diff_files = _find_diff_png(args.shots)
    labels_path = args.labels_out
    os.makedirs(os.path.dirname(labels_path) or ".", exist_ok=True)
    n_pass = n_fail = 0
    label_rows: List[str] = []

    existing_pass = len(
        [f for f in os.listdir(pass_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    )
    existing_fail = len(
        [f for f in os.listdir(fail_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    )

    for src in diff_files:
        norm = os.path.normpath(src)
        cr = meta_by_diff.get(norm)
        if cr is None and args.infer_missing:
            cr = _estimate_changed_ratio_pct(norm, args.size)
        if cr is None:
            continue
        is_fail = cr > float(args.fail_threshold_pct)
        if is_fail and existing_fail + n_fail >= args.max_per_class:
            continue
        if not is_fail and existing_pass + n_pass >= args.max_per_class:
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

    if label_rows:
        mode = "a" if args.append_labels and os.path.isfile(labels_path) else "w"
        with open(labels_path, mode, encoding="utf-8") as f:
            if mode == "a" and os.path.getsize(labels_path) > 0:
                f.write("\n")
            f.write("\n".join(label_rows) + "\n")

    total_pass = existing_pass + n_pass
    total_fail = existing_fail + n_fail
    print(f"добавлено pass: {n_pass}  fail: {n_fail}  (всего pass={total_pass} fail={total_fail}) -> {args.out}")
    print(f"метки: {labels_path}  (+{len(label_rows)} строк)")
    if n_samples:
        print(f"тексты багов: {args.samples_out} (+{n_samples})")
    print(f"метаданных diff: {len(meta_by_diff)}")
    if n_pass + n_fail < 4:
        print("Мало файлов. Сначала прогоните сверку или: python scripts/bootstrap_train_dataset.py")
    else:
        print("Доп. UI-баги: python scripts/import_external_to_train.py --limit 80")
        print("Обучение: python train.py --data data/train --epochs 35 --out weights/diff_cnn.pt")


if __name__ == "__main__":
    main()
