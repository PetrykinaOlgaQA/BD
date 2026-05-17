"""
Импорт внешнего UI-баг датасета (VisionTriage) в data/train/fail для обучения CNN.

  pip install datasets huggingface_hub
  python scripts/import_external_to_train.py --limit 80
  python train.py --data data/train --epochs 30
"""
from __future__ import annotations

import argparse
import json
import os
import re

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _to_train_gray(src_path: str, dst_path: str, size: int) -> None:
    im = Image.open(src_path).convert("L")
    im = im.resize((size, size), Image.Resampling.BILINEAR)
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    im.save(dst_path, format="PNG", optimize=True)


def _save_pil_gray(img: Image.Image, dst_path: str, size: int) -> None:
    im = img.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    im.save(dst_path, format="PNG", optimize=True)


def import_synthetic_ui_fails(train_fail_dir: str, size: int, n: int = 40) -> int:
    """Синтетические «баговые» карты diff по фразам из catalog_ru.json."""
    import random

    import numpy as np

    catalog_path = os.path.join(ROOT, "data", "bugs", "catalog_ru.json")
    phrases = ["layout", "font", "margin_top", "missing_on_mockup"]
    if os.path.isfile(catalog_path):
        with open(catalog_path, encoding="utf-8") as f:
            data = json.load(f)
        ip = data.get("issue_phrases") or {}
        if isinstance(ip, dict):
            phrases = list(ip.keys())

    added = 0
    for i in range(n):
        dst = os.path.join(train_fail_dir, f"syn_ui_{i:03d}.png")
        if os.path.isfile(dst):
            continue
        rng = random.Random(i + 17)
        arr = np.zeros((size, size), dtype=np.uint8)
        for _ in range(rng.randint(8, 24)):
            x0, y0 = rng.randint(0, size - 4), rng.randint(0, size - 4)
            w, h = rng.randint(3, 14), rng.randint(3, 14)
            arr[y0 : y0 + h, x0 : x0 + w] = rng.randint(140, 255)
        Image.fromarray(arr, mode="L").save(dst)
        added += 1
    return added


def import_visiontriage(limit: int, train_fail_dir: str, size: int) -> int:
    try:
        from datasets import load_dataset, load_dataset_builder
    except ImportError as e:
        raise SystemExit("Установите: pip install datasets huggingface_hub") from e

    ds = None
    try:
        builder = load_dataset_builder("tathadn/visiontriage-multimodal", split="train")
        ds = builder.as_streaming_dataset(split="train")
    except Exception:
        try:
            ds = load_dataset(
                "tathadn/visiontriage-multimodal",
                split="train",
                streaming=True,
                verification_mode="no_checks",
            )
        except Exception as e:
            raise RuntimeError(str(e)) from e
    n = 0
    meta_dir = os.path.join(ROOT, "data", "bugs", "external", "visiontriage")
    os.makedirs(meta_dir, exist_ok=True)
    meta_rows = []
    for i, row in enumerate(ds):
        if n >= limit:
            break
        img = row.get("image")
        if img is None or not hasattr(img, "save"):
            continue
        dst = os.path.join(train_fail_dir, f"vt_{i:04d}.png")
        if os.path.isfile(dst):
            n += 1
            continue
        _save_pil_gray(img, dst, size)
        meta_rows.append(
            {
                "bug_type": row.get("bug_type"),
                "severity": row.get("severity"),
                "bug_report": (row.get("bug_report") or row.get("description") or "")[:500],
                "train_path": dst,
            }
        )
        n += 1
    meta_path = os.path.join(meta_dir, "train_import.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_rows, f, ensure_ascii=False, indent=2)
    return n


def import_existing_samples(external_dir: str, train_fail_dir: str, size: int) -> int:
    if not os.path.isdir(external_dir):
        return 0
    n = 0
    for name in sorted(os.listdir(external_dir)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        src = os.path.join(external_dir, name)
        base = re.sub(r"[^\w.-]+", "_", os.path.splitext(name)[0])[:60]
        dst = os.path.join(train_fail_dir, f"ext_{base}.png")
        if os.path.isfile(dst):
            continue
        _to_train_gray(src, dst, size)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=80, help="Сколько кадров VisionTriage")
    ap.add_argument("--size", type=int, default=64, help="Размер для CNN")
    ap.add_argument("--train-dir", default=os.path.join(ROOT, "data", "train"))
    ap.add_argument("--skip-hf", action="store_true", help="Только локальные data/bugs/external")
    args = ap.parse_args()

    fail_dir = os.path.join(args.train_dir, "fail")
    pass_dir = os.path.join(args.train_dir, "pass")
    os.makedirs(fail_dir, exist_ok=True)
    os.makedirs(pass_dir, exist_ok=True)

    n_ext = import_existing_samples(
        os.path.join(ROOT, "data", "bugs", "external", "visiontriage"),
        fail_dir,
        args.size,
    )
    n_syn = import_synthetic_ui_fails(fail_dir, args.size, n=min(48, args.limit))
    n_hf = 0
    if not args.skip_hf:
        try:
            n_hf = import_visiontriage(args.limit, fail_dir, args.size)
        except Exception as e:
            print(f"VisionTriage (HF): пропуск — {e}")

    n_fail = len([f for f in os.listdir(fail_dir) if f.lower().endswith(".png")])
    n_pass = len([f for f in os.listdir(pass_dir) if f.lower().endswith(".png")])
    print(f"Добавлено: локально {n_ext}, синтетика {n_syn}, HF {n_hf}")
    print(f"Итого data/train: pass={n_pass} fail={n_fail}")
    if n_fail < 8:
        print("Мало fail — сначала: python scripts/bootstrap_train_dataset.py --fail 48")
    print("Обучение: python train.py --data data/train --epochs 30 --out weights/diff_cnn.pt")


if __name__ == "__main__":
    main()
