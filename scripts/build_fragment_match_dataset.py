"""
Датасет троек [Фрагмент_A | Помеха | Фрагмент_A/B], метка is_same.

Позитив: правый сегмент = левый (тот же id).
Негатив: правый сегмент = другой фрагмент.

Источник кропов: data/train/pass|fail, shots/diffs (DOMAIN: image).
Для своего домена подставьте пул фрагментов в collect_fragment_pool().

  python scripts/build_fragment_match_dataset.py --n-train 4000 --n-val 600 --n-test 600
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect_fragment_pool() -> List[Tuple[str, str]]:
    """(path, fragment_id) — замените на свой каталог текстов/кода/таблиц."""
    pool: List[Tuple[str, str]] = []
    for sub in ("pass", "fail"):
        d = os.path.join(ROOT, "data", "train", sub)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                pool.append((os.path.join(d, name), f"{sub}_{name}"))
    diffs = os.path.join(ROOT, "shots", "diffs")
    if os.path.isdir(diffs):
        for name in os.listdir(diffs):
            if name.lower().endswith(".png"):
                pool.append((os.path.join(diffs, name), f"diff_{name}"))
    return pool


def _load_patch(path: str, size: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im = im.resize((size, size), Image.Resampling.BILINEAR)
    return im


def _noise_patch(size: int, rng: random.Random, noise_type: str) -> Image.Image:
    if noise_type == "blur":
        base = Image.new("RGB", (size, size), (rng.randint(200, 255),) * 3)
        d = ImageDraw.Draw(base)
        for _ in range(rng.randint(3, 8)):
            x0, y0 = rng.randint(0, size - 10), rng.randint(0, size - 10)
            d.rectangle([x0, y0, x0 + rng.randint(8, 30), y0 + rng.randint(4, 20)], fill=(rng.randint(0, 180),) * 3)
        return base.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.0, 2.5)))
    if noise_type == "stripe":
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        arr[:] = rng.randint(220, 255)
        for x in range(0, size, max(4, size // rng.randint(4, 8))):
            arr[:, x : x + 2] = rng.randint(0, 120)
        return Image.fromarray(arr)
    # random / default
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _compose_triple(left: Image.Image, mid: Image.Image, right: Image.Image) -> Image.Image:
    w = left.width
    canvas = Image.new("RGB", (w * 3, left.height))
    canvas.paste(left, (0, 0))
    canvas.paste(mid, (w, 0))
    canvas.paste(right, (w * 2, 0))
    return canvas


def _write_manifest_row(
    rows: List[Dict[str, Any]],
    *,
    image_rel: str,
    is_same: int,
    left_id: str,
    right_id: str,
    noise_type: str,
    noise_ratio: float,
) -> None:
    rows.append(
        {
            "image": image_rel,
            "is_same": is_same,
            "left_id": left_id,
            "right_id": right_id,
            "noise_type": noise_type,
            "noise_ratio": noise_ratio,
        }
    )


def _class_prefix(fragment_id: str) -> str:
    if fragment_id.startswith("pass_"):
        return "pass"
    if fragment_id.startswith("fail_"):
        return "fail"
    if fragment_id.startswith("diff_"):
        return "diff"
    return ""


def _pick_negative(
    pool: List[Tuple[str, str]],
    left_path: str,
    left_id: str,
    rng: random.Random,
    *,
    hard_ratio: float,
) -> Tuple[str, str]:
    """Случайный или «похожий» негатив (тот же pass/fail, другой кроп)."""
    if rng.random() < hard_ratio:
        pref = _class_prefix(left_id)
        candidates = [
            (p, fid)
            for p, fid in pool
            if fid != left_id and p != left_path and _class_prefix(fid) == pref
        ]
        if candidates:
            return rng.choice(candidates)
    right_path, right_id = rng.choice(pool)
    if right_id == left_id:
        right_path, right_id = rng.choice(pool)
    return right_path, right_id


def generate_split(
    pool: List[Tuple[str, str]],
    out_dir: str,
    n_samples: int,
    rng: random.Random,
    panel_size: int,
    split_name: str,
    *,
    hard_negative_ratio: float = 0.35,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    noise_types = ("random", "blur", "stripe")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_samples):
        left_path, left_id = rng.choice(pool)
        is_same = 1 if rng.random() < 0.5 else 0
        if is_same:
            right_path, right_id = left_path, left_id
        else:
            right_path, right_id = _pick_negative(
                pool, left_path, left_id, rng, hard_ratio=hard_negative_ratio
            )
        noise_type = rng.choice(noise_types)
        noise_ratio = rng.uniform(0.25, 0.45)
        left_im = _load_patch(left_path, panel_size)
        right_im = _load_patch(right_path, panel_size)
        mid_im = _noise_patch(panel_size, rng, noise_type)
        combo = _compose_triple(left_im, mid_im, right_im)
        fname = f"{split_name}_{i:05d}.png"
        combo.save(os.path.join(out_dir, fname), optimize=True)
        _write_manifest_row(
            rows,
            image_rel=f"images/{fname}",
            is_same=is_same,
            left_id=left_id,
            right_id=right_id,
            noise_type=noise_type,
            noise_ratio=noise_ratio,
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "fragment_match"))
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--panel-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=0.35,
        help="Доля негативов из того же класса pass/fail (сложнее для модели)",
    )
    args = ap.parse_args()

    pool = collect_fragment_pool()
    if len(pool) < 4:
        raise SystemExit(
            "Мало фрагментов. Сначала: python scripts/bootstrap_train_dataset.py "
            "или добавьте PNG в data/train/pass|fail"
        )

    rng = random.Random(args.seed)
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    for name, n in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        rows = generate_split(
            pool,
            img_dir,
            n,
            rng,
            args.panel_size,
            name,
            hard_negative_ratio=args.hard_negative_ratio,
        )
        manifest = f"manifest_{name}.jsonl"
        path = os.path.join(args.out, manifest)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        print(f"{manifest}: {len(rows)} samples")

    print(f"OK -> {args.out}")
    print("Обучение: python train_fragment_match.py")


if __name__ == "__main__":
    main()
