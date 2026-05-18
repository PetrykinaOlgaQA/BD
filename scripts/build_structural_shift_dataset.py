"""
Синтетика для обучения: «лишний блок на сайте между двумя секциями» ≠ баг presence.

Создаёт пары baseline/current и дополняет fragment_match тройками is_same=1
(контент на сайте есть и на макете, но со сдвигом по Y).

  python scripts/build_structural_shift_dataset.py
  python scripts/build_fragment_match_dataset.py --append-structural --n-structural 800
  python train_fragment_match.py --epochs 20
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _rand_block(rng: random.Random, w: int, h: int, y0: int, bh: int) -> Image.Image:
    im = Image.new("RGB", (w, bh), (rng.randint(230, 255),) * 3)
    d = ImageDraw.Draw(im)
    for _ in range(rng.randint(3, 10)):
        x0, y0b = rng.randint(0, w - 40), rng.randint(0, bh - 20)
        d.rectangle(
            [x0, y0b, x0 + rng.randint(20, 80), y0b + rng.randint(12, 40)],
            fill=(rng.randint(0, 200), rng.randint(0, 200), rng.randint(0, 200)),
        )
    d.text((12, 12), rng.choice(["Секция", "Блок", "Карточка", "Услуга"]), fill=(20, 20, 20))
    return im


def _compose_page(
    rng: random.Random,
    *,
    w: int = 400,
    insert_extra: bool,
) -> Tuple[Image.Image, Image.Image, int]:
    """Figma: A + A. Сайт: A + [extra] + A. Возвращает (baseline, current, y_extra)."""
    bh = rng.randint(70, 110)
    gap = rng.randint(8, 24)
    extra_h = rng.randint(55, 95) if insert_extra else 0
    block_a = _rand_block(rng, w, 0, bh, bh)
    block_b = _rand_block(rng, w, 0, bh, bh)
    extra = _rand_block(rng, w, 0, extra_h, extra_h) if insert_extra else None

    h_base = bh + gap + bh
    h_cur = bh + (extra_h + gap if insert_extra else 0) + gap + bh
    base = Image.new("RGB", (w, h_base), (248, 248, 250))
    cur = Image.new("RGB", (w, h_cur), (248, 248, 250))
    base.paste(block_a, (0, 0))
    base.paste(block_b, (0, bh + gap))
    cur.paste(block_a, (0, 0))
    y_extra = bh + gap
    if insert_extra and extra is not None:
        cur.paste(extra, (0, y_extra))
        cur.paste(block_b, (0, y_extra + extra_h + gap))
    else:
        cur.paste(block_b, (0, bh + gap))
    return base, cur, y_extra


def _save_pair(out_dir: str, idx: int, base: Image.Image, cur: Image.Image, meta: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    bp = os.path.join(out_dir, f"pair_{idx:05d}_figma.png")
    cp = os.path.join(out_dir, f"pair_{idx:05d}_site.png")
    base.save(bp, optimize=True)
    cur.save(cp, optimize=True)
    meta["baseline"] = bp
    meta["current"] = cp


def build_pairs(n: int, out_dir: str, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        insert = rng.random() < 0.72
        base, cur, y_extra = _compose_page(rng, insert_extra=insert)
        meta: Dict[str, Any] = {
            "id": i,
            "insert_extra_block": insert,
            "y_extra": y_extra,
            "label_presence_is_bug": False,
            "note": "Лишний блок между дубликатами — не сообщать «фрагмента нет на макете»",
        }
        _save_pair(out_dir, i, base, cur, meta)
        rows.append(meta)
    return rows


def append_fragment_manifest(n: int, seed: int, panel_size: int) -> int:
    """Тройки для fragment_matcher: is_same=1 при сдвиге, 0 при реально другом блоке."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bfd",
        os.path.join(ROOT, "scripts", "build_fragment_match_dataset.py"),
    )
    bfd = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(bfd)
    _compose_triple = bfd._compose_triple
    _noise_patch = bfd._noise_patch
    _write_manifest_row = bfd._write_manifest_row

    rng = random.Random(seed + 17)
    img_dir = os.path.join(ROOT, "data", "fragment_match", "images")
    fm_root = os.path.join(ROOT, "data", "fragment_match")
    os.makedirs(img_dir, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    start_idx = len([f for f in os.listdir(img_dir) if f.startswith("struct_")]) if os.path.isdir(img_dir) else 0

    for i in range(n):
        insert = rng.random() < 0.65
        base, cur, y_extra = _compose_page(rng, w=panel_size * 3, insert_extra=insert)
        # Панели: макет | diff-шум | сайт (кроп в зоне «лишнего» или совпадающей)
        pw = panel_size
        left = base.crop((0, y_extra, pw, y_extra + pw)) if insert else base.crop((0, 0, pw, pw))
        if insert:
            right = cur.crop((0, y_extra, pw, y_extra + pw))
            is_same = 1
        else:
            right = cur.crop((0, 0, pw, pw))
            is_same = 1 if rng.random() < 0.85 else 0
            if is_same == 0:
                right = cur.crop((pw, 0, pw * 2, pw))
        mid = _noise_patch(pw, rng, "stripe")
        combo = _compose_triple(left, mid, right)
        fname = f"struct_{start_idx + i:05d}.png"
        combo.save(os.path.join(img_dir, fname), optimize=True)
        _write_manifest_row(
            rows,
            image_rel=f"images/{fname}",
            is_same=is_same,
            left_id=f"struct_figma_{i}",
            right_id=f"struct_site_{i}",
            noise_type="structural_shift",
            noise_ratio=rng.uniform(0.25, 0.45),
        )

    manifest = os.path.join(fm_root, "manifest_train.jsonl")
    mode = "a" if os.path.isfile(manifest) and os.path.getsize(manifest) > 0 else "w"
    with open(manifest, mode, encoding="utf-8") as f:
        if mode == "a":
            f.write("\n")
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=200, help="Пар figma/site PNG")
    ap.add_argument("--n-fragment", type=int, default=600, help="Доп. троек в fragment_match train")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "structural_shift"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--panel-size", type=int, default=64)
    ap.add_argument("--skip-fragment", action="store_true")
    args = ap.parse_args()

    pairs_dir = os.path.join(args.out, "pairs")
    rows = build_pairs(args.n_pairs, pairs_dir, args.seed)
    manifest_path = os.path.join(args.out, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    examples_path = os.path.join(ROOT, "data", "bugs", "structural_shift_examples.jsonl")
    os.makedirs(os.path.dirname(examples_path), exist_ok=True)
    with open(examples_path, "w", encoding="utf-8") as f:
        for r in rows[:40]:
            f.write(
                json.dumps(
                    {
                        "text_bad": "в центре: фрагмента нет на макете",
                        "text_ok": "пропустить — на сайте добавлена секция между повторяющимися блоками",
                        "insert_extra": r.get("insert_extra_block"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    n_frag = 0
    if not args.skip_fragment and args.n_fragment > 0:
        n_frag = append_fragment_manifest(args.n_fragment, args.seed, args.panel_size)

    print(f"pairs: {len(rows)} -> {pairs_dir}")
    print(f"examples: {examples_path}")
    if n_frag:
        print(f"fragment_match: +{n_frag} train rows (structural_shift)")
        print("Дальше: python train_fragment_match.py --epochs 20")


if __name__ == "__main__":
    main()
