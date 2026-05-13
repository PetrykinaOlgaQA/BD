"""
Генерирует простой учебный датасет 64×64 (карты diff): pass — слабый шум, fail — сильный контраст.
Чтобы реально учить под свой проект, замените картинки в data/train на свои кропы diff из shots/diffs.
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_split(sub: str, n: int, strong: bool) -> None:
    d = os.path.join(ROOT, "data", "train", sub)
    os.makedirs(d, exist_ok=True)
    for name in os.listdir(d):
        if name.lower().startswith("synth_") and name.lower().endswith(".png"):
            try:
                os.remove(os.path.join(d, name))
            except OSError:
                pass
    for i in range(n):
        a = np.random.RandomState(i + (1000 if strong else 0)).randint(0, 256, (64, 64), dtype=np.uint8)
        if not strong:
            a = (a.astype(np.float32) * 0.15 + 110).clip(0, 255).astype(np.uint8)
        else:
            for _ in range(8):
                x0, y0 = random.randint(0, 55), random.randint(0, 55)
                a[y0 : y0 + 8, x0 : x0 + 8] = random.choice([0, 255])
        Image.fromarray(a, mode="L").save(os.path.join(d, f"synth_{sub}_{i:03d}.png"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Синтетика 64×64 для первого обучения TinyDiffCNN")
    ap.add_argument("--pass", dest="n_pass", type=int, default=48, help="Сколько PNG класса pass")
    ap.add_argument("--fail", dest="n_fail", type=int, default=48, help="Сколько PNG класса fail")
    args = ap.parse_args()
    random.seed(42)
    _write_split("pass", max(1, args.n_pass), strong=False)
    _write_split("fail", max(1, args.n_fail), strong=True)
    print(f"OK: data/train/pass ({args.n_pass}) и data/train/fail ({args.n_fail}) — дальше: python train.py")


if __name__ == "__main__":
    main()
