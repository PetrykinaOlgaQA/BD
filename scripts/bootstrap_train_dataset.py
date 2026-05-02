"""
Генерирует простой учебный датасет 64×64 (карты diff): pass — слабый шум, fail — сильный контраст.
Чтобы реально учить под свой проект, замените картинки в data/train на свои кропы diff из reports/shots.
"""
from __future__ import annotations

import os
import random

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_split(sub: str, n: int, strong: bool) -> None:
    d = os.path.join(ROOT, "data", "train", sub)
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        a = np.random.RandomState(i + (1000 if strong else 0)).randint(0, 256, (64, 64), dtype=np.uint8)
        if not strong:
            a = (a.astype(np.float32) * 0.15 + 110).clip(0, 255).astype(np.uint8)
        else:
            for _ in range(8):
                x0, y0 = random.randint(0, 55), random.randint(0, 55)
                a[y0 : y0 + 8, x0 : x0 + 8] = random.choice([0, 255])
        Image.fromarray(a, mode="L").save(os.path.join(d, f"synth_{sub}_{i:03d}.png"))


def main():
    random.seed(42)
    _write_split("pass", 32, strong=False)
    _write_split("fail", 32, strong=True)
    print("OK: data/train/pass and data/train/fail (32 PNG each)")


if __name__ == "__main__":
    main()
