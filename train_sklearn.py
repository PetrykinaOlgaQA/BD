"""
Обучение классификатора diff без PyTorch (если c10.dll / WinError 1114).

  pip install scikit-learn joblib
  python train_sklearn.py --epochs 40
  config.json → "model_path": "weights/diff_cnn_sklearn.joblib"
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))


def collect(root: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for label, sub in [(0, "pass"), (1, "fail")]:
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                out.append((os.path.join(d, name), label))
    return out


def load_vec(path: str, size: int) -> np.ndarray:
    im = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return (np.asarray(im, dtype=np.float32).flatten() / 255.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="MLP по diff 64×64 без PyTorch")
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--out", default="weights/diff_cnn_sklearn.joblib")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--min-total", type=int, default=4)
    args = ap.parse_args()

    try:
        import joblib
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise SystemExit("Установите: pip install scikit-learn joblib")

    items = collect(args.data)
    if len(items) < args.min_total:
        raise SystemExit(
            f"Мало данных ({len(items)}). Сначала: python scripts/bootstrap_train_dataset.py --pass 64 --fail 64"
        )
    random.shuffle(items)
    n = max(1, int(len(items) * 0.85))
    train_i, val_i = items[:n], items[n:] or items[-1:]
    X_tr = np.stack([load_vec(p, args.size) for p, _ in train_i])
    y_tr = np.array([y for _, y in train_i], dtype=np.int64)
    X_va = np.stack([load_vec(p, args.size) for p, _ in val_i])
    y_va = np.array([y for _, y in val_i], dtype=np.int64)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    max_iter=400,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=42,
                ),
            ),
        ]
    )
    clf.fit(X_tr, y_tr)
    acc = float((clf.predict(X_va) == y_va).mean())
    print("val_acc", round(acc, 4), "| samples", len(items))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump({"clf": clf, "size": args.size, "kind": "sklearn_mlp"}, args.out)
    print("saved", args.out)
    print('В config.json: "model_path": "weights/diff_cnn_sklearn.joblib"')


if __name__ == "__main__":
    main()
