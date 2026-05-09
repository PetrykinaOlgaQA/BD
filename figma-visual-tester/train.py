"""Обучение TinyDiffCNN на картах diff (папки data/train/pass и data/train/fail)."""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import ensure_runtime_dirs
from src.model import TinyDiffCNN


def _load_gray_tensor(path: str, size: int) -> torch.Tensor:
    im = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(a).unsqueeze(0)


class DiffDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], size: int = 64):
        self.items = items
        self.size = size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, y = self.items[i]
        x = _load_gray_tensor(path, self.size)
        return x, torch.tensor(y, dtype=torch.long)


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


def main() -> None:
    s = ensure_runtime_dirs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(s.data_train_dir))
    ap.add_argument("--out", default=str(s.weights_dir / "diff_cnn_best.pt"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    items = collect(args.data)
    if len(items) < 2:
        raise SystemExit("Нужны изображения в data/train/pass и data/train/fail")

    random.shuffle(items)
    n = max(1, int(len(items) * 0.85))
    train_i, val_i = items[:n], items[n:]
    if not val_i:
        val_i = train_i[-1:]
    tr = DataLoader(DiffDataset(train_i, size=args.size), batch_size=args.batch, shuffle=True)
    va = DataLoader(DiffDataset(val_i, size=args.size), batch_size=args.batch)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDiffCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * x.size(0)
        train_loss = tot / len(train_i)

        model.eval()
        vtot = 0.0
        correct = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                vtot += float(loss_fn(logits, y).item()) * x.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
        val_loss = vtot / max(1, len(val_i))
        acc = correct / max(1, len(val_i))
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"  → сохранено: {args.out}")


if __name__ == "__main__":
    main()
