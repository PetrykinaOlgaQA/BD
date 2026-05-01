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

from src.model_net import TinyDiffCNN


def _load_gray_tensor(path: str, size: int) -> torch.Tensor:
    im = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(a).unsqueeze(0)


class DiffDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], size: int = 64):
        self.items = items
        self.size = size

    def __len__(self):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--out", default="weights/diff_cnn.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    items = collect(args.data)
    if len(items) < 2:
        raise SystemExit("Нужны пары изображений в data/train/pass и data/train/fail")
    random.shuffle(items)
    n = max(1, int(len(items) * 0.85))
    train_i, val_i = items[:n], items[n:]
    if not val_i:
        val_i = train_i[-1:]
    tr = DataLoader(DiffDataset(train_i), batch_size=args.batch, shuffle=True)
    va = DataLoader(DiffDataset(val_i), batch_size=args.batch)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDiffCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in tr:
            x = x.to(dev)
            y = y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * x.size(0)
        model.eval()
        correct = 0
        count = 0
        with torch.no_grad():
            for x, y in va:
                x = x.to(dev)
                y = y.to(dev)
                pred = model(x).argmax(dim=1)
                correct += int((pred == y).sum().item())
                count += x.size(0)
        acc = correct / max(1, count)
        print("epoch", ep + 1, "train_loss", tot / max(1, len(train_i)), "val_acc", round(acc, 4))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
