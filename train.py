from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Any, List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _load_gray_tensor(path: str, size: int, torch_mod: Any) -> Any:
    im = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return torch_mod.from_numpy(a).unsqueeze(0)


class DiffDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], size: int, torch_mod):
        self.items = items
        self.size = size
        self._t = torch_mod

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        path, y = self.items[i]
        x = _load_gray_tensor(path, self.size, self._t)
        return x, self._t.tensor(y, dtype=self._t.long)


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
    ap = argparse.ArgumentParser(
        description="Обучение TinyDiffCNN на кропах diff 64×64 (классы pass=0, fail=1).",
    )
    ap.add_argument("--data", default="data/train", help="Папка с подкаталогами pass/ и fail/")
    ap.add_argument("--out", default="weights/diff_cnn.pt", help="Куда сохранить state_dict")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=64, help="Размер стороны (как в пайплайне)")
    ap.add_argument("--min-total", type=int, default=4, help="Минимум картинок всего (pass+fail)")
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
    except OSError as e:
        print(
            "PyTorch не смог загрузить нативные библиотеки (часто c10.dll / WinError 1114).\n"
            "Установите «Microsoft Visual C++ Redistributable» x64 с сайта Microsoft "
            "или переустановите torch с https://pytorch.org/get-started/locally/\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    from src.model_net import TinyDiffCNN

    items = collect(args.data)
    if len(items) < args.min_total:
        raise SystemExit(
            f"Мало данных: найдено {len(items)} файлов в {args.data}/pass и …/fail. "
            f"Нужно минимум {args.min_total}. Сгенерируй синтетику: "
            f"python scripts/bootstrap_train_dataset.py --pass 48 --fail 48\n"
            "Для «своей» модели замени картинки на свои кропы карт diff (64×64, ч/б) из прогонов."
        )
    random.shuffle(items)
    n = max(1, int(len(items) * 0.85))
    train_i, val_i = items[:n], items[n:]
    if not val_i:
        val_i = train_i[-1:]
    tr = DataLoader(DiffDataset(train_i, args.size, torch), batch_size=args.batch, shuffle=True)
    va = DataLoader(DiffDataset(val_i, args.size, torch), batch_size=args.batch)
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
    print("saved", args.out, "| укажи этот путь в config.json → model_path")


if __name__ == "__main__":
    main()
