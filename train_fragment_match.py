#!/usr/bin/env python3
"""
Обучение сопоставления фрагментов с блоком-помехой (PyTorch + опционально HF ViT).

  python scripts/build_fragment_match_dataset.py
  python train_fragment_match.py
  python eval_fragment_match.py
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.fragment_match.config import FragmentMatchConfig
from src.fragment_match.train_loop import train_fragment_matcher


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune matcher: A + noise + A/B")
    ap.add_argument("--data", default="data/fragment_match")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--encoder", choices=("cnn", "vit_hf"), default="cnn")
    ap.add_argument("--hf-model", default="google/vit-base-patch16-224")
    ap.add_argument("--out", default="weights/fragment_matcher.pt")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    cfg = FragmentMatchConfig()
    cfg.data.root = args.data
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch
    cfg.train.lr = args.lr
    cfg.train.out_path = args.out
    cfg.model.encoder = args.encoder
    cfg.model.hf_model_name = args.hf_model

    train_fragment_matcher(cfg, resume_path=args.resume or None)


if __name__ == "__main__":
    main()
