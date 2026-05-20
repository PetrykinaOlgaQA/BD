#!/usr/bin/env python3
"""
Генерация синтетического датасета для MultiAspectComparator.

  python scripts/generate_comparator_data.py
  python scripts/generate_comparator_data.py --n-train 500 --quick
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.comparator.training.synthetic import generate_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Синтетика Figma vs Site для comparator")
    ap.add_argument("--data-dir", default="data/comparator", help="корень датасета")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=700)
    ap.add_argument("--n-test", type=int, default=700)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="быстрый прогон: 400 / 50 / 50",
    )
    args = ap.parse_args()

    n_train, n_val, n_test = args.n_train, args.n_val, args.n_test
    if args.quick:
        n_train, n_val, n_test = 400, 50, 50

    data_dir = ROOT / args.data_dir
    counts = generate_dataset(
        data_dir,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=args.seed,
    )

    print("\nГотово!")
    print(f"  Train: {counts['train']} пар -> {data_dir / 'manifest_train.jsonl'}")
    print(f"  Val:   {counts['val']} пар -> {data_dir / 'manifest_val.jsonl'}")
    print(f"  Test:  {counts['test']} пар -> {data_dir / 'manifest_test.jsonl'}")
    print(f"  PNG:   {data_dir.resolve()}/train|val|test/")


if __name__ == "__main__":
    main()
