"""
Опционально: выгрузить небольшую выборку VisionTriage (HF) в data/bugs/external/
для справочника формулировок багов. CNN учится на картах diff — см. build_train_dataset.py.

  pip install datasets huggingface_hub
  python scripts/import_visiontriage_sample.py --limit 40
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40, help="Сколько записей сохранить")
    ap.add_argument(
        "--out",
        default=os.path.join(ROOT, "data", "bugs", "external", "visiontriage"),
        help="Каталог выгрузки",
    )
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Установите: pip install datasets huggingface_hub") from e

    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset("tathadn/visiontriage-multimodal", split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= args.limit:
            break
        rec = {
            "bug_type": row.get("bug_type"),
            "severity": row.get("severity"),
            "bug_report": row.get("bug_report") or row.get("description"),
        }
        rows.append(rec)
        img = row.get("image")
        if img is not None and hasattr(img, "save"):
            img.save(os.path.join(args.out, f"sample_{i:03d}.png"))

    meta_path = os.path.join(args.out, "samples.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"OK: {len(rows)} записей → {args.out}")
    print(f"тексты багов: {meta_path}")


if __name__ == "__main__":
    main()
