#!/usr/bin/env python3
"""Проверка подавления ложных «фрагмента нет на макете» при вставке блока."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.structural_shift import filter_structural_shift_bug_items, should_suppress_presence_at_bbox


def main() -> None:
    manifest = os.path.join(ROOT, "data", "structural_shift", "manifest.jsonl")
    if not os.path.isfile(manifest):
        print("skip: no manifest", manifest)
        return
    ok = fail = 0
    for line in open(manifest, encoding="utf-8"):
        rec = json.loads(line)
        if not rec.get("insert_extra_block"):
            continue
        base = np.array(Image.open(rec["baseline"]).convert("RGB"))
        cur = np.array(Image.open(rec["current"]).convert("RGB"))
        mh, ch, w = base.shape[0], cur.shape[0], base.shape[1]
        found = False
        for y in range(max(0, mh - 24), ch - 48, 12):
            if should_suppress_presence_at_bbox(
                base, cur, 0, y, w, 64, missing_on_mockup=True, missing_on_page=False
            ):
                found = True
                break
        items = [{"text": "фрагмента нет на макете", "x": 0, "y": y, "w": w, "h": 64}]
        _, n = filter_structural_shift_bug_items(items, baseline_rgb=base, current_rgb=cur)
        if found and n == 1:
            ok += 1
        else:
            fail += 1
    print(f"insert_extra: suppress ok={ok} fail={fail}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
