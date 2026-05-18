#!/usr/bin/env python3
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.bug_reports import _load_aligned_rgb_pair, build_change_mask
from src.capture import capture_screenshot
from src.section_compare import (
    _crop_rgb,
    _group_site_sections,
    _mask_frac,
    _visual_similarity,
    build_section_bug_items,
)

cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
url = cfg["url_site"]
ws = tuple(cfg["window_size"])
base = os.path.join(ROOT, "shots/figma_cache/baseline.png")
cur = os.path.join(ROOT, "shots/_diag_current.png")
_, layout = capture_screenshot(url, cur, window_size=ws, wait_seconds=2)
els = layout.get("elements", [])
mask, hot = build_change_mask(
    base,
    cur,
    pixel_threshold=cfg.get("pixel_threshold", 30),
    tolerance_shift_px=cfg.get("tolerance_shift_px", 2),
    tolerance_speckle_iter=cfg.get("tolerance_speckle_iter", 1),
)
b, c = _load_aligned_rgb_pair(base, cur)
print("global diff %:", round(float(mask.mean()) * 100, 3))
items = build_section_bug_items(els, b, c, mask)
print("section bugs:", len(items))
for it in items[:8]:
    print(" -", str(it.get("text", ""))[:90])
bgr, cgr = b, c
from src.bug_reports import _score_all_elements_on_mask

scored = _score_all_elements_on_mask(mask, els)
scored.sort(key=lambda t: -t[0])
print("top elements by diff:")
for frac, sn, el, ch in scored[:12]:
    t = str(el.get("innerText", ""))[:50].encode("ascii", "replace").decode()
    print(f"  ch={ch:.1f}% {sn} | {t}")
