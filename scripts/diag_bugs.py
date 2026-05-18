#!/usr/bin/env python3
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.baseline_text_cache import ensure_baseline_text_cache
from src.bug_reports import build_bug_report_items
from src.capture import capture_screenshot
from src.compare import build_change_mask
from src.diff_hotspots import analyze_diff_for_qa

c = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
b = os.path.join(ROOT, "shots/figma_cache/baseline.png")
cur = os.path.join(ROOT, "shots/_diag_bugs.png")
_, lay = capture_screenshot(c["url_site"], cur, window_size=tuple(c["window_size"]), wait_seconds=2)
els = lay.get("elements", [])
hot = analyze_diff_for_qa(
    b,
    cur,
    els,
    pixel_threshold=c.get("pixel_threshold", 30),
    tolerance_shift_px=c.get("tolerance_shift_px", 2),
    tolerance_speckle_iter=c.get("tolerance_speckle_iter", 1),
)
from src.bug_reports import _load_aligned_rgb_pair, build_change_mask as bcm
from src.section_compare import build_section_bug_items

br, cr = _load_aligned_rgb_pair(b, cur)
mask, _ = bcm(b, cur, pixel_threshold=30, tolerance_shift_px=2, tolerance_speckle_iter=1)
cache = ensure_baseline_text_cache(b, br, els)
sec = build_section_bug_items(els, br, cr, mask, baseline_text_cache=cache)
print("section direct:", len(sec))
for it in sec[:5]:
    print("  sec:", str(it.get("text", ""))[:90])

from src.bug_reports import _collect_phrases_with_elements

typo_items = []
try:
    from src.section_compare import build_section_bug_items as bsbi

    typo_items = bsbi(els, br, cr, mask, max_items=16, baseline_text_cache=cache)
except Exception as e:
    print("bsbi exc:", e)
    traceback.print_exc()
print("typo_items len:", len(typo_items))

print("has_rgb", br is not None and cr is not None and br.shape[:2] == cr.shape[:2])
from src.bug_reports import _score_all_elements_on_mask, _compact_phrase_for_element, is_broken_bug_line

scored = _score_all_elements_on_mask(mask, els)
scored.sort(key=lambda t: -t[0])
raw_pairs = []
for _frac, sn, el, ch in scored:
    if ch < 2.5:
        continue
    one = _compact_phrase_for_element(mask, el, ch, snippet=sn)
    if one:
        raw_pairs.append(one)
        print("  phrase:", one[:70], "broken?", is_broken_bug_line(one))
print("raw_pairs", len(raw_pairs))

pairs = _collect_phrases_with_elements(
    hot, els, mask, baseline_path=b, baseline_rgb=br, current_rgb=cr, max_lines=20
)
print("pairs:", len(pairs))
for p, el in pairs[:8]:
    print("  pair:", str(p)[:90])

try:
    items = build_bug_report_items(
        hot, els, baseline_path=b, current_path=cur, max_lines=20
    )
    print("bugs:", len(items))
    for it in items[:10]:
        print(" -", str(it.get("text", ""))[:100])
except Exception:
    traceback.print_exc()
