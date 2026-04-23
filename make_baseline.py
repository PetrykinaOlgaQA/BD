from __future__ import annotations

import argparse
import json
import os
import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.capture import capture_screenshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default="shots/baseline.png")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    w, h = (1280, 720)
    if os.path.isfile(os.path.join(ROOT, args.config)):
        with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
            c = json.load(f)
            w, h = tuple(c.get("window_size", [1280, 720]))
    capture_screenshot(args.url, out, window_size=(w, h))
    cfg_path = os.path.join(ROOT, args.config)
    cfg = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    cfg["url"] = args.url
    cfg["baseline_path"] = os.path.relpath(out, ROOT).replace("\\", "/")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("saved", out)
    print("updated", cfg_path)


if __name__ == "__main__":
    main()
