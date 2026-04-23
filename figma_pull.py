from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.figma_client import export_frame_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="ключ файла из URL figma.com/file/KEY/…")
    ap.add_argument("--node", required=True, help="id фрейма, из ссылки node-id=12-34 → 12:34")
    ap.add_argument("--out", default="storage/designs/from_figma.png")
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()
    tok = os.environ.get("FIGMA_ACCESS_TOKEN") or os.environ.get("FIGMA_TOKEN")
    if not tok:
        raise SystemExit("Задайте переменную окружения FIGMA_ACCESS_TOKEN")
    out = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    p = export_frame_png(args.file, args.node, tok, out, scale=args.scale)
    print(p)


if __name__ == "__main__":
    main()
