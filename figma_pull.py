from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.figma_client import export_frame_png, save_nodes_json


def main():
    ap = argparse.ArgumentParser(
        description="Экспорт макета из Figma (PNG и/или JSON узла). Токен: FIGMA_ACCESS_TOKEN или FIGMA_TOKEN."
    )
    ap.add_argument("--file", required=True, help="ключ файла из URL …/design/KEY/…")
    ap.add_argument("--node", required=True, help="id фрейма: из node-id=19-2 в URL → 19:2")
    ap.add_argument("--out", default="storage/designs/from_figma.png", help="куда сохранить PNG")
    ap.add_argument("--scale", type=int, default=2, help="масштаб рендера (1–4)")
    ap.add_argument(
        "--json-out",
        default=None,
        help="если указан путь — дополнительно сохранить ответ API nodes (JSON)",
    )
    args = ap.parse_args()
    tok = os.environ.get("FIGMA_ACCESS_TOKEN") or os.environ.get("FIGMA_TOKEN")
    if not tok:
        raise SystemExit("Задайте переменную окружения FIGMA_ACCESS_TOKEN (не коммитьте в git).")
    out = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    p = export_frame_png(args.file, args.node, tok, out, scale=args.scale)
    print("PNG:", p)
    if args.json_out:
        jpath = os.path.join(ROOT, args.json_out) if not os.path.isabs(args.json_out) else args.json_out
        save_nodes_json(args.file, args.node, tok, jpath)
        print("JSON:", jpath)


if __name__ == "__main__":
    main()
