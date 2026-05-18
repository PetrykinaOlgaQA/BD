from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import app_config_to_figma_vs_site, parse_app_config, validate_app_config
from src.pipeline import run_figma_vs_site


def main():
    ap = argparse.ArgumentParser(
        description="Тест: сверстанный сайт vs макет Figma (скрин, diff, CNN, текст багов через Ollama)."
    )
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--url", default=None, help="URL страницы под тестом (иначе url_site из config)")
    ap.add_argument("--no-gemma", action="store_true")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--no-gemma-image", action="store_true")
    ap.add_argument("--figma-scale", type=int, default=None, help="масштаб PNG из Figma (1–4)")
    ap.add_argument(
        "--figma-refresh",
        action="store_true",
        help="всегда заново выгрузить макет из Figma (игнор кэша PNG на диске и FIGMA_FORCE_REFRESH не нужен)",
    )
    args = ap.parse_args()

    cfg_path = args.config
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(ROOT, "config.example.json")

    tok = (os.environ.get("FIGMA_ACCESS_TOKEN") or os.environ.get("FIGMA_TOKEN") or "").strip()

    raw = json.load(open(cfg_path, encoding="utf-8"))
    site = (args.url or raw.get("url_site") or raw.get("url_local") or "").strip()
    app_cfg = parse_app_config(raw, ROOT)
    validate_app_config(app_cfg, site_url=site)
    if args.figma_scale is not None:
        app_cfg.figma.scale = max(1, min(4, int(args.figma_scale)))
    figma_use_cached = bool(app_cfg.figma.use_cached_png) and not args.figma_refresh
    cache_png = app_cfg.resolved_design_png(ROOT)
    cache_ok = os.path.isfile(cache_png) and os.path.getsize(cache_png) > 64
    if not tok:
        if args.figma_refresh or not figma_use_cached or not cache_ok:
            raise SystemExit(
                "Задайте FIGMA_ACCESS_TOKEN (нужен для выгрузки макета). "
                "Без токена можно только с кэшем: figma.use_cached_png=true и файл макета на диске."
            )
        tok = "cache-only"
        print(f"FIGMA_ACCESS_TOKEN не задан — макет из кэша: {cache_png}")

    fcfg = app_config_to_figma_vs_site(
        app_cfg,
        ROOT,
        site,
        tok,
        figma_use_cached,
        use_gemma=not args.no_gemma,
        use_model=not args.no_model,
        gemma_use_image=not args.no_gemma_image,
    )

    out = run_figma_vs_site(fcfg, log=print)
    print("PASS" if out.ok else "FAIL", out.report_txt)
    if out.report_html:
        print("HTML:", out.report_html)
    raise SystemExit(0 if out.ok else 1)


if __name__ == "__main__":
    main()
