from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline import FigmaVsSiteConfig, run_figma_vs_site


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _join(root: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))


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
    args = ap.parse_args()

    cfg_path = args.config
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(ROOT, "config.example.json")
    raw = load_json(cfg_path)

    tok = os.environ.get("FIGMA_ACCESS_TOKEN") or os.environ.get("FIGMA_TOKEN")
    if not tok:
        raise SystemExit("Задайте переменную окружения FIGMA_ACCESS_TOKEN (токен Figma, не коммитьте).")

    fg = raw.get("figma") or {}
    fk = (fg.get("file_key") or "").strip()
    nid = (fg.get("node_id") or "").strip()
    site = (args.url or raw.get("url_site") or raw.get("url_local") or "").strip()
    if not fk or not nid or not site:
        raise SystemExit("В config нужны figma.file_key, figma.node_id и url_site (или флаг --url).")

    out_png = _join(ROOT, fg.get("design_png", "storage/designs/figma_baseline_last.png"))
    w, h = tuple(raw.get("window_size", [1280, 720]))
    scale = int(args.figma_scale if args.figma_scale is not None else fg.get("scale", 2))

    fcfg = FigmaVsSiteConfig(
        site_url=site,
        figma_file_key=fk,
        figma_node_id=nid,
        figma_token=tok,
        figma_baseline_png=out_png,
        figma_scale=scale,
        screenshot_dir=_join(ROOT, raw.get("screenshot_dir", "shots")),
        reports_dir=_join(ROOT, raw.get("reports_dir", "reports")),
        diff_threshold_pct=float(raw.get("diff_threshold_pct", 0.5)),
        ollama_url=raw.get("ollama_url", "http://127.0.0.1:11434"),
        gemma_model=raw.get("gemma_model", "gemma3"),
        use_gemma=not args.no_gemma,
        model_path=_join(ROOT, raw.get("model_path", "weights/diff_cnn.pt")),
        use_model=not args.no_model,
        window_size=(int(w), int(h)),
        gemma_use_image=not args.no_gemma_image,
        tolerance_shift_px=int(raw.get("tolerance_shift_px", 2)),
        tolerance_speckle_iter=int(raw.get("tolerance_speckle_iter", 1)),
        pixel_threshold=int(raw.get("pixel_threshold", 30)),
    )

    out = run_figma_vs_site(fcfg, log=print)
    print("PASS" if out.ok else "FAIL", out.report_txt)
    if out.report_html:
        print("HTML:", out.report_html)
    raise SystemExit(0 if out.ok else 1)


if __name__ == "__main__":
    main()
