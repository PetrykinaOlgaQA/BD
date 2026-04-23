from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.manifest import load_manifest, resolve_screen
from src.pipeline import DualRunConfig, RunConfig, run_dual_pipeline, run_pipeline


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _join(root: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--real", default=None, help="URL реального сайта")
    ap.add_argument("--local", default=None, help="URL локалки")
    ap.add_argument("--url", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--screen", default=None)
    ap.add_argument("--no-gemma", action="store_true")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--no-gemma-image", action="store_true")
    ap.add_argument("--batch", default=None)
    args = ap.parse_args()
    cfg_path = args.config
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(ROOT, "config.example.json")
    raw = load_json(cfg_path)

    ur = (args.real or raw.get("url_real") or "").strip()
    ul = (args.local or raw.get("url_local") or "").strip()

    def dual_cfg():
        w, h = tuple(raw.get("window_size", [1280, 720]))
        return DualRunConfig(
            url_real=ur,
            url_local=ul,
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

    if ur and ul and not args.batch:
        out = run_dual_pipeline(dual_cfg(), log=print)
        print("PASS" if out.ok else "FAIL", out.report_txt)
        raise SystemExit(0 if out.ok else 1)

    url = args.url or raw.get("url") or "about:blank"
    baseline = args.baseline or raw.get("baseline_path")
    man_rel = raw.get("manifest_path", "storage/manifest.json")
    if args.screen:
        man = load_manifest(ROOT, man_rel)
        bp, u = resolve_screen(ROOT, man, args.screen)
        baseline = bp
        if not args.url:
            url = u
    baseline = _join(ROOT, baseline) if baseline else ""
    if not baseline or not os.path.isfile(baseline):
        raise SystemExit(
            "Нужны оба URL: url_real и url_local в config.json или флаги --real и --local. "
            "Либо режим эталона: baseline_path + url."
        )

    def one(u: str):
        return RunConfig(
            url=u,
            baseline_path=baseline,
            screenshot_dir=_join(ROOT, raw.get("screenshot_dir", "shots")),
            reports_dir=_join(ROOT, raw.get("reports_dir", "reports")),
            diff_threshold_pct=float(raw.get("diff_threshold_pct", 0.5)),
            ollama_url=raw.get("ollama_url", "http://127.0.0.1:11434"),
            gemma_model=raw.get("gemma_model", "gemma3"),
            use_gemma=not args.no_gemma,
            model_path=_join(ROOT, raw.get("model_path", "weights/diff_cnn.pt")),
            use_model=not args.no_model,
            window_size=tuple(raw.get("window_size", [1280, 720])),
            gemma_use_image=not args.no_gemma_image,
            tolerance_shift_px=int(raw.get("tolerance_shift_px", 0)),
            tolerance_speckle_iter=int(raw.get("tolerance_speckle_iter", 0)),
            pixel_threshold=int(raw.get("pixel_threshold", 30)),
        )

    if args.batch:
        bad = 0
        with open(args.batch, encoding="utf-8") as bf:
            for line in bf:
                u = line.strip()
                if not u or u.startswith("#"):
                    continue
                out = run_pipeline(one(u))
                print(u, "PASS" if out.ok else "FAIL")
                if not out.ok:
                    bad += 1
        raise SystemExit(bad)
    out = run_pipeline(one(url))
    print("PASS" if out.ok else "FAIL", out.report_txt)


if __name__ == "__main__":
    main()
