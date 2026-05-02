from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, jsonify, render_template, request

from src.pipeline import FigmaVsSiteConfig, run_figma_vs_site

app = Flask(__name__, template_folder=os.path.join(ROOT, "templates"), static_folder=os.path.join(ROOT, "static"))


def _load_cfg() -> Dict[str, Any]:
    p = os.path.join(ROOT, "config.json")
    if not os.path.isfile(p):
        p = os.path.join(ROOT, "config.example.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/config")
def api_config():
    c = _load_cfg()
    w, h = tuple(c.get("window_size", [1280, 720]))
    fg = c.get("figma") or {}
    return jsonify(
        {
            "url_site": c.get("url_site", c.get("url_local", "")),
            "figma_file_key": fg.get("file_key", ""),
            "figma_node_id": fg.get("node_id", ""),
            "figma_use_cached_png": bool(fg.get("use_cached_png", True)),
            "window_w": int(w),
            "window_h": int(h),
            "diff_threshold_pct": float(c.get("diff_threshold_pct", 0.5)),
            "tolerance_shift_px": int(c.get("tolerance_shift_px", 2)),
            "tolerance_speckle_iter": int(c.get("tolerance_speckle_iter", 1)),
            "pixel_threshold": int(c.get("pixel_threshold", 30)),
            "ollama_url": c.get("ollama_url", "http://127.0.0.1:11434"),
            "gemma_model": c.get("gemma_model", "gemma3:latest"),
            "figma_scale": int(fg.get("scale", 1)),
            "capture_wait_seconds": float(c.get("capture_wait_seconds", 12)),
        }
    )


@app.post("/api/run")
def api_run():
    body = request.get_json(silent=True) or {}
    c = _load_cfg()
    tok = os.environ.get("FIGMA_ACCESS_TOKEN") or os.environ.get("FIGMA_TOKEN")
    if not tok:
        return jsonify({"error": "Нет FIGMA_ACCESS_TOKEN в окружении процесса (задайте до запуска web_server.py)"}), 400

    try:
        ww = int(body.get("window_w") or c.get("window_size", [1280, 720])[0])
        wh = int(body.get("window_h") or c.get("window_size", [1280, 720])[1])
        thr = float(body.get("diff_threshold_pct", c.get("diff_threshold_pct", 0.5)))
        sh = max(0, min(5, int(body.get("tolerance_shift_px", c.get("tolerance_shift_px", 2)))))
        sp = max(0, min(5, int(body.get("tolerance_speckle_iter", c.get("tolerance_speckle_iter", 1)))))
        px = max(0, min(255, int(body.get("pixel_threshold", c.get("pixel_threshold", 30)))))
        scale = max(1, min(4, int(body.get("figma_scale", (c.get("figma") or {}).get("scale", 1)))))
        cap_wait = float(body.get("capture_wait_seconds", c.get("capture_wait_seconds", 12)))
        cap_wait = max(0.0, min(120.0, cap_wait))
    except (TypeError, ValueError):
        return jsonify({"error": "Некорректные числа"}), 400

    fg = c.get("figma") or {}
    fk = (body.get("figma_file_key") or fg.get("file_key") or "").strip()
    nid = (body.get("figma_node_id") or fg.get("node_id") or "").strip()
    site = (body.get("url_site") or c.get("url_site") or c.get("url_local") or "").strip()
    if not fk or not nid or not site:
        return jsonify({"error": "Нужны url_site и figma file_key + node_id (в теле запроса или config)"}), 400

    out_png = os.path.join(ROOT, fg.get("design_png", "storage/designs/figma_baseline_last.png"))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    use_gemma = bool(body.get("use_gemma", True))
    use_model = bool(body.get("use_model", True))
    gemma_img = bool(body.get("gemma_use_image", True))
    figma_use_cached = bool(fg.get("use_cached_png", True))
    if body.get("figma_refresh") or body.get("figma_force_refresh"):
        figma_use_cached = False
    if "figma_use_cached_png" in body:
        figma_use_cached = bool(body.get("figma_use_cached_png"))
    logs: List[str] = []

    def log(msg: str) -> None:
        logs.append(msg)

    fcfg = FigmaVsSiteConfig(
        site_url=site,
        figma_file_key=fk,
        figma_node_id=nid,
        figma_token=tok,
        figma_baseline_png=out_png,
        figma_scale=scale,
        figma_use_cached_png=figma_use_cached,
        screenshot_dir=os.path.join(ROOT, c.get("screenshot_dir", "shots")),
        reports_dir=os.path.join(ROOT, c.get("reports_dir", "reports")),
        diff_threshold_pct=thr,
        ollama_url=(body.get("ollama_url") or c.get("ollama_url", "http://127.0.0.1:11434")).rstrip("/"),
        gemma_model=body.get("gemma_model") or c.get("gemma_model", "gemma3:latest"),
        use_gemma=use_gemma,
        model_path=os.path.join(ROOT, c.get("model_path", "weights/diff_cnn.pt")),
        use_model=use_model,
        window_size=(ww, wh),
        gemma_use_image=gemma_img,
        tolerance_shift_px=sh,
        tolerance_speckle_iter=sp,
        pixel_threshold=px,
        capture_wait_seconds=cap_wait,
    )

    try:
        out = run_figma_vs_site(fcfg, log=log)
    except Exception as e:
        return jsonify({"error": str(e), "logs": logs}), 500

    return jsonify(
        {
            "ok": out.ok,
            "report_txt": out.report_txt,
            "report_html": out.report_html,
            "witness_dir": out.witness_dir,
            "shot_site": out.current_shot,
            "diff_path": out.compare.diff_path,
            "changed_ratio_pct": round(out.compare.changed_ratio * 100, 4),
            "mse": round(out.compare.mse, 6),
            "model_prob_fail": out.model_prob_fail,
            "gemma_markdown": out.gemma_text,
            "logs": logs,
        }
    )


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Веб-панель: сайт vs макет Figma")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
