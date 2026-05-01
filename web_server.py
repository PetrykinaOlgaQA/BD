from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, jsonify, render_template, request

from src.pipeline import DualRunConfig, run_dual_pipeline

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
    return jsonify(
        {
            "url_real": c.get("url_real", c.get("url", "")),
            "url_local": c.get("url_local", ""),
            "window_w": int(w),
            "window_h": int(h),
            "diff_threshold_pct": float(c.get("diff_threshold_pct", 0.5)),
            "tolerance_shift_px": int(c.get("tolerance_shift_px", 2)),
            "tolerance_speckle_iter": int(c.get("tolerance_speckle_iter", 1)),
            "pixel_threshold": int(c.get("pixel_threshold", 30)),
            "ollama_url": c.get("ollama_url", "http://127.0.0.1:11434"),
            "gemma_model": c.get("gemma_model", "gemma3"),
        }
    )


@app.post("/api/run-dual")
def api_run_dual():
    body = request.get_json(silent=True) or {}
    c = _load_cfg()
    try:
        ww = int(body.get("window_w") or c.get("window_size", [1280, 720])[0])
        wh = int(body.get("window_h") or c.get("window_size", [1280, 720])[1])
        thr = float(body.get("diff_threshold_pct", c.get("diff_threshold_pct", 0.5)))
        sh = max(0, min(5, int(body.get("tolerance_shift_px", c.get("tolerance_shift_px", 2)))))
        sp = max(0, min(5, int(body.get("tolerance_speckle_iter", c.get("tolerance_speckle_iter", 1)))))
        px = max(0, min(255, int(body.get("pixel_threshold", c.get("pixel_threshold", 30)))))
    except (TypeError, ValueError):
        return jsonify({"error": "Некорректные числовые поля"}), 400

    ur = (body.get("url_real") or c.get("url_real") or c.get("url") or "").strip()
    ul = (body.get("url_local") or c.get("url_local") or "").strip()
    if not ur or not ul:
        return jsonify({"error": "Нужны URL эталона и тестируемой страницы"}), 400

    use_gemma = bool(body.get("use_gemma", True))
    use_model = bool(body.get("use_model", False))
    gemma_img = bool(body.get("gemma_use_image", True))
    logs: List[str] = []

    def log(msg: str) -> None:
        logs.append(msg)

    cfg = DualRunConfig(
        url_real=ur,
        url_local=ul,
        screenshot_dir=os.path.join(ROOT, c.get("screenshot_dir", "shots")),
        reports_dir=os.path.join(ROOT, c.get("reports_dir", "reports")),
        diff_threshold_pct=thr,
        ollama_url=(body.get("ollama_url") or c.get("ollama_url", "http://127.0.0.1:11434")).rstrip("/"),
        gemma_model=body.get("gemma_model") or c.get("gemma_model", "gemma3"),
        use_gemma=use_gemma,
        model_path=os.path.join(ROOT, c.get("model_path", "weights/diff_cnn.pt")),
        use_model=use_model,
        window_size=(ww, wh),
        gemma_use_image=gemma_img,
        tolerance_shift_px=sh,
        tolerance_speckle_iter=sp,
        pixel_threshold=px,
    )

    try:
        out = run_dual_pipeline(cfg, log=log)
    except Exception as e:
        return jsonify({"error": str(e), "logs": logs}), 500

    return jsonify(
        {
            "ok": out.ok,
            "report_txt": out.report_txt,
            "witness_dir": out.witness_dir,
            "shot_real": out.shot_real,
            "shot_local": out.shot_local,
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

    ap = argparse.ArgumentParser(
        description="Веб-панель: тестирование вёрстки сайта и сравнение с эталоном (макет/прод)"
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
