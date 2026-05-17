from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, jsonify, render_template, request

from src.figma_client import parse_figma_frame_url, public_design_url
from src.pipeline import run_figma_vs_site
from src.pipeline_types import FigmaVsSiteConfig

app = Flask(__name__, template_folder=os.path.join(ROOT, "templates"), static_folder=os.path.join(ROOT, "static"))

_RUN_JOBS: Dict[str, Dict[str, Any]] = {}
_RUN_JOBS_LOCK = threading.Lock()
API_BUILD = "compact-bug-table-vt-train"


def _outcome_to_response_dict(out: Any, logs: List[str]) -> Dict[str, Any]:
    """Тот же JSON, что раньше отдавал синхронный POST /api/run."""
    return {
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


def _run_pipeline_job(job_id: str, fcfg: FigmaVsSiteConfig) -> None:
    logs: List[str] = []

    def log(msg: str) -> None:
        logs.append(msg)
        with _RUN_JOBS_LOCK:
            st = _RUN_JOBS.get(job_id)
            if st is not None and st.get("status") == "running":
                st["logs"] = list(logs)

    try:
        out = run_figma_vs_site(fcfg, log=log)
        payload = _outcome_to_response_dict(out, logs)
        with _RUN_JOBS_LOCK:
            _RUN_JOBS[job_id] = {"status": "done", **payload}
    except Exception as e:
        with _RUN_JOBS_LOCK:
            _RUN_JOBS[job_id] = {"status": "error", "error": str(e), "logs": logs}


def _load_cfg() -> Dict[str, Any]:
    p = os.path.join(ROOT, "config.json")
    if not os.path.isfile(p):
        p = os.path.join(ROOT, "config.example.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/ping")
def api_ping():
    """Лёгкая проверка: страница и API на одном origin; без токенов и тяжёлых операций."""
    return jsonify({"ok": True, "service": "figma-vs-site", "build": API_BUILD})


@app.get("/api/config")
def api_config():
    c = _load_cfg()
    w, h = tuple(c.get("window_size", [1280, 720]))
    fg = c.get("figma") or {}
    ol = c.get("ollama") or {}
    fk = (fg.get("file_key") or "").strip()
    nid = (fg.get("node_id") or "").strip()
    figma_url_hint = (fg.get("frame_url") or "").strip()
    if not figma_url_hint and fk and nid:
        figma_url_hint = public_design_url(fk, nid)
    return jsonify(
        {
            "url_site": c.get("url_site", c.get("url_local", "")),
            "figma_file_key": fk,
            "figma_node_id": nid,
            "figma_url_hint": figma_url_hint,
            "figma_use_cached_png": bool(fg.get("use_cached_png", True)),
            "window_w": int(w),
            "window_h": int(h),
            "diff_threshold_pct": float(c.get("diff_threshold_pct", 0.5)),
            "tolerance_shift_px": int(c.get("tolerance_shift_px", 2)),
            "tolerance_speckle_iter": int(c.get("tolerance_speckle_iter", 1)),
            "pixel_threshold": int(c.get("pixel_threshold", 30)),
            "ollama_url": ol.get("base_url") or c.get("ollama_url", "http://127.0.0.1:11434"),
            "gemma_model": ol.get("model") or c.get("gemma_model", "gemma3:latest"),
            "ollama_timeout_connect": float(ol.get("timeout_connect", 60)),
            "ollama_timeout_read": float(ol.get("timeout_read", 300)),
            "ollama_model": ol.get("model") or c.get("gemma_model", "gemma3:latest"),
            "figma_scale": int(fg.get("scale", 1)),
            "capture_wait_seconds": float(c.get("capture_wait_seconds", 4)),
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

    figma_url_in = (body.get("figma_url") or "").strip()
    if figma_url_in:
        fk_u, nid_u = parse_figma_frame_url(figma_url_in)
        if not fk_u:
            return jsonify(
                {
                    "error": "Ссылка не похожа на Figma (ожидается …figma.com/design/FILE_KEY/… или /file/…).",
                }
            ), 400
        fk = fk_u
        if nid_u:
            nid = nid_u
        else:
            nid_fb = (body.get("figma_node_id") or fg.get("node_id") or "").strip()
            if nid_fb:
                nid = nid_fb
        if not nid:
            return jsonify(
                {
                    "error": "В ссылке Figma нет node-id=…. Открой нужный фрейм в файле и скопируй URL "
                    "с параметром node-id, либо укажите node вручную в блоке «Дополнительно».",
                }
            ), 400

    if not fk or not nid or not site:
        return jsonify({"error": "Нужны URL сайта и макет Figma (ссылка с node-id или пара file key + node id в config)"}), 400

    design_rel = fg.get("design_png", "storage/designs/figma_baseline_last.png")
    out_png = design_rel if os.path.isabs(design_rel) else os.path.normpath(os.path.join(ROOT, design_rel))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    use_gemma = bool(body.get("use_gemma", True))
    use_model = bool(body.get("use_model", True))
    gemma_img = bool(body.get("gemma_use_image", True))
    figma_use_cached = bool(fg.get("use_cached_png", True))
    if body.get("figma_refresh") or body.get("figma_force_refresh"):
        figma_use_cached = False
    if "figma_use_cached_png" in body:
        figma_use_cached = bool(body.get("figma_use_cached_png"))

    ol = c.get("ollama") or {}

    def _fclamp(name: str, default: float, lo: float, hi: float) -> float:
        try:
            v = float(ol.get(name, default))
        except (TypeError, ValueError):
            v = default
        return max(lo, min(hi, v))

    ollama_tconn = _fclamp("timeout_connect", 60.0, 5.0, 600.0)
    ollama_tread = _fclamp("timeout_read", 300.0, 30.0, 3600.0)
    ollama_img_side = max(256, min(2048, int(ol.get("image_max_side", 384))))
    ollama_max_retries = max(1, min(10, int(ol.get("max_retries", 2))))
    ollama_vision_fb = bool(ol.get("vision_fallback", False))
    ollama_gen_fb = bool(ol.get("try_generate_fallback", False))
    ollama_empty_fb = bool(ol.get("fallback_on_empty", True))
    gemma_model = body.get("gemma_model") or ol.get("model") or c.get("gemma_model", "gemma3:latest")
    if "moondream" in str(gemma_model).lower():
        ollama_empty_fb = True

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
        ollama_url=(body.get("ollama_url") or ol.get("base_url") or c.get("ollama_url", "http://127.0.0.1:11434")).rstrip("/"),
        gemma_model=gemma_model,
        use_gemma=use_gemma,
        model_path=os.path.join(ROOT, c.get("model_path", "weights/diff_cnn.pt")),
        use_model=use_model,
        window_size=(ww, wh),
        gemma_use_image=gemma_img,
        tolerance_shift_px=sh,
        tolerance_speckle_iter=sp,
        pixel_threshold=px,
        capture_wait_seconds=cap_wait,
        ollama_timeout_connect=ollama_tconn,
        ollama_timeout_read=ollama_tread,
        ollama_image_max_side=ollama_img_side,
        ollama_max_retries=ollama_max_retries,
        ollama_vision_fallback=ollama_vision_fb,
        ollama_try_generate_fallback=ollama_gen_fb,
        ollama_fallback_on_empty=ollama_empty_fb,
    )

    job_id = uuid.uuid4().hex[:20]
    with _RUN_JOBS_LOCK:
        _RUN_JOBS[job_id] = {"status": "running", "logs": []}
    threading.Thread(target=_run_pipeline_job, args=(job_id, fcfg), daemon=True).start()
    return (
        jsonify(
            {
                "async": True,
                "job_id": job_id,
                "poll_path": f"/api/job/{job_id}",
                "message": "Сверка запущена в фоне; интерфейс опрашивает статус (долгие прогоны не рвут соединение).",
            }
        ),
        202,
    )


@app.get("/api/job/<job_id>")
def api_run_job_status(job_id: str):
    with _RUN_JOBS_LOCK:
        st = _RUN_JOBS.get(job_id)
    if not st:
        return jsonify({"error": "задача не найдена (устарела или неверный id)"}), 404
    return jsonify(st)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Веб-панель: сайт vs макет Figma")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    listen = args.host
    open_host = "127.0.0.1" if listen in ("0.0.0.0", "::", "[::]") else listen
    print(f"Панель: http://{open_host}:{args.port}/  |  проверка: http://{open_host}:{args.port}/api/ping")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
