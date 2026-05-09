"""CLI: Figma + сайт → diff → CNN + (опционально) Llama Vision JSON."""

from __future__ import annotations

import argparse
import json
import sys

import torch
from loguru import logger

from config import ensure_runtime_dirs, get_settings
from src.diff_utils import create_diff_map
from src.figma_api import export_frame_to_pil
from src.model import load_model, numpy_gray_to_tensor, predict_diff
from src.ollama_vision import analyze_with_llama_vision
from src.selenium_capture import capture_url_to_pil, load_image_file
from src.utils import append_report_history


def main() -> int:
    s = ensure_runtime_dirs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="", help="URL страницы")
    ap.add_argument("--figma-file-key", default="")
    ap.add_argument("--figma-node-id", default="")
    ap.add_argument("--figma-png", default="", help="Локальный PNG вместо Figma")
    ap.add_argument("--site-png", default="", help="Локальный PNG вместо Selenium")
    ap.add_argument("--no-vision", action="store_true", help="Не вызывать Ollama")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loaded = load_model(s.cnn_weights_path, device=dev)
    if not loaded:
        logger.warning("Веса CNN не найдены: {} — вероятности условные.", s.cnn_weights_path)

    if args.figma_png:
        figma = load_image_file(args.figma_png)
    else:
        if not args.figma_file_key or not args.figma_node_id:
            print("Нужны --figma-file-key и --figma-node-id или --figma-png", file=sys.stderr)
            return 2
        tok = s.figma_token.strip()
        if not tok:
            print("Нет FIGMA_ACCESS_TOKEN / FVT_FIGMA_TOKEN", file=sys.stderr)
            return 2
        figma = export_frame_to_pil(
            args.figma_file_key,
            args.figma_node_id,
            tok,
            scale=s.figma_export_scale,
        )

    if args.site_png:
        site = load_image_file(args.site_png)
    else:
        if not args.url:
            print("Нужен --url или --site-png", file=sys.stderr)
            return 2
        site = capture_url_to_pil(
            args.url,
            (s.selenium_window_width, s.selenium_window_height),
            wait_seconds=s.selenium_wait_sec,
        )

    diff_res = create_diff_map(
        figma,
        site,
        target_size=(s.diff_target_size, s.diff_target_size),
        blur_ksize=s.diff_blur_ksize,
    )
    tensor = numpy_gray_to_tensor(diff_res.diff_gray_64)
    cnn = predict_diff(model, tensor, device=dev)
    verdict = "FAIL" if cnn["prob_fail"] >= s.cnn_fail_threshold else "PASS"

    vision_json = None
    raw_vision = ""
    if not args.no_vision:
        rep, raw_vision, err = analyze_with_llama_vision(
            diff_res.aligned_figma,
            diff_res.aligned_website,
            diff_res.diff_display,
            ollama_base_url=s.ollama_base_url,
            model=s.ollama_vision_model,
            timeout_sec=s.ollama_timeout_sec,
        )
        if rep is not None:
            vision_json = rep.model_dump()
        else:
            logger.warning("Vision: {}", err)

    record = {
        "verdict_cnn": verdict,
        "cnn": cnn,
        "vision": vision_json,
        "vision_raw_excerpt": (raw_vision or "")[:8000],
    }
    append_report_history(s.reports_dir, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
