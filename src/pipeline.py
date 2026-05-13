from __future__ import annotations

import os
import shutil
import time
from typing import Any, Callable, Dict, Optional, Tuple

from src.capture import capture_screenshot
from src.compare import CompareResult, compare_screenshots, diff_tensor_gray
from src.diff_hotspots import analyze_diff_for_qa, diff_hotspots_to_task_lines
from src.figma_client import export_frame_png, public_design_url
from src.gemma_client import explain_diff_ru
from src.pipeline_types import FigmaVsSiteConfig, RunConfig, RunOutcome
from src.report import append_text_report, write_html_report, write_json_sidecar


def _verdict(
    cr: CompareResult,
    threshold: float,
    model_prob: Optional[float],
    use_model: bool,
) -> bool:
    if cr.changed_ratio <= threshold / 100.0:
        if use_model and model_prob is not None and model_prob >= 0.5:
            return False
        return True
    return False


def run_pipeline(cfg: RunConfig) -> RunOutcome:
    os.makedirs(cfg.screenshot_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    cur = os.path.join(cfg.screenshot_dir, f"current_{ts}.png")
    _, layout_site = capture_screenshot(
        cfg.url,
        cur,
        window_size=cfg.window_size,
        wait_seconds=float(cfg.capture_wait_seconds),
    )
    diff_dir = os.path.join(cfg.screenshot_dir, "diffs")
    cr = compare_screenshots(
        cfg.baseline_path,
        cur,
        diff_dir,
        tag="diff",
        pixel_threshold=cfg.pixel_threshold,
        tolerance_shift_px=cfg.tolerance_shift_px,
        tolerance_speckle_iter=cfg.tolerance_speckle_iter,
    )
    prob = None
    has_model = False
    if cfg.use_model and cfg.model_path:
        try:
            import torch

            from src.model_net import load_classifier, predict_fail_prob

            device = torch.device("cpu")
            model, has_model = load_classifier(cfg.model_path, device)
            if has_model and model and cr.diff_path:
                x = diff_tensor_gray(cr.diff_path)
                prob = predict_fail_prob(model, x, device)
        except OSError as e:
            # WinError 1114 и т.п. при сломанном PyTorch / VC++ — пайплайн без CNN
            win = getattr(e, "winerror", None)
            if win == 1114 or "c10" in str(e).lower() or "dll" in str(e).lower():
                has_model = False
                prob = None
            else:
                raise
    ok = _verdict(cr, cfg.diff_threshold_pct, prob, has_model)
    raw_layout = dict(layout_site) if isinstance(layout_site, dict) else {}
    els_for_hotspots = raw_layout.get("elements") if isinstance(raw_layout.get("elements"), list) else []

    ls = dict(layout_site) if isinstance(layout_site, dict) else {}
    els = ls.get("elements")
    if isinstance(els, list) and len(els) > 48:
        ls = {**ls, "elements": els[:48], "elements_note": "обрезано до 48 блоков для промпта"}
    stats: Dict[str, Any] = {
        "url": cfg.url,
        "baseline": "figma_png" if cfg.baseline_is_figma else "image",
        "mse": round(cr.mse, 6),
        "changed_ratio_pct": round(cr.changed_ratio * 100, 3),
        "changed_ratio_raw_pct": round(cr.changed_ratio_raw * 100, 3),
        "changed_ratio_shift_pct": round(cr.changed_ratio_shift * 100, 3),
        "threshold_pct": cfg.diff_threshold_pct,
        "tolerance_shift_px": cr.tolerance_shift_px,
        "tolerance_speckle_iter": cr.tolerance_speckle_iter,
        "pixel_threshold": cfg.pixel_threshold,
        "size": [cr.width, cr.height],
        "model_prob_fail": prob,
        "layout_site": ls,
        "capture_wait_seconds": float(cfg.capture_wait_seconds),
    }
    hotspots = analyze_diff_for_qa(
        cfg.baseline_path,
        cur,
        els_for_hotspots,
        pixel_threshold=cfg.pixel_threshold,
        tolerance_shift_px=cfg.tolerance_shift_px,
        tolerance_speckle_iter=cfg.tolerance_speckle_iter,
    )
    stats["diff_hotspots"] = hotspots
    stats["diff_hotspots_tasks"] = diff_hotspots_to_task_lines(hotspots)
    gemma_text = ""
    if cfg.use_gemma:
        if cfg.baseline_is_figma:
            gctx = f"эталон — кадр макета из Figma (PNG {os.path.basename(cfg.baseline_path)}); под тестом страница: {cfg.url}"
        else:
            gctx = f"эталон (файл): {os.path.basename(cfg.baseline_path)}; страница: {cfg.url}"
        gemma_text = explain_diff_ru(
            cfg.ollama_url,
            cfg.gemma_model,
            stats,
            cr.diff_path,
            use_image=cfg.gemma_use_image,
            context_label=gctx,
        )
    lines = [
        f"URL: {cfg.url}",
        f"STATUS: {'PASS' if ok else 'FAIL'}",
        f"Baseline: {cfg.baseline_path}",
        f"Current: {cur}",
        f"Diff: {cr.diff_path}",
        f"MSE: {cr.mse:.6f}",
        f"Changed pixels (итог): {cr.changed_ratio * 100:.3f}%",
        f"Raw / shift: {cr.changed_ratio_raw * 100:.3f}% / {cr.changed_ratio_shift * 100:.3f}%",
        f"Допуск сдвиг px: {cr.tolerance_shift_px}, opening×3: {cr.tolerance_speckle_iter}",
    ]
    if prob is not None:
        lines.append(f"Model P(fail): {prob:.4f}")
    if gemma_text:
        lines.append("Gemma:")
        lines.append(gemma_text)
    if not ok:
        lines.append("Что не так: визуально заметное отличие от эталона (см. diff), либо модель/порог указали на риск регрессии.")
    else:
        lines.append("Эталон и текущий скрин совпали в пределах порога.")
    witness = os.path.join(cfg.reports_dir, f"witness_{ts}")
    os.makedirs(witness, exist_ok=True)
    for p in [cfg.baseline_path, cur, cr.diff_path or ""]:
        if p and os.path.isfile(p):
            shutil.copy2(p, witness)
    fig_url = (
        public_design_url(cfg.figma_file_key, cfg.figma_node_id)
        if (cfg.figma_file_key and cfg.figma_node_id)
        else "https://www.figma.com/"
    )
    shot_b = os.path.join(witness, os.path.basename(cfg.baseline_path))
    shot_c = os.path.join(witness, os.path.basename(cur))
    shot_d = os.path.join(witness, os.path.basename(cr.diff_path)) if cr.diff_path else ""
    html_path = write_html_report(
        cfg.reports_dir,
        site_url=cfg.url,
        figma_url=fig_url,
        ok=ok,
        stats=stats,
        gemma_markdown=gemma_text,
        baseline_path=shot_b if os.path.isfile(shot_b) else cfg.baseline_path,
        current_shot=shot_c if os.path.isfile(shot_c) else cur,
        diff_path=shot_d if (shot_d and os.path.isfile(shot_d)) else cr.diff_path,
    )
    lines.append(f"HTML-отчёт: {html_path}")
    report_path = append_text_report(cfg.reports_dir, lines)
    meta = {
        **stats,
        "ok": ok,
        "baseline": cfg.baseline_path,
        "current": cur,
        "diff": cr.diff_path,
        "gemma": gemma_text,
        "report_html": html_path,
        "figma_url": fig_url,
    }
    write_json_sidecar(report_path, meta)
    return RunOutcome(
        ok=ok,
        current_shot=cur,
        compare=cr,
        model_prob_fail=prob,
        gemma_text=gemma_text,
        report_txt=report_path,
        witness_dir=witness,
        report_html=html_path,
    )


def run_figma_vs_site(
    cfg: FigmaVsSiteConfig,
    log: Optional[Callable[[str], None]] = None,
) -> RunOutcome:
    def L(s: str) -> None:
        if log:
            log(s)

    os.makedirs(os.path.dirname(cfg.figma_baseline_png) or ".", exist_ok=True)
    force_figma = os.environ.get("FIGMA_FORCE_REFRESH", "").strip().lower() in ("1", "true", "yes")
    png_path = cfg.figma_baseline_png
    cached_ok = (
        cfg.figma_use_cached_png
        and not force_figma
        and os.path.isfile(png_path)
        and os.path.getsize(png_path) > 64
    )
    if cached_ok:
        L(
            "Шаг 1/2: кэш макета — беру уже скачанный PNG (без запроса к Figma API). "
            "Чтобы обновить из макета: FIGMA_FORCE_REFRESH=1 или figma.use_cached_png=false в config."
        )
        L(f"         файл: {png_path}")
    else:
        L("Шаг 1/2: загружаю кадр макета из Figma…")
        export_frame_png(
            cfg.figma_file_key,
            cfg.figma_node_id,
            cfg.figma_token,
            png_path,
            scale=max(1, min(4, int(cfg.figma_scale))),
            log=L if log else None,
        )
        L(f"         макет сохранён: {png_path}")
    L("Шаг 2/2: скриншот сайта и сравнение с макетом…")
    rc = RunConfig(
        url=cfg.site_url,
        baseline_path=cfg.figma_baseline_png,
        screenshot_dir=cfg.screenshot_dir,
        reports_dir=cfg.reports_dir,
        diff_threshold_pct=cfg.diff_threshold_pct,
        ollama_url=cfg.ollama_url,
        gemma_model=cfg.gemma_model,
        use_gemma=cfg.use_gemma,
        model_path=cfg.model_path,
        use_model=cfg.use_model,
        window_size=cfg.window_size,
        gemma_use_image=cfg.gemma_use_image,
        tolerance_shift_px=cfg.tolerance_shift_px,
        tolerance_speckle_iter=cfg.tolerance_speckle_iter,
        pixel_threshold=cfg.pixel_threshold,
        capture_wait_seconds=float(cfg.capture_wait_seconds),
        baseline_is_figma=True,
        figma_file_key=cfg.figma_file_key,
        figma_node_id=cfg.figma_node_id,
    )
    out = run_pipeline(rc)
    if log:
        log(f"Отчёт: {out.report_txt}")
        log(f"Артефакты: {out.witness_dir}")
        log("=== " + ("PASS" if out.ok else "FAIL") + " ===")
    return out
