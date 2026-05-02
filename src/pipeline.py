from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from src.capture import capture_screenshot
from src.compare import CompareResult, compare_screenshots, diff_tensor_gray
from src.figma_client import export_frame_png, public_design_url
from src.gemma_client import explain_diff_ru
from src.model_net import load_classifier, predict_fail_prob
from src.report import append_text_report, write_html_report, write_json_sidecar


@dataclass
class RunConfig:
    url: str
    baseline_path: str
    screenshot_dir: str
    reports_dir: str
    diff_threshold_pct: float
    ollama_url: str
    gemma_model: str
    use_gemma: bool
    model_path: Optional[str]
    use_model: bool
    window_size: Tuple[int, int]
    gemma_use_image: bool
    tolerance_shift_px: int = 0
    tolerance_speckle_iter: int = 0
    pixel_threshold: int = 30
    baseline_is_figma: bool = False  # True = эталон из Figma (для промпта к VLM)
    figma_file_key: Optional[str] = None
    figma_node_id: Optional[str] = None


@dataclass
class RunOutcome:
    ok: bool
    current_shot: str
    compare: CompareResult
    model_prob_fail: Optional[float]
    gemma_text: str
    report_txt: str
    witness_dir: str
    report_html: Optional[str] = None


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
    _, layout_site = capture_screenshot(cfg.url, cur, window_size=cfg.window_size)
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
    device = torch.device("cpu")
    model, has_model = load_classifier(cfg.model_path if cfg.use_model else None, device)
    prob = None
    if has_model and model and cr.diff_path:
        x = diff_tensor_gray(cr.diff_path)
        prob = predict_fail_prob(model, x, device)
    ok = _verdict(cr, cfg.diff_threshold_pct, prob, has_model)
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
    }
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


@dataclass
class FigmaVsSiteConfig:
    """Скачать кадр из Figma, снять скрин сайта, сравнить и при необходимости вызвать VLM."""

    site_url: str
    figma_file_key: str
    figma_node_id: str
    figma_token: str
    figma_baseline_png: str
    figma_scale: int = 1
    screenshot_dir: str = "shots"
    reports_dir: str = "reports"
    diff_threshold_pct: float = 0.5
    ollama_url: str = "http://127.0.0.1:11434"
    gemma_model: str = "gemma3"
    use_gemma: bool = True
    model_path: Optional[str] = None
    use_model: bool = False
    window_size: Tuple[int, int] = (1920, 1080)
    gemma_use_image: bool = True
    tolerance_shift_px: int = 2
    tolerance_speckle_iter: int = 1
    pixel_threshold: int = 30


def run_figma_vs_site(
    cfg: FigmaVsSiteConfig,
    log: Optional[Callable[[str], None]] = None,
) -> RunOutcome:
    def L(s: str) -> None:
        if log:
            log(s)

    L("Шаг 1/2: загружаю кадр макета из Figma…")
    os.makedirs(os.path.dirname(cfg.figma_baseline_png) or ".", exist_ok=True)
    export_frame_png(
        cfg.figma_file_key,
        cfg.figma_node_id,
        cfg.figma_token,
        cfg.figma_baseline_png,
        scale=max(1, min(4, int(cfg.figma_scale))),
    )
    L(f"         макет сохранён: {cfg.figma_baseline_png}")
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
