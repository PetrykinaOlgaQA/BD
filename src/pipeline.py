from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from src.capture import capture_screenshot
from src.compare import CompareResult, compare_screenshots, diff_tensor_gray
from src.gemma_client import explain_diff_ru
from src.model_net import load_classifier, predict_fail_prob
from src.report import append_text_report, write_json_sidecar


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


@dataclass
class RunOutcome:
    ok: bool
    current_shot: str
    compare: CompareResult
    model_prob_fail: Optional[float]
    gemma_text: str
    report_txt: str
    witness_dir: str


@dataclass
class DualRunConfig:
    url_real: str
    url_local: str
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
    tolerance_shift_px: int = 2
    tolerance_speckle_iter: int = 1
    pixel_threshold: int = 30


@dataclass
class DualRunOutcome:
    ok: bool
    shot_real: str
    shot_local: str
    compare: CompareResult
    model_prob_fail: Optional[float]
    gemma_text: str
    report_txt: str
    witness_dir: str


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
    capture_screenshot(cfg.url, cur, window_size=cfg.window_size)
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
    stats: Dict[str, Any] = {
        "url": cfg.url,
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
    }
    gemma_text = ""
    if cfg.use_gemma:
        gemma_text = explain_diff_ru(
            cfg.ollama_url,
            cfg.gemma_model,
            stats,
            cr.diff_path,
            use_image=cfg.gemma_use_image,
            context_label=f"эталон (файл): {os.path.basename(cfg.baseline_path)}; проверяемая страница: {cfg.url}",
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
    report_path = append_text_report(cfg.reports_dir, lines)
    witness = os.path.join(cfg.reports_dir, f"witness_{ts}")
    os.makedirs(witness, exist_ok=True)
    for p in [cfg.baseline_path, cur, cr.diff_path or ""]:
        if p and os.path.isfile(p):
            shutil.copy2(p, witness)
    meta = {**stats, "ok": ok, "baseline": cfg.baseline_path, "current": cur, "diff": cr.diff_path, "gemma": gemma_text}
    write_json_sidecar(report_path, meta)
    return RunOutcome(
        ok=ok,
        current_shot=cur,
        compare=cr,
        model_prob_fail=prob,
        gemma_text=gemma_text,
        report_txt=report_path,
        witness_dir=witness,
    )


def run_dual_pipeline(
    cfg: DualRunConfig,
    log: Optional[Callable[[str], None]] = None,
) -> DualRunOutcome:
    def L(s: str) -> None:
        if log:
            log(s)

    os.makedirs(cfg.screenshot_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    shot_real = os.path.join(cfg.screenshot_dir, f"real_{ts}.png")
    shot_local = os.path.join(cfg.screenshot_dir, f"local_{ts}.png")
    L("Шаг 1/3: эталон (прод/стенд) — скриншот…")
    capture_screenshot(cfg.url_real, shot_real, window_size=cfg.window_size)
    L("         скрин эталона сохранён.")
    L("Шаг 2/3: тестируемая вёрстка — скриншот…")
    capture_screenshot(cfg.url_local, shot_local, window_size=cfg.window_size)
    L("         скрин тестируемого сайта сохранён.")
    L("Шаг 3/3: сравнение эталон vs тест (diff)…")
    diff_dir = os.path.join(cfg.screenshot_dir, "diffs")
    cr = compare_screenshots(
        shot_real,
        shot_local,
        diff_dir,
        tag="dual",
        pixel_threshold=cfg.pixel_threshold,
        tolerance_shift_px=cfg.tolerance_shift_px,
        tolerance_speckle_iter=cfg.tolerance_speckle_iter,
    )
    L("         сравнение готово.")
    device = torch.device("cpu")
    model, has_model = load_classifier(cfg.model_path if cfg.use_model else None, device)
    prob = None
    if has_model and model and cr.diff_path:
        x = diff_tensor_gray(cr.diff_path)
        prob = predict_fail_prob(model, x, device)
    ok = _verdict(cr, cfg.diff_threshold_pct, prob, has_model)
    stats: Dict[str, Any] = {
        "url_real": cfg.url_real,
        "url_local": cfg.url_local,
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
    }
    gemma_text = ""
    if cfg.use_gemma:
        L("         опционально: запрос к Gemma…")
        gemma_text = explain_diff_ru(
            cfg.ollama_url,
            cfg.gemma_model,
            stats,
            cr.diff_path,
            use_image=cfg.gemma_use_image,
            context_label="тест вёрстки: эталон (прод) слева, справа — проверяемый сайт",
        )
        L("         Gemma ответила." if gemma_text and not gemma_text.startswith("[") else "         Gemma пропущена или ошибка.")
    lines = [
        f"Эталон: {cfg.url_real}",
        f"Сайт под тестом: {cfg.url_local}",
        f"STATUS: {'PASS' if ok else 'FAIL'}",
        f"Скрин эталона: {shot_real}",
        f"Скрин тестируемой страницы: {shot_local}",
        f"Diff: {cr.diff_path}",
        f"MSE: {cr.mse:.6f}",
        f"Пиксели (итог): {cr.changed_ratio * 100:.3f}%",
        f"Raw / shift: {cr.changed_ratio_raw * 100:.3f}% / {cr.changed_ratio_shift * 100:.3f}%",
        f"Сдвиг px: {cr.tolerance_shift_px}, opening: {cr.tolerance_speckle_iter}",
    ]
    if prob is not None:
        lines.append(f"Model P(fail): {prob:.4f}")
    if gemma_text:
        lines.append("Gemma:")
        lines.append(gemma_text)
    if not ok:
        lines.append("Итог: вёрстка расходится с эталоном сильнее допустимого порога.")
    else:
        lines.append("Итог: в пределах порога — как на эталоне.")
    report_path = append_text_report(cfg.reports_dir, lines)
    witness = os.path.join(cfg.reports_dir, f"witness_dual_{ts}")
    os.makedirs(witness, exist_ok=True)
    for p in [shot_real, shot_local, cr.diff_path or ""]:
        if p and os.path.isfile(p):
            shutil.copy2(p, witness)
    meta = {**stats, "ok": ok, "shot_real": shot_real, "shot_local": shot_local, "diff": cr.diff_path, "gemma": gemma_text}
    write_json_sidecar(report_path, meta)
    L(f"Отчёт: {report_path}")
    L(f"Артефакты: {witness}")
    L("=== " + ("PASS" if ok else "FAIL") + " ===")
    return DualRunOutcome(
        ok=ok,
        shot_real=shot_real,
        shot_local=shot_local,
        compare=cr,
        model_prob_fail=prob,
        gemma_text=gemma_text,
        report_txt=report_path,
        witness_dir=witness,
    )
