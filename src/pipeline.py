from __future__ import annotations

import os
import shutil
import time
from typing import Any, Callable, Dict, Optional, Tuple

from src.capture import capture_screenshot
from src.compare import CompareResult, compare_screenshots
from src.bug_reports import (
    build_bug_report_items,
    bug_items_from_polished_lines,
    draft_lines_to_text,
    has_structured_section_bugs,
    merge_polished_text_into_items,
    sanitize_bug_lines,
)
from src.diff_hotspots import analyze_diff_for_qa
from src.figma_client import FigmaRateLimitError, export_frame_png, public_design_url
from src.gemma_client import explain_diff_ru, refine_bug_lines_ru
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


def run_pipeline(cfg: RunConfig, log: Optional[Callable[[str], None]] = None) -> RunOutcome:
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
    if cfg.use_model and cfg.model_path and cr.diff_path:
        from src.diff_classifier import load_classifier, predict_fail_prob

        handle, has_model = load_classifier(cfg.model_path)
        if has_model and handle:
            try:
                prob = predict_fail_prob(handle, cr.diff_path)
            except OSError as e:
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
    bug_items = build_bug_report_items(
        hotspots,
        els_for_hotspots,
        baseline_path=cfg.baseline_path,
        current_path=cur,
        pixel_threshold=cfg.pixel_threshold,
        tolerance_shift_px=cfg.tolerance_shift_px,
        tolerance_speckle_iter=cfg.tolerance_speckle_iter,
        stats_sink=stats,
    )
    bug_items_pre_fragment = [dict(x) for x in bug_items if isinstance(x, dict)]
    stats["bug_report_items_pre_fragment"] = bug_items_pre_fragment
    stats.setdefault("structural_shift_filtered", 0)
    frag_meta: Dict[str, Any] = {"fragment_matcher_used": False}
    try:
        from src.bug_consolidate import finalize_bug_report_items

        bug_items = finalize_bug_report_items(
            bug_items, layout_elements=els_for_hotspots
        )
    except Exception:
        pass
    if cfg.use_fragment_matcher and bug_items:
        try:
            from src.fragment_match.filter_bugs import apply_fragment_matcher_to_bug_items
            from src.fragment_match.inference import load_fragment_matcher

            fhandle = None
            if cfg.fragment_matcher_path:
                fhandle, _ = load_fragment_matcher(cfg.fragment_matcher_path)
            vp = raw_layout.get("viewport") if isinstance(raw_layout, dict) else {}
            try:
                vw = int(vp.get("w", cfg.window_size[0]))
                vh = int(vp.get("h", cfg.window_size[1]))
            except (TypeError, ValueError, AttributeError):
                vw, vh = cfg.window_size[0], cfg.window_size[1]
            bug_items, frag_meta = apply_fragment_matcher_to_bug_items(
                bug_items,
                baseline_path=cfg.baseline_path,
                current_path=cur,
                diff_path=cr.diff_path,
                matcher=fhandle,
                same_threshold=float(cfg.fragment_match_threshold),
                layout_elements=els_for_hotspots,
                viewport=(vw, vh),
            )
            frag_meta["fragment_matcher_used"] = True
            if log:
                log(
                    f"Fragment matcher: оценено {frag_meta.get('fragment_match_scored', 0)}, "
                    f"отфильтровано ложных {frag_meta.get('fragment_match_filtered', 0)} "
                    f"({frag_meta.get('fragment_match_metric', '')})"
                )
        except OSError as e:
            if log:
                log(f"Fragment matcher: пропуск ({e})")
    sf = int(stats.get("structural_shift_filtered", 0) or 0)
    if sf and log:
        log(
            f"Сдвиг вёрстки: убрано ложных «фрагмента нет на макете/странице» — {sf} "
            f"(вставка блока между одинаковыми секциями)"
        )
    if not bug_items and bug_items_pre_fragment:
        bug_items = bug_items_pre_fragment
        frag_meta["fragment_match_fallback_kept_all"] = True
        if log:
            log(
                "Fragment matcher: после фильтра не осталось пунктов — "
                "оставлен исходный список diff (для Ollama и таблицы)."
            )
    stats.update(frag_meta)
    if cfg.use_comparator:
        try:
            from src.comparator.pipeline_integration import augment_bug_items_with_comparator

            project_root = os.path.dirname(os.path.abspath(cfg.reports_dir))
            crops_dir = os.path.join(cfg.screenshot_dir, "comparator_crops", str(ts))
            bug_items = augment_bug_items_with_comparator(
                bug_items,
                baseline_path=cfg.baseline_path,
                current_path=cur,
                hotspots=hotspots,
                layout_elements=els_for_hotspots,
                project_root=project_root,
                weights_path=str(cfg.comparator_weights_path),
                pass_threshold=float(cfg.comparator_pass_threshold),
                max_regions=int(cfg.comparator_max_regions),
                crops_dir=crops_dir,
                stats_sink=stats,
                log=log,
            )
        except Exception as e:
            stats["comparator_error"] = str(e)
            if log:
                log(f"Comparator: пропуск ({e})")
    try:
        from src.bug_consolidate import finalize_bug_report_items

        bug_items = finalize_bug_report_items(
            bug_items, layout_elements=els_for_hotspots
        )
    except Exception:
        pass
    stats["bug_report_items"] = bug_items
    stats["diff_hotspots_tasks"] = [
        str(it.get("text", "")).strip()
        for it in bug_items
        if isinstance(it, dict) and str(it.get("text", "")).strip()
    ]
    stats["layout_elements_for_crops"] = (
        els_for_hotspots[:120] if isinstance(els_for_hotspots, list) else []
    )
    draft_lines = [
        "- " + str(x).strip().lstrip("-• ").strip()
        for x in (stats.get("diff_hotspots_tasks") or [])
        if str(x).strip()
    ]

    gemma_text = draft_lines_to_text(draft_lines) if draft_lines else ""
    polish = bool(getattr(cfg, "ollama_polish_bugs", getattr(cfg, "refine_bug_text", True)))
    bug_mode = str(getattr(cfg, "ollama_bug_report_mode", "text") or "text").lower()
    structured_bugs = has_structured_section_bugs(bug_items)
    if cfg.use_gemma and polish:
        if cfg.baseline_is_figma:
            gctx = f"эталон — кадр макета из Figma (PNG {os.path.basename(cfg.baseline_path)}); под тестом страница: {cfg.url}"
        else:
            gctx = f"эталон (файл): {os.path.basename(cfg.baseline_path)}; страница: {cfg.url}"
        ollama_kw = dict(
            ollama_timeout=(float(cfg.ollama_timeout_connect), float(cfg.ollama_timeout_read)),
            max_post_retries=int(cfg.ollama_max_retries),
        )
        vision_txt = ""
        if bug_mode in ("vision", "both") and cfg.gemma_use_image and cr.diff_path:
            if log:
                log(
                    f"Ollama ({cfg.gemma_model}): разбор diff по картинке "
                    f"(read<={int(cfg.ollama_timeout_read)} s)..."
                )
            vision_txt = explain_diff_ru(
                cfg.ollama_url,
                cfg.gemma_model,
                stats,
                cr.diff_path,
                use_image=True,
                context_label=gctx,
                vision_fallback=bool(cfg.ollama_vision_fallback),
                try_generate_fallback=bool(cfg.ollama_try_generate_fallback),
                fallback_on_empty=bool(cfg.ollama_fallback_on_empty),
                image_max_side=int(cfg.ollama_image_max_side),
                **ollama_kw,
            )
        refine_in = draft_lines
        if bug_mode == "both" and vision_txt.strip():
            refine_in = [
                ln.strip().lstrip("-• ").strip()
                for ln in vision_txt.splitlines()
                if ln.strip()
            ] or draft_lines
        refined_raw = vision_txt
        if bug_mode in ("text", "both") and refine_in:
            if log:
                log(
                    f"Ollama ({cfg.gemma_model}): полировка баг-репорта "
                    f"(read<={int(cfg.ollama_timeout_read)} s)..."
                )
            refine_draft = [
                (s if str(s).strip().startswith("-") else "- " + str(s).strip())
                for s in refine_in
                if str(s).strip()
            ]
            refined_raw = refine_bug_lines_ru(
                cfg.ollama_url,
                cfg.gemma_model,
                refine_draft,
                stats,
                context_label=gctx,
                **ollama_kw,
            )
        refined_lines = sanitize_bug_lines(
            [ln.strip().lstrip("-• ").strip() for ln in (refined_raw or "").splitlines() if ln.strip()]
        )
        min_ok = max(1, len(draft_lines) // 3) if draft_lines else 1
        if refined_lines and len(refined_lines) >= min_ok:
            gemma_text = draft_lines_to_text(["- " + s for s in refined_lines])
            preserved = list(bug_items)
            if structured_bugs:
                merged = merge_polished_text_into_items(preserved, refined_lines)
                if merged:
                    bug_items = merged
                    stats["ollama_bug_polish_mode"] = "structured_preserve"
                else:
                    stats["ollama_bug_polish_mode"] = "structured_keep_draft"
                    stats["ollama_bug_polish_skipped_replace"] = True
            else:
                bug_items = bug_items_from_polished_lines(
                    refined_lines,
                    els_for_hotspots,
                    bug_items_pre_fragment or bug_items,
                    baseline_path=cfg.baseline_path,
                    current_path=cur,
                )
                stats["ollama_bug_polish_mode"] = "full_replace"
            if not bug_items and preserved:
                bug_items = preserved
                stats["ollama_bug_polish_fallback"] = "kept_heuristic_items"
            try:
                from src.bug_consolidate import finalize_bug_report_items

                bug_items = finalize_bug_report_items(
                    bug_items, layout_elements=els_for_hotspots
                )
            except Exception:
                pass
            from src.structural_shift import filter_items_with_paths

            n_before = len(bug_items)
            bug_items, _ = filter_items_with_paths(
                bug_items,
                cfg.baseline_path,
                cur,
                els_for_hotspots,
            )
            stats["structural_shift_filtered"] = int(stats.get("structural_shift_filtered", 0)) + max(
                0, n_before - len(bug_items)
            )
            if not bug_items and preserved:
                bug_items = preserved
            stats["bug_report_items"] = bug_items
            stats["diff_hotspots_tasks"] = [
                str(it.get("text", "")).strip() for it in bug_items if str(it.get("text", "")).strip()
            ]
            gemma_text = draft_lines_to_text(
                ["- " + str(it.get("text", "")).strip() for it in bug_items if str(it.get("text", "")).strip()]
            )
            stats["ollama_bug_polish"] = True
            stats["ollama_bug_lines"] = list(refined_lines)
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


def _align_baseline_png_to_window(
    png_path: str,
    window_size: Tuple[int, int],
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """Подгоняет PNG макета под window_size (1:1 с фреймом Figma)."""
    try:
        from PIL import Image
    except ImportError:
        return
    if not os.path.isfile(png_path):
        return
    try:
        im = Image.open(png_path).convert("RGB")
    except OSError:
        return
    tw, th = int(window_size[0]), int(window_size[1])
    if tw < 320 or th < 200:
        return
    if im.size == (tw, th):
        return
    im = im.resize((tw, th), Image.Resampling.LANCZOS)
    im.save(png_path, format="PNG", optimize=True)
    if log:
        log(f"         макет приведён к размеру окна {tw}×{th} px (как фрейм Figma)")


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
        _align_baseline_png_to_window(png_path, cfg.window_size, log=L)
    else:
        L("Шаг 1/2: загружаю кадр макета из Figma (свежий экспорт, без кэша)…")
        try:
            export_frame_png(
                cfg.figma_file_key,
                cfg.figma_node_id,
                cfg.figma_token,
                png_path,
                scale=max(1, min(4, int(cfg.figma_scale))),
                log=L if log else None,
            )
            L(f"         макет сохранён: {png_path}")
            _align_baseline_png_to_window(png_path, cfg.window_size, log=L)
        except FigmaRateLimitError:
            if os.path.isfile(png_path) and os.path.getsize(png_path) > 64:
                L(
                    "         Figma API: лимит 429 — беру ранее сохранённый PNG (кэш), "
                    "сверка продолжится. Обновите макет позже без галочки «Заново выгрузить»."
                )
                L(f"         файл: {png_path}")
            else:
                raise
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
        ollama_timeout_connect=float(cfg.ollama_timeout_connect),
        ollama_timeout_read=float(cfg.ollama_timeout_read),
        ollama_image_max_side=int(cfg.ollama_image_max_side),
        ollama_max_retries=int(cfg.ollama_max_retries),
        ollama_vision_fallback=bool(cfg.ollama_vision_fallback),
        ollama_try_generate_fallback=bool(cfg.ollama_try_generate_fallback),
        ollama_fallback_on_empty=bool(cfg.ollama_fallback_on_empty),
        refine_bug_text=bool(getattr(cfg, "refine_bug_text", True)),
        ollama_polish_bugs=bool(getattr(cfg, "ollama_polish_bugs", True)),
        ollama_bug_report_mode=str(getattr(cfg, "ollama_bug_report_mode", "text")),
        fragment_matcher_path=getattr(cfg, "fragment_matcher_path", None),
        use_fragment_matcher=bool(getattr(cfg, "use_fragment_matcher", False)),
        fragment_match_threshold=float(getattr(cfg, "fragment_match_threshold", 0.55)),
        use_comparator=bool(getattr(cfg, "use_comparator", True)),
        comparator_weights_path=str(
            getattr(cfg, "comparator_weights_path", "weights/multi_aspect_comparator_best.pt")
        ),
        comparator_pass_threshold=float(getattr(cfg, "comparator_pass_threshold", 0.68)),
        comparator_max_regions=int(getattr(cfg, "comparator_max_regions", 10)),
    )
    out = run_pipeline(rc, log=log)
    if log:
        log(f"Отчёт: {out.report_txt}")
        log(f"Артефакты: {out.witness_dir}")
        log("=== " + ("PASS" if out.ok else "FAIL") + " ===")
    return out
