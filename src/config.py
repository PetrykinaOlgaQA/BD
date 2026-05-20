from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from src.pipeline_types import FigmaVsSiteConfig


def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _clamp_float(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _pair_size(raw: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            w, h = int(raw[0]), int(raw[1])
            if w > 0 and h > 0:
                return (w, h)
        except (TypeError, ValueError):
            pass
    return default


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_cache_basename(file_key: str, node_id: str, scale: int) -> str:
    fk = _SAFE_NAME.sub("_", (file_key or "figma").strip())[:80]
    nid = _SAFE_NAME.sub("_", (node_id or "node").replace(":", "_").strip())[:40]
    return f"{fk}__{nid}__s{scale}.png"


@dataclass
class FigmaBlock:
    file_key: str
    node_id: str
    design_png: str
    scale: int = 1
    use_cached_png: bool = True
    figma_cache_dir: str = "shots/figma_cache"
    # Полная ссылка на фрейм (как в браузере) для веб-подсказки; пусто — собирается из file_key + node_id.
    frame_url: str = ""


@dataclass
class CompareBlock:
    """Параметры сравнения (часть полей задействуется в compare.py на следующих шагах)."""

    letterbox: bool = True
    ssim: bool = True
    enhanced_diff: bool = True
    diff_bbox_min_area: int = 400
    clahe_clip: float = 2.0
    blur_ksize: int = 3


@dataclass
class ComparatorBlock:
    enabled: bool = True
    weights: str = "weights/multi_aspect_comparator_best.pt"
    pass_threshold: float = 0.68
    max_regions: int = 10


@dataclass
class OllamaVisionBlock:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "gemma3:latest"
    image_max_side: int = 384
    max_retries: int = 2
    vision_fallback: bool = False
    try_generate_fallback: bool = False
    fallback_on_empty: bool = True
    timeout_connect: float = 60.0
    timeout_read: float = 300.0
    # Полировка текста багов (черновик из diff → Ollama → таблица отчёта)
    polish_bug_text: bool = True
    # text = только черновик; vision = картинка diff; both = vision затем полировка
    bug_report_mode: str = "text"


@dataclass
class AppConfig:
    url_site: str
    window_size: Tuple[int, int]
    figma: FigmaBlock
    tolerance_shift_px: int = 2
    tolerance_speckle_iter: int = 1
    pixel_threshold: int = 30
    diff_threshold_pct: float = 0.5
    capture_wait_seconds: float = 4.0
    screenshot_dir: str = "shots"
    reports_dir: str = "reports"
    model_path: str = "weights/diff_cnn.pt"
    fragment_matcher_path: str = "weights/fragment_matcher.pt"
    use_fragment_matcher: bool = True
    fragment_match_threshold: float = 0.55
    compare: CompareBlock = field(default_factory=CompareBlock)
    ollama: OllamaVisionBlock = field(default_factory=OllamaVisionBlock)
    comparator: ComparatorBlock = field(default_factory=ComparatorBlock)
    raw: Dict[str, Any] = field(default_factory=dict)

    def resolved_design_png(self, project_root: str) -> str:
        """Абсолютный путь к PNG макета; при пустом design_png — файл в figma_cache_dir."""
        root = project_root
        cache_dir = abs_if_rel(root, self.figma.figma_cache_dir)
        if (self.figma.design_png or "").strip():
            return abs_if_rel(root, self.figma.design_png)
        os.makedirs(cache_dir, exist_ok=True)
        name = _safe_cache_basename(self.figma.file_key, self.figma.node_id, self.figma.scale)
        return os.path.join(cache_dir, name)


def abs_if_rel(root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(root, path))


def load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_app_config(raw: Dict[str, Any], project_root: str) -> AppConfig:
    fg = raw.get("figma") or {}
    cmp = raw.get("compare") or {}
    ol = raw.get("ollama") or {}
    # Обратная совместимость: ollama_url / gemma_model на верхнем уровне
    ollama_url = str(ol.get("base_url") or raw.get("ollama_url") or "http://127.0.0.1:11434").strip()
    ollama_model = str(ol.get("model") or raw.get("gemma_model") or "gemma3:latest").strip()

    fb = FigmaBlock(
        file_key=str(fg.get("file_key", "")).strip(),
        node_id=str(fg.get("node_id", "")).strip(),
        design_png=str(fg.get("design_png", "")).strip(),
        scale=_clamp_int(fg.get("scale", 1), 1, 4, 1),
        use_cached_png=bool(fg.get("use_cached_png", True)),
        figma_cache_dir=str(fg.get("figma_cache_dir", "shots/figma_cache")).strip() or "shots/figma_cache",
        frame_url=str(fg.get("frame_url", "")).strip(),
    )

    cb = CompareBlock(
        letterbox=bool(cmp.get("letterbox", True)),
        ssim=bool(cmp.get("ssim", True)),
        enhanced_diff=bool(cmp.get("enhanced_diff", True)),
        diff_bbox_min_area=_clamp_int(cmp.get("diff_bbox_min_area", 400), 16, 500_000, 400),
        clahe_clip=_clamp_float(cmp.get("clahe_clip", 2.0), 0.1, 8.0, 2.0),
        blur_ksize=_clamp_int(cmp.get("blur_ksize", 3), 1, 31, 3) | 1,  # нечётное
    )

    cmp_nn = raw.get("comparator") or {}
    cnb = ComparatorBlock(
        enabled=bool(cmp_nn.get("enabled", True)),
        weights=str(cmp_nn.get("weights", "weights/multi_aspect_comparator_best.pt")).strip()
        or "weights/multi_aspect_comparator_best.pt",
        pass_threshold=_clamp_float(cmp_nn.get("pass_threshold", 0.68), 0.0, 1.0, 0.68),
        max_regions=_clamp_int(cmp_nn.get("max_regions", 10), 1, 32, 10),
    )

    ob = OllamaVisionBlock(
        base_url=ollama_url,
        model=ollama_model,
        image_max_side=_clamp_int(ol.get("image_max_side", 384), 256, 2048, 384),
        max_retries=_clamp_int(ol.get("max_retries", 2), 1, 10, 2),
        vision_fallback=bool(ol.get("vision_fallback", False)),
        try_generate_fallback=bool(ol.get("try_generate_fallback", False)),
        fallback_on_empty=bool(ol.get("fallback_on_empty", True)),
        timeout_connect=_clamp_float(ol.get("timeout_connect", 60.0), 5.0, 600.0, 60.0),
        timeout_read=_clamp_float(ol.get("timeout_read", 300.0), 30.0, 3600.0, 300.0),
        polish_bug_text=bool(ol.get("polish_bug_text", raw.get("refine_bug_text", True))),
        bug_report_mode=str(ol.get("bug_report_mode", "text")).strip().lower() or "text",
    )

    url = str(raw.get("url_site") or raw.get("url_local") or "").strip()
    frame_size = _pair_size(fg.get("frame_size"), (0, 0))
    win_default = frame_size if frame_size[0] > 0 and frame_size[1] > 0 else (1280, 720)
    window_size = _pair_size(raw.get("window_size"), win_default)
    cfg = AppConfig(
        url_site=url,
        window_size=window_size,
        figma=fb,
        tolerance_shift_px=_clamp_int(raw.get("tolerance_shift_px", 2), 0, 5, 2),
        tolerance_speckle_iter=_clamp_int(raw.get("tolerance_speckle_iter", 1), 0, 5, 1),
        pixel_threshold=_clamp_int(raw.get("pixel_threshold", 30), 0, 255, 30),
        diff_threshold_pct=_clamp_float(raw.get("diff_threshold_pct", 0.5), 0.0, 100.0, 0.5),
        capture_wait_seconds=_clamp_float(raw.get("capture_wait_seconds", 4.0), 0.0, 120.0, 4.0),
        screenshot_dir=str(raw.get("screenshot_dir", "shots")).strip() or "shots",
        reports_dir=str(raw.get("reports_dir", "reports")).strip() or "reports",
        model_path=str(raw.get("model_path", "weights/diff_cnn.pt")).strip() or "weights/diff_cnn.pt",
        fragment_matcher_path=str(
            raw.get("fragment_matcher_path", "weights/fragment_matcher.pt")
        ).strip()
        or "weights/fragment_matcher.pt",
        use_fragment_matcher=bool(raw.get("use_fragment_matcher", True)),
        fragment_match_threshold=_clamp_float(raw.get("fragment_match_threshold", 0.55), 0.0, 1.0, 0.55),
        compare=cb,
        ollama=ob,
        comparator=cnb,
        raw=dict(raw),
    )
    return cfg


def validate_app_config(cfg: AppConfig, site_url: Optional[str] = None) -> None:
    if not cfg.figma.file_key:
        raise ValueError("figma.file_key обязателен")
    if not cfg.figma.node_id:
        raise ValueError("figma.node_id обязателен")
    url = (site_url or cfg.url_site or "").strip()
    if not url:
        raise ValueError("url_site обязателен (или передайте --url)")


def load_app_config(path: str, project_root: str, *, site_url: Optional[str] = None) -> AppConfig:
    raw = load_json(path)
    cfg = parse_app_config(raw, project_root)
    validate_app_config(cfg, site_url=site_url)
    return cfg


def app_config_to_figma_vs_site(
    cfg: AppConfig,
    project_root: str,
    site_url: str,
    figma_token: str,
    figma_use_cached_png: bool,
    *,
    use_gemma: bool = True,
    use_model: bool = True,
    gemma_use_image: bool = True,
    use_fragment_matcher: Optional[bool] = None,
    use_comparator: Optional[bool] = None,
) -> FigmaVsSiteConfig:
    """Собирает существующий dataclass пайплайна из AppConfig."""
    out_png = cfg.resolved_design_png(project_root)
    w, h = cfg.window_size
    return FigmaVsSiteConfig(
        site_url=site_url,
        figma_file_key=cfg.figma.file_key,
        figma_node_id=cfg.figma.node_id,
        figma_token=figma_token,
        figma_baseline_png=out_png,
        figma_scale=cfg.figma.scale,
        figma_use_cached_png=figma_use_cached_png,
        capture_wait_seconds=float(cfg.capture_wait_seconds),
        screenshot_dir=abs_if_rel(project_root, cfg.screenshot_dir),
        reports_dir=abs_if_rel(project_root, cfg.reports_dir),
        diff_threshold_pct=float(cfg.diff_threshold_pct),
        ollama_url=cfg.ollama.base_url,
        gemma_model=cfg.ollama.model,
        use_gemma=use_gemma,
        model_path=abs_if_rel(project_root, cfg.model_path),
        use_model=use_model,
        window_size=(int(w), int(h)),
        gemma_use_image=gemma_use_image,
        tolerance_shift_px=int(cfg.tolerance_shift_px),
        tolerance_speckle_iter=int(cfg.tolerance_speckle_iter),
        pixel_threshold=int(cfg.pixel_threshold),
        ollama_timeout_connect=float(cfg.ollama.timeout_connect),
        ollama_timeout_read=float(cfg.ollama.timeout_read),
        ollama_image_max_side=int(cfg.ollama.image_max_side),
        ollama_max_retries=int(cfg.ollama.max_retries),
        ollama_vision_fallback=bool(cfg.ollama.vision_fallback),
        ollama_try_generate_fallback=bool(cfg.ollama.try_generate_fallback),
        ollama_fallback_on_empty=bool(cfg.ollama.fallback_on_empty),
        refine_bug_text=bool(cfg.ollama.polish_bug_text),
        ollama_polish_bugs=bool(cfg.ollama.polish_bug_text),
        ollama_bug_report_mode=str(cfg.ollama.bug_report_mode or "text"),
        fragment_matcher_path=abs_if_rel(project_root, cfg.fragment_matcher_path),
        use_fragment_matcher=(
            bool(cfg.use_fragment_matcher)
            if use_fragment_matcher is None
            else bool(use_fragment_matcher)
        ),
        fragment_match_threshold=float(cfg.fragment_match_threshold),
        use_comparator=(
            bool(cfg.comparator.enabled) if use_comparator is None else bool(use_comparator)
        ),
        comparator_weights_path=abs_if_rel(project_root, cfg.comparator.weights),
        comparator_pass_threshold=float(cfg.comparator.pass_threshold),
        comparator_max_regions=int(cfg.comparator.max_regions),
    )
