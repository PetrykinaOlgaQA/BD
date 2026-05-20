from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from src.compare import CompareResult


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
    capture_wait_seconds: float = 4.0
    baseline_is_figma: bool = False
    figma_file_key: Optional[str] = None
    figma_node_id: Optional[str] = None
    ollama_timeout_connect: float = 60.0
    ollama_timeout_read: float = 300.0
    ollama_image_max_side: int = 384
    ollama_max_retries: int = 2
    ollama_vision_fallback: bool = False
    ollama_try_generate_fallback: bool = False
    ollama_fallback_on_empty: bool = True
    refine_bug_text: bool = True
    ollama_polish_bugs: bool = True
    ollama_bug_report_mode: str = "text"
    fragment_matcher_path: Optional[str] = None
    use_fragment_matcher: bool = False
    fragment_match_threshold: float = 0.55
    use_comparator: bool = True
    comparator_weights_path: str = "weights/multi_aspect_comparator_best.pt"
    comparator_pass_threshold: float = 0.68
    comparator_max_regions: int = 10


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


@dataclass
class FigmaVsSiteConfig:
    """Скачать кадр из Figma, снять скрин сайта, сравнить и при необходимости вызвать VLM."""

    site_url: str
    figma_file_key: str
    figma_node_id: str
    figma_token: str
    figma_baseline_png: str
    figma_scale: int = 1
    figma_use_cached_png: bool = True
    capture_wait_seconds: float = 4.0
    screenshot_dir: str = "shots"
    reports_dir: str = "reports"
    diff_threshold_pct: float = 0.5
    ollama_url: str = "http://127.0.0.1:11434"
    gemma_model: str = "gemma3:latest"
    use_gemma: bool = True
    model_path: Optional[str] = None
    use_model: bool = False
    window_size: Tuple[int, int] = (1280, 720)
    gemma_use_image: bool = True
    tolerance_shift_px: int = 2
    tolerance_speckle_iter: int = 1
    pixel_threshold: int = 30
    ollama_timeout_connect: float = 60.0
    ollama_timeout_read: float = 300.0
    ollama_image_max_side: int = 384
    ollama_max_retries: int = 2
    ollama_vision_fallback: bool = False
    ollama_try_generate_fallback: bool = False
    ollama_fallback_on_empty: bool = True
    refine_bug_text: bool = True
    ollama_polish_bugs: bool = True
    ollama_bug_report_mode: str = "text"
    fragment_matcher_path: Optional[str] = None
    use_fragment_matcher: bool = False
    fragment_match_threshold: float = 0.55
    use_comparator: bool = True
    comparator_weights_path: str = "weights/multi_aspect_comparator_best.pt"
    comparator_pass_threshold: float = 0.68
    comparator_max_regions: int = 10
