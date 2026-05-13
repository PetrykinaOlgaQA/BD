from __future__ import annotations

"""
Vision-отчёты через локальную Ollama.

Слой совместимости с ТЗ «ollama_vision.py»: основная реализация запросов и промптов
находится в `gemma_client.py` (ретраи, resize, русский промпт). Здесь — явный фасад
для пайплайна и будущего переноса логики без смены импортов.
"""

from typing import Any, Dict, Optional

from src.gemma_client import explain_diff_ru

__all__ = ["explain_diff_ru", "explain_visual_regression_ru"]


def explain_visual_regression_ru(
    base_url: str,
    model: str,
    stats: Dict[str, Any],
    diff_image_path: Optional[str],
    *,
    use_image: bool = True,
    context_label: str = "",
) -> str:
    """Текстовый баг-репорт на русском по метрикам и (опционально) изображению diff."""
    return explain_diff_ru(
        base_url,
        model,
        stats,
        diff_image_path,
        use_image=use_image,
        context_label=context_label,
    )
