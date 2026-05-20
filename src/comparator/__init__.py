"""
Упрощённый пайплайн сравнения Figma vs сайт (MultiAspectComparator).

Пакет `comparator` — одна обучаемая модель + правила DOM/OCR.
Старый код в `src/bug_reports.py`, `fragment_match/` на этой ветке не используется.
"""

from src.comparator.models.multi_aspect import (
    ASPECT_KEYS,
    MultiAspectComparator,
    build_four_channel_input,
)

__all__ = [
    "ASPECT_KEYS",
    "MultiAspectComparator",
    "build_four_channel_input",
]
