"""Сопоставление фрагментов при блоке-помехе между ними (A + noise + A/B)."""

from src.fragment_match.config import FragmentMatchConfig
from src.fragment_match.models import build_matcher

__all__ = ["FragmentMatchConfig", "build_matcher"]
