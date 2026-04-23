from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple


def load_manifest(root: str, rel_path: str) -> Dict[str, Any]:
    p = rel_path if os.path.isabs(rel_path) else os.path.join(root, rel_path)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def screen_ids(manifest: Dict[str, Any]) -> list[str]:
    screens = manifest.get("screens") or {}
    return sorted(screens.keys())


def resolve_screen(root: str, manifest: Dict[str, Any], screen_id: str) -> Tuple[str, str]:
    screens = manifest.get("screens") or {}
    if screen_id not in screens:
        raise KeyError(screen_id)
    s = screens[screen_id]
    design = s.get("design_png") or s.get("baseline") or ""
    url = s.get("live_url") or s.get("url") or ""
    if not design or not url:
        raise ValueError("screen needs design_png and live_url")
    bp = design if os.path.isabs(design) else os.path.join(root, design)
    return os.path.normpath(bp), url.strip()
