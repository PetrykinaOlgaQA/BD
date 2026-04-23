from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


def append_text_report(
    reports_dir: str,
    lines: list[str],
    basename: str = "runs",
) -> str:
    os.makedirs(reports_dir, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = os.path.join(reports_dir, f"{basename}_{day}.txt")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    block = ["", "=" * 60, stamp, *lines, ""]
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(block))
    return path


def write_json_sidecar(path_txt: str, payload: Dict[str, Any]) -> str:
    path_json = os.path.splitext(path_txt)[0] + "_last.json"
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path_json
