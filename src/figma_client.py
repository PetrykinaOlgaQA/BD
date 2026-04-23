from __future__ import annotations

import os
import time
from typing import Optional
from urllib.parse import quote

import requests


def export_frame_png(
    file_key: str,
    node_id: str,
    token: str,
    out_path: str,
    scale: int = 2,
    timeout: int = 120,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nid = quote(node_id, safe=":")
    api_url = f"https://api.figma.com/v1/images/{file_key}?ids={nid}&format=png&scale={scale}"
    headers = {"X-Figma-Token": token}
    r = requests.get(api_url, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    err = data.get("err")
    if err:
        raise RuntimeError(err)
    images = data.get("images") or {}
    img_url: Optional[str] = None
    for v in images.values():
        if v:
            img_url = v
            break
    if not img_url:
        raise RuntimeError("Figma не вернула URL изображения")
    for _ in range(12):
        ir = requests.get(img_url, timeout=timeout)
        if ir.status_code == 404:
            time.sleep(0.5)
            continue
        ir.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(ir.content)
        return out_path
    raise RuntimeError("Не удалось скачать PNG с временного URL")
