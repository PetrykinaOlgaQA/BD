from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests


def fetch_file_nodes_json(
    file_key: str,
    node_id: str,
    token: str,
    timeout: int = 120,
) -> Dict[str, Any]:
    """GET /v1/files/{key}/nodes — дерево узлов для указанных id (формат id: «19:2»)."""
    nid = quote(node_id, safe=":")
    url = f"https://api.figma.com/v1/files/{file_key}/nodes?ids={nid}"
    headers = {"X-Figma-Token": token}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def save_nodes_json(file_key: str, node_id: str, token: str, out_path: str) -> str:
    data = fetch_file_nodes_json(file_key, node_id, token)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


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
