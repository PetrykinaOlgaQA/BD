from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional

import requests


def ollama_generate(
    base_url: str,
    model: str,
    prompt: str,
    images_b64: Optional[list[str]] = None,
    timeout: int = 120,
) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if images_b64:
        payload["images"] = images_b64
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def explain_diff_ru(
    base_url: str,
    model: str,
    stats: Dict[str, Any],
    diff_image_path: str | None,
    use_image: bool = True,
) -> str:
    lines = [
        "Ты помощник по UI-регрессии. Ответь по-русски, 3–6 коротких предложений.",
        "Метрики сравнения скриншотов:",
        json.dumps(stats, ensure_ascii=False, indent=2),
        "Сформулируй, что могло пойти не так (сдвиг вёрстки, шрифты, цвета, динамический контент, окно браузера).",
    ]
    prompt = "\n".join(lines)
    imgs = None
    if use_image and diff_image_path:
        try:
            imgs = [image_to_b64(diff_image_path)]
        except OSError:
            imgs = None
    try:
        return ollama_generate(base_url, model, prompt, images_b64=imgs)
    except requests.RequestException as e:
        return f"[Gemma/Ollama недоступна: {e}]"
