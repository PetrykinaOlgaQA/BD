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
    timeout: int = 180,
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
    context_label: str = "",
) -> str:
    """
    context_label — короткая подпись сценария, напр. «эталон: прод», «эталон: Figma».
    """
    ctx = f"Контекст: {context_label}\n" if context_label else ""
    lines = [
        "Ты senior QA: тестируешь вёрстку сайта против эталона (макет или прод). Ответ строго по-русски.",
        "",
        "Структура ответа (используй именно эти заголовки в Markdown):",
        "### Резюме",
        "Одно предложение: совпадает ли реализация с ожидаемым видом по diff и метрикам.",
        "",
        "### Вероятные баги",
        "Нумерованный список 2–5 пунктов. Каждый пункт — конкретное несоответствие UI "
        "(что именно отличается: блок, отступ, шрифт, цвет, выравнивание, обрезка, иконка). "
        "Не пиши общие фразы вроде «много красных пикселей» без объяснения причины.",
        "",
        "### Зона экрана",
        "Где сосредоточены отличия: верх / центр / низ / слева / справа / на всю ширину.",
        "",
        "### Почему это полезнее попиксельного отчёта",
        "1–2 предложения: попиксельный diff даёт только % и карту шума; твоя роль — "
        "интерпретировать семантически (какой элемент «сломан» и как это влияет на UX).",
        "",
        ctx + "Метрики сравнения (JSON):",
        json.dumps(stats, ensure_ascii=False, indent=2),
        "",
        "Если есть изображение diff — опирайся на него; учитывай антиалиасинг и допуски по сдвигу из метрик.",
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
