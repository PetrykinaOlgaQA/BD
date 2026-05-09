"""Ollama: Llama 3.2 Vision — три изображения, ответ строго в JSON (BugReport)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import requests
from loguru import logger
from PIL import Image

from src.report import BugReport, parse_bug_report_json
from src.utils import pil_to_base64_png

VISION_SYSTEM_PROMPT = """Ты — ведущий QA-инженер по визуальному регрессионному тестированию веб-интерфейсов.
Тебе даны три изображения в одном порядке:
1) Рендер макета Figma (эталон),
2) Скриншот реальной веб-страницы,
3) Карта различий (diff): яркие пиксели — зоны расхождения.

Твоя задача — сравнить страницу с макетом как для production-release: типографика, отступы, сетка, кнопки, иконки, цвета, обрезка текста, наличие/отсутствие блоков.

Правила:
- Учитывай допустимые отличия: сглаживание шрифтов, 1–2 px антиалиасинг, лёгкие JPEG-артефакты на скрине — не считай их дефектом сами по себе.
- Если расхождения только в мелочах ниже порога заметности для пользователя — вердикт PASS и низкая bug_probability.
- Если есть явный сдвиг блоков, другой размер кнопки, обрезанный текст, неверный цвет секции, пропавший элемент — FAIL и перечисли дефекты.
- Пиши summary_ru и поля defects на русском языке.
- severity: low | medium | high | critical.

ФОРМАТ ОТВЕТА: верни ТОЛЬКО один JSON-объект без markdown и без пояснений до или после. Схема:
{
  "verdict": "PASS" или "FAIL",
  "bug_probability": число от 0 до 1,
  "summary_ru": "строка",
  "defects": [
    {
      "title": "строка",
      "severity": "low|medium|high|critical",
      "location_hint": "строка",
      "expected": "строка",
      "actual": "строка",
      "recommendation": "строка"
    }
  ],
  "notes": "строка (можно пустая)"
}

Если дефектов нет, defects: []. Поле notes используй для оговорок (например «низкое разрешение diff»)."""


def _ollama_chat(
    base_url: str,
    model: str,
    messages: List[dict],
    timeout_sec: int,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    body = {"model": model, "messages": messages, "stream": False}
    r = requests.post(url, json=body, timeout=timeout_sec)
    r.raise_for_status()
    data = r.json()
    msg = (data.get("message") or {}) if isinstance(data, dict) else {}
    content = msg.get("content") or data.get("response") or ""
    return str(content).strip()


def analyze_with_llama_vision(
    figma_img: Image.Image,
    website_img: Image.Image,
    diff_img: Image.Image,
    *,
    ollama_base_url: str,
    model: str,
    timeout_sec: int = 180,
    extra_user_context: str = "",
) -> Tuple[Optional[BugReport], str, Optional[str]]:
    """
    Отправляет три PNG в Ollama (vision).
    Возвращает (BugReport|None, raw_model_text, error|None).
    """
    try:
        b64: List[str] = [
            pil_to_base64_png(figma_img),
            pil_to_base64_png(website_img),
            pil_to_base64_png(diff_img),
        ]
    except Exception as e:
        logger.exception("base64 изображений")
        return None, "", f"Ошибка подготовки изображений: {e}"

    user_text = (
        "Проанализируй три приложенных изображения в порядке: Figma, Website, Diff. "
        "Верни ровно один JSON по схеме из системного сообщения."
    )
    if extra_user_context.strip():
        user_text += "\n\nДоп. контекст от оператора:\n" + extra_user_context.strip()

    messages = [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_text,
            "images": b64,
        },
    ]

    try:
        raw = _ollama_chat(ollama_base_url, model, messages, timeout_sec)
    except requests.RequestException as e:
        logger.warning("Ollama HTTP: {}", e)
        return None, "", f"Ollama недоступен или таймаут: {e}"

    if not raw:
        return None, "", "Пустой ответ Ollama"

    parsed, err = parse_bug_report_json(raw)
    if parsed is not None:
        return parsed, raw, None

    logger.info("JSON не распарсился: {}", err)
    return None, raw, err


def text_fallback_bug_report(raw_text: str, parse_error: Optional[str]) -> BugReport:
    """Явный fallback для UI."""
    return BugReport(
        verdict="PASS",
        bug_probability=0.0,
        summary_ru=raw_text[:4000] if raw_text else "Нет данных",
        defects=[],
        notes=parse_error or "fallback",
    )
