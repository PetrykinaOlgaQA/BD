from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import requests
from PIL import Image

from src.utils import CAPTURE_WAIT_TYPING_OK_SEC, stats_capture_wait_seconds
from urllib3.exceptions import IncompleteRead, ProtocolError

# Vision в Ollama на длинной странице + большой diff → «memory layout cannot be allocated»; держим картинку маленькой.
_OLLAMA_IMAGE_MAX_SIDE = 512
# Умеренные лимиты: огромный num_ctx не помогает vision, но жрёт RAM при KV.
# Делаем ответы максимально стабильными: temperature=0.0.
_OLLAMA_OPTIONS_DEFAULT: Dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 0.9,
    "num_predict": 560,
    "num_ctx": 4096,
}
_OLLAMA_OPTIONS_LIGHT: Dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 0.9,
    "num_predict": 420,
    "num_ctx": 2048,
}
_OLLAMA_OPTIONS_MINIMAL: Dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 0.9,
    "num_predict": 320,
    "num_ctx": 1024,
}
# connect, read: первая генерация после простоя может грузить веса долго; read до 15 мин.
_OLLAMA_TIMEOUT = (90, 900)
_OLLAMA_POST_RETRIES = 4
_OLLAMA_RETRY_SLEEP = 2.5


def _ollama_tag_names(base_url: str) -> List[str]:
    try:
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=30)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
    except Exception:
        return []


def _resolve_model_name(base_url: str, model: str) -> str:
    """Сопоставляет имя из config с тегом из ollama list (llava:latest ↔ llava, gemma3 → gemma3:latest)."""
    want = (model or "").strip()
    if not want:
        return want
    names = _ollama_tag_names(base_url)
    if want in names:
        return want
    # В списке только «llava», в config «llava:latest» — Ollama так не находит.
    if ":" in want:
        bare = want.split(":")[0]
        if bare in names:
            return bare
    base = want.split(":")[0]
    for n in names:
        if n == f"{base}:latest" or (n.startswith(base + ":") and not n.endswith("-runner")):
            return n
    for n in names:
        if n.split(":")[0] == base:
            return n
    # Один образ на машине — подставляем его (часто в config осталось llava, а стоит только gemma3:latest).
    if names and want not in names and len(names) == 1:
        return names[0]
    return want


def _content_from_message_field(content: Any) -> str:
    """Ollama: content строка или список частей [{type,text}, …]."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
                elif p.get("type") == "text" and isinstance(p.get("content"), str):
                    parts.append(p["content"])
        return "\n".join(parts).strip()
    return ""


def _post_json_with_retries(url: str, payload: Dict[str, Any]) -> requests.Response:
    """POST с повторами при обрыве соединения (Ollama под нагрузкой / первый прогон)."""
    last_exc: Optional[BaseException] = None
    transient = (
        requests.ConnectionError,
        requests.Timeout,
        requests.exceptions.ChunkedEncodingError,
        ProtocolError,
        IncompleteRead,
        ConnectionResetError,
        BrokenPipeError,
    )
    for attempt in range(_OLLAMA_POST_RETRIES):
        try:
            # Новая TCP-сессия на попытку — меньше залипаний пула после обрыва.
            with requests.Session() as s:
                r = s.post(url, json=payload, timeout=_OLLAMA_TIMEOUT)
            return r
        except transient as e:
            last_exc = e
            if attempt + 1 < _OLLAMA_POST_RETRIES:
                time.sleep(_OLLAMA_RETRY_SLEEP * (attempt + 1))
                continue
            raise
    assert last_exc is not None
    raise last_exc


def ollama_chat(
    base_url: str,
    model: str,
    prompt: str,
    images_b64: Optional[List[str]] = None,
    ollama_options: Optional[Dict[str, Any]] = None,
) -> str:
    """POST /api/chat — предпочтительный путь для vision-моделей в Ollama."""
    url = base_url.rstrip("/") + "/api/chat"
    msg: Dict[str, Any] = {"role": "user", "content": prompt}
    if images_b64:
        msg["images"] = images_b64
    opts = {**_OLLAMA_OPTIONS_DEFAULT, **(ollama_options or {})}
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [msg],
        "stream": False,
        "keep_alive": "15m",
        "options": opts,
    }
    r = _post_json_with_retries(url, payload)
    r.raise_for_status()
    data = r.json()
    err = (data.get("error") or "").strip()
    if err:
        raise ValueError(f"Ollama /api/chat: {err}")
    m = data.get("message") or {}
    text = _content_from_message_field(m.get("content"))
    if not text:
        text = (data.get("response") or "").strip() if isinstance(data.get("response"), str) else ""
    return text


def ollama_generate(
    base_url: str,
    model: str,
    prompt: str,
    images_b64: Optional[List[str]] = None,
    ollama_options: Optional[Dict[str, Any]] = None,
) -> str:
    """POST /api/generate — запасной вариант для старых сборок / текстовых моделей."""
    url = base_url.rstrip("/") + "/api/generate"
    opts = {**_OLLAMA_OPTIONS_DEFAULT, **(ollama_options or {})}
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "15m",
        "options": opts,
    }
    if images_b64:
        payload["images"] = images_b64
    r = _post_json_with_retries(url, payload)
    r.raise_for_status()
    data = r.json()
    err = (data.get("error") or "").strip()
    if err:
        raise ValueError(f"Ollama /api/generate: {err}")
    out = (data.get("response") or "").strip()
    if not out and isinstance(data.get("message"), dict):
        out = _content_from_message_field(data["message"].get("content"))
    return out


def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def image_to_b64_for_ollama(
    path: str,
    max_side: int = _OLLAMA_IMAGE_MAX_SIDE,
    quality: int = 72,
) -> str:
    """PNG/JPEG → base64; длинная сторона не больше max_side (меньше RAM vision в Ollama)."""
    q = max(40, min(92, int(quality)))
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        m = max(w, h)
        if m > max_side:
            scale = max_side / float(m)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def _ollama_error_body(resp: requests.Response) -> str:
    try:
        j = resp.json()
        return (j.get("error") or "").strip()
    except Exception:
        return ""


def _human_ollama_failure(
    base_url: str,
    model: str,
    exc: BaseException,
    response: Optional[requests.Response] = None,
) -> str:
    if isinstance(exc, requests.Timeout):
        return (
            "[Gemma/Ollama: превышено время ожидания ответа.]\n"
            f"Адрес: {base_url}. Попробуй позже или отключи передачу diff в модель."
        )
    if isinstance(exc, requests.ConnectionError):
        return (
            "[Gemma/Ollama: сервер не отвечает — соединение отклонено или хост недоступен.]\n"
            f"Проверь, что Ollama запущена и слушает {base_url}.\n"
            "Windows: запусти приложение Ollama из меню «Пуск» или в терминале: ollama serve\n"
            f"Затем подтяни модель: ollama pull {model}\n"
            "Если Ollama в Docker — в config.json укажи верный ollama_url (например http://localhost:11434)."
        )
    if isinstance(exc, requests.HTTPError) and response is not None:
        detail = _ollama_error_body(response) or str(exc)
        if response.status_code == 404:
            hint = _ollama_list_models_hint(base_url)
            base = (model or "").strip().split(":")[0] or model
            return (
                "[Gemma/Ollama: модель не найдена (HTTP 404).]\n"
                f"Скачай образ (имя как в «ollama list»), чаще всего: ollama pull {base}\n"
                f"В config.json укажи gemma_model точно как в списке ниже (например llava или llava:7b).\n"
                f"Модели на {base_url}: {hint}\n"
                f"Сообщение сервера: {detail}"
            )
        return f"[Gemma/Ollama: HTTP {response.status_code}]\n{detail}"
    return f"[Gemma/Ollama: {exc}]"


def _ollama_list_models_hint(base_url: str) -> str:
    names = _ollama_tag_names(base_url)
    if names:
        return ", ".join(names[:30])
    try:
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=30)
        r.raise_for_status()
    except Exception as e:
        return f"(запрос /api/tags не удался: {e})"
    return "(список моделей пуст — выполни ollama pull …)"


def _try_ollama(
    fn: Callable[..., str],
    base_url: str,
    model: str,
    prompt: str,
    images_b64: Optional[List[str]],
    ollama_options: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """(текст_ответа_или_None, краткая_ошибка_или_None)."""
    try:
        t = fn(base_url, model, prompt, images_b64, ollama_options)
        text = (t or "").strip()
        return (text if text else None, None)
    except requests.Timeout as e:
        return (None, f"{fn.__name__}: таймаут HTTP (часто первая генерация грузит модель минутами). Детали: {e!r}")
    except requests.ConnectionError as e:
        return (None, f"{fn.__name__}: нет связи с Ollama ({e.__class__.__name__}). Проверь ollama serve и порт.")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            raise
        detail = _ollama_error_body(e.response) if e.response is not None else str(e)
        code = e.response.status_code if e.response is not None else 0
        return (None, f"{fn.__name__}: HTTP {code} {detail}".strip())
    except ValueError as e:
        return (None, f"{fn.__name__}: {e}")
    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        return (None, f"{fn.__name__}: {e!s}")


def _notes_indicate_oom(notes: List[str]) -> bool:
    blob = " ".join(notes).lower()
    return any(
        k in blob
        for k in (
            "memory layout",
            "cannot be allocated",
            "out of memory",
            "insufficient memory",
            "requires more system memory",
            "more system memory than is available",
        )
    )


def _notes_indicate_system_ram_model(notes: List[str]) -> bool:
    """Ollama 500: веса модели не влезают в RAM (типично для llava 7B на 4 ГБ)."""
    blob = "\n".join(notes).lower()
    return "requires more system memory" in blob or "more system memory than is available" in blob


def _ollama_system_ram_reply(model: str, base_url: str) -> str:
    return (
        "### Не хватает RAM под выбранную модель\n\n"
        f"Ollama вернула **HTTP 500**: модель **«{model}»** требует больше оперативной памяти, чем доступно "
        f"(см. сообщение сервера на `{base_url}`).\n\n"
        "**Что сделать (выбери одно):**\n\n"
        "1. **Поставить лёгкую vision-модель** (рекомендуется на ноутбуках с 8 ГБ RAM):\n"
        "   ```\n"
        "   ollama pull moondream\n"
        "   ```\n"
        "   В `config.json` укажи `gemma_model` и `ollama.model`: **`moondream`** (или как в `ollama list`).\n\n"
        "2. Другие относительно компактные варианты (смотри размер на ollama.com): `smolvlm`, `llava-phi3` и т.п.\n\n"
        "3. Закрой браузер и тяжёлые программы, при необходимости **`ollama stop`**, перезапусти Ollama и повтори.\n\n"
        "*Текущая `llava:latest` (~4.7 ГБ на диске) часто не помещается в 4 ГБ **свободной** RAM для загрузки.*\n"
    )


def _notes_suggest_only_connection_failures(notes: List[str]) -> bool:
    """Все попытки упали на сети (Ollama не слушает порт) — не показываем длинный тех. дамп."""
    if not notes:
        return False
    blob = "\n".join(notes).lower()
    if "10061" in blob or "отверг запрос" in blob or "connection refused" in blob:
        return True
    if "max retries exceeded" in blob and "failed to establish" in blob:
        return True
    if "newconnectionerror" in blob and "10061" in blob:
        return True
    return all(
        any(
            k in n.lower()
            for k in (
                "connection",
                "max retries",
                "обрыв",
                "таймаут",
                "timeout",
                "refused",
                "10061",
                "10060",
            )
        )
        for n in notes
    )


def _ollama_unreachable_reply(base_url: str, model: str) -> str:
    return (
        "### Ollama недоступна\n\n"
        f"По адресу `{base_url}` никто не отвечает (часто **WinError 10061** — порт закрыт). "
        "Сервер Ollama не запущен или слушает другой хост/порт.\n\n"
        "**Что сделать:**\n"
        "1. Запусти **Ollama** из меню «Пуск» или в отдельном окне PowerShell: `ollama serve`\n"
        "2. Подтяни **vision**-модель (для картинки diff): `ollama pull llava` или `ollama pull moondream` и укажи имя в `config.json` → `gemma_model` / блок `ollama.model`\n"
        "3. Если не помогает — попробуй в конфиге `http://localhost:11434` вместо `127.0.0.1`\n\n"
        f"*Модель из конфига:* `{model}`\n"
    )


def _call_ollama_with_fallbacks(
    base_url: str,
    model: str,
    prompt: str,
    images_b64: Optional[List[str]],
    diff_image_path: Optional[str] = None,
) -> str:
    """
    Сначала /api/chat с изображением (актуальный путь для Gemma3 vision),
    затем /api/generate с изображением, затем оба варианта только по тексту (метрики).
    При HTTP 500 «memory…» — повтор с меньшим diff и num_ctx.
    """
    try:
        requests.get(base_url.rstrip("/") + "/api/tags", timeout=8)
    except Exception as e:
        if _connection_like(e):
            return _ollama_unreachable_reply(base_url, model)

    resolved = _resolve_model_name(base_url, model)
    notes: List[str] = []
    if resolved != (model or "").strip():
        notes.append(f"Имя модели из config «{model}» заменено на «{resolved}» (как в ollama list).")
    model = resolved

    def one(
        fn: Callable[..., str],
        imgs: Optional[List[str]],
        ollama_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        got, err = _try_ollama(fn, base_url, model, prompt, imgs, ollama_options)
        if err:
            notes.append(err)
        return got

    if images_b64:
        for fn in (ollama_chat, ollama_generate):
            got = one(fn, images_b64, None)
            if got:
                return got
        if _notes_indicate_system_ram_model(notes):
            tiny_ram = [image_to_b64_for_ollama(diff_image_path, max_side=384, quality=60)] if diff_image_path and os.path.isfile(diff_image_path) else images_b64
            for fn in (ollama_chat, ollama_generate):
                got = one(fn, tiny_ram, _OLLAMA_OPTIONS_MINIMAL)
                if got:
                    return got + "\n\n(Ответ с урезанным контекстом и меньшим diff — RAM была на грани.)"

        if diff_image_path and os.path.isfile(diff_image_path) and _notes_indicate_oom(notes):
            notes.append("— повтор: diff 384px, num_ctx↓ (нехватка памяти Ollama)")
            tiny384 = [image_to_b64_for_ollama(diff_image_path, max_side=384, quality=65)]
            for fn in (ollama_chat, ollama_generate):
                got = one(fn, tiny384, _OLLAMA_OPTIONS_LIGHT)
                if got:
                    return (
                        got
                        + "\n\n(Ответ по уменьшенному diff: у Ollama не хватило памяти на полноразмерную картинку.)"
                    )
            notes.append("— повтор: diff 256px")
            tiny256 = [image_to_b64_for_ollama(diff_image_path, max_side=256, quality=58)]
            for fn in (ollama_chat, ollama_generate):
                got = one(fn, tiny256, _OLLAMA_OPTIONS_LIGHT)
                if got:
                    return (
                        got
                        + "\n\n(Ответ по сильно уменьшенному diff из‑за ошибки памяти vision в Ollama.)"
                    )

        note = (
            "\n\n(Ответ без просмотра diff-картинки: не удалось передать изображение в Ollama в штатном режиме; "
            "обнови Ollama или проверь, что gemma_model — vision-модель.)"
        )
        for fn in (ollama_chat, ollama_generate):
            got = one(fn, None, _OLLAMA_OPTIONS_LIGHT)
            if got:
                return got + note
        for fn in (ollama_chat, ollama_generate):
            got = one(fn, None, None)
            if got:
                return got + note
    else:
        for fn in (ollama_chat, ollama_generate):
            got = one(fn, None, _OLLAMA_OPTIONS_LIGHT)
            if got:
                return got
        for fn in (ollama_chat, ollama_generate):
            got = one(fn, None, None)
            if got:
                return got

    if _notes_suggest_only_connection_failures(notes):
        return _ollama_unreachable_reply(base_url, model)

    if _notes_indicate_system_ram_model(notes):
        return _ollama_system_ram_reply(model, base_url)

    tags = _ollama_list_models_hint(base_url)
    detail = "\n".join(notes) if notes else "(подробности не собраны — смотри окно Ollama / journalctl)"
    oom_hint = ""
    if _notes_indicate_oom(notes):
        oom_hint = (
            "\nПамять (RAM/VRAM): закройте браузеры и тяжёлые приложения, в PowerShell выполните `ollama ps` и при необходимости "
            "`ollama stop`, уменьшите окно скрина в config (window_size) или отключите картинку diff в UI (`--no-gemma-image`).\n"
        )
    # Короткий тех. блок без повторения полного repr каждой попытки
    detail_short = detail if len(detail) < 1200 else detail[:1200] + "\n…(обрезано)"
    raise RuntimeError(
        "Пустой ответ от Ollama: все варианты (/api/chat и /api/generate, с картинкой и без) вернули пустой текст.\n"
        f"Использовалась модель: «{model}» (ollama pull {model}).\n"
        + oom_hint
        + "Если в логах только «обрыв соединения» при этом /api/tags работает — закройте лишние программы, "
        "обновите Ollama, либо в config.json попробуйте ollama_url: http://localhost:11434 вместо 127.0.0.1.\n"
        "Для diff нужна vision-модель (llava, qwen2.5vl и т.д.).\n"
        f"Модели на сервере ({base_url}): {tags}\n"
        "Кратко что пробовали:\n"
        f"{detail_short}"
    )


def _connection_like(exc: BaseException) -> bool:
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    cur: BaseException | None = exc
    seen: Set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (ConnectionRefusedError, TimeoutError, BrokenPipeError)):
            return True
        if isinstance(cur, OSError):
            if getattr(cur, "winerror", None) in (10061, 10060):
                return True
        name = type(cur).__name__
        if name in ("NewConnectionError", "MaxRetryError", "NameResolutionError"):
            return True
        cur = cur.__cause__ or cur.__context__
    s = str(exc).lower()
    if "connection refused" in s or "failed to establish" in s or "max retries exceeded" in s:
        return True
    return False


def _is_nonsense_explain_output(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 40:
        return True
    low = t.lower()
    needles = (
        "строка из порядка",
        "строка для поиска",
        "нулевого порядка",
        "вхождений",
        "format_contract",
        "<instructions>",
        "<data>",
        "json (метрики",
    )
    if any(n in low for n in needles):
        return True
    if "###" not in t or "резюме" not in low:
        return True
    if "рекомендации" not in low:
        return True
    return False


def _fallback_explain_markdown(stats: Dict[str, Any], context_label: str) -> str:
    cr = stats.get("changed_ratio_pct", "—")
    mse = stats.get("mse", "—")
    ctx = (context_label or "").strip()
    ctx_line = f" ({ctx})" if ctx else ""
    raw_tasks = stats.get("diff_hotspots_tasks")
    task_lines: List[str] = []
    if isinstance(raw_tasks, list):
        task_lines = [str(t).strip() for t in raw_tasks if str(t).strip()]
    mismatch_para2 = (
        "При очень высоком проценте diff часто не совпадает кадр "
        "(размер окна, масштаб экспорта Figma, длинная страница против одного фрейма, скролл, анимации на момент скрина)."
    )
    if task_lines:
        joined = " ".join(task_lines[:6])
        mismatch_body = (
            "Содержательный разбор vision-модели недоступен; ниже зоны по детерминированной маске diff "
            "(сетка поверх изображения и пересечение с блоками из layout страницы). "
            + joined
            + " "
            + mismatch_para2
        )
    else:
        mismatch_body = (
            "Содержательный текст от модели отсутствует. "
            + mismatch_para2
        )
    cap_wait = stats_capture_wait_seconds(stats)
    delay_bullet = (
        "- Добавить delay 1500ms перед скриншотом из-за анимации typing/fade/transitions\n"
        if cap_wait < CAPTURE_WAIT_TYPING_OK_SEC
        else ""
    )
    rec_bullets = "\n".join(f"- {t}" for t in task_lines) if task_lines else ""
    if not rec_bullets:
        rec_bullets = (
            (delay_bullet if delay_bullet else "")
            + "- Сверить window_size в config с размером фрейма в Figma и figma.scale с масштабом экспорта\n"
            + "- Повторить ollama pull и прогон; при повторе сбоя сменить vision-модель или снизить num_ctx"
        )
    else:
        extra = (delay_bullet if delay_bullet else "") + (
            "- Сверить window_size в config с размером фрейма в Figma и figma.scale с масштабом экспорта"
        )
        rec_bullets = rec_bullets + "\n" + extra
    first_do = (
        "Сверить window_size и figma.scale с макетом; пауза после загрузки (capture_wait_seconds) уже "
        f"≥ {CAPTURE_WAIT_TYPING_OK_SEC} с — при сильном diff смотреть в первую очередь кадр и масштаб, затем повторить прогон.\n\n"
        if cap_wait >= CAPTURE_WAIT_TYPING_OK_SEC
        else "Сверить window_size и figma.scale с макетом, увеличить capture_wait_seconds при typing/fade, повторить прогон.\n\n"
    )
    return (
        "### Резюме\n"
        f"Автоматический разбор vision-модели не удался или ответ не в ожидаемом формате{ctx_line}. "
        f"По метрикам: MSE ≈ {mse}, доля изменённых пикселей ≈ {cr}%.\n\n"
        "### Что не совпало\n"
        + mismatch_body
        + "\n\n"
        "### Что сделать в первую очередь\n"
        + first_do
        + "### Рекомендации по правкам\n"
        + rec_bullets
        + "\n"
    )


def _sanitize_explain_diff_markdown(raw: str, stats: Dict[str, Any], context_label: str) -> str:
    t = (raw or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t).strip()
    if _is_nonsense_explain_output(t):
        return _fallback_explain_markdown(stats, context_label)
    return t


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
    stats_for_fallback = dict(stats)
    st_for_llm: Dict[str, Any] = dict(stats)
    ls = st_for_llm.get("layout_site")
    if isinstance(ls, dict):
        el = ls.get("elements")
        if isinstance(el, list) and len(el) > 14:
            st_for_llm["layout_site"] = {
                **ls,
                "elements": el[:14],
                "elements_omitted": len(el) - 14,
            }
    st_for_llm.pop("diff_hotspots_tasks", None)
    # Жёсткий контракт формата + few-shot: снижает «мусорные» ответы вроде «строка из порядка».
    format_contract = "\n".join(
        [
            "FORMAT_CONTRACT:",
            "Верни ответ СТРОГО в Markdown и СТРОГО в 4 секциях, в точности в таком порядке и с такими заголовками:",
            "### Резюме",
            "### Что не совпало",
            "### Что сделать в первую очередь",
            "### Рекомендации по правкам",
            "",
            "НЕЛЬЗЯ добавлять любой текст до первого заголовка или после последней секции.",
            "НЕЛЬЗЯ менять заголовки, добавлять другие секции, вставлять дисклеймеры.",
            "НЕЛЬЗЯ цитировать или пересказывать правила/инструкции из этого промпта. В ответе — только 4 секции по фактам.",
            "",
            "Правила секций:",
            "- В «Резюме» 1–3 предложения.",
            "- В «Что не совпало» 1–2 абзаца без списков и без нумерации.",
            "- В «Что сделать в первую очередь» 1 короткий абзац.",
            "- В «Рекомендации по правкам» — ТОЛЬКО список строк (каждая строка начинается с '- '), каждая строка — одна конкретная правка. Без подпунктов.",
            "",
            "Формат каждой строки в «Рекомендации по правкам» (ОЧЕНЬ ВАЖНО):",
            "- Разрешены только команды вида:",
            "  1) '<property> у <selector> изменить с <X> на <Y>'",
            "  2) '<property> у <selector> увеличить с <X> до <Y>'",
            "  3) '<property> у <selector> уменьшить с <X> до <Y>'",
            "  4) 'Добавить delay <N>ms перед скриншотом из-за анимации typing/fade/transitions'",
            "- <property>: padding-top | padding-bottom | margin-top | margin-left | font-size | line-height | width | height | gap | border-radius | color | background-color",
            "- <selector>: CSS-селектор (например .corporate-banner, .ui-text--fz94, header.header--white).",
            "- <X>/<Y>: только если видишь конкретные значения в JSON или однозначно на diff; иначе эту строку не добавляй.",
            "- Никаких других формулировок. Никаких «можно», «возможно», «похоже». Только команды.",
            "",
            "Если расхождения несущественные, всё равно заполни 4 секции. В «Рекомендации по правкам» тогда ровно одна строка:",
            "- Критичных расхождений по макету не выявлено.",
            "",
            "Не повторяй текст инструкций и не перечисляй правила ответа.",
        ]
    )

    few_shot_good = "\n".join(
        [
            "ПРИМЕР ИДЕАЛЬНОГО ФОРМАТА (НЕ ЦИТИРОВАТЬ И НЕ ПЕРЕСКАЗЫВАТЬ — только как образец структуры):",
            "### Резюме",
            "Страница в целом близка к макету, но в hero-секции есть заметные расхождения по отступам и позиции подписи.",
            "",
            "### Что не совпало",
            "В hero-блоке верхний отступ заголовка выглядит меньше, чем в Figma, из-за чего текст визуально «прижат» к шапке. Также подпись/typing‑текст смещён и может выглядеть иначе из‑за незавершённой анимации на момент скриншота.",
            "",
            "### Что сделать в первую очередь",
            "Сначала выровнять вертикальные отступы hero-блока по макету и стабилизировать состояние анимаций перед снятием скриншота (увеличить ожидание).",
            "",
            "### Рекомендации по правкам",
            "- padding-top у .hero изменить с 120px на 144px",
            "- Добавить delay 1500ms перед скриншотом из-за анимации typing/fade/transitions",
        ]
    )

    animation_rules = "\n".join(
        [
            "АНИМАЦИИ И ПЕРЕХОДЫ:",
            "- На сайте могут быть typing/fade-in/transitions/карусели. Это часто даёт «шум» на diff.",
            "- Если расхождения выглядят как незавершённая анимация (двойной контур текста, дрожание, полупрозрачные состояния),",
            "  отметь это в «Что не совпало» как вероятную причину и НЕ ставь высокий приоритет.",
            "- Если есть признаки анимаций — в «Рекомендации по правкам» добавь строку только формата 'Добавить delay <N>ms…' (N обычно 1200–2000).",
        ]
    )

    diff_anchor_rules = "\n".join(
        [
            "МАСКА DIFF В JSON (diff_hotspots):",
            "- Если в JSON есть diff_hotspots с grid_cells и elements_overlap, это детерминированные якоря по карте отличий (не выдумывай координаты).",
            "- В «Что не совпало» первым абзацем кратко назови самые сильные зоны: топ ячеек сетки (col, row, origin x,y, % площади) и топ блоков по snippet из elements_overlap.",
            "- Вторым абзацем при необходимости добавь про несовпадение кадра или анимации, если diff очень широкий.",
        ]
    )

    lines = [
        "РОЛЬ: ты ведущий QA по визуальной регрессии. Сравниваешь эталон (рендер из Figma) и скрин реальной страницы.",
        "ЯЗЫК: русский.",
        "",
        "<INSTRUCTIONS>",
        format_contract,
        "",
        few_shot_good,
        "",
        animation_rules,
        "",
        diff_anchor_rules,
        "</INSTRUCTIONS>",
        "",
        "<DATA>",
        "Контекст:",
        ctx.strip() or "—",
        "",
        "JSON (метрики и элементы страницы с margin/padding/font):",
        json.dumps(st_for_llm, ensure_ascii=False, indent=2),
        "</DATA>",
    ]
    prompt = "\n".join(lines)
    imgs: Optional[List[str]] = None
    if use_image and diff_image_path:
        try:
            imgs = [image_to_b64_for_ollama(diff_image_path)]
        except OSError:
            imgs = None
        except Exception:
            try:
                imgs = [image_to_b64(diff_image_path)]
            except OSError:
                imgs = None

    try:
        retry_path = diff_image_path if (use_image and diff_image_path and imgs) else None
        raw = _call_ollama_with_fallbacks(base_url, model, prompt, imgs, diff_image_path=retry_path)
        return _sanitize_explain_diff_markdown(raw, stats_for_fallback, context_label)
    except requests.HTTPError as e:
        return _human_ollama_failure(base_url, model, e, response=e.response)
    except (requests.ConnectionError, requests.Timeout) as e:
        return _human_ollama_failure(base_url, model, e)
    except requests.RequestException as e:
        if _connection_like(e):
            return _human_ollama_failure(base_url, model, e)
        return f"[Gemma/Ollama: ошибка HTTP/сети]\n{e}"
    except RuntimeError as e:
        if _connection_like(e):
            return _human_ollama_failure(base_url, model, e)
        return "[Gemma/Ollama: не удалось получить текст от модели.]\n" + str(e)
