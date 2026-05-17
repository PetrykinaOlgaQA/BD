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
    "num_predict": 420,
    "num_ctx": 2048,
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
# connect, read: первая генерация после простоя может грузить веса долго.
# Чтение по умолчанию 10 мин; переопределяется из config (ollama.timeout_read).
_OLLAMA_TIMEOUT = (90.0, 600.0)
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


def _extract_ollama_text(data: Dict[str, Any]) -> str:
    """Текст ответа: content, response, thinking (некоторые сборки moondream)."""
    if not isinstance(data, dict):
        return ""
    m = data.get("message")
    if isinstance(m, dict):
        text = _content_from_message_field(m.get("content"))
        if text:
            return text
        th = m.get("thinking")
        if isinstance(th, str) and th.strip():
            return th.strip()
    resp = data.get("response")
    if isinstance(resp, str) and resp.strip():
        return resp.strip()
    return ""


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


def _post_json_with_retries(
    url: str,
    payload: Dict[str, Any],
    timeout: Tuple[float, float],
    *,
    max_attempts: int = _OLLAMA_POST_RETRIES,
) -> requests.Response:
    """POST с повторами при обрыве соединения (Ollama под нагрузкой / первый прогон)."""
    last_exc: Optional[BaseException] = None
    n = max(1, min(10, int(max_attempts)))
    transient = (
        requests.ConnectionError,
        requests.Timeout,
        requests.exceptions.ChunkedEncodingError,
        ProtocolError,
        IncompleteRead,
        ConnectionResetError,
        BrokenPipeError,
    )
    for attempt in range(n):
        try:
            # Новая TCP-сессия на попытку — меньше залипаний пула после обрыва.
            with requests.Session() as s:
                r = s.post(url, json=payload, timeout=timeout)
            return r
        except transient as e:
            last_exc = e
            if attempt + 1 < n:
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
    *,
    http_timeout: Optional[Tuple[float, float]] = None,
    max_post_retries: int = _OLLAMA_POST_RETRIES,
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
    to = http_timeout or _OLLAMA_TIMEOUT
    r = _post_json_with_retries(url, payload, to, max_attempts=max_post_retries)
    r.raise_for_status()
    data = r.json()
    err = (data.get("error") or "").strip()
    if err:
        raise ValueError(f"Ollama /api/chat: {err}")
    return _extract_ollama_text(data)


def ollama_generate(
    base_url: str,
    model: str,
    prompt: str,
    images_b64: Optional[List[str]] = None,
    ollama_options: Optional[Dict[str, Any]] = None,
    *,
    http_timeout: Optional[Tuple[float, float]] = None,
    max_post_retries: int = _OLLAMA_POST_RETRIES,
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
    to = http_timeout or _OLLAMA_TIMEOUT
    r = _post_json_with_retries(url, payload, to, max_attempts=max_post_retries)
    r.raise_for_status()
    data = r.json()
    err = (data.get("error") or "").strip()
    if err:
        raise ValueError(f"Ollama /api/generate: {err}")
    return _extract_ollama_text(data)


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
    *,
    http_timeout: Tuple[float, float] = _OLLAMA_TIMEOUT,
    max_post_retries: int = _OLLAMA_POST_RETRIES,
) -> Tuple[Optional[str], Optional[str]]:
    """(текст_ответа_или_None, краткая_ошибка_или_None)."""
    try:
        t = fn(
            base_url,
            model,
            prompt,
            images_b64,
            ollama_options,
            http_timeout=http_timeout,
            max_post_retries=max_post_retries,
        )
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


def _vision_model_priority(name: str) -> int:
    """Меньше — выше приоритет для fallback, если основная модель вернула пустой текст."""
    low = name.lower()
    needles = (
        ("bakllava", 0),
        ("llava", 0),
        ("qwen2.5-vl", 1),
        ("qwen2-vl", 1),
        ("qwen3-vl", 1),
        ("gemma3", 2),
        ("gemma2", 2),
        ("llama3.2-vision", 3),
        ("mistral-small", 4),
        ("granite3.2", 5),
        ("smolvlm", 6),
        ("moondream", 30),
    )
    for needle, p in needles:
        if needle in low:
            return p
    if "vision" in low or re.search(r"\bvl[:_-]", low) or re.search(r"[-_]vl\b", low):
        return 15
    return 99


def _looks_like_vision_model(name: str) -> bool:
    return _vision_model_priority(name) < 99


def _is_moondream_tag(name: str) -> bool:
    return "moondream" in (name or "").strip().lower()


def _ollama_options_for_model(model: str) -> Optional[Dict[str, Any]]:
    """moondream на длинном промпте и большом num_ctx часто отдаёт пустой content."""
    if _is_moondream_tag(model):
        return dict(_OLLAMA_OPTIONS_LIGHT)
    return None


def _compact_verifier_prompt(
    context_label: str,
    mse_v: Any,
    cr_v: Any,
    th_v: Any,
    *,
    low_diff_block: str = "",
    task_lines: Optional[List[str]] = None,
) -> str:
    """Короткий промпт для moondream / малых VLM (без полного JSON)."""
    tasks = task_lines or []
    tasks_txt = ""
    if tasks:
        tasks_txt = "Подсказки по diff:\n" + "\n".join(str(t)[:120] for t in tasks[:12]) + "\n\n"
    pass_rule = ""
    if low_diff_block.strip():
        pass_rule = f"Если diff в пределах порога — одна строка:\n{_VERIFIER_NO_DIFF_LINE}\n\n"
    return (
        "По картинке diff (макет Figma vs сайт) дай только список правок на русском.\n"
        "Формат: каждая строка с '- ', элемент — действие (тире « — » между ними).\n"
        "Без вступления, без ```, без координат px.\n\n"
        f"{pass_rule}"
        f"Контекст: {(context_label or '').strip() or '—'}\n"
        f"Метрики: MSE≈{mse_v}, изменённые пиксели≈{cr_v}%, порог≈{th_v}%.\n\n"
        f"{tasks_txt}"
        "Пример:\n"
        "- Текст «Заказать» — выровнять по центру\n"
        "- .corporate-banner — увеличить отступ сверху\n"
    )


def _vision_fallback_candidates(names: List[str], primary: str) -> List[str]:
    """Другие теги с Ollama, похожие на vision, кроме уже выбранной основной модели."""
    p = (primary or "").strip().lower()
    cand: List[str] = []
    for n in names:
        q = (n or "").strip()
        if not q or q.lower() == p:
            continue
        if not _looks_like_vision_model(q):
            continue
        cand.append(q)
    cand.sort(key=_vision_model_priority)
    return cand


def _ollama_api_fns(try_generate_fallback: bool, model: str) -> Tuple[Callable[..., str], ...]:
    """moondream чаще отвечает через /api/generate, не через chat."""
    if _is_moondream_tag(model) or try_generate_fallback:
        return (ollama_generate, ollama_chat)
    return (ollama_chat,)


def _call_ollama_with_fallbacks(
    base_url: str,
    model: str,
    prompt: str,
    images_b64: Optional[List[str]],
    diff_image_path: Optional[str] = None,
    http_timeout: Tuple[float, float] = _OLLAMA_TIMEOUT,
    *,
    image_max_side: int = _OLLAMA_IMAGE_MAX_SIDE,
    max_post_retries: int = _OLLAMA_POST_RETRIES,
    vision_fallback: bool = False,
    try_generate_fallback: bool = False,
    fallback_on_empty: bool = True,
    compact_prompt: Optional[str] = None,
) -> str:
    """
    Сначала /api/chat с изображением (актуальный путь для Gemma3 vision),
    затем /api/generate с изображением, затем оба варианта только по тексту (метрики).
    При HTTP 500 «memory…» — повтор с меньшим diff и num_ctx.
    Если основная vision-модель стабильно отдаёт пустой JSON (часто moondream + длинный промпт),
    автоматически пробуются другие vision-образы с того же сервера (например llava).
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
    primary_tag = resolved
    img_side = max(256, min(2048, int(image_max_side)))
    compact = (compact_prompt or "").strip() or None

    def try_model(
        cur_model: str,
        *,
        prompt_text: Optional[str] = None,
        use_images: Optional[List[str]] = None,
    ) -> Optional[str]:
        attempt_notes: List[str] = []
        ptxt = prompt_text if prompt_text is not None else prompt
        imgs_in = use_images if use_images is not None else images_b64
        api_fns = _ollama_api_fns(try_generate_fallback, cur_model)
        model_opts = _ollama_options_for_model(cur_model)

        def one(
            fn: Callable[..., str],
            imgs: Optional[List[str]],
            ollama_options: Optional[Dict[str, Any]] = None,
        ) -> Optional[str]:
            opts = ollama_options if ollama_options is not None else model_opts
            got, err = _try_ollama(
                fn,
                base_url,
                cur_model,
                ptxt,
                imgs,
                opts,
                http_timeout=http_timeout,
                max_post_retries=max_post_retries,
            )
            if err:
                attempt_notes.append(f"[{cur_model}] {err}")
            elif not got:
                attempt_notes.append(f"[{cur_model}] {fn.__name__}: пустой контент в ответе")
            return got

        if imgs_in:
            for fn in api_fns:
                got = one(fn, imgs_in, None)
                if got:
                    notes.extend(attempt_notes)
                    return got
            if _notes_indicate_system_ram_model(attempt_notes):
                tiny_ram = (
                    [image_to_b64_for_ollama(diff_image_path, max_side=min(384, img_side), quality=60)]
                    if diff_image_path and os.path.isfile(diff_image_path)
                    else imgs_in
                )
                for fn in api_fns:
                    got = one(fn, tiny_ram, _OLLAMA_OPTIONS_MINIMAL)
                    if got:
                        notes.extend(attempt_notes)
                        return got + "\n\n(Ответ с урезанным контекстом и меньшим diff — RAM была на грани.)"

            if diff_image_path and os.path.isfile(diff_image_path) and _notes_indicate_oom(attempt_notes):
                attempt_notes.append("— повтор: diff 384px, num_ctx↓ (нехватка памяти Ollama)")
                tiny384 = [image_to_b64_for_ollama(diff_image_path, max_side=min(384, img_side), quality=65)]
                for fn in api_fns:
                    got = one(fn, tiny384, _OLLAMA_OPTIONS_LIGHT)
                    if got:
                        notes.extend(attempt_notes)
                        return (
                            got
                            + "\n\n(Ответ по уменьшенному diff: у Ollama не хватило памяти на полноразмерную картинку.)"
                        )
                attempt_notes.append("— повтор: diff 256px")
                tiny256 = [image_to_b64_for_ollama(diff_image_path, max_side=256, quality=58)]
                for fn in api_fns:
                    got = one(fn, tiny256, _OLLAMA_OPTIONS_LIGHT)
                    if got:
                        notes.extend(attempt_notes)
                        return (
                            got
                            + "\n\n(Ответ по сильно уменьшенному diff из‑за ошибки памяти vision в Ollama.)"
                        )

            note = (
                "\n\n(Ответ без просмотра diff-картинки: не удалось передать изображение в Ollama в штатном режиме; "
                "обнови Ollama или проверь, что gemma_model — vision-модель.)"
            )
            for fn in api_fns:
                got = one(fn, None, _OLLAMA_OPTIONS_LIGHT)
                if got:
                    notes.extend(attempt_notes)
                    return got + note
            if try_generate_fallback:
                for fn in api_fns:
                    got = one(fn, None, None)
                    if got:
                        notes.extend(attempt_notes)
                        return got + note
        else:
            for fn in api_fns:
                got = one(fn, None, _OLLAMA_OPTIONS_LIGHT)
                if got:
                    notes.extend(attempt_notes)
                    return got
            if try_generate_fallback:
                for fn in api_fns:
                    got = one(fn, None, None)
                    if got:
                        notes.extend(attempt_notes)
                        return got

        notes.extend(attempt_notes)
        return None

    got = try_model(primary_tag)
    if got:
        return got

    if _is_moondream_tag(primary_tag):
        if compact and compact != prompt:
            notes.append("— moondream: короткий промпт (без полного JSON)")
            got = try_model(primary_tag, prompt_text=compact)
            if got:
                return got
        notes.append("— moondream: только метрики, без картинки diff")
        got = try_model(primary_tag, prompt_text=compact or prompt, use_images=None)
        if got:
            return got + "\n\n(Список по метрикам: moondream не вернула текст с картинкой diff.)"

    use_alt_models = vision_fallback or fallback_on_empty
    if not use_alt_models:
        notes.append(
            "— другая vision-модель не пробуется (vision_fallback и fallback_on_empty выключены)."
        )
        model = primary_tag
        if _notes_suggest_only_connection_failures(notes):
            return _ollama_unreachable_reply(base_url, model)
        if _notes_indicate_system_ram_model(notes):
            return _ollama_system_ram_reply(model, base_url)
        tags = _ollama_list_models_hint(base_url)
        detail = "\n".join(notes) if notes else "(подробности не собраны)"
        detail_short = detail if len(detail) < 1200 else detail[:1200] + "\n…(обрезано)"
        raise RuntimeError(
            "Пустой ответ от Ollama.\n"
            f"Модель: «{primary_tag}». Для moondream включите fallback_on_empty: true или укажите llava в config.\n"
            f"Модели на сервере: {tags}\n{detail_short}"
        )

    alt_prompt = compact or prompt
    for alt in _vision_fallback_candidates(_ollama_tag_names(base_url), primary_tag):
        if vision_fallback:
            notes.append(f"— автопереключение: «{primary_tag}» не дала текста → пробуем «{alt}»")
        else:
            notes.append(f"— fallback_on_empty: «{primary_tag}» пусто → одна попытка «{alt}»")
        got = try_model(alt, prompt_text=alt_prompt)
        if got:
            return got
        if not vision_fallback:
            break

    model = primary_tag

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
        f"Основная модель из config: «{primary_tag}» (ollama pull {primary_tag}).\n"
        + oom_hint
        +         "Если в логах только «обрыв соединения» при этом /api/tags работает — закройте лишние программы, "
        "обновите Ollama, либо в config.json попробуйте ollama_url: http://localhost:11434 вместо 127.0.0.1.\n"
        "Лимит ожидания одного запроса к Ollama задаётся в config: `ollama.timeout_read` (секунды); при fallback на вторую vision-модель время может суммироваться.\n"
        "Для списка правок по diff нужна стабильная vision-модель: чаще всего **llava** или **qwen2.5-vl**; "
        "**moondream** иногда отвечает пустой строкой на длинный промпт — укажите в config `ollama.model`: `llava:latest` "
        "или дождитесь автопереключения, если llava уже есть (`ollama list`).\n"
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


_VERIFIER_NO_DIFF_LINE = "- Критичных расхождений с макетом не выявлено."


def _garbage_needles_in_text(low: str) -> bool:
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
    return any(n in low for n in needles)


def _extract_verifier_bullet_lines(text: str) -> List[str]:
    """Только строки вида '- …' / '* …' → нормализация к '- …'."""
    out: List[str] = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(("– ", "— ", "• ")):
            s = "- " + s[2:].lstrip()
        mo = re.match(r"^[-*•]\s+(.+)$", s)
        if not mo:
            continue
        body = mo.group(1).strip()
        if body:
            out.append("- " + body)
    return out


# Если модель всё же выдала «пройдитесь по блокам» — выкидываем строку (остальное оставляем).
_VAGUE_BULLET_SUBSTRINGS = (
    "на участке",
    "участок экрана",
    "в области экрана",
    "вертикальн",
    "полосе от",
    "полоса от",
    "полосу от",
    "diff красн",
    "сильно красн",
    "«красн",
    "пройдитесь",
    "пройтись по",
    "проверьте блок",
    "проверить блоки",
    "ячейка сетки",
    "карта diff",
    "по маске",
    "по маске diff",
    "сверить зону",
    "подправить margin, padding",
)


def _dedupe_bullet_lines(bullets: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for b in bullets:
        body = b[2:].strip() if b.startswith("- ") else b.strip()
        key = re.sub(r"\s+", " ", body.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(b if b.startswith("- ") else "- " + body)
    return out


def _filter_vague_bullet_lines(bullets: List[str]) -> List[str]:
    kept: List[str] = []
    for b in bullets:
        low = b[2:].lower() if b.startswith("- ") else b.lower()
        if any(sub in low for sub in _VAGUE_BULLET_SUBSTRINGS):
            continue
        kept.append(b)
    return kept


def _is_valid_verifier_task_list(text: str) -> bool:
    """Ровно список задач: каждая непустая строка начинается с '- '."""
    t = (text or "").strip()
    if len(t) < 3:
        return False
    low = t.lower()
    if _garbage_needles_in_text(low):
        return False
    if "###" in t or "<instructions>" in low or "<data>" in low:
        return False
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return False
    for ln in lines:
        if not ln.startswith("- "):
            return False
    return True


def _fallback_verifier_task_list(stats: Dict[str, Any], _context_label: str = "") -> str:
    """Только маркированный список (ответ модели недоступен или невалиден)."""
    lines: List[str] = []
    raw_tasks = stats.get("diff_hotspots_tasks")
    if isinstance(raw_tasks, list):
        for t in raw_tasks:
            s = str(t).strip()
            if not s:
                continue
            if s.startswith("- "):
                lines.append(s)
            else:
                lines.append("- " + s)
            if len(lines) >= 14:
                break
    try:
        cr = float(stats.get("changed_ratio_pct", 0) or 0)
    except (TypeError, ValueError):
        cr = 0.0
    try:
        th = float(stats.get("threshold_pct", 0) or 0)
    except (TypeError, ValueError):
        th = 0.0
    if not lines and cr <= th and th > 0:
        return _VERIFIER_NO_DIFF_LINE + "\n"
    cap_wait = stats_capture_wait_seconds(stats)
    if cap_wait < CAPTURE_WAIT_TYPING_OK_SEC:
        lines.append(
            "- Похоже, анимация (typing/fade) не успела к моменту скрина — увеличить паузу после загрузки "
            "(capture_wait_seconds в config или time.sleep в capture_screenshot, не меньше ~1.5 с)"
        )
    if cr > 20:
        lines.append(
            "- Очень большой diff: часто это не «сломанная вёрстка», а другой кадр или масштаб — сверить размер окна и экспорт Figma (window_size, figma.scale)"
        )
    if not lines:
        lines.append(
            "- Не удалось получить разбор от модели — повторить запрос; при сбое попробовать другую vision-модель или меньший num_ctx"
        )
    lines = _dedupe_bullet_lines(lines)
    return "\n".join(lines) + "\n"


def _sanitize_verifier_task_list(raw: str, stats: Dict[str, Any], context_label: str) -> str:
    t = (raw or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t).strip()
    bullets = _dedupe_bullet_lines(_filter_vague_bullet_lines(_extract_verifier_bullet_lines(t)))
    candidate = "\n".join(bullets)
    if bullets and _is_valid_verifier_task_list(candidate):
        return candidate + "\n"
    alt: List[str] = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s.startswith("- "):
            continue
        low = s.lower()
        if any(sub in low for sub in _VAGUE_BULLET_SUBSTRINGS):
            continue
        alt.append(s)
    cand2 = "\n".join(alt)
    if alt and _is_valid_verifier_task_list(cand2):
        return cand2 + "\n"
    return _fallback_verifier_task_list(stats, context_label)


def explain_diff_ru(
    base_url: str,
    model: str,
    stats: Dict[str, Any],
    diff_image_path: str | None,
    use_image: bool = True,
    context_label: str = "",
    *,
    ollama_timeout: Optional[Tuple[float, float]] = None,
    image_max_side: int = _OLLAMA_IMAGE_MAX_SIDE,
    max_post_retries: int = _OLLAMA_POST_RETRIES,
    vision_fallback: bool = False,
    try_generate_fallback: bool = False,
    fallback_on_empty: bool = True,
) -> str:
    """
    Запрос к Ollama: визуальный разбор diff (Figma vs страница).

    Возвращает только маркированный список: короткие строки «элемент — действие» (префикс '- '), удобно для парсинга в баг-репорт.

    Рекомендуемые ollama_options: temperature 0.0 (обязательно для стабильного списка), top_p 0.85–0.95, num_predict 400–700,
    num_ctx 2048–4096 (для vision не завышать num_ctx). Для typing/fade: пауза перед скрином 1.5–2.5 с или capture_wait_seconds.

    ``ollama_timeout``: (connect, read) в секундах для HTTP к Ollama; read задаёт максимум ожидания одной генерации.

    Вызывать в ``run_pipeline`` после ``compare_screenshots`` с ``cr.diff_path`` и ``stats``.
    """
    stats_for_fallback = dict(stats)
    st_for_llm: Dict[str, Any] = dict(stats)
    ls = st_for_llm.get("layout_site")
    if isinstance(ls, dict):
        el = ls.get("elements")
        if isinstance(el, list) and len(el) > 20:
            st_for_llm["layout_site"] = {
                **ls,
                "elements": el[:20],
                "elements_omitted": len(el) - 20,
            }
    st_for_llm.pop("diff_hotspots_tasks", None)
    st_for_llm.pop("layout_elements_for_crops", None)
    _dh = st_for_llm.get("diff_hotspots")
    if isinstance(_dh, dict):
        # grid_cells даёт модели шаблон «вертикальная полоса от ~Y px» — в промпт не передаём.
        st_for_llm["diff_hotspots"] = {
            "changed_pixels_pct": _dh.get("changed_pixels_pct"),
            "mask_size": _dh.get("mask_size"),
            "elements_overlap": _dh.get("elements_overlap"),
        }

    try:
        cr_m = float(st_for_llm.get("changed_ratio_pct", 999))
        th_m = float(st_for_llm.get("threshold_pct", 0) or 0)
    except (TypeError, ValueError):
        cr_m, th_m = 999.0, 0.0
    low_diff_block = ""
    if th_m > 0 and cr_m <= th_m:
        low_diff_block = (
            "METRICS_PASS_RULE: changed_ratio_pct <= threshold_pct in JSON.\n"
            "Output ONLY this single line, verbatim (ASCII hyphen + space):\n"
            f"{_VERIFIER_NO_DIFF_LINE}\n"
            "Do not add any other line.\n"
        )

    mse_v = st_for_llm.get("mse", "—")
    cr_v = st_for_llm.get("changed_ratio_pct", "—")
    th_v = st_for_llm.get("threshold_pct", "—")

    lines = [
        "Сравни макет Figma и страницу по diff-картинке и JSON. Ответ — только список правок для баг-репорта.",
        "",
        "ФОРМАТ ОТВЕТА (обязательно):",
        "Только строки, начинающиеся с '- '. Русский язык. Без заголовков, без вступления, без ```, без нумерации 1.",
        "Каждая строка — одна правка, максимально коротко: «кто/что» — «что сделать».",
        "Разделитель в строке: длинное тире « — » (пробел, тире, пробел), затем глагол: увеличить, уменьшить, сдвинуть, выровнять, сделать крупнее/мельче, добавить задержку и т.п.",
        "Сначала укажи элемент: видимый текст в «ёлочках» («Заказать», «Корпоративные сайты»); если текста нет — класс из snippet (.corporate-banner, div.banner__title).",
        "Не пиши длинные объяснения, «кажется», «вероятно», проценты, MSE, px, если без них можно обойтись.",
        "",
        *(low_diff_block.splitlines() if low_diff_block else []),
        *([""] if low_diff_block else []),
        "ЗАПРЕЩЕНО: участок, полоса, вертикальн, ячейка сетки, по маске, diff красн, координаты px, «проверьте блоки», общие фразы без имени элемента.",
        "В JSON diff_hotspots без сетки — не описывай зоны координатами; называй блок и действие.",
        "",
        "FEW_SHOT (копируй только формат, факты — со своего экрана):",
        "- Текст «Корпоративные сайты» — увеличить размер",
        "- Заголовок в баннере — сделать крупнее",
        "- Отступ сверху у .corporate-banner — увеличить",
        "- Картинка .corporate-banner__image — сдвинуть вправо",
        "- Боковые отступы у .brand-gallery__inner — уменьшить",
        "- Typing-анимация — добавить time.sleep(1.8) перед скриншотом",
        "- Текст «Заказать» — выровнять по центру",
        "",
        "Оформление текста: в начале пункта можно выделить элемент жирным через Markdown **…** "
        "(например: - **Текст «Корпоративные сайты»** — увеличить размер). Другие теги HTML/Markdown не использовать.",
        "",
        "<DATA>",
        "Контекст:",
        (context_label or "").strip() or "—",
        "",
        "Метрики:",
        f"MSE ≈ {mse_v}, изменённые пиксели ≈ {cr_v}%, порог PASS ≈ {th_v}%.",
        "",
        "JSON:",
        json.dumps(st_for_llm, ensure_ascii=False, indent=2),
        "</DATA>",
    ]
    prompt = "\n".join(lines)
    tasks_raw = stats.get("diff_hotspots_tasks")
    task_lines = [str(t) for t in tasks_raw[:12]] if isinstance(tasks_raw, list) else None
    compact = _compact_verifier_prompt(
        context_label,
        mse_v,
        cr_v,
        th_v,
        low_diff_block=low_diff_block,
        task_lines=task_lines,
    )
    imgs: Optional[List[str]] = None
    side = max(256, min(2048, int(image_max_side)))
    if use_image and diff_image_path:
        try:
            imgs = [image_to_b64_for_ollama(diff_image_path, max_side=side)]
        except OSError:
            imgs = None
        except Exception:
            try:
                imgs = [image_to_b64(diff_image_path)]
            except OSError:
                imgs = None

    try:
        retry_path = diff_image_path if (use_image and diff_image_path and imgs) else None
        raw = _call_ollama_with_fallbacks(
            base_url,
            model,
            prompt,
            imgs,
            diff_image_path=retry_path,
            http_timeout=ollama_timeout or _OLLAMA_TIMEOUT,
            image_max_side=side,
            max_post_retries=max_post_retries,
            vision_fallback=vision_fallback,
            try_generate_fallback=try_generate_fallback,
            fallback_on_empty=fallback_on_empty,
            compact_prompt=compact,
        )
        return _sanitize_verifier_task_list(raw, stats_for_fallback, context_label)
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
        fb = _fallback_verifier_task_list(stats_for_fallback, context_label)
        if fb.strip():
            return (
                fb
                + "\n\n*(Список по diff-hotspots: Ollama не ответила. Перезапустите `python web_server.py` "
                "после обновления кода; в config: `fallback_on_empty: true` или `model: llava:latest`.)*"
            )
        return "[Gemma/Ollama: не удалось получить текст от модели.]\n" + str(e)
