from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import requests
from PIL import Image
from urllib3.exceptions import IncompleteRead, ProtocolError

# Vision в Ollama на длинной странице + большой diff → «memory layout cannot be allocated»; держим картинку маленькой.
_OLLAMA_IMAGE_MAX_SIDE = 512
_OLLAMA_OPTIONS_DEFAULT: Dict[str, Any] = {"num_predict": 768, "num_ctx": 6144}
_OLLAMA_OPTIONS_LIGHT: Dict[str, Any] = {"num_predict": 512, "num_ctx": 4096}
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
        return (None, f"{fn.__name__}: обрыв соединения с Ollama (POST крупнее GET /api/tags). Детали: {e!r}")
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
        )
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

    tags = _ollama_list_models_hint(base_url)
    detail = "\n".join(notes) if notes else "(подробности не собраны — смотри окно Ollama / journalctl)"
    oom_hint = ""
    if _notes_indicate_oom(notes):
        oom_hint = (
            "\nПамять (RAM/VRAM): закройте браузеры и тяжёлые приложения, в PowerShell выполните `ollama ps` и при необходимости "
            "`ollama stop`, уменьшите окно скрина в config (window_size) или отключите картинку diff в UI (`--no-gemma-image`).\n"
        )
    raise RuntimeError(
        "Пустой ответ от Ollama: все варианты (/api/chat и /api/generate, с картинкой и без) вернули пустой текст.\n"
        f"Использовалась модель: «{model}» (ollama pull {model}).\n"
        + oom_hint
        + "Если в логах только «обрыв соединения» при этом /api/tags работает — закройте лишние программы, "
        "обновите Ollama, либо в config.json попробуйте ollama_url: http://localhost:11434 вместо 127.0.0.1.\n"
        "Для diff нужна vision-модель (llava, qwen2.5vl и т.д.).\n"
        f"Модели на сервере ({base_url}): {tags}\n"
        "Что пробовали:\n"
        f"{detail}"
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
    lines = [
        "Ты senior QA: сравниваешь скриншот свёрстанной страницы с кадром макета из Figma. Ответ строго по-русски.",
        "",
        "Структура ответа (используй именно эти заголовки в Markdown):",
        "### Резюме",
        "Одно предложение: совпадает ли реализация с ожидаемым видом по diff и метрикам.",
        "",
        "### Баги (структурированно)",
        "Опиши 3–8 проблем. Для КАЖДОЙ проблемы обязательно три строки с такими же метками (копируй формат):",
        "**Блок:** … (какой визуальный блок: шапка, карточка, кнопка, колонка и т.д.; по возможности совпадение с snippet из JSON)",
        "**Раздел:** … (зона страницы: hero, навигация, сетка услуг, подвал, сайдбар и т.п.)",
        "**Суть бага:** … (чем реализация отличается от макета: отступы, размер, цвет, шрифт, обрезка; оценка в px если видно по diff)",
        "Пиши по-русски, без общих фраз вроде «много красных пикселей» без конкретики.",
        "",
        "### Зона экрана",
        "Где сосредоточены отличия: верх / центр / низ / слева / справа / на всю ширину.",
        "",
        "### Почему это полезнее попиксельного отчёта",
        "1–2 предложения: попиксельный diff даёт только % и карту шума; твоя роль — "
        "интерпретировать семантически (какой элемент «сломан» и как это влияет на UX).",
        "",
        ctx + "Метрики сравнения и снимок отступов со страницы (JSON; elements — margin/padding в px как в браузере):",
        json.dumps(st_for_llm, ensure_ascii=False, indent=2),
        "",
        "Если есть изображение diff — опирайся на него; учитывай антиалиасинг и допуски по сдвигу из метрик.",
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
        return _call_ollama_with_fallbacks(base_url, model, prompt, imgs, diff_image_path=retry_path)
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
