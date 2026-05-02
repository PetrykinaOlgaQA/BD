from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import quote

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError

# Figma отдаёт тяжёлый PNG по временному URL; на нестабильной сети часто рвётся chunked-ответ.
_CONNECT_TIMEOUT = 30
_READ_TIMEOUT = 360
_CHUNK = 256 * 1024
_MAX_NOT_FOUND_STREAK = 12
_MAX_DOWNLOAD_ATTEMPTS = 48
_FIGMA_API_MAX_ATTEMPTS = 8


def _figma_retry_sleep_seconds(response: requests.Response, attempt: int) -> float:
    """429/503: Figma может отдать Retry-After (секунды), иначе экспоненциальная пауза."""
    h = response.headers.get("Retry-After")
    if h:
        try:
            return max(1.0, min(float(h.strip()), 120.0))
        except ValueError:
            pass
    return min(2.0 * (1.45 ** attempt), 45.0)


def _figma_api_get(
    url: str,
    token: str,
    timeout: int = 120,
    log: Optional[Callable[[str], None]] = None,
) -> requests.Response:
    """GET к api.figma.com с повторами при 429 Too Many Requests и 5xx."""
    headers = {"X-Figma-Token": token}
    last: Optional[requests.Response] = None
    for attempt in range(_FIGMA_API_MAX_ATTEMPTS):
        r = requests.get(url, headers=headers, timeout=timeout)
        last = r
        if r.status_code == 429:
            wait = _figma_retry_sleep_seconds(r, attempt)
            if log:
                log(f"         Figma: лимит запросов (429), пауза {wait:.1f} с ({attempt + 1}/{_FIGMA_API_MAX_ATTEMPTS})…")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            if log:
                log(f"         Figma: HTTP {r.status_code}, повтор через пару секунд…")
            time.sleep(min(2.0 + attempt * 1.5, 25.0))
            continue
        r.raise_for_status()
        return r
    if last is not None:
        last.raise_for_status()
    raise RuntimeError("Figma API: пустой ответ после повторов")


def fetch_file_nodes_json(
    file_key: str,
    node_id: str,
    token: str,
    timeout: int = 120,
) -> Dict[str, Any]:
    """GET /v1/files/{key}/nodes — дерево узлов для указанных id (формат id: «19:2»)."""
    nid = quote(node_id, safe=":")
    url = f"https://api.figma.com/v1/files/{file_key}/nodes?ids={nid}"
    r = _figma_api_get(url, token, timeout=timeout, log=None)
    return r.json()


def save_nodes_json(file_key: str, node_id: str, token: str, out_path: str) -> str:
    data = fetch_file_nodes_json(file_key, node_id, token)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def _download_png_stream(
    img_url: str,
    out_path: str,
    timeout: Tuple[int, int] = (_CONNECT_TIMEOUT, _READ_TIMEOUT),
) -> None:
    """Скачивает PNG потоком; 404 — FileNotFoundError (CDN ещё не готов)."""
    tmp = out_path + ".part"
    try:
        with requests.get(img_url, stream=True, timeout=timeout) as ir:
            if ir.status_code == 404:
                raise FileNotFoundError(img_url)
            ir.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in ir.iter_content(chunk_size=_CHUNK):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp, out_path)
    finally:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def export_frame_png(
    file_key: str,
    node_id: str,
    token: str,
    out_path: str,
    scale: int = 2,
    timeout: int = 120,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nid = quote(node_id, safe=":")
    api_url = f"https://api.figma.com/v1/images/{file_key}?ids={nid}&format=png&scale={scale}"
    if log:
        log("         запрос URL рендера к Figma API…")
    r = _figma_api_get(api_url, token, timeout=timeout, log=log)
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

    read_timeout = max(_READ_TIMEOUT, int(timeout) * 3)
    dl_timeout: Tuple[int, int] = (_CONNECT_TIMEOUT, read_timeout)
    transient = (ChunkedEncodingError, ConnectionError, Timeout, ProtocolError, OSError)

    not_found_streak = 0
    last_transient: Optional[BaseException] = None

    for attempt in range(_MAX_DOWNLOAD_ATTEMPTS):
        try:
            if log and attempt > 0 and attempt % 8 == 0:
                log(f"         скачивание PNG с CDN, попытка {attempt + 1}/{_MAX_DOWNLOAD_ATTEMPTS}…")
            _download_png_stream(img_url, out_path, timeout=dl_timeout)
            return out_path
        except FileNotFoundError:
            not_found_streak += 1
            if not_found_streak >= _MAX_NOT_FOUND_STREAK:
                raise RuntimeError(
                    "Не удалось скачать PNG: CDN Figma долго отдаёт 404 (рендер ещё не готов)."
                )
            last_transient = None
            time.sleep(0.6)
        except transient as e:
            not_found_streak = 0
            last_transient = e
            time.sleep(min(1.0 + 0.35 * attempt, 12.0))
        except requests.HTTPError as he:
            not_found_streak = 0
            code = he.response.status_code if he.response is not None else 0
            if code >= 500 and attempt < _MAX_DOWNLOAD_ATTEMPTS - 1:
                last_transient = he
                time.sleep(min(1.0 + 0.35 * attempt, 12.0))
                continue
            raise

    if last_transient is not None:
        raise RuntimeError(
            "Не удалось полностью скачать PNG с сервера Figma (обрыв сети или таймаут). "
            "Повтори запуск, уменьши figma.scale в config или попробуй другую сеть/VPN."
        ) from last_transient

    raise RuntimeError("Не удалось скачать PNG после нескольких попыток.")


def public_design_url(file_key: str, node_id: str) -> str:
    """Публичная ссылка на кадр в Figma (в URL node-id задаётся через дефис)."""
    q = (node_id or "").strip().replace(":", "-")
    fk = (file_key or "").strip()
    return f"https://www.figma.com/design/{fk}?node-id={q}"
