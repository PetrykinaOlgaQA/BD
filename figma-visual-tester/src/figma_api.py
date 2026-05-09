"""Минимальный клиент Figma Images API + устойчивое скачивание PNG."""

from __future__ import annotations

import os
import time
from typing import Callable, Optional, Tuple
from urllib.parse import quote

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError

_CONNECT_TIMEOUT = 30
_READ_TIMEOUT = 360
_CHUNK = 256 * 1024
_MAX_NOT_FOUND_STREAK = 12
_MAX_DOWNLOAD_ATTEMPTS = 48
_MAX_API_ATTEMPTS = 8


def _retry_sleep(response: requests.Response, attempt: int) -> float:
    h = response.headers.get("Retry-After")
    if h:
        try:
            return max(1.0, min(float(h.strip()), 120.0))
        except ValueError:
            pass
    return min(2.0 * (1.45**attempt), 45.0)


def figma_get(url: str, token: str, timeout: int = 120) -> requests.Response:
    headers = {"X-Figma-Token": token}
    last: Optional[requests.Response] = None
    for attempt in range(_MAX_API_ATTEMPTS):
        r = requests.get(url, headers=headers, timeout=timeout)
        last = r
        if r.status_code == 429:
            time.sleep(_retry_sleep(r, attempt))
            continue
        if r.status_code >= 500:
            time.sleep(min(2.0 + attempt * 1.5, 25.0))
            continue
        r.raise_for_status()
        return r
    if last is not None:
        last.raise_for_status()
    raise RuntimeError("Figma API: нет ответа")


def _download_png_stream(img_url: str, out_path: str, timeout: Tuple[int, int]) -> None:
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
        log("Figma: запрос рендера…")
    r = figma_get(api_url, token, timeout=timeout)
    data = r.json()
    if data.get("err"):
        raise RuntimeError(str(data["err"]))
    images = data.get("images") or {}
    img_url = next((v for v in images.values() if v), None)
    if not img_url:
        raise RuntimeError("Figma не вернула URL изображения")

    read_timeout = max(_READ_TIMEOUT, int(timeout) * 3)
    dl_timeout = (_CONNECT_TIMEOUT, read_timeout)
    transient = (ChunkedEncodingError, ConnectionError, Timeout, ProtocolError, OSError)
    not_found_streak = 0
    last_transient: Optional[BaseException] = None

    for attempt in range(_MAX_DOWNLOAD_ATTEMPTS):
        try:
            _download_png_stream(img_url, out_path, timeout=dl_timeout)
            return out_path
        except FileNotFoundError:
            not_found_streak += 1
            if not_found_streak >= _MAX_NOT_FOUND_STREAK:
                raise RuntimeError("CDN Figma долго отдаёт 404 (рендер не готов).") from None
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
        raise RuntimeError("Не удалось скачать PNG (сеть/таймаут).") from last_transient
    raise RuntimeError("Скачивание PNG: исчерпаны попытки.")


def export_frame_to_pil(
    file_key: str,
    node_id: str,
    token: str,
    scale: int = 2,
    timeout: int = 120,
    log: Optional[Callable[[str], None]] = None,
) -> "Image.Image":
    from tempfile import NamedTemporaryFile

    from PIL import Image

    with NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp = tf.name
    try:
        export_frame_png(file_key, node_id, token, tmp, scale=scale, timeout=timeout, log=log)
        return Image.open(tmp).copy()
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
