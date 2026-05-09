"""Скриншот страницы в headless Chrome."""

from __future__ import annotations

import os
import time
from typing import Tuple

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def capture_url_to_png(url: str, out_path: str, window_size: Tuple[int, int], wait_seconds: float = 3.0) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument(f"--window-size={window_size[0]},{window_size[1]}")
    opts.add_argument("--hide-scrollbars")

    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(url)
        time.sleep(wait_seconds)
        driver.save_screenshot(out_path)
    finally:
        driver.quit()
    return out_path


def capture_url_to_pil(url: str, window_size: Tuple[int, int], wait_seconds: float = 3.0) -> Image.Image:
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp = tf.name
    try:
        capture_url_to_png(url, tmp, window_size, wait_seconds=wait_seconds)
        return Image.open(tmp).copy()
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def load_image_file(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
