from __future__ import annotations

import os
import time
from typing import Tuple

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def capture_screenshot(
    url: str,
    out_path: str,
    window_size: Tuple[int, int] = (1280, 720),
    wait_seconds: float = 1.5,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=%d,%d" % window_size)
    opts.add_argument("--hide-scrollbars")
    
    # ✅ Selenium Manager сам скачает драйвер под твой Chrome 147!
    # Просто не передаём service=... — и всё работает
    driver = webdriver.Chrome(options=opts)
    
    try:
        driver.get(url)
        time.sleep(wait_seconds)
        driver.save_screenshot(out_path)
    finally:
        driver.quit()
    return out_path