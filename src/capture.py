from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Снимок отступов видимых блоков (для сравнения с макетом и промпта к VLM).
_LAYOUT_SNAPSHOT_JS = """
return (function() {
  const vh = window.innerHeight, vw = window.innerWidth;
  const nodes = Array.from(document.querySelectorAll('body *')).filter(function(el) {
    if (['SCRIPT','STYLE','NOSCRIPT','META','LINK','SVG','PATH'].indexOf(el.tagName) >= 0) return false;
    const r = el.getBoundingClientRect();
    return r.width >= 12 && r.height >= 12 && r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw;
  });
  nodes.sort(function(a, b) {
    var ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
    return (rb.width * rb.height) - (ra.width * ra.height);
  });
  const out = [];
  const max = 80;
  for (var i = 0; i < nodes.length && out.length < max; i++) {
    var el = nodes[i], r = el.getBoundingClientRect(), cs = getComputedStyle(el);
    var id = el.id ? ('#' + el.id) : '';
    var cn = el.className && typeof el.className === 'string' ? el.className.trim().split(/\\s+/).slice(0, 2).join('.') : '';
    var cls = cn ? ('.' + cn) : '';
    out.push({
      snippet: el.tagName.toLowerCase() + id + cls,
      x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height),
      margin: [cs.marginTop, cs.marginRight, cs.marginBottom, cs.marginLeft].join(' '),
      padding: [cs.paddingTop, cs.paddingRight, cs.paddingBottom, cs.paddingLeft].join(' ')
    });
  }
  return { viewport: { w: vw, h: vh }, elements: out };
})();
"""


def capture_screenshot(
    url: str,
    out_path: str,
    window_size: Tuple[int, int] = (1280, 720),
    wait_seconds: float = 2.0,
    collect_layout: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Скриншот страницы. При collect_layout возвращает также margin/padding
    крупнейших видимых элементов (для отчёта и сравнения отступов с макетом).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=%d,%d" % window_size)
    opts.add_argument("--hide-scrollbars")

    driver = webdriver.Chrome(options=opts)
    layout: Dict[str, Any] = {"viewport": {"w": window_size[0], "h": window_size[1]}, "elements": []}
    try:
        driver.get(url)
        time.sleep(wait_seconds)
        if collect_layout:
            try:
                snap = driver.execute_script(_LAYOUT_SNAPSHOT_JS)
                if isinstance(snap, dict):
                    layout = snap
            except Exception:
                pass
        driver.save_screenshot(out_path)
    finally:
        driver.quit()
    return out_path, layout
