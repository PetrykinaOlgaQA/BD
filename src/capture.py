from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Tuple

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService

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
  const max = 120;
  for (var i = 0; i < nodes.length && out.length < max; i++) {
    var el = nodes[i], r = el.getBoundingClientRect(), cs = getComputedStyle(el);
    var id = el.id ? ('#' + el.id) : '';
    var cn = el.className && typeof el.className === 'string' ? el.className.trim().split(/\\s+/).slice(0, 2).join('.') : '';
    var cls = cn ? ('.' + cn) : '';
    var inner = '';
    try {
      inner = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 120);
    } catch (e) { inner = ''; }
    var section = '';
    var p = el;
    while (p && p !== document.body) {
      var cn = (p.className && typeof p.className === 'string') ? p.className : '';
      if (cn.indexOf('header') >= 0) { section = 'header'; break; }
      if (cn.indexOf('fact-card') >= 0) { section = 'fact-card'; break; }
      if (cn.indexOf('facts-grid') >= 0) { section = 'facts-grid'; break; }
      if (cn.indexOf('stats') >= 0) { section = 'stats'; break; }
      if (cn.indexOf('footer') >= 0) { section = 'footer'; break; }
      p = p.parentElement;
    }
    out.push({
      snippet: el.tagName.toLowerCase() + id + cls,
      section: section,
      x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height),
      margin: [cs.marginTop, cs.marginRight, cs.marginBottom, cs.marginLeft].join(' '),
      padding: [cs.paddingTop, cs.paddingRight, cs.paddingBottom, cs.paddingLeft].join(' '),
      fontFamily: cs.fontFamily,
      fontSize: cs.fontSize,
      fontWeight: cs.fontWeight,
      lineHeight: cs.lineHeight,
      color: cs.color,
      innerText: inner
    });
  }
  return { viewport: { w: vw, h: vh }, elements: out };
})();
"""


def _browser_candidates() -> List[Tuple[str, str]]:
    """(путь к exe, 'chrome' | 'edge')."""
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()

    def _add(path: str | None, kind: str) -> None:
        if not path:
            return
        p = os.path.expandvars(os.path.expanduser(path.strip().strip('"')))
        if p.lower().endswith(".exe") and os.path.isfile(p):
            key = os.path.normcase(p)
            if key not in seen:
                seen.add(key)
                out.append((p, kind))

    _add(os.environ.get("CHROME_BINARY") or os.environ.get("CHROME_PATH"), "chrome")
    _add(os.environ.get("EDGE_BINARY") or os.environ.get("MSEDGE_BINARY"), "edge")

    for path, kind in [
        (r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe", "chrome"),
        (r"%ProgramFiles%\Google\Chrome\Application\chrome.exe", "chrome"),
        (r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe", "chrome"),
        (r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe", "edge"),
        (r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe", "edge"),
        (r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe", "edge"),
    ]:
        _add(path, kind)

    try:
        import winreg

        for subkey, name in (
            (r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe", "chrome"),
            (r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe", "edge"),
        ):
            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    with winreg.OpenKey(hive, subkey) as key:
                        _add(winreg.QueryValue(key, None), name)
                except OSError:
                    pass
    except ImportError:
        pass

    return out


def _headless_args(window_size: Tuple[int, int]) -> List[str]:
    w, h = int(window_size[0]), int(window_size[1])
    return [
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        f"--window-size={w},{h}",
        "--hide-scrollbars",
    ]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _BrowserSession:
    """WebDriver + фоновый процесс браузера (для CDP)."""

    def __init__(self, driver, proc: subprocess.Popen | None = None, tmpdir: str | None = None):
        self.driver = driver
        self._proc = proc
        self._tmpdir = tmpdir

    def quit(self) -> None:
        try:
            self.driver.quit()
        except Exception:
            pass
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=8)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)


def _create_driver_cdp(window_size: Tuple[int, int], binary: str, kind: str) -> _BrowserSession:
    """Headless Chrome/Edge + attach по CDP — без selenium-manager и без сети."""
    w, h = int(window_size[0]), int(window_size[1])
    port = _free_port()
    tmpdir = tempfile.mkdtemp(prefix="capture_cdp_")
    proc = subprocess.Popen(
        [
            binary,
            f"--remote-debugging-port={port}",
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={tmpdir}",
            f"--window-size={w},{h}",
            "--hide-scrollbars",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    addr = f"127.0.0.1:{port}"
    last_err: Exception | None = None
    for _ in range(40):
        if proc.poll() is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError(f"браузер завершился до подключения CDP ({binary})")
        try:
            if kind == "chrome":
                opts = ChromeOptions()
                opts.add_experimental_option("debuggerAddress", addr)
                driver = webdriver.Chrome(options=opts)
            else:
                opts = EdgeOptions()
                opts.add_experimental_option("debuggerAddress", addr)
                driver = webdriver.Edge(options=opts)
            return _BrowserSession(driver, proc, tmpdir)
        except Exception as exc:
            last_err = exc
            time.sleep(0.25)
    try:
        proc.terminate()
    except Exception:
        pass
    shutil.rmtree(tmpdir, ignore_errors=True)
    raise RuntimeError(f"CDP attach ({kind}): {last_err}")


def _create_driver_wdm(window_size: Tuple[int, int], binary: str, kind: str):
    """Chrome/Edge + webdriver-manager (нужен интернет для скачивания драйвера)."""
    args = _headless_args(window_size)
    if kind == "chrome":
        opts = ChromeOptions()
        for a in args:
            opts.add_argument(a)
        opts.binary_location = binary
        from webdriver_manager.chrome import ChromeDriverManager

        service = ChromeService(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=opts)
    opts = EdgeOptions()
    for a in args:
        opts.add_argument(a)
    opts.binary_location = binary
    from webdriver_manager.microsoft import EdgeChromiumDriverManager

    service = EdgeService(EdgeChromiumDriverManager().install())
    return webdriver.Edge(service=service, options=opts)


def _create_driver(window_size: Tuple[int, int]) -> _BrowserSession:
    """Сначала CDP (офлайн), затем webdriver-manager."""
    errors: List[str] = []
    candidates = _browser_candidates()
    if not candidates:
        raise RuntimeError(
            "Не найден Chrome/Edge. Установите браузер или задайте CHROME_BINARY / EDGE_BINARY "
            "с полным путём к .exe"
        )

    prefer_wdm = os.environ.get("CAPTURE_USE_WEBDRIVER_MANAGER", "").strip() in ("1", "true", "yes")
    order = ("wdm", "cdp") if prefer_wdm else ("cdp", "wdm")

    for mode in order:
        for binary, kind in candidates:
            try:
                if mode == "cdp":
                    return _create_driver_cdp(window_size, binary, kind)
                driver = _create_driver_wdm(window_size, binary, kind)
                return _BrowserSession(driver)
            except Exception as exc:
                errors.append(f"{mode}/{kind} ({binary}): {exc}")

    raise RuntimeError(
        "Не удалось запустить браузер для скриншота.\n" + "\n".join(errors[:6])
    )


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

    session = _create_driver(window_size)
    driver = session.driver
    w, h = int(window_size[0]), int(window_size[1])
    layout: Dict[str, Any] = {"viewport": {"w": w, "h": h}, "elements": []}
    try:
        driver.get(url)
        try:
            driver.execute_cdp_cmd(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": w,
                    "height": h,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )
        except Exception:
            pass
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
        session.quit()
    return out_path, layout
