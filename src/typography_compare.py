"""
Сверка текста и типографики: макет (PNG) vs страница (DOM + скрин).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_MIN_CHANGED_PCT = 4.0
_MIN_EDGE_PX = 10
_MAX_ITEMS = 14
_TEXT_TAG = re.compile(r"^(h[1-6]|p|span|a|button|label|li)$", re.I)


def _normalize_text(s: str) -> str:
    t = re.sub(r"\s+", " ", (s or "").strip())
    return t.lower().replace("ё", "е")


def _short(s: str, n: int = 48) -> str:
    t = re.sub(r"\s+", " ", (s or "").strip())
    if len(t) <= n:
        return t
    return t[: n - 1].rstrip() + "…"


def _parse_css_px_one(s: str) -> Optional[float]:
    s = (s or "").strip().lower()
    if not s or s == "normal":
        return None
    m = re.match(r"([\d.]+)\s*px", s)
    if m:
        return float(m.group(1))
    return None


def _crop_rgb(rgb: np.ndarray, x: int, y: int, w: int, h: int, pad: int = 6) -> Optional[np.ndarray]:
    if rgb is None or rgb.size == 0:
        return None
    mh, mw = rgb.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(mw, x + w + pad), min(mh, y + h + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return rgb[y0:y1, x0:x1]


def _ocr_text(rgb: np.ndarray) -> str:
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return ""
    try:
        if rgb.ndim == 2:
            im = Image.fromarray(rgb.astype(np.uint8), mode="L")
        else:
            im = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        w, h = im.size
        if w < 16 or h < 8:
            return ""
        if max(w, h) < 120:
            scale = max(2, int(160 / max(w, h)))
            im = im.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
        txt = pytesseract.image_to_string(im, lang="rus+eng", config="--psm 6")
        return re.sub(r"\s+", " ", txt).strip()
    except Exception:
        return ""


def _mean_rgb(rgb: np.ndarray) -> Tuple[int, int, int]:
    if rgb is None or rgb.size == 0:
        return (0, 0, 0)
    if rgb.ndim == 2:
        v = int(rgb.mean())
        return (v, v, v)
    sub = rgb.reshape(-1, 3).astype(np.float32)
    m = sub.mean(axis=0)
    return (int(m[0]), int(m[1]), int(m[2]))


def _css_color_to_rgb(s: str) -> Optional[Tuple[int, int, int]]:
    s = (s or "").strip().lower()
    if not s:
        return None
    m = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if s.startswith("#") and len(s) >= 7:
        try:
            return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))
        except ValueError:
            return None
    return None


def _color_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def _font_label(el: Dict[str, Any]) -> str:
    fam = str(el.get("fontFamily", "") or "").split(",")[0].strip().strip('"\'')
    fs = _parse_css_px_one(str(el.get("fontSize", "")))
    fw = str(el.get("fontWeight", "") or "").strip()
    parts: List[str] = []
    if fam and fam.lower() not in ("inherit", "initial"):
        parts.append(fam[:40])
    if fs is not None:
        parts.append(f"{fs:g}px")
    if fw and fw not in ("400", "normal"):
        parts.append(f"жирность {fw}")
    return ", ".join(parts) if parts else ""


def _visual_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    try:
        from PIL import Image
    except ImportError:
        return 0.0
    side = 96
    ah, aw = a.shape[:2]
    scale = side / max(aw, ah, 1)
    tw, th = max(8, int(aw * scale)), max(8, int(ah * scale))
    im_a = Image.fromarray(a.astype(np.uint8), mode="RGB" if a.ndim == 3 else "L")
    im_b = Image.fromarray(b.astype(np.uint8), mode="RGB" if b.ndim == 3 else "L")
    a_s = np.array(im_a.resize((tw, th), Image.Resampling.BILINEAR))
    b_s = np.array(im_b.resize((tw, th), Image.Resampling.BILINEAR))
    af = a_s.astype(np.float32).reshape(-1, 3 if a_s.ndim == 3 else 1)
    bf = b_s.astype(np.float32).reshape(-1, 3 if b_s.ndim == 3 else 1)
    mse = float(np.mean((af - bf) ** 2)) / (255.0**2)
    return max(0.0, 1.0 - mse * 12.0)


def _zone_label(snippet: str, y: int, h: int, img_h: int) -> str:
    low = (snippet or "").lower()
    if "header" in low or y < img_h * 0.2:
        return "шапка"
    if "footer" in low or y + h > img_h * 0.78:
        return "подвал"
    if re.match(r"^h1", low.split(".")[0]):
        return "заголовок"
    if "card" in low or "fact" in low:
        return "карточка"
    if "button" in low or "btn" in low:
        return "кнопка"
    if re.match(r"^h[2-6]", low.split(".")[0]):
        return "подзаголовок"
    if low.startswith("p") or low.startswith("span"):
        return "текст"
    return "блок"


def analyze_element_phrase(
    el: Dict[str, Any],
    baseline_rgb: np.ndarray,
    current_rgb: np.ndarray,
    *,
    changed_in_box_pct: float,
    img_h: int = 720,
) -> Optional[str]:
    try:
        x, y, w, h = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if w < 12 or h < 12 or float(changed_in_box_pct) < _MIN_CHANGED_PCT:
        return None

    sn = str(el.get("snippet", "")).strip()
    zone = _zone_label(sn, y, h, img_h)
    site_text = str(el.get("innerText", "") or "").strip()
    tag = sn.split(".")[0].split("#")[0].lower()

    base_crop = _crop_rgb(baseline_rgb, x, y, w, h)
    cur_crop = _crop_rgb(current_rgb, x, y, w, h)
    if base_crop is None or cur_crop is None:
        return None
    if _visual_similarity(base_crop, cur_crop) >= 0.97 and float(changed_in_box_pct) < 5:
        return None

    text_mock = _ocr_text(base_crop)
    text_site = site_text or _ocr_text(cur_crop)
    nm, ns = _normalize_text(text_mock), _normalize_text(text_site)

    issues: List[str] = []

    if nm and ns and nm != ns and len(nm) >= 2 and len(ns) >= 2:
        issues.append(
            f"текст не совпадает ({zone}): в макете «{_short(text_mock, 42)}», "
            f"на сайте «{_short(text_site, 42)}»"
        )
    is_emoji = "emoji" in sn.lower() or "logo" in sn.lower() or len(site_text) <= 3

    if (_TEXT_TAG.match(tag) or (site_text and len(site_text) >= 4)) and not is_emoji:
        br, cr = _mean_rgb(base_crop), _mean_rgb(cur_crop)
        both_light = (
            br[0] > 190 and br[1] > 190 and br[2] > 190
            and cr[0] > 190 and cr[1] > 190 and cr[2] > 190
        )
        if _color_distance(br, cr) >= 48 and not both_light:
            site_css = str(el.get("color", "") or "").strip()
            if site_css:
                issues.append(f"цвет текста не как в макете ({zone}), на сайте {site_css}")
            else:
                issues.append(f"цвет текста не как в макете ({zone})")

        if text_mock and float(changed_in_box_pct) >= 10:
            fs_site = _parse_css_px_one(str(el.get("fontSize", "")))
            if fs_site is not None:
                fl = _font_label(el)
                if fl:
                    issues.append(f"шрифт на сайте ({zone}): {fl} — не совпадает с макетом")

    if not is_emoji:
        for key, label in (
            ("padding", "padding"),
            ("margin", "margin"),
        ):
            vals = str(el.get(key, "") or "").split()
            if len(vals) != 4:
                continue
            try:
                px = [float(v.replace("px", "")) for v in vals]
            except ValueError:
                px = []
            if px and max(px) >= 20 and float(changed_in_box_pct) >= 10:
                issues.append(
                    f"{label} на сайте ({zone}): {' '.join(vals)} — отличается от макета"
                )

    if not issues and not is_emoji and float(changed_in_box_pct) >= 18:
        if site_text and len(_normalize_text(site_text)) >= 8:
            issues.append(f"блок «{_short(site_text, 32)}» ({zone}) не совпадает с макетом")
        elif not site_text:
            issues.append(f"вёрстка блока ({zone}) не совпадает с макетом")

    return issues[0] if issues else None


def build_typography_bug_items(
    scored: List[tuple[float, str, Dict[str, Any], float]],
    baseline_rgb: np.ndarray,
    current_rgb: np.ndarray,
    *,
    img_h: int = 720,
    max_items: int = _MAX_ITEMS,
) -> List[Dict[str, Any]]:
    """Пункты отчёта с привязкой к bbox: текст, шрифт, цвет, отступы."""
    items: List[Dict[str, Any]] = []
    seen_text: set[str] = set()
    for _frac, sn, el, ch in scored:
        if len(items) >= max_items:
            break
        phrase = analyze_element_phrase(
            el, baseline_rgb, current_rgb, changed_in_box_pct=ch, img_h=img_h
        )
        if not phrase:
            continue
        key = _normalize_text(phrase)[:80]
        if key in seen_text:
            continue
        seen_text.add(key)
        row: Dict[str, Any] = {"text": phrase}
        try:
            row.update(
                {
                    "x": int(el["x"]),
                    "y": int(el["y"]),
                    "w": int(el["w"]),
                    "h": int(el["h"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            pass
        snip = str(el.get("snippet", "")).strip()
        if snip:
            row["snippet"] = snip
        items.append(row)
    return items
