"""
Сверка макет (PNG) ↔ страница по разделам и блокам.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.typography_compare import (
    _color_distance,
    _crop_rgb,
    _font_label,
    _mean_rgb,
    _normalize_text,
    _ocr_text,
    _parse_css_px_one,
    _short,
)

_MIN_SECTION_DIFF_PCT = 1.5
_MIN_BLOCK_DIFF_PCT = 2.2
_SIMILAR_TEXT_RATIO = 0.82
_VISUAL_MATCH_SIM = 0.97
_VISUAL_SOFT_SIM = 0.92
_REPORT_BLOCK_DIFF_PCT = 4.5
_REPORT_EMOJI_DIFF_PCT = 10.0


def _union_bbox(els: List[Dict[str, Any]], pad: int = 4) -> Optional[Dict[str, int]]:
    xs, ys, x2s, y2s = [], [], [], []
    for el in els:
        try:
            x, y, w, h = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if w < 8 or h < 8:
            continue
        xs.append(x)
        ys.append(y)
        x2s.append(x + w)
        y2s.append(y + h)
    if not xs:
        return None
    x0, y0, x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad), max(x2s) + pad, max(y2s) + pad
    return {"x": x0, "y": y0, "w": max(8, x1 - x0), "h": max(8, y1 - y0)}


def _mask_frac(mask: Optional[np.ndarray], x: int, y: int, w: int, h: int) -> float:
    if mask is None:
        return 0.0
    mh, mw = mask.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(mw, x + w), min(mh, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean()) * 100.0


def _text_similar(a: str, b: str) -> float:
    na, nb = _normalize_text(a), _normalize_text(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _resize_rgb(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError:
        return rgb
    if rgb.shape[1] == w and rgb.shape[0] == h:
        return rgb
    mode = "RGB" if rgb.ndim == 3 else "L"
    im = Image.fromarray(rgb.astype(np.uint8), mode=mode)
    return np.array(im.resize((w, h), Image.Resampling.BILINEAR))


def _visual_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 — кропы визуально совпадают (рендер шрифтов / сглаживание допускается)."""
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    side = 96
    ah, aw = a.shape[:2]
    bh, bw = b.shape[:2]
    scale = side / max(aw, ah, bh, bw, 1)
    tw, th = max(8, int(aw * scale)), max(8, int(ah * scale))
    a_s = _resize_rgb(a, tw, th)
    b_s = _resize_rgb(b, tw, th)
    if a_s.shape != b_s.shape:
        b_s = _resize_rgb(b_s, a_s.shape[1], a_s.shape[0])
    af = a_s.astype(np.float32)
    bf = b_s.astype(np.float32)
    if af.ndim == 3:
        af = af.reshape(-1, 3)
        bf = bf.reshape(-1, 3)
    else:
        af = af.reshape(-1, 1)
        bf = bf.reshape(-1, 1)
    mse = float(np.mean((af - bf) ** 2)) / (255.0**2)
    return max(0.0, 1.0 - mse * 12.0)


def _section_key(el: Dict[str, Any]) -> str:
    sec = str(el.get("section", "") or "").strip().lower()
    if sec:
        return sec
    sn = str(el.get("snippet", "") or "").lower()
    if "header" in sn or ".h1" in sn or "logo" in sn:
        return "header"
    if "fact-card" in sn:
        return "fact-card"
    if "facts-grid" in sn:
        return "facts-grid"
    if "stats" in sn and "stat-item" not in sn:
        return "stats"
    if "footer" in sn:
        return "footer"
    if "stat-item" in sn:
        return "stats"
    return ""


_SECTION_TITLES = {
    "header": "Шапка",
    "facts-grid": "Сетка карточек",
    "fact-card": "Карточка",
    "stats": "Блок статистики",
    "footer": "Подвал",
}


def _group_site_sections(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Разделы страницы: шапка, 4 карточки, статистика, подвал."""
    by_sec: Dict[str, List[Dict[str, Any]]] = {}
    for el in elements:
        if not isinstance(el, dict):
            continue
        key = _section_key(el)
        if not key:
            continue
        by_sec.setdefault(key, []).append(el)

    sections: List[Dict[str, Any]] = []
    order = ["header", "facts-grid", "fact-card", "fact-card", "fact-card", "fact-card", "stats", "footer"]

    # Карточки — отдельно по y
    cards = sorted(by_sec.get("fact-card", []), key=lambda e: (int(e.get("y", 0)), int(e.get("x", 0))))
    used_cards: set[int] = set()

    def _card_root_indices() -> List[int]:
        roots: List[int] = []
        for i, el in enumerate(cards):
            if i in used_cards:
                continue
            try:
                y, h = int(el["y"]), int(el["h"])
            except (KeyError, TypeError, ValueError):
                continue
            if h >= 80 or "fact-card" in str(el.get("snippet", "")).lower():
                roots.append(i)
        return roots

    card_roots = _card_root_indices()
    if not card_roots and cards:
        card_roots = list(range(min(4, len(cards))))

    for key in ("header", "stats", "footer"):
        els = by_sec.get(key, [])
        if not els:
            continue
        bb = _union_bbox(els, pad=8)
        if not bb:
            continue
        sections.append(
            {
                "id": key,
                "title": _SECTION_TITLES.get(key, key),
                "bbox": bb,
                "elements": els,
            }
        )

    for idx, ci in enumerate(card_roots[:4]):
        root = cards[ci]
        used_cards.add(ci)
        try:
            ry, rh = int(root["y"]), int(root["h"])
            rx = int(root["x"])
        except (KeyError, TypeError, ValueError):
            continue
        group = [root]
        for j, el in enumerate(cards):
            if j in used_cards or j == ci:
                continue
            try:
                y = int(el["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if ry - 8 <= y <= ry + rh + 8:
                group.append(el)
                used_cards.add(j)
        bb = _union_bbox(group, pad=6)
        if not bb:
            continue
        title = f"Карточка {idx + 1}"
        h3 = next(
            (e for e in group if "h3" in str(e.get("snippet", "")).lower() or _normalize_text(str(e.get("innerText", ""))) in _normalize_text(str(root.get("innerText", "")))),
            root,
        )
        sub = _short(str(h3.get("innerText", "") or title), 36)
        sections.append(
            {
                "id": f"fact-card-{idx + 1}",
                "title": f"{title}: {sub}",
                "bbox": bb,
                "elements": group,
            }
        )

    sections.sort(key=lambda s: (int(s["bbox"]["y"]), int(s["bbox"]["x"])))
    return sections


def _site_texts_in_section(els: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Текстовые узлы внутри раздела (заголовки, абзацы)."""
    rows: List[Dict[str, str]] = []
    for el in els:
        txt = str(el.get("innerText", "") or "").strip()
        if len(txt) < 2:
            continue
        if len(txt) > 200 and " " in txt:
            # контейнер с кучей текста — разбить не будем, пропустим крупный
            sn = str(el.get("snippet", "")).lower()
            if "card" in sn or "header" in sn or "stats" in sn:
                continue
        sn = str(el.get("snippet", "")).strip()
        tag = sn.split(".")[0].split("#")[0]
        role = "текст"
        if re.match(r"^h[1-6]", tag) or ".h1" in sn or ".h2" in sn or ".h3" in sn:
            role = "заголовок"
        elif "subtitle" in sn:
            role = "подзаголовок"
        elif "logo" in sn or len(txt) <= 3:
            role = "иконка/лого"
        rows.append(
            {
                "role": role,
                "text": txt,
                "font": _font_label(el),
                "color": str(el.get("color", "") or "").strip(),
                "snippet": sn,
                "el": el,
            }
        )
    rows.sort(key=lambda r: (int(r["el"].get("y", 0)), int(r["el"].get("x", 0))))
    # убрать вложенные дубли: если текст A содержится в B, оставить более короткий узел-лист
    filtered: List[Dict[str, str]] = []
    for r in rows:
        t = _normalize_text(r["text"])
        if any(t != _normalize_text(o["text"]) and t in _normalize_text(o["text"]) for o in rows):
            if len(r["text"]) > 80:
                continue
        filtered.append(r)
    return filtered


def _compare_section(
    section: Dict[str, Any],
    baseline_rgb: np.ndarray,
    current_rgb: np.ndarray,
    mask: Optional[np.ndarray],
    *,
    baseline_text_cache: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    bb = section["bbox"]
    x, y, w, h = int(bb["x"]), int(bb["y"]), int(bb["w"]), int(bb["h"])
    diff_pct = _mask_frac(mask, x, y, w, h)
    site_lines_pre = _site_texts_in_section(section.get("elements") or [])
    max_child_diff = 0.0
    for row in site_lines_pre:
        el = row.get("el") or {}
        try:
            max_child_diff = max(
                max_child_diff,
                _mask_frac(mask, int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])),
            )
        except (KeyError, TypeError, ValueError):
            continue
    if diff_pct < _MIN_SECTION_DIFF_PCT and max_child_diff < _MIN_BLOCK_DIFF_PCT:
        return []

    title = str(section["title"])
    base_crop = _crop_rgb(baseline_rgb, x, y, w, h, pad=0)
    cur_crop = _crop_rgb(current_rgb, x, y, w, h, pad=0)
    if base_crop is None or cur_crop is None:
        return []

    vis_sim = _visual_similarity(base_crop, cur_crop)
    if vis_sim >= _VISUAL_MATCH_SIM and diff_pct < 2.0 and max_child_diff < _MIN_BLOCK_DIFF_PCT:
        return []

    mockup_text = _ocr_text(base_crop)
    site_lines = site_lines_pre
    site_joined = " ".join(r["text"] for r in site_lines)

    issues: List[str] = []

    if baseline_text_cache and site_joined:
        from src.baseline_text_cache import compare_site_to_cached_section

        msg = compare_site_to_cached_section(
            baseline_text_cache,
            str(section.get("id", "")),
            site_joined,
            title=title,
        )
        if msg:
            issues.append(msg)

    if mockup_text and site_joined:
        sim = _text_similar(mockup_text, site_joined)
        if sim < _SIMILAR_TEXT_RATIO:
            issues.append(
                f"[{title}] текст раздела: макет «{_short(mockup_text, 55)}» ≠ сайт «{_short(site_joined, 55)}»"
            )

    # Построчно: заголовки и абзацы
    for row in site_lines:
        el = row["el"]
        try:
            ex, ey, ew, eh = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
        except (KeyError, TypeError, ValueError):
            continue
        el_diff = _mask_frac(mask, ex, ey, ew, eh)
        if el_diff < _MIN_BLOCK_DIFF_PCT:
            continue
        bc = _crop_rgb(baseline_rgb, ex, ey, ew, eh)
        cc = _crop_rgb(current_rgb, ex, ey, ew, eh)
        if bc is None or cc is None:
            continue
        el_vis = _visual_similarity(bc, cc)
        mock_line = _ocr_text(bc)
        site_line = row["text"]
        role = row.get("role", "")
        if mock_line and site_line:
            if _text_similar(mock_line, site_line) < _SIMILAR_TEXT_RATIO:
                issues.append(
                    f"[{title}] {role}: макет «{_short(mock_line, 40)}» → сайт «{_short(site_line, 40)}»"
                )
        elif (
            not mock_line
            and site_line
            and el_diff >= _REPORT_BLOCK_DIFF_PCT
            and el_vis < _VISUAL_MATCH_SIM
            and role not in ("иконка/лого",)
        ):
            issues.append(
                f"[{title}] {role}: визуально отличается от макета (diff {el_diff:.1f}%)"
            )
        elif role == "иконка/лого" and el_diff >= _REPORT_EMOJI_DIFF_PCT:
            issues.append(
                f"[{title}] эмодзи/иконка отличается от макета (diff {el_diff:.0f}%)"
            )
        if role not in ("иконка/лого",):
            br, cr = _mean_rgb(bc), _mean_rgb(cc)
            both_light = (
                br[0] > 190 and br[1] > 190 and br[2] > 190
                and cr[0] > 190 and cr[1] > 190 and cr[2] > 190
            )
            if _color_distance(br, cr) >= 52 and row.get("color") and not both_light:
                issues.append(
                    f"[{title}] {role}: цвет на сайте {row['color']}, не как в макете"
                )
        if (
            mock_line
            and row.get("font")
            and el_diff >= 10
            and role in ("заголовок", "подзаголовок")
            and _visual_similarity(bc, cc) < _VISUAL_SOFT_SIM
        ):
            issues.append(f"[{title}] {role}: на сайте {row['font']}")

    if not issues and diff_pct >= 6.0 and vis_sim < _VISUAL_SOFT_SIM:
        issues.append(f"[{title}] вёрстка раздела не совпадает с макетом (diff {diff_pct:.1f}%)")
    elif not issues and diff_pct >= _MIN_SECTION_DIFF_PCT and vis_sim < _VISUAL_MATCH_SIM:
        issues.append(f"[{title}] блок отличается от макета (diff {diff_pct:.1f}%)")

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for phrase in issues[:4]:
        k = _normalize_text(phrase)[:100]
        if k in seen:
            continue
        seen.add(k)
        out.append(
            {
                "text": phrase,
                "section": section.get("id", ""),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )
    return out


def build_section_bug_items(
    layout_elements: Optional[List[Any]],
    baseline_rgb: np.ndarray,
    current_rgb: np.ndarray,
    mask: Optional[np.ndarray],
    *,
    max_items: int = 16,
    baseline_text_cache: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Баг-репорт по разделам страницы."""
    if not isinstance(layout_elements, list) or not layout_elements:
        return []
    sections = _group_site_sections(layout_elements)
    if not sections:
        return []

    items: List[Dict[str, Any]] = []
    for sec in sections:
        if len(items) >= max_items:
            break
        items.extend(
            _compare_section(
                sec,
                baseline_rgb,
                current_rgb,
                mask,
                baseline_text_cache=baseline_text_cache,
            )
        )
    return items[:max_items]
