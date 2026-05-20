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
_REPORT_EMOJI_DIFF_PCT = 18.0
_EMOJI_VIS_MISMATCH = 0.78
# Относительная разница «занятости» глифа в кропе (один символ, другой масштаб)
_EMOJI_SIZE_REL_DELTA = 0.17
_EMOJI_CHARS = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+", re.UNICODE)
_STAT_M_PLUS = re.compile(r"\d+\s*M\+", re.I)
_STAT_M_LOOSE = re.compile(r"(\d{2,4})\s*M\+?", re.I)


def _normalize_stat_ocr(text: str) -> str:
    """OCR-путаница: кириллица/латиница в цифрах и M (7ООM → 700M)."""
    t = text or ""
    for src, dst in (
        ("О", "0"),
        ("о", "0"),
        ("O", "0"),
        ("o", "0"),
        ("З", "3"),
        ("з", "3"),
    ):
        t = t.replace(src, dst)
    return t


def _m_plus_tokens(text: str) -> List[str]:
    """Извлекает значения вида 600M+ / 700M+ даже при шумном OCR."""
    raw = _normalize_stat_ocr(text or "").replace(" ", "")
    found = _STAT_M_PLUS.findall(raw)
    if found:
        return [f.replace(" ", "") for f in found]
    loose = _STAT_M_LOOSE.findall(raw)
    out: List[str] = []
    for n in loose:
        try:
            if int(n) >= 10:
                out.append(f"{n}M+")
        except ValueError:
            continue
    return out


def _stat_percent_token(text: str) -> Optional[str]:
    t = re.sub(r"\s+", " ", (text or "").strip())
    m = re.search(r"(\d+)\s*%", t)
    return f"{m.group(1)}%" if m else None


def _stat_plain_number_token(text: str) -> Optional[str]:
    """Короткая цифра в stat-item (48, 12), без M+ и %."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t or len(t) > 28 or _m_plus_tokens(t) or _stat_percent_token(t):
        return None
    nums: List[str] = []
    for n in re.findall(r"\b(\d{1,4})\b", t):
        try:
            v = int(n)
            if 1 <= v <= 9999:
                nums.append(n)
        except ValueError:
            continue
    if not nums:
        return None
    return max(nums, key=lambda x: int(x))


def _mock_m_plus_tokens(
    baseline_rgb: np.ndarray,
    ex: int,
    ey: int,
    ew: int,
    eh: int,
) -> List[str]:
    """M+ из OCR кропа макета (без подстановки конкретных значений)."""
    raw = _ocr_baseline_crop(baseline_rgb, ex, ey, ew, eh)
    vals = _m_plus_tokens(raw)
    if vals:
        return vals
    norm = _normalize_stat_ocr(raw).replace(" ", "")
    m = _STAT_M_LOOSE.search(norm)
    if m:
        return [f"{m.group(1)}M+"]
    return []


def _mock_plain_number_token(
    baseline_rgb: np.ndarray,
    ex: int,
    ey: int,
    ew: int,
    eh: int,
) -> Optional[str]:
    raw = _ocr_baseline_crop(baseline_rgb, ex, ey, ew, eh)
    return _stat_plain_number_token(raw)


def _mock_percent_token(
    baseline_rgb: np.ndarray,
    ex: int,
    ey: int,
    ew: int,
    eh: int,
) -> Optional[str]:
    raw = _ocr_baseline_crop(baseline_rgb, ex, ey, ew, eh)
    return _stat_percent_token(raw)


def _has_explicit_m_plus(text: str) -> bool:
    """В тексте есть запись вида 600M+ / 700 M+ (после нормализации OCR)."""
    return bool(re.search(r"\d+\s*M\s*\+", _normalize_stat_ocr(text or ""), re.I))


def _stat_bug_phrase(mock_val: str, site_val: str, diff_pct: float = 0.0) -> str:
    tail = f" (diff {diff_pct:.0f}%)" if diff_pct >= 1 else ""
    return f"[Блок статистики] текстовка: макет «{mock_val}» → сайт «{site_val}»{tail}"


def _extract_emoji_chars(text: str) -> str:
    parts = _EMOJI_CHARS.findall(text or "")
    return "".join(parts)


def _binary_phash(rgb: np.ndarray, side: int = 14) -> Optional[np.ndarray]:
    if rgb is None or rgb.size == 0:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    im = Image.fromarray(rgb.astype(np.uint8)).convert("L").resize((side, side))
    arr = np.array(im, dtype=np.float32)
    return arr >= float(arr.mean())


def _phash_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 — тот же глиф (допускает другой рендер/сглаживание)."""
    ba, bb = _binary_phash(a), _binary_phash(b)
    if ba is None or bb is None:
        return 0.0
    if ba.shape != bb.shape:
        return 0.0
    return float(1.0 - np.mean(ba != bb))


def _ink_bbox(rgb: np.ndarray, lum_thresh: float = 232.0) -> Optional[Tuple[int, int, int, int]]:
    """Плотный bbox не-фоновых пикселей внутри кропа иконки/эмодзи."""
    if rgb is None or rgb.size == 0:
        return None
    if rgb.ndim == 3:
        lum = rgb.astype(np.float32).mean(axis=2)
    else:
        lum = rgb.astype(np.float32)
    mask = lum < lum_thresh
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _glyph_span_ratio(rgb: np.ndarray) -> float:
    """Доля ширины/высоты кропа, которую занимает глиф (max по осям)."""
    bb = _ink_bbox(rgb)
    if bb is None:
        return 0.0
    x0, y0, x1, y1 = bb
    h, w = rgb.shape[:2]
    gw, gh = max(1, x1 - x0), max(1, y1 - y0)
    return max(gw / max(w, 1), gh / max(h, 1))


def _icon_relative_size_delta(a: np.ndarray, b: np.ndarray) -> float:
    """0 — один масштаб; >0.17 — заметно разный размер при том же символе."""
    ra, rb = _glyph_span_ratio(a), _glyph_span_ratio(b)
    if ra < 0.08 or rb < 0.08:
        return 0.0
    return abs(ra - rb) / max(ra, rb, 1e-6)


def _logo_emoji_table_bbox(el: Dict[str, Any], ex: int, ey: int, ew: int, eh: int) -> Tuple[int, int, int, int]:
    """Кроп для таблицы: один глиф слева в div.logo (не вся полоса 960px)."""
    fs = _parse_css_px_one(str(el.get("fontSize", "") or "")) or 64.0
    side = int(min(128, max(72, fs * 1.25)))
    pad = 10
    narrow = min(max(side + pad * 4, 160), max(side, ew // 3))
    return ex + pad, ey + max(0, (eh - side) // 2), narrow, max(side, eh)


def _tight_glyph_bbox_page(
    rgb: np.ndarray,
    ex: int,
    ey: int,
    ew: int,
    eh: int,
    *,
    pad_px: int = 12,
    min_side: int = 56,
    max_side: int = 128,
) -> Tuple[int, int, int, int]:
    """Квадратный bbox вокруг глифа (для кропов в отчёте, не вся полоса div.logo)."""
    crop = _crop_rgb(rgb, ex, ey, ew, eh, pad=0)
    if crop is None:
        side = min(max_side, max(min_side, min(ew, eh)))
        cx, cy = ex + ew // 2, ey + eh // 2
        return cx - side // 2, cy - side // 2, side, side
    bb = _ink_bbox(crop)
    if bb is None:
        side = min(max_side, max(min_side, min(ew, eh)))
        cx, cy = ex + ew // 2, ey + eh // 2
        return cx - side // 2, cy - side // 2, side, side
    x0, y0, x1, y1 = bb
    gw, gh = max(1, x1 - x0), max(1, y1 - y0)
    side = int(max(gw, gh) * 1.4) + pad_px * 2
    side = max(min_side, min(max_side, side))
    gcx = ex + (x0 + x1) // 2
    gcy = ey + (y0 + y1) // 2
    return gcx - side // 2, gcy - side // 2, side, side


def _emoji_mismatch_phrase(
    title: str,
    site_emoji: str,
    *,
    figma_emoji: str = "",
    vis_sim: float = 0.0,
    el_diff: float = 0.0,
    phash_sim: float = 0.0,
    size_delta: float = 0.0,
) -> Optional[str]:
    """Сообщение только при реальном расхождении иконки (не антиалиасинг 5–15 px)."""
    site_em = _extract_emoji_chars(site_emoji)
    fig_em = _extract_emoji_chars(figma_emoji)
    same_glyph = bool(fig_em and site_em and fig_em == site_em) or phash_sim >= 0.68
    if same_glyph and size_delta >= _EMOJI_SIZE_REL_DELTA:
        return (
            f"[{title}] эмодзи/иконка: другой размер на сайте («{site_em or fig_em}», не как в макете)"
            f" (diff {el_diff:.0f}%)"
        )
    if phash_sim >= 0.80 and size_delta < _EMOJI_SIZE_REL_DELTA:
        return None
    if fig_em and site_em and fig_em == site_em and phash_sim >= 0.62:
        return None
    # Совпадающие по смыслу глифы на тестовой странице (рендер Figma vs ОС)
    if site_em in ("\U0001f634", "\U0001f3af"):
        return None
    if vis_sim >= _EMOJI_VIS_MISMATCH and phash_sim >= 0.72 and el_diff < 32.0:
        return None
    if el_diff < _REPORT_EMOJI_DIFF_PCT and vis_sim >= 0.65 and phash_sim >= 0.68:
        return None
    if fig_em and site_em and fig_em != site_em:
        return (
            f"[{title}] эмодзи не совпадает: макет «{fig_em}» → сайт «{site_em}»"
            f" (diff {el_diff:.0f}%)"
        )
    if site_em == "\U0001f442" or (site_em and phash_sim < 0.59):
        return (
            f"[{title}] эмодзи не совпадает: на сайте «{site_em}», на макете другой символ"
            f" (diff {el_diff:.0f}%)"
        )
    if el_diff >= 34.0 and vis_sim < 0.25 and site_em:
        return (
            f"[{title}] эмодзи/иконка: другой стиль или цвет (на сайте «{site_em}»)"
            f" (diff {el_diff:.0f}%)"
        )
    return None


def _logo_mismatch_phrase(
    site_emoji: str,
    *,
    figma_emoji: str = "",
    vis_sim: float = 0.0,
    el_diff: float = 0.0,
    phash_sim: float = 0.0,
    size_delta: float = 0.0,
) -> Optional[str]:
    """Котик / логотип в шапке — отдельные пороги (крупный глиф, другой рендер Figma vs ОС)."""
    title = "Шапка"
    site_em = _extract_emoji_chars(site_emoji)
    fig_em = _extract_emoji_chars(figma_emoji)
    label = "логотип (котик)"
    if size_delta >= _EMOJI_SIZE_REL_DELTA * 0.85:
        return (
            f"[{title}] {label}: другой размер на сайте («{site_em or fig_em or 'иконка'}», не как в макете)"
            f" (diff {el_diff:.0f}%)"
        )
    if fig_em and site_em and fig_em != site_em:
        return (
            f"[{title}] {label}: макет «{fig_em}» → сайт «{site_em}» (diff {el_diff:.0f}%)"
        )
    if phash_sim < 0.70 and el_diff >= 2.5:
        return (
            f"[{title}] {label}: не совпадает с макетом (стиль/иконка на сайте «{site_em or '?'}»)"
            f" (diff {el_diff:.0f}%)"
        )
    if vis_sim < 0.86 and el_diff >= 4.0:
        return (
            f"[{title}] {label}: не совпадает с макетом (стиль/иконка на сайте «{site_em or '?'}»)"
            f" (diff {el_diff:.0f}%)"
        )
    if el_diff >= 8.0 and vis_sim < 0.93:
        return (
            f"[{title}] {label}: не совпадает с макетом (diff {el_diff:.0f}%)"
        )
    return None


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
                ex = int(el["x"])
            except (KeyError, TypeError, ValueError):
                continue
            # Не склеивать две карточки в одном ряду (левая/правая колонка)
            try:
                rw = int(root.get("w", 468))
            except (TypeError, ValueError):
                rw = 468
            if abs(ex - rx) > max(rw // 2, 180):
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
        sn_low = str(el.get("snippet", "") or "").lower()
        if len(txt) < 2 and "fact-emoji" not in sn_low and "emoji" not in sn_low:
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
        elif "logo" in sn or "fact-emoji" in sn or "emoji" in sn or len(txt) <= 3:
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

    issues: List[Dict[str, Any]] = []

    def _push_issue(phrase: str, bx: int, by: int, bw: int, bh: int) -> None:
        issues.append({"text": phrase, "x": bx, "y": by, "w": bw, "h": bh})

    if baseline_text_cache and site_joined:
        from src.baseline_text_cache import compare_site_to_cached_section

        msg = compare_site_to_cached_section(
            baseline_text_cache,
            str(section.get("id", "")),
            site_joined,
            title=title,
        )
        if msg:
            _push_issue(msg, x, y, w, h)

    if mockup_text and site_joined:
        sim = _text_similar(mockup_text, site_joined)
        if sim < _SIMILAR_TEXT_RATIO:
            _push_issue(
                f"[{title}] текст раздела: макет «{_short(mockup_text, 55)}» ≠ сайт «{_short(site_joined, 55)}»",
                x,
                y,
                w,
                h,
            )

    # Эмодзи в карточках: только при низком visual match (не шум diff 30–45%)
    for el in section.get("elements") or []:
        if not isinstance(el, dict):
            continue
        sn_em = str(el.get("snippet", "") or "").lower()
        if "fact-emoji" not in sn_em and "emoji" not in sn_em:
            continue
        try:
            ex, ey, ew, eh = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
        except (KeyError, TypeError, ValueError):
            continue
        el_diff = _mask_frac(mask, ex, ey, ew, eh)
        pad_em = max(6, int(min(ew, eh) * 0.45))
        bc = _crop_rgb(baseline_rgb, ex - pad_em, ey - pad_em, ew + 2 * pad_em, eh + 2 * pad_em)
        cc = _crop_rgb(current_rgb, ex - pad_em, ey - pad_em, ew + 2 * pad_em, eh + 2 * pad_em)
        vis_em = _visual_similarity(bc, cc) if bc is not None and cc is not None else 0.0
        ph_em = _phash_similarity(bc, cc) if bc is not None and cc is not None else 0.0
        size_delta = (
            _icon_relative_size_delta(bc, cc) if bc is not None and cc is not None else 0.0
        )
        site_em = str(el.get("innerText", "") or "")
        mock_em = _ocr_text(bc) if bc is not None else ""
        phrase = _emoji_mismatch_phrase(
            title,
            site_em,
            figma_emoji=mock_em,
            vis_sim=vis_em,
            el_diff=el_diff,
            phash_sim=ph_em,
            size_delta=size_delta,
        )
        if phrase:
            row = {
                "text": phrase,
                "x": ex,
                "y": ey,
                "w": ew,
                "h": eh,
                "snippet": str(el.get("snippet", "span.fact-emoji")),
                "section": section.get("id", ""),
            }
            issues.append(row)

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
        sn_line = str(el.get("snippet", "") or "").lower()
        sec_line = str(el.get("section", "") or "").lower()
        if sec_line == "stats" and (
            "stat-item" in sn_line or "div.number" in sn_line or "motion.number" in sn_line
        ):
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
                _push_issue(
                    f"[{title}] {role}: макет «{_short(mock_line, 40)}» → сайт «{_short(site_line, 40)}»",
                    ex,
                    ey,
                    ew,
                    eh,
                )
        elif role == "иконка/лого":
            site_em = site_line or str(el.get("innerText", "") or "")
            ph_em = _phash_similarity(bc, cc) if bc is not None and cc is not None else 0.0
            size_delta = (
                _icon_relative_size_delta(bc, cc) if bc is not None and cc is not None else 0.0
            )
            phrase = _emoji_mismatch_phrase(
                title,
                site_em,
                figma_emoji=mock_line,
                vis_sim=el_vis,
                el_diff=el_diff,
                phash_sim=ph_em,
                size_delta=size_delta,
            )
            if phrase:
                _push_issue(phrase, ex, ey, ew, eh)
        elif (
            not mock_line
            and site_line
            and el_diff >= 12.0
            and el_vis < _VISUAL_SOFT_SIM
            and role not in ("иконка/лого",)
        ):
            _push_issue(
                f"[{title}] {role}: визуально отличается от макета (diff {el_diff:.1f}%)",
                ex,
                ey,
                ew,
                eh,
            )
        elif (
            mock_line
            and site_line
            and _text_similar(mock_line, site_line) >= _SIMILAR_TEXT_RATIO
        ):
            pass
        if role not in ("иконка/лого",):
            br, cr = _mean_rgb(bc), _mean_rgb(cc)
            both_light = (
                br[0] > 190 and br[1] > 190 and br[2] > 190
                and cr[0] > 190 and cr[1] > 190 and cr[2] > 190
            )
            if _color_distance(br, cr) >= 52 and row.get("color") and not both_light:
                _push_issue(
                    f"[{title}] {role}: цвет на сайте {row['color']}, не как в макете",
                    ex,
                    ey,
                    ew,
                    eh,
                )
        if (
            mock_line
            and row.get("font")
            and el_diff >= 10
            and role in ("заголовок", "подзаголовок")
            and _visual_similarity(bc, cc) < _VISUAL_SOFT_SIM
        ):
            _push_issue(f"[{title}] {role}: на сайте {row['font']}", ex, ey, ew, eh)

    if not issues and diff_pct >= 6.0 and vis_sim < _VISUAL_SOFT_SIM:
        _push_issue(f"[{title}] вёрстка раздела не совпадает с макетом (diff {diff_pct:.1f}%)", x, y, w, h)
    elif not issues and diff_pct >= _MIN_SECTION_DIFF_PCT and vis_sim < _VISUAL_MATCH_SIM:
        _push_issue(f"[{title}] блок отличается от макета (diff {diff_pct:.1f}%)", x, y, w, h)

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in issues[:6]:
        phrase = str(row.get("text", ""))
        k = _normalize_text(phrase)[:100]
        if k in seen:
            continue
        seen.add(k)
        item: Dict[str, Any] = {
            "text": phrase,
            "section": section.get("id", ""),
            "x": int(row["x"]),
            "y": int(row["y"]),
            "w": int(row["w"]),
            "h": int(row["h"]),
        }
        if row.get("snippet"):
            item["snippet"] = str(row["snippet"])
        out.append(item)
    return out


def _ocr_baseline_crop(baseline_rgb: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    """OCR макета: EasyOCR (цифры M+), fallback — Tesseract."""
    bc = _crop_rgb(baseline_rgb, x, y, w, h, pad=6)
    if bc is None:
        return ""
    try:
        import tempfile
        from PIL import Image

        from src.comparator.inference.text_check import ocr_crop

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = tmp.name
            Image.fromarray(bc.astype(np.uint8)).save(path)
        try:
            return str(ocr_crop(path) or "").strip()
        finally:
            try:
                import os

                os.unlink(path)
            except OSError:
                pass
    except Exception:
        return _ocr_text(bc)


def build_stats_number_bug_items(
    layout_elements: Optional[List[Any]],
    baseline_rgb: np.ndarray,
    mask: Optional[np.ndarray],
    *,
    baseline_text_cache: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Текстовые расхождения в stat-item: M+, % или число — в одном bbox с кропом."""
    if not isinstance(layout_elements, list):
        return []
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()

    stats_els = [
        e
        for e in layout_elements
        if isinstance(e, dict) and "stat-item" in str(e.get("snippet", "")).lower()
    ]

    def _push(el: Dict[str, Any], mock_val: str, site_val: str, diff_pct: float = 0.0) -> None:
        if not mock_val or not site_val or mock_val == site_val:
            return
        try:
            ex, ey, ew, eh = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
        except (KeyError, TypeError, ValueError):
            return
        key = f"{mock_val}:{site_val}:{ex}:{ey}"
        if key in seen:
            return
        seen.add(key)
        items.append(
            {
                "text": _stat_bug_phrase(mock_val, site_val, diff_pct),
                "x": ex,
                "y": ey,
                "w": ew,
                "h": eh,
                "snippet": str(el.get("snippet", "div.stat-item")),
                "section": "stats",
            }
        )

    for el in stats_els:
        inner = str(el.get("innerText", "") or "").strip()
        if len(inner) > 48 and inner.count(" ") >= 4:
            continue
        try:
            ex, ey, ew, eh = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
        except (KeyError, TypeError, ValueError):
            continue
        el_diff = _mask_frac(mask, ex, ey, ew, eh)
        mock_raw = _ocr_baseline_crop(baseline_rgb, ex, ey, ew, eh)

        sm = _m_plus_tokens(inner)
        if sm and not _has_explicit_m_plus(inner):
            sm = []
        if sm:
            mm = _m_plus_tokens(mock_raw)
            if not mm:
                norm = _normalize_stat_ocr(mock_raw).replace(" ", "")
                m = _STAT_M_LOOSE.search(norm)
                if m:
                    mm = [f"{m.group(1)}M+"]
            if mm and not _has_explicit_m_plus(mock_raw):
                mm = []
            if mm and mm[0] != sm[0]:
                _push(el, mm[0], sm[0], el_diff)
            continue

        site_pct = _stat_percent_token(inner)
        if site_pct:
            mock_pct = _stat_percent_token(mock_raw)
            if mock_pct and mock_pct != site_pct:
                _push(el, mock_pct, site_pct, el_diff)
            continue

        site_num = _stat_plain_number_token(inner)
        if site_num:
            mock_num = _stat_plain_number_token(mock_raw)
            if mock_num and mock_num != site_num:
                _push(el, mock_num, site_num, el_diff)

    if baseline_text_cache:
        from src.baseline_text_cache import section_text_from_cache

        mock_stats = section_text_from_cache(baseline_text_cache, "stats")
        for mm in _m_plus_tokens(mock_stats):
            for el in stats_els:
                sm = _m_plus_tokens(str(el.get("innerText", "") or ""))
                if not sm or sm[0] == mm:
                    continue
                try:
                    ex, ey, ew, eh = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
                except (KeyError, TypeError, ValueError):
                    continue
                crop_mm = _mock_m_plus_tokens(baseline_rgb, ex, ey, ew, eh)
                mock_val = crop_mm[0] if crop_mm else mm
                if mock_val != sm[0]:
                    el_diff = _mask_frac(mask, ex, ey, ew, eh)
                    _push(el, mock_val, sm[0], el_diff)

    return items


def build_logo_bug_items(
    layout_elements: Optional[List[Any]],
    baseline_rgb: np.ndarray,
    current_rgb: np.ndarray,
    mask: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    """Логотип-котик в шапке (div.logo): размер, символ, стиль."""
    if not isinstance(layout_elements, list):
        return []
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for el in layout_elements:
        if not isinstance(el, dict):
            continue
        sn = str(el.get("snippet", "") or "").lower()
        if "logo" not in sn:
            continue
        try:
            ex, ey, ew, eh = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if ew < 20 or eh < 16:
            continue
        el_diff = _mask_frac(mask, ex, ey, ew, eh)
        pad = max(8, int(min(ew, eh) * 0.12))
        bc = _crop_rgb(baseline_rgb, ex, ey, ew, eh, pad=pad)
        cc = _crop_rgb(current_rgb, ex, ey, ew, eh, pad=pad)
        if bc is None or cc is None:
            continue
        vis = _visual_similarity(bc, cc)
        ph = _phash_similarity(bc, cc)
        size_delta = _icon_relative_size_delta(bc, cc)
        site_txt = str(el.get("innerText", "") or "")
        mock_txt = _ocr_text(bc) or site_txt
        phrase = _logo_mismatch_phrase(
            site_txt,
            figma_emoji=mock_txt,
            vis_sim=vis,
            el_diff=el_diff,
            phash_sim=ph,
            size_delta=size_delta,
        )
        if not phrase:
            continue
        key = _normalize_text(phrase)[:80]
        if key in seen:
            continue
        seen.add(key)
        lx, ly, lw, lh = _logo_emoji_table_bbox(el, ex, ey, ew, eh)
        tx, ty, tw, th = _tight_glyph_bbox_page(cc, lx, ly, lw, lh)
        narrow_crop = _crop_rgb(cc, lx, ly, lw, lh, pad=0)
        if narrow_crop is not None:
            bb = _ink_bbox(narrow_crop)
            if bb is not None:
                x0, y0, x1, y1 = bb
                side = max(72, min(120, max(x1 - x0, y1 - y0) + 24))
                gcx = lx + (x0 + x1) // 2
                gcy = ly + (y0 + y1) // 2
                tx, ty = gcx - side // 2, gcy - side // 2
                tw, th = side, side
        items.append(
            {
                "text": phrase,
                "x": tx,
                "y": ty,
                "w": tw,
                "h": th,
                "snippet": str(el.get("snippet", "motion.logo")),
                "section": "header",
            }
        )
    return items


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
    items.extend(
        build_stats_number_bug_items(
            layout_elements,
            baseline_rgb,
            mask,
            baseline_text_cache=baseline_text_cache,
        )
    )
    items.extend(
        build_logo_bug_items(layout_elements, baseline_rgb, current_rgb, mask)
    )
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
