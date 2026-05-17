"""
Короткий баг-репорт: ищем отличия вёрстки по diff, без CSS-селекторов и без спама по 4 сторонам.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.compare import build_change_mask

_BAND_MIN_FRAC = 0.07
# Подсказку про кадр показываем в HTML-метриках, не вместо списка багов.
_FRAME_HINT_ONLY_IF_EMPTY_PCT = 55.0
# 0 = без верхней границы по числу пунктов
DEFAULT_MAX_BUG_LINES = 0
_MIN_BOX_DIFF_FRAC = 0.008
# «Пустой» фон vs заметное содержимое в кропе
_CONTENT_LUM_THRESH = 235
_PRESENCE_MIN_FRAC = 0.10
_ABSENT_MAX_FRAC = 0.035
_PHRASE_MISSING_ON_MOCKUP = "фрагмента нет на макете"
_PHRASE_MISSING_ON_PAGE = "фрагмента нет на странице"
_EDGE_RU = {"top": "сверху", "bottom": "снизу", "left": "слева", "right": "справа"}


def _load_aligned_rgb_pair(
    baseline_path: str, current_path: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        from PIL import Image
    except ImportError:
        return None, None
    try:
        a = Image.open(baseline_path).convert("RGB")
        b = Image.open(current_path).convert("RGB")
        if a.size != b.size:
            b = b.resize(a.size, Image.Resampling.LANCZOS)
        return np.asarray(a), np.asarray(b)
    except OSError:
        return None, None


def _box_content_frac(rgb: np.ndarray, x0: int, y0: int, w: int, h: int) -> float:
    """Доля «непустых» пикселей в области (не белый фон)."""
    mh, mw = rgb.shape[:2]
    x1, y1 = min(mw, x0 + w), min(mh, y0 + h)
    x0, y0 = max(0, x0), max(0, y0)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub = rgb[y0:y1, x0:x1]
    if sub.size == 0:
        return 0.0
    if sub.ndim == 3:
        lum = sub.mean(axis=2)
    else:
        lum = sub.astype(np.float32)
    return float((lum < _CONTENT_LUM_THRESH).mean())


def _presence_phrases_for_bbox(
    baseline_rgb: np.ndarray,
    current_rgb: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    snippet: str = "",
) -> List[str]:
    """Контент только на сайте или только в макете Figma."""
    if w < 4 or h < 4:
        return []
    base_f = _box_content_frac(baseline_rgb, x, y, w, h)
    curr_f = _box_content_frac(current_rgb, x, y, w, h)
    out: List[str] = []
    if curr_f >= _PRESENCE_MIN_FRAC and base_f <= _ABSENT_MAX_FRAC:
        out.append(_phrase_with_element(_PHRASE_MISSING_ON_MOCKUP, snippet))
    elif base_f >= _PRESENCE_MIN_FRAC and curr_f <= _ABSENT_MAX_FRAC:
        out.append(_phrase_with_element(_PHRASE_MISSING_ON_PAGE, snippet))
    return out


def _presence_phrases_for_element(
    baseline_rgb: np.ndarray,
    current_rgb: np.ndarray,
    el: Dict[str, Any],
    snippet: str = "",
) -> List[str]:
    try:
        x, y, w, h = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
    except (KeyError, TypeError, ValueError):
        return []
    return _presence_phrases_for_bbox(baseline_rgb, current_rgb, x, y, w, h, snippet)


def _parse_css_px_list(s: str) -> List[float]:
    out: List[float] = []
    for part in (s or "").split():
        try:
            out.append(float(part.replace("px", "").strip()))
        except ValueError:
            out.append(0.0)
    return out


def _band_fracs(mask: np.ndarray, x0: int, y0: int, w: int, h: int) -> Dict[str, float]:
    mh, mw = mask.shape
    x1, y1 = min(mw, x0 + w), min(mh, y0 + h)
    x0, y0 = max(0, x0), max(0, y0)
    if x1 <= x0 or y1 <= y0:
        return {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0, "center": 0.0}
    sub = mask[y0:y1, x0:x1]
    sh, sw = sub.shape
    th = max(1, int(sh * 0.15))
    tw = max(1, int(sw * 0.15))
    top = float(sub[:th, :].mean()) if th < sh else float(sub.mean())
    bottom = float(sub[sh - th :, :].mean()) if th < sh else top
    left = float(sub[:, :tw].mean()) if tw < sw else float(sub.mean())
    right = float(sub[:, sw - tw :].mean()) if tw < sw else left
    cy0, cx0 = th, tw
    cy1, cx1 = max(cy0 + 1, sh - th), max(cx0 + 1, sw - tw)
    center = float(sub[cy0:cy1, cx0:cx1].mean()) if cy1 > cy0 and cx1 > cx0 else float(sub.mean())
    return {"top": top, "bottom": bottom, "left": left, "right": right, "center": center}


def _edge_delta_px(mask: np.ndarray, x0: int, y0: int, w: int, h: int, edge: str) -> Optional[int]:
    mh, mw = mask.shape
    x1, y1 = min(mw, x0 + w), min(mh, y0 + h)
    x0, y0 = max(0, x0), max(0, y0)
    if x1 <= x0 or y1 <= y0:
        return None
    sub = mask[y0:y1, x0:x1].astype(np.uint8)
    sh, sw = sub.shape
    thick = max(2, min(20, int(min(sh, sw) * 0.18)))
    if edge == "top":
        band = sub[:thick, :]
    elif edge == "bottom":
        band = sub[sh - thick :, :]
    elif edge == "left":
        band = sub[:, :thick]
    else:
        band = sub[:, sw - thick :]
    if band.size == 0 or float(band.mean()) < _BAND_MIN_FRAC:
        return None
    if edge in ("top", "bottom"):
        rows = np.where(band.max(axis=1) > 0)[0]
        return int(len(rows)) if len(rows) else thick
    cols = np.where(band.max(axis=0) > 0)[0]
    return int(len(cols)) if len(cols) else thick


def _human_zone(snippet: str) -> str:
    low = (snippet or "").lower()
    if "header" in low or "шапк" in low:
        return "в шапке"
    if "footer" in low or "подвал" in low:
        return "в подвале"
    if "banner" in low or "баннер" in low:
        return "в баннере"
    if "card" in low or "карточ" in low:
        return "у карточки"
    if "button" in low or "btn" in low:
        return "у кнопки"
    if "menu" in low or "меню" in low or "nav" in low:
        return "в меню"
    if re.match(r"^h[1-6]", low.split(".")[0].split("#")[0]):
        return "у заголовка"
    if low.startswith("p") or low.startswith("span"):
        return "у текста"
    if low.startswith("img") or "image" in low:
        return "у картинки"
    return ""


def _phrase_with_zone(phrase: str, snippet: str) -> str:
    zone = _human_zone(snippet)
    if zone and zone not in phrase:
        return f"{phrase} {zone}"
    return phrase


def _element_kind_label(snippet: str) -> str:
    """Короткое имя блока без CSS-селектора."""
    low = (snippet or "").lower()
    tag = low.split(".")[0].split("#")[0]
    if "button" in low or "btn" in low or tag == "button":
        return "кнопка"
    if re.match(r"^h[1-6]$", tag):
        return "заголовок"
    if "card" in low:
        return "карточка"
    if "banner" in low:
        return "баннер"
    if "header" in low or tag == "header":
        return "шапка"
    if "footer" in low:
        return "подвал"
    if "nav" in low or "menu" in low:
        return "меню"
    if tag in ("p", "span"):
        return "текст"
    if tag == "img" or "image" in low:
        return "картинка"
    if tag in ("section", "main", "article"):
        return "секция"
    if tag in ("div", "a"):
        return "блок"
    return "элемент"


def _phrase_with_element(phrase: str, snippet: str) -> str:
    """Фраза + зона/тип блока, чтобы пункты не сливались в один."""
    out = _phrase_with_zone(phrase, snippet)
    kind = _element_kind_label(snippet)
    zone = _human_zone(snippet)
    if kind and kind not in out and (not zone or kind not in zone):
        return f"{out}: {kind}"
    return out


def _pair_dedupe_key(phrase: str, el: Optional[Dict[str, Any]]) -> str:
    """Один и тот же текст на том же месте — дубль; на разных блоках — разные пункты."""
    ph = _phrase_key(phrase)
    if isinstance(el, dict):
        try:
            return f"{ph}@{int(el['x'])}:{int(el['y'])}:{int(el['w'])}:{int(el['h'])}"
        except (KeyError, TypeError, ValueError):
            sn = str(el.get("snippet", "")).strip().lower()
            if sn:
                return f"{ph}@{sn}"
    return ph


def _zone_from_bbox(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> str:
    cy = y + max(1, h) / 2.0
    cx = x + max(1, w) / 2.0
    if img_h > 0 and cy < img_h * 0.28:
        return "в верхней части"
    if img_h > 0 and cy > img_h * 0.72:
        return "в нижней части"
    if img_w > 0 and cx < img_w * 0.33:
        return "слева на странице"
    if img_w > 0 and cx > img_w * 0.66:
        return "справа на странице"
    return "по центру страницы"


def _score_all_elements_on_mask(
    mask: np.ndarray,
    layout_elements: Optional[List[Any]],
    *,
    min_frac: float = _MIN_BOX_DIFF_FRAC,
) -> List[tuple[float, str, Dict[str, Any], float]]:
    if not isinstance(layout_elements, list):
        return []
    h, w = mask.shape
    scored: List[tuple[float, str, Dict[str, Any], float]] = []
    for el in layout_elements:
        if not isinstance(el, dict):
            continue
        try:
            x, y, ew, eh = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
        except (KeyError, TypeError, ValueError):
            continue
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + ew), min(h, y + eh)
        if x1 <= x0 or y1 <= y0:
            continue
        frac = float(mask[y0:y1, x0:x1].mean())
        if frac < min_frac:
            continue
        sn = str(el.get("snippet", "")).strip() or "?"
        ch = round(frac * 100, 2)
        row = {**el, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "changed_frac_in_box_pct": ch}
        scored.append((frac, sn, row, ch))
    scored.sort(key=lambda t: -t[0])
    return scored


def _phrases_from_grid(hot: Dict[str, Any], *, max_items: int = 12) -> List[str]:
    size = hot.get("mask_size") or [1280, 720]
    try:
        img_w, img_h = int(size[0]), int(size[1])
    except (TypeError, ValueError, IndexError):
        img_w, img_h = 1280, 720
    out: List[str] = []
    for cell in hot.get("grid_cells") or []:
        if not isinstance(cell, dict):
            continue
        try:
            frac = float(cell.get("changed_frac_pct", 0) or 0)
            x, y, w, h = int(cell["x"]), int(cell["y"]), int(cell["w"]), int(cell["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if frac < 2.0:
            continue
        where = _zone_from_bbox(x, y, w, h, img_w, img_h)
        out.append(f"вёрстка не как в макете {where}")
        if len(out) >= max_items:
            break
    return out


def _phrases_for_element(
    mask: np.ndarray, el: Dict[str, Any], changed_in_box_pct: float, snippet: str = ""
) -> List[str]:
    """Все замечания по одному элементу (отступы по сторонам, шрифт, padding и т.д.)."""
    try:
        x, y, w, h = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
    except (KeyError, TypeError, ValueError):
        return []
    fr = _band_fracs(mask, x, y, w, h)
    sn = str(el.get("snippet", "")).lower()
    tag = sn.split(".")[0].split("#")[0]
    out: List[str] = []

    hot_edges: List[Tuple[str, float]] = []
    for edge in ("top", "bottom", "left", "right"):
        if fr[edge] >= _BAND_MIN_FRAC:
            hot_edges.append((edge, fr[edge]))
    hot_edges.sort(key=lambda t: -t[1])

    for edge, _ in hot_edges:
        px = _edge_delta_px(mask, x, y, w, h, edge)
        side = _EDGE_RU[edge]
        if px and px >= 4:
            out.append(_phrase_with_element(f"большой отступ {side} (~{px}px)", snippet))
        else:
            out.append(_phrase_with_element(f"отступ {side} не как в макете", snippet))

    pad = _parse_css_px_list(str(el.get("padding", "")))
    if len(pad) == 4 and max(pad) >= 12 and fr["center"] >= _BAND_MIN_FRAC:
        out.append(_phrase_with_element("слишком большой внутренний отступ", snippet))

    if fr["center"] >= 0.14 and fr["center"] > max(fr["top"], fr["bottom"], fr["left"], fr["right"]) * 1.1:
        if re.match(r"^h[1-6]$", tag) or tag in ("p", "span", "a"):
            out.append(_phrase_with_element("размер шрифта не как в макете", snippet))
        else:
            out.append(_phrase_with_element("размер блока не как в макете", snippet))

    if not out and float(changed_in_box_pct) > 3:
        out.append(_phrase_with_element("блок не как в макете", snippet))

    return out


def _compact_phrase_for_element(
    mask: np.ndarray, el: Dict[str, Any], changed_in_box_pct: float, snippet: str = ""
) -> Optional[str]:
    phrases = _phrases_for_element(mask, el, changed_in_box_pct, snippet)
    return phrases[0] if phrases else None


def _phrase_key(p: str) -> str:
    return re.sub(r"\s+", " ", p.lower().strip())


def _dedupe_phrases(phrases: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for p in phrases:
        k = _phrase_key(p)
        if not k or k in seen:
            continue
        dup = False
        for prev in list(seen):
            if k in prev or prev in k:
                if len(k) >= len(prev) * 0.75:
                    dup = True
                    break
        if dup:
            continue
        seen.add(k)
        out.append(p.strip())
    return out


def _to_bullet_lines(phrases: List[str]) -> List[str]:
    lines: List[str] = []
    for p in phrases:
        t = p.strip().lstrip("-• ").strip()
        if t:
            lines.append("- " + t)
    return lines


def _el_bbox_dict(el: Dict[str, Any]) -> Optional[Dict[str, int]]:
    try:
        x, y, w, h = int(el["x"]), int(el["y"]), int(el["w"]), int(el["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def is_broken_bug_line(line: str) -> bool:
    """Обрывки, латиница внутри русских слов (артефакт LLM), слишком короткие строки."""
    t = (line or "").strip().lstrip("-• ").strip()
    if len(t) < 12:
        return True
    words = re.findall(r"\S+", t)
    if not words:
        return True
    last = words[-1]
    if len(last) <= 5 and not re.search(r"[.!?…]$", last):
        if last.lower() not in ("макете", "шапке", "кнопке", "текста", "баннере", "меню"):
            return True
    for word in words:
        if re.search(r"[a-zA-Z]", word) and re.search(r"[а-яА-ЯёЁ]", word):
            return True
    return False


def _phrase_dedupe_key(p: str) -> str:
    k = _phrase_key(p)
    k = re.sub(r"[a-zA-Z]+", "", k)
    k = re.sub(r"\s+", " ", k).strip()
    return k


def sanitize_bug_lines(lines: List[str]) -> List[str]:
    """Фильтр мусора и похожих дублей (в т.ч. «Кnoпка» / «Кнопка»)."""
    out: List[str] = []
    seen: set[str] = set()
    for raw in lines:
        s = str(raw).strip().lstrip("-• ").strip()
        if not s or is_broken_bug_line(s) or is_legacy_verbose_bug_line(s):
            continue
        k = _phrase_dedupe_key(s)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _collect_phrases_with_elements(
    hot: Dict[str, Any],
    layout_elements: Optional[List[Any]],
    mask: np.ndarray,
    *,
    baseline_rgb: Optional[np.ndarray] = None,
    current_rgb: Optional[np.ndarray] = None,
    max_lines: int = DEFAULT_MAX_BUG_LINES,
) -> List[tuple[str, Optional[Dict[str, Any]]]]:
    scored = _score_all_elements_on_mask(mask, layout_elements)
    if not scored:
        el_index = {str(e.get("snippet", "")): e for e in (layout_elements or []) if isinstance(e, dict)}
        for ov in hot.get("elements_overlap") or []:
            if not isinstance(ov, dict):
                continue
            sn = str(ov.get("snippet", "")).strip()
            try:
                ch = float(ov.get("changed_frac_in_box_pct", 0) or 0)
                frac = ch / 100.0
            except (TypeError, ValueError):
                ch, frac = 0.0, 0.0
            el = el_index.get(sn) or ov
            scored.append((frac, sn, el, ch))

    pairs: List[tuple[str, Optional[Dict[str, Any]]]] = []
    has_rgb = (
        baseline_rgb is not None
        and current_rgb is not None
        and baseline_rgb.shape[:2] == current_rgb.shape[:2]
    )
    for _frac, sn, el, ch in scored:
        if has_rgb:
            for p in _presence_phrases_for_element(baseline_rgb, current_rgb, el, snippet=sn):
                pairs.append((p, el))
        for p in _phrases_for_element(mask, el, ch, snippet=sn):
            pairs.append((p, el))

    for cell in hot.get("grid_cells") or []:
        if not isinstance(cell, dict):
            continue
        try:
            frac = float(cell.get("changed_frac_pct", 0) or 0)
            x, y, w, h = int(cell["x"]), int(cell["y"]), int(cell["w"]), int(cell["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if frac < 1.5:
            continue
        size = hot.get("mask_size") or [1280, 720]
        img_w, img_h = int(size[0]), int(size[1])
        where = _zone_from_bbox(x, y, w, h, img_w, img_h)
        el_cell = {"x": x, "y": y, "w": w, "h": h, "snippet": f"zone:{where}"}
        if has_rgb:
            for p in _presence_phrases_for_bbox(baseline_rgb, current_rgb, x, y, w, h, f"zone:{where}"):
                pairs.append((p, el_cell))
        text = f"вёрстка не как в макете {where}"
        pairs.append((text, el_cell))

    if not pairs:
        for p in _phrases_from_grid(hot):
            pairs.append((p, None))

    deduped: List[tuple[str, Optional[Dict[str, Any]]]] = []
    seen: set[str] = set()
    for phrase, el in pairs:
        k = _pair_dedupe_key(phrase, el)
        if not k or k in seen or is_broken_bug_line(phrase):
            continue
        seen.add(k)
        deduped.append((phrase.strip(), el))
        if max_lines > 0 and len(deduped) >= max_lines:
            break
    return deduped


def build_bug_report_items(
    hot: Dict[str, Any],
    layout_elements: Optional[List[Any]],
    *,
    baseline_path: Optional[str] = None,
    current_path: Optional[str] = None,
    pixel_threshold: int = 30,
    tolerance_shift_px: int = 2,
    tolerance_speckle_iter: int = 1,
    max_lines: int = DEFAULT_MAX_BUG_LINES,
) -> List[Dict[str, Any]]:
    """Пункты баг-репорта с привязкой к bbox для миниатюр в HTML."""
    mask: Optional[np.ndarray] = None
    baseline_rgb: Optional[np.ndarray] = None
    current_rgb: Optional[np.ndarray] = None
    if baseline_path and current_path:
        baseline_rgb, current_rgb = _load_aligned_rgb_pair(baseline_path, current_path)
        try:
            mask, _ = build_change_mask(
                baseline_path,
                current_path,
                pixel_threshold=pixel_threshold,
                tolerance_shift_px=tolerance_shift_px,
                tolerance_speckle_iter=tolerance_speckle_iter,
            )
        except OSError:
            mask = None
    if mask is None:
        return [{"text": "Нет данных diff — повторите прогон"}]

    items: List[Dict[str, Any]] = []
    for phrase, el in _collect_phrases_with_elements(
        hot,
        layout_elements,
        mask,
        baseline_rgb=baseline_rgb,
        current_rgb=current_rgb,
        max_lines=max_lines,
    ):
        if is_broken_bug_line(phrase):
            continue
        row: Dict[str, Any] = {"text": phrase}
        if isinstance(el, dict):
            sn = str(el.get("snippet", "")).strip()
            if sn:
                row["snippet"] = sn
            bb = _el_bbox_dict(el)
            if bb:
                row.update(bb)
        items.append(row)
    return group_bug_items_by_region(sanitize_bug_items(items))


def _region_group_key(it: Dict[str, Any]) -> str:
    try:
        return f"{int(it['x'])}:{int(it['y'])}:{int(it['w'])}:{int(it['h'])}"
    except (KeyError, TypeError, ValueError):
        return str(it.get("snippet", "")).strip().lower() or _phrase_key(str(it.get("text", "")))


def _bug_phrase_core(text: str) -> str:
    """Короткая часть формулировки без повторяющегося «: кнопка» в конце."""
    t = (text or "").strip()
    if ":" in t:
        left, right = t.rsplit(":", 1)
        if right.strip().lower() in left.lower():
            return left.strip()
    return t


def merge_bug_texts_comma(texts: List[str]) -> str:
    parts: List[str] = []
    seen: set[str] = set()
    for raw in texts:
        core = _bug_phrase_core(str(raw).strip())
        if not core:
            continue
        k = _phrase_key(core)
        if k in seen:
            continue
        seen.add(k)
        parts.append(core)
    return ", ".join(parts)


def group_bug_items_by_region(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Несколько багов одного блока (один кроп) — одна строка, баги через запятую."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        k = _region_group_key(it)
        if k not in buckets:
            buckets[k] = []
            order.append(k)
        buckets[k].append(it)
    out: List[Dict[str, Any]] = []
    for k in order:
        group = buckets[k]
        texts = [str(x.get("text", "")).strip() for x in group if str(x.get("text", "")).strip()]
        if not texts:
            continue
        row = dict(group[0])
        row["text"] = merge_bug_texts_comma(texts)
        row["bugs"] = texts
        out.append(row)
    return out


def sanitize_bug_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for it in items:
        text = str(it.get("text", "")).strip()
        if not text or is_broken_bug_line(text) or is_legacy_verbose_bug_line(text):
            continue
        k = _pair_dedupe_key(text, it)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def build_bug_lines_from_hotspots(
    hot: Dict[str, Any],
    layout_elements: Optional[List[Any]],
    *,
    baseline_path: Optional[str] = None,
    current_path: Optional[str] = None,
    pixel_threshold: int = 30,
    tolerance_shift_px: int = 2,
    tolerance_speckle_iter: int = 1,
    max_lines: int = DEFAULT_MAX_BUG_LINES,
) -> List[str]:
    try:
        global_cr = float(hot.get("changed_pixels_pct", 0) or 0)
    except (TypeError, ValueError):
        global_cr = 0.0

    mask: Optional[np.ndarray] = None
    if baseline_path and current_path:
        try:
            mask, _ = build_change_mask(
                baseline_path,
                current_path,
                pixel_threshold=pixel_threshold,
                tolerance_shift_px=tolerance_shift_px,
                tolerance_speckle_iter=tolerance_speckle_iter,
            )
        except OSError:
            mask = None

    if mask is None:
        return _to_bullet_lines(["Нет данных diff — повторите прогон"])

    pairs = _collect_phrases_with_elements(hot, layout_elements, mask, max_lines=max_lines)
    phrases = [p for p, _ in pairs]
    if not phrases:
        return _to_bullet_lines(["Отличия по diff слабые — проверьте вручную по карте diff"])

    cut = phrases if max_lines <= 0 else phrases[:max_lines]
    lines = _to_bullet_lines(cut)
    if global_cr >= _FRAME_HINT_ONLY_IF_EMPTY_PCT and len(phrases) >= 3:
        return lines
    if global_cr >= _FRAME_HINT_ONLY_IF_EMPTY_PCT:
        lines.append(
            "- При очень большом diff дополнительно проверьте window_size и figma.scale"
        )
    return lines


def draft_lines_to_text(lines: List[str]) -> str:
    return "\n".join(lines) + ("\n" if lines else "")


def is_legacy_verbose_bug_line(line: str) -> bool:
    low = (line or "").lower()
    if "явно не совпадает" in low or "сравни с макетом размер текста" in low:
        return True
    if re.search(r"\b(div|span|main|header|footer|section|picture|a|h[1-6]|img)\.[a-z0-9_-]+\s*—", low):
        return True
    if low.count("отступ") >= 3 and "не совпадает с макетом" in low:
        return True
    if "window_size" in low and "figma.scale" in low and "выровн" in low:
        return True
    if "не мелкие правки" in low:
        return True
    if "кадр или масштаб" in low and "макет" in low:
        return True
    if "выровнять размер окна" in low or ("window_size" in low and "figma.scale" in low):
        return True
    if "повторить сверку" in low and "кадр" in low:
        return True
    return False
