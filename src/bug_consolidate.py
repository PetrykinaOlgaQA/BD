"""
Сжатие и приоритизация пунктов баг-репорта: убираем дубли, сохраняем текст + эмодзи + цифры.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_CARD_NUM = re.compile(r"карточка\s*(\d+)", re.I)
_STAT_M = re.compile(r"\d+\s*M\+", re.I)


def _group_key(it: Dict[str, Any]) -> str:
    text = str(it.get("text", "")).lower()
    sec = str(it.get("section", "")).strip().lower()
    if sec.startswith("fact-card-"):
        return sec
    m = _CARD_NUM.search(text)
    if m:
        return f"fact-card-{m.group(1)}"
    if "блок статистики" in text or "статистик" in text or sec == "stats":
        return "stats"
    if "шапк" in text or sec == "header" or "header" in str(it.get("snippet", "")).lower():
        return "header"
    if "подвал" in text or sec == "footer":
        return "footer"
    try:
        return f"bbox:{int(it['x'])}:{int(it['y'])}:{int(it['w'])}:{int(it['h'])}"
    except (KeyError, TypeError, ValueError):
        return text[:80]


def _bug_category(text: str) -> str:
    low = (text or "").lower()
    if "визуально отличается" in low or "блок отличается" in low:
        return "noise"
    if "цифра:" in low or "текстовка:" in low or _STAT_M.search(text or ""):
        return "digit"
    if "логотип" in low or "котик" in low:
        return "emoji"
    if "эмодзи" in low or ("иконка" in low and ("стил" in low or "цвет" in low or "размер" in low)):
        return "emoji"
    if "текст" in low and ("«" in text or "→" in text or "≠" in text or "макет" in low):
        return "text"
    if "текстовк" in low or "текст не совпадает" in low or "текст отличается" in low:
        return "text"
    if "заголовок" in low and ("«" in text or "→" in text):
        return "text"
    if "шрифт" in low:
        return "font"
    if "цвет" in low and "текст" in low:
        return "color"
    if "не совпадает с макетом" in low:
        return "noise"
    return "other"


def _priority(it: Dict[str, Any]) -> Tuple[int, float]:
    text = str(it.get("text", "")).lower()
    try:
        diff_m = re.search(r"diff\s*([\d.]+)\s*%", text, re.I)
        diff_val = float(diff_m.group(1)) if diff_m else 0.0
    except (TypeError, ValueError):
        diff_val = 0.0

    cat = _bug_category(text)
    cat_prio = {
        "digit": 0,
        "text": 1,
        "emoji": 2,
        "font": 3,
        "color": 4,
        "other": 5,
        "noise": 9,
    }.get(cat, 6)
    return (cat_prio, -diff_val)


def _is_noise(text: str) -> bool:
    return _bug_category(text) == "noise"


def _pick_stats_group(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """В статистике сохраняем несколько текстовок (разные карточки), без шумовых «визуально»."""
    digits: List[Dict[str, Any]] = []
    rest: List[Dict[str, Any]] = []
    for it in group:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text", "")).strip()
        if not text or _is_noise(text):
            continue
        if "текстовка:" in text.lower() or _bug_category(text) == "digit":
            digits.append(it)
        else:
            rest.append(it)
    digits.sort(key=_priority)
    seen_pos: set[Tuple[int, int]] = set()
    picked_digits: List[Dict[str, Any]] = []
    for it in digits:
        try:
            pos = (int(it["x"]) // 24, int(it["y"]) // 24)
        except (KeyError, TypeError, ValueError):
            picked_digits.append(it)
            continue
        if pos in seen_pos:
            continue
        seen_pos.add(pos)
        picked_digits.append(it)
    picked_digits.sort(key=lambda it: (int(it.get("x", 0)), int(it.get("y", 0))))
    return picked_digits + _pick_best_per_category(rest)


def _pick_best_per_category(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Лучший пункт в каждой категории (цифра, текст, эмодзи — не схлопываем в один)."""
    buckets: Dict[str, Dict[str, Any]] = {}
    for it in group:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text", "")).strip()
        if not text or _is_noise(text):
            continue
        cat = _bug_category(text)
        prev = buckets.get(cat)
        if prev is None or _priority(it) < _priority(prev):
            buckets[cat] = it
    order = ["digit", "text", "emoji", "font", "color", "other"]
    out: List[Dict[str, Any]] = []
    for cat in order:
        if cat in buckets:
            out.append(buckets[cat])
    return out


def consolidate_bug_report_items(
    items: List[Dict[str, Any]],
    *,
    layout_elements: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    if not items:
        return items

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        k = _group_key(it)
        if k not in buckets:
            buckets[k] = []
            order.append(k)
        buckets[k].append(it)

    out: List[Dict[str, Any]] = []
    for k in order:
        group = buckets[k]
        if not group:
            continue
        if k == "footer":
            continue
        if k == "stats":
            out.extend(_pick_stats_group(group))
            continue
        picked = _pick_best_per_category(group)
        if k == "header" and picked:
            logos = [
                it
                for it in picked
                if "логотип" in str(it.get("text", "")).lower()
                or "котик" in str(it.get("text", "")).lower()
                or "logo" in str(it.get("snippet", "")).lower()
            ]
            if logos:
                out.append(min(logos, key=_priority))
            else:
                clean = [
                    it
                    for it in picked
                    if _bug_category(str(it.get("text", ""))) != "noise"
                ]
                if clean:
                    out.append(min(clean, key=_priority))
            continue
        out.extend(picked)

    out.sort(
        key=lambda it: (
            0 if "статистик" in str(it.get("text", "")).lower() else 1,
            0 if "логотип" in str(it.get("text", "")).lower() else 1,
            _priority(it)[0],
            int(it.get("y", 0) or 0),
            int(it.get("x", 0) or 0),
        )
    )
    return out


def drop_redundant_nn_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return items
    has_heuristic = any(
        not str(it.get("text", "")).strip().startswith("[NN]") for it in items if isinstance(it, dict)
    )
    has_logo_heuristic = any(
        "логотип" in str(it.get("text", "")).lower()
        or "котик" in str(it.get("text", "")).lower()
        for it in items
        if isinstance(it, dict) and not str(it.get("text", "")).strip().startswith("[NN]")
    )
    if not has_heuristic:
        return items
    out: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        t = str(it.get("text", "")).strip()
        if t.startswith("[NN]"):
            sn = str(it.get("snippet", "")).lower()
            drop_snippets = (
                "fact-card",
                "facts-grid",
                "stats",
                "stats-grid",
                "container",
                "header",
            )
            if any(x in sn for x in drop_snippets):
                continue
            if "logo" in sn and has_logo_heuristic:
                continue
        out.append(it)
    return out


def finalize_bug_report_items(
    items: List[Dict[str, Any]],
    *,
    layout_elements: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    items = consolidate_bug_report_items(items, layout_elements=layout_elements)
    items = drop_redundant_nn_items(items)
    return items


def find_fact_emoji_element(
    card_index: int,
    layout_elements: Optional[List[Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(layout_elements, list):
        return None
    emojis: List[Dict[str, Any]] = []
    for el in layout_elements:
        if not isinstance(el, dict):
            continue
        sn = str(el.get("snippet", "")).lower()
        if "fact-emoji" not in sn and "emoji" not in sn:
            continue
        try:
            if int(el.get("w", 0)) < 8 or int(el.get("h", 0)) < 8:
                continue
        except (TypeError, ValueError):
            continue
        emojis.append(el)
    emojis.sort(key=lambda e: (int(e.get("y", 0)), int(e.get("x", 0))))
    idx = max(0, int(card_index) - 1)
    if idx < len(emojis):
        return emojis[idx]
    return None
