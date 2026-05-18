from __future__ import annotations

import html
import json
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.block_crops import (
    element_bbox,
    find_element_for_bug_item,
    image_size,
    refine_bug_table_bbox,
    save_highlight_crop,
    save_plain_crop,
    scale_bbox,
)
from src.bug_reports import is_broken_bug_line, is_legacy_verbose_bug_line, sanitize_bug_lines
from src.utils import CAPTURE_WAIT_TYPING_OK_SEC, stats_capture_wait_seconds


def append_text_report(
    reports_dir: str,
    lines: list[str],
    basename: str = "runs",
) -> str:
    os.makedirs(reports_dir, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = os.path.join(reports_dir, f"{basename}_{day}.txt")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    block = ["", "=" * 60, stamp, *lines, ""]
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(block))
    return path


def write_json_sidecar(path_txt: str, payload: Dict[str, Any]) -> str:
    path_json = os.path.splitext(path_txt)[0] + "_last.json"
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path_json


def _fragment_match_stats_cell(stats: Dict[str, Any]) -> str:
    if not stats.get("fragment_matcher_used"):
        return "выкл."
    scored = int(stats.get("fragment_match_scored", 0) or 0)
    filt = int(stats.get("fragment_match_filtered", 0) or 0)
    thr = stats.get("fragment_match_threshold", "—")
    return f"вкл., оценено {scored}, отфильтровано {filt}, порог P(same)≥{thr}"


def _parse_structured_bug_blocks(markdown: str) -> List[Dict[str, str]]:
    """Формат **Блок:** / **Раздел:** / **Суть бага:** (повторяется для каждого бага)."""
    pat = re.compile(
        r"(?is)\*\*Блок:\*\*\s*(?P<block>[^\n]+?)\s*"
        r"\*\*Раздел:\*\*\s*(?P<section>[^\n]+?)\s*"
        r"\*\*Суть бага:\*\*\s*(?P<gist>.+?)(?=\n\s*\*\*Блок:\*\*|\Z)"
    )
    rows: List[Dict[str, str]] = []
    for m in pat.finditer(markdown):
        rows.append(
            {
                "id": str(len(rows) + 1),
                "блок": m.group("block").strip(),
                "раздел": m.group("section").strip(),
                "суть": re.sub(r"\s+", " ", m.group("gist").strip()),
            }
        )
    return rows


def _extract_section(markdown: str, titles: tuple[str, ...]) -> Optional[str]:
    """Тело первого раздела ### Title до следующего ### или конца."""
    for title in titles:
        m = re.search(
            rf"(?ms)^###\s*{re.escape(title)}\s*\n(.*?)(?=^###\s|\Z)",
            markdown,
        )
        if m:
            return m.group(1).strip()
    return None


def _extract_section_heading_contains(markdown: str, substring: str) -> Optional[str]:
    """### любой заголовок, содержащий substring (без учёта регистра)."""
    m = re.search(
        rf"(?ms)^###\s*[^\n]*{re.escape(substring)}[^\n]*\s*\n(.*?)(?=^###\s|\Z)",
        markdown,
        flags=re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


def _normalize_rec_key(s: str) -> str:
    """Ключ для сравнения строк (без регистра и лишних пробелов)."""
    t = re.sub(r"\s+", " ", (s or "").lower().strip())
    t = re.sub(r"[«»\"'`]", "", t)
    t = re.sub(r"^[-*•]\s*", "", t)
    return t


def dedupe_recommendation_lines(lines: List[str]) -> List[str]:
    """Убирает дубли; для одного элемента объединяет формулировки через запятую."""
    seen: set[str] = set()
    out: List[str] = []
    by_element: Dict[str, List[str]] = {}
    for raw in lines:
        s = str(raw).strip()
        if (
            not s
            or _is_garbage_recommendation_line(s)
            or is_legacy_verbose_bug_line(s)
            or is_broken_bug_line(s)
        ):
            continue
        if " — " in s:
            el, rest = s.split(" — ", 1)
            el_k = el.strip()
            parts = [p.strip() for p in rest.split(",") if p.strip()]
            bucket = by_element.setdefault(el_k, [])
            for p in parts:
                pk = _normalize_rec_key(p)
                if pk and pk not in {_normalize_rec_key(x) for x in bucket}:
                    bucket.append(p)
            continue
        key = _normalize_rec_key(s)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    for el, parts in by_element.items():
        if not parts:
            continue
        line = f"{el} — {', '.join(parts)}"
        key = _normalize_rec_key(line)
        if key not in seen:
            seen.add(key)
            out.append(line)
    return out


def _is_garbage_recommendation_line(s: str) -> bool:
    low = s.lower()
    needles = (
        "строка из порядка",
        "строка для поиска",
        "нулевого порядка",
        "вхождений",
        "format_contract",
        "<instructions>",
        "<data>",
        "json (метрики",
    )
    return any(n in low for n in needles)


def parse_recommendation_lines(markdown: str) -> List[str]:
    """
    Строки для таблицы «Рекомендации»: раздел ### Рекомендации по правкам,
    маркеры - / * / 1. … При отсутствии — не подставляем обрезок всего текста.

    Если весь ответ модели — только маркированный список (новый контракт explain_diff_ru),
    собираем все строки «- …» без секции ###.
    """
    if not (markdown or "").strip():
        return []

    stripped = markdown.strip()
    if "###" not in stripped:
        out_plain: List[str] = []
        for line in stripped.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith(("---", "***", "```")):
                continue
            mo = re.match(r"^[-*•]\s*(.+)$", s)
            if mo:
                item = mo.group(1).strip()
                if not _is_garbage_recommendation_line(item):
                    out_plain.append(item)
        if out_plain:
            return dedupe_recommendation_lines(out_plain)

    body = _extract_section(
        markdown,
        ("Рекомендации по правкам", "Рекомендации"),
    )
    if not body:
        body = _extract_section_heading_contains(markdown, "Рекомендации")
    lines_out: List[str] = []
    if body:
        for line in body.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith(("---", "***", "```")):
                continue
            # Маркированный или нумерованный список
            mo = re.match(r"^[-*•]\s*(.+)$", s)
            if mo:
                item = mo.group(1).strip()
                if not _is_garbage_recommendation_line(item):
                    lines_out.append(item)
                continue
            mo = re.match(r"^\d{1,2}[\.\)]\s*(.+)$", s)
            if mo:
                item = mo.group(1).strip()
                if not _is_garbage_recommendation_line(item):
                    lines_out.append(item)
                continue
        if lines_out:
            return dedupe_recommendation_lines(lines_out)

    # Модель могла вставить заголовок без перевода строки или список ниже без явного тела
    m = re.search(
        r"(?is)###\s*[^\n]*Рекомендации[^\n]*\n+((?:^\s*[-*•]\s+.+\n?)+)",
        markdown,
    )
    if m:
        for line in m.group(1).splitlines():
            mo = re.match(r"^\s*[-*•]\s+(.+)$", line.strip())
            if mo:
                item = mo.group(1).strip()
                if not _is_garbage_recommendation_line(item):
                    lines_out.append(item)
        if lines_out:
            return dedupe_recommendation_lines(lines_out)

    body2 = _extract_section(markdown, ("Что сделать в первую очередь",))
    if body2:
        # Одна строка абзаца → одна правка
        flat = re.sub(r"\s+", " ", body2).strip()
        if flat and len(flat) > 12:
            return [flat[:500] + ("…" if len(flat) > 500 else "")]

    structured = _parse_structured_bug_blocks(markdown)
    if structured:
        return dedupe_recommendation_lines(
            [
                f"{r.get('блок', '—')}: {r.get('суть', r.get('раздел', ''))}".strip(": ")
                for r in structured[:20]
            ]
        )

    return []


def fallback_recommendation_lines_from_stats(stats: Dict[str, Any]) -> List[str]:
    """Если модель не дала списка — даём осмысленные строки из метрик (не заглушку про «раздел не найден»)."""
    try:
        cr = float(stats.get("changed_ratio_pct", 0) or 0)
    except (TypeError, ValueError):
        cr = 0.0
    out: List[str] = []
    raw_tasks = stats.get("diff_hotspots_tasks")
    if isinstance(raw_tasks, list):
        for t in raw_tasks:
            s = str(t).strip().lstrip("-• ").strip()
            if s and not is_legacy_verbose_bug_line(s):
                out.append(s)
            if len(out) >= 16:
                break
    out = dedupe_recommendation_lines(out)
    if not out and cr >= 55:
        out.append("Сильное расхождение с макетом — проверьте window_size и figma.scale, затем повторите сверку")
    return out


def _parse_legacy_numbered_bugs(body: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        mo = re.match(r"^(\d+)[\.\)]\s+(.+)$", line)
        if mo:
            rows.append(
                {
                    "id": mo.group(1),
                    "блок": "—",
                    "раздел": "—",
                    "суть": mo.group(2).strip(),
                }
            )
            continue
        if line.startswith(("- ", "* ")):
            rows.append(
                {
                    "id": str(len(rows) + 1),
                    "блок": "—",
                    "раздел": "—",
                    "суть": line[2:].strip(),
                }
            )
    return rows


def parse_gemma_bugs(markdown: str) -> List[Dict[str, str]]:
    """Устаревший формат таблицы (Блок/Раздел); для HTML используйте parse_recommendation_lines."""
    recs = parse_recommendation_lines(markdown)
    if recs:
        return [{"id": str(i + 1), "блок": "—", "раздел": "—", "суть": t} for i, t in enumerate(recs)]
    if not (markdown or "").strip():
        return [
            {
                "id": "—",
                "блок": "—",
                "раздел": "—",
                "суть": "Нет ответа модели (проверьте Ollama).",
            }
        ]
    structured = _parse_structured_bug_blocks(markdown)
    if structured:
        return structured

    for title in ("Вероятные баги", "Баги (структурированно)", "Баги"):
        m = re.search(
            rf"^###\s*{re.escape(title)}\s*([\s\S]*?)(?=^###\s|\Z)",
            markdown,
            re.MULTILINE,
        )
        if m:
            body = m.group(1).strip()
            rows = _parse_legacy_numbered_bugs(body)
            if rows:
                return rows

    return [
        {
            "id": "1",
            "блок": "—",
            "раздел": "—",
            "суть": "Полный текст ответа модели — ниже на странице (раздел «Рекомендации по правкам» не найден).",
        }
    ]


def _format_rec_line_html(s: str) -> str:
    """Экранирование + безопасное жирное **…** (как просит модель)."""
    e = html.escape(s)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", e)


def _asset_href(reports_dir: str, abs_path: str) -> str:
    rel = os.path.relpath(abs_path, reports_dir).replace("\\", "/")
    return html.escape(rel, quote=True)


def _crop_cell_html(reports_dir: str, src: str, alt: str) -> str:
    if not src:
        return '<span class="meta">—</span>'
    rel = _asset_href(reports_dir, src)
    return (
        f'<figure class="bug-shot"><img src="{rel}" loading="lazy" '
        f'alt="{html.escape(alt)}" /></figure>'
    )


def _bug_row_label(bug_item: Dict[str, Any]) -> str:
    from src.bug_reports import _element_kind_label

    sn = str(bug_item.get("snippet", "")).strip()
    if sn.startswith("zone:"):
        return sn.replace("zone:", "").strip() or "зона"
    if sn and not sn.startswith("?"):
        kind = _element_kind_label(sn)
        return kind if kind != "элемент" else "блок"
    try:
        y = int(bug_item.get("y", 0))
    except (TypeError, ValueError):
        y = 0
    if y < 200:
        return "верх страницы"
    if y > 500:
        return "низ страницы"
    return "центр"


def _build_bug_report_table_html(
    reports_dir: str,
    stamp: str,
    stats: Dict[str, Any],
    baseline_path: str,
    current_shot: str,
    recommendations: List[str],
    *,
    diff_path: Optional[str] = None,
) -> str:
    """Таблица: ожидаемый (Figma) | фактический (сайт) | баги через запятую."""
    els = stats.get("layout_elements_for_crops")
    if not isinstance(els, list):
        els = []
    raw_items = stats.get("bug_report_items")
    rows_data: List[Dict[str, Any]] = []
    if isinstance(raw_items, list) and raw_items:
        rows_data = [x for x in raw_items if isinstance(x, dict) and str(x.get("text", "")).strip()]
    if not rows_data and recommendations:
        rows_data = [{"text": t} for t in recommendations if str(t).strip()]

    def _sort_key(it: Dict[str, Any]) -> tuple[int, int]:
        try:
            return (int(it.get("y", 0)), int(it.get("x", 0)))
        except (TypeError, ValueError):
            return (0, 0)

    rows_data.sort(key=_sort_key)

    crop_dir = os.path.join(reports_dir, f"bug_table_{stamp}")
    os.makedirs(crop_dir, exist_ok=True)
    ref_wh: Optional[tuple[int, int]] = None
    layout = stats.get("layout_site") if isinstance(stats.get("layout_site"), dict) else {}
    vp = layout.get("viewport") if isinstance(layout, dict) else None
    if isinstance(vp, dict):
        try:
            ref_wh = (int(vp["w"]), int(vp["h"]))
        except (KeyError, TypeError, ValueError):
            ref_wh = None
    if ref_wh is None:
        cur_sz = image_size(current_shot)
        if cur_sz:
            ref_wh = cur_sz
    base_sz = image_size(baseline_path) if baseline_path else None
    cur_sz = image_size(current_shot) if current_shot else None
    diff_sz = image_size(diff_path) if diff_path else None
    hotspots = stats.get("diff_hotspots") if isinstance(stats.get("diff_hotspots"), dict) else {}
    parts: List[str] = []
    for i, bug_item in enumerate(rows_data[:60]):
        t = str(bug_item.get("text", "")).strip()
        if not t:
            continue
        el = find_element_for_bug_item(t, els, bug_item)
        bbox = element_bbox(el) if el else None
        if not bbox and isinstance(bug_item, dict):
            try:
                bbox = (
                    int(bug_item["x"]),
                    int(bug_item["y"]),
                    int(bug_item["w"]),
                    int(bug_item["h"]),
                )
            except (KeyError, TypeError, ValueError):
                bbox = None

        bbox = refine_bug_table_bbox(bbox, bug_item, hotspots, ref_wh=ref_wh)

        exp_src = ""
        act_src = ""
        diff_src = ""
        if bbox:
            exp_png = os.path.join(crop_dir, f"exp_{i}.png")
            act_png = os.path.join(crop_dir, f"act_{i}.png")
            diff_png = os.path.join(crop_dir, f"diff_{i}.png")
            bbox_page = bbox
            bbox_base = bbox
            bbox_cur = bbox
            bbox_diff = bbox
            if ref_wh:
                if base_sz and base_sz != ref_wh:
                    bbox_base = scale_bbox(bbox_page, ref_wh, base_sz)
                if cur_sz and cur_sz != ref_wh:
                    bbox_cur = scale_bbox(bbox_page, ref_wh, cur_sz)
                if diff_sz and diff_sz != ref_wh:
                    bbox_diff = scale_bbox(bbox_page, ref_wh, diff_sz)
            if baseline_path and os.path.isfile(baseline_path):
                if save_highlight_crop(
                    baseline_path,
                    bbox_base,
                    exp_png,
                    outline="#38bdf8",
                    width=3,
                ):
                    exp_src = exp_png
            if current_shot and os.path.isfile(current_shot):
                if save_highlight_crop(current_shot, bbox_cur, act_png):
                    act_src = act_png
            if diff_path and os.path.isfile(diff_path):
                if save_plain_crop(diff_path, bbox_diff, diff_png):
                    diff_src = diff_png

        label = html.escape(_bug_row_label(bug_item))
        bug_html = _format_rec_line_html(t)
        p_same = bug_item.get("fragment_match_p_same")
        if p_same is not None:
            try:
                ps = float(p_same)
                extra = ""
                st = bug_item.get("fragment_match_structure")
                ct = bug_item.get("fragment_match_content")
                if st is not None and ct is not None:
                    extra = f" · S={float(st):.2f} C={float(ct):.2f}"
                bug_html += (
                    f'<br><small class="meta">P(совпадение): {ps:.2f}{extra}'
                    f" <span title=\"Structure×0.7 + Content×0.3, стили не учитываются\">"
                    f"(семантика)</span></small>"
                )
            except (TypeError, ValueError):
                pass
        parts.append(
            "    <tr>"
            f'<td class="bug-num">{i + 1}</td>'
            f'<td class="bug-zone"><small>{label}</small></td>'
            f'<td class="bug-img">{_crop_cell_html(reports_dir, exp_src, "Ожидаемый (макет)")}</td>'
            f'<td class="bug-img">{_crop_cell_html(reports_dir, act_src, "Фактический (сайт)")}</td>'
            f'<td class="bug-img">{_crop_cell_html(reports_dir, diff_src, "Карта отличий в этой зоне")}</td>'
            f'<td class="bug-desc">{bug_html}</td>'
            "</tr>\n"
        )
    if not parts:
        try:
            pct = float(stats.get("changed_ratio_pct", 100) or 100)
        except (TypeError, ValueError):
            pct = 100.0
        if pct < 2.0:
            return (
                '    <tr><td colspan="6">Замечаний не найдено — страница совпадает с макетом '
                "в пределах порога diff.</td></tr>\n"
            )
        return '    <tr><td colspan="6">Нет данных по diff</td></tr>\n'
    return "".join(parts)


def write_html_report(
    reports_dir: str,
    *,
    site_url: str,
    figma_url: str,
    ok: bool,
    stats: Dict[str, Any],
    gemma_markdown: str,
    baseline_path: str,
    current_shot: str,
    diff_path: Optional[str],
) -> str:
    """Одностраничный отчёт: скриншоты (эталон Figma, затем страница), метрики, список правок; без гиперссылок на Figma/сайт/diff."""
    os.makedirs(reports_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(reports_dir, f"qa_report_{stamp}.html")
    last_path = os.path.join(reports_dir, "qa_report_last.html")
    media_dir = os.path.join(reports_dir, f"qa_report_{stamp}_media")

    def _stage_asset(ap: str) -> str:
        if not ap or not os.path.isfile(ap):
            return ap
        os.makedirs(media_dir, exist_ok=True)
        dest = os.path.join(media_dir, os.path.basename(ap))
        if not os.path.isfile(dest):
            shutil.copy2(ap, dest)
        return dest

    baseline_path = _stage_asset(baseline_path)
    current_shot = _stage_asset(current_shot)
    if diff_path:
        diff_path = _stage_asset(diff_path)
    recommendations = sanitize_bug_lines(
        dedupe_recommendation_lines(parse_recommendation_lines(gemma_markdown))
    )
    raw_items = stats.get("bug_report_items")
    if isinstance(raw_items, list) and raw_items:
        from src.bug_reports import sanitize_bug_items

        recommendations = [
            str(it.get("text", "")).strip()
            for it in sanitize_bug_items([x for x in raw_items if isinstance(x, dict)])
            if str(it.get("text", "")).strip()
        ]
    layout = stats.get("layout_site") or {}
    elements = layout.get("elements") if isinstance(layout, dict) else None
    rows_layout = ""
    if isinstance(elements, list):
        for el in elements[:24]:
            if not isinstance(el, dict):
                continue
            rows_layout += (
                "<tr><td><code>"
                + html.escape(str(el.get("snippet", "")), quote=False)
                + "</code></td>"
                f"<td>{html.escape(str(el.get('x', '')))}</td>"
                f"<td>{html.escape(str(el.get('y', '')))}</td>"
                f"<td>{html.escape(str(el.get('w', '')))}</td>"
                f"<td>{html.escape(str(el.get('h', '')))}</td>"
                "<td><small>"
                + html.escape(str(el.get("fontSize", "")), quote=False)
                + " · "
                + html.escape(str(el.get("fontFamily", "")), quote=False)[:40]
                + "</small></td>"
                "<td><small>"
                + html.escape(str(el.get("color", "")), quote=False)
                + "</small></td>"
                "<td><small>"
                + html.escape(str(el.get("margin", "")), quote=False)
                + "</small></td>"
                "<td><small>"
                + html.escape(str(el.get("padding", "")), quote=False)
                + "</small></td></tr>\n"
            )
    if not recommendations:
        recommendations = fallback_recommendation_lines_from_stats(stats)
    ollama_list_html = ""
    if stats.get("ollama_bug_polish") and (gemma_markdown or "").strip():
        ollama_list_html = (
            '  <h2>Список правок (Ollama)</h2>\n'
            f'  <pre class="md">{html.escape(gemma_markdown.strip())}</pre>\n'
        )
    bug_table_rows = _build_bug_report_table_html(
        reports_dir,
        stamp,
        stats,
        baseline_path,
        current_shot,
        recommendations,
        diff_path=diff_path if diff_path and os.path.isfile(diff_path) else None,
    )
    ollama_block = ollama_list_html
    model_block = ""
    if not recommendations and (gemma_markdown or "").strip() and not ollama_list_html:
        model_block = (
            f'  <h2>Полный ответ модели (Markdown)</h2>\n'
            f'  <pre class="md">{html.escape(gemma_markdown or "—")}</pre>\n'
        )
    diff_note_html = ""
    if diff_path and os.path.isfile(diff_path):
        rel = _asset_href(reports_dir, diff_path)
        diff_note_html = (
            '<p class="meta">Полная карта diff (отдельный файл, не вставляется в отчёт): '
            f"<code>{rel}</code></p>"
        )

    try:
        cr = float(stats.get("changed_ratio_pct", 0) or 0)
    except (TypeError, ValueError):
        cr = 0.0
    diff_hint = ""
    if cr > 20:
        diff_hint = (
            "<p class=\"warn\"><strong>Внимание:</strong> очень высокий процент отличий по карте diff часто означает не «сломанную вёрстку», "
            "а несовпадение кадра: размер окна браузера ≠ экспорт Figma, другой масштаб (<code>figma.scale</code>), длинная страница vs один фрейм, скролл. "
            "Сверьте <code>window_size</code> в config с размером фрейма в макете и при необходимости задайте тот же видимый фрагмент.</p>"
        )
    status_cls = "ok" if ok else "bad"
    status_txt = "PASS" if ok else "FAIL"
    vp_raw = layout.get("viewport") if isinstance(layout, dict) else None
    vp: Dict[str, Any] = vp_raw if isinstance(vp_raw, dict) else {}
    vw = vp.get("w", stats.get("size", ["?"])[0] if isinstance(stats.get("size"), list) else "?")
    vh = vp.get("h", stats.get("size", ["?", "?"])[1] if isinstance(stats.get("size"), list) and len(stats["size"]) > 1 else "?")

    page = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>QA: макет vs страница</title>
  <style>
    :root {{
      font-family: "Segoe UI", system-ui, sans-serif;
      background: #0f1419;
      color: #e7ecf1;
    }}
    body {{ max-width: 1400px; margin: 0 auto; padding: 24px; line-height: 1.45; }}
    h1 {{ font-size: 1.35rem; margin-top: 0; }}
    .pill {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 600; }}
    .pill.ok {{ background: #1e3d2f; color: #8fefb0; }}
    .pill.bad {{ background: #3d1e1e; color: #ff9b9b; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #2a3440; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #1a222c; text-align: left; }}
    code {{ font-size: 0.85em; }}
    pre.md {{ white-space: pre-wrap; background: #151b24; padding: 12px; border-radius: 8px; font-size: 0.88rem; overflow: auto; }}
    .meta {{ color: #9aa7b5; font-size: 0.9rem; margin-bottom: 20px; }}
    table.bug-report {{ table-layout: fixed; }}
    table.bug-report th {{ white-space: nowrap; font-size: 0.88rem; }}
    table.bug-report td.bug-num {{ width: 2.2rem; text-align: center; color: #9aa7b5; }}
    table.bug-report td.bug-zone {{ width: 7rem; color: #9aa7b5; }}
    table.bug-report td.bug-img {{ width: 220px; padding: 8px; }}
    table.bug-report td.bug-desc {{ min-width: 200px; line-height: 1.55; vertical-align: middle; }}
    .bug-shot {{ margin: 0; width: 200px; height: 150px; display: flex; align-items: center; justify-content: center; background: #0d1117; border-radius: 8px; border: 1px solid #2a3440; overflow: hidden; }}
    .bug-shot img {{ display: block; max-width: 200px; max-height: 150px; width: auto; height: auto; object-fit: contain; }}
  </style>
</head>
<body>
  <h1>Сверка вёрстки с макетом Figma</h1>
  <p class="meta">Время (UTC): {html.escape(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))}
    · окно браузера: {html.escape(str(vw))}×{html.escape(str(vh))} px</p>
  <p><span class="pill {status_cls}">{html.escape(status_txt)}</span></p>

  {diff_note_html}

  <h2>Метрики diff</h2>
  {diff_hint}
  <table>
    <tr><th>MSE</th><td>{html.escape(str(stats.get("mse", "")))}</td></tr>
    <tr><th>Изменённые пиксели (итог), %</th><td>{html.escape(str(stats.get("changed_ratio_pct", "")))}</td></tr>
    <tr><th>Raw / shift, %</th><td>{html.escape(str(stats.get("changed_ratio_raw_pct", "")))} / {html.escape(str(stats.get("changed_ratio_shift_pct", "")))}</td></tr>
    <tr><th>Порог, %</th><td>{html.escape(str(stats.get("threshold_pct", "")))}</td></tr>
    <tr><th>CNN P(fail)</th><td>{html.escape(str(stats.get("model_prob_fail", "—")))}</td></tr>
    <tr><th>Fragment matcher</th><td>{html.escape(_fragment_match_stats_cell(stats))}</td></tr>
  </table>

{ollama_block}
  <h2>Баг-репорт</h2>
  <p class="meta">Строки таблицы — список правок. Ожидаемый / фактический — та же зона (рамка), третья колонка — фрагмент карты diff в этой зоне.</p>
  <table class="bug-report">
    <thead>
      <tr><th>#</th><th>Блок</th><th>Ожидаемый</th><th>Фактический</th><th>Diff</th><th>Баг</th></tr>
    </thead>
    <tbody>
{bug_table_rows}    </tbody>
  </table>

{model_block}
  <h2>Отступы на странице (computed style)</h2>
  <p class="meta">Крупнейшие видимые блоки в окне просмотра; сверяйте с макетом и diff.</p>
  <table>
    <tr><th>Блок</th><th>x</th><th>y</th><th>w</th><th>h</th><th>шрифт</th><th>цвет</th><th>margin</th><th>padding</th></tr>
    {rows_layout or "<tr><td colspan='9'>Нет данных</td></tr>"}
  </table>

</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    with open(last_path, "w", encoding="utf-8") as f:
        f.write(page)
    return path
