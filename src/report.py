from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
    """
    if not (markdown or "").strip():
        return []

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
            return lines_out

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
            return lines_out

    body2 = _extract_section(markdown, ("Что сделать в первую очередь",))
    if body2:
        # Одна строка абзаца → одна правка
        flat = re.sub(r"\s+", " ", body2).strip()
        if flat and len(flat) > 12:
            return [flat[:500] + ("…" if len(flat) > 500 else "")]

    structured = _parse_structured_bug_blocks(markdown)
    if structured:
        return [
            f"{r.get('блок', '—')}: {r.get('суть', r.get('раздел', ''))}".strip(": ")
            for r in structured[:20]
        ]

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
            s = str(t).strip()
            if s:
                out.append(s)
            if len(out) >= 16:
                break
    if cr > 20:
        out.append(
            "Сверить window_size в config с размером фрейма в Figma и figma.scale с масштабом экспорта макета (высокий % diff часто из-за разного кадра)."
        )
    cap_wait = stats_capture_wait_seconds(stats)
    if cap_wait < CAPTURE_WAIT_TYPING_OK_SEC:
        out.append(
            "Добавить delay 1500ms перед скриншотом из-за анимации typing/fade/transitions (или увеличить capture_wait_seconds в config до ≥1.5 с)."
        )
    else:
        out.append(
            f"Пауза перед скрином (capture_wait_seconds={cap_wait:g} с) уже покрывает короткие анимации; при шуме на diff смотрите кадр/масштаб и длинные переходы."
        )
    out.append("Повторить прогон после выравнивания кадра; при стабильном кадре разбирать оставшийся diff по блокам из таблицы отступов.")
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


def _asset_href(reports_dir: str, abs_path: str) -> str:
    rel = os.path.relpath(abs_path, reports_dir).replace("\\", "/")
    return html.escape(rel, quote=True)


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
    """Одностраничный отчёт: ссылки на макет и сайт, метрики, таблица багов, превью артефактов."""
    os.makedirs(reports_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(reports_dir, f"qa_report_{stamp}.html")
    last_path = os.path.join(reports_dir, "qa_report_last.html")
    recommendations = parse_recommendation_lines(gemma_markdown)
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
                + html.escape(str(el.get("margin", "")), quote=False)
                + "</small></td>"
                "<td><small>"
                + html.escape(str(el.get("padding", "")), quote=False)
                + "</small></td></tr>\n"
            )
    if not recommendations:
        recommendations = fallback_recommendation_lines_from_stats(stats)
    rec_rows = ""
    for i, text in enumerate(recommendations, start=1):
        rec_rows += f"<tr><td>{i}</td><td>{html.escape(text, quote=False)}</td></tr>\n"

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
    vp = layout.get("viewport") if isinstance(layout, dict) else {}
    vw = vp.get("w", stats.get("size", ["?"])[0] if isinstance(stats.get("size"), list) else "?")
    vh = vp.get("h", stats.get("size", ["?", "?"])[1] if isinstance(stats.get("size"), list) and len(stats["size"]) > 1 else "?")

    def img_block(title: str, ap: str) -> str:
        if not ap or not os.path.isfile(ap):
            return ""
        href = _asset_href(reports_dir, ap)
        return (
            f'<figure class="shot"><figcaption>{html.escape(title)}</figcaption>'
            f'<a href="{href}"><img src="{href}" alt="{html.escape(title)}" loading="lazy" /></a></figure>'
        )

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
    body {{ max-width: 1100px; margin: 0 auto; padding: 24px; line-height: 1.45; }}
    h1 {{ font-size: 1.35rem; margin-top: 0; }}
    .links a {{ color: #7eb8ff; margin-right: 16px; }}
    .pill {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 600; }}
    .pill.ok {{ background: #1e3d2f; color: #8fefb0; }}
    .pill.bad {{ background: #3d1e1e; color: #ff9b9b; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #2a3440; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #1a222c; text-align: left; }}
    code {{ font-size: 0.85em; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .shot img {{ max-width: 100%; height: auto; border: 1px solid #2a3440; border-radius: 6px; }}
    figcaption {{ font-size: 0.85rem; color: #9aa7b5; margin-bottom: 6px; }}
    pre.md {{ white-space: pre-wrap; background: #151b24; padding: 12px; border-radius: 8px; font-size: 0.88rem; overflow: auto; }}
    .meta {{ color: #9aa7b5; font-size: 0.9rem; margin-bottom: 20px; }}
    .warn {{ background: #2a2518; border: 1px solid #6b5a2a; border-radius: 8px; padding: 10px 14px; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>Сверка вёрстки с макетом Figma</h1>
  <p class="meta">Время (UTC): {html.escape(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))}
    · окно браузера: {html.escape(str(vw))}×{html.escape(str(vh))} px</p>
  <p class="links">
    <a href="{html.escape(figma_url, quote=True)}" target="_blank" rel="noopener">Открыть макет в Figma</a>
    <a href="{html.escape(site_url, quote=True)}" target="_blank" rel="noopener">Открыть проверяемую страницу</a>
  </p>
  <p><span class="pill {status_cls}">{html.escape(status_txt)}</span></p>

  <h2>Метрики diff</h2>
  {diff_hint}
  <table>
    <tr><th>MSE</th><td>{html.escape(str(stats.get("mse", "")))}</td></tr>
    <tr><th>Изменённые пиксели (итог), %</th><td>{html.escape(str(stats.get("changed_ratio_pct", "")))}</td></tr>
    <tr><th>Raw / shift, %</th><td>{html.escape(str(stats.get("changed_ratio_raw_pct", "")))} / {html.escape(str(stats.get("changed_ratio_shift_pct", "")))}</td></tr>
    <tr><th>Порог, %</th><td>{html.escape(str(stats.get("threshold_pct", "")))}</td></tr>
    <tr><th>CNN P(fail)</th><td>{html.escape(str(stats.get("model_prob_fail", "—")))}</td></tr>
  </table>

  <h2>Отступы на странице (computed style)</h2>
  <p class="meta">Крупнейшие видимые блоки в окне просмотра; сверяйте с макетом и diff.</p>
  <table>
    <tr><th>Блок</th><th>x</th><th>y</th><th>w</th><th>h</th><th>margin</th><th>padding</th></tr>
    {rows_layout or "<tr><td colspan='7'>Нет данных</td></tr>"}
  </table>

  <h2>Рекомендации по правкам</h2>
  <p class="meta">Каждая строка — отдельная правка для вёрстки (формат задаёт модель в разделе «Рекомендации по правкам»).</p>
  <table>
    <tr><th>#</th><th>Правка</th></tr>
    {rec_rows}
  </table>

  <h2>Скриншоты</h2>
  <div class="grid">
    {img_block("Эталон (Figma PNG)", baseline_path)}
    {img_block("Страница", current_shot)}
    {img_block("Diff", diff_path or "")}
  </div>

  <h2>Полный ответ модели (Markdown)</h2>
  <pre class="md">{html.escape(gemma_markdown or "—")}</pre>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    with open(last_path, "w", encoding="utf-8") as f:
        f.write(page)
    return path
