"""
Промпты для Ollama: few-shot и правила из data/bugs/*.json.
Подставьте свой каталог формулировок для другого домена (EN, код-ревью и т.д.).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUGS_DIR = os.path.join(_ROOT, "data", "bugs")


@lru_cache(maxsize=1)
def _load_json(name: str) -> Dict[str, Any]:
    path = os.path.join(_BUGS_DIR, name)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def issue_phrases_ru() -> Dict[str, str]:
    catalog = _load_json("catalog_ru.json")
    few = _load_json("ollama_fewshot_ru.json")
    out: Dict[str, str] = {}
    raw = catalog.get("issue_phrases")
    if isinstance(raw, dict):
        out.update({str(k): str(v) for k, v in raw.items()})
    tpl = few.get("issue_templates")
    if isinstance(tpl, dict):
        for k, v in tpl.items():
            out.setdefault(str(k), str(v))
    return out


def rules_block() -> str:
    few = _load_json("ollama_fewshot_ru.json")
    rules = few.get("rules")
    if not isinstance(rules, list):
        return ""
    lines = [f"- {str(r).strip()}" for r in rules if str(r).strip()]
    return "\n".join(lines)


def fewshot_block() -> str:
    few = _load_json("ollama_fewshot_ru.json")
    good = few.get("good_examples") if isinstance(few.get("good_examples"), list) else []
    bad = few.get("bad_examples") if isinstance(few.get("bad_examples"), list) else []
    parts: List[str] = ["ХОРОШО (пиши так):"]
    parts.extend(str(x).strip() for x in good[:8] if str(x).strip())
    parts.append("")
    parts.append("ПЛОХО (никогда так):")
    parts.extend(str(x).strip() for x in bad[:6] if str(x).strip())
    phrases = issue_phrases_ru()
    if phrases:
        parts.append("")
        parts.append("Шаблоны формулировок:")
        for v in list(phrases.values())[:10]:
            parts.append(f"- …{v}")
    return "\n".join(parts)


def system_prompt_bug_reporter_ru() -> str:
    """Для Modelfile Ollama (ollama create)."""
    return (
        "Ты QA-редактор баг-репортов по вёрстке: макет Figma vs страница.\n"
        "Отвечай только списком строк «- …» на русском.\n\n"
        f"ПРАВИЛА:\n{rules_block()}\n\n"
        f"{fewshot_block()}"
    )


def refine_draft_prompt(
    draft_txt: str,
    *,
    context_label: str,
    changed_ratio_pct: float,
) -> str:
    return (
        "Ты редактор баг-репорта по вёрстке (Figma vs сайт).\n"
        "Перепиши ЧЕРНОВИК в короткий список для разработчика. Сохрани смысл каждого пункта.\n"
        "Если в строке есть [Блок статистики], [Карточка …], [Шапка] — оставь этот префикс.\n"
        "Строк «- …» в ответе должно быть столько же, сколько в черновике.\n\n"
        f"ПРАВИЛА:\n{rules_block()}\n\n"
        f"{fewshot_block()}\n\n"
        f"Контекст: {(context_label or '').strip() or '—'}\n"
        f"Изменённые пиксели (diff): ~{changed_ratio_pct:g}%\n\n"
        "ЧЕРНОВИК (обязательно учти все пункты):\n"
        f"{draft_txt}\n\n"
        "Ответ — только список «- …», без вступления."
    )


def vision_diff_prompt_header(
    *,
    context_label: str,
    mse_v: Any,
    cr_v: Any,
    th_v: Any,
    low_diff_block: str = "",
    task_lines: List[str] | None = None,
) -> str:
    tasks = task_lines or []
    tasks_txt = ""
    if tasks:
        tasks_txt = "Подсказки по diff (используй, не дублируй дословно):\n" + "\n".join(
            f"- {str(t)[:140]}" for t in tasks[:12]
        ) + "\n\n"
    pass_rule = ""
    if low_diff_block.strip():
        pass_rule = low_diff_block + "\n\n"
    return (
        "Сравни макет Figma и страницу по diff-картинке.\n"
        "Ответ — только список багов для таблицы QA.\n\n"
        f"ПРАВИЛА:\n{rules_block()}\n\n"
        f"{fewshot_block()}\n\n"
        f"{pass_rule}"
        f"Контекст: {(context_label or '').strip() or '—'}\n"
        f"MSE ≈ {mse_v}, изменённые пиксели ≈ {cr_v}%, порог ≈ {th_v}%.\n\n"
        f"{tasks_txt}"
        "Формат: каждая строка «- зона/элемент: суть бага».\n"
    )
