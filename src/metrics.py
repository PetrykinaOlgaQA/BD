from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def verdict_from_pixels(changed_ratio: float, threshold_pct: float) -> str:
    """Грубый бинарный вердикт только по доле изменённых пикселей (после допусков)."""
    return "PASS" if changed_ratio <= threshold_pct / 100.0 else "FAIL"


def cnn_risk_flag(prob_fail: Optional[float], threshold: float = 0.5) -> bool:
    """True — CNN считает картину diff «похожей на fail»."""
    if prob_fail is None:
        return False
    return prob_fail >= threshold


def thesis_comparison_row(
    human_label: str,
    pixel_verdict: str,
    cnn_prob_fail: Optional[float],
    llm_severity: Optional[str],
) -> Dict[str, Any]:
    """
    Одна строка таблицы для диплома: сводка «человек / пиксели / CNN / субъективная оценка LLM».

    human_label: «ok» | «bug» (разметка вручную).
    llm_severity: после просмотра ответа модели — «low»|«med»|«high» (заполняет экспериментатор).
    """
    return {
        "human": human_label,
        "pixel_only": pixel_verdict,
        "cnn_p_fail": cnn_prob_fail,
        "llm_severity": llm_severity,
    }


def suggest_diploma_metrics() -> List[Tuple[str, str]]:
    """Короткие идеи метрик для главы «эксперимент» (без обязательной реализации в коде)."""
    return [
        (
            "Согласованность с экспертом",
            "Accuracy / F1 бинарного PASS/FAIL относительно ручной разметки на N страницах.",
        ),
        (
            "Полезность текстового отчёта",
            "Likert 1–5 или чек-лист: «нашёл ли отчёт реальный баг»; сравнение с голым diff без LLM.",
        ),
        (
            "CNN как второй сигнал",
            "Доля случаев, где низкий % пикселей, но высокий P(fail) и эксперт подтверждает баг "
            "(CNN ловит «структурно опасный» diff).",
        ),
        (
            "Попиксельный baseline",
            "Тот же датасет, но вердикт только по threshold; LLM+CNN — сколько дополнительных "
            "истинных срабатываний без роста ложных (precision@k).",
        ),
    ]
