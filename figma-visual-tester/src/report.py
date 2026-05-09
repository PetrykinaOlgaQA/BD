"""Pydantic-схема отчёта и вывод в Streamlit (markdown)."""

from __future__ import annotations

import json
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class DefectItem(BaseModel):
    title: str = Field(..., description="Краткий заголовок на русском")
    severity: Literal["low", "medium", "high", "critical"] = Field(default="medium")
    location_hint: str = Field(default="", description="Область экрана / блок")
    expected: str = Field(default="", description="Как по макету Figma")
    actual: str = Field(default="", description="Как на сайте")
    recommendation: str = Field(default="", description="Что поправить")


class BugReport(BaseModel):
    """Строгая JSON-схема ответа vision-модели."""

    verdict: Literal["PASS", "FAIL"] = Field(..., description="Итог визуального аудита")
    bug_probability: float = Field(..., ge=0.0, le=1.0, description="0..1 субъективная уверенность в дефектах")
    summary_ru: str = Field(..., description="2–4 предложения на русском")
    defects: List[DefectItem] = Field(default_factory=list)
    notes: str = Field(default="", description="Ограничения проверки, если есть")

    @field_validator("defects", mode="before")
    @classmethod
    def _coerce_defects(cls, v):
        if v is None:
            return []
        return v


def bug_report_to_markdown(report: BugReport) -> str:
    sev_ru = {"low": "низкая", "medium": "средняя", "high": "высокая", "critical": "критическая"}
    lines = [
        f"### Вердикт: **{report.verdict}**",
        "",
        f"**Вероятность бага:** {report.bug_probability:.0%}",
        "",
        f"**Краткое резюме:** {report.summary_ru}",
        "",
    ]
    if report.defects:
        lines.append("#### Найденные расхождения")
        lines.append("")
        for i, d in enumerate(report.defects, 1):
            lines.append(f"**{i}. {d.title}** _(важность: {sev_ru.get(d.severity, d.severity)})_")
            if d.location_hint:
                lines.append(f"- **Где:** {d.location_hint}")
            if d.expected:
                lines.append(f"- **По макету:** {d.expected}")
            if d.actual:
                lines.append(f"- **На сайте:** {d.actual}")
            if d.recommendation:
                lines.append(f"- **Рекомендация:** {d.recommendation}")
            lines.append("")
    if report.notes:
        lines.append(f"_Примечание:_ {report.notes}")
    return "\n".join(lines).strip()


def parse_bug_report_json(raw: str) -> tuple[Optional[BugReport], Optional[str]]:
    """
    Достаёт JSON из ответа модели (возможны markdown-обёртки).
    Возвращает (BugReport, None) или (None, error_message).
    """
    text = (raw or "").strip()
    if not text:
        return None, "Пустой ответ модели"

    candidates = []
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        candidates.append(m.group(1).strip())
    candidates.append(text)

    last_val_err: Optional[str] = None
    for c in candidates:
        to_try = [c]
        start, end = c.find("{"), c.rfind("}")
        if start >= 0 and end > start:
            to_try.append(c[start : end + 1])
        for chunk in to_try:
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            try:
                return BugReport.model_validate(data), None
            except ValueError as e:
                last_val_err = str(e)
                continue
    return None, last_val_err or "Не удалось распарсить JSON (ожидается объект BugReport)"
