"""Настройки приложения (env + значения по умолчанию)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FVT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Пути относительно корня пакета figma-visual-tester/
    root_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent)
    data_train_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "train")
    reports_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "reports")
    weights_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "weights")

    cnn_weights_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent / "weights" / "diff_cnn_best.pt"
    )
    cnn_fail_threshold: float = Field(default=0.55, description="P(fail) выше порога → FAIL от CNN")

    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    ollama_vision_model: str = Field(default="llama3.2-vision:11b", description="ollama pull llama3.2-vision:11b")
    ollama_timeout_sec: int = Field(default=180)

    figma_token: str = Field(default="", description="Или переменная FIGMA_ACCESS_TOKEN в окружении")
    figma_export_scale: int = Field(default=2)

    selenium_window_width: int = Field(default=1920)
    selenium_window_height: int = Field(default=1080)
    selenium_wait_sec: float = Field(default=3.0)

    diff_target_size: int = Field(default=64, description="Вход CNN (квадрат)")
    diff_blur_ksize: int = Field(default=3, description="Gaussian blur, нечётное >=3")


def get_settings() -> Settings:
    import os

    s = Settings()
    if not (s.figma_token or "").strip():
        env_tok = (os.environ.get("FIGMA_ACCESS_TOKEN") or os.environ.get("FIGMA_TOKEN") or "").strip()
        if env_tok:
            s = s.model_copy(update={"figma_token": env_tok})
    return s


def ensure_runtime_dirs(settings: Settings | None = None) -> Settings:
    s = settings or get_settings()
    for d in (s.data_train_dir / "pass", s.data_train_dir / "fail", s.reports_dir, s.weights_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s
