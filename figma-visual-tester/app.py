"""
Figma Visual Tester v2 — Streamlit UI.
Запуск из каталога figma-visual-tester:
  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Корень пакета на PYTHONPATH при прямом запуске
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import streamlit as st
from loguru import logger
from PIL import Image

from config import ensure_runtime_dirs, get_settings
from src.diff_utils import create_diff_map
from src.figma_api import export_frame_to_pil
from src.model import load_model, numpy_gray_to_tensor, predict_diff
from src.ollama_vision import analyze_with_llama_vision, text_fallback_bug_report
from src.report import bug_report_to_markdown
from src.selenium_capture import capture_url_to_pil, load_image_file
from src.utils import append_report_history


def _verdict_color(verdict: str) -> str:
    return "#16a34a" if verdict.upper() == "PASS" else "#dc2626"


@st.cache_resource
def _cached_cnn(weights_path: str):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loaded = load_model(weights_path, device=dev)
    return model, loaded, dev


def main() -> None:
    st.set_page_config(page_title="Figma Visual Tester v2", layout="wide")
    st.title("Figma Visual Tester v2")
    st.caption("Сравнение макета Figma со страницей: CNN + Llama 3.2 Vision (Ollama)")

    s = ensure_runtime_dirs()
    with st.sidebar:
        st.header("Настройки")
        s = s.model_copy(
            update={
                "ollama_base_url": st.text_input("Ollama URL", value=s.ollama_base_url),
                "ollama_vision_model": st.text_input("Vision-модель", value=s.ollama_vision_model),
                "figma_token": st.text_input("Figma token (или env FIGMA_ACCESS_TOKEN)", value=s.figma_token, type="password"),
                "cnn_fail_threshold": st.slider("Порог P(fail) CNN", 0.1, 0.95, float(s.cnn_fail_threshold), 0.05),
                "selenium_wait_sec": st.slider("Пауза после загрузки страницы, с", 0.5, 15.0, float(s.selenium_wait_sec), 0.5),
            }
        )
        use_vision = st.checkbox("Вызвать Llama Vision", value=True)
        st.divider()
        st.markdown("Модель: `ollama pull llama3.2-vision:11b` (или ваш тег в Ollama).")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Источники")
        mode = st.radio("Режим", ("Figma API + URL", "Локальные изображения"), horizontal=True)
        figma_key = st.text_input("figma_file_key", placeholder="abcXYZ...")
        figma_node = st.text_input("figma_node_id", placeholder="1:234")
        url = st.text_input("url страницы", placeholder="https://...")
        f_loc = st.file_uploader("Локально: Figma PNG", type=["png", "jpg", "webp"])
        w_loc = st.file_uploader("Локально: Website PNG", type=["png", "jpg", "webp"])
    with col_b:
        st.subheader("Подсказка")
        st.markdown(
            "- Локальные файлы — быстрый тест diff/CNN без API.\n"
            "- Для Figma нужен токен и корректный `node_id`.\n"
            "- Chrome для Selenium должен быть установлен."
        )

    run = st.button("Проверить страницу", type="primary")

    if not run:
        st.info("Заполните поля и нажмите «Проверить страницу».")
        return

    progress = st.progress(0, text="Старт…")
    try:
        progress.progress(10, text="Загрузка изображений…")
        if mode == "Локальные изображения":
            if not f_loc or not w_loc:
                st.error("Нужны два файла: Figma и Website.")
                return
            figma = Image.open(f_loc).convert("RGB")
            site = Image.open(w_loc).convert("RGB")
        else:
            if not figma_key.strip() or not figma_node.strip():
                st.error("Укажите figma_file_key и figma_node_id.")
                return
            if not (s.figma_token or "").strip():
                st.error("Нужен Figma token (sidebar или FIGMA_ACCESS_TOKEN).")
                return
            figma = export_frame_to_pil(
                figma_key.strip(),
                figma_node.strip(),
                s.figma_token.strip(),
                scale=s.figma_export_scale,
            )
            if not url.strip():
                st.error("Укажите URL страницы.")
                return
            site = capture_url_to_pil(
                url.strip(),
                (s.selenium_window_width, s.selenium_window_height),
                wait_seconds=s.selenium_wait_sec,
            )

        progress.progress(40, text="Построение diff…")
        diff_res = create_diff_map(
            figma,
            site,
            target_size=(s.diff_target_size, s.diff_target_size),
            blur_ksize=s.diff_blur_ksize,
        )

        progress.progress(55, text="CNN…")
        model, loaded, dev = _cached_cnn(str(s.cnn_weights_path))
        tensor = numpy_gray_to_tensor(diff_res.diff_gray_64)
        cnn = predict_diff(model, tensor, device=dev)
        c_verdict = "FAIL" if cnn["prob_fail"] >= s.cnn_fail_threshold else "PASS"

        vision_md = ""
        vision_raw = ""
        bug_rep = None
        parse_err = None
        if use_vision:
            progress.progress(70, text="Llama Vision (Ollama)…")
            bug_rep, vision_raw, parse_err = analyze_with_llama_vision(
                diff_res.aligned_figma,
                diff_res.aligned_website,
                diff_res.diff_display,
                ollama_base_url=s.ollama_base_url,
                model=s.ollama_vision_model,
                timeout_sec=s.ollama_timeout_sec,
            )
            if bug_rep is not None:
                vision_md = bug_report_to_markdown(bug_rep)
            else:
                fb = text_fallback_bug_report(vision_raw, parse_err)
                vision_md = bug_report_to_markdown(fb) + "\n\n---\n**Сырой ответ модели:**\n```\n" + (vision_raw or "")[:12000] + "\n```"

        progress.progress(90, text="Сохранение отчёта…")
        record = {
            "verdict": c_verdict,
            "cnn": cnn,
            "cnn_weights_loaded": loaded,
            "vision_parse_error": parse_err,
        }
        if bug_rep is not None:
            record["vision"] = bug_rep.model_dump()
        append_report_history(s.reports_dir, record)
        progress.progress(100, text="Готово")

        st.divider()
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"<div style='text-align:center'><span style='font-size:2.2rem;font-weight:700;color:{_verdict_color(c_verdict)}'>{c_verdict}</span>"
                f"<div style='margin-top:8px'>CNN вердикт (порог {s.cnn_fail_threshold:.2f})</div></div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.metric("P(fail) CNN", f"{cnn['prob_fail']:.1%}")
        with m3:
            if bug_rep is not None:
                st.markdown(
                    f"<div style='text-align:center'><span style='font-size:2.2rem;font-weight:700;color:{_verdict_color(bug_rep.verdict)}'>{bug_rep.verdict}</span>"
                    f"<div style='margin-top:8px'>Vision · P(bug) {bug_rep.bug_probability:.0%}</div></div>",
                    unsafe_allow_html=True,
                )

        st.subheader("Figma · Website · Diff")
        i1, i2, i3 = st.columns(3)
        with i1:
            st.image(diff_res.aligned_figma, use_container_width=True, caption="Figma (aligned)")
        with i2:
            st.image(diff_res.aligned_website, use_container_width=True, caption="Website (aligned)")
        with i3:
            st.image(diff_res.diff_display, use_container_width=True, caption="Diff")

        st.subheader("Bug Report (Vision)")
        st.markdown(vision_md or "_Vision отключён._")

        if parse_err and bug_rep is None:
            with st.expander("Ошибка разбора JSON"):
                st.code(parse_err or "")

        if not loaded:
            st.warning(f"Файл весов CNN не найден: `{s.cnn_weights_path}` — обучите `python train.py`.")

    except Exception as e:
        logger.exception("run")
        st.error(str(e))
        progress.progress(100, text="Ошибка")


if __name__ == "__main__":
    main()
