#!/usr/bin/env python3
"""
«Обучение» Ollama для баг-репортов: кастомная модель с SYSTEM-промптом из few-shot.

  ollama pull llava:latest
  python scripts/setup_ollama_bug_reporter.py --base llava:latest

  В config.json: "gemma_model": "bug-reporter-ru", "ollama": {"model": "bug-reporter-ru"}
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.bug_report_prompts import system_prompt_bug_reporter_ru


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="llava:latest", help="Базовая vision-модель из ollama pull")
    ap.add_argument("--name", default="bug-reporter-ru", help="Имя новой модели")
    ap.add_argument("--out", default=os.path.join(ROOT, "ollama", "Modelfile.bug-reporter-ru"))
    args = ap.parse_args()

    system = system_prompt_bug_reporter_ru()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    body = (
        f"FROM {args.base}\n\n"
        "PARAMETER temperature 0\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER num_predict 480\n"
        "PARAMETER num_ctx 2048\n\n"
        f'SYSTEM """\n{system}\n"""\n'
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"Modelfile: {args.out}")
    cmd = ["ollama", "create", args.name, "-f", args.out]
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)
    print(f"Готово. Укажите в config.json: gemma_model / ollama.model = \"{args.name}\"")


if __name__ == "__main__":
    main()
