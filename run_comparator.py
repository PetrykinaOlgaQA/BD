#!/usr/bin/env python3
"""MultiAspectComparator — сравнение пары кропов Figma vs Site + HTML-отчёт."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.comparator.inference.compare import ComparatorInference
from src.comparator.inference.merge_report import generate_html_report, merge_nn_with_rules


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MultiAspectComparator — визуальное сравнение кропов",
    )
    parser.add_argument("--figma", required=True, help="Путь к кропу Figma")
    parser.add_argument("--site", required=True, help="Путь к кропу сайта")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.68,
        help="Порог overall_similarity для PASS/FAIL",
    )
    parser.add_argument(
        "--weights",
        default="weights/multi_aspect_comparator_best.pt",
        help="Checkpoint модели",
    )
    parser.add_argument(
        "--html",
        default="reports/comparator_final.html",
        help="Путь к HTML-отчёту",
    )
    args = parser.parse_args()

    infer = ComparatorInference(weights_path=args.weights, root=ROOT)
    nn_result = infer.predict_pair(args.figma, args.site, threshold=args.threshold)

    dom_data = None
    ocr_data = None

    bugs = merge_nn_with_rules(nn_result, dom_data, ocr_data)

    html_path = ROOT / args.html
    generate_html_report(
        args.figma,
        args.site,
        bugs,
        output_path=html_path,
        nn_result=nn_result,
        verdict=nn_result["verdict"],
    )

    print(f"\nГотово! Вердикт: {nn_result['verdict']} | Багов: {len(bugs)}")
    print(f"Overall: {nn_result['scores']['overall_similarity']:.4f}")


if __name__ == "__main__":
    main()
