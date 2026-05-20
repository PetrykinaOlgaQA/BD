"""
Синтетический датасет пар (Figma crop | Site crop) с ground-truth по 6 аспектам.

Приоритет обучения:
- text_match: цифры, пропажа текста, другой текст
- image_match: эмодзи/иконки — другая, пропажа, размер
- layout_match: только крупные сдвиги (>15px не компенсируются в train — см. acceptable_shift)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.comparator.models.multi_aspect import ASPECT_KEYS

IMAGE_SIZE: Tuple[int, int] = (224, 224)
BASE_TEXTS: Tuple[str, ...] = (
    "Войти в аккаунт",
    "Купить сейчас",
    "Подробнее",
    "Заказать",
    "Читать все",
)
NUMERIC_TEXTS: Tuple[str, ...] = (
    "700M+",
    "600M+",
    "95% ДНК",
    "Цена 900 руб",
    "Скидка 15%",
    "12 часов сна",
)
EMOJI_CHARS: Tuple[str, ...] = ("🐱", "😴", "👂", "❤️", "🎯", "⭐")
FONT_SIZES: Tuple[int, ...] = (22, 28, 32, 40, 52)
COLOR_HEX: Tuple[str, ...] = ("#111111", "#333333", "#0066cc", "#d32f2f")

AUG_TYPES: Tuple[str, ...] = (
    "ok",
    "minor_noise",
    "acceptable_shift",
    "padding_only",
    "text_minor",
    "text_major",
    "text_critical",
    "text_missing",
    "text_small_change",
    "numeric_text_change",
    "stats_text_change",
    "font_size",
    "color_text_subtle",
    "layout_large",
    "layout_small",
    "emoji_change",
    "emoji_missing",
    "emoji_size_up",
    "emoji_size_down",
    "image_change",
    "critical_break",
)

# Веса выборки: выше у реальных багов
TRAIN_WEIGHTS: Tuple[float, ...] = (
    1.2,  # ok
    1.0,  # minor_noise
    3.0,  # acceptable_shift — учим «после align ≈ ок»
    2.5,  # padding_only
    1.0,  # text_minor
    1.2,  # text_major
    0.6,  # text_critical
    2.5,  # text_missing
    2.0,  # text_small_change — косметика PASS
    5.0,  # numeric
    5.0,  # stats
    0.8,  # font_size
    2.0,  # color subtle PASS
    0.7,  # layout_large
    2.5,  # layout_small PASS
    3.5,  # emoji_change
    3.0,  # emoji_missing
    3.0,  # emoji_size_up
    3.0,  # emoji_size_down
    2.0,  # image_change
    0.5,  # critical_break
)

VAL_TEST_AUGS: Tuple[str, ...] = (
    "ok",
    "acceptable_shift",
    "padding_only",
    "text_major",
    "text_missing",
    "numeric_text_change",
    "stats_text_change",
    "emoji_change",
    "emoji_missing",
    "emoji_size_up",
    "layout_small",
    "critical_break",
)


def _labels_all_ones() -> Dict[str, float]:
    return {k: 1.0 for k in ASPECT_KEYS}


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _random_rgb() -> Tuple[int, int, int]:
    return (random.randint(40, 220), random.randint(40, 220), random.randint(40, 220))


def _draw_card(
    draw: ImageDraw.ImageDraw,
    *,
    text: str = "",
    emoji: str = "",
    font_size: int = 28,
    color: str | Tuple[int, int, int] = "#111111",
    bg: Tuple[int, int, int] = (255, 255, 255),
    icon_box: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    draw.rectangle([0, 0, 224, 224], fill=bg)
    y = 30
    if emoji:
        font = _load_font(font_size)
        bbox = draw.textbbox((0, 0), emoji, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((224 - tw) // 2, y), emoji, fill=color, font=font)
        y += th + 8
    if text:
        font = _load_font(max(14, font_size - 8))
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((224 - tw) // 2, y + 20), text, fill=color, font=font)
    if icon_box:
        x0, y0, x1, y1 = icon_box
        draw.rectangle([x0, y0, x1, y1], fill=_random_rgb(), outline=(0, 0, 0), width=3)
    else:
        draw.rectangle([50, 120, 174, 170], fill=_random_rgb(), outline=(0, 0, 0), width=2)


def generate_pair(aug_type: str = "ok") -> Tuple[Image.Image, Image.Image, Dict[str, float], str]:
    figma = Image.new("RGB", IMAGE_SIZE, color="white")
    site = Image.new("RGB", IMAGE_SIZE, color="white")
    draw_f = ImageDraw.Draw(figma)
    draw_s = ImageDraw.Draw(site)
    labels = _labels_all_ones()

    base_text = random.choice(BASE_TEXTS + NUMERIC_TEXTS)
    font_size = random.choice(FONT_SIZES)
    color = random.choice(COLOR_HEX)

    _draw_card(draw_f, text=base_text, font_size=font_size, color=color)
    site = figma.copy()

    if aug_type == "ok":
        pass

    elif aug_type == "minor_noise":
        site = figma.filter(ImageFilter.GaussianBlur(radius=1))
        labels["overall_similarity"] = 0.94
        labels["text_match"] = 0.93

    elif aug_type == "acceptable_shift":
        # 3–12 px — после align в inference должно быть ≈ PASS
        ox = random.randint(3, 12) * random.choice([-1, 1])
        oy = random.randint(0, 8) * random.choice([-1, 0, 1])
        site = figma.transform(IMAGE_SIZE, Image.AFFINE, (1, 0, ox, 0, 1, oy), fillcolor=(255, 255, 255))
        labels["layout_match"] = round(random.uniform(0.84, 0.96), 3)
        labels["overall_similarity"] = round(random.uniform(0.86, 0.97), 3)
        labels["text_match"] = round(random.uniform(0.90, 0.98), 3)

    elif aug_type == "padding_only":
        site = figma.copy()
        draw_s = ImageDraw.Draw(site)
        draw_s.rectangle([0, 0, 224, 18], fill=(245, 245, 245))
        draw_s.rectangle([0, 206, 224, 224], fill=(245, 245, 245))
        labels["layout_match"] = round(random.uniform(0.88, 0.97), 3)
        labels["overall_similarity"] = round(random.uniform(0.90, 0.98), 3)

    elif aug_type == "text_minor":
        site = figma.copy()
        draw_s = ImageDraw.Draw(site)
        _draw_card(draw_s, text=random.choice(["Войти", "Вход"]), font_size=font_size, color=color)
        labels["text_match"] = round(random.uniform(0.70, 0.84), 3)

    elif aug_type == "text_major":
        site = Image.new("RGB", IMAGE_SIZE, "white")
        draw_s = ImageDraw.Draw(site)
        _draw_card(draw_s, text=random.choice([t for t in BASE_TEXTS if t != base_text] or BASE_TEXTS), font_size=font_size, color=color)
        labels["text_match"] = round(random.uniform(0.10, 0.40), 3)
        labels["overall_similarity"] = round(random.uniform(0.32, 0.55), 3)

    elif aug_type == "text_critical":
        site = Image.new("RGB", IMAGE_SIZE, "white")
        draw_s = ImageDraw.Draw(site)
        _draw_card(draw_s, text="ДРУГОЙ РАЗДЕЛ", font_size=font_size + 6, color=(180, 40, 40))
        labels["text_match"] = 0.06
        labels["overall_similarity"] = 0.22

    elif aug_type == "text_missing":
        site = Image.new("RGB", IMAGE_SIZE, "white")
        draw_s = ImageDraw.Draw(site)
        _draw_card(draw_s, text="", font_size=font_size, color=color)
        labels["text_match"] = round(random.uniform(0.05, 0.25), 3)
        labels["overall_similarity"] = round(random.uniform(0.35, 0.55), 3)
        labels["image_match"] = round(random.uniform(0.70, 0.90), 3)

    elif aug_type == "text_small_change":
        t = random.choice(BASE_TEXTS)
        _draw_card(draw_f, text=t, font_size=font_size, color=color)
        new_t = t[:-1] + "о" if len(t) > 4 else t + "!"
        _draw_card(draw_s, text=new_t, font_size=font_size, color=color)
        labels["text_match"] = round(random.uniform(0.84, 0.95), 3)
        labels["overall_similarity"] = round(random.uniform(0.88, 0.97), 3)

    elif aug_type == "numeric_text_change":
        t = random.choice(["700M+", "600M+", "95% ДНК", "Цена 900 руб", "12 часов"])
        _draw_card(draw_f, text=t, font_size=32, color=color)
        repl = (
            t.replace("700", "600").replace("600", "700")
            .replace("95%", "85%")
            .replace("900", "750")
            .replace("12", "16")
        )
        _draw_card(draw_s, text=repl, font_size=32, color=color)
        labels["text_match"] = round(random.uniform(0.15, 0.42), 3)
        labels["overall_similarity"] = round(random.uniform(0.30, 0.50), 3)
        labels["layout_match"] = round(random.uniform(0.85, 0.96), 3)

    elif aug_type == "stats_text_change":
        t = random.choice(["700M+ кошек", "600M+ в мире"])
        _draw_card(draw_f, text=t, font_size=36, color=color)
        repl = t.replace("700", "600") if "700" in t else t.replace("600", "700")
        _draw_card(draw_s, text=repl, font_size=36, color=color)
        labels["text_match"] = round(random.uniform(0.12, 0.38), 3)
        labels["overall_similarity"] = round(random.uniform(0.28, 0.48), 3)

    elif aug_type == "font_size":
        _draw_card(draw_s, text=base_text, font_size=max(14, int(font_size * 0.55)), color=color)
        labels["typography_match"] = round(random.uniform(0.20, 0.45), 3)
        labels["overall_similarity"] = round(random.uniform(0.50, 0.68), 3)

    elif aug_type == "color_text_subtle":
        site = figma.copy()
        draw_s = ImageDraw.Draw(site)
        _draw_card(draw_s, text=base_text, font_size=font_size, color="#2a2a2a")
        labels["color_match"] = round(random.uniform(0.72, 0.88), 3)
        labels["overall_similarity"] = round(random.uniform(0.82, 0.94), 3)
        labels["text_match"] = round(random.uniform(0.88, 0.97), 3)

    elif aug_type == "layout_large":
        ox, oy = random.randint(18, 40), random.randint(-12, 12)
        site = figma.transform(IMAGE_SIZE, Image.AFFINE, (1, 0, ox, 0, 1, oy), fillcolor=(255, 255, 255))
        labels["layout_match"] = round(random.uniform(0.18, 0.42), 3)
        labels["overall_similarity"] = round(random.uniform(0.35, 0.55), 3)

    elif aug_type == "layout_small":
        ox = random.randint(2, 8) * random.choice([-1, 1])
        site = figma.transform(IMAGE_SIZE, Image.AFFINE, (1, 0, ox, 0, 1, 0), fillcolor=(255, 255, 255))
        labels["layout_match"] = round(random.uniform(0.83, 0.96), 3)
        labels["overall_similarity"] = round(random.uniform(0.88, 0.97), 3)

    elif aug_type == "emoji_change":
        em_f, em_s = random.sample(EMOJI_CHARS, 2)
        _draw_card(draw_f, text=base_text, emoji=em_f, font_size=48, color=color)
        _draw_card(draw_s, text=base_text, emoji=em_s, font_size=48, color=color)
        labels["image_match"] = round(random.uniform(0.12, 0.38), 3)
        labels["text_match"] = round(random.uniform(0.75, 0.92), 3)
        labels["overall_similarity"] = round(random.uniform(0.40, 0.58), 3)

    elif aug_type == "emoji_missing":
        em = random.choice(EMOJI_CHARS)
        _draw_card(draw_f, text=base_text, emoji=em, font_size=48, color=color)
        _draw_card(draw_s, text=base_text, emoji="", font_size=48, color=color)
        labels["image_match"] = round(random.uniform(0.08, 0.28), 3)
        labels["overall_similarity"] = round(random.uniform(0.38, 0.55), 3)

    elif aug_type == "emoji_size_up":
        em = random.choice(EMOJI_CHARS)
        _draw_card(draw_f, text=base_text, emoji=em, font_size=36, color=color)
        _draw_card(draw_s, text=base_text, emoji=em, font_size=58, color=color)
        labels["image_match"] = round(random.uniform(0.22, 0.48), 3)
        labels["overall_similarity"] = round(random.uniform(0.45, 0.62), 3)

    elif aug_type == "emoji_size_down":
        em = random.choice(EMOJI_CHARS)
        _draw_card(draw_f, text=base_text, emoji=em, font_size=56, color=color)
        _draw_card(draw_s, text=base_text, emoji=em, font_size=30, color=color)
        labels["image_match"] = round(random.uniform(0.20, 0.46), 3)
        labels["overall_similarity"] = round(random.uniform(0.42, 0.60), 3)

    elif aug_type == "image_change":
        site = figma.copy()
        draw_s = ImageDraw.Draw(site)
        draw_s.rectangle([45, 115, 179, 175], fill=_random_rgb(), outline=(0, 0, 0), width=6)
        labels["image_match"] = round(random.uniform(0.15, 0.35), 3)
        labels["overall_similarity"] = round(random.uniform(0.48, 0.65), 3)

    elif aug_type == "critical_break":
        site = Image.new("RGB", IMAGE_SIZE, color=_random_rgb())
        draw_s = ImageDraw.Draw(site)
        draw_s.text((30, 90), "FAIL", fill=(255, 255, 255), font=_load_font(32))
        labels = {k: round(random.uniform(0.05, 0.22), 3) for k in ASPECT_KEYS}

    else:
        raise ValueError(f"неизвестный aug_type: {aug_type}")

    return figma, site, labels, aug_type


def save_pair(
    figma: Image.Image,
    site: Image.Image,
    labels: Dict[str, float],
    aug_type: str,
    idx: int,
    split_dir: Path,
    *,
    rel_root: Path,
) -> Dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)
    split_name = split_dir.name
    prefix = f"{split_name}_{idx:05d}"
    figma_name = f"{prefix}_figma.png"
    site_name = f"{prefix}_site.png"
    figma.save(split_dir / figma_name)
    site.save(split_dir / site_name)
    return {
        "figma": str(rel_root / split_name / figma_name).replace("\\", "/"),
        "site": str(rel_root / split_name / site_name).replace("\\", "/"),
        "labels": labels,
        "aug_type": aug_type,
    }


def generate_dataset(
    data_dir: Path | str = "data/comparator",
    *,
    n_train: int = 8000,
    n_val: int = 800,
    n_test: int = 800,
    seed: int = 42,
    progress: bool = True,
) -> Dict[str, int]:
    data_dir = Path(data_dir)
    random.seed(seed)
    np.random.seed(seed)
    rel_root = Path("data/comparator")
    manifests: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    def _iter(n: int, desc: str):
        if progress:
            try:
                from tqdm import tqdm
                return tqdm(range(n), desc=desc)
            except ImportError:
                pass
        return range(n)

    for i in _iter(n_train, "train"):
        aug = random.choices(list(AUG_TYPES), weights=TRAIN_WEIGHTS, k=1)[0]
        f, s, lbl, a = generate_pair(aug)
        manifests["train"].append(save_pair(f, s, lbl, a, i, data_dir / "train", rel_root=rel_root))

    for split, n in (("val", n_val), ("test", n_test)):
        for i in _iter(n, split):
            aug = random.choice(list(VAL_TEST_AUGS))
            f, s, lbl, a = generate_pair(aug)
            manifests[split].append(save_pair(f, s, lbl, a, i, data_dir / split, rel_root=rel_root))

    data_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in manifests.items():
        for record in rows:
            record["source"] = "synthetic"
        # Отдельный manifest для смешивания с Rico (см. scripts/generate_rico_dataset.py --merge-only)
        syn_path = data_dir / f"manifest_{split}_synthetic.jsonl"
        with open(syn_path, "w", encoding="utf-8") as mf:
            for record in rows:
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Если Rico ещё нет — сразу основной train manifest
        main_path = data_dir / f"manifest_{split}.jsonl"
        rico_path = data_dir / f"manifest_{split}_rico.jsonl"
        if not rico_path.is_file():
            with open(main_path, "w", encoding="utf-8") as mf:
                for record in rows:
                    mf.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {k: len(v) for k, v in manifests.items()}


if __name__ == "__main__":
    counts = generate_dataset()
    print("Синтетика:", counts)
    print("Rico:  python scripts/generate_rico_dataset.py --rico-root <path>")
    print("Merge: python scripts/generate_rico_dataset.py --merge-only")
