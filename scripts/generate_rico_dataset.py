#!/usr/bin/env python3
"""Rico → пары figma/site + manifest.jsonl (всё в одном файле)."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.comparator.models.multi_aspect import ASPECT_KEYS

IMAGE_SIZE = (224, 224)
DIGIT_REPLACEMENTS = [
    ("700", "600"), ("600", "700"), ("95%", "85%"), ("85%", "95%"),
    ("900", "750"), ("750", "900"), ("12", "16"), ("16", "12"), ("100", "80"),
]
AUG_TYPES = (
    "ok", "acceptable_shift", "text_changed", "text_missing",
    "image_different", "image_bigger", "image_smaller", "image_missing",
)
AUG_WEIGHTS = [1.5, 3.0, 5.0, 3.5, 3.5, 3.0, 3.0, 3.0]


def _labels(**kw: float) -> Dict[str, float]:
    d = {k: 1.0 for k in ASPECT_KEYS}
    d.update(kw)
    return {k: round(float(d[k]), 3) for k in ASPECT_KEYS}


def labels_for_aug(aug: str) -> Dict[str, float]:
    r = random.uniform
    if aug == "ok":
        return _labels(overall_similarity=r(0.94, 0.99), text_match=r(0.93, 0.99))
    if aug == "acceptable_shift":
        return _labels(layout_match=r(0.86, 0.96), overall_similarity=r(0.88, 0.97), text_match=r(0.90, 0.98))
    if aug == "text_changed":
        return _labels(text_match=r(0.12, 0.35), overall_similarity=r(0.25, 0.48), layout_match=r(0.82, 0.95))
    if aug == "text_missing":
        return _labels(text_match=r(0.05, 0.15), overall_similarity=r(0.28, 0.48))
    if aug == "image_different":
        return _labels(image_match=r(0.08, 0.30), overall_similarity=r(0.32, 0.52))
    if aug in ("image_bigger", "image_smaller"):
        return _labels(image_match=r(0.10, 0.30), overall_similarity=r(0.36, 0.55))
    if aug == "image_missing":
        return _labels(image_match=r(0.05, 0.22), overall_similarity=r(0.30, 0.50))
    return _labels(overall_similarity=0.5)


@dataclass
class UiNode:
    bounds: Tuple[int, int, int, int]
    text: str
    is_text: bool
    is_image: bool


def _walk(root: Dict) -> Iterator[Dict]:
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        for c in n.get("children") or []:
            if isinstance(c, dict):
                stack.append(c)


def extract_nodes(data: Dict, sw: int, sh: int, max_n: int = 8) -> List[UiNode]:
    try:
        root = data["activity"]["root"]
    except (KeyError, TypeError):
        return []
    area_scr = sw * sh
    out: List[UiNode] = []
    for n in _walk(root):
        b = n.get("bounds")
        if not b or len(b) != 4:
            continue
        x0, y0, x1, y1 = map(int, b)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        w, h = x1 - x0, y1 - y0
        if w < 40 or h < 28:
            continue
        if (w * h) / max(1, area_scr) > 0.45:
            continue
        text = str(n.get("text", "") or "").strip()
        cls = str(n.get("class", "")).lower()
        is_text = bool(text) or any(t in cls for t in ("text", "button", "edit"))
        is_image = any(t in cls for t in ("image", "icon"))
        if is_text or is_image or (w > 100 and h > 60):
            out.append(UiNode((x0, y0, x1, y1), text, is_text, is_image))
    out.sort(key=lambda u: (bool(re.search(r"\d", u.text)), u.is_text, u.is_image), reverse=True)
    return out[:max_n]


def discover(rico_root: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    seen: set = set()
    for js in rico_root.glob("**/*.json"):
        if js.stem in seen:
            continue
        img = js.with_suffix(".jpg")
        if not img.is_file():
            img = js.with_suffix(".png")
        if img.is_file():
            seen.add(js.stem)
            pairs.append((img, js))
    return pairs


def crop(screen: Image.Image, bounds: Tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = bounds
    if x1 <= x0 or y1 <= y0:
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        if x1 - x0 < 2:
            x1 = x0 + 2
        if y1 - y0 < 2:
            y1 = y0 + 2
    pad = int(max(x1 - x0, y1 - y0) * 0.06)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(screen.width, x1 + pad), min(screen.height, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return screen.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return screen.crop((x0, y0, x1, y1)).resize(IMAGE_SIZE, Image.Resampling.LANCZOS)


def _font(sz: int = 22):
    for n in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def _bg(img: Image.Image) -> Tuple[int, int, int]:
    a = np.array(img)
    return tuple(int(x) for x in a[0, 0])


def _change_digits(text: str) -> str:
    for a, b in DIGIT_REPLACEMENTS:
        if a in text:
            return text.replace(a, b, 1)
    m = re.search(r"\d+", text)
    if m:
        v = max(0, int(m.group()) + random.randint(-30, 30))
        return text[: m.start()] + str(v) + text[m.end() :]
    return (text + " 600") if text else "600M+"


def _paint_text(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle([0, int(out.height * 0.4), out.width, out.height], fill=_bg(out))
    if text.strip():
        f = _font(20)
        bb = d.textbbox((0, 0), text[:28], font=f)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        d.text(((out.width - tw) // 2, out.height - th - 12), text[:28], fill=(20, 20, 20), font=f)
    return out


def pick_aug(node: UiNode, rng: random.Random) -> str:
    if re.search(r"\d", node.text) and rng.random() < 0.55:
        return "text_changed"
    if node.is_image and rng.random() < 0.45:
        return rng.choices(
            ["image_different", "image_bigger", "image_smaller", "image_missing"],
            weights=[3.5, 3, 3, 2.5], k=1,
        )[0]
    return rng.choices(list(AUG_TYPES), weights=AUG_WEIGHTS, k=1)[0]


def apply_aug(figma: Image.Image, aug: str, text: str = "", alt: Optional[Image.Image] = None) -> Tuple[Image.Image, Dict]:
    figma = figma.convert("RGB")
    site = figma.copy()
    labels = labels_for_aug(aug)
    if aug == "ok" and random.random() < 0.3:
        site = figma.filter(ImageFilter.GaussianBlur(0.5))
    elif aug == "acceptable_shift":
        dx, dy = random.randint(5, 12) * random.choice([-1, 1]), random.randint(0, 8) * random.choice([-1, 0, 1])
        site = figma.transform(figma.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy), fillcolor=_bg(figma))
    elif aug == "text_changed":
        site = _paint_text(figma, _change_digits(text))
    elif aug == "text_missing":
        site = _paint_text(figma, "")
    elif aug == "image_different":
        site = figma.copy()
        d = ImageDraw.Draw(site)
        cx, cy, r = site.width // 2, site.height // 2, 50
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=_bg(figma))
        p = alt.resize((2 * r, 2 * r)) if alt else Image.new("RGB", (2 * r, 2 * r), (200, 50, 50))
        site.paste(p, (cx - r, cy - r))
    elif aug == "image_missing":
        site = figma.copy()
        d = ImageDraw.Draw(site)
        cx, cy, r = site.width // 2, site.height // 2, 55
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=_bg(figma))
    elif aug == "image_bigger":
        w, h = figma.size
        sc = 1.3
        nh, nw = int(h / sc), int(w / sc)
        y0, x0 = (h - nh) // 2, (w - nw) // 2
        site = figma.crop((x0, y0, x0 + nw, y0 + nh)).resize((w, h), Image.Resampling.LANCZOS)
    elif aug == "image_smaller":
        w, h = figma.size
        inner = figma.resize((int(w * 0.58), int(h * 0.58)), Image.Resampling.LANCZOS)
        site = Image.new("RGB", (w, h), _bg(figma))
        site.paste(inner, ((w - inner.width) // 2, (h - inner.height) // 2))
    return site, labels


def print_stats(records: List[Dict]) -> None:
    by_aug = Counter(r["aug_type"] for r in records)
    sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cnt = Counter()
    for r in records:
        a, lbl = r["aug_type"], r["labels"]
        cnt[a] += 1
        for k in ASPECT_KEYS:
            sums[a][k] += float(lbl[k])
    print("\n" + "=" * 58)
    print(f"Всего пар: {len(records)}")
    print("\nПо типу аугментации:")
    for a, n in by_aug.most_common():
        print(f"  {a:20s} {n:6d} ({100*n/len(records):5.1f}%)")
    print("\nСредние лейблы:")
    hdr = "  ".join(f"{k[:8]:>8s}" for k in ASPECT_KEYS)
    print(f"{'aug':20s}  {hdr}")
    for a, _ in by_aug.most_common():
        c = cnt[a]
        print(f"{a:20s}  " + "  ".join(f"{sums[a][k]/c:8.3f}" for k in ASPECT_KEYS))
    print("=" * 58)


def generate(
    rico_root: Path,
    out_dir: Path,
    max_screens: int,
    max_crops: int,
    seed: int,
    splits_filter: Optional[List[str]] = None,
) -> List[Dict]:
    pairs = discover(rico_root)
    if not pairs:
        raise FileNotFoundError(f"Нет jpg+json в {rico_root}")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    pairs = pairs[:max_screens]
    n = len(pairs)
    n_tr, n_va = int(n * 0.85), int(n * 0.10)
    splits = {"train": pairs[:n_tr], "val": pairs[n_tr : n_tr + n_va], "test": pairs[n_tr + n_va :]}
    rel = Path("data/comparator/rico")
    all_recs: List[Dict] = []
    idx_ctr = {"train": 0, "val": 0, "test": 0}

    for split, screens in splits.items():
        if splits_filter and split not in splits_filter:
            continue
        rows: List[Dict] = []
        for img_p, js_p in tqdm(screens, desc=split):
            try:
                screen = Image.open(img_p).convert("RGB")
                data = json.loads(js_p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            nodes = extract_nodes(data, screen.width, screen.height, max_crops)
            patches = [crop(screen, n.bounds) for n in nodes if n.is_image][:3]
            for node in nodes:
                try:
                    figma = crop(screen, node.bounds)
                except (ValueError, OSError):
                    continue
                aug = pick_aug(node, rng)
                alt = rng.choice(patches) if patches else None
                site, labels = apply_aug(figma, aug, node.text, alt)
                i = idx_ctr[split]
                idx_ctr[split] += 1
                sd = out_dir / split
                sd.mkdir(parents=True, exist_ok=True)
                fn, sn = f"rico_{split}_{i:06d}_figma.png", f"rico_{split}_{i:06d}_site.png"
                figma.save(sd / fn)
                site.save(sd / sn)
                rows.append({
                    "figma": f"{rel.as_posix()}/{split}/{fn}",
                    "site": f"{rel.as_posix()}/{split}/{sn}",
                    "labels": labels,
                    "aug_type": aug,
                    "source": "rico",
                })
        man = out_dir.parent / f"manifest_{split}_rico.jsonl"
        with open(man, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_recs.extend(rows)
        print(f"  {split}: {len(rows)} -> {man}")
    return all_recs


def merge_synthetic(data_dir: Path, ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    mul = max(1, int(ratio / max(1e-6, 1 - ratio) * 10))
    for sp in ("train", "val", "test"):
        rows = []
        rp, sp_path = data_dir / f"manifest_{sp}_rico.jsonl", data_dir / f"manifest_{sp}_synthetic.jsonl"
        if rp.is_file():
            rows += [json.loads(l) for l in rp.read_text(encoding="utf-8").splitlines() if l.strip()]
        if sp_path.is_file():
            syn = [json.loads(l) for l in sp_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            rows += syn * mul
        if not rows:
            continue
        rng.shuffle(rows)
        out = data_dir / f"manifest_{sp}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"merge {sp}: {len(rows)} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rico-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "data/comparator/rico")
    ap.add_argument("--max-screens", type=int, default=2500)
    ap.add_argument("--max-crops", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--splits", type=str, default="", help="train,val,test — только эти сплиты")
    ap.add_argument("--synthetic-ratio", type=float, default=0.35)
    args = ap.parse_args()
    data_dir = _ROOT / "data/comparator"
    if args.merge_only:
        merge_synthetic(data_dir, args.synthetic_ratio, args.seed)
        return
    print(f"Rico: {args.rico_root} ({len(discover(args.rico_root))} экранов)")
    sf = [s.strip() for s in args.splits.split(",") if s.strip()] or None
    recs = generate(
        args.rico_root, args.out_dir, args.max_screens, args.max_crops, args.seed, splits_filter=sf
    )
    if recs:
        print_stats(recs)
    if not args.no_merge:
        merge_synthetic(data_dir, args.synthetic_ratio, args.seed)


if __name__ == "__main__":
    main()
