# Обучение MultiAspectComparator на Rico + синтетика

## План и структура папок

```
нейросеть/
├── config/comparator.yaml          # пути, веса loss, preprocess inference
├── data/comparator/
│   ├── train/                      # синтетика (train_00000_figma.png …)
│   ├── val/
│   ├── test/
│   ├── rico/
│   │   ├── train/                  # rico_train_000000_figma.png …
│   │   ├── val/
│   │   └── test/
│   ├── manifest_train_synthetic.jsonl
│   ├── manifest_train_rico.jsonl
│   ├── manifest_train.jsonl        # merged → для train.py
│   ├── manifest_val*.jsonl
│   └── manifest_test*.jsonl
├── scripts/
│   └── generate_rico_dataset.py    # Rico → пары + merge
└── src/comparator/training/
    ├── rico_parser.py              # JSON hierarchy, bbox, фильтры
    ├── crop_augments.py            # аугментации под ваши баги
    ├── labels.py                   # ground-truth 6 аспектов
    ├── manifest_utils.py           # merge jsonl
    ├── synthetic.py                # синтетика + merge_with_rico()
    ├── dataset.py                  # ComparatorDataset (1 или N manifest)
    └── train.py
```

**Rico на диске** (типичные варианты):

```
rico_dataset_v0.1/
├── combined/           # 66k: {id}.jpg + {id}.json
└── unique_uis/{id}/    # альтернативный layout
```

Папка `OwlEye-dl` — это **не** полный Rico, а репозиторий OwlEye. Укажите `--rico-root` на распакованный Rico (`combined/` или `unique_uis/`).

---

## 1. Генерация синтетики

```powershell
python -m src.comparator.training.synthetic
```

→ `manifest_*_synthetic.jsonl`, кропы в `data/comparator/train|val|test/`.

---

## 2. Генерация Rico

```powershell
python scripts/generate_rico_dataset.py `
  --rico-root "D:/datasets/rico_dataset_v0.1/combined" `
  --out-dir data/comparator/rico `
  --max-screens 8000 `
  --max-crops 8
```

Скрипт:

1. Находит пары `*.jpg` + `*.json`
2. Извлекает TextView, Button, ImageView, карточки (фильтр по площади bbox)
3. Кроп 224×224 = **figma** (эталон)
4. Аугментация → **site** + лейблы 6 аспектов

### Типы аугментаций и лейблы

| aug_type | text_match | image_match | layout | Задача |
|----------|------------|-------------|--------|--------|
| ok | 0.93–0.99 | высокий | высокий | Не ловить шум |
| acceptable_shift | 0.90+ | — | 0.84–0.96 | Сдвиг 3–12px → PASS после align |
| padding_only | — | — | 0.88+ | Отступы → PASS |
| numeric_text_change | **0.15–0.42** | — | высокий | 700→600, % |
| text_missing | **0.05–0.25** | — | — | Пропажа текста |
| image_different | — | **0.10–0.38** | — | Другая иконка |
| image_bigger / smaller | — | **0.18–0.48** | — | Размер иконки |

---

## 3. Merge Rico + synthetic

```powershell
python scripts/generate_rico_dataset.py --merge-only --synthetic-ratio 0.35
```

≈65% Rico, 35% синтетика (цифры, эмодзи из synthetic усиливают редкие классы).

---

## 4. Обучение

```powershell
python -m src.comparator.training.train
```

`ComparatorDataset` читает `config/comparator.yaml` → `manifest_train.jsonl`.

Несколько manifest без merge:

```yaml
paths:
  manifest_train_list:
    - "data/comparator/manifest_train_rico.jsonl"
    - "data/comparator/manifest_train_synthetic.jsonl"
```

---

## 5. Preprocess при inference (уже в проекте)

| Шаг | Зачем |
|-----|--------|
| `align_images` до **15px** | Не ловить мелкие сдвиги |
| `GaussianBlur(0.8)` | Антиалиасинг, субпиксель |
| `calibrate_scores()` | Поднять layout после компенсированного shift |

**Важно:** при обучении `Dataset.__getitem__` использует `align=False` — пары уже с аугментацией на сохранённых PNG. Align только на inference.

---

## 6. Рекомендации

- **Первый прогон:** `--max-screens 2000`, epochs 15, `text_weight: 3.0`
- **Rico root:** если 0 пар — проверьте наличие `combined/*.json` рядом с `.jpg`
- **Память:** 8k экранов × 8 кропов ≈ 50k пар train — достаточно для fine-tune MobileNet
- После обучения: `python run_tests.py` + comparator в `pipeline_integration`
