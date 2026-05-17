# Датасет багов и обучение CNN

## Формулировки в отчёте

Краткие строки генерирует `src/bug_reports.py` (по diff и layout):  
`div.wrapper — размер не совпадает с макетом, padding сверху на 12px больше`

## Свой датасет (рекомендуется для диплома)

1. Несколько прогонов сверки → `shots/diffs/*.png` и `reports/*_last.json`
2. Сборка классов pass/fail (64×64 кропы diff):

```bash
python scripts/build_train_dataset.py --fail-threshold-pct 0.5
python train.py --epochs 35 --out weights/diff_cnn.pt
```

3. В `config.json`: `"model_path": "weights/diff_cnn.pt"`

## Внешние датасеты (справочник)

| Источник | Назначение |
|----------|------------|
| [VisionTriage](https://huggingface.co/datasets/tathadn/visiontriage-multimodal) | Тексты баг-репортов + скрины (импорт: `scripts/import_visiontriage_sample.py`) |
| [OwlEye](https://github.com/20200501/OwlEye) | Классификация UI-багов на мобильных скринах |

Для **TinyDiffCNN** в этом проекте используются именно **карты diff 64×64**, а не сырые скриншоты VisionTriage.

## Синтетика (первый запуск без прогонов)

```bash
python scripts/bootstrap_train_dataset.py --pass 64 --fail 64
python train.py
```

## PyTorch не грузится (WinError 1114, c10.dll)

1. Установите [VC++ Redistributable x64](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist), перезагрузите ПК.
2. Или обучите **без PyTorch**:

```bash
pip install scikit-learn joblib
python train_sklearn.py
```

В `config.json`: `"model_path": "weights/diff_cnn_sklearn.joblib"`
