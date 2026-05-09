# Figma Visual Tester v2

Инструмент сравнивает **рендер Figma** со **скриншотом страницы**, строит **diff-карту**, даёт вердикт **TinyDiffCNN** (PASS/FAIL и P(fail)) и формирует **структурированный отчёт на русском** через **Llama 3.2 Vision** в **Ollama**.

## Стек

- Python 3.11  
- PyTorch, torchvision (зависимость для окружения; ядро CNN в `src/model.py`)  
- Selenium + Chrome (headless)  
- Figma Images API (`requests`)  
- Pillow, NumPy, OpenCV (headless) — blur и CLAHE для diff  
- Streamlit — UI (`app.py`)  
- Ollama — `llama3.2-vision:11b` (или другой vision-тег из `ollama list`)  
- Pydantic v2 — схема `BugReport`  
- loguru — логи в консоли  

## Быстрый старт

```powershell
cd figma-visual-tester
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Ollama и vision-модель

```powershell
ollama serve
ollama pull llama3.2-vision:11b
```

Имя модели можно сменить в сайдбаре Streamlit или в переменной окружения `FVT_OLLAMA_VISION_MODEL`.

### Figma

Создайте Personal Access Token в Figma и задайте:

```powershell
$env:FIGMA_ACCESS_TOKEN = "figd_..."
```

Либо введите токен в сайдбаре (не коммитьте в git).

### Запуск UI

```powershell
cd figma-visual-tester
streamlit run app.py
```

### Обучение CNN

Положите карты diff в:

- `data/train/pass/` — эталонные «мелкие» отличия  
- `data/train/fail/` — явные баги  

Затем:

```powershell
python train.py
```

Веса сохраняются в `weights/diff_cnn_best.pt` (файл создаётся при обучении).

### CLI без Streamlit

```powershell
python predict.py --url https://example.com --figma-file-key KEY --figma-node-id "1:2"
python predict.py --figma-png a.png --site-png b.png --no-vision
```

Отчёты JSON — в `reports/run_*.json`, индекс — `reports/history_index.jsonl`.

## Конфигурация

Файл `.env` (опционально) или префикс `FVT_`:

| Переменная | Смысл |
|------------|--------|
| `FVT_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` |
| `FVT_OLLAMA_VISION_MODEL` | тег модели в Ollama |
| `FVT_CNN_FAIL_THRESHOLD` | порог P(fail) для вердикта CNN |
| `FIGMA_ACCESS_TOKEN` | токен Figma |

См. также `config.py`.

## Структура

```
figma-visual-tester/
├── app.py              # Streamlit
├── config.py
├── train.py
├── predict.py
├── src/
│   ├── model.py
│   ├── diff_utils.py
│   ├── figma_api.py
│   ├── selenium_capture.py
│   ├── ollama_vision.py
│   ├── report.py
│   └── utils.py
├── data/train/pass|fail/
├── weights/
└── reports/
```

## Тесты

```powershell
pip install pytest
pytest tests/ -q
```
