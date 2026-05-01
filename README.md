# UI Diff Lab — нейросеть для тестирования сайтов

Это **инструмент автоматизированного тестирования вёрстки**: он открывает страницу в браузере, делает скриншот и **сравнивает реализацию с эталоном** — с **макетом из Figma** (экспорт кадра по API) или с **живым продом**, пока вы верстаете локально.

Внутри: числовой diff, опционально **CNN** по карте отличий и **текстовый отчёт** (что именно не совпало с макетом) через Gemma в Ollama.

Демо-макет в Figma: [Test — node 19:2](https://www.figma.com/design/KqJfoHyA6re2zYzDmriMT2/Test?node-id=19-2).

## Быстрый старт

1. Python 3.10+, установленный **Google Chrome** (Selenium использует встроенный менеджер драйвера).
2. Скопируйте `config.example.json` → `config.json` и пропишите URL.
3. Установка зависимостей:

```bash
pip install -r requirements.txt
```

4. (Опционально) Ollama с моделью, имя которой указано в `gemma_model` (например `gemma3`).

### Десктоп (Tkinter)

```bash
python app.py
```

### Веб-интерфейс

```bash
python web_server.py
```

Откройте `http://127.0.0.1:8765` — панель теста: эталон vs проверяемый сайт, лог и разбор от модели.

### CLI: эталон (прод) vs тестируемая локалка

```bash
python run_tests.py --real https://example.com --local http://127.0.0.1:5173
```

### CLI: **макет Figma vs тестируемый сайт**

Нужен [Personal Access Token](https://www.figma.com/developers/api#access-tokens) Figma. **Не коммитьте токен** — только переменная окружения.

PowerShell:

```powershell
$env:FIGMA_ACCESS_TOKEN = "ваш_токен"
python run_tests.py --figma-vs-local --local http://127.0.0.1:5173
```

Ключ файла и `node_id` берутся из `config.json` → секция `figma` (в примере уже указаны `file_key` и `19:2`). В ссылке Figma `node-id=19-2` в API передаётся как **`19:2`**.

Переопределение из командной строки:

```bash
python run_tests.py --figma-vs-local --figma-file KqJfoHyA6re2zYzDmriMT2 --figma-node 19:2 --local http://127.0.0.1:3000
```

### Скачать PNG и JSON узла из Figma

```bash
set FIGMA_ACCESS_TOKEN=...
python figma_pull.py --file KqJfoHyA6re2zYzDmriMT2 --node 19:2 --out storage/designs/frame.png --json-out figma_node_export.json
```

## Обучение CNN

После обновления архитектуры сети старый файл `weights/diff_cnn.pt` может **не загрузиться** — переобучите:

```bash
python train.py --data data/train --out weights/diff_cnn.pt --epochs 30
```

Структура данных: `data/train/pass/*.png` и `data/train/fail/*.png` — **grayscale-кропы карт diff** (как в пайплайне, 64×64).

## Зачем нейросеть и LLM при тесте «сайт vs макет», если есть попиксельное сравнение?

| Попиксельный diff | CNN + VLM (Gemma) |
|-------------------|-------------------|
| Доля изменённых пикселей и heatmap | CNN: отдельный сигнал «опасный» паттерн diff даже при пограничном % |
| Не объясняет *что* сломано в терминах UI | Текст: **какой блок**, отступ, шрифт, зона экрана |
| Шум от антиалиаса, скролла, даты/времени | Промпт учитывает допуски (сдвиг, opening) и просит **семантическую** интерпретацию |

Для диплома имеет смысл завести небольшую **ручную разметку** (есть баг / нет) и сравнить: только порог по %, порог + CNN, полный конвейер с отчётом LLM — см. идеи в `src/metrics.py` (`suggest_diploma_metrics`).

## Безопасность

- Файл `config.json` и любые токены — в **`.gitignore`** (шаблон уже в репозитории).
- Если токен Figma когда-либо попадал в чат, почту или скриншот — **отзовите его** в настройках Figma и выпустите новый.

## Структура проекта

| Путь | Назначение |
|------|------------|
| `app.py` | Десктоп-панель теста (Tkinter) |
| `web_server.py` | Веб-панель теста (Flask) |
| `run_tests.py` | CLI: прогон тестов, в т.ч. макет Figma vs сайт |
| `figma_pull.py` | Подтягивание кадра макета (PNG / JSON) из Figma |
| `src/pipeline.py` | Скриншоты страниц, сравнение с эталоном, отчёт |
| `src/compare.py` | Метрики diff, heatmap |
| `src/gemma_client.py` | Текстовый отчёт о расхождениях с макетом (Ollama) |
| `src/model_net.py` | CNN по карте diff |
| `src/metrics.py` | Идеи метрик качества теста для диплома |

## Лицензия и диплом

Проект учебный; при оформлении работы укажите версии Python, Chrome, Ollama и модели VLM.
