# Сайт vs макет Figma

Один сценарий: **сверстанная страница** открывается в Chrome, делается скриншот, рядом подтягивается **кадр из Figma** по API, строится **diff**, маленькая **CNN** смотрит на карту отличий, модель в **Ollama** (например Gemma) пишет **что не совпало с макетом**.

Раньше в проекте был режим «два URL друг против друга» — он убран. Осталось только сравнение **вёрстка ↔ Figma**.

### Где «папка с сайтом»

В репозитории есть папка **`site`** — это **демо-вёрстка** (статический HTML/CSS), чтобы сразу было что открыть и сравнить с Figma. Её можно править под свой макет или **не использовать** и указать в `url_site` адрес своего проекта (Vite и т.д.).

Подробные шаги для `site` — файл **`site/ЗАПУСК.txt`**.

---

## Что нужно на компьютере

1. **Python 3.10+**
2. **Google Chrome** (для Selenium)
3. **Токен Figma** — [Figma → Settings → Personal access tokens](https://www.figma.com/developers/api#access-tokens)  
   Токен **ни в git, ни в скриншоты** — только переменная окружения.
4. **Ollama** с vision-моделью (если нужен текстовый отчёт о багах). Имя модели — в `config.json` → `gemma_model`.

---

## Установка

```powershell
cd "путь\к\папке\нейросеть"
pip install -r requirements.txt
```

Скопируй `config.example.json` в `config.json` и поправь:

| Поле | Смысл |
|------|--------|
| `url_site` | URL страницы: для демо из папки `site` обычно **`http://127.0.0.1:8080`** (см. `site/ЗАПУСК.txt`); свой проект — как выдаёт `npm run dev` |
| `window_size` | Ширина×высота окна браузера для скрина (как в макете) |
| `figma.file_key` | Из ссылки `figma.com/design/**KEY**/…` |
| `figma.node_id` | Из `node-id=19-2` в URL → в конфиге пиши **`19:2`** |
| `figma.design_png` | Куда кэшировать PNG кадра из Figma |
| `figma.scale` | Масштаб экспорта (1–4), чаще 2 |

---

## Как запустить тест (3 способа)

### A) Консоль (самый простой)

1. Запусти **сайт** — для демо из этой же репы:
   ```powershell
   cd "путь\к\нейросеть\site"
   python -m http.server 8080
   ```
   В `config.json` тогда **`url_site`**: `http://127.0.0.1:8080`  
   (Свой фронт на npm — в **его** папке `npm run dev`, в конфиге тот URL, что выведет терминал.)
2. В **новом** окне PowerShell:

```powershell
cd "путь\к\нейросеть"
$env:FIGMA_ACCESS_TOKEN = "твой_токен"
python run_tests.py
```

Скрипт сам скачает макет из Figma, снимет страницу с `url_site`, сравнит, положит отчёт в `reports/`, артефакты в `reports/witness_*` и `shots/`.

Другой URL без правки конфига:

```powershell
python run_tests.py --url http://127.0.0.1:3000
```

### B) Веб-панель

Токен должен быть в **том же** окружении, откуда стартуешь сервер:

```powershell
$env:FIGMA_ACCESS_TOKEN = "твой_токен"
python web_server.py
```

Открой в браузере: **http://127.0.0.1:8765** → укажи URL страницы (и при необходимости file key / node id) → **Сравнить с макетом**.

### C) Окно Tkinter

```powershell
$env:FIGMA_ACCESS_TOKEN = "твой_токен"
python app.py
```

---

## Обучить CNN

CNN учится на **маленьких ч/б картинках 64×64** — это кропы **карты diff** (как в пайплайне), классы `pass` / `fail`.

**Быстрый старт (синтетика для проверки, что всё запускается):**

```powershell
python scripts/bootstrap_train_dataset.py
python train.py --epochs 25 --out weights/diff_cnn.pt
```

Появится `weights/diff_cnn.pt`. В реальном дипломе лучше заменить содержимое `data/train/pass` и `data/train/fail` на **свои** кропы diff из папок `shots/diffs` после прогонов (ручная разметка «баг / ок»).

В интерфейсах галочка **CNN по diff** подключает этот файл.

---

## Как это находит и описывает баги

1. **Пиксели** — процент отличающихся пикселей после допусков (сдвиг, opening), порог в `diff_threshold_pct`.
2. **CNN** — дополнительный сигнал по текстуре diff (может подсветить «опасный» diff при пограничном %).
3. **Ollama** — по метрикам и картинке diff генерирует **текст**: резюме, список вероятных багов, зона экрана (см. `src/gemma_client.py`).

Без Ollama останутся числа, diff-картинка и отчёт в `reports/*.txt`.

---

## Вспомогательно: только выгрузить PNG из Figma

```powershell
$env:FIGMA_ACCESS_TOKEN = "…"
python figma_pull.py --file КЛЮЧ_ФАЙЛА --node 19:2 --out storage/designs/maket.png
```

---

## Структура

| Файл / папка | Назначение |
|--------------|------------|
| `run_tests.py` | CLI: один прогон |
| `web_server.py` | Локальный сайт-панель |
| `app.py` | Десктоп-панель |
| `src/pipeline.py` | `run_figma_vs_site` + `run_pipeline` |
| `src/compare.py` | Diff и метрики |
| `src/model_net.py` | CNN |
| `src/gemma_client.py` | Запрос к Ollama |
| `figma_pull.py` | Отдельная выгрузка кадра |
| `scripts/bootstrap_train_dataset.py` | Синтетика для первого обучения |

---

## Безопасность

`config.json` с токенами не коммить — в `.gitignore` уже есть шаблон. Для Figma используй только **`FIGMA_ACCESS_TOKEN`** в окружении.
