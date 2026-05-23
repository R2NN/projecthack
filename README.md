# Shelf Vision: распознавание ценников по видео полки

Shelf Vision принимает MP4-видео прохода робота вдоль стеллажа и возвращает CSV в формате `sample.csv`: название товара, цены, скидки, barcode/sku, координаты ценника, QR-поля и служебные признаки. Решение рассчитано на поток видео из разных магазинов: результаты сохраняются в SQLite, crops для проверки уходят в отдельный HTML/ZIP-отчёт, а неуверенные предсказания складываются для последующего дообучения.

Если при проверке или запуске возникнут проблемы, напишите в Telegram: [@rktqq](https://t.me/rktqq). Быстро поможем с окружением, артефактами или демо-стендом.

Демо-стенд:

https://apt-valley-plugins-therefore.trycloudflare.com

Ссылка работает через Cloudflare quick tunnel и может меняться при выключении компьютера, перезапуске или сбое туннеля. Если ссылка не открывается, настоятельно прошу написать в Telegram: [@rktqq](https://t.me/rktqq).

Health check:

https://apt-valley-plugins-therefore.trycloudflare.com/api/health

## Быстрый старт: 2 режима одной командой

Развертывание было одним из сложных мест: раньше нужно было понимать внутренние скрипты, OCR-зависимости, веса, каталог и runtime-пути. Сейчас проект собран как Docker Compose appliance с двумя понятными режимами.

Перед запуском нужны:

- Docker Desktop или Docker Engine с Docker Compose;
- Git LFS для скачивания модели, каталога и примера видео;
- интернет для первой сборки Docker image и загрузки Python/OCR/RF-DETR артефактов;
- для основного GPU inference и training: NVIDIA driver и NVIDIA Container Toolkit;
- для training: размеченные данные в формате `<name>/<name>.mp4` + `<name>/<name>.csv`.

Скачать проект:

```bash
git lfs install
git clone https://github.com/R2NN/projecthack.git
cd projecthack
git lfs pull
```

Режим 1 — inference с готовыми весами.

Рекомендуемый GPU-запуск для NVIDIA:

```bash
docker compose --profile gpu up --build shelf-vision-gpu
```

Открыть:

```text
http://127.0.0.1:8001/
```

CPU fallback, если NVIDIA GPU или NVIDIA Container Toolkit недоступны:

```bash
docker compose up --build
```

Открыть:

```text
http://127.0.0.1:8000/
```

Режим 2 — обучение с нуля и запуск сайта на новых весах:

```bash
docker compose --profile training up --build shelf-vision-training
```

Для training положите разметку в:

```text
artifacts/training_data/
  25_12-20/
    25_12-20.mp4
    25_12-20.csv
```

После обучения сайт будет доступен на `http://127.0.0.1:8002/`, а новые веса сохранятся в `runtime/training/models/`. Основной checkpoint из `artifacts/models/` не перезаписывается.

## Коротко

- Полный цикл: загрузка видео, детекция ценников, OCR, восстановление по `db_hack`, sanity-check цен, дедупликация, CSV.
- Инференс локальный: модель, OCR и каталог работают без внешних API.
- Результаты сохраняются не только в браузере: CSV, SQLite, manifest, HTML/ZIP-отчёт по crops, uncertain-хранилище.
- Загрузка больших MP4 идёт по частям: браузер отправляет файл блоками, а сервер запускает обработку после полной сборки файла.
- Архитектура готова к горизонтальному масштабированию: видео, кадры и crops можно распределять между workers.
- Для воспроизведения крупные артефакты лежат в Git LFS: checkpoint, `db_hack.csv`, `sample.csv` и пример видео.
- Добавлен worker-ready режим: web/API может только ставить задачи в очередь, а независимые workers забирают их без изменения ML-логики.

## Что важно для проверки

- **Качество распознавания**: pipeline не ограничивается OCR как есть. После детекции идут tracking, выбор лучшего crop, несколько OCR-веток, распознавание QR/barcode, сверка с `db_hack` при уверенном matching, проверка ценовых аномалий и дедупликация повторов.
- **Соответствие формату**: результат экспортируется в CSV со схемой `sample.csv`, включая товарные поля, цены, скидки, barcode, QR-поля, координаты и служебные признаки.
- **Воспроизводимость**: в репозитории есть Dockerfile, `docker-compose.yml`, проверка артефактов, README-инструкция и Git LFS артефакты для полного запуска.
- **Архитектура**: web/API, detector, OCR, catalog recovery, postprocess, хранилище результатов и review-отчёты разделены на понятные этапы.
- **Масштабирование**: обработка естественно раскладывается по видео, кадрам и crops; добавлен выключенный по умолчанию worker-ready слой с очередью, heartbeat, retry и lease-lock, поэтому workers можно подключать без переписывания pipeline.
- **Демонстрация**: есть живой web-стенд, загрузка MP4 через браузер, прогресс обработки, скачивание CSV и отдельный HTML/ZIP-отчёт для проверки сложных случаев.


## Архитектура

```mermaid
flowchart TD
    A[MP4 видео робота] --> B[Web API]
    B --> C[Chunk upload storage]
    C --> D{Execution mode}
    D -->|local demo| E[Local inference thread]
    D -->|worker-ready| W[SQLite queue / future MQ]
    W --> X[GPU/CPU workers]
    E --> Y[RF-DETR detector]
    X --> Y
    Y --> F[Tracking ценников]
    F --> G[Crop quality selection]
    G --> H[OCR zones]
    H --> I[Numeric OCR]
    H --> J[Product-name OCR]
    I --> K[QR / barcode / price parser]
    J --> L[Catalog recovery по db_hack при уверенном matching]
    K --> M[Price sanity]
    L --> M
    M --> N[Dedup]
    N --> O[CSV sample submit]
    N --> P[SQLite]
    N --> Q[HTML review report]
    N --> R[Uncertain storage]
```

Основные компоненты:

- `web/` — UI, API, очередь задач, progress/ETA, SQLite writer, metrics endpoint.
- `pipeline/` — inference/training scripts: детекция, OCR, catalog recovery, postprocess.
- `artifacts/` — внешние артефакты: `db_hack.csv`, веса моделей, закрытые данные.
- `runtime/` — локальные результаты: jobs, CSV, logs, crops, SQLite, HTML-отчёты.

## OCR-стратегия

Мы не используем один универсальный OCR для всех полей. У ценника разные зоны имеют разную визуальную природу, поэтому pipeline использует разные инструменты под разные категории:

- **RF-DETR**: детектирует сам ценник на кадре.
- **OpenCV preprocessing**: нормализация crops, зоны, бинаризация, геометрия и фильтры качества.
- **RapidOCR**: быстрый OCR для числовых и коротких полей: обычная цена, barcode digits, sku/id, дата печати, служебные коды.
- **EasyOCR**: дополнительный OCR для сложных ценовых зон, прежде всего цена по карте и скидочные поля.
- **Tesseract rus+eng**: product-name fast path, где важны длинные строки на русском и английском.
- **PaddleOCR**: альтернативный/расширяемый путь для product-name и additional_info, оставлен в scripts для дальнейшего улучшения.
- **Catalog recovery**: сверка с `db_hack` при достаточно уверенном совпадении, чтобы OCR-шум не превращался в произвольное название товара.
- **QR decode**: QR/barcode зоны декодируются через `zxing` и OpenCV QR; финальный QR-priority pass работает только с валидно распознанным Lenta QR payload и сохраняет debug-артефакты для ручной проверки.

Такой подход устойчивее, чем “один OCR на всё”: цена, barcode и название товара требуют разных preprocessing, разных confidence-порогов и разной логики восстановления.

## Проверка артефактов и окружения

Основной one-command запуск находится в начале README. Этот раздел нужен только для диагностики, если сервис поднялся, но инференс не стартует или `/api/ready` сообщает о проблеме.

После `git lfs pull` должны быть доступны:

```text
artifacts/
  data/
    db_hack.csv
    sample.csv
  models/
    rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth
  special_symbol_templates/
    full_tags/
```

Быстрая проверка:

```bash
python tools/doctor.py
```

`doctor.py` проверяет checkpoint, `db_hack.csv`, runtime-директорию, Tesseract, Python/OCR stack и доступность GPU. Внутри Docker Python, Tesseract и tessdata уже настроены, вручную прописывать их пути не нужно.

## Артефакты модели и каталога

Веса, каталог и пример видео не лежат в обычной Git-истории как большие бинарные blob-файлы. Они хранятся через Git LFS:

```text
artifacts/
  data/
    db_hack.csv
    sample.csv
  models/
    rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth
examples/
  26_12-20.mp4
```

Почему так:

- checkpoint весит около 121 МБ;
- модель является производным артефактом обучения;
- Git LFS даёт воспроизводимость без ручной передачи файлов;
- Git LFS не раздувает обычную историю исходного кода;
- checkpoint можно заменить новым файлом без переписывания структуры проекта.

## Локальный Windows-запуск

```powershell
git lfs install
git clone https://github.com/R2NN/projecthack.git
cd projecthack
git lfs pull
docker compose up --build
```

Если на Windows мало места на системном диске, runtime можно вынести на другой диск через `.env`:

```powershell
Copy-Item .env.example .env
# затем укажите HOST_RUNTIME_DIR в .env
```

## Результаты и отчёты

После обработки создаются:

- `runtime/jobs/<job_id>/final_submission.csv` — итоговая выгрузка;
- `runtime/jobs/<job_id>/final_submission.json` — JSON-версия результата;
- `runtime/jobs/<job_id>/metrics.json` — метрики job;
- `runtime/jobs/<job_id>/pipeline_manifest.json` — служебная информация по запуску;
- `runtime/jobs/<job_id>/review.html` — HTML-отчёт по crops/результату, если сохранён;
- `runtime/jobs/<job_id>/crops/` — crops для проверки, если включено сохранение;
- `runtime/lenta_results.sqlite` — база с jobs, строками CSV, review items и агрегатами.

На главном экране crops не показываются сеткой, чтобы интерфейс оставался чистым. Для проверки сложных случаев есть отдельный HTML-отчёт и записи в SQLite.

## Метрики

API:

```text
GET /api/system/metrics
GET /api/jobs/summary
GET /api/jobs/<job_id>/metrics
```

Уже считаются:

- количество jobs, успешных и упавших задач;
- количество видео и строк ценников;
- fill-rate по `barcode`, `product_name`, `price_card`, `price_default`;
- число скидочных строк;
- средние цены;
- подозрительные случаи, где цена по карте выше обычной;
- количество crops для review;
- количество uncertain predictions;
- топ видео по числу найденных ценников.

Для сети магазинов эти метрики расширяются до:

- качества распознавания по магазину, дате, камере и маршруту;
- доли товаров без barcode или с низкой уверенностью;
- списка магазинов с большим числом ценовых аномалий;
- товаров, которые часто не матчятся с каталогом;
- SLA обработки: очередь, время на видео, throughput, ошибки workers;
- контроля промо: скидочные ценники, расхождения цены карты и QR;
- контроля выкладки относительно планограммы или expected assortment.

## Worker-ready режим

Это отдельная часть решения про промышленную эксплуатацию. В демо можно обработать одно видео на одном компьютере, но в реальной сети магазинов поток будет другим: много магазинов, много роботов, длинные видео, пики нагрузки ночью или после переоценки. Если всё завязано на один процесс, очередь быстро растёт, а любой сбой останавливает обработку.

Поэтому в проект добавлен worker-ready слой. Он не включает “100 workers” на демо-стенде, но заранее отделяет приём видео от обработки. Web/API принимает файл и создаёт задачу, а независимые обработчики забирают задачи из очереди. Так можно добавлять мощности без переписывания распознавания.

Текущий публичный стенд по умолчанию работает в `local` режиме, чтобы сохранить стабильность демонстрации. Worker-режим включается отдельно и использует тот же inference pipeline.

Что уже реализовано:

- `web/server.py` принимает видео, сохраняет upload chunks и либо запускает локальный inference, либо кладёт задачу в очередь;
- `worker_tasks` в SQLite работает как локальный backend очереди для демонстрации контракта;
- atomic claim через `BEGIN IMMEDIATE` и lease-lock защищает от ситуации, когда два worker берут один job;
- `attempts` и `max_attempts` дают retry после временных ошибок;
- `worker_heartbeats` хранит живые обработчики, host, pid и время последнего сигнала;
- queue contract уже отделён от ML-кода; отдельные worker runner/compose-сервисы можно подключать поверх этого контракта без переписывания inference pipeline;
- `GET /api/queue` показывает состояние очереди, recent tasks и активные workers.

Почему это важно:

- можно масштабировать обработку горизонтально: добавить ещё контейнеры `shelf-worker`, когда видео стало больше;
- можно разделить роли машин: GPU workers занимаются детекцией, CPU/OCR workers занимаются распознаванием и postprocess;
- можно переживать временные сбои: задача не теряется, а возвращается в очередь после lease/retry;
- можно мониторить промышленный контур: видно, сколько задач ждёт, какие workers живы, где возникли ошибки;
- можно внедрять это в инфраструктуру компании постепенно: локальная SQLite-очередь заменяется на промышленную очередь, а ML-pipeline остаётся прежним.

Важно для качества: worker-ready режим не меняет ML-логику и не влияет на точность. Детекция, OCR, catalog recovery, price sanity и dedup остаются теми же. Меняется только способ доставки задачи до обработчика.

### Как это подключается к промышленному контуру

Локальная SQLite-очередь нужна для воспроизводимой проверки на одном компьютере. В инфраструктуре сети магазинов меняется только backend очереди и storage:

```text
Web/API
  -> очередь задач: SQLite сейчас, RabbitMQ/Kafka/SQS в промышленном контуре
  -> object storage: runtime сейчас, S3/MinIO в промышленном контуре
  -> GPU workers: RF-DETR detection
  -> CPU/OCR workers: OCR, parsing, postprocess
  -> reducer: merge, tracking/dedup, catalog recovery, CSV, metrics
  -> DB/BI: SQLite сейчас, PostgreSQL/ClickHouse в промышленном контуре
```

Такой контракт позволяет добавлять обработчики горизонтально: один worker может обрабатывать одно видео или батч crops, а общий reducer собирает результат в тот же формат `sample.csv`. При росте нагрузки добавляются новые контейнеры `shelf-worker`, при падении worker задача возвращается в очередь после истечения lease или retry.

### Запуск queue-режима

В clean appliance основной путь — локальный режим `docker compose up --build`. Queue-ready слой оставлен как архитектурный контракт внутри web/API: `WORKER_MODE=queue` кладёт задачи в SQLite-очередь, а внешний worker runner может забирать их через тот же `worker_queue.py` без изменения ML pipeline. Отдельный worker compose-профиль не включён в минимальную поставку, чтобы не плодить конкурирующие способы запуска.

## Почему решение масштабируется

Pipeline естественно раскладывается на map-reduce:

```text
map workers:
  видео / кадры / crops
  -> detection
  -> OCR
  -> field candidates

reduce stage:
  candidates
  -> tracking merge
  -> catalog recovery
  -> price sanity
  -> dedup
  -> CSV + metrics
```

При росте нагрузки можно добавить новые workers:

- GPU workers для RF-DETR detection;
- CPU/OCR workers для OCR и postprocess;
- отдельный reducer для merge/dedup/export;
- общий object storage для видео, crops и результатов;
- PostgreSQL/ClickHouse вместо локального SQLite.

Это важно для крупного бизнеса: система не привязана к одному компьютеру и не требует ручной обработки каждого видео.

## Ресурсы

Минимально для UI/API:

- 2 vCPU — два виртуальных CPU-потока, не два физических процессора;
- 2-4 GB RAM;
- 5 GB disk под runtime;
- Docker или Python 3.11+.

Для полного локального инференса:

- GPU с 6 GB VRAM может быть достаточен для RF-DETR small на текущих настройках;
- 8 GB VRAM — рекомендуемый минимум с запасом для стабильной работы;
- 16 GB RAM;
- 20-30 GB disk под модели, runtime и crops;
- для Docker-сборки с `INSTALL_PIPELINE_DEPS=true` нужно 25-35 GB свободного места в Docker storage, потому что GPU-зависимости включают Torch/CUDA wheels;
- Python 3.12 окружение с RF-DETR/OCR-зависимостями;
- Tesseract + русский language pack;
- `db_hack.csv` и checkpoint детектора.

CPU-only режим технически возможен для части pipeline, но для демонстрации и массовой обработки он слишком медленный. Практичный вариант — GPU для detection и CPU/OCR workers для остального.

## Лицензии и чистота поставки

Ниже — инженерная проверка лицензий ключевых компонентов, а не финальное юридическое заключение. Цель была выбрать стек, который выглядит пригодным для корпоративного использования в Ленте: без AGPL/commercial-only детектора в основном inference path, с permissive-лицензиями у модели детекции, OCR и barcode/QR-decoder библиотек.

Ключевые компоненты runtime/inference:

| Компонент | Роль в решении | Лицензия / статус |
|---|---|---|
| [RF-DETR / `rfdetr`](https://github.com/roboflow/rf-detr) | детекция ценников, обучение нового checkpoint | Apache-2.0 для open-source пакета и Apache-designated весов; используются RF-DETR Small/обычная open-source ветка, не RF-DETR Plus/X/2XL |
| Собственный checkpoint `rfdetr_small_price_tag...pth` | дообученные веса детектора ценников | производный артефакт обучения на размеченных данных проекта; для промышленного использования нужно подтвердить права на training data и факт, что базовые веса RF-DETR использовались в Apache-совместимом варианте |
| [PyTorch / TorchVision](https://github.com/pytorch/pytorch) | inference/training backend для RF-DETR | BSD-style / BSD-3-Clause |
| [OpenCV](https://opencv.org/license/) и `opencv-python-headless` | crops, preprocessing, QR geometry, image processing | OpenCV 4.5+ — Apache-2.0; Python packaging scripts — MIT |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) + `tesseract-ocr-eng/rus` | OCR product-name и длинных строк | Apache-2.0; tessdata/tessdata_fast модели также публикуются как Apache-2.0 |
| [EasyOCR](https://pypi.org/project/easyocr/) | OCR сложных ценовых зон | Apache-2.0 |
| [RapidOCR / `rapidocr-onnxruntime`](https://pypi.org/project/rapidocr-onnxruntime/) | быстрый OCR числовых и коротких зон | Apache-2.0 |
| [zxing-cpp](https://github.com/zxing-cpp/zxing-cpp) | barcode/QR decode | Apache-2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | backend для OCR-моделей | MIT |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | модельные зависимости RF-DETR/OCR stack | Apache-2.0 |
| [Roboflow Supervision](https://supervision.roboflow.com/latest/about/) | CV utility layer | MIT |
| NumPy, SciPy, pandas, scikit-learn, psutil, PyYAML, Pillow | numeric/data/runtime utilities | в основном permissive-лицензии BSD/MIT/HPND-like |
| PaddleOCR/PaddlePaddle | optional/development OCR path в полном requirements | Apache-2.0; не является обязательным для lean appliance inference |

Docker-окружение дополнительно использует Ubuntu packages, PowerShell base image, PyTorch CUDA base image, NVIDIA CUDA runtime/toolkit и `ffmpeg`. Они не меняют ML-логику, но для промышленной поставки контейнера нужно отдельно сохранить/проверить notices и условия этих системных компонентов. Особенно стоит перепроверить `ffmpeg`, потому что его итоговая лицензия зависит от сборки и включённых codecs, а GPU-образ зависит от условий NVIDIA CUDA/PyTorch container stack.

Чистота поставки по данным и артефактам остаётся отдельным требованием: рабочие видео, закрытые обучающие датасеты, runtime-логи, SQLite, crops, debug/review jobs, презентационные файлы и временные архивы не должны попадать в обычную Git-историю. В репозитории остаются код, Docker/Compose, проверки окружения, `sample.csv` как схема, а крупные воспроизводимые артефакты идут через Git LFS.

Перед передачей в production всё равно нужно сделать финальную юридическую проверку: зафиксировать полный список прямых и транзитивных зависимостей, приложить license notices к Docker image/дистрибутиву, подтвердить права на обучающие данные и каталог `db_hack.csv`, а также проверить актуальные условия базовых образов, CUDA/NVIDIA и системных пакетов на дату внедрения.


## Структура

```text
.
├── web/                 # UI, API, jobs, metrics, SQLite writer, worker process
├── pipeline/            # inference/training pipeline без весов и датасетов
├── artifacts/           # Git LFS и подключаемые артефакты модели/каталога
├── tools/               # проверки окружения
├── Dockerfile
├── Dockerfile.gpu
├── docker-compose.yml
└── README.md
```

## Проверки

```bash
python -m py_compile web/server.py web/worker_queue.py tools/doctor.py tools/check_artifacts.py
node --check web/app.js
python -m compileall pipeline/scripts
python tools/doctor.py
docker compose config
```

`doctor.py` покажет понятные ошибки, если Git LFS артефакты ещё не скачаны или runtime недоступен.
