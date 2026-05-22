# Lenta Shelf Vision

Минимальный репозиторий для распознавания ценников на видео: детекция RF-DETR, OCR цен и названий, восстановление по каталогу `db_hack.csv`, QR-priority постобработка и web/API для загрузки видео и скачивания итоговых CSV/JSON.

## Структура

```text
.
  README.md
  Dockerfile
  docker-compose.yml
  .env.example
  configs/
    default.yaml
  artifacts/
    models/                         # RF-DETR checkpoint
    data/                           # db_hack.csv, sample.csv
    special_symbol_templates/       # шаблоны спецсимволов
    training_data/                  # опциональные данные для обучения
  pipeline/
    run_inference.ps1
    run_train_rfdetr.ps1
    requirements.txt
    scripts/
  web/
    server.py
    index.html
    app.js
    styles.css
  tools/
    doctor.py
    check_artifacts.py
  runtime/
    .gitkeep
```

В репозиторий не перенесены старые R&D-эксперименты, debug HTML, review-отчеты, временные jobs/runs/crops, старые архивы, кэши и альтернативные варианты запуска.

## Артефакты

Обязательные файлы по умолчанию:

```text
artifacts/models/rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth
artifacts/data/db_hack.csv
artifacts/data/sample.csv
artifacts/special_symbol_templates/full_tags/track_*.jpg
```

Проверка:

```powershell
python .\tools\check_artifacts.py
python .\tools\doctor.py
```

## Переменные окружения

Скопируйте `.env.example` в `.env` и настройте только то, что отличается на машине:

```powershell
Copy-Item .env.example .env
```

Главные пути:

```text
RUNTIME_DIR=./runtime
ARTIFACTS_DIR=./artifacts
LENTA_PIPELINE_PYTHON=python
LENTA_TESSERACT_EXE=tesseract
LENTA_TESSDATA_DIR=
```

Проект не зависит от диска `A:` или Windows-абсолютных путей. На Windows можно вынести тяжелый runtime на другой диск, например:

```powershell
$env:RUNTIME_DIR = 'A:\lenta_runtime'
```

Большие данные держите в `artifacts/` или подключайте через volume/config. Runtime, outputs и cache по умолчанию пишутся относительно корня проекта.

## Установка

Рекомендуется Python 3.11, PowerShell, Tesseract с `rus` и `eng` tessdata, GPU с CUDA для быстрого RF-DETR inference/training.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\pipeline\requirements.txt
```

Если Tesseract не в `PATH`, задайте в `.env`:

```text
LENTA_TESSERACT_EXE=D:\tools\tesseract\tesseract.exe
LENTA_TESSDATA_DIR=D:\tools\tesseract\tessdata
```

## Инференс без сайта

```powershell
.\pipeline\run_inference.ps1 `
  -VideoPath .\examples\26_12-20.mp4 `
  -VideoId 26_12-20
```

Основные переопределения:

```powershell
.\pipeline\run_inference.ps1 `
  -VideoPath .\input\video.mp4 `
  -RunRoot .\runtime\runs\my_video `
  -Python .\.venv\Scripts\python.exe `
  -TesseractExe D:\tools\tesseract\tesseract.exe `
  -TessdataDir D:\tools\tesseract\tessdata `
  -NumericJobs 8 `
  -TesseractJobs 8
```

Главные outputs:

```text
runtime/runs/<run>/final_submission.csv
runtime/runs/<run>/final_submission_before_qr.csv
runtime/runs/<run>/inference_summary.json
runtime/runs/<run>/timings.csv
runtime/runs/<run>/qr_priority/summary.json
runtime/runs/<run>/qr_priority/qr_decode_candidates.csv
runtime/runs/<run>/qr_priority/qr_priority_debug.html
```

QR-priority применяется последним шагом. Если QR payload валиден, поля `barcode`, `price_default` и `price_card` заполняются из QR с высшим приоритетом.

Для проверки QR на уже сохраненных full-tag кропах без исходного видео:

```powershell
python .\pipeline\scripts\apply_fulltag_qr_priority.py `
  --submission-csv .\runtime\crop_job\submission.csv `
  --full-tags-dir .\runtime\crop_job\full_tags `
  --output-csv .\runtime\crop_job\submission_qr_priority.csv `
  --debug-dir .\runtime\crop_job\qr_priority
```

## Web/API

```powershell
python .\web\server.py --host 127.0.0.1 --port 5173
```

Откройте:

```text
http://127.0.0.1:5173/
```

API:

```text
GET  /api/health
POST /api/infer
POST /api/upload/create
POST /api/jobs/<job_id>/upload-chunk
POST /api/jobs/<job_id>/start
GET  /api/jobs/<job_id>/status
GET  /api/jobs/<job_id>/csv
GET  /api/jobs/<job_id>/json
GET  /api/system/metrics
GET  /api/jobs/summary
GET  /api/jobs/<job_id>/metrics
```

Web UI принимает один или несколько MP4, а также ZIP-архив с MP4 внутри. Результаты доступны как CSV и JSON.

`POST /api/infer` принимает multipart upload с полем `videos` и поддерживает MP4/ZIP:

```powershell
curl.exe -F "videos=@.\examples\26_12-20.mp4;type=video/mp4" http://127.0.0.1:5173/api/infer
```

Web-сервер использует тот же `pipeline/run_inference.ps1`, поэтому все пути берутся из `.env`, переменных окружения или аргументов pipeline.

## Monitoring Dashboard

Web UI включает легкий встроенный раздел `Monitoring` без обязательных Prometheus/Grafana. Он работает локально и в Docker/one-command режиме, читает `runtime/jobs`, state-файлы, `timings.csv`, итоговые CSV и review/debug артефакты.

В дашборде доступны:

- System: CPU, RAM, GPU/VRAM при наличии `nvidia-smi`, свободное место runtime-диска, workers/OCR jobs.
- Queue / Jobs: queued/running/done/failed, среднее время обработки и последние jobs.
- Current Job: статус, текущий этап, progress, ETA с низкой уверенностью во время выполнения.
- Pipeline Timing: breakdown по этапам inference из `timings.csv`.
- Result Quality: rows, fill-rate ключевых полей, suspicious rows, artifact status, ссылки на CSV/JSON/review/debug.

API мониторинга:

```text
GET /api/system/metrics
GET /api/jobs/summary
GET /api/jobs/<job_id>/metrics
```

Если GPU или метрики временно недоступны, API возвращает понятные fallback-значения и UI не падает. Для production эти же данные можно экспортировать наружу в Prometheus/Grafana, но для демо и локальной эксплуатации они доступны прямо на странице.

## Docker

```powershell
docker compose up --build
```

По умолчанию compose подключает:

```text
./artifacts -> /artifacts:ro
./runtime   -> /runtime
```

## Обучение

Обучение вынесено в один основной сценарий:

```powershell
.\pipeline\run_train_rfdetr.ps1
```

По умолчанию скрипт ожидает размеченные данные в:

```text
artifacts/training_data
```

Если данные лежат отдельно:

```powershell
.\pipeline\run_train_rfdetr.ps1 `
  -DataRoot D:\datasets\lenta_training `
  -DatasetDir .\runtime\datasets\rfdetr_price_tags `
  -RunDir .\runtime\runs\rfdetr_train `
  -ModelOutputPath .\artifacts\models\rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth
```

В текущий пакет обучающие видео и разметка не включены, поэтому обучение воспроизводимо после добавления исходных annotated training данных в `artifacts/training_data` или передачи `-DataRoot`.

## Требования к диску

Перед большими прогонами проверьте место:

```powershell
Get-PSDrive
```

Необязательные тяжелые директории можно переносить без изменения кода:

```powershell
$env:RUNTIME_DIR = 'A:\lenta_runtime'
$env:ARTIFACTS_DIR = 'A:\lenta_artifacts'
$env:PADDLE_CACHE_DIR = 'A:\lenta_runtime\cache\paddle'
```

## Быстрая диагностика

```powershell
python .\tools\doctor.py
```

`doctor.py` проверяет Python-модули, PowerShell, Tesseract, обязательные артефакты и свободное место в runtime.
