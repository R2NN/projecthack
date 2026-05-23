# Lenta Shelf Vision

Минимальный воспроизводимый проект для распознавания ценников на видео: RF-DETR детектирует ценники, OCR извлекает цены и названия, `db_hack.csv` помогает восстановить карточки товара, QR-priority постобработка перезаписывает поля ценника данными из валидного QR, а web/API дает загрузку видео и скачивание CSV/JSON.

## Структура

```text
.
  README.md
  Dockerfile
  docker-compose.yml
  .env.example
  configs/default.yaml
  artifacts/
    models/
    data/
    special_symbol_templates/
    training_data/
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
```

В проект не перенесены старые R&D jobs/runs/crops, debug HTML, review-файлы, временные архивы и альтернативные варианты запуска. Основной путь эксплуатации: Docker Compose.

## Артефакты

В GitHub крупные runtime-файлы (`checkpoint`, `db_hack.csv` и пример видео) хранятся через Git LFS. После `git clone` выполните `git lfs pull`; при распространении исходников без LFS-файлов положите вручную:

```text
artifacts/models/rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth
artifacts/data/db_hack.csv
artifacts/data/sample.csv
artifacts/special_symbol_templates/full_tags/
```

Проверка:

```powershell
python .\tools\doctor.py
```

## One-Command Appliance

Запуск CPU/auto-режима:

```powershell
docker compose up --build
```

По умолчанию Dockerfile ставит `pipeline/requirements-appliance.txt`: это lean runtime-набор без CUDA wheel’ов. Полный `pipeline/requirements.txt` оставлен для разработки и воспроизведения исходной ML-среды, но он слишком тяжелый для дефолтного one-command запуска.

Во время Docker build образ заранее загружает модели EasyOCR в `/opt/easyocr`, чтобы job не пытался скачивать OCR-веса уже во время инференса. Для первой сборки все равно нужен интернет: Docker скачивает base image, Python-пакеты и OCR-веса.

После старта:

```text
Web UI:    http://127.0.0.1:8000/
Health:    http://127.0.0.1:8000/api/health
Ready:     http://127.0.0.1:8000/api/ready
Runtime:   ./runtime/jobs/
```

Compose подключает:

```text
./artifacts -> /app/artifacts:ro
./runtime   -> /data/runtime
```

Если модели, каталога, Tesseract или OCR stack не хватает, сервис не должен падать traceback'ом в UI: `/api/health` остается живым, а `/api/ready` и `tools/doctor.py` возвращают понятные коды вроде `MODEL_NOT_FOUND`, `CATALOG_NOT_FOUND`, `TESSERACT_NOT_READY`, `OCR_STACK_NOT_READY`.

GPU-профиль для NVIDIA:

```powershell
docker compose --profile gpu up --build shelf-vision-gpu
```

GPU-сервис по умолчанию доступен на `http://127.0.0.1:8001/`, чтобы не конфликтовать с CPU-сервисом. Нужны NVIDIA driver и NVIDIA Container Toolkit. Для GPU compose использует отдельный `Dockerfile.gpu` на базе `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`: CUDA/PyTorch приходят из базового образа, а `pipeline/requirements-appliance-gpu.txt` ставит только дополнительные OCR/web зависимости. Это переносимый NVIDIA CUDA-профиль, а не привязка к конкретной модели видеокарты; AMD/Intel GPU требуют отдельного ROCm/DirectML-варианта и этим compose-профилем не покрываются. CPU fallback включается через `INFERENCE_DEVICE=auto` или `INFERENCE_DEVICE=cpu`; он может быть существенно медленнее и зависит от поддержки CPU в установленном ML stack.

Проверить, видит ли Docker GPU:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## Переменные

Скопируйте `.env.example` в `.env`, если нужно изменить порты, runtime или volume:

```powershell
Copy-Item .env.example .env
```

Главные настройки:

```text
APP_PORT=8000
RUNTIME_DIR=/data/runtime
MODEL_PATH=/app/artifacts/models/rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth
CATALOG_PATH=/app/artifacts/data/db_hack.csv
SAVE_INPUT_VIDEO=false
SAVE_REVIEW_CROPS=true
WORKER_MODE=local
INFERENCE_DEVICE=auto
```

Проект не зависит от конкретного диска. На Windows тяжелые директории можно вынести на другой диск через `HOST_RUNTIME_DIR` или `RUNTIME_DIR`, без правки кода.

## API

Стабильный appliance API:

```text
GET  /api/v1/health
POST /api/v1/jobs
GET  /api/v1/jobs/<job_id>
GET  /api/v1/jobs/<job_id>/result.csv
GET  /api/v1/jobs/<job_id>/metrics.json
GET  /api/v1/jobs/<job_id>/review.html
```

Пример:

```powershell
curl.exe -F "videos=@.\examples\26_12-20.mp4;type=video/mp4" http://127.0.0.1:8000/api/v1/jobs
curl.exe http://127.0.0.1:8000/api/v1/jobs/<job_id>
curl.exe -o final_submission.csv http://127.0.0.1:8000/api/v1/jobs/<job_id>/result.csv
```

Старые web endpoints сохранены для UI:

```text
POST /api/infer
GET  /api/jobs/<job_id>/status
GET  /api/jobs/<job_id>/csv
GET  /api/jobs/<job_id>/json
GET  /api/system/metrics
GET  /api/jobs/summary
GET  /api/jobs/<job_id>/metrics
```

## Runtime Layout

Каждый запуск сохраняется в:

```text
runtime/jobs/<job_id>/
  input_meta.json
  state.json
  final_submission.csv
  final_submission.json
  metrics.json
  pipeline_manifest.json
  review.html
  crops/
  logs/
```

`pipeline_manifest.json` фиксирует `pipeline_version`, пути и SHA256 модели, каталога и входного видео, а также статусы и время этапов.

## Monitoring Dashboard

Web UI содержит встроенный раздел мониторинга без обязательных Prometheus/Grafana:

- System: CPU, RAM, GPU/VRAM, свободное место runtime-диска, workers/OCR jobs.
- Queue / Jobs: queued/running/done/failed, среднее время и последние jobs.
- Current Job: текущий этап, прогресс и оценка "до конца осталось".
- Pipeline Timing: breakdown времени по этапам из `timings.csv`.
- Result Quality: rows, fill-rate по `product_name`, `price_card`, `price_default`, `barcode`, `discount_amount`, suspicious rows и статусы CSV/JSON/review/debug/crops.

В production эти же данные можно экспортировать во внешнюю систему мониторинга, но для демо они доступны прямо в web UI.

## Инференс Без Сайта

Локальный запуск остается доступен, если установлены Python-зависимости, PowerShell и Tesseract:

```powershell
.\pipeline\run_inference.ps1 `
  -VideoPath .\examples\26_12-20.mp4 `
  -VideoId 26_12-20 `
  -Device auto
```

Основные outputs:

```text
runtime/runs/<run>/final_submission.csv
runtime/runs/<run>/timings.csv
runtime/runs/<run>/qr_priority/summary.json
runtime/runs/<run>/qr_priority/qr_priority_debug.html
```

## Обучение

Один основной сценарий:

```powershell
.\pipeline\run_train_rfdetr.ps1
```

Обучающие видео и разметка не входят в минимальный пакет. Для воспроизводимого обучения положите annotated training data в `artifacts/training_data` или передайте `-DataRoot`; итоговый checkpoint должен попасть в `artifacts/models/`.

## Ограничения

Первый Docker build может быть долгим: ML/OCR зависимости и Torch тяжелые даже в lean-варианте. GPU build скачивает крупный PyTorch CUDA base image и требует заметно больше свободного места в Docker Desktop/WSL, но не собирает CUDA через pip. Для полного inference нужны модель, `db_hack.csv`, `sample.csv`, templates, Tesseract rus+eng и достаточно места в runtime. CPU-режим предназначен как fallback и может быть слишком медленным для больших видео.
