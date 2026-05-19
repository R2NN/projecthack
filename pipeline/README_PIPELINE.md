# Inference pipeline

В этой папке лежат исходники pipeline для детекции ценников, OCR, catalog recovery и сборки итогового CSV.

## Что входит в Git

- PowerShell wrappers:
  - `run_inference_no_ensemble.ps1`
  - `run_full_catalog_recovery_v19.ps1`
  - `run_train_rfdetr_all_data.ps1`
  - `run_train_rfdetr_except_26_12_20.ps1`
  - `run_quality_pipeline_43_15.ps1`
- Python scripts в `scripts/`.
- `requirements-gpu.txt` с зафиксированным окружением рабочего GPU-стенда.
- `data/sample.csv` как схема итоговой отправки.

## Что не входит в Git

- веса моделей;
- `db_hack.csv`;
- видео;
- обучающие датасеты;
- runtime outputs;
- отчёты и handout-файлы хакатона.

Эти артефакты подключаются отдельно через `artifacts/` или через переменные окружения.

## Основной запуск

Через web/API pipeline вызывается командой:

```powershell
.\run_inference_no_ensemble.ps1 `
  -Python python `
  -VideoPath <path-to-video.mp4> `
  -VideoId <video-id> `
  -RunRoot <output-dir> `
  -DetectorCheckpoint <path-to-checkpoint.pth> `
  -CatalogPath <path-to-db_hack.csv>
```

На Windows можно указывать локальный Python из GPU-окружения, например:

```powershell
-Python A:\rfdetr_envs\lenta-rfdetr-gpu\Scripts\python.exe
```

В Docker используется `pwsh` и `python` внутри контейнера.

## Основные этапы

1. RF-DETR детектирует ценники на выбранных кадрах.
2. Tracking объединяет повторные detections в tracks.
3. Crop quality выбирает лучшие crops для OCR.
4. Numeric OCR достаёт цены, скидки, barcode, QR-поля.
5. Product-name OCR распознаёт название товара.
6. Catalog recovery сверяет OCR с `db_hack`.
7. Price sanity исправляет грубые выбросы цен.
8. Dedup убирает повторные строки.
9. Export собирает `final_submission.csv`.

## Обучение

Для полного обучения детектора на всех размеченных данных:

```powershell
.\run_train_rfdetr_all_data.ps1 -Python <python> -Epochs 8
```

Сгенерированные `datasets/`, `runs/`, `models/` не коммитятся. Чекпойнт после обучения хранится как внешний артефакт.
