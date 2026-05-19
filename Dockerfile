FROM mcr.microsoft.com/powershell:7.5-ubuntu-24.04

ARG INSTALL_PIPELINE_DEPS=false

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    LENTA_WEB_RUNTIME_ROOT=/runtime \
    LENTA_PIPELINE_ROOT=/app/pipeline \
    LENTA_PIPELINE_SCRIPT=/app/pipeline/run_inference_no_ensemble.ps1 \
    LENTA_POWERSHELL=pwsh \
    LENTA_PIPELINE_PYTHON=/opt/venv/bin/python \
    LENTA_TESSERACT_EXE=/usr/bin/tesseract \
    LENTA_TESSDATA_DIR=/usr/share/tesseract-ocr/5/tessdata \
    LENTA_CATALOG_PATH=/artifacts/db_hack.csv \
    LENTA_DETECTOR_CHECKPOINT=/artifacts/models/rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth \
    LENTA_RESULTS_DB=/runtime/lenta_results.sqlite \
    LENTA_WORKER_QUEUE_DB=/runtime/worker_queue.sqlite \
    LENTA_JOB_EXECUTION_MODE=local \
    LENTA_RETRAIN_CONFIG=/runtime/retrain_config.json \
    LENTA_UNCERTAIN_ROOT=/runtime/uncertain_predictions

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
      python3 \
      python3-pip \
      python3-venv \
      python-is-python3 \
      tesseract-ocr \
      tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip

WORKDIR /app

COPY web /app/web
COPY pipeline /app/pipeline

RUN if [ "$INSTALL_PIPELINE_DEPS" = "true" ]; then \
      /opt/venv/bin/python -m pip install --no-cache-dir -r /app/pipeline/requirements-gpu.txt; \
    fi

VOLUME ["/runtime", "/artifacts"]
EXPOSE 5173

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5173/api/health', timeout=3).read()" || exit 1

CMD ["python", "/app/web/server.py", "--host", "0.0.0.0", "--port", "5173"]
