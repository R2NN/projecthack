FROM mcr.microsoft.com/powershell:7.4-debian-12

ARG INSTALL_PIPELINE_DEPS=false

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    LENTA_WEB_RUNTIME_ROOT=/runtime \
    LENTA_PIPELINE_ROOT=/app/pipeline \
    LENTA_PIPELINE_SCRIPT=/app/pipeline/run_inference_no_ensemble.ps1 \
    LENTA_POWERSHELL=pwsh \
    LENTA_PIPELINE_PYTHON=python \
    LENTA_CATALOG_PATH=/artifacts/db_hack.csv \
    LENTA_DETECTOR_CHECKPOINT=/artifacts/models/rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth \
    LENTA_RESULTS_DB=/runtime/lenta_results.sqlite \
    LENTA_RETRAIN_CONFIG=/runtime/retrain_config.json \
    LENTA_UNCERTAIN_ROOT=/runtime/uncertain_predictions

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
      python3 \
      python3-pip \
      python-is-python3 \
      tesseract-ocr \
      tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY web /app/web
COPY pipeline /app/pipeline

RUN if [ "$INSTALL_PIPELINE_DEPS" = "true" ]; then \
      python -m pip install --upgrade pip && \
      python -m pip install --no-cache-dir -r /app/pipeline/requirements-gpu.txt; \
    fi

VOLUME ["/runtime", "/artifacts"]
EXPOSE 5173

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5173/api/health', timeout=3).read()" || exit 1

CMD ["python", "/app/web/server.py", "--host", "0.0.0.0", "--port", "5173"]
