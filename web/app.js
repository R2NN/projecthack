const MAX_VIDEO_SIZE_BYTES = 512 * 1024 * 1024;
const UPLOAD_CHUNK_BYTES = 2 * 1024 * 1024;
const API_UPLOAD_CREATE_ENDPOINT = "/api/upload/create";
const API_JOB_UPLOAD_CHUNK_ENDPOINT = (jobId, uploadId, offset) =>
  `/api/jobs/${encodeURIComponent(jobId)}/upload-chunk?upload_id=${encodeURIComponent(uploadId)}&offset=${offset}`;
const API_JOB_START_ENDPOINT = (jobId) => `/api/jobs/${encodeURIComponent(jobId)}/start`;
const API_JOB_STATUS_ENDPOINT = (jobId) => `/api/jobs/${encodeURIComponent(jobId)}/status`;
const API_JOB_CSV_ENDPOINT = (jobId) => `/api/jobs/${encodeURIComponent(jobId)}/csv`;
const API_JOB_JSON_ENDPOINT = (jobId) => `/api/jobs/${encodeURIComponent(jobId)}/json`;
const API_SYSTEM_METRICS_ENDPOINT = "/api/system/metrics";
const API_JOBS_SUMMARY_ENDPOINT = "/api/jobs/summary";
const API_JOB_METRICS_ENDPOINT = (jobId) => `/api/jobs/${encodeURIComponent(jobId)}/metrics`;
const STATUS_POLL_INTERVAL_MS = 1400;
const DASHBOARD_SYSTEM_INTERVAL_MS = 3000;
const DASHBOARD_JOBS_INTERVAL_MS = 7000;

if ("scrollRestoration" in history) {
  history.scrollRestoration = "manual";
}

if (!window.location.hash) {
  window.scrollTo(0, 0);
  window.addEventListener(
    "load",
    () => {
      window.scrollTo(0, 0);
    },
    { once: true },
  );
}

const TIMING_LABELS = {
  "01_detection_tracking": "detection",
  "02_export_ocr_zones": "OCR zones",
  "03_numeric_ocr": "numeric OCR / QR",
  "04_tesseract_product_name": "product OCR",
  "05_crop_quality_reselect": "crop quality",
  "06_catalog_recovery_db_hack": "catalog recovery",
  "07_fill_aux_barcode": "aux / barcode",
  "08_deduplicate_rows": "dedup",
  "09_export_final_submission_with_qr_priority": "final CSV / QR priority",
};

const STATUS_LABELS = {
  uploading: "загрузка",
  queued: "в очереди",
  running: "в работе",
  done: "готово",
  failed: "ошибка",
  idle: "ожидание",
  unknown: "неизвестно",
  unavailable: "недоступно",
};

const state = {
  videos: [],
  activeId: "",
  csvUrl: "",
  jsonUrl: "",
  isProcessing: false,
  previewCollapsed: false,
  dashboard: {
    selectedJobId: "",
    activeJobId: "",
    lastJobs: [],
    systemTimer: 0,
    jobsTimer: 0,
  },
};

const dropzone = document.querySelector("#dropzone");
const uploadCard = document.querySelector("#upload-card");
const videoInput = document.querySelector("#video-input");
const pickFile = document.querySelector("#pick-file");
const errorMessage = document.querySelector("#error-message");
const filesPanel = document.querySelector("#files-panel");
const filesCount = document.querySelector("#files-count");
const fileList = document.querySelector("#file-list");
const clearFiles = document.querySelector("#clear-files");
const previewWrap = document.querySelector("#video-preview");
const previewVideo = document.querySelector("#preview-video");
const previewCaption = document.querySelector("#preview-caption");
const openPreview = document.querySelector("#open-preview");
const togglePreview = document.querySelector("#toggle-preview");
const processingPanel = document.querySelector("#processing-panel");
const processingTitle = document.querySelector("#processing-title");
const processingText = document.querySelector("#processing-text");
const progressBar = document.querySelector("#progress-bar");
const noticeText = document.querySelector("#notice-text");
const runButton = document.querySelector("#run-button");
const downloadCsv = document.querySelector("#download-csv");
const downloadJson = document.querySelector("#download-json");
const downloadReview = document.querySelector("#download-review");
const downloadReviewZip = document.querySelector("#download-review-zip");
const videoModal = document.querySelector("#video-modal");
const videoModalBackdrop = document.querySelector("#video-modal-backdrop");
const modalClose = document.querySelector("#modal-close");
const modalVideo = document.querySelector("#modal-video");
const dashboardRefresh = document.querySelector("#dashboard-refresh");
const metricCpu = document.querySelector("#metric-cpu");
const metricRam = document.querySelector("#metric-ram");
const metricGpu = document.querySelector("#metric-gpu");
const metricDisk = document.querySelector("#metric-disk");
const metricWorkers = document.querySelector("#metric-workers");
const jobsAverage = document.querySelector("#jobs-average");
const jobsQueued = document.querySelector("#jobs-queued");
const jobsRunning = document.querySelector("#jobs-running");
const jobsDone = document.querySelector("#jobs-done");
const jobsFailed = document.querySelector("#jobs-failed");
const jobList = document.querySelector("#job-list");
const currentJobStatus = document.querySelector("#current-job-status");
const currentJobBody = document.querySelector("#current-job-body");
const timingTotal = document.querySelector("#timing-total");
const timingBars = document.querySelector("#timing-bars");
const qualityRows = document.querySelector("#quality-rows");
const qualityBody = document.querySelector("#quality-body");

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 Б";
  }

  const units = ["Б", "КБ", "МБ", "ГБ"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "длительность уточняется";
  }

  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${rest}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatSecondsCompact(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value <= 0) {
    return "--";
  }
  if (value < 60) {
    return `${Math.round(value)}s`;
  }
  if (value < 3600) {
    return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
  }
  const hours = Math.floor(value / 3600);
  const minutes = Math.round((value % 3600) / 60);
  return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
}

function formatPercent(value) {
  const percent = Number(value);
  return Number.isFinite(percent) ? `${percent.toFixed(percent >= 10 ? 0 : 1)}%` : "--";
}

function formatDateShort(value) {
  if (!value) {
    return "--";
  }
  const date = parseApiDate(value);
  if (Number.isNaN(date.getTime())) {
    return "--";
  }
  return date.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function severityClass(percent) {
  const value = Number(percent);
  if (!Number.isFinite(value)) {
    return "is-muted";
  }
  if (value >= 90) {
    return "is-critical";
  }
  if (value >= 75) {
    return "is-warning";
  }
  return "is-ok";
}

function parseApiDate(value) {
  const text = String(value || "").trim();
  if (!text) {
    return new Date(Number.NaN);
  }
  const normalized = text.replace(/([+-]\d{2})(\d{2})$/, "$1:$2");
  return new Date(normalized);
}

function liveProcessingSeconds(metrics) {
  const base = Number(metrics.processing_sec || 0);
  if (!["running", "queued", "uploading"].includes(metrics.status)) {
    return base;
  }
  const job = state.dashboard.lastJobs.find((item) => item.job_id === metrics.job_id);
  const createdAt = parseApiDate(job?.created_at);
  if (Number.isNaN(createdAt.getTime())) {
    return base;
  }
  return Math.max(base, (Date.now() - createdAt.getTime()) / 1000);
}

function etaPhrase(metrics) {
  if (!metrics || !["running", "queued", "uploading"].includes(metrics.status)) {
    return metrics?.status === "done" ? "Обработка завершена" : "";
  }
  if (metrics.eta_sec === null || metrics.eta_sec === undefined) {
    return "До конца осталось: оцениваем";
  }
  return `До конца осталось примерно ${formatSecondsCompact(metrics.eta_sec)}`;
}

function throughputText(metrics) {
  const processing = Number(metrics?.processing_sec || 0);
  const duration = Number(metrics?.video_duration_sec || 0);
  if (!Number.isFinite(processing) || !Number.isFinite(duration) || processing <= 0 || duration <= 0) {
    return "";
  }
  const minutesPerMinute = processing / duration;
  const formatted = minutesPerMinute >= 10 ? Math.round(minutesPerMinute) : minutesPerMinute.toFixed(1);
  return `Скорость: 1 мин видео ≈ ${formatted} мин обработки`;
}

function jobLogUrl(job) {
  if (job.log_url) {
    return job.log_url;
  }
  const jobId = String(job.job_id || "");
  const filename = String(job.filename || "");
  if (!jobId || !filename) {
    return "";
  }
  const stem = filename.replace(/\.[^.]+$/, "");
  return `/api/jobs/${encodeURIComponent(jobId)}/artifact?path=${encodeURIComponent(`outputs/${stem}.log`)}`;
}

function statusLabel(status) {
  return STATUS_LABELS[status] || status || STATUS_LABELS.unknown;
}

function progressHtml(percent, className = "") {
  const value = Math.max(0, Math.min(100, Number(percent) || 0));
  return `<div class="metric-progress ${className}"><span style="width: ${value}%"></span></div>`;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await readErrorResponse(response));
  }
  return response.json();
}

function renderMetricCard(element, title, value, detail, percent, options = {}) {
  if (!element) {
    return;
  }
  const severity = options.severity || severityClass(percent);
  element.innerHTML = `
    <div class="metric-top">
      <span>${escapeHtml(title)}</span>
      <strong>${escapeHtml(value)}</strong>
    </div>
    ${Number.isFinite(Number(percent)) ? progressHtml(percent, severity) : ""}
    <p>${escapeHtml(detail || "")}</p>
  `;
  element.classList.remove("is-ok", "is-warning", "is-critical", "is-muted");
  element.classList.add(severity);
}

function renderSystemMetrics(metrics) {
  const ram = metrics.ram || {};
  const disk = metrics.disk || {};
  const gpu = metrics.gpu || {};
  const workers = metrics.workers || {};
  renderMetricCard(metricCpu, "CPU", formatPercent(metrics.cpu_percent), "локальный inference host", Number(metrics.cpu_percent));
  renderMetricCard(
    metricRam,
    "RAM",
    formatPercent(ram.percent),
    `${ram.used_gb ?? 0} / ${ram.total_gb ?? 0} GB`,
    Number(ram.percent),
  );
  if (gpu.available) {
    const vramPercent = gpu.vram_total_gb > 0 ? (Number(gpu.vram_used_gb) / Number(gpu.vram_total_gb)) * 100 : 0;
    renderMetricCard(
      metricGpu,
      "GPU",
      formatPercent(gpu.util_percent),
      `${gpu.name || "NVIDIA"} · ${gpu.vram_used_gb} / ${gpu.vram_total_gb} GB VRAM`,
      vramPercent,
    );
  } else {
    renderMetricCard(metricGpu, "GPU", "offline", "GPU не обнаружена", Number.NaN, { severity: "is-muted" });
  }
  renderMetricCard(
    metricDisk,
    "Disk",
    `${disk.free_gb ?? 0} GB free`,
    disk.runtime_path || "runtime",
    Number(disk.percent),
  );
  const activeWorkers = Number(workers.active || 0);
  const availableWorkers = Number(workers.available || 0);
  const totalWorkers = Math.max(1, activeWorkers + availableWorkers);
  renderMetricCard(
    metricWorkers,
    "Workers",
    `${activeWorkers}/${totalWorkers}`,
    `${workers.mode || "local"} · занято ${activeWorkers}, свободно ${availableWorkers} · OCR jobs ${workers.ocr_jobs ?? "--"}`,
    (activeWorkers / totalWorkers) * 100,
    { severity: activeWorkers ? "is-warning" : "is-ok" },
  );
  dashboardRefresh.textContent = `Обновлено ${formatDateShort(metrics.timestamp)}`;
}

function renderSystemUnavailable() {
  renderMetricCard(metricCpu, "CPU", "--", "метрики недоступны", Number.NaN, { severity: "is-muted" });
  renderMetricCard(metricRam, "RAM", "--", "метрики недоступны", Number.NaN, { severity: "is-muted" });
  renderMetricCard(metricGpu, "GPU", "offline", "GPU не обнаружена", Number.NaN, { severity: "is-muted" });
  renderMetricCard(metricDisk, "Disk", "--", "метрики недоступны", Number.NaN, { severity: "is-muted" });
  renderMetricCard(metricWorkers, "Workers", "--", "метрики недоступны", Number.NaN, { severity: "is-muted" });
  dashboardRefresh.textContent = "Метрики временно недоступны";
}

function renderJobsSummary(summary) {
  const jobs = Array.isArray(summary.last_jobs) ? summary.last_jobs : [];
  state.dashboard.lastJobs = jobs;
  jobsQueued.textContent = Number(summary.queued || 0);
  jobsRunning.textContent = Number(summary.running || 0);
  jobsDone.textContent = Number(summary.done || 0);
  jobsFailed.textContent = Number(summary.failed || 0);
  jobsAverage.textContent = summary.avg_processing_sec ? `среднее ${formatSecondsCompact(summary.avg_processing_sec)}` : "среднее --";

  if (!jobs.length) {
    jobList.innerHTML = `<div class="empty-state">Задач пока нет</div>`;
    state.dashboard.selectedJobId = "";
    state.dashboard.activeJobId = "";
    return;
  }

  const active = jobs.find((job) => ["running", "queued", "uploading"].includes(job.status));
  state.dashboard.activeJobId = active ? active.job_id : "";
  const hasSelected = jobs.some((job) => job.job_id === state.dashboard.selectedJobId);
  if (!hasSelected) {
    state.dashboard.selectedJobId = (active || jobs[0]).job_id;
  }

  jobList.innerHTML = jobs
    .map((job) => {
      const selected = job.job_id === state.dashboard.selectedJobId ? "is-selected" : "";
      const logUrl = job.status === "failed" ? jobLogUrl(job) : "";
      return `
        <article class="job-row ${selected}" data-job-id="${escapeHtml(job.job_id)}">
          <button class="job-row-main" type="button" data-job-id="${escapeHtml(job.job_id)}">
            <span>
              <strong>${escapeHtml(job.filename || job.job_id)}</strong>
              <small>${escapeHtml(job.job_id)} · ${formatDateShort(job.finished_at || job.created_at)}</small>
            </span>
            <span class="job-status ${escapeHtml(job.status)}">${escapeHtml(statusLabel(job.status))}</span>
          </button>
          ${logUrl ? `<a class="job-log-link" href="${escapeHtml(logUrl)}" target="_blank" rel="noreferrer">log</a>` : ""}
        </article>
      `;
    })
    .join("");
}

function renderCurrentJob(metrics) {
  if (!metrics) {
    currentJobStatus.textContent = STATUS_LABELS.idle;
    currentJobBody.innerHTML = `<div class="empty-state">Нет активных задач</div>`;
    return;
  }
  currentJobStatus.textContent = statusLabel(metrics.status);
  const processingSeconds = liveProcessingSeconds(metrics);
  const etaText = etaPhrase(metrics);
  currentJobBody.innerHTML = `
    <div class="job-hero">
      <strong>${escapeHtml(metrics.filename || metrics.job_id)}</strong>
      <span>${escapeHtml(metrics.current_stage || metrics.status)}</span>
    </div>
    ${progressHtml(metrics.progress || 0, severityClass(metrics.progress || 0))}
    ${etaText ? `<div class="eta-note">${escapeHtml(etaText)}</div>` : ""}
    <div class="job-detail-grid">
      <div><span>прогресс</span><strong>${formatPercent(metrics.progress || 0)}</strong></div>
      <div><span>видео</span><strong>${formatSecondsCompact(metrics.video_duration_sec)}</strong></div>
      <div><span>обработка</span><strong>${formatSecondsCompact(processingSeconds)}</strong></div>
    </div>
  `;
}

function renderTiming(metrics) {
  const timings = metrics && Array.isArray(metrics.timings) ? metrics.timings : [];
  if (!timings.length) {
    timingTotal.textContent = "--";
    timingBars.innerHTML = `<div class="empty-state">Timing появится после старта pipeline</div>`;
    return;
  }
  const total = timings.reduce((sum, item) => sum + Number(item.seconds || 0), 0);
  const max = Math.max(...timings.map((item) => Number(item.seconds || 0)), 1);
  timingTotal.textContent = formatSecondsCompact(total);
  timingBars.innerHTML = timings
    .map((item) => {
      const seconds = Number(item.seconds || 0);
      const width = Math.max(2, (seconds / max) * 100);
      const label = TIMING_LABELS[item.step] || item.step;
      return `
        <div class="timing-row">
          <div class="timing-row-label">
            <span>${escapeHtml(label)}</span>
            <strong>${formatSecondsCompact(seconds)}</strong>
          </div>
          <div class="timing-track"><span style="width: ${width}%"></span></div>
        </div>
      `;
    })
    .join("");
}

function renderQuality(metrics) {
  if (!metrics) {
    qualityRows.textContent = "0 rows";
    qualityBody.innerHTML = `<div class="empty-state">Готовых результатов пока нет</div>`;
    return;
  }
  const fillRate = metrics.fill_rate || {};
  const quality = metrics.quality || {};
  const fields = [
    ["product_name", "product_name"],
    ["price_card", "price_card"],
    ["price_default", "price_default"],
    ["barcode", "barcode"],
    ["discount_amount", "discount"],
  ];
  const links = metrics.artifacts || {};
  qualityRows.textContent = `${metrics.rows} rows`;
  const artifactItems = [
    ["CSV", "ready", Boolean(links.csv_url || links.csv_ready)],
    ["Review", "ready", Boolean(links.review_url || links.review_ready)],
    ["Debug", "ready", Boolean(links.debug_url || links.debug_ready)],
    ["Crops", "saved", Boolean(links.crops_saved)],
  ];
  const artifactStatusHtml = artifactItems
    .map(
      ([label, readyText, ready]) => `
        <div class="artifact-status ${ready ? "is-ready" : "is-missing"}">
          <strong>${escapeHtml(`${label} ${ready ? readyText : "missing"}`)}</strong>
        </div>
      `,
    )
    .join("");
  const throughputHtml = throughputText(metrics)
    ? `<div class="business-throughput"><span>Throughput</span><strong>${escapeHtml(throughputText(metrics))}</strong></div>`
    : "";
  if (!metrics.rows) {
    qualityBody.innerHTML = `
      ${throughputHtml}
      <div class="artifact-status-grid">${artifactStatusHtml}</div>
      <div class="empty-state">Готовых результатов пока нет</div>
    `;
    return;
  }
  const fieldHtml = fields
    .map(([key, label]) => {
      const percent = Number(fillRate[key] || 0) * 100;
      return `
        <div class="quality-rate">
          <span>${escapeHtml(label)}</span>
          <strong>${formatPercent(percent)}</strong>
          ${progressHtml(percent, severityClass(100 - percent))}
        </div>
      `;
    })
    .join("");
  const linkHtml = [
    links.csv_url ? `<a href="${escapeHtml(links.csv_url)}" download>CSV</a>` : "",
    links.json_url ? `<a href="${escapeHtml(links.json_url)}" download>JSON</a>` : "",
    links.review_url ? `<a href="${escapeHtml(links.review_url)}" target="_blank" rel="noreferrer">review</a>` : "",
    links.debug_url ? `<a href="${escapeHtml(links.debug_url)}" target="_blank" rel="noreferrer">debug</a>` : "",
  ]
    .filter(Boolean)
    .join("");
  qualityBody.innerHTML = `
    ${throughputHtml}
    <div class="artifact-status-grid">${artifactStatusHtml}</div>
    <div class="quality-rates">${fieldHtml}</div>
    <div class="quality-alerts">
      <div><strong>${quality.suspicious_rows ?? 0}</strong><span>подозрительные строки</span></div>
      <div><strong>${quality.price_outliers ?? 0}</strong><span>price outliers</span></div>
      <div><strong>${quality.empty_barcode ?? 0}</strong><span>empty barcode</span></div>
    </div>
    <div class="artifact-links">${linkHtml || "<span>Артефактов пока нет</span>"}</div>
  `;
}

async function refreshSystemMetrics() {
  try {
    renderSystemMetrics(await fetchJson(API_SYSTEM_METRICS_ENDPOINT));
  } catch (_error) {
    renderSystemUnavailable();
  }
}

async function refreshSelectedJobMetrics() {
  const jobId = state.dashboard.selectedJobId;
  if (!jobId) {
    renderCurrentJob(null);
    renderTiming(null);
    renderQuality(null);
    return;
  }
  try {
    const metrics = await fetchJson(API_JOB_METRICS_ENDPOINT(jobId));
    let activeMetrics = null;
    if (state.dashboard.activeJobId) {
      activeMetrics =
        state.dashboard.activeJobId === jobId
          ? metrics
          : await fetchJson(API_JOB_METRICS_ENDPOINT(state.dashboard.activeJobId));
    }
    renderCurrentJob(activeMetrics);
    renderTiming(metrics);
    renderQuality(metrics);
  } catch (_error) {
    currentJobStatus.textContent = STATUS_LABELS.unavailable;
    currentJobBody.innerHTML = `<div class="empty-state">Метрики задачи временно недоступны</div>`;
    renderTiming(null);
    renderQuality(null);
  }
}

async function refreshJobsDashboard() {
  try {
    const summary = await fetchJson(API_JOBS_SUMMARY_ENDPOINT);
    renderJobsSummary(summary);
    await refreshSelectedJobMetrics();
  } catch (_error) {
    jobList.innerHTML = `<div class="empty-state">Список задач временно недоступен</div>`;
    await refreshSelectedJobMetrics();
  }
}

function pluralizeVideo(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;

  if (mod10 === 1 && mod100 !== 11) {
    return `${count} видео`;
  }

  return `${count} видео`;
}

function pluralizeUpload(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) {
    return `${count} файл`;
  }
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) {
    return `${count} файла`;
  }
  return `${count} файлов`;
}

function isMp4(file) {
  const name = file.name.toLowerCase();
  return file.type === "video/mp4" || name.endsWith(".mp4");
}

function isZip(file) {
  const name = file.name.toLowerCase();
  return file.type === "application/zip" || file.type === "application/x-zip-compressed" || name.endsWith(".zip");
}

function isProcessableFile(file) {
  return isMp4(file) || isZip(file);
}

function isTooLarge(file) {
  return file.size > MAX_VIDEO_SIZE_BYTES;
}

function createVideoId(file) {
  return `${file.name}-${file.size}-${file.lastModified}-${crypto.randomUUID()}`;
}

function setNotice(text, mode = "default") {
  noticeText.textContent = text;
  const dot = document.querySelector(".notice-dot");
  dot.style.background =
    mode === "success" ? "#05bf6a" : mode === "error" ? "#f33387" : mode === "warn" ? "#ffd22e" : "#27b9ff";
}

function setError(text = "") {
  errorMessage.textContent = text;
  errorMessage.hidden = !text;
  uploadCard.classList.toggle("has-error", Boolean(text));
}

function resetCsv() {
  if (state.csvUrl) {
    URL.revokeObjectURL(state.csvUrl);
  }

  state.csvUrl = "";
  state.jsonUrl = "";
  downloadCsv.href = "#";
  downloadCsv.hidden = true;
  downloadCsv.classList.add("disabled");
  downloadCsv.setAttribute("aria-disabled", "true");
  downloadJson.href = "#";
  downloadJson.hidden = true;
  downloadJson.classList.add("disabled");
  downloadJson.setAttribute("aria-disabled", "true");
  downloadReview.href = "#";
  downloadReview.hidden = true;
  downloadReview.classList.add("disabled");
  downloadReview.setAttribute("aria-disabled", "true");
  downloadReviewZip.href = "#";
  downloadReviewZip.hidden = true;
  downloadReviewZip.classList.add("disabled");
  downloadReviewZip.setAttribute("aria-disabled", "true");
  progressBar.style.width = "0%";
  processingPanel.hidden = true;
  processingPanel.classList.remove("is-complete");
  uploadCard.classList.remove("is-processing", "has-result");
  processingTitle.textContent = "Распознаем ценники";
  processingText.textContent = "Подготавливаем кадры робота";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function sanitizeFileBase(name) {
  return name
    .replace(/\.[^.]+$/, "")
    .replace(/[^\p{L}\p{N}_-]+/gu, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64) || "lenta_video";
}

function getActiveVideo() {
  return state.videos.find((video) => video.id === state.activeId);
}

function updateControls() {
  const hasVideos = state.videos.length > 0;
  const activeVideo = getActiveVideo();
  const canPreviewActive = Boolean(activeVideo && isMp4(activeVideo.file));
  uploadCard.classList.toggle("has-files", hasVideos);
  uploadCard.classList.toggle("preview-collapsed", hasVideos && state.previewCollapsed && !state.csvUrl);
  runButton.disabled = !hasVideos || state.isProcessing;
  filesPanel.hidden = !hasVideos;
  previewWrap.hidden = !hasVideos || !canPreviewActive;
  filesCount.textContent = pluralizeUpload(state.videos.length);
  togglePreview.textContent = state.previewCollapsed ? "Показать" : "Скрыть";
  togglePreview.insertAdjacentHTML(
    "beforeend",
    `
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="m6 9 6 6 6-6" />
      </svg>
    `,
  );
  togglePreview.setAttribute("aria-expanded", String(!state.previewCollapsed));

  if (state.previewCollapsed) {
    previewVideo.pause();
  }
}

function renderFileList() {
  fileList.innerHTML = state.videos
    .map((video) => {
      const isActive = video.id === state.activeId;
      const isArchive = isZip(video.file);
      const meta = isArchive
        ? `${formatBytes(video.file.size)} · ZIP архив с MP4`
        : video.meta.loaded
        ? `${formatBytes(video.file.size)} · ${formatDuration(video.meta.duration)} · ${video.meta.width}×${video.meta.height}`
        : `${formatBytes(video.file.size)} · метаданные читаются`;
      const badge = isArchive ? "ZIP" : "MP4";

      return `
        <li class="file-item ${isActive ? "is-active" : ""}" data-id="${escapeHtml(video.id)}">
          <button class="file-item-button" type="button" aria-label="Показать ${escapeHtml(video.file.name)}">
            <span class="file-badge">${badge}</span>
            <span class="file-text">
              <strong>${escapeHtml(video.file.name)}</strong>
              <span>${escapeHtml(meta)}</span>
            </span>
          </button>
          <button class="icon-button remove-video" type="button" aria-label="Удалить ${escapeHtml(video.file.name)}">
            <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </li>
      `;
    })
    .join("");
}

function renderPreview() {
  const activeVideo = getActiveVideo();

  if (!activeVideo || !isMp4(activeVideo.file)) {
    previewVideo.removeAttribute("src");
    previewVideo.load();
    previewCaption.textContent = activeVideo ? `${activeVideo.file.name} будет распакован на сервере` : "";
    return;
  }

  if (previewVideo.src !== activeVideo.objectUrl) {
    previewVideo.src = activeVideo.objectUrl;
    previewVideo.load();
  }

  previewCaption.textContent = activeVideo.file.name;
}

function render() {
  renderFileList();
  renderPreview();
  updateControls();
}

function addFiles(fileListLike) {
  const incomingFiles = Array.from(fileListLike || []);
  if (!incomingFiles.length) {
    return;
  }

  const accepted = [];
  let rejected = 0;
  let tooLarge = 0;

  incomingFiles.forEach((file) => {
    if (!isProcessableFile(file)) {
      rejected += 1;
      return;
    }

    if (isTooLarge(file)) {
      tooLarge += 1;
      return;
    }

    const duplicate = state.videos.some(
      (video) =>
        video.file.name === file.name &&
        video.file.size === file.size &&
        video.file.lastModified === file.lastModified,
    );

    if (duplicate) {
      return;
    }

    accepted.push({
      id: createVideoId(file),
      file,
      objectUrl: isMp4(file) ? URL.createObjectURL(file) : "",
      meta: {
        duration: 0,
        width: 1920,
        height: 1080,
        loaded: false,
      },
    });
  });

  if (!accepted.length) {
    const errorText = tooLarge
      ? `Файл слишком большой. Максимальный размер для прототипа: ${formatBytes(MAX_VIDEO_SIZE_BYTES)}.`
      : rejected
        ? "Нужны файлы MP4 или ZIP с MP4."
        : "Эти видео уже добавлены.";
    setError(errorText);
    setNotice(errorText, rejected || tooLarge ? "error" : "warn");
    return;
  }

  resetCsv();
  setError(
    tooLarge
      ? `Часть файлов не добавлена: размер больше ${formatBytes(MAX_VIDEO_SIZE_BYTES)}.`
      : rejected
        ? "Часть файлов пропущена: нужен формат MP4 или ZIP."
        : "",
  );
  state.videos.push(...accepted);
  state.activeId = state.activeId || accepted[0].id;
  setNotice(
    state.videos.length === 1
      ? "Файл принят и готов к обработке"
      : `${pluralizeUpload(state.videos.length)} готовы к пакетной обработке`,
  );
  render();
}

function removeVideo(id) {
  const index = state.videos.findIndex((video) => video.id === id);
  if (index === -1 || state.isProcessing) {
    return;
  }

  const [removed] = state.videos.splice(index, 1);
  if (removed.objectUrl) {
    URL.revokeObjectURL(removed.objectUrl);
  }

  if (state.activeId === id) {
    state.activeId = state.videos[index]?.id || state.videos[index - 1]?.id || "";
  }

  resetCsv();
  if (!state.videos.length) {
    setError("");
  }
  setNotice(state.videos.length ? `${pluralizeUpload(state.videos.length)} в очереди` : "Видео ожидает загрузки");
  render();
}

function clearAllVideos() {
  if (state.isProcessing) {
    return;
  }

  state.videos.forEach((video) => {
    if (video.objectUrl) {
      URL.revokeObjectURL(video.objectUrl);
    }
  });
  state.videos = [];
  state.activeId = "";
  state.previewCollapsed = false;
  videoInput.value = "";
  previewVideo.removeAttribute("src");
  previewVideo.load();
  resetCsv();
  setError("");
  setNotice("Видео ожидает загрузки");
  render();
}

function getDownloadBaseName() {
  if (state.videos.length === 1) {
    return `${sanitizeFileBase(state.videos[0].file.name)}_price_tags`;
  }

  return `lenta_price_tags_${state.videos.length}_files`;
}

function getDownloadName() {
  return `${getDownloadBaseName()}.csv`;
}

function getDownloadJsonName() {
  return `${getDownloadBaseName()}.json`;
}

function wait(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function setProcessingProgress(progress, text = "") {
  progressBar.style.width = `${Math.max(0, Math.min(progress, 100))}%`;
  if (text) {
    processingText.textContent = text;
  }
}

async function readErrorResponse(response) {
  const contentType = response.headers.get("Content-Type") || "";
  if (contentType.includes("application/json")) {
    const payload = await response.json().catch(() => ({}));
    return payload.error || `HTTP ${response.status}`;
  }
  const text = await response.text().catch(() => "");
  return text || `HTTP ${response.status}`;
}

function userFacingError(error, fallback) {
  const message = error instanceof Error ? error.message : fallback;
  if (/failed to fetch|network\s*error|load failed/i.test(message)) {
    return "Сервер временно недоступен. Обновите страницу и запустите обработку снова.";
  }
  return message || fallback;
}

async function postJson(url, payload = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(await readErrorResponse(response));
  }
  return response.json();
}

async function uploadChunk(jobId, uploadId, file, offset) {
  const end = Math.min(file.size, offset + UPLOAD_CHUNK_BYTES);
  const chunk = file.slice(offset, end);
  const response = await fetch(API_JOB_UPLOAD_CHUNK_ENDPOINT(jobId, uploadId, offset), {
    method: "POST",
    headers: {
      "Content-Type": "application/octet-stream",
    },
    body: chunk,
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(await readErrorResponse(response));
  }
  return end - offset;
}

async function createChunkedInferenceJob() {
  const files = state.videos.map((video) => ({
    name: video.file.name,
    size: video.file.size,
  }));
  const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
  const created = await postJson(API_UPLOAD_CREATE_ENDPOINT, { files });
  const jobId = created.job_id;
  if (!jobId) {
    throw new Error("Сервер не вернул job_id");
  }

  const uploadTargets = Array.isArray(created.files) ? created.files : [];
  let uploadedBytes = 0;
  for (let fileIndex = 0; fileIndex < state.videos.length; fileIndex += 1) {
    const video = state.videos[fileIndex];
    const target = uploadTargets[fileIndex];
    if (!target || target.upload_id === undefined) {
      throw new Error("Сервер не вернул upload_id");
    }
    for (let offset = 0; offset < video.file.size; offset += UPLOAD_CHUNK_BYTES) {
      const written = await uploadChunk(jobId, target.upload_id, video.file, offset);
      uploadedBytes += written;
      const progress = totalBytes > 0 ? Math.min(8, 1 + (uploadedBytes / totalBytes) * 7) : 1;
      setProcessingProgress(
        progress,
        `Загрузка файлов кусками: ${formatBytes(uploadedBytes)} из ${formatBytes(totalBytes)}`,
      );
    }
  }

  setProcessingProgress(8, "Файлы загружены, запускаем обработку");
  return postJson(API_JOB_START_ENDPOINT(jobId), {});
}

function formatEta(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "";
  }

  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  if (minutes <= 0) {
    return `${Math.max(5, rest)} сек`;
  }
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60);
    const tailMinutes = minutes % 60;
    return tailMinutes > 0 ? `${hours} ч ${tailMinutes} мин` : `${hours} ч`;
  }
  return rest > 0 ? `${minutes} мин ${rest.toString().padStart(2, "0")} сек` : `${minutes} мин`;
}

function formatEtaRange(lowSeconds, highSeconds) {
  const low = Number(lowSeconds);
  const high = Number(highSeconds);
  if (!Number.isFinite(low) || !Number.isFinite(high) || high <= 0) {
    return "";
  }
  if (Math.abs(high - low) < 20) {
    return formatEta(high);
  }
  return `${formatEta(low)}-${formatEta(high)}`;
}

function renderJobStatus(status) {
  const progress = Number(status.progress || 0);
  const stage = status.stage || "Пайплайн запущен";
  const detail = status.detail ? `: ${status.detail}` : "";
  const eta = formatEtaRange(status.eta_low_seconds, status.eta_high_seconds);
  const etaText = eta ? ` · осталось примерно ${eta}` : "";
  const videoText =
    status.total_videos > 1
      ? ` · видео ${status.current_video_index}/${status.total_videos}`
      : "";

  processingTitle.textContent = status.current_video
    ? `Обработка: ${status.current_video}`
    : `Обработка: ${pluralizeUpload(state.videos.length)}`;
  setProcessingProgress(progress, `${stage}${detail}${videoText}${etaText}`);
}

async function fetchJobStatus(jobId) {
  const response = await fetch(API_JOB_STATUS_ENDPOINT(jobId), { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await readErrorResponse(response));
  }
  return response.json();
}

async function waitForJob(jobId) {
  while (true) {
    const status = await fetchJobStatus(jobId);
    renderJobStatus(status);
    if (status.status === "done") {
      return status;
    }
    if (status.status === "failed") {
      throw new Error(status.error || "Инференс завершился с ошибкой");
    }
    await wait(STATUS_POLL_INTERVAL_MS);
  }
}

async function runProcessing() {
  if (!state.videos.length || state.isProcessing) {
    return;
  }

  state.isProcessing = true;
  runButton.disabled = true;
  runButton.textContent = "Идет распознавание";
  downloadCsv.hidden = true;
  downloadCsv.classList.add("disabled");
  downloadJson.hidden = true;
  downloadJson.classList.add("disabled");
  processingPanel.hidden = false;
  processingPanel.classList.remove("is-complete");
  uploadCard.classList.add("is-processing");
  uploadCard.classList.remove("has-result");
  processingTitle.textContent = `Обработка: ${pluralizeUpload(state.videos.length)}`;
  progressBar.style.width = "0%";
  setNotice("Видео обрабатывается локальной моделью");

  try {
    setProcessingProgress(1, "Создаем задачу и загружаем файлы кусками");
    const job = await createChunkedInferenceJob();
    const jobId = job.job_id;
    state.dashboard.selectedJobId = jobId || state.dashboard.selectedJobId;
    refreshJobsDashboard();
    if (!jobId) {
      throw new Error("Сервер не вернул job_id");
    }

    setProcessingProgress(8, `Задача ${jobId} создана, ждем первые строки лога`);
    const finalStatus = await waitForJob(jobId);
    const response = await fetch(finalStatus.csv_url || API_JOB_CSV_ENDPOINT(jobId), { cache: "no-store" });
    if (!response.ok) {
      throw new Error(await readErrorResponse(response));
    }

    const blob = await response.blob();
    const rows = Number(finalStatus.rows || response.headers.get("X-Lenta-Rows") || 0);

    if (state.csvUrl) {
      URL.revokeObjectURL(state.csvUrl);
    }

    state.csvUrl = URL.createObjectURL(blob);
    downloadCsv.href = state.csvUrl;
    downloadCsv.download = getDownloadName();
    downloadCsv.hidden = false;
    downloadCsv.classList.remove("disabled");
    downloadCsv.setAttribute("aria-disabled", "false");
    state.jsonUrl = finalStatus.json_url || API_JOB_JSON_ENDPOINT(jobId);
    downloadJson.href = state.jsonUrl;
    downloadJson.download = getDownloadJsonName();
    downloadJson.hidden = false;
    downloadJson.classList.remove("disabled");
    downloadJson.setAttribute("aria-disabled", "false");
    if (finalStatus.review_html_url) {
      downloadReview.href = finalStatus.review_html_url;
      downloadReview.hidden = false;
      downloadReview.classList.remove("disabled");
      downloadReview.setAttribute("aria-disabled", "false");
    }
    if (finalStatus.review_zip_url) {
      downloadReviewZip.href = finalStatus.review_zip_url;
      downloadReviewZip.hidden = false;
      downloadReviewZip.classList.remove("disabled");
      downloadReviewZip.setAttribute("aria-disabled", "false");
    }
    setProcessingProgress(100, `${Number.isFinite(rows) && rows > 0 ? rows : "CSV"} строк готово, job ${jobId}`);
    processingTitle.textContent = "CSV и JSON готовы";
    processingPanel.classList.add("is-complete");
    uploadCard.classList.remove("is-processing");
    uploadCard.classList.add("has-result");
    refreshJobsDashboard();
    setNotice("CSV и JSON готовы к скачиванию", "success");
    runButton.textContent = "Запустить заново";
    runButton.disabled = false;
  } catch (error) {
    const message = userFacingError(error, "Не удалось запустить инференс");
    setError(message);
    setNotice("Инференс завершился с ошибкой", "error");
    processingTitle.textContent = "Ошибка обработки";
    processingText.textContent = message;
    uploadCard.classList.remove("is-processing");
    runButton.textContent = "Запустить распознавание";
    runButton.disabled = false;
  } finally {
    state.isProcessing = false;
    refreshJobsDashboard();
  }
}

function openVideoModal() {
  const activeVideo = getActiveVideo();
  if (!activeVideo || !isMp4(activeVideo.file)) {
    return;
  }

  modalVideo.src = activeVideo.objectUrl;
  videoModal.classList.add("is-open");
  videoModal.setAttribute("aria-hidden", "false");
  modalVideo.play().catch(() => {});
}

function closeVideoModal() {
  videoModal.classList.remove("is-open");
  videoModal.setAttribute("aria-hidden", "true");
  modalVideo.pause();
  modalVideo.removeAttribute("src");
  modalVideo.load();
}

dropzone.addEventListener("click", (event) => {
  if (event.target !== pickFile) {
    videoInput.click();
  }
});

dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    videoInput.click();
  }
});

pickFile.addEventListener("click", (event) => {
  event.stopPropagation();
  videoInput.click();
});

videoInput.addEventListener("change", (event) => {
  addFiles(event.target.files);
  videoInput.value = "";
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  addFiles(event.dataTransfer?.files);
});

fileList.addEventListener("click", (event) => {
  const item = event.target.closest(".file-item");
  if (!item) {
    return;
  }

  const id = item.dataset.id;
  if (event.target.closest(".remove-video")) {
    removeVideo(id);
    return;
  }

  if (state.activeId !== id) {
    state.activeId = id;
    render();
  }
});

jobList.addEventListener("click", (event) => {
  const row = event.target.closest(".job-row-main");
  if (!row) {
    return;
  }
  state.dashboard.selectedJobId = row.dataset.jobId || "";
  jobList.querySelectorAll(".job-row").forEach((item) => {
    item.classList.toggle("is-selected", item.dataset.jobId === state.dashboard.selectedJobId);
  });
  refreshSelectedJobMetrics();
});

clearFiles.addEventListener("click", clearAllVideos);
runButton.addEventListener("click", runProcessing);
openPreview.addEventListener("click", openVideoModal);
togglePreview.addEventListener("click", () => {
  state.previewCollapsed = !state.previewCollapsed;
  render();
});
videoModalBackdrop.addEventListener("click", closeVideoModal);
modalClose.addEventListener("click", closeVideoModal);

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && videoModal.classList.contains("is-open")) {
    closeVideoModal();
  }
});

previewVideo.addEventListener("loadedmetadata", () => {
  const activeVideo = getActiveVideo();
  if (!activeVideo) {
    return;
  }

  activeVideo.meta = {
    duration: previewVideo.duration || 0,
    width: previewVideo.videoWidth || 1920,
    height: previewVideo.videoHeight || 1080,
    loaded: true,
  };
  renderFileList();
});

refreshSystemMetrics();
refreshJobsDashboard();
state.dashboard.systemTimer = window.setInterval(refreshSystemMetrics, DASHBOARD_SYSTEM_INTERVAL_MS);
state.dashboard.jobsTimer = window.setInterval(refreshJobsDashboard, DASHBOARD_JOBS_INTERVAL_MS);

window.addEventListener("beforeunload", () => {
  state.videos.forEach((video) => {
    if (video.objectUrl) {
      URL.revokeObjectURL(video.objectUrl);
    }
  });
  if (state.csvUrl) {
    URL.revokeObjectURL(state.csvUrl);
  }
  window.clearInterval(state.dashboard.systemTimer);
  window.clearInterval(state.dashboard.jobsTimer);
});
