const MAX_VIDEO_SIZE_BYTES = 512 * 1024 * 1024;
const UPLOAD_CHUNK_BYTES = 2 * 1024 * 1024;
const API_UPLOAD_CREATE_ENDPOINT = "/api/upload/create";
const API_JOB_UPLOAD_CHUNK_ENDPOINT = (jobId, uploadId, offset) =>
  `/api/jobs/${encodeURIComponent(jobId)}/upload-chunk?upload_id=${encodeURIComponent(uploadId)}&offset=${offset}`;
const API_JOB_START_ENDPOINT = (jobId) => `/api/jobs/${encodeURIComponent(jobId)}/start`;
const API_JOB_STATUS_ENDPOINT = (jobId) => `/api/jobs/${encodeURIComponent(jobId)}/status`;
const API_JOB_CSV_ENDPOINT = (jobId) => `/api/jobs/${encodeURIComponent(jobId)}/csv`;
const STATUS_POLL_INTERVAL_MS = 1400;

const state = {
  videos: [],
  activeId: "",
  csvUrl: "",
  isProcessing: false,
  previewCollapsed: false,
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
const downloadReview = document.querySelector("#download-review");
const downloadReviewZip = document.querySelector("#download-review-zip");
const videoModal = document.querySelector("#video-modal");
const videoModalBackdrop = document.querySelector("#video-modal-backdrop");
const modalClose = document.querySelector("#modal-close");
const modalVideo = document.querySelector("#modal-video");

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

function pluralizeVideo(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;

  if (mod10 === 1 && mod100 !== 11) {
    return `${count} видео`;
  }

  return `${count} видео`;
}

function isMp4(file) {
  const name = file.name.toLowerCase();
  return file.type === "video/mp4" || name.endsWith(".mp4");
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
  downloadCsv.href = "#";
  downloadCsv.hidden = true;
  downloadCsv.classList.add("disabled");
  downloadCsv.setAttribute("aria-disabled", "true");
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
  uploadCard.classList.toggle("has-files", hasVideos);
  uploadCard.classList.toggle("preview-collapsed", hasVideos && state.previewCollapsed && !state.csvUrl);
  runButton.disabled = !hasVideos || state.isProcessing;
  filesPanel.hidden = !hasVideos;
  previewWrap.hidden = !hasVideos;
  filesCount.textContent = pluralizeVideo(state.videos.length);
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
      const meta = video.meta.loaded
        ? `${formatBytes(video.file.size)} · ${formatDuration(video.meta.duration)} · ${video.meta.width}×${video.meta.height}`
        : `${formatBytes(video.file.size)} · метаданные читаются`;

      return `
        <li class="file-item ${isActive ? "is-active" : ""}" data-id="${escapeHtml(video.id)}">
          <button class="file-item-button" type="button" aria-label="Показать ${escapeHtml(video.file.name)}">
            <span class="file-badge">MP4</span>
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

  if (!activeVideo) {
    previewVideo.removeAttribute("src");
    previewVideo.load();
    previewCaption.textContent = "";
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
    if (!isMp4(file)) {
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
      objectUrl: URL.createObjectURL(file),
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
        ? "Нужны файлы в формате MP4."
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
        ? "Часть файлов пропущена: нужен формат MP4."
        : "",
  );
  state.videos.push(...accepted);
  state.activeId = state.activeId || accepted[0].id;
  setNotice(
    state.videos.length === 1
      ? "Видео принято и готово к обработке"
      : `${pluralizeVideo(state.videos.length)} готовы к пакетной обработке`,
  );
  render();
}

function removeVideo(id) {
  const index = state.videos.findIndex((video) => video.id === id);
  if (index === -1 || state.isProcessing) {
    return;
  }

  const [removed] = state.videos.splice(index, 1);
  URL.revokeObjectURL(removed.objectUrl);

  if (state.activeId === id) {
    state.activeId = state.videos[index]?.id || state.videos[index - 1]?.id || "";
  }

  resetCsv();
  if (!state.videos.length) {
    setError("");
  }
  setNotice(state.videos.length ? `${pluralizeVideo(state.videos.length)} в очереди` : "Видео ожидает загрузки");
  render();
}

function clearAllVideos() {
  if (state.isProcessing) {
    return;
  }

  state.videos.forEach((video) => URL.revokeObjectURL(video.objectUrl));
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

function getDownloadName() {
  if (state.videos.length === 1) {
    return `${sanitizeFileBase(state.videos[0].file.name)}_price_tags.csv`;
  }

  return `lenta_price_tags_${state.videos.length}_videos.csv`;
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
        `Загрузка видео кусками: ${formatBytes(uploadedBytes)} из ${formatBytes(totalBytes)}`,
      );
    }
  }

  setProcessingProgress(8, "Видео загружено, запускаем обработку");
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
    : `Обработка: ${pluralizeVideo(state.videos.length)}`;
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
  processingPanel.hidden = false;
  processingPanel.classList.remove("is-complete");
  uploadCard.classList.add("is-processing");
  uploadCard.classList.remove("has-result");
  processingTitle.textContent = `Обработка: ${pluralizeVideo(state.videos.length)}`;
  progressBar.style.width = "0%";
  setNotice("Видео обрабатывается локальной моделью");

  try {
    setProcessingProgress(1, "Создаем задачу и загружаем видео кусками");
    const job = await createChunkedInferenceJob();
    const jobId = job.job_id;
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
    processingTitle.textContent = "CSV готов";
    processingPanel.classList.add("is-complete");
    uploadCard.classList.remove("is-processing");
    uploadCard.classList.add("has-result");
    setNotice("CSV готов к скачиванию", "success");
    runButton.textContent = "Запустить заново";
    runButton.disabled = false;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Не удалось запустить инференс";
    setError(message);
    setNotice("Инференс завершился с ошибкой", "error");
    processingTitle.textContent = "Ошибка обработки";
    processingText.textContent = message;
    uploadCard.classList.remove("is-processing");
    runButton.textContent = "Запустить распознавание";
    runButton.disabled = false;
  } finally {
    state.isProcessing = false;
  }
}

function openVideoModal() {
  const activeVideo = getActiveVideo();
  if (!activeVideo) {
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

window.addEventListener("beforeunload", () => {
  state.videos.forEach((video) => URL.revokeObjectURL(video.objectUrl));
  if (state.csvUrl) {
    URL.revokeObjectURL(state.csvUrl);
  }
});
