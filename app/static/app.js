const form = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#videoFile");
const fileName = document.querySelector("#fileName");
const dropHint = document.querySelector("#dropHint");
const dropzone = document.querySelector("#dropzone");
const submitButton = document.querySelector("#submitButton");
const resultPanel = document.querySelector("#resultPanel");
const jobTitle = document.querySelector("#jobTitle");
const jobMessage = document.querySelector("#jobMessage");
const jobOptions = document.querySelector("#jobOptions");
const jobStatus = document.querySelector("#jobStatus");
const progressBar = document.querySelector("#progressBar");
const downloadSrt = document.querySelector("#downloadSrt");
const downloadTxt = document.querySelector("#downloadTxt");
const cancelJobButton = document.querySelector("#cancelJob");
const errorBox = document.querySelector("#errorBox");
const serviceState = document.querySelector("#serviceState");
const modeSelect = document.querySelector("#modeSelect");
const modeHint = document.querySelector("#modeHint");
const advancedPanel = document.querySelector("#advancedPanel");
const engineSelect = document.querySelector("#engineSelect");
const engineHint = document.querySelector("#engineHint");
const languageSelect = document.querySelector("#languageSelect");
const engineList = document.querySelector("#engineList");
const installStatus = document.querySelector("#installStatus");
const installLog = document.querySelector("#installLog");
const regionPanel = document.querySelector("#regionPanel");
const regionVideo = document.querySelector("#regionVideo");
const videoRegion = document.querySelector("#videoRegion");
const selectionLayer = document.querySelector("#selectionLayer");
const selectionBox = document.querySelector("#selectionBox");
const regionStatus = document.querySelector("#regionStatus");
const selectRegionButton = document.querySelector("#selectRegionButton");
const useBottomRegion = document.querySelector("#useBottomRegion");
const clearRegion = document.querySelector("#clearRegion");
const previewPlayButton = document.querySelector("#previewPlayButton");
const previewSeek = document.querySelector("#previewSeek");
const previewTime = document.querySelector("#previewTime");
const ocrLogPanel = document.querySelector("#ocrLogPanel");
const ocrLogRows = document.querySelector("#ocrLogRows");
const ocrLogCount = document.querySelector("#ocrLogCount");
const llmCorrection = document.querySelector("#llmCorrection");
const llmProvider = document.querySelector("#llmProvider");
const llmModel = document.querySelector("#llmModel");
const llmBaseUrl = document.querySelector("#llmBaseUrl");
const llmApiKey = document.querySelector("#llmApiKey");
const testLlmConfigButton = document.querySelector("#testLlmConfig");
const saveLlmConfigButton = document.querySelector("#saveLlmConfig");
const resetLlmConfigButton = document.querySelector("#resetLlmConfig");
const llmConfigStatus = document.querySelector("#llmConfigStatus");
const historyPanel = document.querySelector("#historyPanel");
const historyList = document.querySelector("#historyList");
const historyCount = document.querySelector("#historyCount");

const modeHints = {
  balanced: "自动分析视频时长、分辨率和字幕区域，适合大多数视频。",
  fast: "优先速度，抽帧更少，适合先快速看结果。",
  accurate: "优先准确率，抽帧更多，适合短视频或字幕较小的视频。",
  manual: "使用你填写的高级参数。",
};

const statusLabels = {
  queued: "排队中",
  running: "处理中",
  done: "完成",
  failed: "失败",
  canceled: "已停止",
};

const phaseLabels = {
  "analyzing-region": "\u5206\u6790\u533a\u57df",
  "detecting-subtitles": "检查软字幕",
  extracting: "抽帧中",
  "loading-ocr": "加载 OCR",
  "llm-correction": "大模型纠错",
  ocr: "识别中",
  merging: "合并字幕",
  done: "完成",
  failed: "失败",
  canceled: "已停止",
};

const phaseMessages = {
  "analyzing-region": "\u6b63\u5728\u8de8\u591a\u5e27\u5206\u6790\u5b57\u5e55\u533a\u57df\uff0c\u51cf\u5c11 Logo \u3001\u8f66\u724c\u548c\u753b\u9762\u6587\u5b57\u5e72\u6270\u3002",
  "detecting-subtitles": "正在检查视频里是否存在可直接导出的软字幕。",
  extracting: "正在从视频中抽取字幕帧。",
  "loading-ocr": "正在加载 OCR 模型，首次运行会稍慢。",
  "llm-correction": "正在用已配置的大模型修正字幕文本，不会改动时间轴。",
  merging: "正在合并重复字幕并生成文件。",
  done: "处理完成，可以下载字幕文件。",
  failed: "处理失败，请查看错误信息。",
  canceled: "任务已停止，临时视频和抽帧会被清理。",
};

let pollTimer = null;
let dragDepth = 0;
let engineStatuses = new Map();
let installProfiles = new Map();
let installPollTimer = null;
let selectedVideoUrl = "";
let selectedRegion = null;
let regionDraft = null;
let regionDragStart = null;
let regionSelectMode = false;
let previewSeeking = false;
let currentJobId = "";
let currentUploadController = null;

initApp();

modeSelect.addEventListener("change", () => {
  const mode = modeSelect.value;
  modeHint.textContent = modeHints[mode] || modeHints.balanced;
  advancedPanel.open = mode === "manual";
});

engineSelect.addEventListener("change", updateEngineHint);
llmProvider?.addEventListener("change", updateLlmDefaults);
testLlmConfigButton?.addEventListener("click", testLlmConfig);
saveLlmConfigButton?.addEventListener("click", saveLlmConfig);
resetLlmConfigButton?.addEventListener("click", resetLlmConfig);

document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-install-profile]");
  if (!button) return;
  startInstall(button.dataset.installProfile);
});

fileInput.addEventListener("change", () => {
  updateSelectedFile(fileInput.files[0]);
});

dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (eventName === "dragenter") dragDepth += 1;
    dropzone.classList.add("drag-over");
    dropHint.textContent = "松开鼠标导入这个视频";
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (eventName === "dragleave") dragDepth = Math.max(0, dragDepth - 1);
    if (eventName === "drop" || dragDepth === 0) {
      dropzone.classList.remove("drag-over");
      resetDropHint();
    }
  });
}

dropzone.addEventListener("drop", (event) => {
  const file = findVideoFile(event.dataTransfer?.files);
  if (!file) {
    showInlineHint("没有找到可导入的视频文件");
    return;
  }
  setSelectedFile(file);
});

document.addEventListener("paste", (event) => {
  const file = findVideoFileFromClipboard(event.clipboardData);
  if (!file) return;

  event.preventDefault();
  setSelectedFile(file);
  dropzone.classList.add("paste-flash");
  window.setTimeout(() => dropzone.classList.remove("paste-flash"), 520);
});

selectRegionButton.addEventListener("click", () => {
  if (!regionVideo.src) return;
  setRegionSelectMode(!regionSelectMode);
});

selectionLayer.addEventListener("pointerdown", (event) => {
  if (!regionSelectMode || !regionVideo.src) return;
  event.preventDefault();
  selectionLayer.setPointerCapture(event.pointerId);
  regionDragStart = pointInVideo(event);
  regionDraft = { x: regionDragStart.x, y: regionDragStart.y, w: 0, h: 0 };
  showRegion(regionDraft);
});

selectionLayer.addEventListener("pointermove", (event) => {
  if (!regionDragStart) return;
  event.preventDefault();
  regionDraft = normalizeRect(regionDragStart, pointInVideo(event));
  showRegion(regionDraft);
});

selectionLayer.addEventListener("pointerup", finishRegionDrag);
selectionLayer.addEventListener("pointercancel", () => {
  regionDragStart = null;
  regionDraft = null;
  if (selectedRegion) {
    showRegion(selectedRegion);
  } else {
    selectionBox.hidden = true;
  }
});

useBottomRegion.addEventListener("click", () => {
  selectedRegion = { x: 0, y: 0.45, w: 1, h: 0.55 };
  setRegionSelectMode(false);
  showRegion(selectedRegion);
  regionStatus.textContent = "已使用底部 55% 区域";
});

clearRegion.addEventListener("click", clearSelectedRegion);

regionVideo.addEventListener("loadedmetadata", () => {
  if (selectedRegion) showRegion(selectedRegion);
  updatePreviewControls();
});

regionVideo.addEventListener("durationchange", updatePreviewControls);
regionVideo.addEventListener("timeupdate", updatePreviewControls);
regionVideo.addEventListener("play", updatePreviewControls);
regionVideo.addEventListener("pause", updatePreviewControls);
regionVideo.addEventListener("ended", updatePreviewControls);

previewPlayButton.addEventListener("click", async () => {
  if (!regionVideo.src) return;
  try {
    if (regionVideo.paused || regionVideo.ended) {
      await regionVideo.play();
    } else {
      regionVideo.pause();
    }
  } catch {
    regionStatus.textContent = "浏览器阻止了自动播放，请再点一次播放。";
  }
});

previewSeek.addEventListener("input", () => {
  previewSeeking = true;
  seekPreviewToSlider();
  updatePreviewControls();
});

previewSeek.addEventListener("change", () => {
  seekPreviewToSlider();
  previewSeeking = false;
  updatePreviewControls();
});

window.addEventListener("resize", () => {
  if (selectedRegion) showRegion(selectedRegion);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && regionSelectMode) {
    setRegionSelectMode(false);
  }
});

cancelJobButton.addEventListener("click", async () => {
  if (currentUploadController) {
    currentUploadController.abort();
    currentUploadController = null;
    submitButton.disabled = false;
    cancelJobButton.disabled = true;
    setServiceState("已停止", "error");
    return;
  }
  if (!currentJobId) return;
  cancelJobButton.disabled = true;
  try {
    const response = await fetch(`/api/jobs/${currentJobId}/cancel`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "停止任务失败");
    setJobUi(payload);
  } catch (error) {
    showError(error.message);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) {
    showInlineHint("先选择、拖入或粘贴一个视频");
    return;
  }

  setRegionSelectMode(false);
  clearInterval(pollTimer);
  currentJobId = "";
  currentUploadController = new AbortController();
  submitButton.disabled = true;
  cancelJobButton.disabled = false;
  setServiceState("上传中", "busy");
  resultPanel.hidden = false;
  clearLogPanel();
  setJobUi({
    status: "queued",
    filename: fileInput.files[0].name,
    progress: 0,
    message: "正在上传视频",
    phase: "uploading",
  });

  try {
    const body = new FormData(form);
    if (selectedRegion) {
      body.append("crop_x", selectedRegion.x.toFixed(6));
      body.append("crop_y", selectedRegion.y.toFixed(6));
      body.append("crop_w", selectedRegion.w.toFixed(6));
      body.append("crop_h", selectedRegion.h.toFixed(6));
    }

    const response = await fetch("/api/jobs", {
      method: "POST",
      body,
      signal: currentUploadController.signal,
    });
    currentUploadController = null;
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "上传失败");
    }
    currentJobId = payload.id;
    setJobUi(payload);
    pollTimer = setInterval(() => pollJob(payload.id), 1200);
  } catch (error) {
    if (error.name === "AbortError") {
      showError("上传已停止");
    } else {
      showError(error.message);
    }
    currentUploadController = null;
    submitButton.disabled = false;
    cancelJobButton.disabled = true;
    setServiceState("失败", "error");
  }
});

async function initApp() {
  updateLlmDefaults();
  await Promise.all([refreshHealth(), refreshInstallProfiles(), refreshEngines(), pollInstallOnce(), loadLlmConfig()]);
  await loadHistory();
}

async function loadLlmConfig() {
  if (!llmProvider) return;
  try {
    const response = await fetch("/api/config/llm");
    const config = await response.json();
    if (!response.ok) throw new Error(config.detail || "读取大模型配置失败");
    applyLlmConfig(config);
  } catch (error) {
    setLlmConfigStatus(error.message || "读取配置失败", "error");
  }
}

function applyLlmConfig(config) {
  if (!config) return;
  if (llmCorrection) llmCorrection.checked = Boolean(config.enabled);
  if (llmProvider) llmProvider.value = config.provider || "openai";
  if (llmModel) llmModel.value = config.model || "";
  if (llmBaseUrl) llmBaseUrl.value = config.base_url || "";
  if (llmApiKey) {
    llmApiKey.value = "";
    llmApiKey.placeholder = config.has_api_key ? "已保存，留空继续使用" : "Ollama 本地可留空";
  }
  updateLlmDefaults();
  setLlmConfigStatus(config.has_api_key ? "已加载保存配置" : "已加载配置", "ready");
}

function collectLlmConfig() {
  return {
    enabled: Boolean(llmCorrection?.checked),
    provider: llmProvider?.value || "openai",
    model: llmModel?.value.trim() || "",
    base_url: llmBaseUrl?.value.trim() || "",
    api_key: llmApiKey?.value.trim() || "",
  };
}

async function saveLlmConfig() {
  setLlmButtonsEnabled(false);
  setLlmConfigStatus("正在保存", "busy");
  try {
    const response = await fetch("/api/config/llm", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectLlmConfig()),
    });
    const config = await response.json();
    if (!response.ok) throw new Error(config.detail || "保存配置失败");
    applyLlmConfig(config);
    setLlmConfigStatus("配置已保存", "success");
  } catch (error) {
    setLlmConfigStatus(error.message || "保存配置失败", "error");
  } finally {
    setLlmButtonsEnabled(true);
  }
}

async function resetLlmConfig() {
  setLlmButtonsEnabled(false);
  setLlmConfigStatus("正在重置", "busy");
  try {
    const response = await fetch("/api/config/llm/reset", { method: "POST" });
    const config = await response.json();
    if (!response.ok) throw new Error(config.detail || "重置配置失败");
    applyLlmConfig(config);
    setLlmConfigStatus("配置已重置", "success");
  } catch (error) {
    setLlmConfigStatus(error.message || "重置配置失败", "error");
  } finally {
    setLlmButtonsEnabled(true);
  }
}

async function testLlmConfig() {
  setLlmButtonsEnabled(false);
  setLlmConfigStatus("正在测试连接", "busy");
  try {
    const response = await fetch("/api/config/llm/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectLlmConfig()),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "测试连接失败");
    if (!result.ok) throw new Error(result.message || "测试连接失败");
    const suffix = result.elapsed_ms ? ` · ${result.elapsed_ms}ms` : "";
    setLlmConfigStatus(`LLM 测试通过${suffix}`, "success");
  } catch (error) {
    setLlmConfigStatus(error.message || "测试连接失败", "error");
  } finally {
    setLlmButtonsEnabled(true);
  }
}

function setLlmButtonsEnabled(enabled) {
  for (const button of [testLlmConfigButton, saveLlmConfigButton, resetLlmConfigButton]) {
    if (button) button.disabled = !enabled;
  }
}

function setLlmConfigStatus(message, tone = "neutral") {
  if (!llmConfigStatus) return;
  llmConfigStatus.textContent = message;
  llmConfigStatus.dataset.tone = tone;
}

async function refreshHealth() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    if (!response.ok) throw new Error(health.detail || "服务状态检查失败");
    setServiceState(health.ffmpeg_available ? "就绪" : "缺少 FFmpeg", health.ffmpeg_available ? "ready" : "error");
    serviceState.title = formatResourcePlan(health);
  } catch {
    setServiceState("离线", "error");
    serviceState.title = "";
  }
}

async function refreshEngines() {
  try {
    const response = await fetch("/api/engines");
    const engines = await response.json();
    if (!response.ok) throw new Error(engines.detail || "OCR 引擎检测失败");

    engineStatuses = new Map(engines.map((engine) => [engine.id, engine]));
    for (const option of Array.from(engineSelect.options)) {
      if (option.value === "auto") {
        option.disabled = false;
        continue;
      }
      const engine = engineStatuses.get(option.value);
      if (!engine) continue;
      option.textContent = engine.available ? engine.name : `${engine.name}（未安装）`;
      option.disabled = !engine.available;
    }

    const selected = engineSelect.value === "auto" ? { available: true } : engineStatuses.get(engineSelect.value);
    if (!selected?.available) {
      const preferred = engines.find((engine) => engine.available && engine.default) || engines.find((engine) => engine.available);
      if (preferred) engineSelect.value = preferred.id;
    }

    submitButton.disabled = !engines.some((engine) => engine.available);
    updateEngineHint();
    renderEngineManager();
  } catch {
    engineHint.textContent = "暂时无法检测 OCR 引擎";
  }
}

async function refreshInstallProfiles() {
  try {
    const response = await fetch("/api/install/profiles");
    const profiles = await response.json();
    if (!response.ok) throw new Error(profiles.detail || "安装档位检测失败");
    installProfiles = new Map(profiles.map((profile) => [profile.id, profile]));
    renderEngineManager();
  } catch {
    installStatus.textContent = "暂时无法读取安装档位";
  }
}

async function pollInstallOnce() {
  try {
    const response = await fetch("/api/install/current");
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || "安装状态查询失败");
    setInstallUi(job);
    if (job?.status === "running" && !installPollTimer) {
      installPollTimer = setInterval(pollInstallOnce, 1500);
    }
  } catch {
    installStatus.textContent = "暂时无法读取安装状态";
  }
}

async function startInstall(profile) {
  if (!profile) return;
  setInstallButtonsEnabled(false);
  installStatus.textContent = "正在启动安装任务";
  installLog.hidden = false;
  installLog.textContent = "";

  try {
    const response = await fetch(`/api/install/${profile}`, {
      method: "POST",
      headers: {
        "X-VSO-Action": "install",
      },
    });
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || "启动安装失败");
    setInstallUi(job);
    clearInterval(installPollTimer);
    installPollTimer = setInterval(pollInstallOnce, 1500);
  } catch (error) {
    installStatus.textContent = error.message;
    setInstallButtonsEnabled(true);
  }
}

async function pollJob(id) {
  try {
    const response = await fetch(`/api/jobs/${id}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "查询任务失败");
    }
    setJobUi(payload);
    refreshOcrLog(id);
    if (["done", "failed", "canceled"].includes(payload.status)) {
      clearInterval(pollTimer);
      submitButton.disabled = false;
      cancelJobButton.disabled = true;
      setServiceState(payload.status === "done" ? "完成" : payload.status === "canceled" ? "已停止" : "失败", payload.status === "done" ? "success" : "error");
      loadHistory();
    }
  } catch (error) {
    clearInterval(pollTimer);
    showError(error.message);
    submitButton.disabled = false;
    cancelJobButton.disabled = true;
    setServiceState("失败", "error");
  }
}

async function refreshOcrLog(id) {
  try {
    const response = await fetch(`/api/jobs/${id}/ocr-log?limit=80`);
    const payload = await response.json();
    if (!response.ok) return;
    renderOcrLog(payload.rows || [], payload.total || 0);
  } catch {
    // 日志刷新失败不影响主流程。
  }
}

function setSelectedFile(file) {
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  updateSelectedFile(file);
}

function updateSelectedFile(file) {
  if (!file) {
    fileName.textContent = "选择、拖入或粘贴视频";
    resetDropHint();
    clearVideoPreview();
    return;
  }
  fileName.textContent = file.name;
  dropHint.textContent = `${formatBytes(file.size)} · 已准备导入`;
  setServiceState("就绪", "ready");
  showVideoPreview(file);
}

function showVideoPreview(file) {
  clearVideoPreview();
  form.classList.add("has-preview");
  selectedVideoUrl = URL.createObjectURL(file);
  regionVideo.src = selectedVideoUrl;
  regionVideo.load();
  regionPanel.hidden = false;
  selectedRegion = null;
  setRegionSelectMode(false);
  selectionBox.hidden = true;
  regionStatus.textContent = "先定位到有字幕的画面，再框选字幕区域；不框选则使用智能区域。";
  updatePreviewControls();
}

function clearVideoPreview() {
  form.classList.remove("has-preview");
  if (selectedVideoUrl) URL.revokeObjectURL(selectedVideoUrl);
  selectedVideoUrl = "";
  regionVideo.removeAttribute("src");
  regionVideo.load();
  regionPanel.hidden = true;
  selectedRegion = null;
  setRegionSelectMode(false);
  selectionBox.hidden = true;
  updatePreviewControls();
}

function seekPreviewToSlider() {
  const duration = Number(regionVideo.duration) || 0;
  if (!duration) return;
  const ratio = Number(previewSeek.value) / Number(previewSeek.max || 1000);
  regionVideo.currentTime = clamp(ratio, 0, 1) * duration;
}

function updatePreviewControls() {
  const hasVideo = Boolean(regionVideo.src);
  const duration = Number.isFinite(regionVideo.duration) ? regionVideo.duration : 0;
  const currentTime = Number.isFinite(regionVideo.currentTime) ? regionVideo.currentTime : 0;
  previewPlayButton.disabled = !hasVideo;
  previewSeek.disabled = !hasVideo || duration <= 0;
  previewPlayButton.textContent = !regionVideo.paused && !regionVideo.ended ? "暂停" : "播放";
  if (!previewSeeking && duration > 0) {
    previewSeek.value = String(Math.round((currentTime / duration) * Number(previewSeek.max || 1000)));
  } else if (!hasVideo) {
    previewSeek.value = "0";
  }
  previewTime.textContent = `${formatTime(currentTime)} / ${formatTime(duration)}`;
}

function setRegionSelectMode(active) {
  regionSelectMode = Boolean(active && regionVideo.src);
  videoRegion.classList.toggle("region-selecting", regionSelectMode);
  selectRegionButton.classList.toggle("active", regionSelectMode);
  selectRegionButton.setAttribute("aria-pressed", String(regionSelectMode));
  selectRegionButton.textContent = regionSelectMode ? "正在框选" : "框选区域";
  if (regionSelectMode) {
    regionStatus.textContent = "在视频画面上拖动鼠标框选字幕区域。";
  } else if (selectedRegion) {
    regionStatus.textContent = `已框选 ${Math.round(selectedRegion.w * 100)}% × ${Math.round(selectedRegion.h * 100)}% 区域`;
  }
}

function finishRegionDrag(event) {
  if (!regionDragStart) return;
  event.preventDefault();
  const rect = normalizeRect(regionDragStart, pointInVideo(event));
  regionDragStart = null;
  regionDraft = null;

  if (rect.w < 0.03 || rect.h < 0.03) {
    if (selectedRegion) {
      showRegion(selectedRegion);
    } else {
      selectionBox.hidden = true;
    }
    regionStatus.textContent = "框选范围太小，请重新拖动。";
    return;
  }

  selectedRegion = rect;
  showRegion(selectedRegion);
  setRegionSelectMode(false);
}

function getVideoDisplayRect() {
  const container = videoRegion.getBoundingClientRect();
  const videoWidth = regionVideo.videoWidth || 16;
  const videoHeight = regionVideo.videoHeight || 9;
  const containerRatio = container.width / container.height;
  const videoRatio = videoWidth / videoHeight;

  let width = container.width;
  let height = container.height;
  let left = 0;
  let top = 0;

  if (containerRatio > videoRatio) {
    height = container.height;
    width = height * videoRatio;
    left = (container.width - width) / 2;
  } else {
    width = container.width;
    height = width / videoRatio;
    top = (container.height - height) / 2;
  }

  return { container, left, top, width, height };
}

function pointInVideo(event) {
  const fit = getVideoDisplayRect();
  return {
    x: clamp((event.clientX - fit.container.left - fit.left) / fit.width, 0, 1),
    y: clamp((event.clientY - fit.container.top - fit.top) / fit.height, 0, 1),
  };
}

function normalizeRect(start, end) {
  const x1 = Math.min(start.x, end.x);
  const y1 = Math.min(start.y, end.y);
  const x2 = Math.max(start.x, end.x);
  const y2 = Math.max(start.y, end.y);
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}

function showRegion(rect) {
  const fit = getVideoDisplayRect();
  selectionBox.hidden = false;
  selectionBox.style.left = `${fit.left + rect.x * fit.width}px`;
  selectionBox.style.top = `${fit.top + rect.y * fit.height}px`;
  selectionBox.style.width = `${rect.w * fit.width}px`;
  selectionBox.style.height = `${rect.h * fit.height}px`;
}

function clearSelectedRegion() {
  selectedRegion = null;
  regionDraft = null;
  regionDragStart = null;
  setRegionSelectMode(false);
  selectionBox.hidden = true;
  regionStatus.textContent = "未框选区域，将使用智能区域。";
}

function updateEngineHint() {
  if (engineSelect.value === "auto") {
    engineHint.textContent = "按识别模式和本机可用后端自动选择：快速优先加速，均衡/精细优先 OpenVINO 或 ONNXRuntime，并用 PaddleOCR 复核低可信帧。";
    return;
  }
  const engine = engineStatuses.get(engineSelect.value);
  if (!engine) {
    engineHint.textContent = "默认使用 PaddleOCR";
    return;
  }
  if (engine.available) {
    engineHint.textContent = engine.description || `${engine.name} 可用`;
    return;
  }
  engineHint.textContent = `${engine.reason || "当前不可用"} ${engine.install_hint || ""}`.trim();
}

function updateLlmDefaults() {
  if (!llmProvider || !llmModel || !llmBaseUrl) return;
  const defaults = {
    openai: { model: "gpt-4.1-mini", baseUrl: "https://api.openai.com/v1" },
    anthropic: { model: "claude-3-5-haiku-latest", baseUrl: "https://api.anthropic.com" },
    ollama: { model: "qwen2.5:7b", baseUrl: "http://127.0.0.1:11434" },
    "openai-compatible": { model: "gpt-4.1-mini", baseUrl: "" },
  };
  const value = defaults[llmProvider.value] || defaults.openai;
  if (!llmModel.value.trim()) llmModel.placeholder = value.model;
  if (!llmBaseUrl.value.trim()) llmBaseUrl.placeholder = value.baseUrl || "填写兼容接口地址";
}

function renderEngineManager() {
  if (!engineList) return;
  engineList.textContent = "";

  const rows = [
    ["paddle", "recommended"],
    ["openvino", "openvino"],
    ["onnxruntime", "onnxruntime"],
    ["easyocr", "easyocr"],
    ["tesseract", "tesseract"],
  ];

  for (const [engineId, profileId] of rows) {
    const engine = engineStatuses.get(engineId);
    const profile = installProfiles.get(profileId);
    if (!engine) continue;

    const row = document.createElement("div");
    row.className = "engine-row";

    const text = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = engine.name;
    const meta = document.createElement("span");
    meta.textContent = engine.available ? engine.description : engine.reason || profile?.description || "未安装";
    text.append(title, meta);

    const state = document.createElement("span");
    state.className = `engine-state ${engine.available ? "available" : "missing"}`;
    state.textContent = engine.available ? "可用" : "未安装";

    const button = document.createElement("button");
    button.className = "secondary compact";
    button.type = "button";
    button.dataset.installProfile = profileId;
    button.disabled = engine.available;
    button.textContent = engine.available ? "已安装" : "安装";

    row.append(text, state, button);
    engineList.append(row);
  }
}

function setInstallUi(job) {
  if (!job) {
    setInstallButtonsEnabled(true);
    return;
  }

  installStatus.textContent = `${job.profile_name} · ${job.message}`;
  if (job.log?.length) {
    installLog.hidden = false;
    installLog.textContent = job.log.join("\n");
    installLog.scrollTop = installLog.scrollHeight;
  }

  const running = job.status === "running";
  setInstallButtonsEnabled(!running);
  if (!running) {
    clearInterval(installPollTimer);
    installPollTimer = null;
    refreshEngines();
  }
}

function setInstallButtonsEnabled(enabled) {
  for (const button of document.querySelectorAll("[data-install-profile]")) {
    const engine = profileEngine(button.dataset.installProfile);
    const alreadyAvailable = engine ? engineStatuses.get(engine)?.available : false;
    button.disabled = !enabled || Boolean(alreadyAvailable);
  }
}

function profileEngine(profile) {
  return {
    recommended: "paddle",
    openvino: "openvino",
    onnxruntime: "onnxruntime",
    easyocr: "easyocr",
    tesseract: "tesseract",
  }[profile] || "";
}

function findVideoFile(files) {
  if (!files) return null;
  return Array.from(files).find(isVideoFile) || null;
}

function findVideoFileFromClipboard(clipboardData) {
  if (!clipboardData) return null;

  const directFile = findVideoFile(clipboardData.files);
  if (directFile) return directFile;

  for (const item of Array.from(clipboardData.items || [])) {
    if (item.kind !== "file") continue;
    const file = item.getAsFile();
    if (isVideoFile(file)) return file;
  }
  return null;
}

function isVideoFile(file) {
  if (!file) return false;
  if (file.type?.startsWith("video/")) return true;
  return /\.(mp4|mkv|mov|avi|webm|m4v|ts)$/i.test(file.name || "");
}

function resetDropHint() {
  dropHint.textContent = "支持 MP4、MKV、MOV、WEBM，也可以直接 Ctrl+V 粘贴文件";
}

function showInlineHint(message) {
  dropHint.textContent = message;
  dropzone.classList.add("input-warn");
  window.setTimeout(() => {
    dropzone.classList.remove("input-warn");
    if (!fileInput.files.length) resetDropHint();
  }, 1800);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function setJobUi(job) {
  errorBox.hidden = true;
  errorBox.textContent = "";
  jobTitle.textContent = job.filename || "字幕识别任务";
  jobMessage.textContent = formatJobMessage(job);
  jobStatus.textContent = formatStatus(job);
  setServiceState(formatServiceState(job), job.status === "failed" || job.status === "canceled" ? "error" : job.status === "done" ? "success" : "busy");
  progressBar.style.width = `${Math.round((job.progress || 0) * 100)}%`;
  jobOptions.textContent = formatOptions(job.options);
  cancelJobButton.disabled = !["queued", "running"].includes(job.status) || Boolean(job.cancel_requested);

  updateDownload(downloadSrt, job.srt_url);
  updateDownload(downloadTxt, job.txt_url);

  if (job.status === "failed") {
    showError(job.error || "处理失败");
  }
}

function updateDownload(link, url) {
  if (url) {
    link.href = url;
    link.classList.remove("disabled");
  } else {
    link.href = "#";
    link.classList.add("disabled");
  }
}

function formatJobMessage(job) {
  if (job.cancel_requested && !["done", "failed", "canceled"].includes(job.status)) {
    return "正在停止任务，会在当前 OCR 批次结束后停止。";
  }
  return job.message || phaseMessages[job.phase] || job.phase || "";
}

function formatStatus(job) {
  if (job.cancel_requested && !["done", "failed", "canceled"].includes(job.status)) return "停止中";
  return phaseLabels[job.phase] || statusLabels[job.status] || job.status || "排队中";
}

function formatServiceState(job) {
  if (job.phase === "uploading") return "上传中";
  return formatStatus(job);
}

function formatOptions(options) {
  if (!options) return "";
  const modeName = {
    fast: "快速预览",
    balanced: "智能均衡",
    accurate: "精细识别",
    manual: "手动高级",
  }[options.mode] || "智能配置";
  const engineName = engineStatuses.get(options.engine)?.name || options.engine;
  const languageName = {
    ch: "中英混合",
    en: "英文优先",
  }[options.language] || options.language;
  const manualRegion = options.crop_x != null ? " · 已框选区域" : "";
  const autoRegion = options.auto_crop_x != null ? " · 智能锁定区域" : "";
  const region = manualRegion || autoRegion;
  const reuse = options.skip_unchanged_frames ? " · 智能跳帧" : "";
  const strip = options.auto_subtitle_strip ? " · 字幕细条" : "";
  const review = options.review_engine ? ` · ${engineStatuses.get(options.review_engine)?.name || options.review_engine} 复核` : "";
  const llm = options.llm_correction ? ` · 大模型纠错 ${options.llm_provider || ""}` : "";
  return `${modeName} · ${engineName}${review} · ${languageName} · ${options.fps} FPS · 底部 ${Math.round(options.crop_bottom * 100)}% · 批量 ${options.ocr_batch_size}${region}${reuse}${strip}${llm}`;
}

function renderOcrLog(rows, total) {
  ocrLogPanel.hidden = rows.length === 0;
  const reused = rows.filter((row) => row.reused).length;
  const empty = rows.filter((row) => row.empty).length;
  const reviewed = rows.filter((row) => row.reviewed).length;
  const suffix = rows.length ? ` · 近 ${rows.length} 条里复核 ${reviewed}，复用 ${reused}，空帧 ${empty}` : "";
  ocrLogCount.textContent = `${total} 条${suffix}`;
  ocrLogRows.textContent = "";
  for (const row of rows) {
    const item = document.createElement("div");
    item.className = `ocr-log-row ${row.empty ? "empty" : ""}`;
    const time = document.createElement("span");
    time.textContent = formatTime(row.timestamp);
    const text = document.createElement("p");
    text.textContent = row.empty ? "空帧" : row.text;
    if (row.reused) {
      const badge = document.createElement("em");
      badge.textContent = "复用";
      text.prepend(badge, " ");
    }
    if (row.reviewed) {
      const badge = document.createElement("em");
      badge.textContent = "复核";
      text.prepend(badge, " ");
    }
    item.append(time, text);
    ocrLogRows.append(item);
  }
  ocrLogRows.scrollTop = ocrLogRows.scrollHeight;
}

function clearLogPanel() {
  ocrLogPanel.hidden = true;
  ocrLogRows.textContent = "";
  ocrLogCount.textContent = "0 条";
}

function formatTime(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function showError(message) {
  resultPanel.hidden = false;
  errorBox.hidden = false;
  errorBox.textContent = message;
}

function setServiceState(label, tone = "neutral") {
  serviceState.textContent = label;
  serviceState.dataset.tone = tone;
}

function formatResourcePlan(health) {
  const parts = [];
  if (health.version) parts.push(`版本 ${health.version}`);
  if (health.worker_count) {
    const source = health.worker_source === "manual" ? "手动" : "自动";
    parts.push(`${source} worker ${health.worker_count}/${health.max_workers || health.worker_count}`);
  }
  if (health.cpu_count && health.cpu_threads) {
    parts.push(`CPU ${health.cpu_count} 线程，单 worker ${health.cpu_threads} 线程`);
  }
  if (health.available_memory_gb != null && health.total_memory_gb != null) {
    parts.push(`内存可用 ${health.available_memory_gb}GB / ${health.total_memory_gb}GB`);
  }
  return parts.join(" · ");
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

async function loadHistory() {
  try {
    const response = await fetch("/api/jobs?limit=30");
    if (!response.ok) return;
    const jobs = await response.json();
    renderHistory(jobs);
  } catch {
    // 历史记录加载失败不影响主流程。
  }
}

function renderHistory(jobs) {
  if (!historyPanel || !historyList || !historyCount) return;
  // 过滤掉当前正在展示的任务（进行中的），只展示已结束的
  const finished = jobs.filter((j) => ["done", "failed", "canceled"].includes(j.status));
  if (finished.length === 0) {
    historyPanel.hidden = true;
    return;
  }
  historyPanel.hidden = false;
  historyCount.textContent = `共 ${finished.length} 条`;
  historyList.textContent = "";

  for (const job of finished) {
    const item = document.createElement("div");
    item.className = "history-item";

    // 左侧：文件名 + 时间
    const meta = document.createElement("div");
    meta.className = "history-item-meta";

    const name = document.createElement("div");
    name.className = "history-item-name";
    name.textContent = job.filename || "未知文件";
    name.title = job.filename || "";

    const timeEl = document.createElement("div");
    timeEl.className = "history-item-time";
    const timeStr = job.created_at ? formatRelativeTime(job.created_at * 1000) : "";
    const msgStr = job.message ? ` · ${job.message}` : "";
    timeEl.textContent = timeStr + msgStr;

    meta.append(name, timeEl);

    // 右侧：状态 + 操作链接
    const actions = document.createElement("div");
    actions.className = "history-item-actions";

    const pill = document.createElement("span");
    pill.className = "status-pill";
    const toneMap = { done: "success", failed: "error", canceled: "neutral" };
    pill.dataset.tone = toneMap[job.status] || "neutral";
    const statusLabels = { queued: "排队中", running: "进行中", done: "完成", failed: "失败", canceled: "已停止" };
    pill.textContent = statusLabels[job.status] || job.status;
    actions.append(pill);

    if (job.srt_url) {
      const srtLink = document.createElement("a");
      srtLink.href = job.srt_url;
      srtLink.textContent = "SRT";
      srtLink.download = "";
      actions.append(srtLink);
    }
    if (job.txt_url) {
      const txtLink = document.createElement("a");
      txtLink.href = job.txt_url;
      txtLink.textContent = "TXT";
      txtLink.download = "";
      actions.append(txtLink);
    }

    item.append(meta, actions);
    historyList.append(item);
  }
}

function formatRelativeTime(timestampMs) {
  const diffSec = Math.round((Date.now() - timestampMs) / 1000);
  if (diffSec < 5) return "刚刚";
  if (diffSec < 60) return `${diffSec} 秒前`;
  const diffMin = Math.round(diffSec / 60);
  if (diffMin < 60) return `${diffMin} 分钟前`;
  const diffHour = Math.round(diffMin / 60);
  if (diffHour < 24) return `${diffHour} 小时前`;
  const diffDay = Math.round(diffHour / 24);
  if (diffDay < 30) return `${diffDay} 天前`;
  return new Date(timestampMs).toLocaleDateString("zh-CN");
}
