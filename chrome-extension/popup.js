const API_BASE = "http://127.0.0.1:5000";

const serverState = document.getElementById("serverState");
const statusEl = document.getElementById("status");
const labelEl = document.getElementById("label");
const confidenceEl = document.getElementById("confidence");
const analyzePageBtn = document.getElementById("analyzePage");
const analyzeFileBtn = document.getElementById("analyzeFile");
const fileInput = document.getElementById("fileInput");
const MAX_PAGE_VIDEO_BYTES = 180 * 1024 * 1024;

function setBusy(isBusy) {
  analyzePageBtn.disabled = isBusy;
  analyzeFileBtn.disabled = isBusy;
}

function setStatus(message) {
  statusEl.textContent = message;
}

function clearResult() {
  labelEl.textContent = "";
  labelEl.className = "";
  confidenceEl.textContent = "";
}

function showResult(result) {
  const label = result.label || "UNKNOWN";
  labelEl.textContent = label;
  labelEl.className = result.status === "unsupported" ? "unsupported" : label.toLowerCase();

  if (result.status === "unsupported") {
    confidenceEl.textContent = result.message || "This video is not suitable for deepfake processing.";
    setStatus("Video is not suitable for deepfake analysis.");
    return;
  }

  confidenceEl.textContent = `Confidence: ${(Number(result.probability) * 100).toFixed(1)}%`;
  setStatus("Analysis complete.");
}

async function checkServer() {
  try {
    const response = await fetch(`${API_BASE}/api/health`);
    const data = await response.json();
    serverState.textContent = data.models_loaded ? "Ready" : "Model missing";
  } catch (_error) {
    serverState.textContent = "Offline";
  }
}

async function postPrediction(body, options = {}) {
  setBusy(true);
  clearResult();
  setStatus("Analyzing video. This can take a minute on CPU.");

  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      body,
      ...options
    });
    const data = await response.json();
    if (!response.ok || !["ok", "unsupported"].includes(data.status)) {
      throw new Error(data.error || "Prediction failed.");
    }
    showResult(data);
  } catch (error) {
    clearResult();
    setStatus(error.message);
  } finally {
    setBusy(false);
  }
}

function filenameFromUrl(url, fallback = "page-video.mp4") {
  try {
    const parsed = new URL(url);
    const lastPart = parsed.pathname.split("/").filter(Boolean).pop();
    return lastPart || fallback;
  } catch (_error) {
    return fallback;
  }
}

function extensionFromMime(mimeType) {
  if (mimeType.includes("webm")) return "webm";
  if (mimeType.includes("quicktime")) return "mov";
  if (mimeType.includes("x-msvideo")) return "avi";
  if (mimeType.includes("matroska")) return "mkv";
  return "mp4";
}

function collectPageVideoInfo() {
  const absoluteUrl = (value) => {
    if (!value) return "";
    try {
      return new URL(value, document.baseURI).href;
    } catch (_error) {
      return "";
    }
  };

  const urls = [];
  const addUrl = (value) => {
    const url = absoluteUrl(value);
    if (url && !urls.includes(url)) urls.push(url);
  };

  for (const video of document.querySelectorAll("video")) {
    addUrl(video.currentSrc);
    addUrl(video.src);
    for (const source of video.querySelectorAll("source")) {
      addUrl(source.src);
    }
  }

  for (const selector of [
    "meta[property='og:video']",
    "meta[property='og:video:url']",
    "meta[property='og:video:secure_url']",
    "meta[name='twitter:player:stream']"
  ]) {
    addUrl(document.querySelector(selector)?.content);
  }

  for (const link of document.querySelectorAll("a[href]")) {
    const href = link.getAttribute("href");
    if (/\.(mp4|mov|avi|mkv|webm)(\?|#|$)/i.test(href || "")) {
      addUrl(href);
    }
  }

  return {
    pageTitle: document.title || "page-video",
    videoElementCount: document.querySelectorAll("video").length,
    urls
  };
}

async function readBlobVideoFromPage(blobUrl, maxBytes) {
  const response = await fetch(blobUrl);
  if (!response.ok) {
    throw new Error(`Could not read blob video: ${response.status}`);
  }

  const blob = await response.blob();
  if (blob.size > maxBytes) {
    throw new Error("Page video is too large to transfer from the tab.");
  }

  return {
    bytes: Array.from(new Uint8Array(await blob.arrayBuffer())),
    mimeType: blob.type || "video/mp4",
    size: blob.size
  };
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) {
    throw new Error("No active tab found.");
  }
  return tab;
}

async function getPageVideoInfo(tabId) {
  const [injection] = await chrome.scripting.executeScript({
    target: { tabId },
    func: collectPageVideoInfo
  });
  return injection?.result || { urls: [], videoElementCount: 0 };
}

async function fetchPageVideoAsBlob(url) {
  const response = await fetch(url, { credentials: "include" });
  if (!response.ok) {
    throw new Error(`Chrome could not fetch the video URL: ${response.status}`);
  }

  const blob = await response.blob();
  if (blob.size > MAX_PAGE_VIDEO_BYTES) {
    throw new Error("Page video is too large for extension upload.");
  }
  return blob;
}

async function uploadBlobForPrediction(blob, filename) {
  const formData = new FormData();
  formData.append("file", blob, filename);
  await postPrediction(formData);
}

async function analyzeUploadedFile() {
  const file = fileInput.files?.[0];
  if (!file) {
    setStatus("Choose a video file first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  await postPrediction(formData);
}

async function analyzePageVideo() {
  setBusy(true);
  clearResult();
  setStatus("Looking for video sources on this page.");

  try {
    const tab = await getActiveTab();
    const pageInfo = await getPageVideoInfo(tab.id);
    const urls = pageInfo.urls || [];

    if (!urls.length) {
      throw new Error(pageInfo.videoElementCount > 0
        ? "Video found, but the page does not expose a readable source URL."
        : "No video source found on this page.");
    }

    const blobUrl = urls.find((url) => url.startsWith("blob:"));
    const directUrl = urls.find((url) => /^https?:\/\//i.test(url));

    if (directUrl) {
      setStatus("Downloading page video through Chrome.");
      const blob = await fetchPageVideoAsBlob(directUrl);
      await uploadBlobForPrediction(blob, filenameFromUrl(directUrl));
      return;
    }

    if (blobUrl) {
      setStatus("Reading blob video from the current tab.");
      const [injection] = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: readBlobVideoFromPage,
        args: [blobUrl, MAX_PAGE_VIDEO_BYTES]
      });
      const blobData = injection?.result;
      if (!blobData?.bytes?.length) {
        throw new Error("Could not read the blob video from this page.");
      }

      const bytes = new Uint8Array(blobData.bytes);
      const blob = new Blob([bytes], { type: blobData.mimeType });
      const ext = extensionFromMime(blobData.mimeType);
      await uploadBlobForPrediction(blob, `${pageInfo.pageTitle}.${ext}`);
      return;
    }

    throw new Error("No supported video URL found on this page.");
  } catch (error) {
    clearResult();
    setStatus(error.message);
    setBusy(false);
  }
}

analyzeFileBtn.addEventListener("click", analyzeUploadedFile);
analyzePageBtn.addEventListener("click", analyzePageVideo);
checkServer();
