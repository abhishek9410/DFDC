const API_BASE = "http://127.0.0.1:5000";
const MENU_ID = "analyze-deepfake-video";
const MAX_CONTEXT_VIDEO_BYTES = 180 * 1024 * 1024;

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: MENU_ID,
    title: "Analyze this video",
    contexts: ["video"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId !== MENU_ID || !tab?.id) return;
  analyzeContextVideo(info, tab);
});

async function analyzeContextVideo(info, tab) {
  try {
    await setRunningState(tab.id);
    showNotification("DeepFake analysis started", "Sending this video to the local model.");

    const sourceInfo = await getContextVideoSource(info, tab.id);
    const blob = await getVideoBlob(sourceInfo.url, tab.id);
    const filename = filenameFromSource(sourceInfo.url, sourceInfo.pageTitle, blob.type);
    const result = await uploadForPrediction(blob, filename);

    if (result.status === "unsupported") {
      await chrome.action.setBadgeText({ tabId: tab.id, text: "INFO" });
      await chrome.action.setBadgeBackgroundColor({ tabId: tab.id, color: "#854d0e" });
      showNotification("Video not suitable for analysis", result.message || "No clear human face was detected.");
      return;
    }

    const confidence = `${(Number(result.probability) * 100).toFixed(1)}%`;
    await chrome.action.setBadgeText({ tabId: tab.id, text: result.label === "FAKE" ? "FAKE" : "REAL" });
    await chrome.action.setBadgeBackgroundColor({
      tabId: tab.id,
      color: result.label === "FAKE" ? "#b42318" : "#087443"
    });
    showNotification(`DeepFake result: ${result.label}`, `Confidence: ${confidence}`);
  } catch (error) {
    await chrome.action.setBadgeText({ tabId: tab.id, text: "ERR" });
    await chrome.action.setBadgeBackgroundColor({ tabId: tab.id, color: "#71717a" });
    showNotification("DeepFake analysis failed", error.message || "Could not analyze this video.");
  }
}

async function setRunningState(tabId) {
  await chrome.action.setBadgeText({ tabId, text: "..." });
  await chrome.action.setBadgeBackgroundColor({ tabId, color: "#1f6feb" });
}

async function getContextVideoSource(info, tabId) {
  const srcUrl = info.srcUrl || "";
  if (isSupportedSource(srcUrl)) {
    return { url: srcUrl, pageTitle: "context-video" };
  }

  const pageVideo = await askContentScriptForContextVideo(tabId);
  const sources = pageVideo.sources || [];
  const source = sources.find(isSupportedSource) || sources[0];
  if (!source) {
    throw new Error("This video does not expose a readable source URL.");
  }

  return {
    url: source,
    pageTitle: pageVideo.pageTitle || "context-video"
  };
}

function isSupportedSource(url) {
  return /^https?:\/\//i.test(url || "") || /^blob:/i.test(url || "");
}

async function askContentScriptForContextVideo(tabId) {
  try {
    return await chrome.tabs.sendMessage(tabId, { type: "GET_CONTEXT_VIDEO" });
  } catch (_error) {
    return injectContextVideoReader(tabId);
  }
}

async function injectContextVideoReader(tabId) {
  const [injection] = await chrome.scripting.executeScript({
    target: { tabId },
    func: () => {
      const video = document.querySelector("video");
      if (!video) return { sources: [], pageTitle: document.title || "context-video" };
      const sources = [video.currentSrc, video.src]
        .concat(Array.from(video.querySelectorAll("source")).map((source) => source.src))
        .filter(Boolean);
      return { sources, pageTitle: document.title || "context-video" };
    }
  });
  return injection?.result || { sources: [], pageTitle: "context-video" };
}

async function getVideoBlob(url, tabId) {
  if (/^blob:/i.test(url)) {
    const blobData = await readBlobInPage(tabId, url);
    const bytes = new Uint8Array(blobData.bytes);
    return new Blob([bytes], { type: blobData.mimeType || "video/mp4" });
  }

  const response = await fetch(url, { credentials: "include" });
  if (!response.ok) {
    throw new Error(`Chrome could not download this video: ${response.status}`);
  }

  const blob = await response.blob();
  if (blob.size > MAX_CONTEXT_VIDEO_BYTES) {
    throw new Error("This video is too large for extension upload.");
  }
  return blob;
}

async function readBlobInPage(tabId, blobUrl) {
  const [injection] = await chrome.scripting.executeScript({
    target: { tabId },
    func: async (url, maxBytes) => {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Could not read blob video: ${response.status}`);
      }
      const blob = await response.blob();
      if (blob.size > maxBytes) {
        throw new Error("This video is too large to transfer from the page.");
      }
      return {
        bytes: Array.from(new Uint8Array(await blob.arrayBuffer())),
        mimeType: blob.type || "video/mp4"
      };
    },
    args: [blobUrl, MAX_CONTEXT_VIDEO_BYTES]
  });

  const result = injection?.result;
  if (!result?.bytes?.length) {
    throw new Error("Could not read this page video.");
  }
  return result;
}

async function uploadForPrediction(blob, filename) {
  const formData = new FormData();
  formData.append("file", blob, filename);

  const response = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    body: formData
  });
  const data = await response.json();
  if (!response.ok || !["ok", "unsupported"].includes(data.status)) {
    throw new Error(data.error || "Prediction failed.");
  }
  return data;
}

function filenameFromSource(url, pageTitle, mimeType) {
  if (/^https?:\/\//i.test(url)) {
    try {
      const parsed = new URL(url);
      const name = parsed.pathname.split("/").filter(Boolean).pop();
      if (name) return name;
    } catch (_error) {
      // Fall back to page title below.
    }
  }

  const cleanTitle = (pageTitle || "context-video")
    .replace(/[^a-z0-9._-]+/gi, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80) || "context-video";
  return `${cleanTitle}.${extensionFromMime(mimeType || "")}`;
}

function extensionFromMime(mimeType) {
  if (mimeType.includes("webm")) return "webm";
  if (mimeType.includes("quicktime")) return "mov";
  if (mimeType.includes("x-msvideo")) return "avi";
  if (mimeType.includes("matroska")) return "mkv";
  return "mp4";
}

function showNotification(title, message) {
  chrome.notifications.create({
    type: "basic",
    iconUrl: "icon.svg",
    title,
    message
  });
}
