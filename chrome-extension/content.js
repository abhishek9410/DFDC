let lastContextVideo = null;

const OVERLAY_CLASS = "dfd-video-overlay";
const OVERLAY_STYLE_ID = "dfd-video-overlay-styles";
const MIN_VISIBLE_VIDEO_AREA = 160 * 90;
const overlays = new Map();
const dismissedVideos = new WeakSet();
let hoveredVideo = null;
let hideOverlayTimer = null;

function getVideoSources(video) {
  const sources = [];
  const addSource = (value) => {
    const absolute = toAbsoluteUrl(value);
    if (!absolute || sources.includes(absolute)) return;
    sources.push(absolute);
  };

  addSource(video.currentSrc);
  addSource(video.src);
  for (const source of video.querySelectorAll("source")) {
    addSource(source.src);
  }
  return sources;
}

function toAbsoluteUrl(value) {
  if (!value) return "";
  if (/^blob:/i.test(value)) return value;

  try {
    return new URL(value, document.baseURI).href;
  } catch (_error) {
    return "";
  }
}

function isVideoVisible(video) {
  if (!video.isConnected || dismissedVideos.has(video)) return false;

  const rect = video.getBoundingClientRect();
  if (rect.width * rect.height < MIN_VISIBLE_VIDEO_AREA) return false;
  if (rect.bottom <= 0 || rect.right <= 0 || rect.top >= window.innerHeight || rect.left >= window.innerWidth) {
    return false;
  }

  const style = window.getComputedStyle(video);
  if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity) === 0) {
    return false;
  }

  return true;
}

function selectBestSource(video) {
  const sources = getVideoSources(video);
  const usable = sources.filter((source) => /^https?:\/\//i.test(source) || /^blob:/i.test(source));
  return usable[0] || sources[0] || "";
}

function ensureOverlayStyles() {
  if (document.getElementById(OVERLAY_STYLE_ID)) return;

  const style = document.createElement("style");
  style.id = OVERLAY_STYLE_ID;
  style.textContent = `
    .${OVERLAY_CLASS} {
      position: fixed;
      z-index: 2147483647;
      width: min(220px, calc(100vw - 24px));
      box-sizing: border-box;
      padding: 10px;
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 10px;
      background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(2, 6, 23, 0.9));
      color: #f8fafc;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.08);
      backdrop-filter: blur(16px) saturate(1.2);
      -webkit-backdrop-filter: blur(16px) saturate(1.2);
      transform: translateY(-4px) scale(0.98);
      opacity: 0;
      pointer-events: auto;
      transition: opacity 180ms ease, transform 180ms ease, box-shadow 180ms ease;
      user-select: none;
    }

    .${OVERLAY_CLASS}.is-visible {
      opacity: 1;
      transform: translateY(0) scale(1);
    }

    .${OVERLAY_CLASS}.is-dragging {
      transition: none;
      box-shadow: 0 22px 56px rgba(0, 0, 0, 0.44);
      cursor: grabbing;
    }

    .dfd-overlay-title {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 0 0 9px;
      font-size: 13px;
      font-weight: 750;
      line-height: 1.25;
      letter-spacing: 0;
      cursor: grab;
    }

    .dfd-overlay-mark {
      display: inline-grid;
      place-items: center;
      width: 22px;
      height: 22px;
      flex: 0 0 22px;
      border-radius: 7px;
      color: #67e8f9;
      background: rgba(8, 145, 178, 0.18);
      border: 1px solid rgba(103, 232, 249, 0.24);
      font-size: 11px;
      font-weight: 800;
      line-height: 1;
    }

    .dfd-overlay-actions {
      display: flex;
      align-items: center;
      gap: 7px;
    }

    .dfd-overlay-button {
      appearance: none;
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 7px;
      min-height: 30px;
      padding: 0 11px;
      color: #f8fafc;
      font: 700 12px/1 Inter, ui-sans-serif, system-ui, sans-serif;
      letter-spacing: 0;
      background: rgba(15, 23, 42, 0.72);
      cursor: pointer;
      transition: background 160ms ease, border-color 160ms ease, transform 160ms ease;
    }

    .dfd-overlay-button:hover {
      background: rgba(255, 255, 255, 0.14);
      border-color: rgba(255, 255, 255, 0.22);
      transform: translateY(-1px);
    }

    .dfd-overlay-button:disabled {
      cursor: wait;
      opacity: 0.72;
      transform: none;
    }

    .dfd-overlay-primary {
      border-color: rgba(34, 211, 238, 0.36);
      background: linear-gradient(135deg, #0e7490, #2563eb);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.13);
    }

    .dfd-overlay-primary:hover {
      background: linear-gradient(135deg, #0e7490, #1d4ed8);
    }

    .dfd-overlay-status {
      display: none;
      margin-top: 9px;
      color: #cbd5e1;
      font-size: 12px;
      line-height: 1.35;
      letter-spacing: 0;
      word-break: break-word;
    }

    .dfd-overlay-status.is-visible {
      display: block;
    }

    .dfd-overlay-result {
      display: none;
      margin-top: 10px;
      padding: 9px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: rgba(2, 6, 23, 0.42);
      font-size: 12px;
      line-height: 1.35;
      letter-spacing: 0;
    }

    .dfd-overlay-result.is-visible {
      display: block;
    }

    .dfd-overlay-label {
      font-weight: 800;
      letter-spacing: 0;
    }

    .dfd-overlay-label.fake {
      color: #fca5a5;
    }

    .dfd-overlay-label.real {
      color: #86efac;
    }

    .dfd-overlay-label.unsupported,
    .dfd-overlay-label.error {
      color: #fde68a;
    }

    @media (max-width: 520px) {
      .${OVERLAY_CLASS} {
        width: min(206px, calc(100vw - 16px));
        padding: 10px;
        border-radius: 10px;
      }

      .dfd-overlay-actions {
        gap: 6px;
      }

      .dfd-overlay-button {
        min-height: 30px;
        padding: 0 10px;
      }
    }
  `;
  document.documentElement.appendChild(style);
}

function createOverlay(video) {
  ensureOverlayStyles();

  const overlay = document.createElement("div");
  overlay.className = OVERLAY_CLASS;
  overlay.innerHTML = `
    <div class="dfd-overlay-title" data-drag-handle="true">
      <span class="dfd-overlay-mark" aria-hidden="true">AI</span>
      <span>Analyze Video?</span>
    </div>
    <div class="dfd-overlay-actions">
      <button type="button" class="dfd-overlay-button dfd-overlay-primary" data-action="analyze">Analyze</button>
      <button type="button" class="dfd-overlay-button" data-action="dismiss">Dismiss</button>
    </div>
    <div class="dfd-overlay-status"></div>
    <div class="dfd-overlay-result"></div>
  `;

  const state = {
    overlay,
    video,
    dragging: false,
    dragOffsetX: 0,
    dragOffsetY: 0,
    positionRatio: null,
    analyzeButton: overlay.querySelector('[data-action="analyze"]'),
    dismissButton: overlay.querySelector('[data-action="dismiss"]'),
    status: overlay.querySelector(".dfd-overlay-status"),
    result: overlay.querySelector(".dfd-overlay-result")
  };

  state.analyzeButton.addEventListener("click", () => analyzeVideoFromOverlay(state));
  state.dismissButton.addEventListener("click", () => dismissOverlay(state));
  overlay.addEventListener("pointerdown", (event) => startDrag(event, state));
  overlay.addEventListener("pointerenter", () => {
    clearTimeout(hideOverlayTimer);
  });
  overlay.addEventListener("pointerleave", () => {
    scheduleOverlayHide(video);
  });

  document.documentElement.appendChild(overlay);
  overlays.set(video, state);
  requestAnimationFrame(() => overlay.classList.add("is-visible"));
  return state;
}

function updateOverlayPosition(state) {
  if (!state.overlay.isConnected) return;

  const videoRect = state.video.getBoundingClientRect();
  const overlayRect = state.overlay.getBoundingClientRect();
  const margin = 12;
  const controlSafeInset = Math.min(72, Math.max(44, videoRect.height * 0.18));

  let left;
  let top;
  if (state.positionRatio) {
    left = videoRect.left + (videoRect.width - overlayRect.width) * state.positionRatio.x;
    top = videoRect.top + (videoRect.height - overlayRect.height - controlSafeInset) * state.positionRatio.y;
  } else {
    left = videoRect.right - overlayRect.width - margin;
    top = videoRect.top + margin;
  }

  const maxLeft = Math.min(window.innerWidth - overlayRect.width - 8, videoRect.right - overlayRect.width - 8);
  const minLeft = Math.max(8, videoRect.left + 8);
  const maxTop = Math.min(window.innerHeight - overlayRect.height - 8, videoRect.bottom - overlayRect.height - controlSafeInset);
  const minTop = Math.max(8, videoRect.top + 8);

  state.overlay.style.left = `${clamp(left, minLeft, maxLeft)}px`;
  state.overlay.style.top = `${clamp(top, minTop, maxTop)}px`;
}

function startDrag(event, state) {
  if (event.target.closest("button")) return;
  if (!event.target.closest('[data-drag-handle="true"], .' + OVERLAY_CLASS)) return;

  const rect = state.overlay.getBoundingClientRect();
  state.dragging = true;
  state.dragOffsetX = event.clientX - rect.left;
  state.dragOffsetY = event.clientY - rect.top;
  state.overlay.classList.add("is-dragging");
  state.overlay.setPointerCapture?.(event.pointerId);

  const onMove = (moveEvent) => moveOverlay(moveEvent, state);
  const onUp = (upEvent) => {
    state.dragging = false;
    state.overlay.classList.remove("is-dragging");
    state.overlay.releasePointerCapture?.(upEvent.pointerId);
    state.overlay.removeEventListener("pointermove", onMove);
    state.overlay.removeEventListener("pointerup", onUp);
    state.overlay.removeEventListener("pointercancel", onUp);
    saveDragRatio(state);
  };

  state.overlay.addEventListener("pointermove", onMove);
  state.overlay.addEventListener("pointerup", onUp);
  state.overlay.addEventListener("pointercancel", onUp);
  event.preventDefault();
}

function moveOverlay(event, state) {
  if (!state.dragging) return;

  const videoRect = state.video.getBoundingClientRect();
  const overlayRect = state.overlay.getBoundingClientRect();
  const controlSafeInset = Math.min(72, Math.max(44, videoRect.height * 0.18));
  const minLeft = Math.max(8, videoRect.left + 8);
  const maxLeft = Math.min(window.innerWidth - overlayRect.width - 8, videoRect.right - overlayRect.width - 8);
  const minTop = Math.max(8, videoRect.top + 8);
  const maxTop = Math.min(window.innerHeight - overlayRect.height - 8, videoRect.bottom - overlayRect.height - controlSafeInset);

  state.overlay.style.left = `${clamp(event.clientX - state.dragOffsetX, minLeft, maxLeft)}px`;
  state.overlay.style.top = `${clamp(event.clientY - state.dragOffsetY, minTop, maxTop)}px`;
}

function saveDragRatio(state) {
  const videoRect = state.video.getBoundingClientRect();
  const overlayRect = state.overlay.getBoundingClientRect();
  const controlSafeInset = Math.min(72, Math.max(44, videoRect.height * 0.18));
  const availableX = Math.max(1, videoRect.width - overlayRect.width);
  const availableY = Math.max(1, videoRect.height - overlayRect.height - controlSafeInset);

  state.positionRatio = {
    x: clamp((overlayRect.left - videoRect.left) / availableX, 0, 1),
    y: clamp((overlayRect.top - videoRect.top) / availableY, 0, 1)
  };
}

function dismissOverlay(state) {
  dismissedVideos.add(state.video);
  if (hoveredVideo === state.video) hoveredVideo = null;
  removeOverlay(state.video);
}

function removeOverlay(video) {
  const state = overlays.get(video);
  if (!state) return;

  state.overlay.classList.remove("is-visible");
  setTimeout(() => state.overlay.remove(), 180);
  overlays.delete(video);
}

async function analyzeVideoFromOverlay(state) {
  const source = selectBestSource(state.video);
  if (!source) {
    setOverlayError(state, "This video does not expose a readable source URL.");
    return;
  }

  state.analyzeButton.disabled = true;
  state.dismissButton.disabled = true;
  setOverlayStatus(state, "Analyzing video. This can take a minute on CPU.");
  state.result.classList.remove("is-visible");
  state.result.textContent = "";

  try {
    const response = await chrome.runtime.sendMessage({
      type: "ANALYZE_VIDEO_SOURCE",
      source,
      sources: getVideoSources(state.video),
      currentTime: state.video.currentTime,
      pageTitle: document.title || "page-video"
    });

    if (!response?.ok) {
      throw new Error(response?.error || "Prediction failed.");
    }

    showOverlayResult(state, response.result);
  } catch (error) {
    setOverlayError(state, error.message || "Could not analyze this video.");
  } finally {
    state.analyzeButton.disabled = false;
    state.dismissButton.disabled = false;
  }
}

function setOverlayStatus(state, message) {
  state.status.textContent = message;
  state.status.classList.add("is-visible");
}

function setOverlayError(state, message) {
  state.status.classList.remove("is-visible");
  state.result.innerHTML = `<span class="dfd-overlay-label error">Error</span><br>${escapeHtml(message)}`;
  state.result.classList.add("is-visible");
}

function showOverlayResult(state, result) {
  state.status.classList.remove("is-visible");

  if (result.status === "unsupported") {
    state.result.innerHTML = `
      <span class="dfd-overlay-label unsupported">${escapeHtml(result.label || "Unsupported")}</span>
      <br>${escapeHtml(result.message || "This video is not suitable for analysis.")}
    `;
    state.result.classList.add("is-visible");
    return;
  }

  const label = result.label || "UNKNOWN";
  const confidence = Number.isFinite(Number(result.probability))
    ? `${(Number(result.probability) * 100).toFixed(1)}% confidence`
    : "Prediction complete";
  state.result.innerHTML = `
    <span class="dfd-overlay-label ${label.toLowerCase()}">${escapeHtml(label)}</span>
    <br>${escapeHtml(confidence)}
  `;
  state.result.classList.add("is-visible");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[char]);
}

function clamp(value, min, max) {
  if (max < min) return min;
  return Math.min(Math.max(value, min), max);
}

function refreshVideoOverlays() {
  if (hoveredVideo && isVideoVisible(hoveredVideo)) {
    const state = overlays.get(hoveredVideo) || createOverlay(hoveredVideo);
    updateOverlayPosition(state);
  }

  for (const [video, state] of overlays.entries()) {
    if (!video.isConnected || video !== hoveredVideo || !isVideoVisible(video)) {
      state.overlay.remove();
      overlays.delete(video);
    }
  }
}

function getVideoFromPoint(event) {
  return event.target?.closest?.("video") || null;
}

function showOverlayForVideo(video) {
  if (!video || dismissedVideos.has(video) || !isVideoVisible(video)) return;
  clearTimeout(hideOverlayTimer);

  if (hoveredVideo && hoveredVideo !== video) {
    removeOverlay(hoveredVideo);
  }

  hoveredVideo = video;
  scheduleOverlayRefresh();
}

function scheduleOverlayHide(video) {
  clearTimeout(hideOverlayTimer);
  hideOverlayTimer = setTimeout(() => {
    const state = overlays.get(video);
    if (!state || state.dragging || state.analyzeButton.disabled) return;
    if (hoveredVideo === video) hoveredVideo = null;
    removeOverlay(video);
  }, 140);
}

const scheduleOverlayRefresh = (() => {
  let scheduled = false;
  return () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      refreshVideoOverlays();
    });
  };
})();

document.addEventListener("contextmenu", (event) => {
  const video = event.target?.closest?.("video");
  if (!video) {
    lastContextVideo = null;
    return;
  }

  lastContextVideo = {
    sources: getVideoSources(video),
    currentTime: video.currentTime,
    pageTitle: document.title || "context-video"
  };
}, true);

document.addEventListener("pointerover", (event) => {
  const video = getVideoFromPoint(event);
  if (video) {
    showOverlayForVideo(video);
  }
}, true);

document.addEventListener("pointerout", (event) => {
  const video = getVideoFromPoint(event);
  if (!video || event.relatedTarget?.closest?.("video") === video) return;
  scheduleOverlayHide(video);
}, true);

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "GET_CONTEXT_VIDEO") {
    sendResponse(lastContextVideo || { sources: [], pageTitle: document.title || "context-video" });
    return true;
  }

  if (message?.type !== "GET_PAGE_VIDEO") {
    return false;
  }

  const videos = Array.from(document.querySelectorAll("video"));
  const candidates = videos
    .map((video) => video.currentSrc || video.src)
    .filter(Boolean)
    .filter((src) => src.startsWith("http://") || src.startsWith("https://"));

  sendResponse({
    url: candidates[0] || "",
    count: videos.length,
    foundDirectUrl: candidates.length > 0
  });
  return true;
});

new MutationObserver(scheduleOverlayRefresh).observe(document.documentElement, {
  childList: true,
  subtree: true,
  attributes: true,
  attributeFilter: ["src", "style", "class"]
});

window.addEventListener("scroll", scheduleOverlayRefresh, true);
window.addEventListener("resize", scheduleOverlayRefresh);
document.addEventListener("fullscreenchange", scheduleOverlayRefresh);
setInterval(scheduleOverlayRefresh, 1200);
