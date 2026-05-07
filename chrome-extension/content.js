let lastContextVideo = null;

function getVideoSources(video) {
  const sources = [];
  const addSource = (value) => {
    if (!value || sources.includes(value)) return;
    sources.push(value);
  };

  addSource(video.currentSrc);
  addSource(video.src);
  for (const source of video.querySelectorAll("source")) {
    addSource(source.src);
  }
  return sources;
}

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
