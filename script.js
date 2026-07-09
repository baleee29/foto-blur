const MEDIAPIPE_BASE_URLS = [
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22",
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest",
];

const MODEL_PATH = "./hand_landmarker.task";
const LOVE_IMAGE_PATH = "./assets/Lovesign.jpeg";
const MAX_BLUR = 45;
const BLUR_STEP_UP = 3;
const BLUR_STEP_DOWN = 2;
const STABLE_POSE_FRAMES = 3;
const STABLE_LOVE_FRAMES = 8;
const LOVE_RAIN_BATCH_SIZE = 5;
const LOVE_RAIN_INTERVAL_MS = 120;
const MAX_LOVE_PHOTOS = 90;

const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

const elements = {
  video: document.querySelector("#inputVideo"),
  canvas: document.querySelector("#outputCanvas"),
  loveRain: document.querySelector("#loveRain"),
  emptyState: document.querySelector("#emptyState"),
  startCameraButton: document.querySelector("#startCameraButton"),
  toggleCameraButton: document.querySelector("#toggleCameraButton"),
  toggleLandmarksButton: document.querySelector("#toggleLandmarksButton"),
  toggleMirrorButton: document.querySelector("#toggleMirrorButton"),
  statusDot: document.querySelector("#statusDot"),
  appStatus: document.querySelector("#appStatus"),
  modelState: document.querySelector("#modelState"),
  handCount: document.querySelector("#handCount"),
  poseFrames: document.querySelector("#poseFrames"),
  effectState: document.querySelector("#effectState"),
};

const ctx = elements.canvas.getContext("2d");
const blurCanvas = document.createElement("canvas");
const blurCtx = blurCanvas.getContext("2d");

function shouldUseMobileBlurFallback() {
  const userAgent = navigator.userAgent || "";
  const isPhoneOrTablet = /Android|iPhone|iPad|iPod/i.test(userAgent);
  const isTouchIpadDesktopMode = /Macintosh/i.test(userAgent) && navigator.maxTouchPoints > 1;

  return isPhoneOrTablet || isTouchIpadDesktopMode;
}

const state = {
  mediaPipeModule: null,
  mediaPipeBaseUrl: null,
  handLandmarker: null,
  stream: null,
  rafId: null,
  isRunning: false,
  showLandmarks: true,
  mirror: true,
  blurStrength: 0,
  poseFrameCount: 0,
  loveFrameCount: 0,
  lastLoveRainAt: 0,
  lastResults: null,
  useNativeCanvasBlur: "filter" in ctx && !shouldUseMobileBlurFallback(),
};

function setStatus(text, tone = "idle") {
  elements.appStatus.textContent = text;
  elements.statusDot.classList.toggle("is-live", tone === "live");
  elements.statusDot.classList.toggle("is-error", tone === "error");
}

function setBusy(isBusy) {
  elements.startCameraButton.disabled = isBusy;
  elements.toggleCameraButton.disabled = isBusy || !state.isRunning;
}

function isFingerUp(handLandmarks, tipId, pipId) {
  return handLandmarks[tipId].y < handLandmarks[pipId].y;
}

function isPeaceSign(handLandmarks) {
  const indexUp = isFingerUp(handLandmarks, 8, 6);
  const middleUp = isFingerUp(handLandmarks, 12, 10);
  const ringUp = isFingerUp(handLandmarks, 16, 14);
  const pinkyUp = isFingerUp(handLandmarks, 20, 18);

  return indexUp && middleUp && !ringUp && !pinkyUp;
}

function landmarkDistance(firstLandmark, secondLandmark) {
  const deltaX = firstLandmark.x - secondLandmark.x;
  const deltaY = firstLandmark.y - secondLandmark.y;
  return Math.hypot(deltaX, deltaY);
}

function midpoint(firstLandmark, secondLandmark) {
  return {
    x: (firstLandmark.x + secondLandmark.x) / 2,
    y: (firstLandmark.y + secondLandmark.y) / 2,
  };
}

function getHandScale(handLandmarks) {
  return Math.max(landmarkDistance(handLandmarks[0], handLandmarks[9]), 0.001);
}

function isTwoHandLovePair(firstHand, secondHand) {
  const indexTipDistance = landmarkDistance(firstHand[8], secondHand[8]);
  const thumbTipDistance = landmarkDistance(firstHand[4], secondHand[4]);
  const averageHandScale = (getHandScale(firstHand) + getHandScale(secondHand)) / 2;
  const closeThreshold = Math.min(0.085, Math.max(0.035, averageHandScale * 0.55));
  const indexMidpoint = midpoint(firstHand[8], secondHand[8]);
  const thumbMidpoint = midpoint(firstHand[4], secondHand[4]);
  const heartHeight = thumbMidpoint.y - indexMidpoint.y;
  const centerAligned = Math.abs(indexMidpoint.x - thumbMidpoint.x) < closeThreshold * 1.35;
  const indexTipsAboveThumbs = firstHand[8].y < firstHand[4].y && secondHand[8].y < secondHand[4].y;
  const wristDistance = Math.abs(firstHand[0].x - secondHand[0].x);
  const wristsReasonablyApart = wristDistance > closeThreshold * 1.6 && wristDistance < 0.72;

  return (
    wristsReasonablyApart &&
    centerAligned &&
    indexTipsAboveThumbs &&
    indexTipDistance < closeThreshold &&
    thumbTipDistance < closeThreshold &&
    heartHeight > closeThreshold * 0.8
  );
}

function isTwoHandLoveSign(handLandmarks) {
  if (handLandmarks.length < 2) {
    return false;
  }

  for (let firstIndex = 0; firstIndex < handLandmarks.length - 1; firstIndex += 1) {
    for (let secondIndex = firstIndex + 1; secondIndex < handLandmarks.length; secondIndex += 1) {
      if (isTwoHandLovePair(handLandmarks[firstIndex], handLandmarks[secondIndex])) {
        return true;
      }
    }
  }

  return false;
}

function updatePoseState(handLandmarks) {
  const rawPoseDetected = handLandmarks.some(isPeaceSign);

  if (rawPoseDetected) {
    state.poseFrameCount += 1;
  } else {
    state.poseFrameCount = 0;
  }

  const poseDetected = state.poseFrameCount >= STABLE_POSE_FRAMES;

  if (poseDetected) {
    state.blurStrength = Math.min(MAX_BLUR, state.blurStrength + BLUR_STEP_UP);
  } else {
    state.blurStrength = Math.max(0, state.blurStrength - BLUR_STEP_DOWN);
  }

  return poseDetected;
}

function updateLoveState(handLandmarks) {
  const rawLoveDetected = isTwoHandLoveSign(handLandmarks);

  if (rawLoveDetected) {
    state.loveFrameCount += 1;
  } else {
    state.loveFrameCount = 0;
  }

  const loveDetected = state.loveFrameCount >= STABLE_LOVE_FRAMES;

  return loveDetected;
}

function randomBetween(min, max) {
  return min + Math.random() * (max - min);
}

function spawnLovePhoto() {
  const particle = document.createElement("span");
  const size = randomBetween(44, 92);
  const drift = randomBetween(-120, 120);
  const startRotation = randomBetween(-28, 28);
  const endRotation = startRotation + randomBetween(-130, 130);
  const duration = randomBetween(2.8, 5.4);

  particle.className = "love-photo";
  particle.style.left = `${randomBetween(-4, 96)}%`;
  particle.style.setProperty("--size", `${size}px`);
  particle.style.setProperty("--drift", `${drift}px`);
  particle.style.setProperty("--start-rotation", `${startRotation}deg`);
  particle.style.setProperty("--end-rotation", `${endRotation}deg`);
  particle.style.setProperty("--duration", `${duration}s`);
  particle.style.backgroundImage = `url("${LOVE_IMAGE_PATH}")`;

  particle.addEventListener("animationend", () => particle.remove(), { once: true });
  elements.loveRain.appendChild(particle);

  while (elements.loveRain.childElementCount > MAX_LOVE_PHOTOS) {
    elements.loveRain.firstElementChild.remove();
  }
}

function updateLoveRain(loveDetected) {
  if (!loveDetected) {
    state.lastLoveRainAt = 0;
    clearLoveRain();
    return;
  }

  const now = performance.now();
  if (now - state.lastLoveRainAt < LOVE_RAIN_INTERVAL_MS) {
    return;
  }

  state.lastLoveRainAt = now;
  for (let index = 0; index < LOVE_RAIN_BATCH_SIZE; index += 1) {
    spawnLovePhoto();
  }
}

function clearLoveRain() {
  elements.loveRain.replaceChildren();
}

function resizeCanvasToVideo() {
  const width = elements.video.videoWidth || 1280;
  const height = elements.video.videoHeight || 720;

  if (elements.canvas.width !== width || elements.canvas.height !== height) {
    elements.canvas.width = width;
    elements.canvas.height = height;
  }
}

function getLandmarkPoint(landmark, width, height) {
  const x = state.mirror ? (1 - landmark.x) * width : landmark.x * width;
  return {
    x,
    y: landmark.y * height,
  };
}

function drawLandmarks(handLandmarks) {
  if (!state.showLandmarks || !handLandmarks.length) {
    return;
  }

  const { width, height } = elements.canvas;

  ctx.save();
  ctx.lineWidth = Math.max(2, width / 520);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const landmarks of handLandmarks) {
    ctx.strokeStyle = "rgba(65, 217, 138, 0.88)";
    for (const [startIndex, endIndex] of HAND_CONNECTIONS) {
      const start = getLandmarkPoint(landmarks[startIndex], width, height);
      const end = getLandmarkPoint(landmarks[endIndex], width, height);

      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(end.x, end.y);
      ctx.stroke();
    }

    for (const landmark of landmarks) {
      const point = getLandmarkPoint(landmark, width, height);
      ctx.beginPath();
      ctx.fillStyle = "#f3ba4d";
      ctx.arc(point.x, point.y, Math.max(3, width / 240), 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.restore();
}

function drawVideoSource(targetCtx, width, height) {
  targetCtx.save();

  if (state.mirror) {
    targetCtx.translate(width, 0);
    targetCtx.scale(-1, 1);
  }

  targetCtx.drawImage(elements.video, 0, 0, width, height);
  targetCtx.restore();
}

function resizeBlurCanvas(width, height) {
  if (blurCanvas.width !== width || blurCanvas.height !== height) {
    blurCanvas.width = width;
    blurCanvas.height = height;
  }
}

function drawFallbackBlurredVideo(width, height) {
  const blurRatio = state.blurStrength / MAX_BLUR;
  const scale = Math.max(0.08, 1 - blurRatio * 0.92);
  const scaledWidth = Math.max(24, Math.round(width * scale));
  const scaledHeight = Math.max(24, Math.round(height * scale));

  resizeBlurCanvas(scaledWidth, scaledHeight);

  blurCtx.clearRect(0, 0, scaledWidth, scaledHeight);
  blurCtx.imageSmoothingEnabled = true;
  blurCtx.imageSmoothingQuality = "high";
  drawVideoSource(blurCtx, scaledWidth, scaledHeight);

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(blurCanvas, 0, 0, scaledWidth, scaledHeight, 0, 0, width, height);
}

function drawVideoFrame(handLandmarks) {
  resizeCanvasToVideo();

  const { width, height } = elements.canvas;
  ctx.clearRect(0, 0, width, height);

  if (state.blurStrength <= 0) {
    drawVideoSource(ctx, width, height);
  } else if (state.useNativeCanvasBlur) {
    ctx.save();
    ctx.filter = `blur(${state.blurStrength}px)`;
    drawVideoSource(ctx, width, height);
    ctx.restore();
  } else {
    drawFallbackBlurredVideo(width, height);
  }

  drawLandmarks(handLandmarks);
}

function updateDisplay(handLandmarks, loveDetected) {
  elements.handCount.textContent = String(handLandmarks.length);
  elements.poseFrames.textContent = String(Math.min(state.poseFrameCount, STABLE_POSE_FRAMES));
  elements.effectState.textContent = loveDetected ? "Love" : state.blurStrength > 0 ? "Blur" : "Normal";
}

async function loadMediaPipe() {
  if (state.mediaPipeModule) {
    return state.mediaPipeModule;
  }

  let lastError = null;

  for (const baseUrl of MEDIAPIPE_BASE_URLS) {
    try {
      const module = await import(baseUrl);
      state.mediaPipeModule = module;
      state.mediaPipeBaseUrl = baseUrl;
      return module;
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError;
}

async function setupHandLandmarker() {
  if (state.handLandmarker) {
    return state.handLandmarker;
  }

  elements.modelState.textContent = "Loading";
  const { FilesetResolver, HandLandmarker } = await loadMediaPipe();
  const vision = await FilesetResolver.forVisionTasks(`${state.mediaPipeBaseUrl}/wasm`);

  state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_PATH,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.7,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  elements.modelState.textContent = "Ready";
  return state.handLandmarker;
}

async function setupCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Browser tidak mendukung akses kamera.");
  }

  state.stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user",
    },
    audio: false,
  });

  elements.video.srcObject = state.stream;
  await elements.video.play();
  resizeCanvasToVideo();
}

function renderLoop() {
  if (!state.isRunning) {
    return;
  }

  let handLandmarks = [];

  if (elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    const results = state.handLandmarker.detectForVideo(elements.video, performance.now());
    handLandmarks = results.landmarks || [];
    state.lastResults = results;
  }

  updatePoseState(handLandmarks);
  const loveDetected = updateLoveState(handLandmarks);
  updateLoveRain(loveDetected);
  drawVideoFrame(handLandmarks);
  updateDisplay(handLandmarks, loveDetected);

  state.rafId = window.requestAnimationFrame(renderLoop);
}

async function startCamera() {
  try {
    setBusy(true);
    setStatus("Loading model");
    await setupHandLandmarker();

    setStatus("Opening camera");
    await setupCamera();

    state.isRunning = true;
    elements.emptyState.classList.add("is-hidden");
    elements.toggleCameraButton.disabled = false;
    elements.startCameraButton.disabled = true;
    setStatus("Live", "live");
    renderLoop();
  } catch (error) {
    setStatus("Camera error", "error");
    elements.modelState.textContent = state.handLandmarker ? "Ready" : "Error";
    elements.emptyState.classList.remove("is-hidden");
    elements.startCameraButton.disabled = false;
    console.error(error);
    alert(error.message || "Kamera gagal dibuka.");
  } finally {
    setBusy(false);
  }
}

function stopCamera() {
  state.isRunning = false;

  if (state.rafId) {
    window.cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }

  if (state.stream) {
    for (const track of state.stream.getTracks()) {
      track.stop();
    }
    state.stream = null;
  }

  state.blurStrength = 0;
  state.poseFrameCount = 0;
  state.loveFrameCount = 0;
  state.lastLoveRainAt = 0;
  ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
  elements.video.srcObject = null;
  clearLoveRain();
  elements.emptyState.classList.remove("is-hidden");
  elements.startCameraButton.disabled = false;
  elements.toggleCameraButton.disabled = true;
  updateDisplay([], false);
  setStatus("Ready");
}

function toggleLandmarks() {
  state.showLandmarks = !state.showLandmarks;
  elements.toggleLandmarksButton.classList.toggle("is-active", state.showLandmarks);
}

function toggleMirror() {
  state.mirror = !state.mirror;
  elements.toggleMirrorButton.classList.toggle("is-active", state.mirror);
}

elements.startCameraButton.addEventListener("click", startCamera);
elements.toggleCameraButton.addEventListener("click", stopCamera);
elements.toggleLandmarksButton.addEventListener("click", toggleLandmarks);
elements.toggleMirrorButton.addEventListener("click", toggleMirror);

window.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() === "q" && state.isRunning) {
    stopCamera();
  }
});

window.addEventListener("beforeunload", stopCamera);

window.addEventListener("load", () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }
});
