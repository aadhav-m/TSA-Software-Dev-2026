import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("video");
const startBtn = document.getElementById("startButton");
const stopBtn = document.getElementById("stopButton");
const letterDiv = document.getElementById("translation-letter");
const sentenceDiv = document.getElementById("translation-sentence");
const clearBtn = document.getElementById("clearTranslation");

const HOLD_MS = 1200;
const REPEAT_DELAY_MS = 1800;
const SMOOTH_WINDOW = 9;
const NOTHING_FRAMES = 4;
const MOTION_FRAMES = 8;
const MOTION_T = 0.014;

const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

const FINGER_COLORS = ["#ffffff","#ffffff","#ffffff","#ffffff","#ffffff"];
const FINGER_RANGES = [[1,4],[5,8],[9,12],[13,16],[17,20]];

let handLandmarker = null;
let stream = null;
let rafId = null;
let running = false;
let lastVideoTime = -1;
let canvas = null;
let ctx = null;

let sentenceChars = [];
let lastLetter = "";
let holdStart = 0;
let lastCommitTime = 0;
let lastCommitChar = "";
let nothingCount = 0;

const predictionBuffer = [];
const jHistory = [];
const zHistory = [];
let prevPinky = null;
let prevIndex = null;

function renderSentence() {
  sentenceDiv.textContent = sentenceChars.join("") || "\u00a0";
}

function setupCanvas() {
  if (canvas) canvas.remove();
  canvas = document.createElement("canvas");
  canvas.style.position = "absolute";
  canvas.style.pointerEvents = "none";
  canvas.style.zIndex = "10";
  canvas.width = 640;
  canvas.height = 480;
  video.parentNode.insertBefore(canvas, video.nextSibling);
  ctx = canvas.getContext("2d");
  positionCanvas();
}

function positionCanvas() {
  if (!canvas) return;
  const r = video.getBoundingClientRect();
  const pr = video.parentNode.getBoundingClientRect();
  canvas.style.top = (r.top - pr.top) + "px";
  canvas.style.left = (r.left - pr.left) + "px";
  canvas.style.width = r.width + "px";
  canvas.style.height = r.height + "px";
}

function getFingerForLandmark(idx) {
  for (let f = 0; f < FINGER_RANGES.length; f++) {
    const [s, e] = FINGER_RANGES[f];
    if (idx >= s && idx <= e) return f;
  }
  return -1;
}

function drawSkeleton(lm) {
  if (!ctx || !canvas) return;
  positionCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const sx = (v) => v * canvas.width;
  const sy = (v) => v * canvas.height;
  for (const [a, b] of CONNECTIONS) {
    const fi = getFingerForLandmark(a);
    const color = fi >= 0 ? FINGER_COLORS[fi] : "#ffffff";
    ctx.beginPath();
    ctx.moveTo(sx(lm[a].x), sy(lm[a].y));
    ctx.lineTo(sx(lm[b].x), sy(lm[b].y));
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
  for (let i = 0; i < lm.length; i++) {
    const fi = getFingerForLandmark(i);
    const color = fi >= 0 ? FINGER_COLORS[fi] : "#ffffff";
    const isTip = [4,8,12,16,20].includes(i);
    const r = isTip ? 6 : 4;
    ctx.beginPath();
    ctx.arc(sx(lm[i].x), sy(lm[i].y), r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.arc(sx(lm[i].x), sy(lm[i].y), r * 0.45, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
  }
}

function clearCanvas() {
  if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
}

async function init() {
  letterDiv.textContent = "Loading model…";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
  });
  letterDiv.textContent = "Model ready — press Start";
  startBtn.addEventListener("click", startRecording);
  stopBtn.addEventListener("click", stopRecording);
  stopBtn.disabled = true;
}

async function startRecording() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  setupCanvas();
  running = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  letterDiv.textContent = "Show a sign…";
  rafId = requestAnimationFrame(detect);
}

function stopRecording() {
  running = false;
  cancelAnimationFrame(rafId);
  stream?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  if (canvas) { canvas.remove(); canvas = null; ctx = null; }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  letterDiv.textContent = "Stopped";
  predictionBuffer.length = 0;
  jHistory.length = 0;
  zHistory.length = 0;
  prevPinky = null;
  prevIndex = null;
  lastLetter = "";
  holdStart = 0;
  nothingCount = 0;
}

function detect(now) {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const results = handLandmarker.detectForVideo(video, now);
    if (results.landmarks && results.landmarks.length > 0) {
      nothingCount = 0;
      drawSkeleton(results.landmarks[0]);
      window.__lastLM = results.landmarks[0];
      processLandmarks(results.landmarks[0], now);
    } else {
      nothingCount++;
      if (nothingCount >= NOTHING_FRAMES) {
        clearCanvas();
        predictionBuffer.length = 0;
        jHistory.length = 0;
        zHistory.length = 0;
        prevPinky = null;
        prevIndex = null;
        lastLetter = "";
        holdStart = 0;
        letterDiv.textContent = "Show a sign…";
      }
    }
  }
  rafId = requestAnimationFrame(detect);
}

function smoothedPrediction(raw) {
  predictionBuffer.push(raw);
  if (predictionBuffer.length > SMOOTH_WINDOW) predictionBuffer.shift();
  const counts = {};
  for (const label of predictionBuffer) counts[label] = (counts[label] || 0) + 1;
  let best = raw, bestCount = 0;
  for (const [label, count] of Object.entries(counts)) {
    if (count > bestCount) { best = label; bestCount = count; }
  }
  const MIN_CONFIDENCE = Math.ceil(SMOOTH_WINDOW * 0.4);
  return bestCount >= MIN_CONFIDENCE ? best : "NOTHING";
}

function trackMotion(lm) {
  const px = lm[20].x, py = lm[20].y;
  const ix = lm[8].x,  iy = lm[8].y;
  const pdx = prevPinky ? px - prevPinky.x : 0;
  const pdy = prevPinky ? py - prevPinky.y : 0;
  const idx = prevIndex ? ix - prevIndex.x : 0;
  prevPinky = { x: px, y: py };
  prevIndex = { x: ix, y: iy };
  jHistory.push({ dx: pdx, dy: pdy });
  if (jHistory.length > MOTION_FRAMES) jHistory.shift();
  zHistory.push({ dx: idx, dy: 0 });
  if (zHistory.length > MOTION_FRAMES) zHistory.shift();
}

function detectJ() {
  if (jHistory.length < MOTION_FRAMES) return false;
  const sumDy = jHistory.reduce((s, f) => s + f.dy, 0);
  const sumAbsDx = jHistory.reduce((s, f) => s + Math.abs(f.dx), 0);
  return sumDy > MOTION_T * 4 && sumAbsDx > MOTION_T * 2;
}

function detectZ() {
  if (zHistory.length < MOTION_FRAMES) return false;
  const h = Math.floor(zHistory.length / 2);
  const dx1 = zHistory.slice(0, h).reduce((s, f) => s + f.dx, 0);
  const dx2 = zHistory.slice(h).reduce((s, f) => s + f.dx, 0);
  return dx1 > MOTION_T * 2 && dx2 > MOTION_T * 1.5;
}

function processLandmarks(lm, now) {
  trackMotion(lm);
  const raw = classify(lm);
  const letter = smoothedPrediction(raw);

  if (letter === "NOTHING") {
    letterDiv.textContent = "—";
    lastLetter = "";
    holdStart = 0;
    return;
  }

  letterDiv.textContent =
    letter === "SPACE"  ? "[ SPACE ]"  :
    letter === "DELETE" ? "[ DELETE ]" :
    letter;

  if (letter !== lastLetter) {
    lastLetter = letter;
    holdStart = now;
    return;
  }

  if (letter === "I" && detectJ()) {
    commitOnce("J", now);
    jHistory.length = 0;
    return;
  }

  if (letter === "G" && detectZ()) {
    commitOnce("Z", now);
    zHistory.length = 0;
    return;
  }

  const held     = now - holdStart >= HOLD_MS;
  const sameChar = letter === lastCommitChar;
  const cooldown = now - lastCommitTime >= REPEAT_DELAY_MS;
  if (held && (!sameChar || cooldown)) commitOnce(letter, now);
}

function commitOnce(letter, now) {
  commit(letter);
  holdStart = now;
  lastCommitTime = now;
  lastCommitChar = letter;
}

function commit(letter) {
  if (letter === "SPACE") {
    sentenceChars.push(" ");
  } else if (letter === "DELETE") {
    sentenceChars.pop();
  } else {
    sentenceChars.push(letter);
  }
  renderSentence();
}

function classify(lm) {
  const lx = (i) => lm[i].x;
  const ly = (i) => lm[i].y;
  const lz = (i) => lm[i].z;

  const d3 = (a, b) => Math.sqrt(
    (lx(a)-lx(b))**2 + (ly(a)-ly(b))**2 + (lz(a)-lz(b))**2
  );
  const d2 = (a, b) => Math.sqrt(
    (lx(a)-lx(b))**2 + (ly(a)-ly(b))**2
  );

  const palm = d3(0, 9) || 0.0001;
  const n3 = (a, b) => d3(a, b) / palm;
  const n2 = (a, b) => d2(a, b) / palm;

  function ang(a, b, c) {
    const abx = lx(a)-lx(b), aby = ly(a)-ly(b), abz = lz(a)-lz(b);
    const cbx = lx(c)-lx(b), cby = ly(c)-ly(b), cbz = lz(c)-lz(b);
    const ma = Math.sqrt(abx*abx+aby*aby+abz*abz);
    const mc = Math.sqrt(cbx*cbx+cby*cby+cbz*cbz);
    const cos = (abx*cbx+aby*cby+abz*cbz)/(ma*mc||0.0001);
    return Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI;
  }

  const curl = (a, b, c, d) => (ang(a, b, c) + ang(b, c, d)) / 2;

  const IC = curl(5, 6, 7, 8);
  const MC = curl(9, 10, 11, 12);
  const RC = curl(13, 14, 15, 16);
  const PC = curl(17, 18, 19, 20);

  const EXT = 150;
  const CRL = 120;

  const iExt = IC > EXT;
  const mExt = MC > EXT;
  const rExt = RC > EXT;
  const pExt = PC > EXT;

  const iCrl = IC < CRL;
  const mCrl = MC < CRL;
  const rCrl = RC < CRL;
  const pCrl = PC < CRL;

  const iMid = !iExt && !iCrl;
  const mMid = !mExt && !mCrl;

  const t_i = n3(4, 8);
  const t_m = n3(4, 12);
  const t_r = n3(4, 16);
  const t_p = n3(4, 20);
  const i_m = n3(8, 12);

  const imGap = n2(8, 12);
  const mrGap = n2(12, 16);

  const iRelY = (ly(8)  - ly(5))  / palm;
  const mRelY = (ly(12) - ly(9))  / palm;
  const rRelY = (ly(16) - ly(13)) / palm;
  const pRelY = (ly(20) - ly(17)) / palm;

  const iRelX = (lx(8) - lx(5)) / palm;

  const thumbRelX = (lx(4) - lx(2)) / palm;
  const thumbRelY = (ly(4) - ly(2)) / palm;

  const iDirX  = lx(8) - lx(5);
  const iDirY  = ly(8) - ly(5);
  const iDirMag = Math.sqrt(iDirX*iDirX + iDirY*iDirY) || 0.0001;
  const iDirYn  = iDirY / iDirMag;

  const mDirY   = ly(12) - ly(9);
  const mDirX   = lx(12) - lx(9);
  const mDirMag = Math.sqrt(mDirX*mDirX + mDirY*mDirY) || 0.0001;
  const mDirYn  = mDirY / mDirMag;

  if (iExt && mExt && rExt && pExt && thumbRelX > 0.8) return "SPACE";
  if (iCrl && mCrl && rCrl && pCrl && thumbRelY > 0.5) return "DELETE";
  if (iExt && mExt && rExt && pExt && iRelY < -0.5 && mRelY < -0.5 && Math.abs(thumbRelX) < 0.35) return "B";
  if (iExt && mExt && rExt && !pExt && imGap > 0.15 && mrGap > 0.15) return "W";

  if (iExt && mExt && !rExt && !pExt) {
    if (iRelX > 0.8 && Math.abs(iDirYn) < 0.4) return "H";
    if (iRelY < -0.5 && mRelY < -0.5 && imGap < 0.13) return "R";
    if (iRelY < -0.5 && mRelY < -0.5 && imGap >= 0.13 && imGap < 0.30) return "U";
    if (iRelY < -0.5 && mRelY < -0.5 && imGap >= 0.30) return "V";
    if (mRelY < -0.5 && iRelX > 0.3 && imGap > 0.30 && thumbRelY < -0.35) return "K";
    if (iRelX > 0.5 && mRelY > 0.5) return "P";
  }

  if (iExt && mCrl && rCrl && iRelY < -0.7 && thumbRelX > 0.35) return "L";
  if (iExt && !mExt && iRelY < -0.6 && imGap > 0.7 && t_m < 0.30 && t_r < 0.25) return "D";
  if (iExt && mCrl && rCrl && pCrl && iRelX > 0.7 && Math.abs(iRelY) < 0.25) return "G";
  if (iExt && !mExt && iRelX > 0.5 && iRelY > -0.2 && mRelY > 0.5) return "Q";
  if (iMid && mCrl && rCrl && pCrl && iRelY < -0.3 && imGap > 0.5) return "X";
  if (iMid && mMid && !iExt && !mExt && !rExt && !pExt && iRelX > 0.3 && imGap < 0.12 && iRelY < -0.1 && iRelY > -0.5) return "C";
  if (!iExt && mExt && rExt && pExt && t_i < 0.15 && mRelY < -0.5) return "F";
  if (!iExt && !mExt && !rExt && pExt && pRelY < -0.5 && thumbRelY < -0.4) return "Y";
  if (!iExt && !mExt && !rExt && pExt && pRelY < -0.5) return "I";

  if (!iExt && !mExt && !rExt && !pExt) {
    if (t_i < 0.22 && t_m < 0.22 && t_r < 0.22 && imGap < 0.10) return "O";
    if (IC < 115 && MC < 115 && RC < 115 && PC < 115 && t_i < 0.20 && t_m < 0.20 && Math.abs(iRelY) < 0.15) return "E";
    if (IC < 120 && MC < 120 && thumbRelY < -0.65 && t_i < 0.60 && Math.abs(iRelY) < 0.20) return "T";
    if (IC < 120 && MC < 120 && RC < 120 && PC < 120 && iRelY > 0.10 && mRelY > 0.10 && t_m < 0.70 && t_r < 0.55 && t_p > 0.50) return "M";
    if (IC < 120 && MC < 120 && RC < 120 && PC < 120 && iRelY > 0.05 && mRelY > 0.05 && t_m < 0.55 && t_r > 0.60) return "N";
    if (IC < 115 && MC < 115 && RC < 115 && PC < 115 && t_i < 0.35 && thumbRelX > -0.40 && thumbRelX < 0.15) return "S";
    if (IC < 120 && MC < 120 && RC < 120 && PC < 120 && thumbRelX > 0.10 && t_i > 0.45) return "A";
  }

  return "NOTHING";
}

init().catch((err) => {
  console.error(err);
  letterDiv.textContent = "Error loading model.";
});

const toggle = document.getElementById("mobile-menu");
const menu = document.querySelector(".navbar_menu");
toggle.addEventListener("click", () => {
  menu.classList.toggle("active");
  toggle.classList.toggle("active");
});

const buttons = document.querySelectorAll(".button-neon");
buttons.forEach(btn => {
  const duration = 3.5 + Math.random() * 1.5;
  btn.style.setProperty("--pulse-speed", `${duration}s`);
  function randomize() {
    const hue = Math.floor(Math.random() * 260 + 100);
    btn.style.setProperty("--hue", hue);
    setTimeout(randomize, 50 + Math.random() * 200);
  }
  setTimeout(randomize, Math.random() * 500);
});

clearBtn.addEventListener("click", () => {
  sentenceChars = [];
  renderSentence();
});