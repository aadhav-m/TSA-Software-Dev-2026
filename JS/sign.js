import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video       = document.getElementById("video");
const startBtn    = document.getElementById("startButton");
const stopBtn     = document.getElementById("stopButton");
const letterDiv   = document.getElementById("translation-letter");
const sentenceDiv = document.getElementById("translation-sentence");
const clearBtn    = document.getElementById("clearTranslation");

const HOLD_MS         = 800;
const REPEAT_DELAY_MS = 1800;
const SMOOTH_WINDOW   = 7;
const NOTHING_FRAMES  = 4;
const MOTION_FRAMES   = 10;
const MOTION_T        = 0.012;

const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

const FINGER_COLORS = ["#ff6b6b","#ffa94d","#69db7c","#74c0fc","#da77f2"];
const FINGER_RANGES = [[1,4],[5,8],[9,12],[13,16],[17,20]];

let handLandmarker  = null;
let stream          = null;
let rafId           = null;
let running         = false;
let lastVideoTime   = -1;
let canvas          = null;
let ctx             = null;

let sentenceChars   = [];
let lastLetter      = "";
let holdStart       = 0;
let lastCommitTime  = 0;
let lastCommitChar  = "";
let nothingCount    = 0;

const predictionBuffer = [];
const jHistory         = [];
const zHistory         = [];
let prevPinky          = null;
let prevIndex          = null;

function renderSentence() {
  sentenceDiv.textContent = sentenceChars.join("") || "\u00a0";
}

function setupCanvas() {
  if (canvas) canvas.remove();
  canvas = document.createElement("canvas");
  canvas.style.position     = "absolute";
  canvas.style.pointerEvents = "none";
  canvas.style.zIndex       = "10";
  canvas.width              = 640;
  canvas.height             = 480;
  video.parentNode.insertBefore(canvas, video.nextSibling);
  ctx = canvas.getContext("2d");
  positionCanvas();
}

function positionCanvas() {
  if (!canvas) return;
  const r  = video.getBoundingClientRect();
  const pr = video.parentNode.getBoundingClientRect();
  canvas.style.top    = (r.top  - pr.top)  + "px";
  canvas.style.left   = (r.left - pr.left) + "px";
  canvas.style.width  = r.width  + "px";
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
    const fi    = getFingerForLandmark(a);
    const color = fi >= 0 ? FINGER_COLORS[fi] : "#ffffff";
    ctx.beginPath();
    ctx.moveTo(sx(lm[a].x), sy(lm[a].y));
    ctx.lineTo(sx(lm[b].x), sy(lm[b].y));
    ctx.strokeStyle  = color;
    ctx.lineWidth    = 2.5;
    ctx.shadowColor  = color;
    ctx.shadowBlur   = 8;
    ctx.stroke();
    ctx.shadowBlur   = 0;
  }

  for (let i = 0; i < lm.length; i++) {
    const fi    = getFingerForLandmark(i);
    const color = fi >= 0 ? FINGER_COLORS[fi] : "#ffffff";
    const isTip = [4,8,12,16,20].includes(i);
    const r     = isTip ? 6 : 4;
    ctx.beginPath();
    ctx.arc(sx(lm[i].x), sy(lm[i].y), r, 0, Math.PI * 2);
    ctx.fillStyle   = color;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 12;
    ctx.fill();
    ctx.shadowBlur  = 0;
    ctx.beginPath();
    ctx.arc(sx(lm[i].x), sy(lm[i].y), r * 0.45, 0, Math.PI * 2);
    ctx.fillStyle   = "#ffffff";
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

  letterDiv.textContent = "Press Start for Sign Language Translations";
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
  running           = true;
  startBtn.disabled = true;
  stopBtn.disabled  = false;
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
  stopBtn.disabled  = true;
  letterDiv.textContent = "Stopped";
  predictionBuffer.length = 0;
  jHistory.length         = 0;
  zHistory.length         = 0;
  prevPinky               = null;
  prevIndex               = null;
  lastLetter              = "";
  holdStart               = 0;
  nothingCount            = 0;
}

function detect(now) {
  if (!running) return;

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const results = handLandmarker.detectForVideo(video, now);

    if (results.landmarks && results.landmarks.length > 0) {
      nothingCount = 0;
      drawSkeleton(results.landmarks[0]);
      processLandmarks(results.landmarks[0], now);
    } else {
      nothingCount++;
      if (nothingCount >= NOTHING_FRAMES) {
        clearCanvas();
        predictionBuffer.length = 0;
        jHistory.length         = 0;
        zHistory.length         = 0;
        prevPinky               = null;
        prevIndex               = null;
        lastLetter              = "";
        holdStart               = 0;
        letterDiv.textContent   = "Show a sign…";
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
  return best;
}

function trackMotion(lm) {
  const px  = lm[20].x, py = lm[20].y;
  const ix  = lm[8].x,  iy = lm[8].y;
  const pdx = prevPinky ? px - prevPinky.x : 0;
  const pdy = prevPinky ? py - prevPinky.y : 0;
  const idx = prevIndex ? ix - prevIndex.x : 0;
  prevPinky = { x: px, y: py };
  prevIndex = { x: ix, y: iy };
  jHistory.push({ dx: pdx, dy: pdy });
  if (jHistory.length > MOTION_FRAMES) jHistory.shift();
  zHistory.push({ dx: idx });
  if (zHistory.length > MOTION_FRAMES) zHistory.shift();
}

function detectJ() {
  if (jHistory.length < MOTION_FRAMES) return false;
  const dy = jHistory.reduce((s, f) => s + f.dy, 0);
  const dx = jHistory.reduce((s, f) => s + Math.abs(f.dx), 0);
  return dy > MOTION_T * 4 && dx > MOTION_T * 2;
}

function detectZ() {
  if (zHistory.length < MOTION_FRAMES) return false;
  const h   = Math.floor(zHistory.length / 2);
  const dx1 = zHistory.slice(0, h).reduce((s, f) => s + f.dx, 0);
  const dx2 = zHistory.slice(h).reduce((s, f) => s + f.dx, 0);
  return Math.abs(dx1) > MOTION_T * 2 && Math.sign(dx1) !== Math.sign(dx2);
}

function processLandmarks(lm, now) {
  trackMotion(lm);
  const raw    = classify(lm);
  const letter = smoothedPrediction(raw);

  if (letter === "NOTHING") {
    letterDiv.textContent = "—";
    lastLetter            = "";
    holdStart             = 0;
    return;
  }

  letterDiv.textContent =
    letter === "SPACE"  ? "[ SPACE ]"  :
    letter === "DELETE" ? "[ DELETE ]" :
    letter;

  if (letter !== lastLetter) {
    lastLetter = letter;
    holdStart  = now;
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
  holdStart      = now;
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

  const dist3 = (a, b) => Math.sqrt(
    (lx(a)-lx(b))**2 + (ly(a)-ly(b))**2 + (lz(a)-lz(b))**2
  );
  const dist2 = (a, b) => Math.sqrt(
    (lx(a)-lx(b))**2 + (ly(a)-ly(b))**2
  );

  const palm   = dist3(0, 9) || 0.0001;
  const d3     = (a, b) => dist3(a, b) / palm;
  const d2     = (a, b) => dist2(a, b) / palm;

  function angle(a, b, c) {
    const ax = lx(a)-lx(b), ay = ly(a)-ly(b), az = lz(a)-lz(b);
    const cx = lx(c)-lx(b), cy = ly(c)-ly(b), cz = lz(c)-lz(b);
    const ma  = Math.sqrt(ax*ax+ay*ay+az*az);
    const mc  = Math.sqrt(cx*cx+cy*cy+cz*cz);
    const dot = (ax*cx+ay*cy+az*cz) / (ma*mc||0.0001);
    return Math.acos(Math.max(-1, Math.min(1, dot))) * 180 / Math.PI;
  }

  const fingerAngle = (a, b, c, d) => (angle(a,b,c) + angle(b,c,d)) / 2;

  const IC = fingerAngle(5,6,7,8);
  const MC = fingerAngle(9,10,11,12);
  const RC = fingerAngle(13,14,15,16);
  const PC = fingerAngle(17,18,19,20);

  const FULL = 160;
  const HALF = 115;
  const BENT = 90;

  const iStr  = IC > FULL;
  const mStr  = MC > FULL;
  const rStr  = RC > FULL;
  const pStr  = PC > FULL;

  const iHalf = IC > HALF && IC <= FULL;
  const mHalf = MC > HALF && MC <= FULL;

  const iBent = IC > BENT && IC <= HALF;

  const iCurl = IC <= BENT;
  const mCurl = MC <= BENT;
  const rCurl = RC <= BENT;
  const pCurl = PC <= BENT;

  const allStr  = iStr && mStr && rStr && pStr;
  const allCurl = iCurl && mCurl && rCurl && pCurl;

  const isRight = lx(17) < lx(5);

  const thumbTipX  = lx(4);
  const thumbTipY  = ly(4);
  const thumbTipZ  = lz(4);
  const thumbBaseX = lx(2);
  const thumbBaseY = ly(2);
  const thumbMidY  = ly(3);

  const thumbSideOut    = isRight
    ? thumbTipX < thumbBaseX - 0.045
    : thumbTipX > thumbBaseX + 0.045;
  const thumbUp         = thumbTipY < thumbBaseY - 0.06;
  const thumbDown       = thumbTipY > thumbMidY  + 0.05;
  const thumbForward    = thumbTipZ < lz(2) - 0.015;
  const thumbAcrossFist = d2(4,17) < 0.58 && !thumbSideOut;

  const tipTouching = (a, b) => d3(a,b) < 0.13;
  const tipNear     = (a, b) => d3(a,b) < 0.20;
  const tipClose    = (a, b) => d3(a,b) < 0.30;

  const imGap = d2(8,12);
  const mrGap = d2(12,16);
  const rpGap = d2(16,20);

  const tip8y  = ly(8);
  const tip12y = ly(12);
  const tip16y = ly(16);
  const tip20y = ly(20);
  const mcp5y  = ly(5);
  const mcp9y  = ly(9);
  const mcp13y = ly(13);
  const mcp17y = ly(17);

  const indexTipAboveMcp  = tip8y  < mcp5y  - 0.04;
  const indexTipBelowMcp  = tip8y  > mcp5y  + 0.04;
  const indexHorizontal   = Math.abs(tip8y - mcp5y) < 0.07;
  const middleTipBelowMcp = tip12y > mcp9y  + 0.04;

  const imTogether = imGap < 0.10;
  const imSpread   = imGap > 0.19;

  if (allStr && thumbSideOut) return "SPACE";
  if (allCurl && thumbDown && !thumbSideOut) return "DELETE";

  if (allStr && !thumbSideOut && !thumbUp && thumbAcrossFist) return "B";

  if (iStr && mStr && rStr && !pStr && pCurl && imGap > 0.12 && mrGap > 0.12) return "W";

  if (iStr && mStr && !rStr && !pStr && rCurl && pCurl) {
    if (d3(7,11) < 0.11) return "R";
    if (imSpread && indexTipAboveMcp) {
      if (thumbUp && tipNear(4,10)) return "K";
      return "V";
    }
    if (imSpread && indexTipBelowMcp && thumbUp) return "P";
    if (imTogether && indexHorizontal) return "H";
    if (imTogether) return "U";
  }

  if (iStr && !mStr && !rStr && !pStr && mCurl && rCurl && pCurl) {
    if (thumbSideOut && thumbUp && indexTipAboveMcp) return "L";
    if (indexHorizontal && !thumbSideOut) return "G";
    if (indexTipBelowMcp && !thumbSideOut) return "Q";
    if (indexTipAboveMcp && tipNear(4,12) && !thumbSideOut) return "D";
  }

  if (iBent && !mStr && !rStr && !pStr && mCurl && rCurl && pCurl) return "X";
  if (!iStr && mStr && rStr && pStr && iCurl && tipTouching(4,8)) return "F";

  if (!iStr && !mStr && !rStr && pStr && iCurl && mCurl && rCurl) {
    if (thumbSideOut || thumbUp) return "Y";
    return "I";
  }

  if (tipTouching(4,8) && tipTouching(4,12) && tipTouching(4,16)) return "O";

  if (!iStr && !mStr && !rStr && !pStr && !allCurl) {
    const midCurved = IC > BENT && IC < FULL && MC > BENT && MC < FULL;
    if (midCurved && !tipTouching(4,8) && d2(4,8) > 0.20) return "C";
  }

  if (!iStr && !mStr && !rStr && !pStr && allCurl) {
    if (tipNear(4,8) && tipNear(4,12) && tip8y < mcp5y) return "E";

    if (tip8y > mcp5y && tip12y > mcp9y && tip16y > mcp13y) {
      const i_over = tipClose(4,8);
      const m_over = tipClose(4,12);
      const r_over = tipClose(4,16);
      if (i_over && m_over && r_over) return "M";
      if (i_over && m_over && !r_over) return "N";
      if (i_over && !m_over) return "T";
    }

    if (thumbForward && !thumbSideOut && thumbAcrossFist) return "S";
    if (thumbSideOut && !thumbDown) return "A";
    if (!thumbSideOut && !thumbDown && !thumbForward) return "A";
  }

  return "NOTHING";
}

init().catch((err) => {
  console.error(err);
  letterDiv.textContent = "Error loading model. Check console.";
});

const toggle = document.getElementById("mobile-menu");
const menu   = document.querySelector(".navbar_menu");

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