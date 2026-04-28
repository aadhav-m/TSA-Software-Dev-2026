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

// ─────────────────────────────────────────────────────────────────────────────
// CLASSIFIER — all thresholds derived from real captured debug data
// ─────────────────────────────────────────────────────────────────────────────
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

  // From real data: extended = 165–179, curled = 88–115, mid = 120–150
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

  // Mid-bent: between CRL and EXT
  const iMid = !iExt && !iCrl;
  const mMid = !mExt && !mCrl;

  // Palm-normalised tip distances
  const t_i = n3(4, 8);
  const t_m = n3(4, 12);
  const t_r = n3(4, 16);
  const t_p = n3(4, 20);
  const i_m = n3(8, 12);

  // 2D tip spread
  const imGap = n2(8, 12);
  const mrGap = n2(12, 16);

  // Tip Y relative to MCP, palm-normalised. Negative = tip above base = pointing up.
  // From data: extended upward = -0.69 to -1.00 | drooping/curled = +0.10 to +0.40
  const iRelY = (ly(8)  - ly(5))  / palm;
  const mRelY = (ly(12) - ly(9))  / palm;
  const rRelY = (ly(16) - ly(13)) / palm;
  const pRelY = (ly(20) - ly(17)) / palm;

  // Tip X relative to MCP, palm-normalised. Positive = tip to the right of base.
  const iRelX = (lx(8) - lx(5)) / palm;

  // Thumb position relative to landmark 2 (thumb base), palm-normalised
  const thumbRelX = (lx(4) - lx(2)) / palm;
  const thumbRelY = (ly(4) - ly(2)) / palm;

  // Index finger direction vector (MCP→tip), unit-normalised
  const iDirX  = lx(8) - lx(5);
  const iDirY  = ly(8) - ly(5);
  const iDirMag = Math.sqrt(iDirX*iDirX + iDirY*iDirY) || 0.0001;
  const iDirYn  = iDirY / iDirMag;

  // Middle finger direction Y component, normalised
  const mDirY   = ly(12) - ly(9);
  const mDirX   = lx(12) - lx(9);
  const mDirMag = Math.sqrt(mDirX*mDirX + mDirY*mDirY) || 0.0001;
  const mDirYn  = mDirY / mDirMag;

  // ── SPACE ─────────────────────────────────────────────────────────────────
  // All fingers extended + thumb clearly out to the side (open hand)
  if (iExt && mExt && rExt && pExt && thumbRelX > 0.8) return "SPACE";

  // ── DELETE ────────────────────────────────────────────────────────────────
  // All fingers curled, thumb pointing downward
  if (iCrl && mCrl && rCrl && pCrl && thumbRelY > 0.5) return "DELETE";

  // ── B ─────────────────────────────────────────────────────────────────────
  // All 4 fingers fully extended upward, thumb tucked (not out to side)
  // Data: IC=171, MC=173, RC=174, PC=179, iRelY=-0.69, thumbRelX=-0.06
  if (iExt && mExt && rExt && pExt
      && iRelY < -0.5 && mRelY < -0.5
      && Math.abs(thumbRelX) < 0.35) return "B";

  // ── W ─────────────────────────────────────────────────────────────────────
  // Index + middle + ring extended with spread, pinky curled
  if (iExt && mExt && rExt && !pExt && imGap > 0.15 && mrGap > 0.15) return "W";

  // ── Two fingers up: U / R / V / K / H / P ────────────────────────────────
  if (iExt && mExt && !rExt && !pExt) {

    // H: index pointing strongly sideways (horizontal)
    // Data: H iRelX=1.28, iDirYn=0.12 — clearly horizontal, both fingers sideways
    if (iRelX > 0.8 && Math.abs(iDirYn) < 0.4) return "H";

    // R: both fingers pointing straight up, very close together (crossed)
    // Data: R iRelY=-0.87, mRelY=-0.91, imGap=0.108
    // U has imGap=0.196, so 0.13 is a clean dividing line
    if (iRelY < -0.5 && mRelY < -0.5 && imGap < 0.13) return "R";

    // U: both fingers up, together but not crossed
    // Data: U iRelY=-0.78, mRelY=-0.91, imGap=0.196
    if (iRelY < -0.5 && mRelY < -0.5 && imGap >= 0.13 && imGap < 0.30) return "U";

    // V: both fingers up, spread apart
    // Data: V iRelY=-0.87, mRelY=-1.00, imGap=0.386
    if (iRelY < -0.5 && mRelY < -0.5 && imGap >= 0.30) return "V";

    // K: index diagonal, middle up, thumb up — imGap large, iRelX noticeable
    // Data: K iRelY=-0.46, mRelY=-0.85, iRelX=0.52, imGap=0.468, thumbRelY=-0.50
    if (mRelY < -0.5 && iRelX > 0.3 && imGap > 0.30 && thumbRelY < -0.35) return "K";

    // P: index sideways, middle drooping downward
    // Data: P iRelY=-0.09, mRelY=+0.83, iRelX=0.71
    if (iRelX > 0.5 && mRelY > 0.5) return "P";
  }

  // ── L ─────────────────────────────────────────────────────────────────────
  // Index strongly up, rest curled, thumb out to the right
  // Data: L iRelY=-1.00, MC=84 (crl), thumbRelX=+0.51
  if (iExt && mCrl && rCrl && iRelY < -0.7 && thumbRelX > 0.35) return "L";

  // ── D ─────────────────────────────────────────────────────────────────────
  // Index fully up, middle/ring make a circle with thumb (but not curled tight)
  // Data: D IC=175, MC=124 (mid), RC=125 (mid), imGap=0.976 (huge), t_m=0.252, t_r=0.175
  // The "O" shape of middle+ring+thumb is the key: t_m and t_r small while index is fully up
  if (iExt && !mExt && iRelY < -0.6 && imGap > 0.7 && t_m < 0.30 && t_r < 0.25) return "D";

  // ── G ─────────────────────────────────────────────────────────────────────
  // Index pointing sideways, all others curled
  // Data: G IC=176 (ext), mCrl (MC=101), iRelX=0.92, iRelY=0.036 (flat/horizontal)
  if (iExt && mCrl && rCrl && pCrl && iRelX > 0.7 && Math.abs(iRelY) < 0.25) return "G";

  // ── Q ─────────────────────────────────────────────────────────────────────
  // Index pointing sideways but not fully extended (IC=165), middle drooping
  // Data: Q IC=165, iRelX=0.69, iRelY=-0.057, mRelY=+0.76
  if (iExt && !mExt && iRelX > 0.5 && iRelY > -0.2 && mRelY > 0.5) return "Q";

  // ── X ─────────────────────────────────────────────────────────────────────
  // Index hooked (mid-bent), others curled, tip still above MCP level
  // Data: X IC=132 (mid), MC=99 (crl), iRelY=-0.56, imGap=0.892
  if (iMid && mCrl && rCrl && pCrl && iRelY < -0.3 && imGap > 0.5) return "X";

  // ── C ─────────────────────────────────────────────────────────────────────
  // All fingers curved into C — all mid-bent, hand angled sideways, tips close
  // Data: C IC=134, MC=120, RC=123, PC=138 — all mid-range
  // iRelX=0.45, imGap=0.059, iRelY=-0.26
  if (iMid && mMid && !iExt && !mExt && !rExt && !pExt
      && iRelX > 0.3 && imGap < 0.12 && iRelY < -0.1 && iRelY > -0.5) return "C";

  // ── F ─────────────────────────────────────────────────────────────────────
  // Middle/ring/pinky extended up, index tip touches thumb tip
  // Data: F MC=172, RC=175, PC=178, t_i=0.098, mRelY=-0.82
  if (!iExt && mExt && rExt && pExt && t_i < 0.15 && mRelY < -0.5) return "F";

  // ── Y ─────────────────────────────────────────────────────────────────────
  // Only pinky extended, thumb out — index/middle/ring curled
  // Data: Y PC=172, IC=92, MC=92, RC=94, pRelY=-0.90, thumbRelY=-0.76
  if (!iExt && !mExt && !rExt && pExt && pRelY < -0.5 && thumbRelY < -0.4) return "Y";

  // ── I ─────────────────────────────────────────────────────────────────────
  // Only pinky extended, thumb neutral
  if (!iExt && !mExt && !rExt && pExt && pRelY < -0.5) return "I";

  // ── All-curled group: O / E / T / M / N / S / A ──────────────────────────
  if (!iExt && !mExt && !rExt && !pExt) {

    // O: all fingertips forming a circle close to thumb — all t_x small, imGap tiny
    // Data: O t_i=0.167, t_m=0.149, t_r=0.146, t_p=0.187, imGap=0.035
    if (t_i < 0.22 && t_m < 0.22 && t_r < 0.22 && imGap < 0.10) return "O";

    // E: all fingers curled, tips hovering near (but not quite touching) thumb
    // Data: E IC=98, MC=94, RC=90, PC=92, t_i=0.146, t_m=0.157, iRelY≈0
    // All curls ~90–100, tips near thumb, fingers roughly level with knuckles
    if (IC < 115 && MC < 115 && RC < 115 && PC < 115
        && t_i < 0.20 && t_m < 0.20 && Math.abs(iRelY) < 0.15) return "E";

    // T: thumb pushes up between index and middle, index hooks forward
    // Data: T IC=107, thumbRelY=-0.83 (thumb strongly UP), t_i=0.490, iRelY=-0.06
    // Key signature: thumbRelY very negative (thumb pointing up) in a curled fist
    if (IC < 120 && MC < 120 && thumbRelY < -0.65 && t_i < 0.60 && Math.abs(iRelY) < 0.20) return "T";

    // M: thumb tucked under index+middle+ring, 3 fingers droop over it
    // Data: M IC=107, MC=103, RC=97, PC=111, iRelY=+0.25, mRelY=+0.27
    // t_m=0.598, t_r=0.432 — thumb is reaching across under middle/ring
    if (IC < 120 && MC < 120 && RC < 120 && PC < 120
        && iRelY > 0.10 && mRelY > 0.10
        && t_m < 0.70 && t_r < 0.55 && t_p > 0.50) return "M";

    // N: thumb under index+middle only, ring/pinky not covered
    // Data: N IC=104, MC=97, t_m=0.468, t_r=0.782 (thumb doesn't reach ring)
    if (IC < 120 && MC < 120 && RC < 120 && PC < 120
        && iRelY > 0.05 && mRelY > 0.05
        && t_m < 0.55 && t_r > 0.60) return "N";

    // S: closed fist, thumb wraps across front of fingers
    // Data: S IC=101, MC=94, RC=89, PC=90, t_i=0.281, thumbRelX=-0.22
    // Thumb is centered/slightly left — not to the right like A
    if (IC < 115 && MC < 115 && RC < 115 && PC < 115
        && t_i < 0.35 && thumbRelX > -0.40 && thumbRelX < 0.15) return "S";

    // A: fist with thumb resting to the side (not over fingers)
    // Data: A IC=101, MC=99, RC=97, PC=97, thumbRelX=+0.189, t_i=0.600
    // Thumb clearly to the right side, not touching index tip
    if (IC < 120 && MC < 120 && RC < 120 && PC < 120
        && thumbRelX > 0.10 && t_i > 0.45) return "A";
  }

  return "NOTHING";
}

init().catch((err) => {
  console.error(err);
  letterDiv.textContent = "Error loading model. Check console.";
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
