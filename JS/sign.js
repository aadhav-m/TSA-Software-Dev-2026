import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video       = document.getElementById("video");
const startBtn    = document.getElementById("startButton");
const stopBtn     = document.getElementById("stopButton");
const letterDiv   = document.getElementById("translation-letter");
const sentenceDiv = document.getElementById("translation-sentence");

const HOLD_MS          = 800;
const REPEAT_DELAY_MS  = 1800;
const SMOOTH_WINDOW    = 6;
const NOTHING_FRAMES   = 4;
const MOTION_FRAMES    = 8;
const MOTION_T         = 0.014;

let handLandmarker = null;
let stream         = null;
let rafId          = null;
let running        = false;
let lastVideoTime  = -1;

let sentenceChars  = [];
let lastLetter     = "";
let holdStart      = 0;
let lastCommitTime = 0;
let lastCommitChar = "";
let nothingCount   = 0;

const predictionBuffer = [];
const jHistory         = [];
const zHistory         = [];
let prevPinky          = null;
let prevIndex          = null;

function renderSentence() {
  sentenceDiv.textContent = sentenceChars.join("") || "\u00a0";
}

async function init() {
  letterDiv.textContent = "Loading model…";

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
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

  running = true;
  startBtn.disabled = true;
  stopBtn.disabled  = false;
  letterDiv.textContent = "Show a sign…";
  rafId = requestAnimationFrame(detect);
}

function stopRecording() {
  running = false;
  cancelAnimationFrame(rafId);
  stream?.getTracks().forEach((t) => t.stop());
  video.srcObject   = null;
  startBtn.disabled = false;
  stopBtn.disabled  = true;
  letterDiv.textContent   = "Stopped";
  predictionBuffer.length = 0;
  jHistory.length         = 0;
  zHistory.length         = 0;
  prevPinky    = null;
  prevIndex    = null;
  lastLetter   = "";
  holdStart    = 0;
  nothingCount = 0;
}

function detect(now) {
  if (!running) return;

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const results = handLandmarker.detectForVideo(video, now);

    if (results.landmarks && results.landmarks.length > 0) {
      nothingCount = 0;
      processLandmarks(results.landmarks[0], now);
    } else {
      nothingCount++;
      if (nothingCount >= NOTHING_FRAMES) {
        predictionBuffer.length = 0;
        jHistory.length         = 0;
        zHistory.length         = 0;
        prevPinky    = null;
        prevIndex    = null;
        lastLetter   = "";
        holdStart    = 0;
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
  return best;
}

function trackMotion(lm) {
  const px = lm[20].x, py = lm[20].y;
  const ix = lm[8].x,  iy = lm[8].y;

  const pdx = prevPinky ? px - prevPinky.x : 0;
  const pdy = prevPinky ? py - prevPinky.y : 0;
  const idx = prevIndex ? ix - prevIndex.x : 0;
  const idy = prevIndex ? iy - prevIndex.y : 0;

  prevPinky = { x: px, y: py };
  prevIndex = { x: ix, y: iy };

  jHistory.push({ dx: pdx, dy: pdy });
  if (jHistory.length > MOTION_FRAMES) jHistory.shift();

  zHistory.push({ dx: idx, dy: idy });
  if (zHistory.length > MOTION_FRAMES) zHistory.shift();
}

function detectJ() {
  if (jHistory.length < MOTION_FRAMES) return false;
  const sumDy    = jHistory.reduce((s, f) => s + f.dy, 0);
  const sumAbsDx = jHistory.reduce((s, f) => s + Math.abs(f.dx), 0);
  return sumDy > MOTION_T * 4 && sumAbsDx > MOTION_T * 2;
}

function detectZ() {
  if (zHistory.length < MOTION_FRAMES) return false;
  const h    = Math.floor(zHistory.length / 2);
  const dx1  = zHistory.slice(0, h).reduce((s, f) => s + f.dx, 0);
  const dy1  = zHistory.slice(0, h).reduce((s, f) => s + f.dy, 0);
  const dx2  = zHistory.slice(h).reduce((s, f) => s + f.dx, 0);
  return dx1 > MOTION_T * 2 && dy1 > MOTION_T && dx2 > MOTION_T * 1.5;
}

function processLandmarks(lm, now) {
  trackMotion(lm);

  const raw    = classify(lm);
  const letter = smoothedPrediction(raw);

  if (letter === "NOTHING") {
    letterDiv.textContent = "—";
    lastLetter = "";
    holdStart  = 0;
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

  if (held && (!sameChar || cooldown)) {
    commitOnce(letter, now);
  }
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
    return Math.acos(Math.max(-1,Math.min(1,cos)))*180/Math.PI;
  }

  const curl = (a,b,c,d) => (ang(a,b,c)+ang(b,c,d))/2;

  const IC = curl(5,6,7,8);
  const MC = curl(9,10,11,12);
  const RC = curl(13,14,15,16);
  const PC = curl(17,18,19,20);

  const S = 155;
  const H = 115;
  const B = 85;

  const iStr = IC > S;
  const mStr = MC > S;
  const rStr = RC > S;
  const pStr = PC > S;

  const iHalf = IC >= H && IC < S;
  const mHalf = MC >= H && MC < S;
  const rHalf = RC >= H && RC < S;
  const pHalf = PC >= H && PC < S;

  const iCurl = IC < B;
  const mCurl = MC < B;
  const rCurl = RC < B;
  const pCurl = PC < B;

  const allStr  = iStr && mStr && rStr && pStr;
  const allCurl = iCurl && mCurl && rCurl && pCurl;

  const isRight = lx(17) < lx(5);

  const thumbOutX = isRight
    ? lx(4) < lx(2) - 0.04
    : lx(4) > lx(2) + 0.04;

  const thumbUpY   = ly(4) < ly(2) - 0.06;
  const thumbDownY = ly(4) > ly(3) + 0.05;

  const thumbOverFront = n2(4, 9) < 0.38 && !thumbOutX;
  const thumbAcrossPalm = n2(4, 17) < 0.55 && !thumbOutX && !thumbUpY;

  const tipTouch = (a,b) => n3(a,b) < 0.13;
  const tipNear  = (a,b) => n3(a,b) < 0.20;
  const tipMid   = (a,b) => n3(a,b) < 0.28;

  const imGap = n2(8,12);
  const mrGap = n2(12,16);
  const rpGap = n2(16,20);

  const i8y   = ly(8);
  const i5y   = ly(5);
  const m12y  = ly(12);
  const m9y   = ly(9);

  const indexUp    = i8y < i5y - 0.04;
  const indexDown  = i8y > i5y + 0.05;
  const indexSide  = Math.abs(i8y - i5y) < 0.07;

  const imHoriz = Math.abs(i8y - m12y) < 0.07 && Math.abs(i5y - m9y) < 0.07;
  const imTog   = imGap < 0.09;
  const imSprd  = imGap > 0.18;

  const tipAvgY = (ly(8)+ly(12)+ly(16)+ly(20))/4;

  if (allStr && thumbOutX)                                        return "SPACE";
  if (allCurl && thumbDownY && !thumbOutX)                        return "DELETE";

  if (allStr && thumbAcrossPalm && !thumbOutX && tipAvgY < ly(0)) return "B";

  if (iStr && mStr && rStr && !pStr && imGap > 0.12 && mrGap > 0.12) return "W";

  if (iStr && mStr && !rStr && !pStr) {
    if (imTog && imHoriz)                                         return "H";
    if (n3(8,12) < 0.07)                                         return "R";
    if (!imTog && thumbUpY && n2(4,6) < 0.35)                    return "K";
    if (imTog && !imHoriz)                                        return "U";
    if (imSprd && !thumbUpY && !indexDown)                        return "V";
    if (imSprd && indexDown && thumbUpY)                          return "P";
  }

  if (iStr && !mStr && !rStr && !pStr) {
    if (thumbUpY && thumbOutX && indexUp)                         return "L";
    if (indexSide && !thumbOutX && mCurl && rCurl && pCurl)       return "G";
    if (indexDown && !thumbOutX && mCurl && rCurl && pCurl)       return "Q";
    if (indexUp && tipNear(4,12) && mCurl && rCurl && !thumbOutX) return "D";
    if (iHalf || (IC >= B && IC < S && !indexSide && !indexDown)) return "X";
  }

  if (iHalf && !mStr && !rStr && !pStr && mCurl && rCurl && pCurl) return "X";

  if (iHalf && mHalf && rHalf && pHalf) {
    const thumbCurl = ang(1,2,3);
    if (n2(4,8) > 0.20)                                          return "C";
  }

  if (!iStr && !mStr && !rStr && pStr) {
    if (thumbOutX || thumbUpY)                                    return "I";
    return "Y";
  }

  if (!iStr && mStr && rStr && pStr && tipTouch(4,8) && iCurl)    return "F";

  if (!iStr && !mStr && !rStr && !pStr) {
    if (tipTouch(4,8) && tipTouch(4,12) && tipTouch(4,16))       return "O";

    if (allCurl) {
      if (tipNear(4,8) && tipNear(4,12) && ly(8)>ly(6) && ly(12)>ly(10)) return "E";

      if (ly(8)>ly(5) && ly(12)>ly(9) && ly(16)>ly(13)) {
        if (tipMid(4,16) && tipMid(4,8))                         return "M";
        if (tipMid(4,12) && !tipMid(4,16))                       return "N";
      }

      if (!thumbOutX && !thumbDownY) {
        if (n2(4,6) < 0.22 && n2(4,10) < 0.28 && ly(4) > ly(5)) return "T";
        if (ly(4) < ly(8) && n2(4,8) < 0.28 && !thumbUpY)        return "S";
        if (thumbOutX)                                            return "A";
      }

      if (thumbOutX && !thumbDownY && !thumbUpY)                  return "A";
    }
  }

  return "NOTHING";
}

init().catch((err) => {
  console.error(err);
  letterDiv.textContent = "Error loading model. Check console.";
});

const toggle = document.getElementById('mobile-menu');
const menu = document.querySelector('.navbar_menu');

toggle.addEventListener('click', () => {
  menu.classList.toggle('active');
  toggle.classList.toggle('active');
});

const buttons = document.querySelectorAll('.button-neon');

buttons.forEach(btn => {
  const duration = 3.5 + Math.random() * 1.5;
  btn.style.setProperty('--pulse-speed', `${duration}s`);
  function randomize() {
    const hue = Math.floor(Math.random() * 260 + 100);
    btn.style.setProperty('--hue', hue);
    const nextDelay = 50 + Math.random() * 200;
    setTimeout(randomize, nextDelay);
  }
  setTimeout(randomize, Math.random() * 500);
});

const clear = document.getElementById("clearTranslation");
clear.addEventListener("click", () => {
  sentenceChars = [];
  renderSentence();
});