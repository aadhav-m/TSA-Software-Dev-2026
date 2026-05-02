import {
  FilesetResolver,
  HandLandmarker,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const ALL_LETTERS       = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const SAMPLES_PER_LETTER = 5;
const JPEG_QUALITY       = 0.85;

const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

const els = {};
let landmarker = null;
let stream = null;
let lastResult = null;
let lastVideoTime = -1;
let detectedHandedness = null;

let letterIdx = 0;
let sampleIdx = 0;
const captured = {};
for (const L of ALL_LETTERS) captured[L] = [];

const snapCanvas = document.createElement('canvas');
const snapCtx    = snapCanvas.getContext('2d', { willReadFrequently: false });

init().catch((err) => {
  console.error('[captureTraining] init failed:', err);
  const s = document.getElementById('cap-status');
  if (s) s.textContent = 'Init failed: ' + err.message + ' — see console.';
});

async function init() {
  for (const id of [
    'cap-video', 'cap-canvas', 'cap-status', 'cap-letter', 'cap-grid',
    'cap-start', 'cap-capture', 'cap-retry', 'cap-skip', 'cap-download',
    'train-sample', 'train-total',
    'mobile-menu',
  ]) els[id] = document.getElementById(id);

  els._ctx = els['cap-canvas'].getContext('2d');
  els['train-total'].textContent = SAMPLES_PER_LETTER;

  buildGrid();
  updateUI();

  els['cap-status'].textContent = 'Loading MediaPipe…';
  try {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );
    landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 1,
    });
  } catch (err) {
    els['cap-status'].textContent = 'Failed to load MediaPipe: ' + err.message;
    return;
  }
  els['cap-status'].textContent = 'Click Start Camera, then sign each letter and click Capture.';

  els['cap-start'].addEventListener('click', startCam);
  els['cap-capture'].addEventListener('click', captureFrame);
  els['cap-retry'].addEventListener('click', retry);
  els['cap-skip'].addEventListener('click', skipLetter);
  els['cap-download'].addEventListener('click', download);

  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input, textarea')) return;
    if (e.code === 'Space') { e.preventDefault(); captureFrame(); }
    else if (e.code === 'Backspace') { e.preventDefault(); retry(); }
  });

  const toggle = els['mobile-menu'];
  const menu = document.querySelector('.navbar_menu');
  if (toggle && menu) {
    toggle.addEventListener('click', () => {
      menu.classList.toggle('active');
      toggle.classList.toggle('active');
    });
  }
  document.querySelectorAll('.button-neon').forEach((btn) => {
    btn.style.setProperty('--pulse-speed', `${3.5 + Math.random() * 1.5}s`);
    (function spin() {
      btn.style.setProperty('--hue', Math.floor(Math.random() * 260 + 100));
      setTimeout(spin, 50 + Math.random() * 200);
    })();
  });
}

async function startCam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false,
    });
  } catch (err) {
    els['cap-status'].textContent = 'Camera denied: ' + err.message;
    return;
  }
  els['cap-video'].srcObject = stream;
  await els['cap-video'].play();
  els['cap-start'].disabled = true;
  loop();
}

function loop() {
  const v = els['cap-video'];
  if (v.readyState >= 2 && v.currentTime !== lastVideoTime) {
    lastVideoTime = v.currentTime;
    lastResult = landmarker.detectForVideo(v, performance.now());
    drawOverlay(lastResult);
  }
  requestAnimationFrame(loop);
}

function drawOverlay(r) {
  const ctx = els._ctx;
  const c = els['cap-canvas'];
  ctx.clearRect(0, 0, c.width, c.height);
  if (!r || !r.landmarks || r.landmarks.length === 0) return;
  const lm = r.landmarks[0];
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = '#E8845A';
  ctx.fillStyle = '#C45020';
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(lm[a].x * c.width, lm[a].y * c.height);
    ctx.lineTo(lm[b].x * c.width, lm[b].y * c.height);
    ctx.stroke();
  }
  for (const p of lm) {
    ctx.beginPath();
    ctx.arc(p.x * c.width, p.y * c.height, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function snapshotFrame() {
  const v = els['cap-video'];
  const w = v.videoWidth || 640;
  const h = v.videoHeight || 480;
  if (snapCanvas.width !== w)  snapCanvas.width  = w;
  if (snapCanvas.height !== h) snapCanvas.height = h;
  snapCtx.drawImage(v, 0, 0, w, h);
  return snapCanvas.toDataURL('image/jpeg', JPEG_QUALITY);
}

function captureFrame() {
  if (letterIdx >= ALL_LETTERS.length) return;
  if (!lastResult || !lastResult.landmarks || lastResult.landmarks.length === 0) {
    els['cap-status'].textContent = 'No hand visible — try again.';
    return;
  }
  const letter = ALL_LETTERS[letterIdx];
  const lm     = lastResult.landmarks[0];
  const wlm    = lastResult.worldLandmarks?.[0] || lm;
  const handed = lastResult.handedness?.[0]?.[0]?.categoryName ?? 'Right';
  if (!detectedHandedness) detectedHandedness = handed;

  const sample = {
    letter,
    sampleIndex: sampleIdx,
    handedness: handed,
    landmarks:      lm.map((p) => ({ x: p.x, y: p.y, z: p.z })),
    worldLandmarks: wlm.map((p) => ({ x: p.x, y: p.y, z: p.z })),
    image: snapshotFrame(),
    capturedAt: new Date().toISOString(),
  };
  captured[letter].push(sample);
  sampleIdx++;
  if (sampleIdx >= SAMPLES_PER_LETTER) {
    sampleIdx = 0;
    letterIdx++;
  }
  updateUI();
}

function retry() {
  if (sampleIdx > 0) {
    captured[ALL_LETTERS[letterIdx]].pop();
    sampleIdx--;
  } else if (letterIdx > 0) {
    letterIdx--;
    captured[ALL_LETTERS[letterIdx]].pop();
    sampleIdx = captured[ALL_LETTERS[letterIdx]].length;
    if (sampleIdx >= SAMPLES_PER_LETTER) sampleIdx = SAMPLES_PER_LETTER - 1;
  }
  updateUI();
}

function skipLetter() {
  if (letterIdx >= ALL_LETTERS.length) return;
  sampleIdx = 0;
  letterIdx++;
  updateUI();
}

function buildGrid() {
  els['cap-grid'].innerHTML = '';
  for (const L of ALL_LETTERS) {
    const cell = document.createElement('div');
    cell.className = 'cap-cell';
    cell.dataset.letter = L;
    cell.textContent = L;
    const prog = document.createElement('span');
    prog.className = 'train-progress';
    prog.textContent = '0/' + SAMPLES_PER_LETTER;
    cell.appendChild(prog);
    els['cap-grid'].appendChild(cell);
  }
}

function refreshGrid() {
  for (const cell of els['cap-grid'].children) {
    const L = cell.dataset.letter;
    const n = captured[L].length;
    const isFull   = n >= SAMPLES_PER_LETTER;
    const isPart   = n > 0 && !isFull;
    const isActive = ALL_LETTERS[letterIdx] === L;
    cell.classList.toggle('done',    isFull);
    cell.classList.toggle('partial', isPart);
    cell.classList.toggle('active',  isActive);
    cell.querySelector('.train-progress').textContent = `${n}/${SAMPLES_PER_LETTER}`;
  }
}

function totalCaptured() {
  let n = 0;
  for (const L of ALL_LETTERS) n += captured[L].length;
  return n;
}

function updateUI() {
  refreshGrid();
  const total = totalCaptured();
  const targetTotal = ALL_LETTERS.length * SAMPLES_PER_LETTER;

  if (letterIdx >= ALL_LETTERS.length) {
    els['cap-letter'].textContent = '✓';
    els['train-sample'].textContent = String(SAMPLES_PER_LETTER);
    els['cap-status'].textContent = `Done. Captured ${total}/${targetTotal} total samples. Click Download.`;
    els['cap-capture'].disabled = true;
    els['cap-skip'].disabled    = true;
    els['cap-download'].disabled = total === 0;
  } else {
    const L = ALL_LETTERS[letterIdx];
    els['cap-letter'].textContent = L;
    els['train-sample'].textContent = String(sampleIdx + 1);
    const handTag = detectedHandedness ? ` (hand: ${detectedHandedness})` : '';
    els['cap-status'].textContent =
      `Sign "${L}" — sample ${sampleIdx + 1}/${SAMPLES_PER_LETTER}${handTag}. ` +
      `[Space = Capture, Backspace = Re-do]  Total: ${total}/${targetTotal}.`;
    els['cap-capture'].disabled  = false;
    els['cap-skip'].disabled     = false;
    els['cap-download'].disabled = total === 0;
  }
}

function download() {
  const out = {
    version: 1,
    capturedAt: new Date().toISOString(),
    handedness: detectedHandedness ?? 'Right',
    samplesPerLetter: SAMPLES_PER_LETTER,
    imageFormat: 'image/jpeg',
    imageQuality: JPEG_QUALITY,
    imageDimensions: {
      width:  snapCanvas.width  || 640,
      height: snapCanvas.height || 480,
    },
    letters: captured,
  };
  const blob = new Blob([JSON.stringify(out)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const ts = new Date().toISOString().replace(/[:.]/g, '-').replace(/T/, '_').slice(0, 19);
  a.download = `asl_training_${ts}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
