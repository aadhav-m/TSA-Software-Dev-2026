import {
  FilesetResolver,
  HandLandmarker,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const ALL_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const STATIC_END_POSE_LETTERS = ['J', 'Z'];

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

let idx = 0;
const captured = {};
let detectedHandedness = null;

init().catch((err) => {
  console.error('[captureLandmarks] init failed:', err);
  const s = document.getElementById('cap-status');
  if (s) s.textContent = 'Init failed: ' + err.message + ' — see console.';
});

async function init() {
  for (const id of [
    'cap-video', 'cap-canvas', 'cap-status', 'cap-letter', 'cap-grid',
    'cap-start', 'cap-capture', 'cap-retry', 'cap-download',
    'mobile-menu',
  ]) els[id] = document.getElementById(id);

  const ctx = els['cap-canvas'].getContext('2d');
  els._ctx = ctx;

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
  els['cap-download'].addEventListener('click', download);

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

function captureFrame() {
  if (!lastResult || !lastResult.worldLandmarks || lastResult.worldLandmarks.length === 0) {
    els['cap-status'].textContent = 'No hand visible — try again.';
    return;
  }
  const letter = ALL_LETTERS[idx];
  const wlm    = lastResult.worldLandmarks[0];
  const handed = lastResult.handedness?.[0]?.[0]?.categoryName ?? 'Right';
  if (!detectedHandedness) detectedHandedness = handed;
  captured[letter] = {
    landmarks: wlm.map((p) => ({ x: p.x, y: p.y, z: p.z })),
    handedness: handed,
    capturedAt: new Date().toISOString(),
  };
  if (STATIC_END_POSE_LETTERS.includes(letter)) {
    captured[letter].note = 'end pose only — motion path TODO';
  }
  idx++;
  updateUI();
}

function retry() {
  if (idx > 0) idx--;
  delete captured[ALL_LETTERS[idx]];
  updateUI();
}

function buildGrid() {
  els['cap-grid'].innerHTML = '';
  for (const L of ALL_LETTERS) {
    const cell = document.createElement('div');
    cell.className = 'cap-cell';
    cell.dataset.letter = L;
    cell.textContent = L;
    els['cap-grid'].appendChild(cell);
  }
}

function refreshGrid() {
  for (const cell of els['cap-grid'].children) {
    const L = cell.dataset.letter;
    cell.classList.toggle('done', !!captured[L]);
    cell.classList.toggle('active', ALL_LETTERS[idx] === L);
  }
}

function updateUI() {
  refreshGrid();
  if (idx >= ALL_LETTERS.length) {
    els['cap-letter'].textContent = '✓';
    els['cap-status'].textContent = 'All captured. Download the JSON, then drop it into Assets/asl_letters.json.';
    els['cap-capture'].disabled = true;
    els['cap-download'].disabled = false;
  } else {
    const L = ALL_LETTERS[idx];
    const tag = STATIC_END_POSE_LETTERS.includes(L) ? '   (end pose only)' : '';
    els['cap-letter'].textContent = L + tag;
    const handTag = detectedHandedness ? ` Detected hand: ${detectedHandedness}.` : '';
    els['cap-status'].textContent = `Sign "${L}" then click Capture.${handTag}`;
    els['cap-capture'].disabled = false;
    els['cap-download'].disabled = true;
  }
}

function download() {
  const out = {
    version: 1,
    handedness: detectedHandedness ?? 'Right',
    capturedAt: new Date().toISOString(),
    letters: captured,
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'asl_letters.json';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
