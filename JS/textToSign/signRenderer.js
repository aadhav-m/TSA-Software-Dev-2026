import { synthesize, validateAgainstClassifier } from './syntheticLetters.js';

const ALL_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const STATIC_END_POSE_LETTERS = ['J', 'Z'];

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const FINGER_LANDMARKS = {
  thumb:  [1, 2, 3, 4],
  index:  [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring:   [13, 14, 15, 16],
  pinky:  [17, 18, 19, 20],
};
const TIPS = new Set([4, 8, 12, 16, 20]);

const COLORS = {
  thumb:  '#E8845A',
  index:  '#C45020',
  middle: '#2ABFB0',
  ring:   '#1A9E91',
  pinky:  '#7A6F68',
  palm:   '#3F362F',
  dot:    '#1E1A17',
  inner:  '#FAF7F4',
  bg:     '#F2EDE8',
};

const TRANSITION_MS = 150;
const HOLD_MS       = 250;
const RELAX_FRAC    = 0.30;
const RELAX_MS      = 100;

const _fingerByLandmark = (() => {
  const m = new Array(21).fill('palm');
  for (const [name, ids] of Object.entries(FINGER_LANDMARKS)) {
    for (const i of ids) m[i] = name;
  }
  return m;
})();

const _fingerByEdge = CONNECTIONS.map(([a, b]) => {
  if (a === 0 || b === 0) return _fingerByLandmark[a === 0 ? b : a];
  const fa = _fingerByLandmark[a], fb = _fingerByLandmark[b];
  return fa === fb ? fa : 'palm';
});

export class SignRenderer {
  constructor(canvas, opts = {}) {
    this.canvas   = canvas;
    this.ctx      = canvas.getContext('2d');
    this.dataURL  = opts.dataURL ?? '/Assets/asl_letters.json';
    this.mirror   = opts.mirror  ?? null;
    this.onStatus = opts.onStatus ?? (() => {});

    this._rawLetters    = null;
    this.letterFrames   = null;
    this.dataHandedness = null;
    this.currentLandmarks = null;
    this.queue   = [];
    this.current = null;

    this._tick = this._tick.bind(this);
    this._resize();
    window.addEventListener('resize', () => this._resize());
    requestAnimationFrame(this._tick);
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.canvas.width  = Math.floor(w * dpr);
    this.canvas.height = Math.floor(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  async load() {
    this.onStatus('Loading letter data…');
    let data;
    try {
      const resp = await fetch(this.dataURL, { cache: 'no-store' });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      data = await resp.json();
    } catch (err) {
      this.onStatus('⚠ Letter data missing — click "Run capture" below to record A–Z first.');
      console.warn('[signRenderer] letter data unavailable:', err.message);
      return;
    }
    this.dataHandedness = data.handedness ?? 'Right';
    if (this.mirror === null) {
      this.mirror = !!(this.dataHandedness && this.dataHandedness.toLowerCase().startsWith('l'));
    }

    const merged = {};
    let nCaptured = 0, nSynth = 0, nRejected = 0;
    for (const L of ALL_LETTERS) {
      const cap = data.letters?.[L];
      if (cap && cap.landmarks && validateAgainstClassifier(L, cap.landmarks)) {
        merged[L] = cap.landmarks;
        nCaptured++;
      } else {
        const synthLm = synthesize(L);
        if (synthLm) {
          merged[L] = synthLm;
          if (cap) nRejected++; else nSynth++;
        }
      }
    }
    this._rawLetters = merged;
    this.letterFrames = this._applyMirror(merged, this.mirror);
    this.currentLandmarks = this._cloneLandmarks(this.letterFrames.B || synthesize('B'));

    const have = Object.keys(this.letterFrames).length;
    const note = (nSynth + nRejected) > 0
      ? ` (${nCaptured} captured, ${nRejected} replaced, ${nSynth} synth-only)`
      : '';
    this.onStatus(`Ready. ${have}/26 letters loaded${note}. Hand: ${this.dataHandedness}${this.mirror ? ' (mirrored)' : ''}.`);
  }

  setMirror(m) {
    if (this.mirror === !!m) return;
    this.mirror = !!m;
    if (this._rawLetters) {
      this.letterFrames = this._applyMirror(this._rawLetters, this.mirror);
      // Re-mirror current pose too, so it doesn't snap when toggled mid-anim.
      this.currentLandmarks = this.currentLandmarks.map((p) => ({ x: -p.x, y: p.y, z: p.z }));
      if (this.current) {
        this.current.start  = this.current.start.map((p)  => ({ x: -p.x, y: p.y, z: p.z }));
        this.current.target = this.current.target.map((p) => ({ x: -p.x, y: p.y, z: p.z }));
      }
    }
    this.onStatus(`Mirror ${this.mirror ? 'on' : 'off'}.`);
  }

  playText(text) {
    if (!this.letterFrames) {
      this.onStatus('⚠ Letter data missing — run the capture page first ("Run capture" link below).');
      return;
    }
    const letters = String(text).toUpperCase().replace(/[^A-Z]/g, '').split('');
    if (letters.length === 0) { this.onStatus('Type at least one letter.'); return; }
    this.queue = [];
    let prev = null;
    for (const L of letters) {
      if (L === prev) this.queue.push({ kind: 'relax' });
      this.queue.push({ kind: 'letter', letter: L });
      prev = L;
    }
    this.queue.push({ kind: 'rest' });
    if (!this.current) this._advance();
  }

  stop() {
    this.queue = [];
    this.current = null;
  }

  _applyMirror(letters, mirror) {
    const out = {};
    for (const [L, lm] of Object.entries(letters)) {
      out[L] = lm.map((p) => ({ x: mirror ? -p.x : p.x, y: p.y, z: p.z }));
    }
    return out;
  }

  _cloneLandmarks(lm) { return lm.map((p) => ({ x: p.x, y: p.y, z: p.z })); }

  _restPose() {
    return this.letterFrames?.B
        ?? this.letterFrames?.U
        ?? Object.values(this.letterFrames ?? {})[0]
        ?? null;
  }

  _advance() {
    if (this.queue.length === 0) { this.current = null; return; }
    const step = this.queue.shift();
    let target;
    let duration = TRANSITION_MS;
    let hold     = HOLD_MS;

    if (step.kind === 'relax') {
      const rest = this._restPose() || this.currentLandmarks;
      target = this.currentLandmarks.map((p, i) => ({
        x: p.x * (1 - RELAX_FRAC) + rest[i].x * RELAX_FRAC,
        y: p.y * (1 - RELAX_FRAC) + rest[i].y * RELAX_FRAC,
        z: p.z * (1 - RELAX_FRAC) + rest[i].z * RELAX_FRAC,
      }));
      duration = RELAX_MS;
      hold     = 0;
    } else if (step.kind === 'rest') {
      target = this._restPose() || this.currentLandmarks;
      hold = 0;
    } else if (step.kind === 'letter') {
      target = this.letterFrames[step.letter];
      if (!target) { this.onStatus(`Skipping "${step.letter}" — not loaded.`); return this._advance(); }
      const tag = STATIC_END_POSE_LETTERS.includes(step.letter) ? ' (end pose only — motion path TODO)' : '';
      this.onStatus(`Signing "${step.letter}"${tag}…`);
    }

    this.current = {
      start: this._cloneLandmarks(this.currentLandmarks),
      target: this._cloneLandmarks(target),
      duration, hold,
      t0: performance.now(),
      phase: 'transition',
    };
  }

  _tick(now) {
    requestAnimationFrame(this._tick);
    if (this.current) {
      const c = this.current;
      if (c.phase === 'transition') {
        const t = Math.min(1, (now - c.t0) / Math.max(1, c.duration));
        const e = ease(t);
        for (let i = 0; i < 21; i++) {
          this.currentLandmarks[i].x = c.start[i].x * (1 - e) + c.target[i].x * e;
          this.currentLandmarks[i].y = c.start[i].y * (1 - e) + c.target[i].y * e;
          this.currentLandmarks[i].z = c.start[i].z * (1 - e) + c.target[i].z * e;
        }
        if (t >= 1) {
          c.phase = 'hold';
          c.holdStart = now;
          if (c.hold === 0) this._advance();
        }
      } else if (c.phase === 'hold') {
        if (now - c.holdStart >= c.hold) this._advance();
      }
    }
    this._draw();
  }

  _draw() {
    const ctx = this.ctx;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h || !this.currentLandmarks) {
      if (w && h) { ctx.clearRect(0, 0, w, h); }
      return;
    }
    ctx.clearRect(0, 0, w, h);

    const ref = this._rawLetters?.B || this.currentLandmarks;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of ref) {
      const x = this.mirror ? -p.x : p.x;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    }
    const handW = Math.max(maxX - minX, 0.001);
    const handH = Math.max(maxY - minY, 0.001);
    const padding = Math.min(w, h) * 0.08;
    const scale = Math.min((w - 2 * padding) / handW, (h - 2 * padding) / handH) * 0.85;
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;

    const zMin = -0.08, zMax = 0.08;
    const project = (p) => ({
      x: (p.x - cx) * scale + w / 2,
      y: (p.y - cy) * scale + h / 2,
      z: clamp((p.z - zMin) / (zMax - zMin), 0, 1),
    });

    const edges = CONNECTIONS.map(([a, b], i) => ({
      a, b, finger: _fingerByEdge[i],
      depth: (this.currentLandmarks[a].z + this.currentLandmarks[b].z) * 0.5,
    })).sort((u, v) => v.depth - u.depth);

    ctx.lineCap = 'round';
    for (const e of edges) {
      const pa = project(this.currentLandmarks[e.a]);
      const pb = project(this.currentLandmarks[e.b]);
      const colour = COLORS[e.finger] || COLORS.palm;
      const widthBase = e.finger === 'palm' ? 7 : 9;
      const width = widthBase * (0.7 + 0.6 * (1 - (pa.z + pb.z) * 0.5));
      ctx.strokeStyle = colour + '33';
      ctx.lineWidth = width + 6;
      ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
      ctx.strokeStyle = colour;
      ctx.lineWidth = width;
      ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
    }

    const points = this.currentLandmarks.map((p, i) => ({
      i, p: project(p), depth: p.z,
    })).sort((u, v) => v.depth - u.depth);

    for (const { i, p } of points) {
      const finger = _fingerByLandmark[i];
      const colour = COLORS[finger] || COLORS.palm;
      const isAnchor = i === 0;
      const isTip = TIPS.has(i);
      const baseR = isAnchor ? 9 : isTip ? 8 : 6;
      const r = baseR * (0.75 + 0.5 * (1 - p.z));
      ctx.fillStyle = colour;
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = COLORS.inner;
      ctx.beginPath(); ctx.arc(p.x, p.y, r * 0.42, 0, Math.PI * 2); ctx.fill();
    }
  }

  verify(letter, mode) {
    const lm = this.letterFrames?.[letter.toUpperCase()];
    if (!lm) return null;
    if (mode === '2d') {
      return lm.map((p, i) => ({ i, x: p.x, y: p.y, z: p.z, finger: _fingerByLandmark[i] }));
    }
    return lm;
  }
}

function clamp(x, lo, hi) { return Math.min(hi, Math.max(lo, x)); }
function ease(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }

(async function autoInit() {
  if (typeof document === 'undefined') return;
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoInit, { once: true });
    return;
  }
  const canvas    = document.getElementById('t2s-canvas');
  if (!canvas) return;
  const playBtn   = document.getElementById('t2s-play');
  const stopBtn   = document.getElementById('t2s-stop');
  const sourceEl  = document.getElementById('t2s-source') || document.getElementById('tts-input');
  const statusEl  = document.getElementById('t2s-status');
  const mirrorChk = document.getElementById('t2s-mirror');

  const renderer = new SignRenderer(canvas, {
    onStatus: (msg) => {
      if (!statusEl) return;
      statusEl.textContent = msg;
      statusEl.classList.toggle('warn', msg.startsWith('⚠'));
    },
  });
  window.__t2s = renderer;
  await renderer.load();

  if (mirrorChk) {
    mirrorChk.checked = !!renderer.mirror;
    mirrorChk.addEventListener('change', () => renderer.setMirror(mirrorChk.checked));
  }
  if (playBtn) playBtn.addEventListener('click', () => {
    const txt = sourceEl ? (sourceEl.value || '').trim() : '';
    renderer.playText(txt);
  });
  if (stopBtn) stopBtn.addEventListener('click', () => renderer.stop());
})();

export { ALL_LETTERS };
