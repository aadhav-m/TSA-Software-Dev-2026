// 3D fingerspelling renderer.
//
// Loads /Assets/HandRigs.glb (rigged right-hand model with 26 animation clips
// named "A"-"Z", one 12-frame held pose per ASL letter; J/Z include motion).
// Plays clips in sequence with crossfades between letters. Public API matches
// the previous 2D renderer so the auto-init wiring on TextToSign.html keeps
// working unchanged: load(), playText(), stop(), setMirror() (no-op shim).
//
// Non-letter characters in the input are filtered out before sequencing.
// Unknown clips (e.g. a stray "B.001" leftover from the source rig) are
// ignored; only single-uppercase-letter clip names are accepted.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const LETTER_RE = /^[A-Z]$/;
const MOTION_LETTERS = new Set(['J', 'Z']); // clips that contain real motion

// Crossfade between letters, then dwell on the held end-pose. We normalise
// every clip to a fixed playback duration via setDuration() so timing doesn't
// depend on whatever the Blender timeline range was at export time — that
// avoids the "sometimes a letter takes way longer" problem when one clip
// happens to have been exported with a long trailing empty range.
const TRANSITION_MS      = 200;
const STATIC_DURATION_S  = 0.40;  // held-pose letters
const MOTION_DURATION_S  = 0.70;  // J/Z — trace needs to be readable
const HOLD_MS            = 250;   // dwell on the end-pose before the next letter
const REBOUND_MS         = 140;   // brief drop-to-neutral between repeated letters

// Camera framing. PALM_FORWARD_AXIS is the axis (in the model's local space
// after import) that points OUT from the palm toward a viewer. The Rigify
// export's exact orientation is hard to predict, so this is the one knob to
// flip if the back of the hand is facing the camera instead of the palm:
//   '+z' / '-z' / '+x' / '-x' / '+y' / '-y'
const PALM_FORWARD_AXIS = '+z';
const FRAME_PADDING     = 1.6;  // bigger = hand smaller in frame

// Verbose timing diagnostics. Toggle at runtime: window.__t2s.debug = false.
// Logs: clip durations on load, every _advance with planned + actual ms,
// every rAF stall > 80ms, every queue play.
const DEBUG_DEFAULT = true;
const TICK_STALL_MS = 80;     // warn when frames take longer than this
const log = (...args) => console.log('%c[t2s]', 'color:#C45020;font-weight:bold', ...args);
const warn = (...args) => console.warn('%c[t2s]', 'color:#C45020;font-weight:bold', ...args);

export class SignRenderer {
  constructor(canvas, opts = {}) {
    this.canvas   = canvas;
    this.glbURL   = opts.glbURL  ?? '/Assets/HandRigs.glb';
    this.onStatus = opts.onStatus ?? (() => {});

    this.scene  = new THREE.Scene();
    this.scene.background = null; // let panel CSS gradient show through

    this.camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;

    this.scene.add(new THREE.HemisphereLight(0xfff5e8, 0xc4b5a8, 1.0));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(2, 3, 4);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffe4cc, 0.5);
    fill.position.set(-2, 1, 2);
    this.scene.add(fill);

    this.mixer         = null;
    this.actions       = new Map();   // letter -> AnimationAction
    this.currentAction = null;
    this.queue         = [];
    this.step          = null;        // { letter, endsAt, startedAt } | { kind:'rest', endsAt } | null
    this._lastTickMs   = 0;
    this.debug         = opts.debug ?? DEBUG_DEFAULT;
    this.speed         = opts.speed ?? 1.0;  // 1.0 = normal, 0.5 = half speed, 0.25 = quarter

    this._tick = this._tick.bind(this);
    this._resize();
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(canvas);
    requestAnimationFrame(this._tick);
  }

  async load() {
    this.onStatus('Loading hand model…');
    const loader = new GLTFLoader();
    let gltf;
    try {
      gltf = await loader.loadAsync(this.glbURL);
    } catch (err) {
      this.onStatus('⚠ Could not load hand model.');
      console.error('[signRenderer] GLB load failed:', err);
      return;
    }

    this.model = gltf.scene;
    this.scene.add(this.model);
    this._frameCamera(this.model);

    this.mixer = new THREE.AnimationMixer(this.model);
    this.mixer.timeScale = this.speed;
    let nLoaded = 0, nSkipped = 0;
    const clipReport = [];
    const skippedNames = [];
    for (const clip of gltf.animations) {
      if (!LETTER_RE.test(clip.name)) { nSkipped++; skippedNames.push(clip.name); continue; }
      const action = this.mixer.clipAction(clip);
      action.setLoop(THREE.LoopOnce, 1);
      action.clampWhenFinished = true;   // hold the last frame as a static pose
      // NOTE: do NOT setEffectiveWeight(0) here. three.js computes effective
      // weight as `this.weight * interpolantValue`. fadeIn() ramps the
      // interpolant 0→1 but doesn't touch base weight; if base is 0 the
      // result is 0×(0..1)=0 and nothing renders. Default weight is 1 — keep
      // it. Inactive actions don't render anyway because we never call play().
      // Force every clip to play in a known time, irrespective of how its
      // native duration was exported. setDuration() re-scales timeScale.
      const nativeSec = clip.duration;
      const targetSec = MOTION_LETTERS.has(clip.name) ? MOTION_DURATION_S : STATIC_DURATION_S;
      action.setDuration(targetSec);
      this.actions.set(clip.name, action);
      clipReport.push({
        letter: clip.name,
        nativeMs: +(nativeSec * 1000).toFixed(1),
        forcedMs: +(targetSec * 1000).toFixed(1),
        timeScale: +(action.getEffectiveTimeScale()).toFixed(3),
        tracks: clip.tracks.length,
      });
      nLoaded++;
    }

    const skipNote = nSkipped ? ` (${nSkipped} non-letter clip${nSkipped === 1 ? '' : 's'} ignored)` : '';
    this.onStatus(`Ready. ${nLoaded}/26 letters loaded${skipNote}.`);

    if (this.debug) {
      log(`loaded ${nLoaded}/26 clips (skipped: ${skippedNames.join(', ') || 'none'})`);
      console.table(clipReport);
      const odd = clipReport.filter((r) => r.nativeMs > 600 || r.nativeMs < 100);
      if (odd.length) warn(`unusual native clip durations — likely the source of inconsistent timing if setDuration didn't fix it:`, odd);
    }
  }

  playText(text) {
    if (!this.mixer) {
      this.onStatus('⚠ Hand model not ready yet.');
      return;
    }
    const letters = String(text).toUpperCase().replace(/[^A-Z]/g, '').split('');
    if (letters.length === 0) {
      this.onStatus('Type at least one letter.');
      return;
    }
    // Insert a brief rebound between same-letter repeats so the second one is
    // visually distinct (static-pose clips have 12 identical frames, so just
    // re-playing the action looks like nothing happened).
    this.queue = [];
    let prev = null;
    for (const L of letters) {
      if (L === prev) this.queue.push({ kind: 'rebound' });
      this.queue.push({ kind: 'letter', letter: L });
      prev = L;
    }
    this.queue.push({ kind: 'rest' });
    if (this.debug) {
      const shape = this.queue.map((s) => s.kind === 'letter' ? s.letter : s.kind[0]).join(',');
      log(`playText("${text}") → queue: [${shape}] @ t=${performance.now().toFixed(0)}ms${this.step ? ' (step in progress, will continue)' : ''}`);
      this._playStartedAt = performance.now();
    }
    if (!this.step) this._advance();
  }

  stop() {
    this.queue = [];
    this.step = null;
    if (this.currentAction) {
      this.currentAction.fadeOut(TRANSITION_MS / 1000);
      this.currentAction = null;
    }
    this.onStatus('Stopped.');
  }

  // No-op: model is already a right hand and the page has no mirror toggle.
  // Kept so the auto-init's optional checkbox handler can call it safely.
  setMirror() {}

  // Slow down (or speed up) playback. 1.0 = normal, 0.5 = half speed, etc.
  // Scales BOTH the mixer's clip-playback timeScale AND the queue's
  // wait times in _advance, so per-letter on-screen duration matches the
  // queue's idea of when to crossfade. Clamped to a sane range.
  setSpeed(speed) {
    const v = Number(speed);
    this.speed = Math.max(0.1, Math.min(2.0, isFinite(v) && v > 0 ? v : 1.0));
    if (this.mixer) this.mixer.timeScale = this.speed;
    if (this.debug) log(`speed → ${this.speed}× (mixer.timeScale=${this.speed})`);
  }

  // ─── Internals ────────────────────────────────────────────────────────

  _advance() {
    const now = performance.now();
    if (this.debug && this.step) {
      const planned = this.step.endsAt - this.step.startedAt;
      const actual  = now - this.step.startedAt;
      const drift   = actual - planned;
      const tag     = this.step.kind === 'letter' ? `"${this.step.letter}"` : this.step.kind;
      const drag    = drift > 30 ? ` ⚠ DRIFT +${drift.toFixed(0)}ms` : '';
      log(`step ${tag} ended → planned=${planned.toFixed(0)}ms, actual=${actual.toFixed(0)}ms${drag}`);
    }

    if (this.queue.length === 0) {
      this.step = null;
      if (this.debug && this._playStartedAt) {
        log(`queue drained — total wall time = ${(now - this._playStartedAt).toFixed(0)}ms`);
        this._playStartedAt = null;
      }
      return;
    }
    const next = this.queue.shift();

    // Wait times are scaled by 1/speed because three.js mixer.timeScale=speed
    // makes every fade and clip take 1/speed real seconds — our queue runs
    // on real wall time, so we have to wait that long too.
    const sf = 1 / this.speed;

    if (next.kind === 'rest') {
      if (this.currentAction) this.currentAction.fadeOut(TRANSITION_MS / 1000);
      this.currentAction = null;
      const ms = TRANSITION_MS * sf;
      this.step = { kind: 'rest', endsAt: now + ms, startedAt: now };
      this.onStatus('Done.');
      if (this.debug) log(`step rest started @ t=${now.toFixed(0)}ms (planned ${ms.toFixed(0)}ms @ ${this.speed}×)`);
      return;
    }

    if (next.kind === 'rebound') {
      // Drop the current letter's weight to 0 briefly so the next instance
      // visibly fades back in. Setting currentAction=null routes the next
      // letter through the "first (no fade)" path → clean fadeIn from 0.
      if (this.currentAction) this.currentAction.fadeOut(REBOUND_MS / 1000);
      this.currentAction = null;
      const ms = REBOUND_MS * sf;
      this.step = { kind: 'rebound', endsAt: now + ms, startedAt: now };
      if (this.debug) log(`step rebound started @ t=${now.toFixed(0)}ms (planned ${ms.toFixed(0)}ms @ ${this.speed}×)`);
      return;
    }

    const action = this.actions.get(next.letter);
    if (!action) {
      this.onStatus(`Skipping "${next.letter}" — clip not in model.`);
      if (this.debug) warn(`skipping "${next.letter}" — no clip in model`);
      return this._advance();
    }

    this.onStatus(`Signing "${next.letter}"…`);

    // Canonical three.js crossfade pattern: reset() the new action, set base
    // weight to 1, schedule the interpolant 0→1 via fadeIn, then play(). If
    // there's an outgoing action, fadeOut it in parallel — the two fades
    // overlap into a crossfade. Same shape works for the first letter too.
    // (Same-action case is impossible here because playText inserts a
    // rebound step between any two identical letters, which sets
    // currentAction back to null.)
    const path = this.currentAction ? 'crossfade' : 'fade-in from rest';
    const fadeS = TRANSITION_MS / 1000;
    if (this.currentAction) this.currentAction.fadeOut(fadeS);
    action.reset();
    action.setEffectiveWeight(1);   // base weight; fadeIn's interpolant multiplies onto this
    action.fadeIn(fadeS);
    action.play();
    this.currentAction = action;

    // Wait for the clip to play out (its duration is fixed by setDuration in
    // load()), then the dwell. The transition overlaps the start of the clip
    // and is shorter than the clip, so it doesn't add to the total. Scaled
    // by 1/speed so a 0.5× slider gives 2× real-time per letter.
    const playMs = (MOTION_LETTERS.has(next.letter)
      ? MOTION_DURATION_S * 1000
      : STATIC_DURATION_S * 1000) * sf;
    const holdMs = HOLD_MS * sf;
    this.step = {
      kind: 'letter',
      letter: next.letter,
      endsAt: now + playMs + holdMs,
      startedAt: now,
    };
    if (this.debug) {
      log(`step "${next.letter}" started @ t=${now.toFixed(0)}ms — ${path}, play=${playMs.toFixed(0)}ms + hold=${holdMs.toFixed(0)}ms = ${(playMs+holdMs).toFixed(0)}ms total @ ${this.speed}× | actionTime=${action.time.toFixed(3)}s, weight=${action.getEffectiveWeight().toFixed(2)}, queue remaining=${this.queue.length}`);
    }
  }

  _tick() {
    requestAnimationFrame(this._tick);
    const now = performance.now();
    const gap = this._lastTickMs ? (now - this._lastTickMs) : 0;
    const dt  = this._lastTickMs ? Math.min(0.1, gap / 1000) : 0;
    this._lastTickMs = now;
    if (this.mixer) this.mixer.update(dt);

    // Detect rAF stalls — these would manifest as "delays between letters"
    // because the queue advance is gated by the rAF loop hitting endsAt.
    if (this.debug && gap > TICK_STALL_MS && this.step) {
      warn(`rAF stall: ${gap.toFixed(0)}ms gap (during step ${this.step.kind === 'letter' ? '"' + this.step.letter + '"' : this.step.kind})`);
    }

    if (this.step && now >= this.step.endsAt) {
      this._advance();
    }
    this.renderer.render(this.scene, this.camera);
  }

  _resize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _frameCamera(object) {
    const box    = new THREE.Box3().setFromObject(object);
    const size   = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    const maxDim   = Math.max(size.x, size.y, size.z);
    const fovRad   = this.camera.fov * (Math.PI / 180);
    const distance = (maxDim / 2) / Math.tan(fovRad / 2) * FRAME_PADDING;

    const dir = new THREE.Vector3();
    switch (PALM_FORWARD_AXIS) {
      case '+x': dir.set( 1,  0,  0); break;
      case '-x': dir.set(-1,  0,  0); break;
      case '+y': dir.set( 0,  1,  0); break;
      case '-y': dir.set( 0, -1,  0); break;
      case '-z': dir.set( 0,  0, -1); break;
      case '+z': default: dir.set(0, 0, 1); break;
    }

    this.camera.position.copy(center).addScaledVector(dir, distance);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(center);
    this.camera.near = distance / 100;
    this.camera.far  = distance * 100;
    this.camera.updateProjectionMatrix();
  }
}

// ─── Auto-wire on the TextToSign page ──────────────────────────────────────
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

  const renderer = new SignRenderer(canvas, {
    onStatus: (msg) => {
      if (!statusEl) return;
      statusEl.textContent = msg;
      statusEl.classList.toggle('warn', msg.startsWith('⚠'));
    },
  });
  window.__t2s = renderer;
  await renderer.load();

  if (playBtn) playBtn.addEventListener('click', () => {
    const txt = sourceEl ? (sourceEl.value || '').trim() : '';
    renderer.playText(txt);
  });
  if (stopBtn) stopBtn.addEventListener('click', () => renderer.stop());

  // Optional speed slider — bind if present. Format: <input type="range"
  // id="t2s-speed" min="0.25" max="1" step="0.05" value="1"> with a sibling
  // <span id="t2s-speed-value"> that gets a "1×" / "0.5×" readout.
  const speedEl    = document.getElementById('t2s-speed');
  const speedValEl = document.getElementById('t2s-speed-value');
  if (speedEl) {
    const apply = () => {
      const v = parseFloat(speedEl.value);
      renderer.setSpeed(v);
      if (speedValEl) speedValEl.textContent = `${v}×`;
    };
    speedEl.addEventListener('input', apply);
    apply();
  }
})();
