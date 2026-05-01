const WRIST = [0.000, +0.090, 0.000];

const MCP_INDEX  = [+0.025, 0.000, 0.000];
const MCP_MIDDLE = [+0.005, -0.005, 0.000];
const MCP_RING   = [-0.020, 0.000, 0.000];
const MCP_PINKY  = [-0.042, +0.012, 0.000];

const THUMB_CMC  = [+0.015, +0.072, -0.005];

const LEN = {
  index:  [0.040, 0.025, 0.020],
  middle: [0.045, 0.027, 0.022],
  ring:   [0.040, 0.025, 0.020],
  pinky:  [0.030, 0.020, 0.018],
};

function _fingerExtended(mcp, lengths, spreadDeg = 0) {
  const a = (spreadDeg * Math.PI) / 180;
  const dx = Math.sin(a);
  const dy = -Math.cos(a);
  const dz = -0.05; // slight forward tilt
  return _walk(mcp, [dx, dy, dz], lengths, [0, 0, 0]);
}

function _fingerCurled(mcp, lengths, spreadDeg = 0) {
  const a = (spreadDeg * Math.PI) / 180;
  const dx = Math.sin(a);
  const dy = -Math.cos(a);
  return _walk(mcp, [dx, dy, 0], lengths, [95, 90, 50]);
}

function _fingerMid(mcp, lengths, spreadDeg = 0) {
  const a = (spreadDeg * Math.PI) / 180;
  const dx = Math.sin(a);
  const dy = -Math.cos(a);
  return _walk(mcp, [dx, dy, -0.08], lengths, [55, 35, 20]);
}

function _fingerHook(mcp, lengths, spreadDeg = 0) {
  const a = (spreadDeg * Math.PI) / 180;
  const dx = Math.sin(a);
  const dy = -Math.cos(a);
  return _walk(mcp, [dx, dy, -0.05], lengths, [25, 70, 60]);
}

function _walk(mcp, dir0, lengths, bends) {
  const norm = _norm(dir0);
  let d = norm.slice();
  let p = mcp.slice();
  const points = [p.slice()];
  for (let i = 0; i < 3; i++) {
    p = [p[0] + d[0] * lengths[i], p[1] + d[1] * lengths[i], p[2] + d[2] * lengths[i]];
    points.push(p.slice());
    // Rotate direction around X axis: y' = y cos − z sin, z' = y sin + z cos.
    // Positive angle bends finger forward (+Z) — into the palm.
    const a = (bends[i] * Math.PI) / 180;
    const ny = d[1] * Math.cos(a) - d[2] * Math.sin(a);
    const nz = d[1] * Math.sin(a) + d[2] * Math.cos(a);
    d = [d[0], ny, nz];
  }
  return points;
}

function _norm(v) {
  const m = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / m, v[1] / m, v[2] / m];
}

function _thumbSide()      { return _thumb([+0.040, -0.020, -0.010], [+0.040, -0.020, +0.000]); }
function _thumbAcross()    { return _thumb([+0.020, -0.005, +0.025], [-0.005, -0.005, +0.030]); }
function _thumbUp()        { return _thumb([-0.005, -0.030, -0.005], [-0.010, -0.030, -0.010]); }
function _thumbDown()      { return _thumb([+0.010, +0.040, -0.005], [+0.005, +0.045, -0.010]); }
function _thumbOut()       { return _thumb([+0.045, -0.025, -0.005], [+0.060, -0.040, -0.005]); }
function _thumbBetween()   { return _thumb([+0.015, +0.005, +0.020], [+0.005, +0.020, +0.015]); }
function _thumbTucked()    { return _thumb([+0.018, -0.010, +0.005], [+0.012, -0.020, +0.020]); }

function _thumb(midDelta, tipDelta) {
  const cmc = THUMB_CMC.slice();
  const mcp = [cmc[0] + 0.025, cmc[1] - 0.020, cmc[2] - 0.005];
  const ip  = [mcp[0] + midDelta[0], mcp[1] + midDelta[1], mcp[2] + midDelta[2]];
  const tip = [ip[0]  + tipDelta[0], ip[1]  + tipDelta[1], ip[2]  + tipDelta[2]];
  return [cmc, mcp, ip, tip];
}

const LETTERS = {
  B: { I: 'ext', M: 'ext', R: 'ext', P: 'ext', T: 'tucked' },

  A: { I: 'crl', M: 'crl', R: 'crl', P: 'crl', T: 'side'    },
  S: { I: 'crl', M: 'crl', R: 'crl', P: 'crl', T: 'across'  },
  T: { I: 'crl', M: 'crl', R: 'crl', P: 'crl', T: 'between' },
  M: { I: 'crl', M: 'crl', R: 'crl', P: 'crl', T: 'across', _thumbDeep: true  },
  N: { I: 'crl', M: 'crl', R: 'crl', P: 'crl', T: 'across', _thumbDeepN: true },
  E: { I: 'crl', M: 'crl', R: 'crl', P: 'crl', T: 'tucked'  },

  D: { I: 'ext', M: 'mid', R: 'mid', P: 'crl', T: 'tucked'  }, 
  L: { I: 'ext', M: 'crl', R: 'crl', P: 'crl', T: 'out'     }, 
  I: { I: 'crl', M: 'crl', R: 'crl', P: 'ext', T: 'tucked'  }, 
  J: { I: 'crl', M: 'crl', R: 'crl', P: 'ext', T: 'tucked'  }, 
  Y: { I: 'crl', M: 'crl', R: 'crl', P: 'ext', T: 'out'     }, 
  X: { I: 'hook', M: 'crl', R: 'crl', P: 'crl', T: 'tucked' }, 

  U: { I: 'ext', M: 'ext', R: 'crl', P: 'crl', T: 'tucked', spread: [0, 0, 0, 0] },
  R: { I: 'ext', M: 'ext', R: 'crl', P: 'crl', T: 'tucked', spread: [3, -3, 0, 0] }, 
  V: { I: 'ext', M: 'ext', R: 'crl', P: 'crl', T: 'tucked', spread: [-15, 15, 0, 0] }, 
  W: { I: 'ext', M: 'ext', R: 'ext', P: 'crl', T: 'tucked', spread: [-12, 0, 12, 0] },
  K: { I: 'ext', M: 'ext', R: 'crl', P: 'crl', T: 'up', spread: [-25, 5, 0, 0] }, 
  F: { I: 'crl', M: 'ext', R: 'ext', P: 'ext', T: 'tucked', _indexThumbCircle: true },

  G: { I: 'ext', M: 'crl', R: 'crl', P: 'crl', T: 'tucked', _sideways: true },
  H: { I: 'ext', M: 'ext', R: 'crl', P: 'crl', T: 'tucked', _sideways: true },
  P: { I: 'ext', M: 'mid', R: 'crl', P: 'crl', T: 'down',    _sideways: true },
  Q: { I: 'ext', M: 'mid', R: 'crl', P: 'crl', T: 'down',    _sideways: true, _down: true },
  Z: { I: 'ext', M: 'crl', R: 'crl', P: 'crl', T: 'tucked', _sideways: true }, 

  C: { I: 'mid', M: 'mid', R: 'mid', P: 'mid', T: 'out', _cShape: true },
  O: { I: 'mid', M: 'mid', R: 'mid', P: 'mid', T: 'tucked', _oShape: true },
};

const FINGER_FN = {
  ext:  _fingerExtended,
  mid:  _fingerMid,
  crl:  _fingerCurled,
  hook: _fingerHook,
};

const THUMB_FN = {
  side:    _thumbSide,
  across:  _thumbAcross,
  up:      _thumbUp,
  down:    _thumbDown,
  out:     _thumbOut,
  between: _thumbBetween,
  tucked:  _thumbTucked,
};

export function synthesize(letter) {
  const spec = LETTERS[letter.toUpperCase()];
  if (!spec) return null;

  const sp = spec.spread || [0, 0, 0, 0];
  const idx = FINGER_FN[spec.I](MCP_INDEX,  LEN.index,  sp[0]);
  const mid = FINGER_FN[spec.M](MCP_MIDDLE, LEN.middle, sp[1]);
  const rng = FINGER_FN[spec.R](MCP_RING,   LEN.ring,   sp[2]);
  const pky = FINGER_FN[spec.P](MCP_PINKY,  LEN.pinky,  sp[3]);
  const thb = THUMB_FN[spec.T]();

  let lm = [
    WRIST,
    thb[0], thb[1], thb[2], thb[3],
    idx[0], idx[1], idx[2], idx[3],
    mid[0], mid[1], mid[2], mid[3],
    rng[0], rng[1], rng[2], rng[3],
    pky[0], pky[1], pky[2], pky[3],
  ];

  if (spec._sideways) lm = _rotateZ(lm, -75);
  if (spec._down)     lm = _translate(lm, [0.02, +0.03, 0]);
  if (spec._cShape) {
    const t = thb[3];
    for (const i of [8, 12, 16, 20]) {
      lm[i] = [
        lm[i][0] * 0.55 + t[0] * 0.45,
        lm[i][1] * 0.65 + t[1] * 0.35,
        lm[i][2] - 0.015,
      ];
    }
  }
  if (spec._oShape) {
    const t = thb[3];
    for (const i of [8, 12, 16, 20]) {
      lm[i] = [
        lm[i][0] * 0.25 + t[0] * 0.75,
        lm[i][1] * 0.40 + t[1] * 0.60,
        lm[i][2] - 0.010,
      ];
    }
  }
  if (spec._indexThumbCircle) {
    const t = thb[3];
    lm[8] = [t[0], t[1], t[2]];
    lm[7] = [(t[0] + lm[6][0]) / 2, (t[1] + lm[6][1]) / 2, (t[2] + lm[6][2]) / 2];
  }
  if (spec._thumbDeep) {
    lm[4] = [-0.008, +0.030, +0.030];
    lm[3] = [+0.005, +0.020, +0.025];
  }
  if (spec._thumbDeepN) {
    lm[4] = [+0.012, +0.030, +0.030];
    lm[3] = [+0.018, +0.020, +0.020];
  }

  return lm.map(([x, y, z]) => ({ x, y, z }));
}

function _rotateZ(landmarks, deg) {
  const a = (deg * Math.PI) / 180;
  const c = Math.cos(a), s = Math.sin(a);
  return landmarks.map(([x, y, z]) => [x * c - y * s, x * s + y * c, z]);
}

function _translate(landmarks, [dx, dy, dz]) {
  return landmarks.map(([x, y, z]) => [x + dx, y + dy, z + dz]);
}

export function validateAgainstClassifier(letter, lm) {
  if (!lm || lm.length !== 21) return false;
  const px = (i) => lm[i].x, py = (i) => lm[i].y, pz = (i) => lm[i].z;
  const d3 = (a, b) => Math.hypot(px(a) - px(b), py(a) - py(b), pz(a) - pz(b));
  const palm = d3(0, 9) || 0.0001;
  const ang = (a, b, c) => {
    const abx = px(a) - px(b), aby = py(a) - py(b), abz = pz(a) - pz(b);
    const cbx = px(c) - px(b), cby = py(c) - py(b), cbz = pz(c) - pz(b);
    const ma = Math.hypot(abx, aby, abz), mc = Math.hypot(cbx, cby, cbz);
    const cos = (abx * cbx + aby * cby + abz * cbz) / (ma * mc || 0.0001);
    return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
  };
  const curl = (a, b, c, d) => (ang(a, b, c) + ang(b, c, d)) / 2;
  const IC = curl(5, 6, 7, 8),  MC = curl(9, 10, 11, 12);
  const RC = curl(13, 14, 15, 16), PC = curl(17, 18, 19, 20);
  const ext = (c) => c > 150;
  const crl = (c) => c < 120;
  const L = letter.toUpperCase();
  const rule = {
    A: () => crl(IC) && crl(MC) && crl(RC) && crl(PC),
    B: () => ext(IC) && ext(MC) && ext(RC) && ext(PC),
    C: () => !ext(IC) && !ext(MC) && !ext(RC) && !ext(PC) && !crl(IC),
    D: () => ext(IC) && !ext(MC),
    E: () => crl(IC) && crl(MC) && crl(RC) && crl(PC),
    F: () => !ext(IC) && ext(MC) && ext(RC) && ext(PC),
    G: () => ext(IC) && crl(MC),
    H: () => ext(IC) && ext(MC) && !ext(RC) && !ext(PC),
    I: () => !ext(IC) && !ext(MC) && !ext(RC) && ext(PC),
    J: () => !ext(IC) && !ext(MC) && !ext(RC) && ext(PC),
    K: () => ext(IC) && ext(MC) && !ext(RC) && !ext(PC),
    L: () => ext(IC) && crl(MC) && crl(RC),
    M: () => crl(IC) && crl(MC) && crl(RC) && crl(PC),
    N: () => crl(IC) && crl(MC) && crl(RC) && crl(PC),
    O: () => !ext(IC) && !ext(MC) && !ext(RC) && !ext(PC),
    P: () => ext(IC) && !ext(MC) && !ext(RC) && !ext(PC),
    Q: () => ext(IC) && !ext(MC) && !ext(RC) && !ext(PC),
    R: () => ext(IC) && ext(MC) && !ext(RC) && !ext(PC),
    S: () => crl(IC) && crl(MC) && crl(RC) && crl(PC),
    T: () => crl(IC) && crl(MC) && crl(RC) && crl(PC),
    U: () => ext(IC) && ext(MC) && !ext(RC) && !ext(PC),
    V: () => ext(IC) && ext(MC) && !ext(RC) && !ext(PC),
    W: () => ext(IC) && ext(MC) && ext(RC) && !ext(PC),
    X: () => !ext(IC) && !crl(IC) && crl(MC) && crl(RC) && crl(PC),
    Y: () => !ext(IC) && !ext(MC) && !ext(RC) && ext(PC),
    Z: () => ext(IC) && crl(MC),
  };
  const fn = rule[L];
  return fn ? !!fn() : true;
}
