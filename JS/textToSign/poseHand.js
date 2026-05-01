// Pure pose math for text-to-fingerspelling.
//
// Maps MediaPipe 21-point hand landmarks onto the rigged hand skeleton in
// hands.glb (Elbow → Wrist → finger chains, each 1..4). Tip bones (*4) and
// the Elbow stay at rest; the Wrist orientation also stays at rest.
//
// The math: for each non-tip finger bone, compute the world-space target
// direction from the corresponding landmark pair, transform it into the
// bone's parent's local space, then set bone.quaternion via
// Quaternion.setFromUnitVectors(restDir, targetDir). restDir is captured
// once at load from the model's rest pose so we don't assume any axis.

import * as THREE from 'three';

// Bone-name → MediaPipe landmark index pair (a, b). Direction = lm[b] - lm[a].
// Object insertion order matters: parents must come before children so that
// when we pose a bone, its parent's world matrix is already up-to-date.
export const BONE_LANDMARK_PAIRS = {
  Thumb1:  [1,  2],
  Thumb2:  [2,  3],
  Thumb3:  [3,  4],
  Index1:  [5,  6],
  Index2:  [6,  7],
  Index3:  [7,  8],
  Middle1: [9,  10],
  Middle2: [10, 11],
  Middle3: [11, 12],
  Ring1:   [13, 14],
  Ring2:   [14, 15],
  Ring3:   [15, 16],
  Pinky1:  [17, 18],
  Pinky2:  [18, 19],
  Pinky3:  [19, 20],
};

export const TIP_BONES = ['Thumb4', 'Index4', 'Middle4', 'Ring4', 'Pinky4'];
export const REST_BONES = [...TIP_BONES, 'Wrist', 'Elbow'];

export const ALL_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
// J and Z need motion paths in real ASL. We capture only the end pose for now.
// TODO: animate motion path for J and Z.
export const STATIC_END_POSE_LETTERS = ['J', 'Z'];

// Capture each bone's rest direction in its OWN local frame (the child's
// local-position direction, normalized). This is independent of any bind
// rotation the GLB rig may have on the bone — important here because the
// hands.glb rig encodes finger spread as rotations rather than position
// offsets, so every finger MCP has Q_bind ≠ identity.
//
// At pose time we set bone.quaternion = setFromUnitVectors(restDir, target),
// which REPLACES the bind quaternion. Using the bone-local rest direction
// (child.position direction) is the version of the math that gives the
// bone's WORLD direction the value we ask for; using a parent-local rest
// direction baked through Q_bind gives a different rotation that's wrong
// whenever Q_bind ≠ identity.
export function extractRestDirs(bones) {
  const rest = {};
  for (const name of Object.keys(BONE_LANDMARK_PAIRS)) {
    const bone = bones[name];
    if (!bone) continue;
    const child = bone.children.find((c) => c.isBone);
    if (!child) continue;
    rest[name] = child.position.clone().normalize();
  }
  return rest;
}

// Apply landmark pose to bones in place. Iterates parent-first so each child
// reads its parent's freshly-updated world quaternion.
//
// landmarks: 21-point array of {x,y,z}. MediaPipe worldLandmarks are meters
//   from wrist with Y-down; we flip Y by default so up matches three.js.
// options.mirror: mirror landmark X (use when the captured hand and the GLB
//   hand are different handedness).
// options.flipY: default true. Set false if your data is already Y-up.
export function poseHandFromLandmarks(landmarks, bones, restDirs, options = {}) {
  const { mirror = false, flipY = true } = options;
  const lm = landmarks.map((p) => new THREE.Vector3(
    mirror ? -p.x : p.x,
    flipY  ? -p.y : p.y,
    p.z,
  ));
  const tmpDir = new THREE.Vector3();
  const tmpQInv = new THREE.Quaternion();
  for (const [name, [a, b]] of Object.entries(BONE_LANDMARK_PAIRS)) {
    const bone = bones[name];
    const restDir = restDirs[name];
    if (!bone || !restDir) continue;
    tmpDir.copy(lm[b]).sub(lm[a]).normalize();
    if (bone.parent) {
      bone.parent.updateMatrixWorld(true);
      bone.parent.getWorldQuaternion(tmpQInv).invert();
      tmpDir.applyQuaternion(tmpQInv);
    }
    bone.quaternion.setFromUnitVectors(restDir, tmpDir);
  }
}

// Build per-letter local-space quaternion targets from a {letter: {landmarks}}
// map. Resets bones to rest before each letter so targets are reproducible
// regardless of the bones' current pose.
export function buildLetterTargets(letterData, bones, restDirs, options = {}) {
  const restQ = {};
  for (const name of Object.keys(BONE_LANDMARK_PAIRS)) {
    const bone = bones[name];
    if (bone) restQ[name] = bone.quaternion.clone();
  }
  const out = {};
  for (const [letter, data] of Object.entries(letterData)) {
    if (!data || !data.landmarks) continue;
    for (const name of Object.keys(restQ)) bones[name].quaternion.copy(restQ[name]);
    poseHandFromLandmarks(data.landmarks, bones, restDirs, options);
    const target = {};
    for (const name of Object.keys(BONE_LANDMARK_PAIRS)) {
      const bone = bones[name];
      if (bone) target[name] = bone.quaternion.clone();
    }
    out[letter] = target;
  }
  for (const name of Object.keys(restQ)) bones[name].quaternion.copy(restQ[name]);
  return out;
}
