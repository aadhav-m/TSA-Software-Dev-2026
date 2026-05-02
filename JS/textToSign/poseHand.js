import * as THREE from 'three';

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
export const STATIC_END_POSE_LETTERS = ['J', 'Z'];

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
