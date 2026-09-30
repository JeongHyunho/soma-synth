"""PRISM -> paired Small/Large **PoC DEMO** generator.

Purpose: a **structure and format example** that shows teammates "what a pair looks like": a
Small (6-IMU) / Large (pelvis + 15 joints) pair in the format of the contract
(`PAIRED_SMALL_LARGE_DATASET_CONTRACT.md`), from one PRISM sample (subj001/take002.pkl).

What this is not (read this):
  * Not a canonical dataset. `experimental_non_candidate`.
  * Not a physically validated output. The frame is the native PRISM world, and no
    SOMA_WORLD_FUR conversion, resampling, SMPL FK or quality gate was applied.
  * Not a quality-gate PASS; must not be used as training or evaluation input.
  * INTERNAL-ONLY. No upload to external, cloud or public links.

Design principles:
  * Arrays PRISM gives directly are filled with real data and their provenance recorded as
    `source_derived`/`measured`.
  * Values derived in the demo (specific force, angular velocity, chest proxy) are `estimated`/`proxy`.
  * Fields the demo does not make (most joint rotations, moments, powers, angular momentum and so
    on) are not filled with 0 but with NaN + valid_mask=false + provenance=`unavailable`.
  * External pickles are read only with an unpickler restricted to a numpy whitelist (no
    arbitrary code execution).

Input: ${SOMA_SOURCE_ROOT}/prism/subj001/take002.pkl (default ${SOMA_DATA_ROOT}/extracted/prism/...)
Output: ${SOMA_DATA_ROOT}/runs/experimental_generation_poc_demo/<run_id>/
Both are resolved in __main__ with `soma_synth.pipeline.paths`, so importing reads no environment.

A retired demo: nothing in the repository calls this script. Generation is under hold, so it is not
run without approval.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from soma_synth.pipeline import paths

# --------------------------------------------------------------------------
# 0. Config
# --------------------------------------------------------------------------
#: The one take this demo reads, relative to the PRISM source folder (paths.source_dir("prism")).
SOURCE_TAKE = ("subj001", "take002.pkl")
# sha256 of the published PRISM release archive (lower case).
PRISM_ARCHIVE_SHA256 = "565082abbbce8df2459dffd711d58ea889c84270e90f77b75a2a552f798f7fb2"

RUN_ID = "prism-subj001-take002-poc-demo-v1"

WINDOW_START = 1000          # the same window as the pilot design [1000, 2000)
T = 1000                     # 10 s @ 100 Hz
TARGET_RATE_HZ = 100

# Demo-only contract/version identifiers — kept apart on purpose so they are not mistaken for a real v3 pair.
CONTRACT_ID = "soma_paired_small_large_v3_POC_DEMO"
CONTRACT_VERSION = "0.0.0-poc-demo"

# 4.1 canonical sensor registry (Small)
SENSOR_CODES = ["chest", "wrist_l", "wrist_r", "foot_l", "foot_r", "head"]
# Mapping to the keys of PRISM's `imu` dict. PRISM has no chest, so Pelvis is used as a proxy.
SENSOR_SOURCE = {
    "chest":   ("Pelvis",  "proxy"),        # not provided by PRISM → Pelvis instead (approximation)
    "wrist_l": ("L_Wrist", "source_derived"),
    "wrist_r": ("R_Wrist", "source_derived"),
    "foot_l":  ("L_Foot",  "source_derived"),
    "foot_r":  ("R_Foot",  "source_derived"),
    "head":    ("Head",    "source_derived"),
}

# 5.1 canonical 15-joint registry (Large)
JOINT_CODES = [
    "mtp_l", "mtp_r", "ankle_l", "ankle_r", "knee_l", "knee_r",
    "hip_l", "hip_r", "shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
    "wrist_l", "wrist_r", "neck",
]
# Only the joints whose position PRISM imu_gt gives directly are filled. The rest are unavailable (NaN+mask=false).
JOINT_GT_SOURCE = {
    "ankle_l": ("L_Foot",  "proxy"),         # foot marker ~ ankle, approximately
    "ankle_r": ("R_Foot",  "proxy"),
    "knee_l":  ("L_Knee",  "source_derived"),
    "knee_r":  ("R_Knee",  "source_derived"),
    "wrist_l": ("L_Wrist", "source_derived"),
    "wrist_r": ("R_Wrist", "source_derived"),
    "neck":    ("Head",    "proxy"),          # head marker ~ neck, approximately
}

GRAVITY = 9.81  # m/s^2, a demo value assuming PRISM world is +Y-up


# --------------------------------------------------------------------------
# 1. Safe PRISM pickle loading (numpy-only whitelist)
# --------------------------------------------------------------------------
class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Allows only numpy array reconstruction and blocks every other global."""

    _ALLOWED = {"_reconstruct", "scalar", "ndarray", "dtype"}

    def find_class(self, module, name):
        if module.startswith("numpy") and name in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"BLOCKED non-numpy global: {module}.{name}")


def load_prism_safe(path: str) -> dict:
    with open(path, "rb") as fh:
        return _NumpyOnlyUnpickler(fh).load()


# --------------------------------------------------------------------------
# 2. Math helpers (standard formulas; demo accuracy)
# --------------------------------------------------------------------------
def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """[N,3,3] rotation matrices -> [N,4] quaternion (w,x,y,z), with sign continuity enforced over time."""
    R = np.asarray(R, dtype=np.float64)
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.zeros((R.shape[0], 4), dtype=np.float64)
    # stable branches (trace-based + largest-diagonal-based)
    for i in range(R.shape[0]):
        Ri = R[i]
        t = tr[i]
        if t > 0.0:
            s = np.sqrt(t + 1.0) * 2.0
            w = 0.25 * s
            x = (Ri[2, 1] - Ri[1, 2]) / s
            y = (Ri[0, 2] - Ri[2, 0]) / s
            z = (Ri[1, 0] - Ri[0, 1]) / s
        elif Ri[0, 0] > Ri[1, 1] and Ri[0, 0] > Ri[2, 2]:
            s = np.sqrt(1.0 + Ri[0, 0] - Ri[1, 1] - Ri[2, 2]) * 2.0
            w = (Ri[2, 1] - Ri[1, 2]) / s
            x = 0.25 * s
            y = (Ri[0, 1] + Ri[1, 0]) / s
            z = (Ri[0, 2] + Ri[2, 0]) / s
        elif Ri[1, 1] > Ri[2, 2]:
            s = np.sqrt(1.0 + Ri[1, 1] - Ri[0, 0] - Ri[2, 2]) * 2.0
            w = (Ri[0, 2] - Ri[2, 0]) / s
            x = (Ri[0, 1] + Ri[1, 0]) / s
            y = 0.25 * s
            z = (Ri[1, 2] + Ri[2, 1]) / s
        else:
            s = np.sqrt(1.0 + Ri[2, 2] - Ri[0, 0] - Ri[1, 1]) * 2.0
            w = (Ri[1, 0] - Ri[0, 1]) / s
            x = (Ri[0, 2] + Ri[2, 0]) / s
            y = (Ri[1, 2] + Ri[2, 1]) / s
            z = 0.25 * s
        q[i] = (w, x, y, z)
    # normalize
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # sign continuity
    for i in range(1, q.shape[0]):
        if np.dot(q[i], q[i - 1]) < 0.0:
            q[i] = -q[i]
    return q


def angular_velocity_deg_s(R: np.ndarray, dt: float) -> np.ndarray:
    """[N,3,3] sequence of world rotation matrices -> body-frame angular velocity [N,3] (deg/s).

    omega_body_hat = R^T dR/dt ; central differences (forward/backward at the ends).
    """
    R = np.asarray(R, dtype=np.float64)
    N = R.shape[0]
    dR = np.empty_like(R)
    dR[1:-1] = (R[2:] - R[:-2]) / (2.0 * dt)
    dR[0] = (R[1] - R[0]) / dt
    dR[-1] = (R[-1] - R[-2]) / dt
    omega = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        W = R[i].T @ dR[i]              # skew-symmetric (approx)
        omega[i] = (W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1])
        omega[i] *= 0.5
    return np.degrees(omega)


def specific_force(acc_world: np.ndarray, R_world_from_sensor: np.ndarray) -> np.ndarray:
    """f_sensor = R^T (a_world - g_world). Assumes PRISM world is +Y-up. m/s^2 (not deg)."""
    # accelerometer specific force = R^T (a_world - g_world), g_world = (0,-9.81,0)
    g_world = np.array([0.0, -GRAVITY, 0.0])
    a = np.asarray(acc_world, dtype=np.float64) - g_world
    R = np.asarray(R_world_from_sensor, dtype=np.float64)
    return np.einsum("nji,nj->ni", R, a)       # R^T @ a  (R[n].T @ a[n])


def canonical_json_bytes(obj) -> bytes:
    """Approximates RFC8785: sorted keys, no whitespace, UTF-8, no BOM, no trailing newline, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def f64_be_hex(x: float) -> str:
    x = 0.0 if x == 0.0 else float(x)          # -0.0 → +0.0
    return struct.pack(">d", x).hex()


# --------------------------------------------------------------------------
# 3. Generation
# --------------------------------------------------------------------------
def sl(a):
    """window slice + float32 view helper."""
    return np.asarray(a[WINDOW_START:WINDOW_START + T])


def build(source_pkl: str):
    """`source_pkl` is the take SOURCE_TAKE names under the PRISM source folder."""
    raw = load_prism_safe(source_pkl)
    fps = int(raw["info"]["data_info"]["fps"])
    assert fps == TARGET_RATE_HZ, f"demo expects native 100 Hz, got {fps}"
    dt = 1.0 / TARGET_RATE_HZ
    imu = raw["imu"]
    imu_gt = raw["imu_gt"]
    insole = raw["insole"]

    # ---- time axis ----
    timestamps_s = (np.arange(T, dtype=np.float64) * dt)     # window-relative, exact 100 Hz
    source_interval_s = [WINDOW_START / fps, (WINDOW_START + T) / fps]
    ts_bytes = np.ascontiguousarray(timestamps_s, dtype="<f8").tobytes()
    timestamps_sha256 = sha256_hex(ts_bytes)

    # ---- canonicalization config (demo) + hash ----
    canon_cfg = {
        "kind": "poc_demo_canonicalization",
        "window_start": WINDOW_START, "frame_count": T, "target_rate_hz": TARGET_RATE_HZ,
        "frame_convention": "prism_world_native",  # SOMA_WORLD_FUR conversion not applied
        "gravity_m_s2": GRAVITY,
        "notes": "no resample, no SMPL FK, no JCS, no quality gate",
    }
    canon_cfg_sha256 = sha256_hex(canonical_json_bytes(canon_cfg))

    # ---- pair_id (approximates the order of contract 3.1 within the demo's scope) ----
    subject_id = "prism_subj001"
    trial_id = "take002"
    preimage = [
        "prism",                                  # source_name
        "prism-archive-565082ab-v1",              # source_version: archive id (sha256 prefix)
        PRISM_ARCHIVE_SHA256,                     # source_asset_sha256
        subject_id, trial_id,
        f64_be_hex(source_interval_s[0]),
        f64_be_hex(source_interval_s[1]),
        canon_cfg_sha256, CONTRACT_ID, CONTRACT_VERSION,
        str(TARGET_RATE_HZ),
    ]
    pair_id = sha256_hex(canonical_json_bytes(preimage))

    # ---------------------------------------------------------------
    # SMALL (6-IMU)
    # ---------------------------------------------------------------
    imu_orientation = np.zeros((T, 6, 4), dtype=np.float32)
    imu_acceleration = np.zeros((T, 6, 3), dtype=np.float32)
    imu_angular_velocity = np.zeros((T, 6, 3), dtype=np.float32)
    imu_valid_mask = np.zeros((T, 6), dtype=bool)
    imu_confidence = np.zeros((T, 6), dtype=np.float32)
    mount_id = np.array([f"poc_demo_mount::{c}" for c in SENSOR_CODES], dtype=object)

    for j, code in enumerate(SENSOR_CODES):
        src_key, prov = SENSOR_SOURCE[code]
        site = imu[src_key]
        R = sl(site["ori_world"])                       # [T,3,3]
        acc_world = sl(site["acc_world_filt"])          # [T,3]
        imu_orientation[:, j, :] = rotmat_to_quat_wxyz(R).astype(np.float32)
        imu_acceleration[:, j, :] = specific_force(acc_world, R).astype(np.float32)
        imu_angular_velocity[:, j, :] = angular_velocity_deg_s(R, dt).astype(np.float32)
        imu_valid_mask[:, j] = True
        # demo confidence: 0.6 for measured-derived, 0.3 for proxy (arbitrary, illustrative)
        imu_confidence[:, j] = 0.3 if prov == "proxy" else 0.6

    small = {
        "sensor_codes": np.array(SENSOR_CODES, dtype=object),
        "imu_orientation": imu_orientation,
        "imu_acceleration": imu_acceleration,
        "imu_angular_velocity": imu_angular_velocity,
        "imu_valid_mask": imu_valid_mask,
        "imu_confidence": imu_confidence,
        "mount_id": mount_id,
        "small_mode": "ideal",
        "pair_id": pair_id,
        "timestamps_s": timestamps_s,
        "frame_count": np.int64(T),
    }

    # ---------------------------------------------------------------
    # LARGE (pelvis + 15 joints)
    # ---------------------------------------------------------------
    pelvis_position = sl(imu_gt["Pelvis"]["pos_world"]).astype(np.float32)
    pelvis_orientation = rotmat_to_quat_wxyz(sl(imu_gt["Pelvis"]["ori_world"])).astype(np.float32)

    NaN = np.float32(np.nan)
    joint_centers_world = np.full((T, 15, 3), NaN, dtype=np.float32)
    joint_valid_mask = np.zeros((T, 15), dtype=bool)
    joint_provenance = ["unavailable"] * 15
    for j, code in enumerate(JOINT_CODES):
        if code in JOINT_GT_SOURCE:
            gt_key, prov = JOINT_GT_SOURCE[code]
            joint_centers_world[:, j, :] = sl(imu_gt[gt_key]["pos_world"]).astype(np.float32)
            joint_valid_mask[:, j] = True
            joint_provenance[j] = prov

    # rotations/angles/moments/powers the demo does not make → NaN + mask=false + unavailable
    joint_rotations = np.full((T, 15, 4), NaN, dtype=np.float32)
    joint_angles_jcs_deg = np.full((T, 15, 3), NaN, dtype=np.float32)
    joint_angular_velocity_jcs_deg_s = np.full((T, 15, 3), NaN, dtype=np.float32)
    joint_component_valid_mask = np.zeros((T, 15, 3), dtype=bool)
    joint_confidence = np.zeros((T, 15), dtype=np.float32)
    com_position = np.full((T, 3), NaN, dtype=np.float32)
    com_velocity = np.full((T, 3), NaN, dtype=np.float32)
    com_acceleration = np.full((T, 3), NaN, dtype=np.float32)
    whole_body_angular_momentum = np.full((T, 3), NaN, dtype=np.float32)
    whole_body_angular_momentum_rate = np.full((T, 3), NaN, dtype=np.float32)

    # ---- Kinetics: PRISM insole measured GRF/CoP/contact (native units, unverified) ----
    grf_feet = np.stack([sl(insole["L_Foot"]["force_world"]),
                         sl(insole["R_Foot"]["force_world"])], axis=1).astype(np.float32)  # [T,2,3]
    grf = sl(insole["combined"]["force_world"]).astype(np.float32)                          # [T,3]
    # CoP world [T,3] → ground (X,Z)
    cop_l = sl(insole["L_Foot"]["CoP_world"])[:, [0, 2]]
    cop_r = sl(insole["R_Foot"]["CoP_world"])[:, [0, 2]]
    cop_feet = np.stack([cop_l, cop_r], axis=1).astype(np.float32)                          # [T,2,2]
    cop = sl(insole["combined"]["CoP_world"])[:, [0, 2]].astype(np.float32)                 # [T,2]
    foot_contact_mask = np.stack([sl(insole["L_Foot"]["contacts"])[:, 0],
                                  sl(insole["R_Foot"]["contacts"])[:, 0]], axis=1)          # [T,2] bool
    grf_valid_mask = np.ones((T,), dtype=bool)
    cop_valid_mask = (grf[:, 1] > 1e-6) | (grf[:, 0] != 0)   # only under load

    large = {
        "joint_codes": np.array(JOINT_CODES, dtype=object),
        "pelvis_position": pelvis_position,
        "pelvis_orientation": pelvis_orientation,
        "joint_centers_world": joint_centers_world,
        "joint_rotations": joint_rotations,
        "joint_angles_jcs_deg": joint_angles_jcs_deg,
        "joint_angular_velocity_jcs_deg_s": joint_angular_velocity_jcs_deg_s,
        "joint_valid_mask": joint_valid_mask,
        "joint_component_valid_mask": joint_component_valid_mask,
        "joint_confidence": joint_confidence,
        "com_position": com_position,
        "com_velocity": com_velocity,
        "com_acceleration": com_acceleration,
        "whole_body_angular_momentum": whole_body_angular_momentum,
        "whole_body_angular_momentum_rate": whole_body_angular_momentum_rate,
        # kinetics (SOMA bilateral)
        "grf": grf,
        "cop": cop,
        "grf_feet": grf_feet,
        "cop_feet": cop_feet,
        "foot_contact_mask": foot_contact_mask,
        "grf_valid_mask": grf_valid_mask,
        "cop_valid_mask": cop_valid_mask,
        # identity (the same as Small)
        "pair_id": pair_id,
        "timestamps_s": timestamps_s,
        "frame_count": np.int64(T),
    }

    # ---------------------------------------------------------------
    # per-field provenance table (manifest)
    # ---------------------------------------------------------------
    field_provenance = {
        # Small
        "small.imu_orientation": {"shape": [T, 6, 4], "unit": "quaternion(w,x,y,z)",
            "frame": "prism_world_native", "provenance": "source_derived",
            "note": "PRISM imu[*].ori_world (3x3) -> quaternion. The chest slot is a Pelvis proxy."},
        "small.imu_acceleration": {"shape": [T, 6, 3], "unit": "m/s^2(specific force)",
            "frame": "sensor", "provenance": "estimated",
            "note": "f=R^T(a_world-g); from PRISM acc_world_filt, lever arm ignored, not converted to SOMA_WORLD_FUR."},
        "small.imu_angular_velocity": {"shape": [T, 6, 3], "unit": "deg/s", "frame": "sensor",
            "provenance": "estimated", "note": "Derived from central differences of ori_world (the raw gyro is not used)."},
        "small.sensor_codes[0]=chest": {"provenance": "proxy",
            "note": "PRISM has no chest IMU → Pelvis instead. The other 5 sites are source_derived."},
        # Large kinematics
        "large.pelvis_position": {"shape": [T, 3], "unit": "m", "frame": "prism_world_native",
            "provenance": "source_derived", "note": "imu_gt.Pelvis.pos_world."},
        "large.pelvis_orientation": {"shape": [T, 4], "unit": "quaternion", "frame": "prism_world_native",
            "provenance": "source_derived", "note": "imu_gt.Pelvis.ori_world -> quaternion."},
        "large.joint_centers_world": {"shape": [T, 15, 3], "unit": "m", "frame": "prism_world_native",
            "provenance": "mixed",
            "note": "Only 7 joints (ankle/knee/wrist l·r, neck) are filled from PRISM imu_gt (source_derived/proxy). "
                    "The other 8 (mtp/hip/shoulder/elbow l·r) need SMPL FK → NaN+mask=false+unavailable.",
            "per_joint_provenance": dict(zip(JOINT_CODES, joint_provenance))},
        "large.joint_rotations/angles/velocity": {"provenance": "unavailable",
            "note": "Needs SMPL FK + the JCS convention. Not made by the demo → NaN + mask=false (not filled with 0)."},
        "large.com_*/angular_momentum*": {"provenance": "unavailable",
            "note": "Needs a whole-body inertial model. Not made by the demo → NaN + mask=false."},
        # Large kinetics
        "large.grf / grf_feet": {"shape_total": [T, 3], "shape_feet": [T, 2, 3],
            "unit": "PRISM_source_native_UNVERIFIED", "frame": "prism_world_native",
            "provenance": "measured",
            "note": "PRISM insole force_world. Units unverified (the range is not N; possibly normalised). grf = L+R sum."},
        "large.cop / cop_feet": {"unit": "m(estimated, PRISM native)", "frame": "prism_world_native ground(X,Z)",
            "provenance": "source_derived", "note": "(X,Z) of insole CoP_world."},
        "large.foot_contact_mask": {"shape": [T, 2], "provenance": "source_derived",
            "note": "insole contacts[:,0] (L,R)."},
    }

    manifest = {
        "distribution_scope": "internal_only",
        "artifact_class": "experimental_non_candidate",
        "artifact_label": "POC DEMO — format illustration only. NOT physically validated. "
                          "NOT a canonical dataset. NOT quality-gate PASS. Do not train/evaluate on this.",
        "excluded_modalities": ["emg"],
        "excluded_modality_reasons": {"emg": "not_in_synthetic_artifact_scope"},
        "run_id": RUN_ID,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "source_name": "prism", "subject_id": subject_id, "trial_id": trial_id,
            "trial_name": raw["info"]["data_info"]["trial_name"],
            "source_asset_sha256": PRISM_ARCHIVE_SHA256,
            "source_relative_path": "subj001/take002.pkl",
            "native_frames_total": int(next(iter(imu.values()))["ori_world"].shape[0]),
            "native_rate_hz": fps,
        },
        "window": {"window_start": WINDOW_START, "frame_count": T,
                   "source_interval_s": source_interval_s},
        "identity": {
            "pair_id": pair_id, "contract_id": CONTRACT_ID, "contract_version": CONTRACT_VERSION,
            "timestamps_sha256": timestamps_sha256, "target_rate_hz": TARGET_RATE_HZ,
            "canonicalization_config_sha256": canon_cfg_sha256,
        },
        "canonicalization_config": canon_cfg,
        "provenance_enum": ["measured", "source_derived", "estimated", "proxy", "unavailable"],
        "field_provenance": field_provenance,
        "not_applied_pipeline_steps": [
            "SOMA_WORLD_FUR frame conversion", "continuous full-trial reconstruction",
            "100Hz canonical resampling (native already 100Hz)", "SMPL forward kinematics for all 15 joints",
            "ISB/JCS joint-angle convention", "physics-informed hybrid GRF", "inverse dynamics (moments/powers)",
            "quality gates (PHYSICS_QUALITY_GATES.md)", "unit normalization/verification for insole force",
        ],
        "safety": {"pickle_loader": "numpy-whitelist restricted unpickler (no arbitrary code execution)"},
    }
    return small, large, manifest


def write_outputs(out_dir: str, small, large, manifest):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "small.npz"), **small)
    np.savez(os.path.join(out_dir, "large.npz"), **large)
    with open(os.path.join(out_dir, "pair_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    readme = f"""# PRISM → Small/Large PoC DEMO  ({RUN_ID})

**Note: this is a format/structure example (a demo).** It is not a canonical dataset, not a
physically validated output and not a quality-gate PASS. `INTERNAL-ONLY`. Do not use it as training
or evaluation input.

## What it is
An example mapping the {T} frames [{WINDOW_START},{WINDOW_START+T}) of PRISM `subj001/take002.pkl`
("Walking Square" gait, 100 Hz) into the paired Small/Large contract format.

## Files
- `small.npz` — 6-IMU (chest, wrist_l, wrist_r, foot_l, foot_r, head)
- `large.npz` — pelvis + 15-joint kinematics + bilateral GRF/CoP
- `pair_manifest.json` — per-field shape/unit/frame/provenance and "pipeline steps not applied"

## Real data vs demo fill
- **Real data (from PRISM):** IMU orientation (quaternion), pelvis trajectory/orientation, knee/wrist
  joint centres, insole GRF/CoP/contact.
- **Derived by the demo (estimated):** specific force, angular velocity (orientation derivative),
  chest (= Pelvis proxy).
- **Not made (unavailable, NaN+mask=false, not filled with 0):** mtp/hip/shoulder/elbow joint centres,
  all joint rotations/JCS angles/moments/powers, COM/angular momentum.
- **Steps not applied:** SOMA_WORLD_FUR conversion, SMPL FK (all joints), JCS, physics GRF, inverse
  dynamics, quality gates, insole force unit verification. (see manifest `not_applied_pipeline_steps`)

## Safety
External pickles were read only with an unpickler restricted to a numpy whitelist (no arbitrary
code execution).
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme)


def main() -> int:
    try:
        prism = paths.source_dir("prism")
        out_dir = paths.lineage_dir(RUN_ID)
        paths.check_output_dir(out_dir, inputs=[prism])
    except paths.PathConfigError as error:
        print(error)
        return 2
    small, large, manifest = build(str(prism.joinpath(*SOURCE_TAKE)))
    write_outputs(str(out_dir), small, large, manifest)
    print("OK ->", out_dir)
    print("pair_id:", manifest["identity"]["pair_id"])
    print("small keys:", list(small.keys()))
    print("large keys:", list(large.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
