"""PRISM -> paired Small/Large **PoC DEMO** generator.

목적: PRISM 한 샘플(subj001/take002.pkl)에서 계약(`PAIRED_SMALL_LARGE_DATASET_CONTRACT.md`)
형식의 Small(6-IMU) / Large(pelvis+15관절) 페어가 "어떻게 생겼는지"를 팀원에게 보여주기
위한 **구조·포맷 예시**다.

이것은 무엇이 아닌가 (반드시 읽을 것):
  * 캐노니컬 데이터셋이 아니다. `experimental_non_candidate`.
  * 물리적으로 검증된 산출물이 아니다. 프레임 좌표계는 PRISM world 원본이며
    SOMA_WORLD_FUR 변환/리샘플/SMPL FK/품질 게이트를 적용하지 않았다.
  * quality-gate PASS가 아니며 학습/평가 입력으로 쓰면 안 된다.
  * INTERNAL-ONLY. 외부/클라우드/공개 링크 업로드 금지.

설계 원칙:
  * PRISM이 직접 주는 배열은 실데이터로 채우고 provenance를 `source_derived`/`measured`로 기록.
  * 데모에서 유도한 값(specific force, angular velocity, chest proxy)은 `estimated`/`proxy`.
  * 데모가 만들지 않는 필드(대부분의 관절 회전/모멘트/파워/각운동량 등)는 0으로 채우지 않고
    NaN + valid_mask=false + provenance=`unavailable`.
  * 외부 pickle은 numpy 화이트리스트 제한 unpickler로만 읽는다(임의코드 실행 차단).

입력: ${SOMA_SOURCE_ROOT}/prism/subj001/take002.pkl (기본 ${SOMA_DATA_ROOT}/extracted/prism/...)
출력: ${SOMA_DATA_ROOT}/runs/experimental_generation_poc_demo/<run_id>/
둘 다 __main__에서 `soma_synth.pipeline.paths`로 정하므로, import만으로는 환경을 읽지 않는다.

은퇴한 데모다: 저장소 안에서 이 스크립트를 부르는 곳이 없다. 생성은 hold 아래 있으므로 승인 없이
실행하지 않는다.
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

WINDOW_START = 1000          # 파일럿 설계와 동일한 창 [1000, 2000)
T = 1000                     # 10 s @ 100 Hz
TARGET_RATE_HZ = 100

# 데모 전용 계약/버전 식별자 — 실제 v3 페어로 오인되지 않도록 일부러 분리.
CONTRACT_ID = "soma_paired_small_large_v3_POC_DEMO"
CONTRACT_VERSION = "0.0.0-poc-demo"

# 4.1 canonical sensor registry (Small)
SENSOR_CODES = ["chest", "wrist_l", "wrist_r", "foot_l", "foot_r", "head"]
# PRISM `imu` dict 키로의 매핑. chest는 PRISM에 없으므로 Pelvis를 proxy로 사용.
SENSOR_SOURCE = {
    "chest":   ("Pelvis",  "proxy"),        # PRISM 미제공 → Pelvis 대체(근사)
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
# PRISM imu_gt 가 직접 위치를 주는 관절만 채운다. 나머지는 unavailable(NaN+mask=false).
JOINT_GT_SOURCE = {
    "ankle_l": ("L_Foot",  "proxy"),         # foot 마커 ~ 발목 근사
    "ankle_r": ("R_Foot",  "proxy"),
    "knee_l":  ("L_Knee",  "source_derived"),
    "knee_r":  ("R_Knee",  "source_derived"),
    "wrist_l": ("L_Wrist", "source_derived"),
    "wrist_r": ("R_Wrist", "source_derived"),
    "neck":    ("Head",    "proxy"),          # head 마커 ~ 목 근사
}

GRAVITY = 9.81  # m/s^2, PRISM world +Y-up 가정 하의 데모 값


# --------------------------------------------------------------------------
# 1. 안전한 PRISM pickle 로딩 (numpy 전용 화이트리스트)
# --------------------------------------------------------------------------
class _NumpyOnlyUnpickler(pickle.Unpickler):
    """numpy 배열 재구성만 허용하고 그 외 모든 전역은 차단한다."""

    _ALLOWED = {"_reconstruct", "scalar", "ndarray", "dtype"}

    def find_class(self, module, name):
        if module.startswith("numpy") and name in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"BLOCKED non-numpy global: {module}.{name}")


def load_prism_safe(path: str) -> dict:
    with open(path, "rb") as fh:
        return _NumpyOnlyUnpickler(fh).load()


# --------------------------------------------------------------------------
# 2. 수학 헬퍼 (표준 공식; 데모 정확도)
# --------------------------------------------------------------------------
def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """[N,3,3] 회전행렬 -> [N,4] quaternion (w,x,y,z), 시간축 부호 연속성 강제."""
    R = np.asarray(R, dtype=np.float64)
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.zeros((R.shape[0], 4), dtype=np.float64)
    # 안정적 분기 (trace 기반 + 대각 최대 성분 기반)
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
    """[N,3,3] world 회전행렬 시퀀스 -> body-frame 각속도 [N,3] (deg/s).

    omega_body_hat = R^T dR/dt ; 중앙차분(경계는 전/후진차분).
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
    """f_sensor = R^T (a_world - g_world). PRISM world +Y-up 가정. m/s^2 (deg 아님)."""
    # accelerometer specific force = R^T (a_world - g_world), g_world = (0,-9.81,0)
    g_world = np.array([0.0, -GRAVITY, 0.0])
    a = np.asarray(acc_world, dtype=np.float64) - g_world
    R = np.asarray(R_world_from_sensor, dtype=np.float64)
    return np.einsum("nji,nj->ni", R, a)       # R^T @ a  (R[n].T @ a[n])


def canonical_json_bytes(obj) -> bytes:
    """RFC8785 근사: 정렬키, 공백없음, UTF-8, no BOM, no trailing newline, NaN 금지."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def f64_be_hex(x: float) -> str:
    x = 0.0 if x == 0.0 else float(x)          # -0.0 → +0.0
    return struct.pack(">d", x).hex()


# --------------------------------------------------------------------------
# 3. 생성
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

    # ---- 시간축 ----
    timestamps_s = (np.arange(T, dtype=np.float64) * dt)     # window-relative, exact 100 Hz
    source_interval_s = [WINDOW_START / fps, (WINDOW_START + T) / fps]
    ts_bytes = np.ascontiguousarray(timestamps_s, dtype="<f8").tobytes()
    timestamps_sha256 = sha256_hex(ts_bytes)

    # ---- canonicalization config (데모) + hash ----
    canon_cfg = {
        "kind": "poc_demo_canonicalization",
        "window_start": WINDOW_START, "frame_count": T, "target_rate_hz": TARGET_RATE_HZ,
        "frame_convention": "prism_world_native",  # SOMA_WORLD_FUR 변환 미적용
        "gravity_m_s2": GRAVITY,
        "notes": "no resample, no SMPL FK, no JCS, no quality gate",
    }
    canon_cfg_sha256 = sha256_hex(canonical_json_bytes(canon_cfg))

    # ---- pair_id (계약 3.1 순서를 데모 범위에서 근사) ----
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
        # 데모 confidence: 실측 유래는 0.6, proxy는 0.3 (임의, 설명용)
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

    # 데모가 만들지 않는 회전/각도/모멘트/파워 계열 → NaN + mask=false + unavailable
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

    # ---- Kinetics: PRISM insole 실측 GRF/CoP/contact (단위 원본-미검증) ----
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
    cop_valid_mask = (grf[:, 1] > 1e-6) | (grf[:, 0] != 0)   # 하중 있을 때만

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
        # identity (Small과 동일)
        "pair_id": pair_id,
        "timestamps_s": timestamps_s,
        "frame_count": np.int64(T),
    }

    # ---------------------------------------------------------------
    # 필드별 provenance 표 (manifest)
    # ---------------------------------------------------------------
    field_provenance = {
        # Small
        "small.imu_orientation": {"shape": [T, 6, 4], "unit": "quaternion(w,x,y,z)",
            "frame": "prism_world_native", "provenance": "source_derived",
            "note": "PRISM imu[*].ori_world (3x3) -> quaternion. chest 슬롯은 Pelvis proxy."},
        "small.imu_acceleration": {"shape": [T, 6, 3], "unit": "m/s^2(specific force)",
            "frame": "sensor", "provenance": "estimated",
            "note": "f=R^T(a_world-g); PRISM acc_world_filt 기반, lever-arm 무시, SOMA_WORLD_FUR 미변환."},
        "small.imu_angular_velocity": {"shape": [T, 6, 3], "unit": "deg/s", "frame": "sensor",
            "provenance": "estimated", "note": "ori_world 중앙차분에서 유도(자이로 원본 미사용)."},
        "small.sensor_codes[0]=chest": {"provenance": "proxy",
            "note": "PRISM chest IMU 없음 → Pelvis 대체. 나머지 5개 site는 source_derived."},
        # Large kinematics
        "large.pelvis_position": {"shape": [T, 3], "unit": "m", "frame": "prism_world_native",
            "provenance": "source_derived", "note": "imu_gt.Pelvis.pos_world."},
        "large.pelvis_orientation": {"shape": [T, 4], "unit": "quaternion", "frame": "prism_world_native",
            "provenance": "source_derived", "note": "imu_gt.Pelvis.ori_world -> quaternion."},
        "large.joint_centers_world": {"shape": [T, 15, 3], "unit": "m", "frame": "prism_world_native",
            "provenance": "mixed",
            "note": "7개 관절(ankle/knee/wrist l·r, neck)만 PRISM imu_gt로 채움(source_derived/proxy). "
                    "나머지 8개(mtp/hip/shoulder/elbow l·r)는 SMPL FK 필요 → NaN+mask=false+unavailable.",
            "per_joint_provenance": dict(zip(JOINT_CODES, joint_provenance))},
        "large.joint_rotations/angles/velocity": {"provenance": "unavailable",
            "note": "SMPL FK + JCS 규약 필요. 데모 미생성 → NaN + mask=false (0으로 채우지 않음)."},
        "large.com_*/angular_momentum*": {"provenance": "unavailable",
            "note": "전신 관성 모델 필요. 데모 미생성 → NaN + mask=false."},
        # Large kinetics
        "large.grf / grf_feet": {"shape_total": [T, 3], "shape_feet": [T, 2, 3],
            "unit": "PRISM_source_native_UNVERIFIED", "frame": "prism_world_native",
            "provenance": "measured",
            "note": "PRISM insole force_world. 단위 미검증(범위상 N 아님, 정규화 가능성). grf=L+R 합."},
        "large.cop / cop_feet": {"unit": "m(추정, PRISM native)", "frame": "prism_world_native ground(X,Z)",
            "provenance": "source_derived", "note": "insole CoP_world의 (X,Z)."},
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

**주의: 이것은 포맷/구조 예시(데모)입니다.** 캐노니컬 데이터셋도, 물리 검증 산출물도,
quality-gate PASS도 아닙니다. `INTERNAL-ONLY`. 학습/평가 입력으로 쓰지 마세요.

## 무엇인가
PRISM `subj001/take002.pkl`("Walking Square" 보행, 100 Hz)의 프레임 [{WINDOW_START},{WINDOW_START+T})
구간 {T}프레임을, paired Small/Large 계약 형식으로 매핑한 예시입니다.

## 파일
- `small.npz` — 6-IMU (chest, wrist_l, wrist_r, foot_l, foot_r, head)
- `large.npz` — pelvis + 15관절 kinematics + bilateral GRF/CoP
- `pair_manifest.json` — 필드별 shape/단위/frame/provenance 와 "적용하지 않은 파이프라인 단계"

## 실데이터 vs 데모 채움
- **실데이터(PRISM 유래):** IMU 방향(quaternion), pelvis 궤적/방향, knee/wrist 관절중심,
  insole GRF/CoP/contact.
- **데모 유도(estimated):** specific force, angular velocity(방향 미분), chest(=Pelvis proxy).
- **미생성(unavailable, NaN+mask=false, 0으로 안 채움):** mtp/hip/shoulder/elbow 관절중심,
  모든 관절 회전/JCS 각도/모멘트/파워, COM/각운동량.
- **미적용 단계:** SOMA_WORLD_FUR 변환, SMPL FK(전 관절), JCS, physics GRF, 역동역학, 품질게이트,
  insole force 단위 검증. (manifest `not_applied_pipeline_steps` 참조)

## 안전
외부 pickle은 numpy 화이트리스트 제한 unpickler로만 읽었습니다(임의코드 실행 차단).
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
