"""PRISM -> paired Small/Large **물리적으로 충실한(faithful) 참조** 생성기 (독립 replica).

이 스크립트는 상위 프로젝트의 정식 PRISM experimental pilot 설계
(`configs/experiments/prism_experimental_pilot_v1.json`)의 알고리즘/pin을 그대로 따라,
독립적으로 동일한 faithful 참조 페어를 생성한다. 정식 파이프라인 코드는 건드리지 않는다.

정식 접근(=현실 제약에 맞춤). 필드명은 업데이트된 qmd 스펙(small/00_sensors·01_streams, large/02_kinematics) 준수:
  * SMPL 신체모델 파일 없이도 돌도록 **관절 '위치'가 아니라 '회전/상대회전'** 기반이다.
    - SMPL-24 pose chain FK로 글로벌 관절 회전을 poses에서 복원(모델 불필요).
    - 17관절(골반 제외)은 SMPL local(부모 대비) 회전을 joint_rotation[T,18,4]로 담는다(0=골반 global).
  * Small 8채널 통합 IMU: 본체 6(back_T4·wrist_l/r·shank_l/r·occiput) + 인솔 2(foot_l/r).
    - 본체 5개는 PRISM `imu_gt`, back_T4는 SMPL spine3(관절9)+pelvis→head α=2/3 proxy.
    - 인솔 2채널도 orientation 제공(중력기준 자세·heading 드리프트, 자기계 없음). ideal이라 GT로
      채우고 imu_orientation_absolute_heading=False로 실제 한계를 표기.
    - specific force f = R^T (a_world - g),  g=[0,0,-9.80665] (PRISM world = Z-up).
    - angular velocity = 인접 방향의 rotation-log(SO3) / dt (deg/s).
  * 프레임은 PRISM world(Z-up) = 갱신된 spec G(large/99_conventions.qmd: Z-up·PRISM world 정렬). 별도 변환 불필요.
  * PRISM insole(force/CoP/contact)는 **별도 development_reference.npz**(measured/source_derived).
    계약상 measured kinetics는 synthetic namespace와 분리(§6.5) → Large에 넣지 않는다.
  * Large 골반 병진은 `root_velocity`[T,3] m/s(프레임간 변위/dt)로 저장(spec). 절대 위치는
    pelvis_position_world_aux(참고)로 병기하며 `pos[t]=pos[0]+dt·cumsum(root_velocity)`로 복원.
  * 전체 take(13160f)에서 미분/FK 후 window[1000,2000)로 슬라이스(경계 아티팩트 방지).

한계: 관절 '중심 위치'(joint_centers_world)와 ISB JCS 각도는 SMPL 모델/ISB 규약이 필요해
  생성하지 않는다(unavailable). physics GRF/역동역학은 DETERMINISTIC_EXECUTION_CONFIG_HOLD로 금지.
  따라서 이건 여전히 `experimental_non_candidate`이며 quality-gate PASS가 아니다. INTERNAL-ONLY.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from datetime import datetime, timezone

import importlib.util as _ilu
import sys

import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation

# Anthro (anthropometry) deliverable helper — sibling module in this directory.
_ANTHRO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anthro_smpl.py")
_anthro_spec = _ilu.spec_from_file_location("anthro_smpl", _ANTHRO_PATH)
anthro_smpl = _ilu.module_from_spec(_anthro_spec)
_anthro_spec.loader.exec_module(anthro_smpl)

# The reduced model (smpl18.reduce, through the pipeline module): anthro_smpl has already put
# packages/smpl18/src on the path; src/ goes on beside it, as the AMASS generator does.
_SRC_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
from soma_synth.pipeline import paths, reduced_model  # noqa: E402

# --------------------------------------------------------------------------
# 0. Pins (정식 config prism_experimental_pilot_v1.json 과 동일)
# --------------------------------------------------------------------------
# Standalone single-take (subj001/take002) golden bundle: written to $SOMA_POC_OUT_DIR, else
# paths.lineage_dir(RUN_ID) under $SOMA_DATA_ROOT; the source is paths.source_dir("prism").
# Both are resolved in __main__, so importing this module reads no environment.
RUN_ID = "prism-subj001-take002-faithful-code-v2"

TARGET_RATE_HZ = 100
GRAVITY_PRISM_WORLD = np.array([0.0, 0.0, -9.80665], dtype=np.float64)  # Z-up
CHEST_ALPHA = 2.0 / 3.0
CHEST_SPINE3_JOINT = 9
FILTER_ORDER, FILTER_CUTOFF_HZ, FILTER_FS = 4, 7.0, 100.0
FILTER_PADLEN, FILTER_PADTYPE = 15, "odd"

CONTRACT_ID = "soma_paired_small_large_v3_POC_FAITHFUL_DOCX_CODE"
CONTRACT_VERSION = "0.0.1-poc-faithful-docx-code"

DATA_SPECIFICATION_DOCX = "01_Data_Specifipication.docx"
DATA_SPECIFICATION_DOCX_SHA256 = "ddceed2dbd6e1ba515d9a12b8337c8d2643494f86305afb31b617226e912a1a0"
DOCX_LARGE_FLAT_CODE_ALIASES = {
    "joint_positions": "pelvis_position",
    "joint_rotations": "regional_orientation_parent_from_child",
}
DOCX_LARGE_QUALIFIED_CODE_MAP = {
    "joint_positions(root)": "joint_positions",
    "joint_rotations(root)": "pelvis_orientation",
    "joint_rotations(joints)": "joint_rotations",
}

SMPL24_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)

# Small 8채널 통합 IMU (small/00_sensors.qmd, small/01_streams.qmd).
#   본체 6(back_T4..occiput): 9-DOF → orientation+accel+gyro.
#   인솔 2(foot_l/r): 6축 → accel+gyro만, 융합 사원수(orientation) 없음(자기계 미포함).
SENSOR_CODES = ["back_T4", "wrist_l", "wrist_r", "shank_l", "shank_r", "occiput", "foot_l", "foot_r"]
N_SENSORS = 8
N_BODY = 6  # ch0..ch5 = 9-DOF(사원수 있음). ch6/7(foot) = 6축(사원수 없음).
# 본체 IMU site -> PRISM imu_gt 키. back_T4는 SMPL spine3(별도 처리).
#   shank(정강이 상부, 무릎쪽 3/4)은 PRISM 무릎부 GT(L_Knee/R_Knee)를 근위 하퇴 proxy로 사용.
SENSOR_IMU_GT = {
    "wrist_l": "L_Wrist", "wrist_r": "R_Wrist",
    "shank_l": "L_Knee", "shank_r": "R_Knee", "occiput": "Head",
}
# 인솔(Moticon OpenGo) 6축 IMU — 좌/우, 발 GT(L_Foot/R_Foot) 기반. orientation 없음.
FOOT_CODES = ["foot_l", "foot_r"]
FOOT_IMU_GT = {"foot_l": "L_Foot", "foot_r": "R_Foot"}

# Large 운동학(large/02_kinematics.qmd): 골반(root)+17관절 = smpl18이 남기는 18관절.
#   제외: spine1(3)·spine2(6)·collar(13,14)·hand(22,23). joint_names는 SMPL 인덱스 순서. 목록은 패키지 소유.
JOINT18_NAMES = list(reduced_model.JOINT18_NAMES)
SMPL18_INDICES = np.array(reduced_model.KEEP18, np.int64)

# Reduced-model QC (spec sec-anthro-fit): each fixed joint's downstream affected joints, and the
# p95 position-residual threshold above which the fixed assumption is deemed broken and the joint is
# flagged as a Large-motion promotion candidate. Collars -> their arm; spine1/spine2 -> the trunk.
PROMOTE_TAU_P95_M = 0.05
FIXED_JOINT_DOWNSTREAM = {
    "left_collar": [16, 18, 20],     # left_shoulder, left_elbow, left_wrist
    "right_collar": [17, 19, 21],    # right_shoulder, right_elbow, right_wrist
    "spine1": [9, 12, 15],           # spine3, neck, head
    "spine2": [9, 12, 15],
}


class TakeSpec:
    """One PRISM take to generate: identity, source, gender, and window.

    Plain class (not a dataclass) so it constructs correctly even when this module is loaded via
    importlib without a sys.modules entry — dataclass string-annotation resolution needs one, and
    both the tests and the batch runner import this module that way.
    """

    def __init__(self, subject_id, take_id, source_pkl, gender, window=None, relative_path=""):
        self.subject_id = subject_id            # e.g. "prism_subj001"
        self.take_id = take_id                  # e.g. "take002"
        self.source_pkl = source_pkl            # absolute path to the take .pkl
        self.gender = gender                    # "M" / "F" (from info.subj_info.gender)
        self.window = window                    # (start, end_exclusive); None => full take
        self.relative_path = relative_path      # e.g. "extracted/prism/subj001/take002.pkl"

    def __repr__(self):
        return "TakeSpec(%s/%s, gender=%s, window=%r)" % (
            self.subject_id, self.take_id, self.gender, self.window)


# --------------------------------------------------------------------------
# 1. 안전 로딩
# --------------------------------------------------------------------------
class _NumpyOnlyUnpickler(pickle.Unpickler):
    # `_frombuffer` reconstructs an ndarray from an in-band (BYTEARRAY8) buffer for PRISM's two
    # protocol-5 takes (subj001/take011, subj005/take022). It is a pure array reconstruction (no
    # code execution) — the same numpy global the governed source audit's protocol-5 strategy
    # (symbolic_pickle_vm_v2) verified as safe. Out-of-band buffers are never passed to load().
    _ALLOWED = {"_reconstruct", "scalar", "ndarray", "dtype", "_frombuffer"}

    def find_class(self, module, name):
        if module.startswith("numpy") and name in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"BLOCKED non-numpy global: {module}.{name}")


def load_prism_safe(path):
    with open(path, "rb") as fh:
        return _NumpyOnlyUnpickler(fh).load()


# --------------------------------------------------------------------------
# 2. 헬퍼
# --------------------------------------------------------------------------
def project_so3(R):
    """SVD로 가장 가까운 정규직교 회전행렬로 투영(det=+1 보장)."""
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    det = np.linalg.det(Rp)
    # det<0면 마지막 특이벡터 반전
    neg = det < 0
    if np.any(neg):
        U2 = U.copy()
        U2[neg, :, -1] *= -1.0
        Rp = U2 @ Vt
    return Rp


def rotmat_to_quat_wxyz(R):
    """[...,3,3] -> [...,4] (w,x,y,z), 시간축 부호 연속성."""
    q_xyzw = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat().reshape(R.shape[:-2] + (4,))
    q = np.concatenate((q_xyzw[..., 3:4], q_xyzw[..., :3]), axis=-1)
    if q.ndim == 2:  # [N,4] 시간열에 부호 연속성
        for i in range(1, q.shape[0]):
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
    return q


def rotvec_to_quat_wxyz(rotvec):
    """[...,3] axis-angle -> [...,4] (w,x,y,z), 2D면 시간축 부호 연속성."""
    q_xyzw = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_quat().reshape(rotvec.shape[:-1] + (4,))
    q = np.concatenate((q_xyzw[..., 3:4], q_xyzw[..., :3]), axis=-1)
    if q.ndim == 2:
        for i in range(1, q.shape[0]):
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
    return q


def smpl24_global_rotation(poses_rotvec):
    """poses [T,24,3] axis-angle -> global rotmats [T,24,3,3] (PRISM world), FK 누적."""
    local = Rotation.from_rotvec(poses_rotvec.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    out = np.empty_like(local)
    out[:, 0] = local[:, 0]
    for j in range(1, 24):
        out[:, j] = out[:, SMPL24_PARENTS[j]] @ local[:, j]
    return out


def angular_velocity_deg_s(R, dt):
    """[N,3,3] world 방향열 -> body-frame 각속도 [N,3] deg/s. rotation-log 중앙차분."""
    Rp = project_so3(R)
    N = Rp.shape[0]
    omega = np.zeros((N, 3), dtype=np.float64)
    # 중앙차분: rel = R[k-1]^T R[k+1], omega = log(rel)/(2dt)
    rel_c = np.swapaxes(Rp[:-2], -1, -2) @ Rp[2:]
    omega[1:-1] = Rotation.from_matrix(rel_c).as_rotvec() / (2.0 * dt)
    rel0 = np.swapaxes(Rp[0], -1, -2) @ Rp[1]
    omega[0] = Rotation.from_matrix(rel0).as_rotvec() / dt
    rel1 = np.swapaxes(Rp[-2], -1, -2) @ Rp[-1]
    omega[-1] = Rotation.from_matrix(rel1).as_rotvec() / dt
    return np.degrees(omega)


def specific_force(acc_world, R_world_from_sensor):
    """f_sensor = R^T (a_world - g_world)."""
    a = np.asarray(acc_world, np.float64) - GRAVITY_PRISM_WORLD
    R = np.asarray(R_world_from_sensor, np.float64)
    return np.einsum("nji,nj->ni", R, a)  # R[n].T @ a[n]


def butter_filtfilt(x):
    b, a = signal.butter(FILTER_ORDER, FILTER_CUTOFF_HZ, btype="low", fs=FILTER_FS, output="ba")
    return signal.filtfilt(b, a, x, axis=0, padtype=FILTER_PADTYPE, padlen=FILTER_PADLEN, method="pad")


def centered_diff(x, dt):
    r = np.empty_like(x, dtype=np.float64)
    r[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    r[0] = (x[1] - x[0]) / dt
    r[-1] = (x[-1] - x[-2]) / dt
    return r


def canonical_json_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256_hex(b):
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_docx_code_aliases(large):
    """팀 Data Specification의 Large code *이름*을 기존 PoC 배열에 무손실 alias한다."""
    compatible = dict(large)
    for specification_code, internal_code in DOCX_LARGE_FLAT_CODE_ALIASES.items():
        if internal_code not in compatible:
            raise KeyError(f"missing internal Large field for {specification_code}: {internal_code}")
        compatible[specification_code] = compatible[internal_code]
    return compatible


def docx_code_compatibility_metadata():
    """업데이트된 qmd 스펙(00_sensors·01_streams·02_kinematics) 코드명 준수 현황과 의미 경계."""
    return {
        "spec_documents": {
            "small_sensors": "small/00_sensors.qmd",
            "small_streams": "small/01_streams.qmd",
            "large_kinematics": "large/02_kinematics.qmd",
        },
        "historical_docx": {
            "document": DATA_SPECIFICATION_DOCX,
            "document_sha256": DATA_SPECIFICATION_DOCX_SHA256,
            "note": "이전 DOCX flat/qualified 별칭(joint_positions/joint_rotations)은 qmd 코드명으로 대체됨(superseded).",
        },
        "small_codes": ["sensor_codes(8)", "imu_orientation", "imu_acceleration",
                        "imu_angular_velocity", "imu_orientation_absolute_heading"],
        "large_codes": ["joint_names(18)", "root_velocity", "joint_rotation",
                        "joint_velocity", "pelvis_position_world_aux"],
        "status": "code_name_and_shape_compatibility_only",
        "semantic_limits": {
            "small": "코드명·shape은 qmd 준수; 단위/장착보정(mount extrinsic) 의미는 미인증. "
                     "인솔(foot_l/r) orientation은 중력기준 자세(heading 드리프트); ideal 참조라 GT로 채움.",
            "root_velocity": "PRISM world(=spec G, Z-up) 프레임의 프레임간 변위/dt; heading 정규화는 스펙상 미확정.",
            "joint_rotation": "SMPL local(부모 대비) 쿼터니언; ISB JCS 각 분해는 미적용.",
            "joint_velocity": "joint_rotation과 동일 규약의 각속도(0=골반, 1-17=관절).",
        },
        "unavailable_codes": {
            "grf": "canonical SOMA world 지면반력[T,3] 미생성(insole는 development_reference에 source-native).",
            "cop": "canonical [T,2] SOMA world CoP 미생성.",
            "joint_torques": "2차 범위 역동역학 미생성.",
        },
    }


# --------------------------------------------------------------------------
# 3. 생성 (full-take 계산 후 window 슬라이스)
# --------------------------------------------------------------------------
def subject_fit_inputs(spec):
    """(native pose [N,24,3], rest joints (24,3)) of one take, for the subject's pooled fit.

    The rest skeleton is the one `build` uses: the subject's gendered model at the take's betas."""
    raw = load_prism_safe(spec.source_pkl)
    poses = np.asarray(raw["smpl_params"]["poses"], np.float64).reshape(-1, 24, 3)
    win_start = spec.window[0] if spec.window is not None else 0
    betas0 = np.asarray(raw["smpl_params"]["betas"], np.float64)[win_start]
    model = anthro_smpl.load_smpl_model_for_gender(spec.gender)
    return poses, anthro_smpl.rest_joints(model, betas0)


def build(spec, reduction=None):
    """One take -> (small, large, development_reference, anthro, manifest).

    `reduction` is the subject's frozen-joint fit, pooled over all of the subject's takes by the batch
    runner; without it the constants are fitted on this take and the records say `take` scope."""
    raw = load_prism_safe(spec.source_pkl)
    fps = int(raw["info"]["data_info"]["fps"])
    assert fps == TARGET_RATE_HZ
    dt = 1.0 / TARGET_RATE_HZ
    N = int(raw["smpl_params"]["poses"].shape[0])
    win_start, win_end = spec.window if spec.window is not None else (0, N)
    T = win_end - win_start
    imu_gt, insole = raw["imu_gt"], raw["insole"]

    def win(a):
        return np.asarray(a)[win_start:win_end]

    poses = np.asarray(raw["smpl_params"]["poses"], np.float64).reshape(N, 24, 3)
    gR_full = smpl24_global_rotation(poses)                       # [N,24,3,3]

    pelvis_pos_full = np.asarray(imu_gt["Pelvis"]["pos_world"], np.float64)
    head_pos_full = np.asarray(imu_gt["Head"]["pos_world"], np.float64)

    # ---- 시간축 ----
    timestamps_s = np.arange(T, dtype=np.float64) * dt
    source_interval_s = [win_start / fps, win_end / fps]
    timestamps_sha256 = sha256_hex(np.ascontiguousarray(timestamps_s, "<f8").tobytes())

    take_sha256 = _sha256_file(spec.source_pkl)
    canon_cfg = {"kind": "poc_faithful", "window_start": win_start, "frame_count": T,
                 "target_rate_hz": TARGET_RATE_HZ, "frame_convention": "prism_world_native_Zup",
                 "gravity_prism_world_m_s2": GRAVITY_PRISM_WORLD.tolist(),
                 "chest_alpha": CHEST_ALPHA, "chest_orientation_joint": CHEST_SPINE3_JOINT,
                 "filter": "butter4_7hz_filtfilt_odd15", "angular_velocity": "rotation_log",
                 "smpl_fk": "smpl24_pose_chain"}
    canon_cfg_sha256 = sha256_hex(canonical_json_bytes(canon_cfg))

    subject_id, trial_id = spec.subject_id, spec.take_id
    preimage = ["prism", "prism-archive-565082ab-v1", take_sha256, subject_id, trial_id,
                __import__("struct").pack(">d", source_interval_s[0]).hex(),
                __import__("struct").pack(">d", source_interval_s[1]).hex(),
                canon_cfg_sha256, CONTRACT_ID, CONTRACT_VERSION, str(TARGET_RATE_HZ)]
    pair_id = sha256_hex(canonical_json_bytes(preimage))

    # ---------------- Chest proxy (full-take -> window) ----------------
    chest_pos_full = pelvis_pos_full + CHEST_ALPHA * (head_pos_full - pelvis_pos_full)
    chest_pos_filt = butter_filtfilt(chest_pos_full)
    chest_vel_full = butter_filtfilt(centered_diff(chest_pos_filt, dt))
    chest_acc_full = centered_diff(chest_vel_full, dt)

    # ---------------- SMALL: 8채널 통합 IMU (본체 6 + 인솔 2) ----------------
    #   본체 6(ch0..5): 9축 → 절대 heading 사원수. 인솔 2(ch6/7): 6축 → 중력기준 자세(heading 드리프트).
    #   01_streams.qmd 갱신으로 인솔도 orientation 제공. ideal 참조이므로 인솔 orientation을
    #   GT(참 heading 포함)로 채우고, 실제 6축 한계는 imu_orientation_absolute_heading 플래그로 표기.
    imu_orientation = np.zeros((T, N_SENSORS, 4), np.float32)
    imu_acceleration = np.zeros((T, N_SENSORS, 3), np.float32)
    imu_angular_velocity = np.zeros((T, N_SENSORS, 3), np.float32)
    imu_valid_mask = np.ones((T, N_SENSORS), bool)
    imu_confidence = np.zeros((T, N_SENSORS), np.float32)
    # 절대(자기계) heading 보유 여부: 본체 True, 인솔 False(중력기준 자세·heading 드리프트).
    imu_orientation_absolute_heading = np.array([True] * N_BODY + [False] * len(FOOT_CODES))
    site_prov = {}

    def emit_site(j, R_full, acc_world_full, conf, prov):
        imu_orientation[:, j, :] = rotmat_to_quat_wxyz(win(R_full)).astype(np.float32)
        imu_acceleration[:, j, :] = win(specific_force(acc_world_full, R_full)).astype(np.float32)
        imu_angular_velocity[:, j, :] = win(angular_velocity_deg_s(R_full, dt)).astype(np.float32)
        imu_confidence[:, j] = conf
        site_prov[SENSOR_CODES[j]] = prov

    # ch0 back_T4: spine3 orientation + axial proxy accel
    emit_site(0, gR_full[:, CHEST_SPINE3_JOINT], chest_acc_full, 0.4, "proxy:back_T4_spine3+axial")
    # ch1..5 (wrist_l/r, shank_l/r, occiput): imu_gt
    for j in range(1, N_BODY):
        gt = imu_gt[SENSOR_IMU_GT[SENSOR_CODES[j]]]
        emit_site(j, np.asarray(gt["ori_world"], np.float64),
                  np.asarray(gt["acc_world"], np.float64), 0.7, "source_derived:imu_gt")

    # ch6/7 인솔(foot_l/r): 인솔도 orientation 추출(중력기준, heading 드리프트).
    #   ideal 참조라 orientation을 GT(참 heading 포함)로 채우고, 한계는 heading 플래그로 표기.
    for jj, code in enumerate(FOOT_CODES):
        j = N_BODY + jj
        gt = imu_gt[FOOT_IMU_GT[code]]
        emit_site(j, np.asarray(gt["ori_world"], np.float64),
                  np.asarray(gt["acc_world"], np.float64), 0.7,
                  "source_derived:imu_gt(real_insole=gravity_ref_heading_drift)")

    small = {
        "sensor_codes": np.array(SENSOR_CODES, dtype=object),          # 8채널 통합
        "imu_orientation": imu_orientation,                           # [T,8,4] 전 채널 유효(인솔=중력기준 자세)
        "imu_acceleration": imu_acceleration,                         # [T,8,3] specific force
        "imu_angular_velocity": imu_angular_velocity,                 # [T,8,3] deg/s
        "imu_valid_mask": imu_valid_mask,                             # [T,8] 채널 연결 상태
        "imu_confidence": imu_confidence,                             # [T,8]
        # 절대 heading 보유 여부: 인솔 2채널 False(6축, 중력기준·heading 드리프트)
        "imu_orientation_absolute_heading": imu_orientation_absolute_heading,  # [8]
        "mount_id": np.array([f"prism_faithful::{c}" for c in SENSOR_CODES], dtype=object),
        "small_mode": "ideal", "pair_id": pair_id, "timestamps_s": timestamps_s,
        "frame_count": np.int64(T),
    }

    # ---------------- LARGE: 골반(root) + 17관절 = SMPL 18 부분집합 (large/02_kinematics.qmd) ----------------
    pelvis_position = win(pelvis_pos_full).astype(np.float64)                          # 절대 위치(참고 aux 계산용)
    smpl_global_orientation = rotmat_to_quat_wxyz(win(gR_full)).astype(np.float32)     # [T,24,4] lossless
    # Reduced model (smpl18.reduce): the 4 frozen joints take the subject's constants and spine3/the
    # shoulders absorb what was removed, so distal global orientation is preserved exactly. `reduction`
    # is the subject's fit (all of its takes, pooled by the batch runner); when omitted this take is
    # fitted alone. Large's motion joints then carry the reduced-model values.
    Lfull = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(N, 24, 3, 3)
    betas0 = np.asarray(raw["smpl_params"]["betas"], np.float64)[win_start]
    smpl_gender = str(raw["smpl_params"].get("gender", "")).lower()
    assert smpl_gender[:1] == str(spec.gender).lower()[:1], \
        "subj_info.gender vs smpl_params.gender mismatch"
    smpl_model = anthro_smpl.load_smpl_model_for_gender(spec.gender)
    j_rest_fit = anthro_smpl.rest_joints(smpl_model, betas0)
    if reduction is None:
        reduction = reduced_model.fit_subject(
            [poses], j_rest_fit, reduced_model.settings_for_source("prism"),
            group=f"{spec.subject_id}/{spec.take_id}", basis="this take, whole take", scope="take")
    fixed_C = reduction.constants
    Lrefit = reduced_model.reduce_local(Lfull, fixed_C)

    # root_velocity [T,3] m/s: 프레임 간 골반 변위(backward diff, [0]=0). 누적으로 시작기준 궤적 정확 복원:
    #   pos[t] = pos[0] + dt * cumsum(root_velocity)[t].  프레임=PRISM world(spec G 프레임 변환 미적용; 스펙상 미확정).
    root_velocity = np.zeros((T, 3), np.float64)
    root_velocity[1:] = (pelvis_position[1:] - pelvis_position[:-1]) / dt
    root_velocity = root_velocity.astype(np.float32)

    # joint_rotation [T,18,4]: [0]=골반 global 자세(q_world_from_pelvis), [1:]=17관절 SMPL local rotation(부모 대비)
    joint_rotation = np.zeros((T, 18, 4), np.float32)
    joint_rotation[:, 0, :] = rotmat_to_quat_wxyz(win(gR_full[:, 0])).astype(np.float32)
    # joint_velocity [T,18,3] deg/s: [0]=골반 각속도(정렬용 extra), [1:]=17관절 각속도(spec)
    joint_velocity = np.zeros((T, 18, 3), np.float32)
    joint_velocity[:, 0, :] = win(angular_velocity_deg_s(gR_full[:, 0], dt)).astype(np.float32)
    for k in range(1, 18):
        idx = int(SMPL18_INDICES[k])
        # reduced-model local rotations (spine3/shoulders re-fitted; others == raw)
        joint_rotation[:, k, :] = rotmat_to_quat_wxyz(win(Lrefit[:, idx])).astype(np.float32)
        joint_velocity[:, k, :] = win(angular_velocity_deg_s(Lrefit[:, idx], dt)).astype(np.float32)

    large = {
        "joint_names": np.array(JOINT18_NAMES, dtype=object),            # 18 (SMPL 인덱스 순서)
        "root_velocity": root_velocity,                                 # [T,3] m/s 골반 root 속도(spec; 프레임간 변위/dt)
        "joint_rotation": joint_rotation,                               # [T,18,4] 0=root global, 1-17=관절 local
        "joint_velocity": joint_velocity,                               # [T,18,3] deg/s (0=root, 1-17=관절)
        "pelvis_position_world_aux": pelvis_position.astype(np.float32),  # [T,3] m 절대 위치(참고 aux; root_velocity 누적 기준)
        "smpl_global_orientation_prism_world": smpl_global_orientation,  # [T,24,4] lossless 보조
        "pair_id": pair_id, "timestamps_s": timestamps_s, "frame_count": np.int64(T),
    }

    # ---------------- DEVELOPMENT REFERENCE: PRISM insole (measured/source_derived) ----------------
    dev = {
        "namespace": "development_reference/prism_insole_v1",
        "grf_feet_source_native": np.stack([win(insole["L_Foot"]["force_world"]),
                                            win(insole["R_Foot"]["force_world"])], axis=1).astype(np.float32),
        "grf_combined_source_native": win(insole["combined"]["force_world"]).astype(np.float32),
        "cop_feet_world": np.stack([win(insole["L_Foot"]["CoP_world"]),
                                    win(insole["R_Foot"]["CoP_world"])], axis=1).astype(np.float32),
        "cop_combined_world": win(insole["combined"]["CoP_world"]).astype(np.float32),
        "foot_contact_mask": np.stack([  # contacts normalized to bool (usage policy C급8)
            win(insole["L_Foot"]["contacts"])[:, 0].astype(bool),
            win(insole["R_Foot"]["contacts"])[:, 0].astype(bool),
        ], axis=1),
        "vertical_force_provenance": "measured",
        "cop_provenance": "source_derived",
        "contacts_provenance": "source_derived",
        "usage": "development_reference_only (generator_input=false, supervision=false)",
        "pair_id": pair_id, "timestamps_s": timestamps_s,
    }

    # ---------------- ANTHRO: subject skeleton constants (anthro/00_anthropometry.qmd) ----------------
    #   Large 모션 18관절(축소모델) + Anthro 고정 4관절(spine1/spine2/collars, fit 상수) = SMPL 22-체인.
    #   fixed_C는 재구성 위치잔차 최소화로 fit(위 재적합 블록); spine3/어깨가 편차 흡수 → 말단 방향 exact.
    #   rest-pose 관절 위치/분절 길이는 SMPL 모델 + 실제 betas로 회귀. height/mass/sex는 측정값(subj_info).
    subj_info = dict(raw["info"]["subj_info"])
    anthro = anthro_smpl.compute_anthro(smpl_model, betas0, subj_info, pair_id,
                                        fixed_joint_local_R=fixed_C,
                                        fixed_joint_provenance=reduced_model.anthro_provenance(reduction))
    anthro["subject_id"] = subject_id
    anthro["source_asset_id"] = "prism/%s/%s.pkl" % (spec.subject_id.replace("prism_", ""), spec.take_id)

    # ---------------- validation ----------------
    v = {}
    qn = np.linalg.norm(imu_orientation, axis=2)   # 전 8채널(인솔 포함, NaN 없음)
    v["imu_quat_norm_max_dev"] = float(np.max(np.abs(qn - 1)))
    v["pelvis_quat_norm_max_dev"] = float(np.max(np.abs(np.linalg.norm(joint_rotation[:, 0, :], axis=1) - 1)))
    v["joint_rotation_quat_norm_max_dev"] = float(np.max(np.abs(
        np.linalg.norm(joint_rotation, axis=2) - 1)))
    dts = np.diff(timestamps_s)
    v["timestamp_step_max_err"] = float(np.max(np.abs(dts - dt)))
    # specific-force 역변환: a = R f + g, imu_gt.Head(occiput 채널) 로 검증
    Rh = win(np.asarray(imu_gt["Head"]["ori_world"], np.float64))
    f_head = imu_acceleration[:, SENSOR_CODES.index("occiput"), :].astype(np.float64)
    a_recon = np.einsum("nij,nj->ni", Rh, f_head) + GRAVITY_PRISM_WORLD
    a_true = win(np.asarray(imu_gt["Head"]["acc_world"], np.float64))
    v["specific_force_inverse_max_abs_err_m_s2"] = float(np.max(np.abs(a_recon - a_true)))
    # root_velocity 누적 복원 검증: pos[t] = pos[0] + dt*cumsum(root_velocity)
    pos_recon = pelvis_position[0] + dt * np.cumsum(root_velocity.astype(np.float64), axis=0)
    v["root_velocity_cumsum_recon_max_abs_err_m"] = float(np.max(np.abs(pos_recon - pelvis_position)))
    # FK 검증: SMPL FK 글로벌 방향 vs PRISM imu_gt 방향(부위별) — 상대변화 일치도(회전 offset 제거)
    fk_vs_gt = {}
    for code, joint in [("wrist_l", 20), ("wrist_r", 21), ("occiput", 15), ("shank_l", 4), ("shank_r", 5)]:
        R_fk = win(gR_full[:, joint])
        R_gt = win(np.asarray(imu_gt[SENSOR_IMU_GT[code]]["ori_world"], np.float64))
        # 고정 offset = R_gt[0]^T R_fk[0] 제거 후 잔차 각도
        offset = R_gt[0].T @ R_fk[0]
        resid = np.swapaxes(R_gt, -1, -2) @ R_fk @ np.swapaxes(offset[None], -1, -2)
        ang = np.degrees(np.linalg.norm(Rotation.from_matrix(project_so3(resid)).as_rotvec(), axis=1))
        fk_vs_gt[code] = {"resid_deg_mean": float(np.mean(ang)), "resid_deg_p95": float(np.percentile(ang, 95))}
    v["fk_vs_imu_gt_orientation_residual"] = fk_vs_gt

    # Reduced-model QC (spec sec-anthro-fit): joint-center position residual of the reduced model
    # (Large refit + Anthro fixed constants) vs raw SMPL, pelvis-relative over the window. Distal
    # global ORIENTATION is exact by construction; this residual is the small trunk position offset.
    raw_pos = anthro_smpl.fk_positions_batch(j_rest_fit, win(Lfull))
    ref_pos = anthro_smpl.fk_positions_batch(j_rest_fit, win(Lrefit))
    qc = np.linalg.norm(raw_pos - ref_pos, axis=2)   # [T,24]
    aff = list(reduced_model.AFFECTED_BY_FREEZE)
    promotion = {}
    for fname, js in FIXED_JOINT_DOWNSTREAM.items():
        p95 = float(np.percentile(qc[:, js], 95))
        promotion[fname] = {"p95_m": p95, "max_m": float(qc[:, js].max()),
                            "promotion_candidate": bool(p95 > PROMOTE_TAU_P95_M)}
    v["reduced_model_fit_residual_m"] = {
        "overall_max": float(qc.max()),
        "affected_mean": float(qc[:, aff].mean()),
        "per_joint_max": {anthro_smpl.JOINT24_NAMES[j]: float(qc[:, j].max()) for j in aff},
        "promotion_threshold_p95_m": PROMOTE_TAU_P95_M,
        "fixed_joint_promotion": promotion,
        "promotion_candidates": [f for f, d in promotion.items() if d["promotion_candidate"]],
        "note": "reduced-model (Large 18 refit + Anthro 4 fixed constants) vs raw SMPL joint-center "
                "position residual; distal global orientation exact (see sec-anthro-fit). A fixed joint "
                "whose downstream p95 residual exceeds promotion_threshold_p95_m is flagged as a "
                "Large-motion promotion candidate.",
    }

    manifest = {
        "distribution_scope": "internal_only", "artifact_class": "experimental_non_candidate",
        "artifact_label": "PoC FAITHFUL reference (independent replica of sanctioned PRISM pilot recipe). "
                          "NOT canonical, NOT contract-compliant, NOT quality-gate PASS. INTERNAL-ONLY.",
        "excluded_modalities": ["emg"], "excluded_modality_reasons": {"emg": "not_in_synthetic_artifact_scope"},
        "run_id": "%s-%s-faithful" % (spec.subject_id.replace("prism_", "prism-"), spec.take_id),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"source_name": "prism", "subject_id": subject_id, "trial_id": trial_id,
                   "trial_name": raw["info"]["data_info"]["trial_name"],
                   "source_asset_sha256_take": take_sha256,
                   "relative_path": spec.relative_path or (
                       "extracted/prism/%s/%s.pkl" % (spec.subject_id.replace("prism_", ""), spec.take_id)),
                   "native_frames_total": N, "native_rate_hz": fps,
                   "note": "some wrist/head IMU frames are synth (info.data_info.synth_imu_frames); "
                           "본 참조는 imu_gt(model-derived kinematics) 기반이라 직접 영향은 없으나 provenance상 유의."},
        "window": {"frame_start": win_start, "frame_end_exclusive": win_end,
                   "frame_count": T, "source_interval_s": source_interval_s},
        "identity": {"pair_id": pair_id, "contract_id": CONTRACT_ID, "contract_version": CONTRACT_VERSION,
                     "timestamps_sha256": timestamps_sha256, "target_rate_hz": TARGET_RATE_HZ,
                     "canonicalization_config_sha256": canon_cfg_sha256},
        # `data_specification_code_compatibility` and `recipe_alignment` were dropped from the
        # manifest on 2026-09-15, the regeneration the registry's waivers named as the moment to
        # drop them (dataset_profiles_v1.yaml, prism waivers of 2026-09-08). Both were legacy keys
        # of the pre-faithful-v2 lineage; docx_code_compatibility_metadata() stays for its callers.
        "canonicalization_config": canon_cfg,
        "spec_documents": {
            "small_sensors": "small/00_sensors.qmd",
            "small_streams": "small/01_streams.qmd",
            "large_kinematics": "large/02_kinematics.qmd",
            "large_conventions": "large/99_conventions.qmd",
        },
        "global_frame": {
            "convention": "PRISM world, right-handed, Z-up, gravity along -Z",
            "conforms_to_spec_G": True,
            "note": ("updated large/99_conventions.qmd defines global frame G as Z-up aligned to "
                     "PRISM world; data is already in this frame, so no frame conversion is needed."),
        },
        "small_sites_provenance": site_prov,
        "prism_usage_policy": {
            "config_id": "prism_usage_policy_v1",
            "note": "Small IMU derived from imu_gt (model-derived), permitted for all sites incl. the "
                    "excluded wrist/head (A급1 PERMITTED_AS_IMU_GT_ONLY). Pelvis trajectory uses "
                    "imu_gt.Pelvis.pos_world (released stream), consistent with trans-only world "
                    "(A급2; root_offset not added). Insole contacts normalized to bool (C급8). anthro "
                    "height/body_mass in SOMA spec units (m/kg); PRISM source subj_info units "
                    "undeclared upstream (C급7); arm_length excluded.",
        },
        "large_joint_names": JOINT18_NAMES,
        "deliverables": [
            "small_reference.npz", "large_reference.npz", "anthro_reference.npz",
            "development_reference.npz", "manifest.json", "README.md",
        ],
        "faithful_content": {
            "small": "8채널 통합 IMU(back_T4·wrist_l/r·shank_l/r·occiput·foot_l/r), 전 채널 orientation+accel+gyro. "
                     "인솔 2(foot_l/r)는 6축이라 실제로는 중력기준 자세(heading 드리프트); "
                     "ideal 참조라 orientation은 GT로 채우고 imu_orientation_absolute_heading=False로 표기. "
                     "shank=PRISM 무릎부 GT(proxy), 손목/머리/발=imu_gt, back_T4=spine3+α proxy.",
            "large": "골반(root) root_velocity[T,3] + joint_rotation[T,18,4](0=골반 global, 1-17=관절 SMPL local, "
                     "축소모델: spine3·좌우 어깨는 고정관절 편차 흡수하도록 재적합) + joint_velocity[T,18,3]; "
                     "절대 위치는 pelvis_position_world_aux(참고). raw SMPL은 smpl_global_orientation_prism_world 보존.",
            "anthro": "대상자 뼈대 상수(anthro/00_anthropometry.qmd): joint_position[2,22,3](T-pose·직립중립, "
                      "SMPL rest 회귀) + segment_length[13] + fixed_joint_rotation[4,4](spine1/spine2/collars, "
                      "모션 전체 위치잔차 최소화 fit; sec-anthro-fit) + sex/height/body_mass(측정) + betas(source_derived). "
                      "Large 축소모델 모션과 합쳐 SMPL 22-체인 복원(말단 방향 exact). betas는 참조 레이어로 이전.",
            "development_reference": "PRISM insole GRF/CoP/contacts (measured/source_derived), 별도 namespace. "
                                     "인솔 전용(betas 등 SMPL 형상은 anthro_reference로 이전됨).",
        },
        "anthro_reconstruction": {
            **reduced_model.reconstruction_block(
                reduction,
                "large.smpl_global_orientation_prism_world[T,24,4] + raw prism_take002_smplx.npz"),
            "joint_position_frame": "smpl_canonical_rest_yup (rest geometry; world via Large pelvis global)",
        },
        "unavailable_or_not_applied": [
            "joint_angles_jcs_deg (needs ISB/JCS convention — not applied)",
            "physics-informed GRF / inverse dynamics / moments / powers / COM (DETERMINISTIC_EXECUTION_CONFIG_HOLD)",
            "sensor mount extrinsics (segment→sensor lever arm; sites at joint/segment origin)",
            "quality gates (PHYSICS_QUALITY_GATES.md)",
        ],
        "validation": v,
        "safety": {"pickle_loader": "numpy-whitelist restricted unpickler (no arbitrary code execution)"},
    }
    return small, large, dev, anthro, manifest


def write_outputs(out_dir, small, large, dev, anthro, manifest):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "small_reference.npz"), **small)
    np.savez(os.path.join(out_dir, "large_reference.npz"), **large)
    np.savez(os.path.join(out_dir, "anthro_reference.npz"), **anthro)
    np.savez(os.path.join(out_dir, "development_reference.npz"), **dev)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def write_readme(out_dir, run_id):
    readme = f"""# PRISM → Small/Large **FAITHFUL** reference ({run_id})

정식 PRISM experimental pilot 레시피(`prism_experimental_pilot_v1`)를 독립적으로 replicate한
**물리적으로 충실한 참조 페어**입니다. 여전히 `experimental_non_candidate` · 계약 비준수 · 품질게이트 미평가 ·
`INTERNAL-ONLY`. 학습/평가 입력 금지.

외국인 연구자용 상세 영문 설명서는 동봉 `DATA_DESCRIPTION_EN.md`(번들에 별도 유지) 참조.
필드/shape/규약의 정본은 `manifest.json`이다.

## 파일
- `small_reference.npz` — 8채널 통합 IMU `sensor_codes`(back_T4·wrist_l/r·shank_l/r·occiput·foot_l/r):
  `imu_orientation[T,8,4]`(전 채널) · `imu_acceleration[T,8,3]` specific force ·
  `imu_angular_velocity[T,8,3]` deg/s · `imu_valid_mask` · `imu_confidence` · `imu_orientation_absolute_heading`[8].
  인솔(foot_l/r)도 orientation 제공 — 6축이라 중력기준 자세(heading 드리프트), ideal 참조라 GT로 채움.
- `large_reference.npz` — `joint_names`(18) + `root_velocity[T,3]`(골반 속도 m/s) + `joint_rotation[T,18,4]`
  (`[:,0]`=골반 global, `[:,1:]`=17관절 SMPL local) + `joint_velocity[T,18,3]`(deg/s) +
  `pelvis_position_world_aux[T,3]`(절대위치 참고) + `smpl_global_orientation_prism_world[T,24,4]`(무손실 보조). ISB JCS 각 아님.
  - 궤적 복원: `pos[t] = pelvis_position_world_aux[0] + dt·cumsum(root_velocity)[t]`.
- `anthro_reference.npz` — 대상자 뼈대 상수(시행·프레임 무관): `joint_position[2,22,3]`(T-pose·직립중립, root 기준) +
  `segment_length[13]` + `fixed_joint_rotation[2,4,4]`(spine1·spine2·좌우 collar, take 평균) +
  `sex`·`height`·`body_mass`(측정) + `betas`(SMPL 형상, source_derived). **Large 모션 + Anthro → SMPL 22-체인 복원.**
- `development_reference.npz` — PRISM insole GRF(source-native 단위)/CoP(world m)/contact (measured/source_derived; 인솔 전용, synthetic과 분리)
- `manifest.json` — provenance, validation(재구성 잔차 포함), 미적용 단계

## 실데이터 유래
- **source_derived:** 손목/머리/정강이 IMU(PRISM imu_gt), 인솔 IMU(발 GT), 골반 궤적, SMPL FK 17관절 회전, SMPL betas(형상).
- **derived:** anthro 뼈대(SMPL rest-pose J 회귀), fixed_joint_rotation(take 평균).
- **measured:** insole 수직력(development_reference), 신장·체중·성별(subj_info).
- **proxy:** back_T4(spine3 방향 + pelvis→head α=2/3 축위치), shank(무릎부 GT → 정강이 상부 proxy).
- **복원 주의:** 고정 4관절을 take 평균으로 얼리면 손목 재구성 잔차 ~0.1m(identity로 얼리면 ~0.3m).
  프레임별 변동·무손실 원본은 `large.smpl_global_orientation_prism_world`와 raw SMPL 참조 레이어에 보존.
- **unavailable(생성 안 함):** ISB JCS 각도, canonical `grf`/`cop`, `joint_torques`,
  physics 모멘트/파워/COM, mount extrinsic(분절→센서 lever arm), 품질게이트.

## 규약
프레임 PRISM world(Z-up 우수) = 갱신 spec `G`(Z-up·PRISM world 정렬), 쿼터니언 `(w,x,y,z)`, 중력 `g=[0,0,-9.80665]`,
specific force `f=Rᵀ(a_world−g)`, 각속도 rotation-log deg/s. 방향 프레임은 PRISM GT 분절 프레임이며
`00_sensors.qmd`의 물리 센서 축 정렬(mount extrinsic)은 미적용(목표 규약).
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme)


if __name__ == "__main__":
    try:
        _prism = paths.source_dir("prism")
        default_pkl = os.path.join(str(_prism), "subj001", "take002.pkl")
        _out = os.environ.get("SOMA_POC_OUT_DIR") or str(paths.lineage_dir(RUN_ID))
        paths.body_model_dir()
        paths.check_output_dir(_out, inputs=[_prism])
    except paths.PathConfigError as _error:
        print(_error)
        raise SystemExit(2) from None
    _spec = TakeSpec(subject_id="prism_subj001", take_id="take002", source_pkl=default_pkl,
                     gender="M", window=(1000, 2000),
                     relative_path="extracted/prism/subj001/take002.pkl")
    small, large, dev, anthro, manifest = build(_spec)
    write_outputs(_out, small, large, dev, anthro, manifest)
    write_readme(_out, manifest["run_id"])
    print("OK ->", _out)
    print("pair_id:", manifest["identity"]["pair_id"])
    print("validation:", json.dumps(manifest["validation"], indent=2)[:1800])
