"""AMASS -> paired Small/Large/Anthro **faithful** synthetic reference (independent replica).

Mirrors the PRISM faithful generator (`generate_prism_faithful.py`) for AMASS. The essential
difference: AMASS ships **only** SMPL-H pose parameters (poses/betas/trans/gender/mocap_framerate)
— there is no measured IMU (`imu_gt`) and no insole/GRF. So every one of the 8 Small IMU channels is
**synthesized from the SMPL body via forward kinematics** (segment orientation + double-differentiated
world acceleration + rotation-log angular velocity), there is **no development_reference** (no measured
kinetics exist), and there is no measured subject height/mass.

Governance: the AMASS field registry is `activation_state: non_authorizing` (like PRISM_SOURCE_PASS) —
this is NOT a generation authorization. The artifact stays `experimental_non_candidate`, INTERNAL-ONLY,
an independent replica, not the canonical governed artifact. AMASS npz are read via
`np.load(allow_pickle=False)` using only the audited core fields.

Conventions mirror the qmd spec (small/00_sensors, small/01_streams, large/02_kinematics,
anthro/00_anthropometry) and the PRISM faithful recipe:
  * Frame = AMASS world, empirically **Z-up** for every sub-dataset (head-foot dz > 0) = spec frame G;
    gravity g=[0,0,-9.80665]; no frame conversion.
  * SMPL-H poses[:, :66] = the 22 body joints == SMPL joints 0..21; a 24-joint pose is formed with the
    two hands (22,23) at identity. The tested `anthro_smpl` module supplies rest joints / FK / the
    reduced-model refit.
  * Source framerate is resampled to 100 Hz (per-joint Slerp on local rotations + linear interp on trans);
    a no-op for KIT (already 100 Hz).
  * Small 8ch (back_T4,wrist_l/r,shank_l/r,occiput,foot_l/r): orientation from the SMPL segment global
    rotation, specific force f=R^T(a_world-g) from butter-filtered double-diff world position, angular
    velocity from the rotation-log. Insole channels keep the 6-axis heading=False flag (sensor model),
    though the synthetic orientation is model-exact.
  * Large = root_velocity + joint_rotation[18] (reduced-model) + joint_velocity[18] + pelvis position aux
    + raw smpl_global_orientation[24], identical in shape/convention to PRISM.
  * Anthro = SMPL rest skeleton constants + fixed-joint weld; betas are AMASS 16-dim but the clean model
    carries 10 (smpl10_beta_truncation); height is an SMPL stature estimate, body_mass is unavailable.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util as _ilu
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation, Slerp

# Anthro helper — the same tested module the PRISM generator uses (sibling file).
_ANTHRO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anthro_smpl.py")
_anthro_spec = _ilu.spec_from_file_location("anthro_smpl", _ANTHRO_PATH)
anthro_smpl = _ilu.module_from_spec(_anthro_spec)
_anthro_spec.loader.exec_module(anthro_smpl)

# The attribution registry is a config value, never a code constant; this generator writes
# source_attribution itself.
_SRC_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
from soma_synth.contracts import source_attribution as _attribution  # noqa: E402
from soma_synth.pipeline import paths, reduced_model  # noqa: E402

# --------------------------------------------------------------------------
# 0. Pins / constants (mirror the PRISM faithful recipe)
# --------------------------------------------------------------------------
# The AMASS source folder is paths.source_dir("amass") ($SOMA_SOURCE_ROOT/amass, by default
# $SOMA_DATA_ROOT/extracted/amass), resolved by whoever reads it; nothing is read at import.

#: The spec lineage every manifest and INDEX names (layout section 6). Written by the generator
#: since 2026-09-15; earlier bundles received these two keys in a reconcile pass after the fact.
SPEC_ID = "qmd_unified8_smpl18"
SPEC_VERSION = "faithful-v2"
SEX_PROVENANCE = ("source_derived:amass npz 'gender' field (SMPL-H subject gender; 'neutral' when "
                  "the field is absent)")
SMPL_TRANS_PROVENANCE = ("source_derived:amass npz trans, resampled to the 100 Hz grid with the pose "
                         "(linear interpolation); equals large.pelvis_position_world_aux, which in "
                         "this lineage is the SMPL root translation")


@functools.lru_cache(maxsize=1)
def attribution_block():
    """manifest.source_attribution for amass, read from the registry rather than typed here."""
    return dict(_attribution.resolved_for("amass"))


TARGET_RATE_HZ = 100
GRAVITY_WORLD = np.array([0.0, 0.0, -9.80665], dtype=np.float64)  # Z-up (empirically verified for AMASS)
CHEST_ALPHA = 2.0 / 3.0
FILTER_ORDER, FILTER_CUTOFF_HZ, FILTER_FS = 4, 7.0, 100.0
FILTER_PADLEN, FILTER_PADTYPE = 15, "odd"

CONTRACT_ID = "soma_paired_small_large_v3_POC_FAITHFUL_AMASS_SYNTH"
CONTRACT_VERSION = "0.0.1-poc-faithful-amass-synth"

AMASS_FIELD_REGISTRY = ("amass_field_registry_v2 (activation_state=non_authorizing); the yaml lives in the "
                        "parent project's configs/datasets/, not in soma-synth")

# AMASS sub-datasets and their standard mocap framerate (fallback when the npz omits mocap_framerate).
DATASET_FPS_FALLBACK = {
    "BMLmovi": 120.0, "CMU": 120.0, "Transitions": 120.0, "KIT": 100.0, "TotalCapture": 60.0,
}
AMASS_DATASETS = tuple(DATASET_FPS_FALLBACK)

SMPL24_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)

# Small 8-channel synthetic IMU (small/00_sensors.qmd, small/01_streams.qmd).
SENSOR_CODES = ["back_T4", "wrist_l", "wrist_r", "shank_l", "shank_r", "occiput", "foot_l", "foot_r"]
N_SENSORS = 8
N_BODY = 6  # ch0..5 = 9-DOF (orientation w/ absolute heading). ch6/7 (foot/insole) = 6-axis.
FOOT_CODES = ["foot_l", "foot_r"]

# Spec sensor frame S (small/00_sensors.qmd): +Y=proximal, +Z=outward lateral, +X=Y x Z; left limbs
# mounted 180 deg about Y (left +X=posterior). Built from the SMPL segment: R_S = gR[:,J] @ M_J @ C_LR,
# with M_J the constant anatomical relabel from rest geometry (no imu_gt needed — the AMASS orientation
# IS the SMPL FK segment frame, so we relabel it directly; matches the PRISM measured construction).
#   J = driving SMPL joint (wrist->forearm/elbow, shank->knee, occiput->head, foot->foot, back->spine3),
#   kind = limb|foot, (pj,dj) = proximal/distal joints for the long axis, side = L (180 flip) / R|M.
SITE_GEOM = {
    "back_T4": (9, "limb", 12, 9, "M"),
    "wrist_l": (18, "limb", 18, 20, "L"), "wrist_r": (19, "limb", 19, 21, "R"),
    "shank_l": (4, "limb", 4, 7, "L"), "shank_r": (5, "limb", 5, 8, "R"),
    "occiput": (15, "limb", 15, 12, "M"),
    "foot_l": (10, "foot", 10, 7, "L"), "foot_r": (11, "foot", 11, 8, "R"),
}
_ROT_Y_180 = np.diag([-1.0, 1.0, -1.0])
# Site -> SMPL joint whose WORLD position is differentiated for the linear acceleration. back_T4 is the
# non-rigid chest proxy (pelvis + alpha*(head-pelvis)); feet use the toe joint (10/11) on the foot segment.
SITE_POS_JOINT = {
    "wrist_l": 20, "wrist_r": 21, "shank_l": 4, "shank_r": 5,
    "occiput": 15, "foot_l": 10, "foot_r": 11,
}
SITE_CONFIDENCE = {"back_T4": 0.4, "foot_l": 0.5, "foot_r": 0.5}  # others default below

# Large kinematics (large/02_kinematics.qmd): pelvis(root) + 17 joints, the 18 smpl18 keeps
# (spine1, spine2, both collars frozen; hands dropped). The package owns the list.
JOINT18_NAMES = list(reduced_model.JOINT18_NAMES)
SMPL18_INDICES = np.array(reduced_model.KEEP18, np.int64)

# Reduced-model QC (spec sec-anthro-fit): fixed-joint downstream sets + the p95 position-residual bar.
RIGID_WELD_TAU_P95_M = 0.05
FIXED_JOINT_DOWNSTREAM = {
    "left_collar": [16, 18, 20], "right_collar": [17, 19, 21],
    "spine1": [9, 12, 15], "spine2": [9, 12, 15],
}


class AmassSeqSpec:
    """One AMASS sequence to synthesize: dataset/subject/sequence identity + source npz.

    Plain class (not a dataclass) so it constructs correctly when this module is imported via
    importlib without a sys.modules entry (tests and the batch runner load it that way).
    """

    def __init__(self, dataset, subject_id, sequence_id, source_npz, relative_path=""):
        self.dataset = dataset                  # "CMU" / "KIT" / ...
        self.subject_id = subject_id            # e.g. "amass_CMU_01"
        self.sequence_id = sequence_id          # e.g. "01_02_poses"
        self.source_npz = source_npz            # absolute path to the *_poses.npz
        self.relative_path = relative_path      # e.g. "extracted/amass/CMU/01/01_02_poses.npz"

    def __repr__(self):
        return "AmassSeqSpec(%s/%s/%s)" % (self.dataset, self.subject_id, self.sequence_id)


# --------------------------------------------------------------------------
# 1. Safe loading + gender / framerate normalization
# --------------------------------------------------------------------------
def normalize_gender(g):
    """AMASS `gender` (numpy <U str or bytes, or a python str) -> 'male'/'female'/'neutral'."""
    s = g.item() if hasattr(g, "item") else g
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    s = str(s).strip().lower()
    if s.startswith("m"):
        return "male"
    if s.startswith("f"):
        return "female"
    if s.startswith("n"):
        return "neutral"
    raise ValueError("unrecognized AMASS gender: %r" % (g,))


def _source_label(path) -> str:
    """How an error names a source file: its logical id (``extracted/amass/...``) when it lies in
    the AMASS source folder, else its file name. Never the absolute path of the PC that ran it:
    the error text becomes the failed take's reason in INDEX.json, which ships with the bundle."""
    try:
        return paths.logical_source_id("amass", path)
    except (paths.PathConfigError, ValueError, OSError):
        return os.path.basename(os.fspath(path))


def load_amass_safe(path, dataset=None):
    """Load an AMASS *_poses.npz using only the audited core fields, no pickle execution.

    Returns {poses[N,156], trans[N,3], betas[B], gender, fps, fps_source}. Raises if `poses` absent.
    `dataset` supplies the framerate fallback when `mocap_framerate` is missing.
    """
    with np.load(path, allow_pickle=False) as d:
        files = set(d.files)
        if "poses" not in files:
            raise ValueError("AMASS npz has no `poses` field (shape-only?): %s" % _source_label(path))
        # Fail closed on missing core fields rather than fabricating zeros (which would be marked valid).
        for req in ("trans", "betas"):
            if req not in files:
                raise ValueError("AMASS npz missing required field `%s`: %s" % (req, _source_label(path)))
        poses = np.asarray(d["poses"], np.float64)
        trans = np.asarray(d["trans"], np.float64)
        betas = np.asarray(d["betas"], np.float64)
        # Non-finite input would silently produce NaN synthetic IMU; reject so the sequence is recorded
        # as `failed` in the run index instead (the AMASS Transitions archive has known non-finite frames).
        for name, arr in (("poses", poses), ("trans", trans), ("betas", betas)):
            if not np.all(np.isfinite(arr)):
                raise ValueError("AMASS npz has non-finite `%s`: %s" % (name, _source_label(path)))
        gender = normalize_gender(d["gender"]) if "gender" in files else "neutral"
        if "mocap_framerate" in files:
            fps, fps_source = float(np.asarray(d["mocap_framerate"])), "native"
        elif "mocap_frame_rate" in files:
            fps, fps_source = float(np.asarray(d["mocap_frame_rate"])), "native"
        else:
            fps = DATASET_FPS_FALLBACK.get(dataset)
            if fps is None:
                raise ValueError("no mocap_framerate and no dataset fallback for %s" % path)
            fps_source = "dataset_fallback"
    return {"poses": poses, "trans": trans, "betas": betas, "gender": gender,
            "fps": float(fps), "fps_source": fps_source}


def build_pose24(poses):
    """AMASS poses [N,156] (SMPL-H) -> [N,24,3] axis-angle: 0..21 = body, hands (22,23) = identity(0)."""
    poses = np.asarray(poses, np.float64)
    n = poses.shape[0]
    p24 = np.zeros((n, 24, 3), np.float64)
    p24[:, :22] = poses[:, :66].reshape(n, 22, 3)
    return p24


# --------------------------------------------------------------------------
# 2. Resampling to the 100 Hz target grid
# --------------------------------------------------------------------------
def resampled_length(n, src_fps, dst_fps=TARGET_RATE_HZ):
    """Frame count on the dst grid covering [0, (n-1)/src_fps] with no extrapolation.

    The ``+1e-9`` guards against float underflow in ``floor`` (e.g. 138/120*100 evaluates to
    114.9999999999 instead of 115), which would otherwise silently drop the final endpoint frame.
    """
    if float(src_fps) == float(dst_fps):
        return int(n)
    duration = (n - 1) / float(src_fps)
    return int(np.floor(duration * dst_fps + 1e-9)) + 1


def _resample_grids(n, src_fps, dst_fps=TARGET_RATE_HZ):
    t_src = np.arange(n) / float(src_fps)
    t_dst = np.arange(resampled_length(n, src_fps, dst_fps)) / float(dst_fps)
    return t_src, t_dst


def resample_rotations(local_R, src_fps, dst_fps=TARGET_RATE_HZ):
    """[N,J,3,3] local rotation matrices -> [M,J,3,3] on the dst grid via per-joint Slerp.

    Exact no-op (copy) when src_fps == dst_fps.
    """
    local_R = np.asarray(local_R, np.float64)
    if float(src_fps) == float(dst_fps):
        return local_R.copy()
    n, jn = local_R.shape[0], local_R.shape[1]
    t_src, t_dst = _resample_grids(n, src_fps, dst_fps)
    out = np.empty((t_dst.shape[0], jn, 3, 3), np.float64)
    for j in range(jn):
        out[:, j] = Slerp(t_src, Rotation.from_matrix(local_R[:, j]))(t_dst).as_matrix()
    return out


def resample_trans(trans, src_fps, dst_fps=TARGET_RATE_HZ):
    """[N,3] translation -> [M,3] via per-axis linear interpolation. No-op copy when rates match."""
    trans = np.asarray(trans, np.float64)
    if float(src_fps) == float(dst_fps):
        return trans.copy()
    t_src, t_dst = _resample_grids(trans.shape[0], src_fps, dst_fps)
    return np.stack([np.interp(t_dst, t_src, trans[:, k]) for k in range(3)], axis=1)


# --------------------------------------------------------------------------
# 3. Pure kinematic helpers (duplicated from the PRISM generator: keep PRISM golden tests untouched)
# --------------------------------------------------------------------------
def project_so3(R):
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    neg = np.linalg.det(Rp) < 0
    if np.any(neg):
        U2 = U.copy()
        U2[neg, :, -1] *= -1.0
        Rp = U2 @ Vt
    return Rp


def rotmat_to_quat_wxyz(R):
    q_xyzw = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat().reshape(R.shape[:-2] + (4,))
    q = np.concatenate((q_xyzw[..., 3:4], q_xyzw[..., :3]), axis=-1)
    if q.ndim == 2:
        for i in range(1, q.shape[0]):
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
    return q


def _unit(v):
    return np.asarray(v, np.float64) / np.linalg.norm(v)


def canonical_axes(j_rest):
    """Subject up / forward / right unit axes in the SMPL canonical rest frame (from rest geometry)."""
    j_rest = np.asarray(j_rest, np.float64)
    up = _unit(j_rest[15] - j_rest[0])                          # pelvis -> head
    right = j_rest[2] - j_rest[1]                               # left hip -> right hip = subject RIGHT
    right = _unit(right - right.dot(up) * up)
    fwd = np.cross(up, right)
    if fwd.dot(j_rest[10] - j_rest[7]) < 0:                     # orient to toe-forward
        fwd = -fwd
    return up, _unit(fwd), right


def anatomical_frame(kind, j_rest, canon, pj, dj):
    """Constant relabel M (3,3) from the SMPL bone frame to the spec anatomical frame A.

    Columns are [X_A | Y_A | Z_A] in bone/canonical coords. limb: Y=proximal long axis, X=anterior
    (body-forward, orthogonalized), Z=X x Y. foot: X=anterior(toe), Y=up(toward ankle), Z=X x Y.
    (Identical construction to the PRISM measured generator.)
    """
    j_rest = np.asarray(j_rest, np.float64)
    up, fwd, _right = canon
    if kind == "limb":
        yA = _unit(j_rest[pj] - j_rest[dj])
        xA = _unit(fwd - fwd.dot(yA) * yA)
        zA = np.cross(xA, yA)
    else:                                                       # foot
        xA = _unit(fwd)
        yA = _unit(up - up.dot(xA) * xA)
        zA = np.cross(xA, yA)
    return np.stack([xA, yA, zA], axis=1)


def fk_global_rotation(local_R):
    """[T,24,3,3] local rotation matrices -> [T,24,3,3] global rotations (parent-accumulate)."""
    local_R = np.asarray(local_R, np.float64)
    out = np.empty_like(local_R)
    out[:, 0] = local_R[:, 0]
    for j in range(1, 24):
        out[:, j] = out[:, int(SMPL24_PARENTS[j])] @ local_R[:, j]
    return out


def angular_velocity_deg_s(R, dt):
    """[N,3,3] world orientation sequence -> body-frame angular velocity [N,3] deg/s (rotation-log)."""
    Rp = project_so3(R)
    n = Rp.shape[0]
    omega = np.zeros((n, 3), np.float64)
    if n >= 3:
        rel_c = np.swapaxes(Rp[:-2], -1, -2) @ Rp[2:]
        omega[1:-1] = Rotation.from_matrix(rel_c).as_rotvec() / (2.0 * dt)
    if n >= 2:
        omega[0] = Rotation.from_matrix(np.swapaxes(Rp[0], -1, -2) @ Rp[1]).as_rotvec() / dt
        omega[-1] = Rotation.from_matrix(np.swapaxes(Rp[-2], -1, -2) @ Rp[-1]).as_rotvec() / dt
    return np.degrees(omega)


def specific_force(acc_world, R_world_from_sensor):
    a = np.asarray(acc_world, np.float64) - GRAVITY_WORLD
    R = np.asarray(R_world_from_sensor, np.float64)
    return np.einsum("nji,nj->ni", R, a)  # R[n].T @ a[n]


def butter_filtfilt(x):
    """4th-order 7 Hz low-pass, zero-phase. Copy-through for sequences too short to pad-filter."""
    x = np.asarray(x, np.float64)
    m = x.shape[0]
    if m <= FILTER_PADLEN + 1:
        return x.copy()
    b, a = signal.butter(FILTER_ORDER, FILTER_CUTOFF_HZ, btype="low", fs=FILTER_FS, output="ba")
    return signal.filtfilt(b, a, x, axis=0, padtype=FILTER_PADTYPE,
                           padlen=min(FILTER_PADLEN, m - 1), method="pad")


def centered_diff(x, dt):
    x = np.asarray(x, np.float64)
    r = np.empty_like(x)
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


def smpl_stature_m(model, betas):
    """SMPL rest-pose stature estimate: vertical (Y-up canonical) extent of the shaped vertices, in m."""
    n = model["shapedirs"].shape[2]
    b = np.zeros(n, np.float64)
    b[: min(n, len(betas))] = np.asarray(betas, np.float64)[: min(n, len(betas))]
    v = model["v_template"] + np.einsum("vij,j->vi", model["shapedirs"], b)
    return float(v[:, 1].max() - v[:, 1].min())


def sanitize_dirname(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(s))


# --------------------------------------------------------------------------
# 4. Build one sequence -> (small, large, anthro, manifest)
# --------------------------------------------------------------------------
def model_gender_letter(gender):
    """The letter `build` selects the body model with. Neutral AMASS sequences get the male model,
    as they always have; the fit group keys on this so it pools sequences posed on one skeleton."""
    return "F" if gender.startswith("f") else "M"


def fit_group_of(spec):
    """The fit group a motion sequence belongs to: (group name, model gender letter, betas).

    A group is what one set of frozen-joint constants may be shared across: one subject, one body
    model, one set of MoSh betas. AMASS stores betas per sequence, so a subject whose sequences
    disagree on betas is split, and the group name carries a betas digest to say so. Reads only
    the two small fields; the poses are loaded per group, not all at once."""
    with np.load(spec.source_npz, allow_pickle=False) as d:
        betas = np.asarray(d["betas"], np.float64)
        gender = normalize_gender(d["gender"]) if "gender" in d.files else "neutral"
    letter = model_gender_letter(gender)
    digest = hashlib.sha256(np.ascontiguousarray(betas, "<f8").tobytes()).hexdigest()[:12]
    return f"{spec.subject_id}|{letter}|betas:{digest}", letter, betas


def build(spec, reduction=None):
    """One sequence -> (small, large, anthro, manifest).

    ``reduction`` is the subject's fit (``reduced_model.SubjectFit``); the batch runner pools the
    subject's sequences first. Without it the constants are fitted on this sequence alone, and
    every record says ``take`` scope."""
    d = load_amass_safe(spec.source_npz, dataset=spec.dataset)
    src_fps = d["fps"]
    gender = d["gender"]
    betas = d["betas"]
    n_src = d["poses"].shape[0]

    p24 = build_pose24(d["poses"])                                     # [N,24,3] axis-angle
    local_full = Rotation.from_rotvec(p24.reshape(-1, 3)).as_matrix().reshape(n_src, 24, 3, 3)
    local = resample_rotations(local_full, src_fps, TARGET_RATE_HZ)    # [M,24,3,3]
    trans = resample_trans(d["trans"], src_fps, TARGET_RATE_HZ)        # [M,3]
    m = local.shape[0]
    if m < 4:
        raise ValueError("sequence too short after resample (M=%d): %s"
                         % (m, spec.relative_path or _source_label(spec.source_npz)))
    dt = 1.0 / TARGET_RATE_HZ

    gR = fk_global_rotation(local)                                     # [M,24,3,3] world orientations
    model = anthro_smpl.load_smpl_model_for_gender(model_gender_letter(gender))
    j_rest = anthro_smpl.rest_joints(model, betas)                     # (24,3)
    Jp = anthro_smpl.fk_positions_batch(j_rest, local)                 # [M,24,3] offsets, pelvis at ORIGIN
    world = trans[:, None, :] + Jp                                     # [M,24,3] world joint centres; pelvis = trans

    # ---- time / identity ----
    timestamps_s = np.arange(m, dtype=np.float64) * dt
    timestamps_sha256 = sha256_hex(np.ascontiguousarray(timestamps_s, "<f8").tobytes())
    take_sha256 = _sha256_file(spec.source_npz)
    canon_cfg = {"kind": "amass_faithful_synth", "frame_count": m, "target_rate_hz": TARGET_RATE_HZ,
                 "source_fps": src_fps, "fps_source": d["fps_source"], "resample": "slerp_local+linear_trans",
                 "frame_convention": "amass_world_native_Zup", "gravity_world_m_s2": GRAVITY_WORLD.tolist(),
                 "chest_alpha": CHEST_ALPHA, "filter": "butter4_7hz_filtfilt_odd15",
                 "angular_velocity": "rotation_log", "smpl_fk": "smpl24_pose_chain",
                 "imu_synthesis": "fk_orientation+doublediff_accel+rotationlog_gyro"}
    canon_cfg_sha256 = sha256_hex(canonical_json_bytes(canon_cfg))
    preimage = ["amass", spec.dataset, spec.subject_id, spec.sequence_id, take_sha256,
                canon_cfg_sha256, CONTRACT_ID, CONTRACT_VERSION, str(TARGET_RATE_HZ)]
    pair_id = sha256_hex(canonical_json_bytes(preimage))

    # ---- SMALL: 8-channel synthetic IMU (all sites FK-derived) ----
    # back_T4 axial position = non-rigid chest proxy: pelvis + alpha*(head - pelvis).
    chest_pos = world[:, 0] + CHEST_ALPHA * (world[:, 15] - world[:, 0])
    imu_orientation = np.zeros((m, N_SENSORS, 4), np.float32)
    imu_acceleration = np.zeros((m, N_SENSORS, 3), np.float32)
    imu_angular_velocity = np.zeros((m, N_SENSORS, 3), np.float32)
    imu_valid_mask = np.ones((m, N_SENSORS), bool)
    imu_confidence = np.zeros((m, N_SENSORS), np.float32)
    imu_orientation_absolute_heading = np.array([True] * N_BODY + [False] * len(FOOT_CODES))
    site_prov = {}

    def emit_site(j, R_full, pos_world, conf, prov):
        pos_filt = butter_filtfilt(pos_world)
        vel = butter_filtfilt(centered_diff(pos_filt, dt))
        acc_world = centered_diff(vel, dt)
        imu_orientation[:, j, :] = rotmat_to_quat_wxyz(R_full).astype(np.float32)
        imu_acceleration[:, j, :] = specific_force(acc_world, R_full).astype(np.float32)
        imu_angular_velocity[:, j, :] = angular_velocity_deg_s(R_full, dt).astype(np.float32)
        imu_confidence[:, j] = conf
        site_prov[SENSOR_CODES[j]] = prov

    canon = canonical_axes(j_rest)
    for j, code in enumerate(SENSOR_CODES):
        jj, kind, pj, dj, side = SITE_GEOM[code]
        relabel = anatomical_frame(kind, j_rest, canon, pj, dj) @ (_ROT_Y_180 if side == "L" else np.eye(3))
        R_S = np.einsum("nij,jk->nik", gR[:, jj], relabel)   # SMPL segment frame -> spec sensor frame S
        pos = chest_pos if code == "back_T4" else world[:, SITE_POS_JOINT[code]]
        conf = SITE_CONFIDENCE.get(code, 0.6)
        prov = ("synthetic_fk_specS:back_T4=spine3_ori+chest_axial_proxy(pelvis+2/3(head-pelvis))"
                if code == "back_T4" else
                "synthetic_fk_specS:smpl_segment_ori(joint %d)+doublediff_world_accel(joint %s)"
                % (jj, SITE_POS_JOINT.get(code)))
        emit_site(j, R_S, pos, conf, prov)

    small = {
        # String arrays are numpy 'U' (ADR-0040 D2): an object array forces allow_pickle on the reader.
        "sensor_codes": np.array(SENSOR_CODES, dtype="U"),
        "imu_orientation": imu_orientation,
        "imu_acceleration": imu_acceleration,
        "imu_angular_velocity": imu_angular_velocity,
        "imu_valid_mask": imu_valid_mask,
        "imu_confidence": imu_confidence,
        "imu_orientation_absolute_heading": imu_orientation_absolute_heading,
        "mount_id": np.array([f"amass_faithful::{c}" for c in SENSOR_CODES], dtype="U"),
        "small_mode": "synthetic_from_smpl", "axis_convention": "spec_S_v2",
        "pair_id": pair_id, "timestamps_s": timestamps_s,
        "frame_count": np.int64(m),
    }

    # ---- LARGE: pelvis(root) + 17 joints (reduced-model), identical convention to PRISM ----
    pelvis_position = world[:, 0].astype(np.float64)   # == trans (pelvis anchored at origin in Jp); self-consistent
                                                       # with anthro_reconstruction (offsets are pelvis-relative)
    # [M,24,4] lossless. Per-joint so the (w,x,y,z) sign-continuity in rotmat_to_quat_wxyz (2-D path)
    # is applied along time for every joint, not skipped as it would be on the bulk [M,24,3,3] array.
    smpl_global_orientation = np.stack(
        [rotmat_to_quat_wxyz(gR[:, j]) for j in range(24)], axis=1).astype(np.float32)
    if reduction is None:
        reduction = reduced_model.fit_subject(
            [p24], j_rest, reduced_model.settings_for_source("amass"),
            group=f"{spec.subject_id}/{spec.sequence_id}", basis="this sequence, native rate",
            scope="take")
    fixed_C = reduction.constants
    Lrefit = reduced_model.reduce_local(local, fixed_C)

    root_velocity = np.zeros((m, 3), np.float64)
    root_velocity[1:] = (pelvis_position[1:] - pelvis_position[:-1]) / dt
    root_velocity = root_velocity.astype(np.float32)

    joint_rotation = np.zeros((m, 18, 4), np.float32)
    joint_velocity = np.zeros((m, 18, 3), np.float32)
    joint_rotation[:, 0, :] = rotmat_to_quat_wxyz(gR[:, 0]).astype(np.float32)
    joint_velocity[:, 0, :] = angular_velocity_deg_s(gR[:, 0], dt).astype(np.float32)
    for k in range(1, 18):
        idx = int(SMPL18_INDICES[k])
        joint_rotation[:, k, :] = rotmat_to_quat_wxyz(Lrefit[:, idx]).astype(np.float32)
        joint_velocity[:, k, :] = angular_velocity_deg_s(Lrefit[:, idx], dt).astype(np.float32)

    large = {
        "joint_names": np.array(JOINT18_NAMES, dtype="U"),
        "root_velocity": root_velocity,
        "joint_rotation": joint_rotation,
        "joint_velocity": joint_velocity,
        "pelvis_position_world_aux": pelvis_position.astype(np.float32),
        "smpl_global_orientation_world": smpl_global_orientation,   # AMASS world frame (was prism_world in PRISM)
        "pair_id": pair_id, "timestamps_s": timestamps_s, "frame_count": np.int64(m),
    }

    # ---- ANTHRO: SMPL rest skeleton constants + fixed-joint weld (AMASS-specific overrides) ----
    subj_info = {"gender": "F" if gender.startswith("f") else ("M" if gender.startswith("m") else "U")}
    anthro = anthro_smpl.compute_anthro(
        model, betas, subj_info, pair_id, fixed_joint_local_R=fixed_C,
        fixed_joint_provenance=reduced_model.anthro_provenance(reduction))
    anthro["namespace"] = "anthro_reference/amass_smpl_v1"
    # AMASS stores MoSh betas per sequence. The frozen-joint constants are shared by every sequence
    # of the subject posed on the same betas and body model (one fit group); a take-scope fit shares
    # nothing, and the label says which.
    if reduction.record["scope"] == "subject":
        anthro["subject_scope"] = "subject_constant_trial_frame_invariant"
        anthro["subject_scope_note"] = (
            "betas are the sequence's MoSh betas; the frozen-joint constants are fitted once over "
            f"the {reduction.record['takes']} sequence(s) of this subject that share these betas and "
            "body model, and are the same in each of them.")
    else:
        anthro["subject_scope"] = "sequence_constant_frame_invariant"
        anthro["subject_scope_note"] = ("betas and the frozen-joint constants are this sequence's own; "
                                        "nothing here is shared with the subject's other sequences.")
    anthro["subject_id"] = spec.subject_id
    anthro["source_asset_id"] = spec.relative_path or ("extracted/amass/%s/%s.npz" % (spec.dataset, spec.sequence_id))
    anthro["gender"] = gender
    anthro["height"] = np.float32(smpl_stature_m(model, betas))
    anthro["height_provenance"] = ("estimated:smpl_rest_stature_yup (shaped-vertex Y-extent; AMASS has "
                                   "no measured subject height)")
    anthro["body_mass"] = np.float32(np.nan)
    anthro["body_mass_provenance"] = "unavailable:amass_no_subject_mass"
    anthro["betas_provenance"] = ("source_derived:amass npz betas (16-dim); the clean SMPL model carries "
                                  "10 shape dims, so rest geometry uses the first 10 (smpl10_beta_truncation)")
    # compute_anthro stamps PRISM's string on every source; this generator's sex comes from the npz.
    anthro["sex_provenance"] = SEX_PROVENANCE

    # ---- validation ----
    v = {}
    qn = np.linalg.norm(imu_orientation, axis=2)
    v["imu_quat_norm_max_dev"] = float(np.max(np.abs(qn - 1)))
    v["joint_rotation_quat_norm_max_dev"] = float(np.max(np.abs(np.linalg.norm(joint_rotation, axis=2) - 1)))
    v["timestamp_step_max_err"] = float(np.max(np.abs(np.diff(timestamps_s) - dt)))
    # specific-force round-trip on a body site (occiput): a = R_S f + g must match the differentiated accel.
    occ = SENSOR_CODES.index("occiput")
    ojj, okind, opj, odj, _os = SITE_GEOM["occiput"]
    Rocc_S = np.einsum("nij,jk->nik", gR[:, ojj], anatomical_frame(okind, j_rest, canon, opj, odj))
    pos_filt = butter_filtfilt(world[:, SITE_POS_JOINT["occiput"]])
    a_true = centered_diff(butter_filtfilt(centered_diff(pos_filt, dt)), dt)
    f_occ = imu_acceleration[:, occ, :].astype(np.float64)
    a_recon = np.einsum("nij,nj->ni", Rocc_S, f_occ) + GRAVITY_WORLD
    v["specific_force_inverse_max_abs_err_m_s2"] = float(np.max(np.abs(a_recon - a_true)))
    pos_recon = pelvis_position[0] + dt * np.cumsum(root_velocity.astype(np.float64), axis=0)
    v["root_velocity_cumsum_recon_max_abs_err_m"] = float(np.max(np.abs(pos_recon - pelvis_position)))
    v["accel_filter_edge_transient_frames"] = int(FILTER_PADLEN)
    v["accel_boundary_note"] = (
        "per-site acceleration is a filtered double-difference; the first/last "
        "~accel_filter_edge_transient_frames carry a butter-filtfilt edge transient + one-sided boundary "
        "derivative (mild on real mocap, a few percent of |a| near gravity). PRISM avoided this by interior "
        "windowing; AMASS sequences are standalone, so their boundaries are included.")

    raw_pos = anthro_smpl.fk_positions_batch(j_rest, local)
    ref_pos = anthro_smpl.fk_positions_batch(j_rest, Lrefit)
    qc = np.linalg.norm(raw_pos - ref_pos, axis=2)
    aff = list(reduced_model.AFFECTED_BY_FREEZE)
    weld = {}
    for fname, js in FIXED_JOINT_DOWNSTREAM.items():
        p95 = float(np.percentile(qc[:, js], 95))
        weld[fname] = {"p95_m": p95, "max_m": float(qc[:, js].max()),
                       "exceeds_tau": bool(p95 > RIGID_WELD_TAU_P95_M)}
    v["reduced_model_fit_residual_m"] = {
        "overall_max": float(qc.max()), "affected_mean": float(qc[:, aff].mean()),
        "per_joint_max": {anthro_smpl.JOINT24_NAMES[j]: float(qc[:, j].max()) for j in aff},
        "rigid_weld_threshold_p95_m": RIGID_WELD_TAU_P95_M, "fixed_joint_weld_residual": weld,
        "note": "reduced-model (Large 18 refit + Anthro 4 fixed constants) vs raw SMPL joint-center "
                "position residual; distal global orientation exact (sec-anthro-fit). This is the "
                "accepted rigid-weld cost, reported as a quality metric (no promotion).",
    }

    manifest = {
        "spec_id": SPEC_ID, "spec_version": SPEC_VERSION,
        "distribution_scope": "internal_only", "artifact_class": "experimental_non_candidate",
        "artifact_label": "PoC FAITHFUL AMASS synthetic reference (SMPL-derived synthetic IMU; independent "
                          "replica of the PRISM faithful recipe). NOT canonical, NOT contract-compliant, "
                          "NOT quality-gate PASS. INTERNAL-ONLY.",
        "quality_gate": "NOT_EVALUATED",
        "up_axis": "z",
        "source_attribution": attribution_block(),
        "excluded_modalities": ["emg"],
        "excluded_modality_reasons": {"emg": "not_in_synthetic_artifact_scope"},
        "governance_disposition": {
            "field_registry_activation_state": "non_authorizing",
            "not_the_governed_pipeline": True,
            "generation_basis": "experimental replica (poc-demo faithful pipeline) mirroring the "
                                "PRISM faithful recipe; the governed candidate pipeline's source holds and "
                                "SPEC_DRIFT gating are intentionally out of scope for this "
                                "experimental_non_candidate replica, as with generate_prism_faithful.py.",
        },
        "determinism": "small/large/anthro npz are byte-deterministic for a fixed (sequence, code); "
                       "manifest.generated_utc is a wall-clock timestamp and is the sole non-deterministic field.",
        "run_id": "amass-%s-%s-%s-faithful" % (spec.dataset, sanitize_dirname(spec.subject_id), spec.sequence_id),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"source_name": "amass", "dataset": spec.dataset, "subject_id": spec.subject_id,
                   "sequence_id": spec.sequence_id, "source_asset_sha256": take_sha256,
                   "relative_path": spec.relative_path, "native_frames_total": n_src,
                   "native_rate_hz": src_fps, "framerate_source": d["fps_source"],
                   "field_registry": AMASS_FIELD_REGISTRY,
                   "note": "AMASS ships only SMPL-H pose params; every IMU channel is synthesized from the "
                           "SMPL body (no measured imu_gt/insole exist)."},
        "resample": {"source_fps": src_fps, "target_fps": TARGET_RATE_HZ, "frames_in": n_src,
                     "frames_out": m, "method": "per-joint Slerp (local rotations) + linear interp (trans)",
                     "noop": bool(src_fps == TARGET_RATE_HZ)},
        "window": {"frame_start": 0, "frame_end_exclusive": m, "frame_count": m},
        "identity": {"pair_id": pair_id, "contract_id": CONTRACT_ID, "contract_version": CONTRACT_VERSION,
                     "timestamps_sha256": timestamps_sha256, "target_rate_hz": TARGET_RATE_HZ,
                     "canonicalization_config_sha256": canon_cfg_sha256},
        "canonicalization_config": canon_cfg,
        "global_frame": {"convention": "AMASS world, right-handed, Z-up, gravity along -Z",
                         "conforms_to_spec_G": True,
                         "note": "empirically Z-up for every AMASS sub-dataset (head-foot dz>0)."},
        "small_sites_provenance": site_prov,
        "spec_documents": {
            "small_sensors": "small/00_sensors.qmd",
            "small_streams": "small/01_streams.qmd",
            "large_kinematics": "large/02_kinematics.qmd",
            "anthropometry": "anthro/00_anthropometry.qmd",
        },
        "large_joint_names": JOINT18_NAMES,
        "deliverables": ["small_reference.npz", "large_reference.npz", "anthro_reference.npz",
                         "smpl_root_translation.npz", "manifest.json"],
        "faithful_content": {
            "small": "8ch synthetic IMU (back_T4,wrist_l/r,shank_l/r,occiput,foot_l/r) fully FK-derived: "
                     "orientation in the spec sensor frame S (small/00_sensors.qmd: +Y=proximal, "
                     "+Z=outward lateral, +X=Y x Z; left limbs Rot(Y,180)), built as gR[:,J] @ M_J @ C_LR "
                     "from SMPL rest geometry (no imu_gt needed); specific force (gravity-included) + "
                     "rotation-log gyro in S. No measured imu_gt/insole; small_mode=synthetic_from_smpl, "
                     "axis_convention=spec_S_v2.",
            "large": "root_velocity[M,3] + joint_rotation[M,18,4] (reduced-model: spine3/shoulders refit) + "
                     "joint_velocity[M,18,3] + pelvis_position_world_aux; raw smpl_global_orientation[M,24,4] kept.",
            "anthro": "SMPL rest skeleton constants (joint_position[2,22,3], segment_length[13], "
                      "fixed_joint_rotation[4,4]) + betas(16, model uses 10) + gender. height=SMPL stature "
                      "estimate; body_mass unavailable (AMASS has no subject anthropometry).",
            "development_reference": "NONE — AMASS has no measured GRF/insole/kinetics.",
        },
        "anthro_reconstruction": reduced_model.reconstruction_block(
            reduction, "large.smpl_global_orientation_world[M,24,4] + resampled raw SMPL poses"),
        "unavailable_or_not_applied": [
            "development_reference / measured GRF / CoP / insole (AMASS has none)",
            "measured subject height & body_mass (height is an SMPL stature estimate; mass is NaN)",
            "joint_angles_jcs_deg (needs ISB/JCS convention)",
            "physics-informed GRF / inverse dynamics / moments / powers / COM",
            "sensor mount extrinsics (segment->sensor lever arm)",
            "quality gates",
        ],
        "validation": v,
        "safety": {"npz_loader": "np.load(allow_pickle=False); audited core fields only; no code execution"},
    }
    return small, large, anthro, manifest


def write_outputs(out_dir, small, large, anthro, manifest):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "small_reference.npz"), **small)
    np.savez(os.path.join(out_dir, "large_reference.npz"), **large)
    np.savez(os.path.join(out_dir, "anthro_reference.npz"), **anthro)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import glob
    _amass_root = str(paths.source_dir("amass"))
    _fs = sorted(glob.glob(os.path.join(_amass_root, "KIT", "**", "*_poses.npz"), recursive=True))
    if _fs:
        _p = _fs[0]
        _spec = AmassSeqSpec(dataset="KIT", subject_id="amass_KIT_demo",
                             sequence_id=os.path.splitext(os.path.basename(_p))[0], source_npz=_p,
                             relative_path=paths.logical_source_id("amass", _p, root=_amass_root))
        _small, _large, _anthro, _manifest = build(_spec)
        print("OK", _spec, "M=", int(_small["frame_count"]))
        print("validation:", json.dumps(_manifest["validation"], indent=2)[:1200])
