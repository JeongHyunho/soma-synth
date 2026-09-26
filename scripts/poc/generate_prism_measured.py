"""PRISM small revision: simulate the REAL measured IMU from PRISM's PHYSICAL `imu.*`.

The faithful generator builds `small` from PRISM's kinematic GT `imu_gt.*` (small_mode="ideal", a
gravity-free segment-frame signal). This module rebuilds ONLY the `small` deliverable from the
**physical** sensor stream `imu.*`, producing a gravity-included accelerometer signal expressed in the
spec sensor frame S. large / anthro / development_reference are reused unchanged from the faithful build.

Design: PRISM measured-physical small design note (parent project record, 2026-08-13)
Key points (all verified against the take):
  * PRISM physical acc is gravity-free linear accel; a real accelerometer reads specific force INCLUDING
    gravity, so imu_acceleration = R_S2G^T (a_world - g), g=[0,0,-9.80665] (|.|~9.8).
  * The physical `imu.ori` is slightly NON-orthonormal (Foot/Knee/Wrist det 0.96-1.02); it is projected
    to SO(3) so the deliverable is internally consistent (orientation quat, specific force, gyro agree).
  * Body sites (wrist/knee/head) have no stored gyro -> derive from the (calibrated physical) orientation;
    feet store gyr_local_raw (deg/s) -> rotate the physical sensor frame into S.
  * SPEC AXES (small/00_sensors.qmd, v2): sensor frame S has +Y=proximal, +Z=outward lateral, +X=Y x Z,
    and left limbs are mounted 180 deg about Y (left +X=posterior). We build the anatomical frame A per
    segment from SMPL rest geometry (Y=proximal long axis, X=body-anterior, Z=X x Y), carried into world
    by the SMPL global joint rotation; S = A for trunk/head/right limbs, S = A.Rot(Y,180) for left limbs.
    The constant relabel imu_gt-frame -> S is fit per take (imu_gt is EXACTLY a rigid relabel of the SMPL
    bone frame: residual 0.0 deg). Verified: right/trunk/head +Z -> subject-right, left +Z -> subject-left.
  * back_T4 has NO PRISM trunk sensor -> the non-physical spine3 proxy is retained (flagged), but relabeled
    into the same spec back frame (X=forward, Y=up, Z=subject-right).
  * R_Wrist has 56% gap-filled (synthetic) frames -> exposed per-frame in imu_synth_mask.
"""
from __future__ import annotations

import functools
import importlib.util as _ilu
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

# Reuse the faithful generator (build + pure helpers); it in turn imports anthro_smpl.
_FAITHFUL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_prism_faithful.py")
_fspec = _ilu.spec_from_file_location("generate_prism_faithful", _FAITHFUL_PATH)
faithful = _ilu.module_from_spec(_fspec)
_fspec.loader.exec_module(faithful)

# The attribution registry is a config value, never a code constant; this generator writes
# source_attribution, up_axis and resample itself.
_SRC_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
from soma_synth.contracts import source_attribution as _attribution  # noqa: E402


@functools.lru_cache(maxsize=1)
def attribution_block():
    """manifest.source_attribution for prism, read from the registry rather than typed here."""
    return dict(_attribution.resolved_for("prism"))


RELABEL_EVIDENCE = "PRISM insole-heading investigation (parent project record, 2026-09-09) sec 5"

GRAVITY = faithful.GRAVITY_PRISM_WORLD
SENSOR_CODES = faithful.SENSOR_CODES
N_SENSORS = 8
BACK_JOINT = faithful.CHEST_SPINE3_JOINT       # SMPL spine3 (9): carrier of the small_ideal back_T4 frame

# spec channel -> PRISM physical `imu` site. back_T4 has no physical trunk sensor.
SITE_PHYS = {
    "wrist_l": "L_Wrist", "wrist_r": "R_Wrist", "shank_l": "L_Knee", "shank_r": "R_Knee",
    "occiput": "Head", "foot_l": "L_Foot", "foot_r": "R_Foot",
}
FOOT_CODES = ("foot_l", "foot_r")
SYNTH_SITE = {"wrist_l": "L_Wrist", "wrist_r": "R_Wrist"}   # gap-filled physical frames live here

#: The insole reports `gyr_local_raw` on axes that are a fixed relabel away from the body frame
#: its own `ori_world` maps out of: (a, b, c) -> (-a, c, b). Not an assumption -- solved for by
#: Kabsch on all 150 takes x 2 feet, which converge on this one matrix (median 1.0 deg from it,
#: max 5.5, 300/300 within 10), while the six body channels return the identity because their
#: gyro is derived from their orientation. Applying it makes a healthy foot's gyro integrate to
#: its own shipped orientation within 2.9 deg, against 134.6 without.
#: Evidence: PRISM insole-heading investigation (parent project record, 2026-09-09) sec 5.
GYR_LOCAL_AXIS_RELABEL = np.array(
    [[-1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0]]
)

# Per-channel segment geometry for building the anatomical frame A (spec small/00_sensors.qmd).
#   J     = SMPL joint whose global rotation carries the segment frame (verified: imu_gt is a rigid
#           constant relabel of gR[:,J], residual 0.0 deg — the wrist maps to the forearm/elbow joint).
#   kind  = "limb" (Y=proximal long axis, X=anterior) or "foot" (X=toe-anterior, Y=up-toward-ankle).
#   pj/dj = proximal/distal SMPL joints defining the long axis (rest geometry).
#   side  = "L" left limb (S = A.Rot(Y,180)), "R"/"M" no flip (S = A).
SITE_GEOM = {
    "back_T4": (BACK_JOINT, "limb", 12, 9, "M"),    # trunk: up = neck(12)-spine3(9)
    "wrist_l": (18, "limb", 18, 20, "L"),           # forearm: elbow(prox)-wrist(distal)
    "wrist_r": (19, "limb", 19, 21, "R"),
    "shank_l": (4, "limb", 4, 7, "L"),              # shank: knee(prox)-ankle(distal)
    "shank_r": (5, "limb", 5, 8, "R"),
    "occiput": (15, "limb", 15, 12, "M"),           # head: vertex-up = head(15)-neck(12)
    "foot_l": (10, "foot", 10, 7, "L"),
    "foot_r": (11, "foot", 11, 8, "R"),
}
_ROT_Y_180 = np.diag([-1.0, 1.0, -1.0])             # Rot about Y by 180 deg (left-limb mount rule)


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


def prepare_take_geometry(raw, gender):
    """(gR [N,24,3,3] SMPL global rotations, j_rest [24,3], canonical axes) for the take."""
    poses = np.asarray(raw["smpl_params"]["poses"], np.float64).reshape(-1, 24, 3)
    gR = faithful.smpl24_global_rotation(poses)
    betas0 = np.asarray(raw["smpl_params"]["betas"], np.float64)[0]
    model = faithful.anthro_smpl.load_smpl_model_for_gender(gender)
    j_rest = faithful.anthro_smpl.rest_joints(model, betas0)
    return gR, j_rest, canonical_axes(j_rest)


def _static_mask(raw, n):
    """Quasi-static frames (low pelvis speed) — the mount calibration is fit over these."""
    pel = np.asarray(raw["imu_gt"]["Pelvis"]["pos_world"], np.float64)
    vel = np.zeros(n)
    vel[1:] = np.linalg.norm(np.diff(pel, axis=0), axis=1) * faithful.TARGET_RATE_HZ
    return vel < np.percentile(vel, 20)


def compute_site_relabels(raw, gR, j_rest):
    """Per-channel constant relabel to the spec sensor frame S (see the module docstring).

    Physical sites: C_site (imu_gt-frame -> S) = median_t(imu_gt.ori^T @ A) @ C_LR, applied inside
      fit_site_calibration as the C term. back_T4: M_back (bone -> A), applied to the spine3 proxy frame.
    Returns {code: (3,3)}.
    """
    canon = canonical_axes(j_rest)
    st = _static_mask(raw, gR.shape[0])
    rel = {}
    for code, (J, kind, pj, dj, side) in SITE_GEOM.items():
        MJ = anatomical_frame(kind, j_rest, canon, pj, dj)
        c_lr = _ROT_Y_180 if side == "L" else np.eye(3)
        if code not in SITE_PHYS:                              # back_T4 proxy: relabel gR[:,spine3] -> A
            rel[code] = MJ @ c_lr
            continue
        A = np.einsum("nij,jk->nik", gR[:, J], MJ)            # R_A2world = gR[:,J] @ MJ
        Rgt = faithful.project_so3(np.asarray(raw["imu_gt"][SITE_PHYS[code]]["ori_world"], np.float64))
        c_gt2a = Rotation.from_matrix(faithful.project_so3(
            np.einsum("nji,njk->nik", Rgt[st], A[st]))).mean().as_matrix()
        rel[code] = c_gt2a @ c_lr
    return rel


def fit_site_calibration(imu_ori, gt_ori, C=None):
    """R_cal (3,3): physical sensor frame S_phys -> spec sensor frame S.

    median over time of (imu.ori^T @ imu_gt.ori), then the per-site relabel C (imu_gt-frame -> S).
    Rigid-mount assumption: the physical<->imu_gt offset is ~constant per site.
    """
    C = np.eye(3) if C is None else np.asarray(C, np.float64)
    imu_ori = faithful.project_so3(np.asarray(imu_ori, np.float64))
    gt_ori = faithful.project_so3(np.asarray(gt_ori, np.float64))
    rel = np.swapaxes(imu_ori, -1, -2) @ gt_ori
    r_med = Rotation.from_matrix(faithful.project_so3(rel)).mean().as_matrix()
    return r_med @ C


def measured_small(raw, small_ideal, relabels, gR, win_start, win_end, dt):
    """Build the 8-channel PHYSICAL-measured small dict for a window (spec sensor frame S)."""
    imu, gt = raw["imu"], raw["imu_gt"]
    t = win_end - win_start
    ori = np.zeros((t, N_SENSORS, 4), np.float32)
    acc = np.zeros((t, N_SENSORS, 3), np.float32)
    gyr = np.zeros((t, N_SENSORS, 3), np.float32)
    valid = np.ones((t, N_SENSORS), bool)
    conf = np.zeros((t, N_SENSORS), np.float32)
    synth = np.zeros((t, N_SENSORS), bool)
    q_anat = np.zeros((N_SENSORS, 4), np.float32)
    gyro_prov = [""] * N_SENSORS

    def win(a):
        return np.asarray(a, np.float64)[win_start:win_end]

    for j, code in enumerate(SENSOR_CODES):
        c_site = np.asarray(relabels[code], np.float64)
        if code == "back_T4":
            # No PRISM trunk IMU -> retain the faithful spine3 proxy, but relabel into the spec back
            # frame (X=forward, Y=up, Z=subject-right) so axes are spec-consistent. Flagged non-physical.
            r_back = np.einsum("nij,jk->nik", faithful.project_so3(win(gR[:, BACK_JOINT])), c_site)
            ori[:, j, :] = faithful.rotmat_to_quat_wxyz(r_back).astype(np.float32)
            # acc/gyro of a constant right-multiplied frame rotate by c_site^T (reuse the proxy values).
            acc[:, j, :] = np.einsum("ji,nj->ni", c_site, small_ideal["imu_acceleration"][:, j, :]).astype(np.float32)
            gyr[:, j, :] = np.einsum("ji,nj->ni", c_site, small_ideal["imu_angular_velocity"][:, j, :]).astype(np.float32)
            conf[:, j] = 0.4
            q_anat[j] = np.array([1.0, 0.0, 0.0, 0.0], np.float32)   # no physical sensor
            gyro_prov[j] = "proxy_non_physical:spine3+axial(relabeled_to_spec_back)"
            continue
        site = SITE_PHYS[code]
        r_o = faithful.project_so3(np.asarray(imu[site]["ori_world"], np.float64))   # SO(3)-projected physical ori
        r_g = np.asarray(gt[site]["ori_world"], np.float64)
        r_cal = fit_site_calibration(r_o, r_g, c_site)                # S_phys -> spec S
        # q_anatomical_from_sensor = A <- S_phys = C_LR @ r_cal^T (C_LR folded into c_site already: A=S here)
        q_anat[j] = faithful.rotmat_to_quat_wxyz(r_cal.T).astype(np.float32)
        r_s2g = win(r_o) @ r_cal                                      # physical ori, calibrated to spec S
        ori[:, j, :] = faithful.rotmat_to_quat_wxyz(r_s2g).astype(np.float32)
        # specific force in S: R^T (a_world - g); gravity-included (~9.8) per a real accelerometer
        acc[:, j, :] = faithful.specific_force(win(imu[site]["acc_world_filt"]), r_s2g).astype(np.float32)
        if code in FOOT_CODES:
            gl = win(imu[site]["gyr_local_raw"])                      # deg/s in the physical insole frame
            # r_cal^T alone assumes gyr_local_raw shares the body frame imu.ori maps out of. It
            # does not: the two are a fixed axis relabel apart, and the omitted M made the shipped
            # foot gyro disagree with the shipped foot orientation in every take. See
            # the PRISM insole-heading investigation (parent project record, 2026-09-09) sec 5.
            gyr[:, j, :] = np.einsum(
                "ji,jk,nk->ni", r_cal, GYR_LOCAL_AXIS_RELABEL, gl
            ).astype(np.float32)                                      # r_cal^T @ M @ gl -> spec S
            gyro_prov[j] = "measured:gyr_local_raw"
        else:
            gyr[:, j, :] = faithful.angular_velocity_deg_s(r_s2g, dt).astype(np.float32)
            gyro_prov[j] = "derived_from_orientation:rotation_log"
        conf[:, j] = 0.7

    sif = raw["info"]["data_info"]["synth_imu_frames"]
    for code, site in SYNTH_SITE.items():
        j = SENSOR_CODES.index(code)
        fr = np.asarray(sif.get(site, []), np.int64)
        fr = fr[(fr >= win_start) & (fr < win_end)] - win_start
        synth[fr, j] = True

    heading = np.array([True] * (N_SENSORS - len(FOOT_CODES)) + [False] * len(FOOT_CODES))
    return {
        "sensor_codes": np.array(SENSOR_CODES, dtype=object),
        "imu_orientation": ori, "imu_acceleration": acc, "imu_angular_velocity": gyr,
        "imu_valid_mask": valid, "imu_confidence": conf, "imu_synth_mask": synth,
        "imu_orientation_absolute_heading": heading,
        "q_anatomical_from_sensor": q_anat,
        "p_segment_to_sensor_m": np.zeros((N_SENSORS, 3), np.float32),  # sites at segment origin (unavailable)
        "gyro_provenance": np.array(gyro_prov, dtype=object),
        "mount_id": np.array([f"prism_measured::{c}" for c in SENSOR_CODES], dtype=object),
        "small_mode": "measured_physical", "axis_convention": "spec_S_v2",
        "pair_id": small_ideal["pair_id"], "timestamps_s": small_ideal["timestamps_s"],
        "frame_count": np.int64(t),
    }


def build_measured(spec, reduction=None):
    """Faithful build with `small` swapped for the PHYSICAL-measured version (Small-only revision).

    `reduction` is the subject's frozen-joint fit, passed straight to the faithful build."""
    small_ideal, large, dev, anthro, manifest = faithful.build(spec, reduction)
    raw = faithful.load_prism_safe(spec.source_pkl)
    n = int(raw["smpl_params"]["poses"].shape[0])
    win_start, win_end = spec.window if spec.window is not None else (0, n)
    dt = 1.0 / faithful.TARGET_RATE_HZ
    gR, j_rest, _canon = prepare_take_geometry(raw, spec.gender)
    relabels = compute_site_relabels(raw, gR, j_rest)
    small = measured_small(raw, small_ideal, relabels, gR, win_start, win_end, dt)

    manifest["faithful_content"]["small"] = (
        "8ch PHYSICAL-measured IMU (raw['imu']): orientation=physical imu.ori (SO(3)-projected) calibrated "
        "into the spec sensor frame S (per-segment anatomical relabel, left limbs Rot(Y,180)); "
        "imu_acceleration=R^T(a_world-g) specific force (gravity-included, ~9.8 m/s^2); "
        "foot gyro=measured gyr_local_raw (deg/s) rotated to S, body gyro=derived from orientation; "
        "back_T4=non-physical proxy (no PRISM trunk IMU), relabeled to the spec back frame. "
        "small_mode=measured_physical, axis_convention=spec_S_v2.")
    manifest["small_measured"] = {
        "source": "raw['imu'] physical sensor stream (NOT imu_gt kinematic GT)",
        "acceleration": "specific force f=R^T(a_world-g), gravity-included per real accelerometer (~9.8)",
        "orientation_projection": "physical imu.ori projected to SO(3) (physical stream is non-orthonormal)",
        "frame": "spec sensor S (small/00_sensors.qmd): +Y=proximal, +Z=outward lateral, +X=Y x Z; "
                 "left limbs mounted 180deg about Y (left +X=posterior)",
        "anatomical_frame": "A built from SMPL rest geometry (Y=proximal long axis, X=body-anterior, Z=X x Y) "
                            "carried into world by the SMPL global joint rotation; imu_gt is a rigid relabel "
                            "of that bone frame (residual 0.0 deg), so C_site is fit per take",
        "calibration": "q_anatomical_from_sensor = A <- S_phys (from median_t(imu.ori^T imu_gt.ori) and C_site)",
        "gyro": {"feet": "measured gyr_local_raw (deg/s) rotated to S",
                 "body": "derived: rotation_log of calibrated orientation"},
        "back_T4": "no PRISM trunk/T4 IMU -> non_physical spine3 proxy retained (flagged), relabeled to spec back frame",
        "synth_frames": {k: len(v) for k, v in raw["info"]["data_info"]["synth_imu_frames"].items()},
        "notes": "PRISM 100Hz; spec 1kHz grid + MTi-630 400Hz band recorded as metadata, not resampled.",
    }
    manifest["small_sites_provenance"] = {
        c: ("measured_physical:imu.%s" % SITE_PHYS[c]) if c in SITE_PHYS else "proxy_non_physical:spine3+axial"
        for c in SENSOR_CODES
    }
    # ---- faithful-v2 schema reconcile (deployed measured lineage; the ideal faithful build is untouched,
    #      so its golden stays valid; the legacy faithful-v1 bundle keeps _prism_world and is superseded) ----
    # large: canonical world key (retire the drifted _prism_world) + U-dtype joint names
    if "smpl_global_orientation_prism_world" in large:
        large["smpl_global_orientation_world"] = large.pop("smpl_global_orientation_prism_world")
    large["joint_names"] = np.asarray(large["joint_names"], dtype="U")
    # anthro: subject_scope_note is CORE at faithful-v2 (PRISM lacked it) + U-dtype string arrays
    anthro.setdefault(
        "subject_scope_note",
        "PRISM subject-constant anthropometry: identical across every frame of a take "
        "(subject_constant_trial_frame_invariant).")
    for _k in ("joint_names", "segment_names", "fixed_joint_names", "reference_poses"):
        if _k not in anthro:
            continue
        _a = np.asarray(anthro[_k])
        if _a.dtype == object and _a.size and all(isinstance(x, str) for x in _a.ravel()):
            anthro[_k] = _a.astype("U")
    # small: U-dtype string arrays
    for _k in ("sensor_codes", "mount_id", "gyro_provenance"):
        small[_k] = np.asarray(small[_k], dtype="U")
    # manifest: spec lineage (layout §6) + corrected deliverables (drop the absent README, add root translation)
    manifest["spec_id"] = "qmd_unified8_smpl18"
    manifest["spec_version"] = "faithful-v2"
    manifest["quality_gate"] = "NOT_EVALUATED"
    manifest["deliverables"] = [
        "small_reference.npz", "large_reference.npz", "anthro_reference.npz",
        "development_reference.npz", "smpl_root_translation.npz", "manifest.json",
    ]
    # The three fields ADR-0040 D1 promoted to universal, written at emit since 2026-09-15 with
    # exactly the values the 2026-09-09 backfill derived from this manifest's own contents
    # (scripts/backfill_manifest_fields.py derive_up_axis / derive_resample), so a regenerated take
    # and a backfilled one read the same.
    manifest["up_axis"] = "z"                       # global_frame.convention "Z-up", gravity [0,0,-g]
    native_rate = float(manifest["source"]["native_rate_hz"])
    target_rate = manifest["canonicalization_config"]["target_rate_hz"]
    frames_out = int(manifest["window"]["frame_count"])
    frames_in = int(manifest["source"]["native_frames_total"])
    same_grid = native_rate == float(target_rate) and frames_in == frames_out
    manifest["resample"] = {
        "source_fps": native_rate,
        "target_fps": target_rate,
        "frames_in": frames_in,
        "frames_out": frames_out,
        "method": ("none: the source grid is already the target grid, and this lineage has no "
                   "resampler" if same_grid else
                   "none: the take was windowed from the source; no rate change was applied"),
        "noop": bool(same_grid),
        "direction": "none",
    }
    manifest["source_attribution"] = attribution_block()
    # The foot gyro is rotated onto its true axes at emit (GYR_LOCAL_AXIS_RELABEL above). The
    # marker says so in the shape the 2026-09-14 rewrite tool left on the retired bundle, so
    # relabel_insole_gyro_axes.py and the heading comparison see a take that is already right and
    # never apply M a second time (M is its own inverse; a second application would undo the fix).
    manifest["small_measured"]["gyro_axis_relabel"] = {
        "applied": True,
        "channels": list(FOOT_CODES),
        "matrix": GYR_LOCAL_AXIS_RELABEL.astype(int).tolist(),
        "formula": "w_spec = r_cal^T @ M @ gyr_local_raw; r_cal = transpose(q_anatomical_from_sensor); "
                   "applied by generate_prism_measured.py at emit, not by a rewrite",
        "applied_utc": manifest["generated_utc"],
        "evidence": RELABEL_EVIDENCE,
    }
    manifest["faithful_content"]["large"] = manifest["faithful_content"]["large"].replace(
        "smpl_global_orientation_prism_world", "smpl_global_orientation_world")
    manifest["anthro_reconstruction"]["lossless_alternative"] = (
        manifest["anthro_reconstruction"]["lossless_alternative"].replace(
            "smpl_global_orientation_prism_world", "smpl_global_orientation_world"))
    return small, large, dev, anthro, manifest
