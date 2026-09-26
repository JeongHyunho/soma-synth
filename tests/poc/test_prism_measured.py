from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import numpy as np
from data_root_skips import DATA_ROOT, needs_data

SCRIPT = Path(__file__).parents[2] / "scripts" / "poc" / "generate_prism_measured.py"
PKL = DATA_ROOT / "extracted/prism/subj001/take002.pkl"
NEEDS_DATA = needs_data(PKL.exists(), "the PRISM take extracted/prism/subj001/take002.pkl")


def _mod():
    spec = importlib.util.spec_from_file_location("prism_measured", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _spec(m, window):
    return m.faithful.TakeSpec(
        subject_id="prism_subj001", take_id="take002", source_pkl=str(PKL),
        gender="M", window=window, relative_path="extracted/prism/subj001/take002.pkl")


def _npz_bytes(d):
    b = io.BytesIO()
    np.savez(b, **d)
    return b.getvalue()


def _body_dirs(m, raw, gR, canon):
    """Subject forward/right/up unit vectors in world at each frame + quasi-static mask."""
    up, fwd, right = canon
    fwd_w = np.einsum("nij,j->ni", gR[:, 0], fwd)
    right_w = np.einsum("nij,j->ni", gR[:, 0], right)
    up_w = np.einsum("nij,j->ni", gR[:, 0], up)
    pel = np.asarray(raw["imu_gt"]["Pelvis"]["pos_world"], np.float64)
    n = pel.shape[0]
    vel = np.zeros(n)
    vel[1:] = np.linalg.norm(np.diff(pel, axis=0), axis=1) * 100.0
    st = vel < np.percentile(vel, 20)
    return fwd_w, right_w, up_w, st


def test_helpers_exist():
    m = _mod()
    assert callable(m.build_measured) and callable(m.measured_small) and callable(m.fit_site_calibration)
    assert callable(m.compute_site_relabels) and callable(m.prepare_take_geometry)
    assert callable(m.canonical_axes) and callable(m.anatomical_frame)
    assert m.SITE_PHYS["occiput"] == "Head" and "back_T4" not in m.SITE_PHYS
    # every channel has a geometry entry (driving joint, kind, proximal/distal, side)
    assert set(m.SITE_GEOM) == set(m.SENSOR_CODES)


@NEEDS_DATA
def test_site_relabels_are_proper_rotations():
    m = _mod()
    raw = m.faithful.load_prism_safe(str(PKL))
    gR, j_rest, canon = m.prepare_take_geometry(raw, "M")
    rel = m.compute_site_relabels(raw, gR, j_rest)
    assert set(rel) == set(m.SENSOR_CODES)
    for code, C in rel.items():
        C = np.asarray(C, np.float64)
        assert C.shape == (3, 3), code
        assert np.allclose(C @ C.T, np.eye(3), atol=1e-6), code
        assert abs(np.linalg.det(C) - 1.0) < 1e-6, code


@NEEDS_DATA
def test_relabeled_S_frame_matches_spec_axes():
    """STRONG spec constraint (small/00_sensors): +Y=proximal, +Z=outward lateral, left limbs 180 about Y."""
    m = _mod()
    raw = m.faithful.load_prism_safe(str(PKL))
    gR, j_rest, canon = m.prepare_take_geometry(raw, "M")
    rel = m.compute_site_relabels(raw, gR, j_rest)
    fwd_w, right_w, up_w, st = _body_dirs(m, raw, gR, canon)

    # limbs: +Z outward (right limb -> subject right; left limb -> subject left), +Y proximal (up-ish)
    for code, out_sign in [("shank_r", +1), ("shank_l", -1), ("foot_r", +1), ("foot_l", -1),
                           ("wrist_r", +1), ("wrist_l", -1)]:
        key = m.SITE_PHYS[code]
        Rgt = np.asarray(raw["imu_gt"][key]["ori_world"], np.float64)
        S = np.einsum("nij,jk->nik", Rgt, rel[code])
        z_right = np.mean((S[:, :, 2] * right_w)[st].sum(-1))    # lateral . subject-right
        y_up = np.mean((S[:, :, 1] * up_w)[st].sum(-1))          # proximal . up
        assert np.sign(z_right) == out_sign and abs(z_right) > 0.4, (code, z_right)
        assert y_up > 0.4, (code, y_up)

    # left limb +X = posterior, right limb +X = anterior (the 180-about-Y consequence)
    for code, fwd_sign in [("shank_r", +1), ("shank_l", -1), ("foot_r", +1), ("foot_l", -1)]:
        key = m.SITE_PHYS[code]
        Rgt = np.asarray(raw["imu_gt"][key]["ori_world"], np.float64)
        S = np.einsum("nij,jk->nik", Rgt, rel[code])
        x_fwd = np.mean((S[:, :, 0] * fwd_w)[st].sum(-1))
        assert np.sign(x_fwd) == fwd_sign and abs(x_fwd) > 0.4, (code, x_fwd)

    # midline occiput + back_T4: +Z = subject right, +X = forward, +Y = up
    Rocc = np.asarray(raw["imu_gt"]["Head"]["ori_world"], np.float64)
    Socc = np.einsum("nij,jk->nik", Rocc, rel["occiput"])
    assert np.mean((Socc[:, :, 2] * right_w)[st].sum(-1)) > 0.6
    assert np.mean((Socc[:, :, 0] * fwd_w)[st].sum(-1)) > 0.4
    Sback = np.einsum("nij,jk->nik", gR[:, m.BACK_JOINT], rel["back_T4"])   # small_ideal back = gR[:,spine3]
    assert np.mean((Sback[:, :, 2] * right_w)[st].sum(-1)) > 0.8
    assert np.mean((Sback[:, :, 0] * fwd_w)[st].sum(-1)) > 0.8
    assert np.mean((Sback[:, :, 1] * up_w)[st].sum(-1)) > 0.8


@NEEDS_DATA
def test_acceleration_is_gravity_included_specific_force():
    m = _mod()
    small, large, dev, anthro, manifest = m.build_measured(_spec(m, (1000, 2000)))
    acc = small["imu_acceleration"]
    for code in ("occiput", "wrist_l", "shank_r"):
        j = m.SENSOR_CODES.index(code)
        mag = np.linalg.norm(acc[:, j, :], axis=1).mean()
        assert 7.0 < mag < 13.0, (code, mag)
    assert str(small["small_mode"]) == "measured_physical"


@NEEDS_DATA
def test_orientation_is_calibrated_physical_and_unit_quat():
    """small's own orientation must reconstruct the physical world accel: a = R_S2G @ acc + g."""
    m = _mod()
    small, *_ = m.build_measured(_spec(m, (1000, 2000)))
    ori = small["imu_orientation"]
    assert float(np.max(np.abs(np.linalg.norm(ori, axis=2) - 1.0))) < 1e-3
    raw = m.faithful.load_prism_safe(str(PKL))
    from scipy.spatial.transform import Rotation
    for code in ("occiput", "shank_r", "foot_l"):
        site = m.SITE_PHYS[code]
        j = m.SENSOR_CODES.index(code)
        q = ori[:, j, :].astype(np.float64)                      # (w,x,y,z)
        R = Rotation.from_quat(np.concatenate([q[:, 1:], q[:, :1]], axis=1)).as_matrix()
        f = small["imu_acceleration"][:, j, :].astype(np.float64)
        a_recon = np.einsum("nij,nj->ni", R, f) + m.GRAVITY
        a_true = np.asarray(raw["imu"][site]["acc_world_filt"], np.float64)[1000:2000]
        assert float(np.max(np.abs(a_recon - a_true))) < 1e-2, code


@NEEDS_DATA
def test_foot_gyro_is_measured_body_gyro_is_derived():
    m = _mod()
    small, *_ = m.build_measured(_spec(m, (1000, 2000)))
    prov = list(small["gyro_provenance"])
    assert prov[m.SENSOR_CODES.index("foot_l")].startswith("measured:gyr_local_raw")
    assert prov[m.SENSOR_CODES.index("occiput")].startswith("derived_from_orientation")
    # foot gyro equals R_cal^T @ gyr_local_raw, with R_cal built from the correct v2 relabel
    raw = m.faithful.load_prism_safe(str(PKL))
    gR, j_rest, _ = m.prepare_take_geometry(raw, "M")
    rel = m.compute_site_relabels(raw, gR, j_rest)
    r_cal = m.fit_site_calibration(np.asarray(raw["imu"]["L_Foot"]["ori_world"], np.float64),
                                   np.asarray(raw["imu_gt"]["L_Foot"]["ori_world"], np.float64),
                                   rel["foot_l"])
    gl = np.asarray(raw["imu"]["L_Foot"]["gyr_local_raw"], np.float64)[1000:2000]
    # r_cal^T @ M @ gl. The relabel M is not optional: gyr_local_raw does not share the body
    # frame imu.ori maps out of, and without it the shipped foot gyro contradicts the shipped
    # foot orientation in every take.
    expect = np.einsum("ji,jk,nk->ni", r_cal, m.GYR_LOCAL_AXIS_RELABEL, gl)
    got = small["imu_angular_velocity"][:, m.SENSOR_CODES.index("foot_l"), :].astype(np.float64)
    assert float(np.max(np.abs(got - expect))) < 1e-2


def test_the_gyro_axis_relabel_is_a_proper_rotation_and_the_one_that_was_solved_for():
    """Runs without the raw take: the matrix is a fact about the source, pinned here so a future
    edit cannot quietly change what the foot gyro means.

    Solved by Kabsch over all 150 takes x 2 feet, which converge on this one matrix while the six
    body channels return the identity.
    """
    m = _mod()
    M = m.GYR_LOCAL_AXIS_RELABEL
    assert M.shape == (3, 3)
    assert np.isclose(np.linalg.det(M), 1.0), "a relabel that flips handedness is not a rotation"
    assert np.allclose(M @ M.T, np.eye(3))
    # (a, b, c) -> (-a, c, b)
    assert np.allclose(M @ np.array([1.0, 2.0, 3.0]), [-1.0, 3.0, 2.0])
    assert not np.allclose(M, np.eye(3)), "the identity is what the defect assumed"


@NEEDS_DATA
def test_back_T4_is_flagged_non_physical_and_axis_relabeled():
    m = _mod()
    small, _, _, _, manifest = m.build_measured(_spec(m, (1000, 2000)))
    j = m.SENSOR_CODES.index("back_T4")
    assert str(small["gyro_provenance"][j]).startswith("proxy_non_physical")
    assert np.allclose(small["q_anatomical_from_sensor"][j], [1, 0, 0, 0])
    assert manifest["small_sites_provenance"]["back_T4"].startswith("proxy_non_physical")
    assert manifest["small_sites_provenance"]["occiput"].startswith("measured_physical")
    # back accel is gravity-included (proxy specific force), still ~9.8
    mag = np.linalg.norm(small["imu_acceleration"][:, j, :], axis=1).mean()
    assert 7.0 < mag < 13.0, mag


@NEEDS_DATA
def test_synth_mask_marks_r_wrist_gapfilled_frames():
    m = _mod()
    small, *_ = m.build_measured(_spec(m, (5700, 5900)))
    jw = m.SENSOR_CODES.index("wrist_r")
    jl = m.SENSOR_CODES.index("wrist_l")
    assert small["imu_synth_mask"][:, jw].any()
    assert not small["imu_synth_mask"][:, jl].any()
    small2, *_ = m.build_measured(_spec(m, (1000, 2000)))
    assert not small2["imu_synth_mask"].any()


@NEEDS_DATA
def test_large_anthro_reconciled_to_faithful_v2():
    # faithful-v2 reconcile lives in build_measured (the deployed lineage); the ideal faithful build is
    # untouched, so its golden stays valid while the deployed dataset gets the canonical schema.
    m = _mod()
    sp = _spec(m, (1000, 2000))
    _, large_m, dev_m, anthro_m, manifest_m = m.build_measured(sp)
    _small_f, _large_f, dev_f, anthro_f, _ = m.faithful.build(sp)
    assert _npz_bytes(dev_m) == _npz_bytes(dev_f)                       # dev unchanged
    assert "smpl_global_orientation_world" in large_m                  # canonical key
    assert "smpl_global_orientation_prism_world" not in large_m        # drift retired
    assert large_m["joint_names"].dtype.kind == "U"                    # U-dtype strings
    assert "subject_scope_note" in anthro_m and "subject_scope_note" not in anthro_f
    assert manifest_m["spec_id"] == "qmd_unified8_smpl18"
    assert manifest_m["spec_version"] == "faithful-v2"
    assert "README.md" not in manifest_m["deliverables"]
    assert "smpl_root_translation.npz" in manifest_m["deliverables"]
    assert _npz_bytes(_small_f) != _npz_bytes(m.build_measured(sp)[0])  # measured small != ideal small


@NEEDS_DATA
def test_build_measured_is_deterministic():
    m = _mod()
    sp = _spec(m, (1000, 2000))
    s1 = m.build_measured(sp)[0]
    s2 = m.build_measured(sp)[0]
    assert _npz_bytes(s1) == _npz_bytes(s2)
