from __future__ import annotations

import glob
import importlib.util
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from data_root_skips import DATA_ROOT, needs_data

SCRIPT = Path(__file__).parents[2] / "scripts" / "poc" / "generate_amass_faithful.py"
AMASS = DATA_ROOT / "extracted" / "amass"
NEEDS_DATA = needs_data((AMASS / "KIT").exists(), "the AMASS source extracted/amass/KIT")


def _mod():
    spec = importlib.util.spec_from_file_location("amass_gen", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _spec(m, dataset, path):
    sub = os.path.basename(os.path.dirname(path))
    seq = os.path.splitext(os.path.basename(path))[0]
    return m.AmassSeqSpec(dataset=dataset, subject_id=f"amass_{dataset}_{sub}", sequence_id=seq,
                          source_npz=path, relative_path=os.path.relpath(path, DATA_ROOT).replace("\\", "/"))


def _first(dataset):
    fs = sorted(glob.glob(str(AMASS / dataset / "**" / "*_poses.npz"), recursive=True))
    return fs[0] if fs else None


def _npz_bytes(d):
    b = io.BytesIO()
    np.savez(b, **d)
    return b.getvalue()


@NEEDS_DATA
def test_build_returns_four_no_development_reference():
    m = _mod()
    out = m.build(_spec(m, "KIT", _first("KIT")))
    assert len(out) == 4  # small, large, anthro, manifest — no development_reference
    small, large, anthro, manifest = out
    # The root-translation sidecar is written from large's pelvis aux by the corpus writer, so it is
    # a deliverable of the take without being a fifth return value of build().
    assert manifest["deliverables"] == [
        "small_reference.npz", "large_reference.npz", "anthro_reference.npz",
        "smpl_root_translation.npz", "manifest.json"]
    assert manifest["faithful_content"]["development_reference"].startswith("NONE")


@NEEDS_DATA
def test_small_is_synthetic_8ch_imu():
    m = _mod()
    small, large, anthro, manifest = m.build(_spec(m, "KIT", _first("KIT")))
    mm = int(small["frame_count"])
    assert list(small["sensor_codes"]) == m.SENSOR_CODES
    assert small["imu_orientation"].shape == (mm, 8, 4)
    assert small["imu_acceleration"].shape == (mm, 8, 3)
    assert small["imu_angular_velocity"].shape == (mm, 8, 3)
    assert str(small["small_mode"]) == "synthetic_from_smpl"
    heading = np.asarray(small["imu_orientation_absolute_heading"])
    assert heading.tolist() == [True] * 6 + [False] * 2
    qn = np.linalg.norm(small["imu_orientation"], axis=2)
    assert float(np.max(np.abs(qn - 1))) < 1e-3
    # specific-force self-consistency round-trip (documented validation)
    assert manifest["validation"]["specific_force_inverse_max_abs_err_m_s2"] < 1e-3


@NEEDS_DATA
def test_small_orientation_is_spec_sensor_frame_s_v2():
    """STRONG spec axis constraint (small/00_sensors): +Y=proximal, +Z=outward lateral; left limbs 180."""
    from scipy.spatial.transform import Rotation
    m = _mod()
    path = _first("KIT")
    small, *_ = m.build(_spec(m, "KIT", path))
    assert str(small["axis_convention"]) == "spec_S_v2"
    assert set(m.SITE_GEOM) == set(m.SENSOR_CODES)

    d = m.load_amass_safe(path, "KIT")
    local = m.resample_rotations(
        Rotation.from_rotvec(m.build_pose24(d["poses"]).reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3),
        d["fps"])
    gR = m.fk_global_rotation(local)
    model = m.anthro_smpl.load_smpl_model_for_gender("F" if d["gender"].startswith("f") else "M")
    canon = m.canonical_axes(m.anthro_smpl.rest_joints(model, d["betas"]))
    right_w = np.einsum("nij,j->ni", gR[:, 0], canon[2])
    up_w = np.einsum("nij,j->ni", gR[:, 0], canon[0])
    tr = m.resample_trans(d["trans"], d["fps"])
    vel = np.zeros(tr.shape[0])
    vel[1:] = np.linalg.norm(np.diff(tr, axis=0), axis=1) * 100.0
    st = vel < np.percentile(vel, 20)

    for code, out_sign in [("shank_r", +1), ("shank_l", -1), ("foot_r", +1), ("foot_l", -1),
                           ("wrist_r", +1), ("wrist_l", -1), ("occiput", +1), ("back_T4", +1)]:
        j = m.SENSOR_CODES.index(code)
        q = small["imu_orientation"][:, j, :].astype(np.float64)
        R = Rotation.from_quat(np.concatenate([q[:, 1:], q[:, :1]], axis=1)).as_matrix()
        z_right = np.mean((R[:, :, 2] * right_w)[st].sum(-1))    # lateral . subject-right
        y_up = np.mean((R[:, :, 1] * up_w)[st].sum(-1))          # proximal . up
        assert np.sign(z_right) == out_sign and abs(z_right) > 0.4, (code, z_right)
        assert y_up > 0.4, (code, y_up)


@NEEDS_DATA
def test_large_reduced_model_and_trajectory_reconstruction():
    m = _mod()
    small, large, anthro, manifest = m.build(_spec(m, "KIT", _first("KIT")))
    mm = int(large["frame_count"])
    assert list(large["joint_names"]) == m.JOINT18_NAMES
    assert large["joint_rotation"].shape == (mm, 18, 4)
    assert large["joint_velocity"].shape == (mm, 18, 3)
    assert large["root_velocity"].shape == (mm, 3)
    assert large["smpl_global_orientation_world"].shape == (mm, 24, 4)
    assert "smpl_global_orientation_prism_world" not in large   # PRISM-specific name dropped for AMASS
    assert manifest["validation"]["root_velocity_cumsum_recon_max_abs_err_m"] < 1e-3


@NEEDS_DATA
def test_anthro_amass_overrides():
    m = _mod()
    small, large, anthro, manifest = m.build(_spec(m, "KIT", _first("KIT")))
    assert anthro["joint_position"].shape == (2, 22, 3)
    assert anthro["betas"].shape == (16,)
    # AMASS constants are sequence-level (per-sequence betas), not subject-shared
    assert str(anthro["subject_scope"]) == "sequence_constant_frame_invariant"
    # height is an SMPL stature estimate (finite, human-plausible); body_mass unavailable (NaN)
    assert 1.0 < float(anthro["height"]) < 2.5
    assert np.isnan(float(anthro["body_mass"]))
    assert "smpl10_beta_truncation" in str(anthro["betas_provenance"])
    assert "estimated:smpl_rest_stature" in str(anthro["height_provenance"])
    assert "unavailable" in str(anthro["body_mass_provenance"]).lower()
    # no measured GRF/insole anywhere in the manifest's unavailable list
    ua = json.dumps(manifest["unavailable_or_not_applied"]).lower()
    assert "grf" in ua and "body_mass" in ua


@NEEDS_DATA
def test_resample_120_to_100_recorded():
    m = _mod()
    cmu = _first("CMU")
    if cmu is None:
        pytest.skip("no CMU")
    small, large, anthro, manifest = m.build(_spec(m, "CMU", cmu))
    r = manifest["resample"]
    assert r["source_fps"] == 120.0 and r["target_fps"] == 100
    assert r["noop"] is False
    assert r["frames_out"] == m.resampled_length(r["frames_in"], 120.0, 100)


@NEEDS_DATA
def test_build_is_deterministic():
    m = _mod()
    sp = _spec(m, "KIT", _first("KIT"))
    s1, l1, a1, _ = m.build(sp)
    s2, l2, a2, _ = m.build(sp)
    assert _npz_bytes(s1) == _npz_bytes(s2)
    assert _npz_bytes(l1) == _npz_bytes(l2)
    assert _npz_bytes(a1) == _npz_bytes(a2)


@NEEDS_DATA
def test_manifest_governance_labels():
    m = _mod()
    small, large, anthro, manifest = m.build(_spec(m, "KIT", _first("KIT")))
    assert manifest["distribution_scope"] == "internal_only"
    assert manifest["artifact_class"] == "experimental_non_candidate"
    assert manifest["source"]["source_name"] == "amass"
    assert "non_authorizing" in manifest["source"]["field_registry"]
    assert manifest["quality_gate"] == "NOT_EVALUATED"
    assert manifest["up_axis"] == "z"
    assert manifest["excluded_modalities"] == ["emg"]
    assert manifest["governance_disposition"]["not_the_governed_pipeline"] is True
