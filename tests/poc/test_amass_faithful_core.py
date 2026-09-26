from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from data_root_skips import DATA_ROOT, needs_data
from scipy.spatial.transform import Rotation

SCRIPT = Path(__file__).parents[2] / "scripts" / "poc" / "generate_amass_faithful.py"
AMASS = DATA_ROOT / "extracted" / "amass"
NEEDS_DATA = needs_data((AMASS / "KIT").exists(), "the AMASS source extracted/amass/KIT")


def _mod():
    spec = importlib.util.spec_from_file_location("amass_gen", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_build_pose24_maps_22_body_and_zeros_hands():
    m = _mod()
    poses = np.arange(2 * 156, dtype=np.float64).reshape(2, 156)
    p24 = m.build_pose24(poses)
    assert p24.shape == (2, 24, 3)
    assert np.array_equal(p24[:, :22], poses[:, :66].reshape(2, 22, 3))
    assert np.all(p24[:, 22:24] == 0.0)


def test_resampled_length_formula():
    m = _mod()
    # 121 frames @120Hz => duration 1.0 s => 100Hz => 101 frames
    assert m.resampled_length(121, 120.0, 100.0) == 101
    # 61 frames @60Hz => duration 1.0 s => 101 frames
    assert m.resampled_length(61, 60.0, 100.0) == 101
    # no-op
    assert m.resampled_length(500, 100.0, 100.0) == 500
    # float-underflow endpoints (138/120*100 == 114.9999999999) must NOT drop the final frame
    assert m.resampled_length(139, 120.0, 100.0) == 116
    assert m.resampled_length(70, 60.0, 100.0) == 116


def test_load_amass_missing_core_field_raises(tmp_path):
    m = _mod()
    p = tmp_path / "notrans_poses.npz"
    np.savez(p, poses=np.zeros((4, 156)), betas=np.zeros(16),
             gender=np.array("male"), mocap_framerate=np.array(120.0))
    with pytest.raises(ValueError):
        m.load_amass_safe(str(p))
    p2 = tmp_path / "nobetas_poses.npz"
    np.savez(p2, poses=np.zeros((4, 156)), trans=np.zeros((4, 3)),
             gender=np.array("male"), mocap_framerate=np.array(120.0))
    with pytest.raises(ValueError):
        m.load_amass_safe(str(p2))


def test_load_amass_nonfinite_raises(tmp_path):
    m = _mod()
    poses = np.zeros((4, 156))
    poses[0, 0] = np.nan
    p = tmp_path / "nan_poses.npz"
    np.savez(p, poses=poses, trans=np.zeros((4, 3)), betas=np.zeros(16),
             gender=np.array("male"), mocap_framerate=np.array(120.0))
    with pytest.raises(ValueError):
        m.load_amass_safe(str(p))


def test_resample_is_exact_noop_when_rates_equal():
    m = _mod()
    rng = np.random.default_rng(0)
    R = Rotation.from_rotvec(rng.standard_normal((10, 24, 3)).reshape(-1, 3)).as_matrix().reshape(10, 24, 3, 3)
    out = m.resample_rotations(R, 100.0, 100.0)
    assert np.array_equal(out, R)
    tr = rng.standard_normal((10, 3))
    assert np.array_equal(m.resample_trans(tr, 100.0, 100.0), tr)


def test_resample_rotations_endpoints_preserved():
    m = _mod()
    angs = np.linspace(0.0, np.pi / 2, 121)
    R = np.tile(np.eye(3), (121, 2, 1, 1)).astype(np.float64)
    R[:, 0] = Rotation.from_rotvec(np.c_[np.zeros(121), np.zeros(121), angs]).as_matrix()
    out = m.resample_rotations(R, 120.0, 100.0)
    assert out.shape == (101, 2, 3, 3)
    assert np.allclose(out[0], R[0], atol=1e-9)
    assert np.allclose(out[-1], R[-1], atol=1e-6)
    # monotone slerp: middle frame angle strictly between endpoints
    mid = Rotation.from_matrix(out[50, 0]).as_rotvec()[2]
    assert 0.0 < mid < np.pi / 2


def test_resample_trans_is_linear_interp():
    m = _mod()
    t_src = np.arange(121) / 120.0
    tr = np.c_[t_src, 2.0 * t_src, 3.0 * t_src]
    out = m.resample_trans(tr, 120.0, 100.0)
    assert out.shape == (101, 3)
    assert np.allclose(out[0], tr[0])
    t_dst = np.arange(101) / 100.0
    assert np.allclose(out[:, 0], t_dst, atol=1e-9)
    assert np.allclose(out[:, 2], 3.0 * t_dst, atol=1e-9)


def test_normalize_gender_str_bytes_and_short():
    m = _mod()
    assert m.normalize_gender(np.array("male")) == "male"
    assert m.normalize_gender(np.array(b"female")) == "female"
    assert m.normalize_gender("M") == "male"
    assert m.normalize_gender("F") == "female"


def test_load_amass_safe_synthetic(tmp_path):
    m = _mod()
    p = tmp_path / "seq_poses.npz"
    np.savez(p, poses=np.zeros((5, 156)), trans=np.ones((5, 3)), betas=np.zeros(16),
             gender=np.array("male"), mocap_framerate=np.array(120.0))
    d = m.load_amass_safe(str(p))
    assert d["poses"].shape == (5, 156)
    assert d["trans"].shape == (5, 3)
    assert d["fps"] == 120.0
    assert d["gender"] == "male"
    assert d["fps_source"] == "native"


def test_load_amass_missing_framerate_uses_dataset_fallback(tmp_path):
    m = _mod()
    p = tmp_path / "x_poses.npz"
    np.savez(p, poses=np.zeros((3, 156)), trans=np.zeros((3, 3)), betas=np.zeros(16),
             gender=np.array("male"))
    d = m.load_amass_safe(str(p), dataset="BMLmovi")
    assert d["fps"] == 120.0
    assert d["fps_source"] == "dataset_fallback"


def test_load_amass_no_poses_raises(tmp_path):
    m = _mod()
    p = tmp_path / "shape.npz"
    np.savez(p, betas=np.zeros(16), gender=np.array("male"))
    with pytest.raises((ValueError, KeyError)):
        m.load_amass_safe(str(p))


@NEEDS_DATA
def test_load_real_kit_sequence():
    m = _mod()
    import glob
    fs = sorted(glob.glob(str(AMASS / "KIT" / "**" / "*_poses.npz"), recursive=True))
    d = m.load_amass_safe(fs[0], dataset="KIT")
    assert d["poses"].shape[1] == 156
    assert d["gender"] in ("male", "female", "neutral")
    assert d["fps"] > 0
