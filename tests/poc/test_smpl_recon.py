"""Regression tests for the Small/Large viewer reconstruction math, on synthetic data only (no
real data, no Blender, no pkl): the inverse-FK round-trip and the side-aware foot frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

POC = Path(__file__).parents[2] / "scripts" / "poc"
if str(POC) not in sys.path:
    sys.path.insert(0, str(POC))

import generate_prism_faithful as g   # noqa: E402  (numpy/scipy only; no bpy)
import smpl_recon as sr               # noqa: E402


def _random_poses(T=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, 24, 3)) * 0.6


def test_reconstruct_pose_is_exact_inverse_of_forward_fk():
    """reconstruct_pose_from_global must invert smpl24_global_rotation to SO(3) closure."""
    poses = _random_poses()
    G = g.smpl24_global_rotation(poses)                    # [T,24,3,3]
    quats = g.rotmat_to_quat_wxyz(G)                       # [T,24,4] (w,x,y,z)
    poses_rec = sr.reconstruct_pose_from_global(quats)     # [T,24,3]
    G_rec = g.smpl24_global_rotation(poses_rec)
    max_deg, _ = sr.geodesic_angle_deg(G, G_rec)
    assert max_deg < 1e-9, f"inverse-FK not lossless: {max_deg} deg"


def test_root_local_rotation_equals_root_global():
    """The SMPL root has no parent, so its reconstructed local rotation == its global orientation."""
    poses = _random_poses(seed=3)
    G = g.smpl24_global_rotation(poses)
    poses_rec = sr.reconstruct_pose_from_global(g.rotmat_to_quat_wxyz(G))
    R_root_rec = Rotation.from_rotvec(poses_rec[:, 0]).as_matrix()
    max_deg, _ = sr.geodesic_angle_deg(G[:, 0], R_root_rec)
    assert max_deg < 1e-9


def test_global_orientation_quaternions_are_unit_norm():
    quats = g.rotmat_to_quat_wxyz(g.smpl24_global_rotation(_random_poses(seed=7)))
    assert np.allclose(np.linalg.norm(quats, axis=-1), 1.0, atol=1e-6)


def test_foot_imu_frames_are_side_aware():
    """Guard the bilateral-axis fix: the reused render_prism_mesh_imu foot frames must be side-tagged
    so left/right insoles get outward-lateral +Z. Parsed from source (that module imports bpy)."""
    src = (POC / "render_prism_mesh_imu.py").read_text(encoding="utf-8")
    assert '("foot", "L")' in src and '("foot", "R")' in src, "foot IMU frames must be side-tagged"
    assert 'frame[0] == "foot"' in src, "sensor_frame_quat must handle the side-aware foot frame"


def test_shape_reference_prefers_anthro_over_development(tmp_path):
    """A dataset bundle may ship either name; the newer anthro_reference wins when both are present."""
    (tmp_path / "development_reference.npz").write_bytes(b"")
    assert sr.shape_reference_path(str(tmp_path)).endswith("development_reference.npz")
    (tmp_path / "anthro_reference.npz").write_bytes(b"")
    assert sr.shape_reference_path(str(tmp_path)).endswith("anthro_reference.npz")


def test_shape_reference_missing_returns_empty_and_load_raises(tmp_path):
    """No shape reference → "" (so discovery can skip the folder) and a named error when loaded."""
    assert sr.shape_reference_path(str(tmp_path)) == ""
    try:
        sr.load_shape_reference(str(tmp_path))
    except SystemExit as exc:
        assert "no shape reference npz" in str(exc)
    else:
        raise AssertionError("load_shape_reference must fail loudly when neither name exists")


def _npz(path, **arrays):
    np.savez(path, **arrays)
    return np.load(path, allow_pickle=True)


def test_root_translation_prefers_the_most_authoritative_source(tmp_path):
    """Published runs carry no translation, so several fallbacks exist — order must be deterministic."""
    T = 4
    shape_t = np.full((T, 3), 1.0, np.float32)
    large_t = np.full((T, 3), 2.0, np.float32)
    side_t = np.full((T, 3), 3.0, np.float32)
    smplx_t = np.full((T, 3), 4.0, np.float32)

    _npz(tmp_path / "large_reference.npz", smpl_trans=large_t)
    _npz(tmp_path / sr.ROOT_TRANSLATION_SIDECAR, smpl_trans=side_t)
    _npz(tmp_path / "take_smplx.npz", trans=smplx_t)
    shape = _npz(tmp_path / "anthro_reference.npz", betas=np.zeros(10), smpl_trans=shape_t)

    trans, src = sr.root_translation(str(tmp_path), shape, T)
    assert np.allclose(trans, 1.0) and src == "shape_reference.smpl_trans"

    bare = _npz(tmp_path / "bare.npz", betas=np.zeros(10))       # no smpl_trans on the shape ref
    trans, src = sr.root_translation(str(tmp_path), bare, T)
    assert np.allclose(trans, 2.0) and src == "large_reference.smpl_trans"

    (tmp_path / "large_reference.npz").unlink()
    trans, src = sr.root_translation(str(tmp_path), bare, T)
    assert np.allclose(trans, 3.0) and src.startswith(sr.ROOT_TRANSLATION_SIDECAR)

    (tmp_path / sr.ROOT_TRANSLATION_SIDECAR).unlink()
    trans, src = sr.root_translation(str(tmp_path), bare, T)
    assert np.allclose(trans, 4.0) and src.startswith("take_smplx.npz")


def test_root_translation_ignores_wrong_length_and_falls_back_in_place(tmp_path):
    """A translation of the wrong length is not this run's, so it is refused — and a run with no
    translation at all still opens, in place, under a label that says so."""
    shape = _npz(tmp_path / "anthro_reference.npz", betas=np.zeros(10))
    _npz(tmp_path / sr.ROOT_TRANSLATION_SIDECAR, smpl_trans=np.zeros((9, 3), np.float32))
    trans, src = sr.root_translation(str(tmp_path), shape, 4)
    assert trans.shape == (4, 3) and np.allclose(trans, 0.0)
    assert src == sr.IN_PLACE_TRANSLATION


def test_global_rotation_key_accepts_any_dataset_prefix(tmp_path):
    """The rotation array is the same quantity whatever the dataset prefixed it with."""
    T = 3
    q = np.zeros((T, 24, 4), np.float32)
    for name in ("smpl_global_orientation_prism_world", "smpl_global_orientation_world",
                 "smpl_global_orientation_someotherlab_world"):
        z = _npz(tmp_path / "large_reference.npz", **{name: q, "other": np.zeros(3)})
        assert sr.global_rotation_key(z) == name

    z = _npz(tmp_path / "large_reference.npz", joint_rotation=np.zeros((T, 18, 4), np.float32))
    assert sr.global_rotation_key(z) == ""            # 18 joints is the reduced set, not SMPL-24
    z = _npz(tmp_path / "large_reference.npz", smpl_global_orientation_bad=np.zeros((T, 18, 4)))
    assert sr.global_rotation_key(z) == ""            # right prefix, wrong shape -> not a match
