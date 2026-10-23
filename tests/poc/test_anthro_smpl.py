"""Tests for the Anthro (anthropometry) deliverable computation.

Anthro is the third dataset (per ``anthro/00_anthropometry.qmd``): subject-level,
trial/frame-invariant skeleton constants that, combined with Large's joint rotations,
reconstruct the SMPL 24-joint body / rigid skeleton.

These tests exercise the pure math on a *tiny synthetic* SMPL-like model so they run
without the real 6890-vertex body-model asset.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


MODULE_PATH = Path(__file__).parents[2] / "scripts" / "poc" / "anthro_smpl.py"


def _load_anthro_module():
    spec = importlib.util.spec_from_file_location("anthro_smpl", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_model(anthro, n_betas: int = 2):
    """A 24-vertex fake SMPL model: J_regressor = I(24) so rest_joints(betas=0)==v_template.

    v_template is a crude but valid skeleton laid out along +Y (up), arms along +X,
    mirroring the real SMPL canonical (Y-up) frame so upright-neutral logic is exercised.
    """
    parents = anthro.SMPL24_PARENTS
    # Build a deterministic skeleton: each joint offset a fixed step from its parent.
    v = np.zeros((24, 3), dtype=np.float32)
    # spine chain up +Y; legs down -Y; arms out +/-X
    v[0] = (0.0, 0.0, 0.0)
    step = 0.12
    for j in range(1, 24):
        p = int(parents[j])
        off = np.array([0.0, step, 0.0], dtype=np.float32)
        # legs go down (-Y), arms go out (+/-X)
        if j in (1, 4, 7, 10):      # left leg chain
            off = np.array([0.08, -step, 0.0], dtype=np.float32)
        elif j in (2, 5, 8, 11):    # right leg chain
            off = np.array([-0.08, -step, 0.0], dtype=np.float32)
        elif j in (16, 18, 20, 22):  # left arm chain -> +X
            off = np.array([step, 0.0, 0.0], dtype=np.float32)
        elif j in (17, 19, 21, 23):  # right arm chain -> -X
            off = np.array([-step, 0.0, 0.0], dtype=np.float32)
        v[j] = v[p] + off
    shapedirs = np.zeros((24, 3, n_betas), dtype=np.float32)
    shapedirs[:, 1, 0] = 0.01 * np.arange(24)  # beta0 stretches +Y
    model = {
        "v_template": v,
        "shapedirs": shapedirs,
        "J_regressor": np.eye(24, dtype=np.float32),
        "parents": parents.copy(),
    }
    return model


def test_rest_joints_betas_zero_equals_regressed_template():
    anthro = _load_anthro_module()
    model = _tiny_model(anthro)
    J = anthro.rest_joints(model, np.zeros(2, np.float64))
    assert J.shape == (24, 3)
    np.testing.assert_allclose(J, model["v_template"], atol=1e-6)


def test_rest_joints_betas_apply_shapedirs():
    anthro = _load_anthro_module()
    model = _tiny_model(anthro)
    betas = np.array([1.0, 0.0])
    J = anthro.rest_joints(model, betas)
    expected = model["v_template"] + model["shapedirs"][:, :, 0]
    np.testing.assert_allclose(J, expected, atol=1e-6)


def test_segment_lengths_match_bone_norms_and_are_positive():
    anthro = _load_anthro_module()
    model = _tiny_model(anthro)
    J = anthro.rest_joints(model, np.zeros(2))
    lengths = anthro.segment_lengths(J)
    assert lengths.shape == (len(anthro.SEGMENTS),)
    assert np.all(lengths > 0)
    for i, (_name, p, c) in enumerate(anthro.SEGMENTS):
        assert lengths[i] == pytest.approx(float(np.linalg.norm(J[c] - J[p])), abs=1e-6)


def test_segment_lengths_pose_invariant():
    anthro = _load_anthro_module()
    model = _tiny_model(anthro)
    J = anthro.rest_joints(model, np.zeros(2))
    local = np.repeat(np.eye(3)[None], 24, axis=0)
    local[16] = Rotation.from_rotvec([0, 0, -np.pi / 2]).as_matrix()
    local[17] = Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
    Jp, _ = anthro.smpl_fk_positions(J, local)
    # segment lengths from posed skeleton equal rest lengths (rigid bones)
    np.testing.assert_allclose(anthro.segment_lengths(J), anthro.segment_lengths(Jp), atol=1e-5)


def test_fk_identity_returns_rest():
    anthro = _load_anthro_module()
    model = _tiny_model(anthro)
    J = anthro.rest_joints(model, np.zeros(2))
    local = np.repeat(np.eye(3)[None], 24, axis=0)
    Jp, G = anthro.smpl_fk_positions(J, local)
    np.testing.assert_allclose(Jp, J, atol=1e-6)
    np.testing.assert_allclose(G, np.repeat(np.eye(3)[None], 24, axis=0), atol=1e-6)


def test_fk_known_rotation_moves_child():
    anthro = _load_anthro_module()
    model = _tiny_model(anthro)
    J = anthro.rest_joints(model, np.zeros(2))
    # rotate left_shoulder(16) by -90 about Z: its +X child chain should swing toward -Y
    local = np.repeat(np.eye(3)[None], 24, axis=0)
    local[16] = Rotation.from_rotvec([0, 0, -np.pi / 2]).as_matrix()
    Jp, _ = anthro.smpl_fk_positions(J, local)
    # left_wrist(20) should now be below (smaller Y) its rest Y
    assert Jp[20][1] < J[20][1] - 0.1


def test_upright_neutral_local_rotations_bring_arms_down():
    anthro = _load_anthro_module()
    model = _tiny_model(anthro)
    J = anthro.rest_joints(model, np.zeros(2))
    local = anthro.upright_neutral_local_R()
    assert local.shape == (24, 3, 3)
    Jp, _ = anthro.smpl_fk_positions(J, local)
    # both wrists end up lower than shoulders in the neutral pose
    assert Jp[20][1] < Jp[16][1]
    assert Jp[21][1] < Jp[17][1]
    # non-arm joints unchanged vs rest
    for j in (0, 1, 2, 7, 9, 12, 15):
        np.testing.assert_allclose(Jp[j], J[j], atol=1e-6)


def test_compute_anthro_shapes_and_values():
    anthro = _load_anthro_module()
    model = _tiny_model(anthro)
    betas = np.array([0.5, -0.2])
    subj_info = {"gender": "M", "height": 163.0, "weight": 51.0}
    out = anthro.compute_anthro(model, betas, subj_info, pair_id="deadbeef")

    assert out["joint_position"].shape == (2, 24, 3)
    assert out["joint_position"].dtype == np.float32
    assert out["segment_length"].shape == (15,)
    assert not any(key.startswith("fixed_joint") for key in out)
    assert out["betas"].shape == betas.shape
    # sex / measured anthropometry
    assert str(out["sex"]) == "M"
    assert float(out["height"]) == pytest.approx(1.63, abs=1e-6)   # cm -> m
    assert float(out["body_mass"]) == pytest.approx(51.0, abs=1e-6)
    assert str(out["pair_id"]) == "deadbeef"
    # joint_position is root-relative (pelvis at origin, both poses)
    np.testing.assert_allclose(out["joint_position"][:, 0, :], 0.0, atol=1e-6)
    # names line up: the 24 SMPL joints, and the segments end with the two hands
    assert list(out["joint_names"]) == list(anthro.JOINT24_NAMES)
    assert list(out["segment_names"])[-2:] == ["hand_l", "hand_r"]


def test_tpose_joint_position_is_the_root_relative_rest_skeleton():
    anthro = _load_anthro_module()
    model = _tiny_model(anthro)
    betas = np.array([0.3, 0.0])
    J = anthro.rest_joints(model, betas)
    out = anthro.compute_anthro(model, betas, {"gender": "F"}, "x")
    np.testing.assert_allclose(out["joint_position"][0], J - J[0], atol=1e-6)


def test_anthro_rest_and_local_rotations_reproduce_the_source_body():
    """FK of the published rest skeleton with the source's own local rotations is the source body."""
    anthro = _load_anthro_module()
    model = _tiny_model(anthro)
    J = anthro.rest_joints(model, np.zeros(2))
    out = anthro.compute_anthro(model, np.zeros(2), {"gender": "M"}, "x")
    rng = np.random.default_rng(0)
    local = np.stack([np.stack([Rotation.from_rotvec(rng.normal(0, 0.4, 3)).as_matrix()
                                for _ in range(24)]) for _ in range(5)])
    rest = out["joint_position"][0].astype(np.float64)
    np.testing.assert_allclose(anthro.fk_positions_batch(rest, local),
                               anthro.fk_positions_batch(J, local), atol=1e-6)


def test_load_smpl_model_for_gender_maps_paths(monkeypatch, tmp_path):
    anthro = _load_anthro_module()
    male = tmp_path / "SMPL_MALE_clean.npz"
    female = tmp_path / "SMPL_FEMALE_clean.npz"
    monkeypatch.setenv("SOMA_SMPL_MODEL_MALE", str(male))
    monkeypatch.setenv("SOMA_SMPL_MODEL_FEMALE", str(female))
    assert anthro.smpl_model_path_for_gender("M") == str(male)
    assert anthro.smpl_model_path_for_gender("male") == str(male)
    assert anthro.smpl_model_path_for_gender("F") == str(female)
    assert anthro.smpl_model_path_for_gender("female") == str(female)
    with pytest.raises(ValueError):
        anthro.smpl_model_path_for_gender("X")


def test_mean_rotation_of_constant_returns_that_rotation():
    anthro = _load_anthro_module()
    R = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix()
    stack = np.repeat(R[None], 7, axis=0)
    np.testing.assert_allclose(anthro.mean_rotation(stack), R, atol=1e-9)
