"""Tests for the Anthro (anthropometry) deliverable computation.

Anthro is the third dataset (per ``anthro/00_anthropometry.qmd``): subject-level,
trial/frame-invariant skeleton constants that, combined with Large's motion joints,
reconstruct the full SMPL 22-chain / rigid skeleton.

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

    assert out["joint_position"].shape == (2, 22, 3)
    assert out["joint_position"].dtype == np.float32
    assert out["segment_length"].shape == (len(anthro.SEGMENTS),)
    assert out["fixed_joint_rotation"].shape == (4, 4)
    assert out["betas"].shape == betas.shape
    # sex / measured anthropometry
    assert str(out["sex"]) == "M"
    assert float(out["height"]) == pytest.approx(1.63, abs=1e-6)   # cm -> m
    assert float(out["body_mass"]) == pytest.approx(51.0, abs=1e-6)
    assert str(out["pair_id"]) == "deadbeef"
    # joint_position is root-relative (pelvis at origin, both poses)
    np.testing.assert_allclose(out["joint_position"][:, 0, :], 0.0, atol=1e-6)
    # names line up
    assert len(list(out["joint_names"])) == 22
    assert list(out["fixed_joint_names"]) == list(anthro.FIXED_JOINT_NAMES)


def test_fixed_joint_rotation_is_unit_identity_quat():
    anthro = _load_anthro_module()
    model = _tiny_model(anthro)
    out = anthro.compute_anthro(model, np.zeros(2), {"gender": "M", "height": 170.0, "weight": 60.0}, "x")
    fjr = out["fixed_joint_rotation"]
    assert fjr.shape == (4, 4)
    # identity quaternion (w,x,y,z) = (1,0,0,0), unit norm
    np.testing.assert_allclose(fjr[:, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(fjr[:, 1:], 0.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(fjr, axis=1), 1.0, atol=1e-6)


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


def test_reconstruction_frozen_equals_full_when_fixed_joints_are_identity():
    """Large(18)+Anthro(4 frozen) reconstruction == full-24 FK when the 4 fixed joints
    are identity in the source pose (the exactness boundary of the SOMA freeze model)."""
    anthro = _load_anthro_module()
    model = _tiny_model(anthro)
    J = anthro.rest_joints(model, np.zeros(2))
    rng = np.random.default_rng(0)
    local_full = np.stack([Rotation.from_rotvec(rng.normal(0, 0.3, 3)).as_matrix() for _ in range(24)])
    # zero-out the 4 fixed joints in the source (identity) -> freeze is lossless here
    for j in anthro.FIXED_JOINTS:
        local_full[j] = np.eye(3)
    full_pos, _ = anthro.smpl_fk_positions(J, local_full)
    frozen_pos = anthro.reconstruct_positions_frozen_fixed(J, local_full)
    np.testing.assert_allclose(frozen_pos, full_pos, atol=1e-6)


def test_reconstruction_frozen_deviates_when_fixed_joints_move():
    anthro = _load_anthro_module()
    model = _tiny_model(anthro)
    J = anthro.rest_joints(model, np.zeros(2))
    local_full = np.repeat(np.eye(3)[None], 24, axis=0)
    # give a collar a real rotation -> frozen reconstruction must differ downstream (wrist)
    local_full[13] = Rotation.from_rotvec([0, 0, 0.5]).as_matrix()
    full_pos, _ = anthro.smpl_fk_positions(J, local_full)
    frozen_pos = anthro.reconstruct_positions_frozen_fixed(J, local_full)
    assert np.linalg.norm(full_pos[20] - frozen_pos[20]) > 1e-3


def test_mean_rotation_of_constant_returns_that_rotation():
    anthro = _load_anthro_module()
    R = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix()
    stack = np.repeat(R[None], 7, axis=0)
    np.testing.assert_allclose(anthro.mean_rotation(stack), R, atol=1e-9)


def test_fixed_local_R_matching_source_makes_reconstruction_exact():
    """When the frozen constant equals the (constant) source fixed-joint rotation, the
    reconstruction is exact even though the collar is far from identity."""
    anthro = _load_anthro_module()
    model = _tiny_model(anthro)
    J = anthro.rest_joints(model, np.zeros(2))
    collar = Rotation.from_rotvec([0, 0, 0.42]).as_matrix()      # ~24 deg, like the real take
    local_full = np.repeat(np.eye(3)[None], 24, axis=0)
    local_full[13] = collar
    fixed = np.stack([np.eye(3), np.eye(3), collar, np.eye(3)])  # FIXED_JOINTS order (3,6,13,14)
    full_pos, _ = anthro.smpl_fk_positions(J, local_full)
    frozen = anthro.reconstruct_positions_frozen_fixed(J, local_full, fixed_local_R=fixed)
    np.testing.assert_allclose(frozen, full_pos, atol=1e-9)
    # and identity freeze would NOT be exact here
    id_frozen = anthro.reconstruct_positions_frozen_fixed(J, local_full)
    assert np.linalg.norm(id_frozen[20] - full_pos[20]) > 1e-2


def test_compute_anthro_stores_provided_fixed_rotations_as_unit_quats():
    anthro = _load_anthro_module()
    model = _tiny_model(anthro)
    fixed = np.stack([
        Rotation.from_rotvec([0, 0, 0.04]).as_matrix(),
        Rotation.from_rotvec([0, 0, 0.16]).as_matrix(),
        Rotation.from_rotvec([0, 0, 0.42]).as_matrix(),
        Rotation.from_rotvec([0, 0, 0.43]).as_matrix(),
    ])
    out = anthro.compute_anthro(model, np.zeros(2), {"gender": "M", "height": 170.0, "weight": 60.0},
                                "pid", fixed_joint_local_R=fixed)
    fjr = out["fixed_joint_rotation"]
    assert fjr.shape == (4, 4)
    np.testing.assert_allclose(np.linalg.norm(fjr, axis=1), 1.0, atol=1e-6)
    # not identity (collars ~24 deg)
    assert abs(float(fjr[2, 0])) < 0.9999
    assert "reduced_model_fit" in str(out["fixed_joint_rotation_provenance"])


def _random_local(rng, T):
    return np.stack([
        np.stack([Rotation.from_rotvec(rng.normal(0, 0.4, 3)).as_matrix() for _ in range(24)])
        for _ in range(T)
    ])


def _global(anthro, local):
    P = anthro.SMPL24_PARENTS
    G = np.zeros_like(local)
    G[:, 0] = local[:, 0]
    for j in range(1, 24):
        G[:, j] = G[:, int(P[j])] @ local[:, j]
    return G


def _reduced_model():
    _load_anthro_module()          # puts packages/smpl18/src on the path
    from soma_synth.pipeline import reduced_model
    return reduced_model


def test_refit_preserves_distal_global_orientation_for_any_constant():
    anthro = _load_anthro_module()
    reduced_model = _reduced_model()
    rng = np.random.default_rng(1)
    local = _random_local(rng, 5)
    fixed = np.stack([Rotation.from_rotvec(rng.normal(0, 0.5, 3)).as_matrix() for _ in range(4)])
    refit = reduced_model.reduce_local(local, fixed)
    Graw, Gref = _global(anthro, local), _global(anthro, refit)
    # distal joints below the fixed welds: spine3, head, shoulders, wrists — orientation exact
    for j in (9, 12, 15, 16, 17, 20, 21):
        d = (np.swapaxes(Graw[:, j], -1, -2) @ Gref[:, j]).reshape(-1, 3, 3)
        ang = np.degrees(np.linalg.norm(Rotation.from_matrix(d).as_rotvec(), axis=1))
        assert ang.max() < 1e-6, (j, ang.max())
    # the 4 fixed joints now hold the constants (np.allclose broadcasts (3,3) over frames)
    for i, j in enumerate(anthro.FIXED_JOINTS):
        assert np.allclose(refit[:, j], fixed[i], atol=1e-12)


def test_fit_reduces_position_residual_vs_identity():
    anthro = _load_anthro_module()
    reduced_model = _reduced_model()
    model = _tiny_model(anthro)
    J = anthro.rest_joints(model, np.zeros(2))
    rng = np.random.default_rng(2)
    local = _random_local(rng, 40)
    pose = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(40, 24, 3)
    settings = reduced_model.settings_for_source("amass")
    fit = reduced_model.fit_subject([pose], J, settings, group="tiny", basis="test", scope="take")
    assert fit.constants.shape == (4, 3, 3)
    affected = list(reduced_model.AFFECTED_BY_FREEZE)

    def mean_res(fixed):
        r = np.linalg.norm(
            anthro.fk_positions_batch(J, reduced_model.reduce_local(local, fixed))
            - anthro.fk_positions_batch(J, local), axis=2)
        return r[:, affected].mean()

    ident = np.repeat(np.eye(3)[None], 4, axis=0)
    assert mean_res(fit.constants) <= mean_res(ident) + 1e-9
