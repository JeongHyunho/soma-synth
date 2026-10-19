"""The validator's large_kinetics invariants (validation.checks.check_take_l2_kinetics)."""

import numpy as np
import pytest

from soma_synth.contracts import qmd_unified8_smpl18_spec as spec
from soma_synth.validation import checks

NAMES18 = ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
           "spine3", "left_foot", "right_foot", "neck", "head", "left_shoulder", "right_shoulder",
           "left_elbow", "right_elbow", "left_wrist", "right_wrist"]


def _large(T=20):
    contact = np.zeros((T, 2), bool)
    contact[:12, 0] = True
    contact[8:, 1] = True
    grf = np.where(contact[..., None], np.array([1.0, 2.0, 600.0], np.float32), 0.0).astype(np.float32)
    cop = np.where(contact[..., None], np.full((T, 2, 2), 0.1), np.nan).astype(np.float32)
    free = np.where(contact, np.float32(1.5), np.nan).astype(np.float32)
    return {
        "joint_names": np.array(NAMES18),
        "kinetics_joint_names": np.array(NAMES18[1:]),
        "com_acceleration": np.zeros((T, 3), np.float32),
        "angular_momentum": np.zeros((T, 3), np.float32),
        "grf": grf, "cop": cop, "free_moment": free, "foot_contact": contact,
        "joint_torque": np.zeros((T, 17), np.float32),
        "joint_power": np.zeros((T, 17), np.float32),
        "joint_moments_jcs": np.zeros((T, 17, 3), np.float32),
        "joint_power_total": np.zeros((T, 17), np.float32),
        "grf_valid_mask": np.ones(T, bool),
        "kinetics_valid_mask": np.ones(T, bool),
        "inverse_dynamics_residual": np.zeros((T, 6), np.float32),
    }


def _check(tmp_path, large):
    np.savez(tmp_path / "large_reference.npz", **large)
    return [f.message for f in checks.check_take_l2_kinetics(tmp_path, "t")]


def test_consistent_kinetics_pass(tmp_path):
    assert _check(tmp_path, _large()) == []


def test_a_bundle_without_kinetics_is_not_judged(tmp_path):
    large = {k: v for k, v in _large().items() if k == "joint_names"}
    assert _check(tmp_path, large) == []


@pytest.mark.parametrize("break_it, expected", [
    (lambda a: a["grf"].__setitem__((15, 0, 2), 3.0), "grf is not zero on frames without foot_contact"),
    (lambda a: a["cop"].__setitem__((15, 0, 0), 0.2), "cop is finite on frames without foot_contact"),
    (lambda a: a["cop"].__setitem__((2, 0, 0), np.nan), "cop is NaN on frames with foot_contact"),
    (lambda a: a["free_moment"].__setitem__((2, 0), np.nan), "free_moment is NaN on frames with foot_contact"),
    (lambda a: a["joint_torque"].__setitem__((3, 4), np.nan), "non-finite values in joint_torque"),
    (lambda a: a["grf_valid_mask"].__setitem__(5, False), "kinetics_valid_mask is True on frames grf_valid_mask"),
    (lambda a: a.__setitem__("kinetics_joint_names", np.array(NAMES18[:17])), "kinetics_joint_names is not"),
])
def test_each_broken_invariant_fails(tmp_path, break_it, expected):
    large = _large()
    break_it(large)
    messages = _check(tmp_path, large)
    assert any(expected in m for m in messages), messages


def test_the_spec_gates_the_kinetics_keys_by_capability():
    with_caps = spec.expected_deliverables(frozenset({"large_kinetics"}))
    assert "large_reference.npz" in with_caps
    arrays = {k: (v.shape, spec.dtype_class(v.dtype)) for k, v in _large().items()}
    arrays.update({"joint_rotation": ((20, 18, 4), "f4"), "joint_velocity": ((20, 18, 3), "f4"),
                   "root_velocity": ((20, 3), "f4"), "pelvis_position_world_aux": ((20, 3), "f4"),
                   "smpl_global_orientation_world": ((20, 24, 4), "f4"), "pair_id": ((), "U"),
                   "timestamps_s": ((20,), "f8"), "frame_count": ((), "i8")})
    assert spec.check_arrays("large", frozenset({"large_kinetics"}), arrays) == ()
    without = spec.check_arrays("large", frozenset(), arrays)
    assert any("forbidden-key:grf" in message for _, message in without)
