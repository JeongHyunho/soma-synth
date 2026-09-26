"""`smpl18` computes what this repository's own copies of the same maths compute.

The SMPL-24 conversion rules moved out into the `smpl18` package (mounted at `packages/smpl18`),
while `scripts/poc/anthro_smpl.py`, `addbio_retarget/shape_fit.py`, `addbio_retarget/world_frame.py`
and `kinematics/rigid_body.py` still carry the versions the generated corpora were produced with.
Until those copies are retired, the two must agree: a bundle regenerated through the package has to
land on the same numbers as the bundle on disk.

This test lives here rather than in the package because it is about this repository's copies. It
compares them directly, on random inputs with a fixed seed.

Equality is exact where the arithmetic is the same sequence of operations. Where the package
batches what a local copy does frame by frame (`einsum` against a per-frame `@`), the
floating-point order differs and the check is `allclose` at 1e-12; each such case says so.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from smpl18.model.load import Model
from smpl18.skeleton import definition, frames, kinematics, rotations

from soma_synth.addbio_retarget import shape_fit, smpl_correspondence, world_frame
from soma_synth.kinematics import quaternion, rigid_body

REPO = Path(__file__).resolve().parents[2]
ANTHRO_PATH = REPO / "scripts" / "poc" / "anthro_smpl.py"


@pytest.fixture(scope="module")
def anthro():
    """`scripts/poc/anthro_smpl.py`, which is a script rather than an importable module."""
    spec = importlib.util.spec_from_file_location("_anthro_smpl_for_parity", ANTHRO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260915)


@pytest.fixture
def model_arrays(rng) -> dict:
    vertices = 60
    regressor = rng.random((24, vertices))
    regressor /= regressor.sum(axis=1, keepdims=True)
    return {
        "v_template": rng.normal(0, 0.3, (vertices, 3)),
        "shapedirs": rng.normal(0, 0.01, (vertices, 3, 10)),
        "J_regressor": regressor,
        "kintree_parents": np.array(definition.PARENTS, dtype=np.int64),
    }


def test_the_skeleton_definition_is_the_same_table(anthro):
    assert tuple(definition.PARENTS[1:]) == tuple(int(p) for p in anthro.SMPL24_PARENTS[1:])
    assert tuple(definition.PARENTS[1:]) == tuple(int(p) for p in smpl_correspondence.SMPL24_PARENTS[1:])
    assert tuple(definition.JOINT_NAMES) == tuple(anthro.JOINT24_NAMES)
    assert tuple(definition.JOINT_NAMES) == tuple(smpl_correspondence.SMPL24_NAMES)
    assert tuple(definition.SEGMENTS) == tuple(tuple(s) for s in anthro.SEGMENTS)


def test_rest_joints_match_both_local_copies(anthro, model_arrays, rng):
    betas = rng.normal(0, 1, 10)
    new = kinematics.rest_joints(Model(**model_arrays), betas)
    np.testing.assert_array_equal(new, anthro.rest_joints(model_arrays, betas))
    np.testing.assert_array_equal(new, shape_fit.smpl_rest_joints(model_arrays, betas))


def test_betas_wider_than_the_model_are_truncated_the_same_way(anthro, model_arrays, rng):
    betas = rng.normal(0, 1, 16)
    np.testing.assert_array_equal(
        kinematics.rest_joints(Model(**model_arrays), betas),
        anthro.rest_joints(model_arrays, betas),
    )


def test_segment_lengths_match(anthro, rng):
    rest = rng.normal(0, 0.3, (24, 3))
    np.testing.assert_array_equal(kinematics.segment_lengths(rest), anthro.segment_lengths(rest))


def test_single_frame_fk_matches_with_the_pelvis_at_its_rest_position(anthro, rng):
    # the local copy puts the pelvis at j_rest[0] and takes no translation -> package with trans = 0
    rest = rng.normal(0, 0.3, (24, 3))
    local = rotations.axis_angle_to_matrix(rng.normal(0, 0.6, (24, 3)))
    expected_positions, expected_global = anthro.smpl_fk_positions(rest, local)
    positions, world = kinematics.fk(rest, local, np.zeros(3))
    np.testing.assert_array_equal(world, expected_global)
    # per-frame `@` locally against `einsum` in the package: allclose, not bit-equal
    np.testing.assert_allclose(positions, expected_positions, rtol=0, atol=1e-12)


def test_batched_fk_matches_with_the_pelvis_at_the_origin(anthro, rng):
    # the local batch copy puts the pelvis at the origin -> package with trans = -j_rest[0]
    rest = rng.normal(0, 0.3, (24, 3))
    local = rotations.axis_angle_to_matrix(rng.normal(0, 0.6, (7, 24, 3)))
    expected = anthro.fk_positions_batch(rest, local)
    positions, _ = kinematics.fk_batch(rest, local, np.tile(-rest[0], (7, 1)))
    np.testing.assert_allclose(positions, expected, rtol=0, atol=1e-12)


def test_gravity_frame_change_matches_to_z_up():
    for gravity in ((0.0, -9.80665, 0.0), (0.0, 0.0, -9.81), (0.3, -9.7, 0.2), (0.0, 9.81, 0.0)):
        new = frames.transform_for_gravity(gravity, target_up="z")
        old = world_frame.transform_for_gravity(gravity)
        np.testing.assert_allclose(new.rotation, old.rotation, rtol=0, atol=1e-15)
        vectors = np.random.default_rng(1).normal(0, 1, (5, 3))
        np.testing.assert_allclose(new.apply(vectors), old.apply(vectors), rtol=0, atol=1e-15)


def test_quaternion_algebra_matches(rng):
    a = rotations.normalise(rng.normal(0, 1, (9, 4)))
    b = rotations.normalise(rng.normal(0, 1, (9, 4)))
    np.testing.assert_array_equal(rotations.normalise(a), quaternion.normalise(a))
    np.testing.assert_array_equal(rotations.conjugate(a), quaternion.conjugate(a))
    np.testing.assert_array_equal(rotations.multiply(a, b), quaternion.multiply(a, b))
    np.testing.assert_array_equal(rotations.canonicalise_sign(a), quaternion.canonicalise_sign(a))
    np.testing.assert_array_equal(rotations.log_unit(a), quaternion.log_unit(a))
    v = rng.normal(0, 0.5, (9, 3))
    np.testing.assert_array_equal(rotations.exp_pure(v), quaternion.exp_pure(v))
    t = rng.random(9)
    np.testing.assert_array_equal(rotations.slerp(a, b, t), quaternion.slerp(a, b, t))
    series = rotations.normalise(np.cumsum(rng.normal(0, 0.05, (30, 4)), axis=0) + [1, 0, 0, 0])
    np.testing.assert_array_equal(
        rotations.body_angular_velocity(series, 0.01), quaternion.body_angular_velocity(series, 0.01)
    )


def test_kabsch_with_reflection_correction_matches(rng):
    for _ in range(20):
        covariance = rng.normal(0, 1, (3, 3))
        np.testing.assert_array_equal(
            rotations.proper_rotation_from_covariance(covariance),
            rigid_body.proper_rotation_from_covariance(covariance),
        )
