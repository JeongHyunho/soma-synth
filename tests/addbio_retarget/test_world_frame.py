"""Y-up source into the Z-up world the rest of the corpus already lives in.

AddBiomechanics models declare `gravity 0 -9.80665 0`; PRISM and AMASS artifacts are
right-handed Z-up with gravity along -Z. The rotation is derived from the gravity the model
itself declares rather than assumed, so a study that ever ships a different convention is
rotated correctly instead of silently tipped on its side.

Gravity fixes two of the three degrees of freedom. The remaining spin about the vertical is
not recoverable from gravity and is not comparable between capture labs anyway, so it is
pinned to zero and written into the artifact as a constant.
"""

import numpy as np
import pytest

from soma_synth.addbio_retarget.world_frame import (
    SOMA_GRAVITY_ZUP,
    SOURCE_GRAVITY_YUP,
    WorldFrameTransform,
    rotation_from_gravity,
    transform_for_gravity,
)


@pytest.fixture(scope="module")
def transform():
    return transform_for_gravity(SOURCE_GRAVITY_YUP)


def test_the_rotation_is_a_rotation(transform):
    rotation = transform.rotation
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_gravity_ends_up_exactly_on_the_corpus_constant(transform):
    """Source and corpus agree on the magnitude, so the rotation alone must land on it.

    Nothing here rescales; if this ever needs a tolerance wider than floating point, the
    source stopped declaring the same gravity and that is worth failing over.
    """
    rotated = transform.rotation @ np.asarray(SOURCE_GRAVITY_YUP)
    np.testing.assert_allclose(rotated, SOMA_GRAVITY_ZUP, atol=1e-12)
    assert np.linalg.norm(SOURCE_GRAVITY_YUP) == pytest.approx(
        np.linalg.norm(SOMA_GRAVITY_ZUP), abs=1e-12
    )


def test_the_basis_images_are_pinned(transform):
    """A change here silently reorients every artifact, so it has to fail a test first."""
    np.testing.assert_allclose(
        transform.rotation,
        [[1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0],
         [0.0, 1.0, 0.0]],
        atol=1e-12,
    )
    assert transform.quaternion_wxyz == pytest.approx(
        (np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0)
    )


def test_up_and_forward_land_where_expected(transform):
    np.testing.assert_allclose(transform.apply([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(transform.apply([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(transform.apply([0.0, 0.0, 1.0]), [0.0, -1.0, 0.0], atol=1e-12)


def test_it_applies_to_stacks_of_points_without_reshaping_by_hand(transform):
    points = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
    out = transform.apply(points)
    assert out.shape == points.shape
    for i in range(2):
        for j in range(4):
            np.testing.assert_allclose(out[i, j], transform.rotation @ points[i, j])


def test_the_inverse_returns_the_source_frame(transform):
    points = np.array([[0.3, 1.2, -0.7], [-2.0, 0.9, 4.4]])
    np.testing.assert_allclose(transform.invert(transform.apply(points)), points, atol=1e-12)


def test_distances_and_angles_survive(transform):
    a = np.array([0.4, 1.1, -0.2])
    b = np.array([-0.9, 0.3, 2.0])
    assert np.linalg.norm(transform.apply(a) - transform.apply(b)) == pytest.approx(
        np.linalg.norm(a - b)
    )


def test_an_already_z_up_source_is_left_alone():
    identity = transform_for_gravity(SOMA_GRAVITY_ZUP)
    np.testing.assert_allclose(identity.rotation, np.eye(3), atol=1e-12)


def test_a_source_hanging_the_other_way_up_is_still_handled():
    """+Y up would need the opposite turn; the minimal rotation supplies it."""
    flipped = transform_for_gravity((0.0, 9.80665, 0.0))
    rotated = flipped.rotation @ np.asarray([0.0, 9.80665, 0.0])
    np.testing.assert_allclose(rotated, [0.0, 0.0, -9.80665], atol=1e-12)
    assert np.linalg.det(flipped.rotation) == pytest.approx(1.0)


def test_gravity_of_zero_length_is_refused():
    with pytest.raises(ValueError):
        rotation_from_gravity((0.0, 0.0, 0.0))


def test_the_transform_reports_itself_for_the_artifact(transform):
    assert isinstance(transform, WorldFrameTransform)
    assert transform.frame_convention == "addbio_world_converted_Zup"
    assert transform.source_gravity == pytest.approx(SOURCE_GRAVITY_YUP)
