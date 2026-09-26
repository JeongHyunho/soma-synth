"""Cluster pose extraction: reflection guard, exactness, residual QC."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from soma_synth.kinematics import rigid_body as rb

PLATE_LOCAL_M = np.array(
    [
        [0.0000, 0.0000, 0.0],
        [0.0531, 0.0000, 0.0],
        [0.0531, 0.0550, 0.0],
        [-0.0109, 0.0641, 0.0],
    ]
)
NON_PLANAR_LOCAL_M = np.vstack([PLATE_LOCAL_M, [[0.02, 0.02, 0.03]]])


def place(local: np.ndarray, rotation: Rotation, translation: np.ndarray) -> np.ndarray:
    centred = local - local.mean(axis=0)
    return (rotation.as_matrix() @ centred.T).T + translation


class TestProperRotation:
    def test_recovered_rotation_is_proper_for_a_planar_plate(self) -> None:
        """A coplanar constellation is the case where a naive SVD can hand back a mirror."""
        rotation = Rotation.from_euler("xyz", [0.3, -1.1, 2.4])
        translation = np.array([0.4, -0.2, 1.1])
        observed = place(PLATE_LOCAL_M, rotation, translation)[None]
        local = rb.mean_local_shape(observed)

        pose = rb.solve_pose_series(local, observed)

        assert np.linalg.det(pose.rotation[0]) == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(
            pose.rotation[0] @ pose.rotation[0].T, np.eye(3), atol=1e-12
        )

    def test_a_mirrored_observation_does_not_yield_a_reflection(self) -> None:
        """Mirroring the constellation must not be absorbed as an improper rotation."""
        rotation = Rotation.from_euler("xyz", [0.2, 0.5, -0.9])
        base = place(NON_PLANAR_LOCAL_M, rotation, np.array([0.1, 0.2, 0.9]))
        local = rb.mean_local_shape(base[None])

        mirrored = base.copy()
        mirrored[:, 2] *= -1.0

        _, _, _ = rb.solve_single_pose(local, mirrored)
        recovered, _, residual = rb.solve_single_pose(local, mirrored)

        assert np.linalg.det(recovered) == pytest.approx(1.0, abs=1e-12)
        # A proper rotation cannot fit a mirrored body, so the misfit must surface as a
        # residual rather than be hidden by silently flipping an axis.
        assert residual > 1e-3

    def test_a_collinear_constellation_still_returns_an_orthonormal_matrix(self) -> None:
        collinear = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.10, 0.0, 0.0]])
        observed = (collinear + np.array([0.2, 0.3, 0.4]))[None]

        pose = rb.solve_pose_series(collinear - collinear.mean(axis=0), observed)

        matrix = pose.rotation[0]
        assert np.isfinite(matrix).all()
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-9)


class TestExactRecovery:
    @pytest.mark.parametrize(
        "euler,translation",
        [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([0.7, -0.3, 1.9], [1.2, -0.7, 0.4]),
            ([np.pi, 0.0, 0.0], [0.0, 0.0, 2.0]),
            ([-2.9, 1.4, 0.05], [-3.0, 0.5, 1.0]),
        ],
    )
    def test_pose_of_a_rigid_body_is_recovered_to_machine_precision(
        self, euler: list[float], translation: list[float]
    ) -> None:
        rotation = Rotation.from_euler("xyz", euler)
        offset = np.asarray(translation)
        observed = place(NON_PLANAR_LOCAL_M, rotation, offset)[None]
        local = NON_PLANAR_LOCAL_M - NON_PLANAR_LOCAL_M.mean(axis=0)

        pose = rb.solve_pose_series(local, observed)

        angle_error = Rotation.from_matrix(
            rotation.as_matrix().T @ pose.rotation[0]
        ).magnitude()
        assert angle_error < 1e-9
        assert np.linalg.norm(pose.translation[0] - offset) < 1e-9
        assert pose.residual_rms_m[0] < 1e-9

    def test_translation_convention_is_world_equals_rotation_times_local_plus_translation(
        self,
    ) -> None:
        rotation = Rotation.from_euler("xyz", [0.4, 0.9, -1.3])
        offset = np.array([0.5, -1.5, 2.5])
        local = NON_PLANAR_LOCAL_M - NON_PLANAR_LOCAL_M.mean(axis=0)
        observed = place(NON_PLANAR_LOCAL_M, rotation, offset)[None]

        pose = rb.solve_pose_series(local, observed)
        rebuilt = (pose.rotation[0] @ local.T).T + pose.translation[0]

        np.testing.assert_allclose(rebuilt, observed[0], atol=1e-12)


class TestResidualQuality:
    def test_residual_is_reported_per_frame(self) -> None:
        rotation = Rotation.from_euler("xyz", [0.1, 0.2, 0.3])
        frames = 6
        observed = np.repeat(
            place(NON_PLANAR_LOCAL_M, rotation, np.array([0.0, 0.0, 1.0]))[None],
            frames,
            axis=0,
        )
        local = NON_PLANAR_LOCAL_M - NON_PLANAR_LOCAL_M.mean(axis=0)
        observed[3, 0] += np.array([0.01, 0.0, 0.0])  # 10 mm of non-rigidity

        pose = rb.solve_pose_series(local, observed)

        assert pose.residual_rms_m.shape == (frames,)
        assert pose.residual_rms_m[3] > pose.residual_rms_m[0]
        assert pose.residual_rms_m[0] < 1e-9

    def test_high_residual_frames_are_flagged_and_kept(self) -> None:
        rotation = Rotation.from_euler("xyz", [0.1, 0.2, 0.3])
        observed = np.repeat(
            place(NON_PLANAR_LOCAL_M, rotation, np.array([0.0, 0.0, 1.0]))[None], 5, axis=0
        )
        local = NON_PLANAR_LOCAL_M - NON_PLANAR_LOCAL_M.mean(axis=0)
        observed[2, 0] += np.array([0.02, 0.0, 0.0])

        pose = rb.solve_pose_series(local, observed)
        flagged = rb.flag_high_residual(pose, threshold_m=0.003)

        assert flagged[2]
        assert flagged.sum() == 1
        # Flagged, not dropped: the pose is still solved and still available.
        assert pose.solved[2]
        assert np.isfinite(pose.rotation[2]).all()

    def test_threshold_must_be_supplied_and_positive(self) -> None:
        observed = place(NON_PLANAR_LOCAL_M, Rotation.identity(), np.zeros(3))[None]
        local = NON_PLANAR_LOCAL_M - NON_PLANAR_LOCAL_M.mean(axis=0)
        pose = rb.solve_pose_series(local, observed)

        with pytest.raises(TypeError):
            rb.flag_high_residual(pose)  # type: ignore[call-arg]
        with pytest.raises(rb.RigidBodyError, match="threshold_m"):
            rb.flag_high_residual(pose, threshold_m=0.0)


class TestSharedProperRotation:
    """One helper serves the single-frame fit, the batched fit and the mounting solve."""

    def test_batched_and_single_covariances_give_identical_rotations(self) -> None:
        rng = np.random.default_rng(3)
        local = NON_PLANAR_LOCAL_M - NON_PLANAR_LOCAL_M.mean(axis=0)
        rotations = Rotation.random(6, random_state=7).as_matrix()
        worlds = np.einsum("fij,mj->fmi", rotations, local) + rng.normal(0, 1e-4, (6, 5, 3))
        covariances = np.einsum("mi,fmj->fij", local, worlds - worlds.mean(axis=1, keepdims=True))

        batched = rb.proper_rotation_from_covariance(covariances)
        singles = np.stack([rb.proper_rotation_from_covariance(c) for c in covariances])

        np.testing.assert_allclose(batched, singles, atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(batched), 1.0, atol=1e-12)

    def test_a_mirrored_covariance_still_yields_a_proper_rotation(self) -> None:
        local = NON_PLANAR_LOCAL_M - NON_PLANAR_LOCAL_M.mean(axis=0)
        mirrored = local.copy()
        mirrored[:, 2] *= -1.0

        rotation = rb.proper_rotation_from_covariance(local.T @ mirrored)

        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)


class TestShapeGuards:
    def test_a_two_marker_constellation_is_refused(self) -> None:
        with pytest.raises(rb.RigidBodyError, match="at least 3"):
            rb.solve_pose_series(np.zeros((2, 3)), np.zeros((4, 2, 3)))

    def test_local_shape_must_match_the_constellation(self) -> None:
        with pytest.raises(rb.RigidBodyError, match="does not match"):
            rb.solve_pose_series(np.zeros((3, 3)), np.zeros((4, 4, 3)))

    def test_wrong_position_rank_is_refused(self) -> None:
        with pytest.raises(rb.RigidBodyError, match="frames, markers, 3"):
            rb.solve_pose_series(np.zeros((4, 3)), np.zeros((4, 4)))
