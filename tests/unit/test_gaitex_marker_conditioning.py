"""Stage A conditioning and Stage B rigid reconstruction."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from soma_synth.kinematics import markers as mk
from soma_synth.kinematics import rigid_body as rb

# GAITEX mounts one standardised planar plate on every IMU. Its six edge lengths are all
# different (53.1 / 55.1 / 64.0 / 70.3 / 82.2 / 90.0 mm measured across 13 trials), which is
# what makes a swapped label detectable from geometry alone.
PLATE_LOCAL_M = np.array(
    [
        [0.0000, 0.0000, 0.0],
        [0.0531, 0.0000, 0.0],
        [0.0531, 0.0550, 0.0],
        [-0.0109, 0.0641, 0.0],
    ]
)


def rigid_track(frames: int, *, seed: int = 0) -> np.ndarray:
    """Move the plate through a smooth rigid trajectory: (frames, 4, 3) world positions."""
    time = np.arange(frames) * 0.01
    angles = np.column_stack([0.6 * time, 0.4 * np.sin(2.0 * time), 0.2 * time])
    rotations = Rotation.from_euler("xyz", angles)
    translations = np.column_stack(
        [0.3 * np.sin(time), 0.2 * time, 1.0 + 0.05 * np.cos(3.0 * time)]
    )
    centred = PLATE_LOCAL_M - PLATE_LOCAL_M.mean(axis=0)
    matrices = rotations.as_matrix()
    return np.einsum("fij,mj->fmi", matrices, centred) + translations[:, None, :]


class TestOcclusionSentinel:
    def test_exact_zero_triplet_becomes_nan(self) -> None:
        positions = np.ones((5, 2, 3))
        positions[2, 1, :] = 0.0

        cleaned, sentinel = mk.apply_occlusion_sentinel(positions)

        assert sentinel[2, 1]
        assert sentinel.sum() == 1
        assert np.isnan(cleaned[2, 1]).all()
        assert np.isfinite(cleaned[np.arange(5) != 2]).all()

    def test_marker_genuinely_near_the_origin_is_left_alone(self) -> None:
        """Only a bit-exact triplet is a sentinel; real data near the origin must survive."""
        positions = np.ones((3, 3, 3))
        positions[0, 0] = [1e-9, 1e-9, 1e-9]  # tiny but nonzero
        positions[1, 1] = [0.0, 0.0, 1e-12]  # two axes zero, third not
        positions[2, 2] = [0.0, 0.0, 0.0]  # the only true sentinel

        cleaned, sentinel = mk.apply_occlusion_sentinel(positions)

        assert sentinel.sum() == 1
        assert sentinel[2, 2]
        assert np.isfinite(cleaned[0, 0]).all()
        assert np.isfinite(cleaned[1, 1]).all()


class TestSpeedOutliers:
    def test_isolated_spike_is_flagged_and_its_neighbours_are_not(self) -> None:
        track = rigid_track(40)
        track[20, 0] += np.array([0.5, 0.0, 0.0])  # 50 m/s over one 10 ms step

        flagged = mk.flag_speed_outliers(track, dt_s=0.01, max_speed_m_per_s=10.0)

        assert flagged[20, 0]
        assert not flagged[19, 0]
        assert not flagged[21, 0]
        assert flagged[:, 1:].sum() == 0

    def test_a_clean_track_is_untouched(self) -> None:
        flagged = mk.flag_speed_outliers(rigid_track(60), dt_s=0.01, max_speed_m_per_s=10.0)
        assert flagged.sum() == 0

    def test_non_positive_parameters_are_rejected(self) -> None:
        track = rigid_track(5)
        with pytest.raises(ValueError, match="dt_s"):
            mk.flag_speed_outliers(track, dt_s=0.0, max_speed_m_per_s=10.0)
        with pytest.raises(ValueError, match="max_speed"):
            mk.flag_speed_outliers(track, dt_s=0.01, max_speed_m_per_s=-1.0)


class TestRigidInvariant:
    def test_reference_distances_recover_the_plate_geometry(self) -> None:
        reference = mk.reference_pair_distances(rigid_track(50))

        expected = np.linalg.norm(
            PLATE_LOCAL_M[:, None, :] - PLATE_LOCAL_M[None, :, :], axis=2
        )
        np.testing.assert_allclose(reference, expected, atol=1e-9)

    def test_a_displaced_marker_is_flagged_and_its_neighbours_are_not(self) -> None:
        track = rigid_track(30)
        reference = mk.reference_pair_distances(track)
        track[10, 2] += np.array([0.02, 0.0, 0.0])  # 20 mm off its rigid position

        flagged = mk.flag_rigid_violations(
            track, reference_distances=reference, tolerance_m=0.002
        )

        assert flagged[10, 2]
        assert flagged[10].sum() == 1
        assert flagged[np.arange(30) != 10].sum() == 0

    def test_a_swapped_label_pair_is_detected(self) -> None:
        """An asymmetric plate makes a swap change both markers' distance signatures."""
        track = rigid_track(30)
        reference = mk.reference_pair_distances(track)
        track[[15], 0], track[[15], 1] = track[[15], 1].copy(), track[[15], 0].copy()

        flagged = mk.flag_rigid_violations(
            track, reference_distances=reference, tolerance_m=0.002
        )

        assert flagged[15, 0] and flagged[15, 1]
        assert flagged[np.arange(30) != 15].sum() == 0

    def test_tolerance_must_be_positive(self) -> None:
        track = rigid_track(5)
        reference = mk.reference_pair_distances(track)
        with pytest.raises(ValueError, match="tolerance_m"):
            mk.flag_rigid_violations(track, reference_distances=reference, tolerance_m=0.0)


class TestConditioningPipeline:
    def test_all_three_defect_classes_are_separated(self) -> None:
        track = rigid_track(60)
        track[5, 0] = 0.0  # occlusion sentinel
        track[30, 1] += np.array([0.6, 0.0, 0.0])  # speed spike
        track[45, 2] += np.array([0.02, 0.0, 0.0])  # rigid violation

        report = mk.condition_cluster(
            track, dt_s=0.01, max_speed_m_per_s=10.0, rigid_tolerance_m=0.002
        )

        assert report.sentinel_mask[5, 0] and report.sentinel_mask.sum() == 1
        assert report.speed_mask[30, 1] and report.speed_mask.sum() == 1
        assert report.rigid_mask[45, 2] and report.rigid_mask.sum() == 1
        assert report.withdrawn_mask.sum() == 3
        assert np.isnan(report.positions_m[report.withdrawn_mask]).all()
        assert report.visible_count[0] == 4
        assert report.visible_count[5] == 3


class TestRigidReconstruction:
    def test_pose_is_solved_from_three_of_four_markers(self) -> None:
        track = rigid_track(20)
        local = rb.mean_local_shape(track)
        observed = track.copy()
        observed[7, 3] = np.nan

        pose = rb.solve_pose_series(local, observed)

        assert pose.visible_count[7] == 3
        assert pose.solved[7]
        assert pose.residual_rms_m[7] == pytest.approx(0.0, abs=1e-9)

    def test_two_visible_markers_leave_the_pose_unsolved(self) -> None:
        track = rigid_track(20)
        local = rb.mean_local_shape(track)
        observed = track.copy()
        observed[7, 2:] = np.nan

        pose = rb.solve_pose_series(local, observed)

        assert pose.visible_count[7] == 2
        assert not pose.solved[7]
        assert np.isnan(pose.rotation[7]).all()
        assert np.isnan(pose.translation[7]).all()

    def test_a_removed_marker_is_reconstructed_to_better_than_half_a_millimetre(
        self,
    ) -> None:
        track = rigid_track(40)
        local = rb.mean_local_shape(track)
        observed = track.copy()
        observed[np.arange(10, 30), 1] = np.nan

        pose = rb.solve_pose_series(local, observed)
        filled = rb.reconstruct_markers(local, pose, observed)

        error = np.linalg.norm(filled[10:30, 1] - track[10:30, 1], axis=1)
        assert np.isfinite(filled[10:30, 1]).all()
        assert error.max() < 0.0005
        # Untouched markers must come back bit-identical.
        np.testing.assert_array_equal(filled[:, 0], track[:, 0])

    def test_reconstruction_leaves_unsolved_frames_as_nan(self) -> None:
        track = rigid_track(20)
        local = rb.mean_local_shape(track)
        observed = track.copy()
        observed[9, 1:] = np.nan  # only one marker left

        filled = rb.reconstruct_markers(local, rb.solve_pose_series(local, observed), observed)

        assert np.isnan(filled[9, 1:]).all()

    def test_a_shape_cannot_be_learned_without_a_complete_frame(self) -> None:
        track = rigid_track(10)
        track[:, 0] = np.nan
        with pytest.raises(rb.RigidBodyError, match="every marker visible"):
            rb.mean_local_shape(track)
