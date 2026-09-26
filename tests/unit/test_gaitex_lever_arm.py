"""Plate frame, lever arm and its sensitivity."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from soma_synth.sensors import lever_arm as la

# The GAITEX plate as measured across 13 trials and five clusters: six distinct edge
# lengths in millimetres, reproducible to 0.1-0.6 mm, planar to 0.05-0.23 mm.
REFERENCE_EDGES_M = np.sort(np.array([53.1, 55.0, 64.1, 69.6, 82.6, 89.4]) / 1000.0)

# Coplanar coordinates fitted to those six edges (max edge error 0.075 mm). Solving for the
# constellation rather than inventing one keeps the QC test independent of the fixture: the
# reference it is checked against is the measured signature, not the fixture's own geometry.
PLATE_LOCAL_M = np.array(
    [
        [0.036029, -0.018453, 0.0],
        [-0.038487, 0.017024, 0.0],
        [0.028511, 0.036076, 0.0],
        [-0.026052, -0.034647, 0.0],
    ]
)


def measured_edges() -> np.ndarray:
    return la.edge_lengths(PLATE_LOCAL_M)


class TestEdgeSignature:
    def test_a_four_marker_plate_has_six_edges(self) -> None:
        assert measured_edges().shape == (6,)

    def test_the_signature_is_asymmetric(self) -> None:
        """Distinct edges are what make marker correspondence unique on this plate."""
        edges = measured_edges()
        assert np.min(np.diff(edges)) > 0.001

    def test_a_wrong_rank_shape_is_refused(self) -> None:
        with pytest.raises(la.LeverArmError, match="markers, 3"):
            la.edge_lengths(np.zeros((4, 2)))


class TestPlateFrame:
    def test_the_normal_is_perpendicular_to_the_plate(self) -> None:
        plate = la.estimate_plate_frame(
            PLATE_LOCAL_M, inward_reference_local=np.array([0.0, 0.0, -1.0])
        )

        in_plane = PLATE_LOCAL_M - PLATE_LOCAL_M.mean(axis=0)
        assert np.abs(in_plane @ plate.normal_local).max() < 1e-12
        assert np.linalg.norm(plate.normal_local) == pytest.approx(1.0)

    def test_planarity_is_reported_and_near_zero_for_a_flat_plate(self) -> None:
        plate = la.estimate_plate_frame(
            PLATE_LOCAL_M, inward_reference_local=np.array([0.0, 0.0, -1.0])
        )
        assert plate.planarity_m < 1e-12

    def test_planarity_grows_when_a_marker_leaves_the_plane(self) -> None:
        bent = PLATE_LOCAL_M.copy()
        bent[2, 2] += 0.004
        plate = la.estimate_plate_frame(
            bent, inward_reference_local=np.array([0.0, 0.0, -1.0])
        )
        assert plate.planarity_m > 0.001

    @pytest.mark.parametrize("sign", [-1.0, 1.0])
    def test_the_normal_follows_the_anatomical_reference(self, sign: float) -> None:
        """The sign comes from anatomy; the measured IMU is not consulted."""
        plate = la.estimate_plate_frame(
            PLATE_LOCAL_M, inward_reference_local=np.array([0.0, 0.0, sign])
        )
        assert np.sign(plate.normal_local[2]) == sign

    def test_axes_form_a_right_handed_orthonormal_triad(self) -> None:
        plate = la.estimate_plate_frame(
            PLATE_LOCAL_M, inward_reference_local=np.array([0.0, 0.0, -1.0])
        )
        matrix = plate.rotation_local_from_plate()
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
        assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-12)

    def test_a_reference_lying_in_the_plane_is_refused(self) -> None:
        with pytest.raises(la.LeverArmError, match="undecidable"):
            la.estimate_plate_frame(
                PLATE_LOCAL_M, inward_reference_local=np.array([1.0, 0.0, 0.0])
            )

    def test_a_degenerate_reference_is_refused(self) -> None:
        with pytest.raises(la.LeverArmError, match="finite and non-zero"):
            la.estimate_plate_frame(
                PLATE_LOCAL_M, inward_reference_local=np.zeros(3)
            )


class TestInwardReferenceFromAnatomy:
    def test_a_world_direction_is_carried_into_the_local_frame(self) -> None:
        frames = 40
        rotations = Rotation.from_euler(
            "xyz", np.column_stack([0.01 * np.arange(frames), np.zeros(frames), np.zeros(frames)])
        ).as_matrix()
        # A fixed local direction, expressed in world on every frame.
        local_truth = np.array([0.0, 0.0, -1.0])
        world = np.einsum("fij,j->fi", rotations, local_truth)

        recovered = la.inward_reference_local(rotations, world)

        np.testing.assert_allclose(recovered, local_truth, atol=1e-9)

    def test_frames_without_a_pose_are_skipped(self) -> None:
        frames = 10
        rotations = np.repeat(np.eye(3)[None], frames, axis=0)
        rotations[3:6] = np.nan
        world = np.repeat(np.array([[0.0, 0.0, -1.0]]), frames, axis=0)
        world[7] = np.nan

        recovered = la.inward_reference_local(rotations, world)

        np.testing.assert_allclose(recovered, [0.0, 0.0, -1.0], atol=1e-12)

    def test_no_usable_frame_is_refused(self) -> None:
        rotations = np.full((5, 3, 3), np.nan)
        world = np.zeros((5, 3))
        with pytest.raises(la.LeverArmError, match="no frame"):
            la.inward_reference_local(rotations, world)


class TestLeverArm:
    def test_a_pure_depth_offset_runs_along_the_inward_normal(self) -> None:
        plate = la.estimate_plate_frame(
            PLATE_LOCAL_M, inward_reference_local=np.array([0.0, 0.0, -1.0])
        )

        lever = la.lever_arm_local(
            plate, depth_m=0.014, in_plane_offset_m=np.zeros(2)
        )

        np.testing.assert_allclose(lever, 0.014 * plate.normal_local, atol=1e-12)
        assert np.linalg.norm(lever) == pytest.approx(0.014)

    def test_an_in_plane_offset_stays_in_the_plane(self) -> None:
        plate = la.estimate_plate_frame(
            PLATE_LOCAL_M, inward_reference_local=np.array([0.0, 0.0, -1.0])
        )

        lever = la.lever_arm_local(
            plate, depth_m=0.0, in_plane_offset_m=np.array([0.003, -0.002])
        )

        assert abs(float(lever @ plate.normal_local)) < 1e-12
        assert np.linalg.norm(lever) == pytest.approx(np.hypot(0.003, 0.002))

    def test_depth_and_offset_have_no_defaults(self) -> None:
        """The hold forbids a module-level execution constant; callers must supply both."""
        plate = la.estimate_plate_frame(
            PLATE_LOCAL_M, inward_reference_local=np.array([0.0, 0.0, -1.0])
        )
        with pytest.raises(TypeError):
            la.lever_arm_local(plate)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            la.lever_arm_local(plate, depth_m=0.014)  # type: ignore[call-arg]

    def test_a_wrong_offset_rank_is_refused(self) -> None:
        plate = la.estimate_plate_frame(
            PLATE_LOCAL_M, inward_reference_local=np.array([0.0, 0.0, -1.0])
        )
        with pytest.raises(la.LeverArmError, match="2-vector"):
            la.lever_arm_local(plate, depth_m=0.014, in_plane_offset_m=np.zeros(3))


class TestPlateGeometryQc:
    def _plate(self, shape: np.ndarray) -> la.PlateFrame:
        return la.estimate_plate_frame(
            shape, inward_reference_local=np.array([0.0, 0.0, -1.0])
        )

    def test_the_reference_plate_passes(self) -> None:
        report = la.check_plate_geometry(
            self._plate(PLATE_LOCAL_M),
            reference_edges_m=REFERENCE_EDGES_M,
            edge_tolerance_m=0.001,
            planarity_tolerance_m=0.0005,
        )
        assert report.passed
        assert report.max_edge_deviation_m < 0.001

    def test_a_stretched_plate_fails_the_edge_check(self) -> None:
        stretched = PLATE_LOCAL_M * 1.05
        report = la.check_plate_geometry(
            self._plate(stretched),
            reference_edges_m=REFERENCE_EDGES_M,
            edge_tolerance_m=0.001,
            planarity_tolerance_m=0.0005,
        )
        assert not report.edge_within_tolerance
        assert not report.passed

    def test_a_buckled_plate_fails_the_planarity_check(self) -> None:
        bent = PLATE_LOCAL_M.copy()
        bent[1, 2] += 0.005
        report = la.check_plate_geometry(
            self._plate(bent),
            reference_edges_m=REFERENCE_EDGES_M,
            edge_tolerance_m=0.010,
            planarity_tolerance_m=0.0005,
        )
        assert not report.planarity_within_tolerance
        assert not report.passed

    def test_tolerances_must_be_positive(self) -> None:
        with pytest.raises(la.LeverArmError, match="tolerances"):
            la.check_plate_geometry(
                self._plate(PLATE_LOCAL_M),
                reference_edges_m=REFERENCE_EDGES_M,
                edge_tolerance_m=0.0,
                planarity_tolerance_m=0.0005,
            )


class TestCentripetalSensitivity:
    def test_the_error_scales_with_the_square_of_the_rate(self) -> None:
        speeds = np.array([0.0, 5.0, 10.0])

        result = la.centripetal_sensitivity(speeds, depth_uncertainty_m=0.006)

        assert result.max_m_per_s2 == pytest.approx(100.0 * 0.006)
        assert result.mean_m_per_s2 == pytest.approx((0.0 + 25.0 + 100.0) / 3.0 * 0.006)

    def test_a_realistic_shank_rate_series_gives_a_sub_gravity_error(self) -> None:
        """A depth of 14 +/- 6 mm is assumed; this is the budget that implies."""
        rng = np.random.default_rng(0)
        speeds = np.abs(rng.normal(2.6, 1.5, size=20000))

        result = la.centripetal_sensitivity(speeds, depth_uncertainty_m=0.006)

        assert result.p95_m_per_s2 < 9.80665
        assert result.mean_m_per_s2 > 0.0

    def test_non_finite_samples_are_ignored(self) -> None:
        speeds = np.array([np.nan, 4.0, np.inf])
        result = la.centripetal_sensitivity(speeds, depth_uncertainty_m=0.01)
        assert result.max_m_per_s2 == pytest.approx(16.0 * 0.01)

    def test_an_all_nan_series_is_refused(self) -> None:
        with pytest.raises(la.LeverArmError, match="no finite"):
            la.centripetal_sensitivity(np.full(4, np.nan), depth_uncertainty_m=0.01)

    def test_a_negative_uncertainty_is_refused(self) -> None:
        with pytest.raises(la.LeverArmError, match="negative"):
            la.centripetal_sensitivity(np.array([1.0]), depth_uncertainty_m=-0.001)
