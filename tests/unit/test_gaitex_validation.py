"""Held-out validation checks."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from soma_synth.validation import gaitex_reference as gr

DT_S = 0.01


def to_wxyz(rotations: Rotation) -> np.ndarray:
    xyzw = rotations.as_quat()
    return np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])


def wandering_rotation(frames: int, *, seed: int = 0) -> Rotation:
    """A rotation that turns about all three axes so its axis is well determined."""
    time = np.arange(frames) * DT_S
    angles = np.column_stack(
        [1.3 * np.sin(1.7 * time), 0.9 * time, 1.1 * np.cos(2.3 * time)]
    )
    return Rotation.from_euler("xyz", angles)


class TestMountingRotation:
    def test_a_known_constant_rotation_is_recovered(self) -> None:
        """The convention is measured_row = synthesised_row @ M."""
        frames = 800
        truth = Rotation.from_euler("xyz", [0.4, -1.2, 2.1]).as_matrix()
        # Real body-frame angular velocity, not the derivative of a rotation vector, which
        # only coincides with it for rotation about a fixed axis.
        rates = gr._body_rates(to_wxyz(wandering_rotation(frames)), DT_S)
        measured_rates = rates @ truth

        recovered, used = gr.solve_mounting_rotation(
            rates, measured_rates, minimum_rate_rad_per_s=0.2
        )

        angle = Rotation.from_matrix(truth.T @ recovered).magnitude()
        assert np.degrees(angle) < 0.5
        assert used > 100

    def test_a_stationary_trial_is_refused_rather_than_fitted(self) -> None:
        rates = np.zeros((200, 3))
        with pytest.raises(gr.ValidationError, match="underdetermined"):
            gr.solve_mounting_rotation(rates, rates, minimum_rate_rad_per_s=0.2)

    def test_the_rate_threshold_must_be_positive(self) -> None:
        rates = np.ones((10, 3))
        with pytest.raises(gr.ValidationError, match="must be positive"):
            gr.solve_mounting_rotation(rates, rates, minimum_rate_rad_per_s=0.0)


class TestOrientationComparison:
    def test_identical_streams_differ_by_nothing(self) -> None:
        frames = 600
        stream = to_wxyz(wandering_rotation(frames))

        result = gr.compare_orientation(
            stream, stream, dt_s=DT_S, minimum_rate_rad_per_s=0.2
        )

        assert result.median_error_deg < 1e-6
        assert result.p95_error_deg < 1e-6
        assert result.angular_velocity_correlation == pytest.approx(1.0, abs=1e-9)
        assert result.compared_frames == frames

    def test_a_constant_mounting_and_world_offset_are_both_removed(self) -> None:
        """Neither constant is an error, so neither should show up as one."""
        frames = 600
        synthesised_rotations = wandering_rotation(frames)
        mounting = Rotation.from_euler("xyz", [0.3, -0.8, 1.4])
        world = Rotation.from_euler("z", 0.39)  # the optical-to-inertial heading offset

        synthesised = to_wxyz(synthesised_rotations)
        measured = to_wxyz(world * synthesised_rotations * mounting)

        result = gr.compare_orientation(
            synthesised, measured, dt_s=DT_S, minimum_rate_rad_per_s=0.2
        )

        assert result.median_error_deg < 0.5
        assert result.angular_velocity_correlation > 0.999

    def test_a_real_discrepancy_shows_up_as_error(self) -> None:
        frames = 600
        synthesised_rotations = wandering_rotation(frames)
        drift = Rotation.from_rotvec(
            np.column_stack(
                [np.zeros(frames), np.zeros(frames), np.linspace(0.0, 0.15, frames)]
            )
        )

        synthesised = to_wxyz(synthesised_rotations)
        measured = to_wxyz(drift * synthesised_rotations)

        result = gr.compare_orientation(
            synthesised, measured, dt_s=DT_S, minimum_rate_rad_per_s=0.2
        )

        assert result.p95_error_deg > 1.0

    def test_mismatched_shapes_are_refused(self) -> None:
        with pytest.raises(gr.ValidationError, match="same shape"):
            gr.compare_orientation(
                np.zeros((10, 4)),
                np.zeros((11, 4)),
                dt_s=DT_S,
                minimum_rate_rad_per_s=0.2,
            )


class TestMeasuredStreamStaysOut:
    def test_the_comparison_returns_statistics_only(self) -> None:
        """A held-out reference must not be able to leak into a synthesis path."""
        frames = 400
        stream = to_wxyz(wandering_rotation(frames))

        result = gr.compare_orientation(
            stream, stream, dt_s=DT_S, minimum_rate_rad_per_s=0.2
        )

        for field, value in vars(result).items():
            if isinstance(value, np.ndarray):
                # Only the 3x3 mounting rotation is an array, and it is a calibration
                # constant, not a per-frame signal that could be mistaken for data.
                assert value.shape == (3, 3), field
            else:
                assert isinstance(value, (int, float)), field

    def test_the_synthesis_module_does_not_import_the_validation_module(self) -> None:
        from soma_synth.sensors import virtual_imu

        source = inspect.getsource(virtual_imu)
        assert "validation" not in source
        assert "gaitex_reference" not in source


class TestPhysicsCheck:
    def _still(self, frames: int, magnitude: float = 9.80665):
        force = np.tile(np.array([0.0, 0.0, magnitude]), (frames, 1))
        rate = np.zeros((frames, 3))
        return force, rate

    def test_a_still_sensor_matches_gravity(self) -> None:
        force, rate = self._still(500)

        result = gr.check_physics(
            force,
            rate,
            quasi_static_rate_deg_per_s=5.0,
            quasi_static_magnitude_tolerance_m_per_s2=0.5,
            magnitude_bound_m_per_s2=100.0,
        )

        assert result.quasi_static_frames == 500
        assert result.gravity_error_m_per_s2 < 1e-9
        assert result.gravity_error_ppm < 1e-3
        assert result.magnitude_breach_frames == 0

    def test_a_biased_magnitude_is_reported_as_gravity_error(self) -> None:
        force, rate = self._still(500, magnitude=9.90)

        result = gr.check_physics(
            force,
            rate,
            quasi_static_rate_deg_per_s=5.0,
            quasi_static_magnitude_tolerance_m_per_s2=0.5,
            magnitude_bound_m_per_s2=100.0,
        )

        assert result.gravity_error_m_per_s2 == pytest.approx(0.09335, abs=1e-4)

    def test_a_fast_turning_frame_is_not_counted_as_quasi_static(self) -> None:
        frames = 500
        force, rate = self._still(frames)
        rate[:250] = [0.0, 0.0, 200.0]

        result = gr.check_physics(
            force,
            rate,
            quasi_static_rate_deg_per_s=5.0,
            quasi_static_magnitude_tolerance_m_per_s2=0.5,
            magnitude_bound_m_per_s2=100.0,
        )

        assert result.quasi_static_frames == 250

    def test_a_translating_sensor_at_constant_orientation_is_excluded(self) -> None:
        """Slow rotation alone does not mean still; the magnitude test catches this."""
        frames = 400
        force = np.tile(np.array([0.0, 0.0, 25.0]), (frames, 1))
        rate = np.zeros((frames, 3))

        result = gr.check_physics(
            force,
            rate,
            quasi_static_rate_deg_per_s=5.0,
            quasi_static_magnitude_tolerance_m_per_s2=0.5,
            magnitude_bound_m_per_s2=100.0,
        )

        assert result.quasi_static_frames == 0
        assert np.isnan(result.quasi_static_mean_magnitude)

    def test_magnitude_breaches_are_counted(self) -> None:
        frames = 300
        force, rate = self._still(frames)
        force[10:20] = [0.0, 0.0, 250.0]

        result = gr.check_physics(
            force,
            rate,
            quasi_static_rate_deg_per_s=5.0,
            quasi_static_magnitude_tolerance_m_per_s2=0.5,
            magnitude_bound_m_per_s2=100.0,
        )

        assert result.magnitude_breach_frames == 10
        assert result.checked_frames == frames

    def test_nan_frames_are_skipped(self) -> None:
        force, rate = self._still(300)
        force[50:80] = np.nan

        result = gr.check_physics(
            force,
            rate,
            quasi_static_rate_deg_per_s=5.0,
            quasi_static_magnitude_tolerance_m_per_s2=0.5,
            magnitude_bound_m_per_s2=100.0,
        )

        assert result.checked_frames == 270

    def test_thresholds_must_be_supplied_and_positive(self) -> None:
        force, rate = self._still(10)
        with pytest.raises(TypeError):
            gr.check_physics(force, rate)  # type: ignore[call-arg]
        with pytest.raises(gr.ValidationError, match="magnitude_bound"):
            gr.check_physics(
                force,
                rate,
                quasi_static_rate_deg_per_s=5.0,
                quasi_static_magnitude_tolerance_m_per_s2=0.5,
                magnitude_bound_m_per_s2=0.0,
            )
