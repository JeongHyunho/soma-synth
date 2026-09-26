"""Held-out checks on a synthesised GAITEX inertial signal.

Two independent checks, with different standing.

**T1 compares against the measured IMU.** GAITEX ships each sensor's own orientation, which
is exactly what the marker-derived pipeline also produces, so the two can be differenced.
The measured stream enters here and nowhere else: it is a reference, never an input to
synthesis. The check is only held-out while that separation holds, so
:func:`compare_orientation` takes the measured stream and returns statistics --
it never returns anything a synthesis path could consume.

The comparison first has to remove the constant rotation between the marker plate frame and
the sensor's own axes. Both streams observe the same rigid body, so their body-frame angular
velocities differ by that constant rotation alone, and a Procrustes fit over the whole trial
recovers it. Orientation error is then the residual.

**T2 checks physics and needs no reference at all.** On a quasi-static frame the specific
force must equal gravity, so its magnitude is a check against a constant of nature rather
than against another measurement. That makes it the stronger of the two where it applies --
but it applies only where the body is nearly still, and it says nothing about dynamic
accuracy. GAITEX publishes no accelerometer, so nothing here can validate synthesised
acceleration during movement; that requires a source which pairs markers with a measured
accelerometer.

Thresholds are required arguments, never code constants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from soma_synth.kinematics import quaternion as quat
from soma_synth.kinematics.rigid_body import proper_rotation_from_covariance

__all__ = [
    "OrientationComparison",
    "PhysicsCheck",
    "ValidationError",
    "check_physics",
    "compare_orientation",
    "solve_mounting_rotation",
]

_STANDARD_GRAVITY = 9.80665
"""CODATA standard gravity. A defined constant, not a tunable threshold."""


class ValidationError(ValueError):
    """Raised when a validation comparison cannot be made as configured."""


@dataclass(frozen=True)
class OrientationComparison:
    """T1: synthesised orientation against the measured sensor's own."""

    mounting_rotation: np.ndarray
    median_error_deg: float
    p95_error_deg: float
    max_error_deg: float
    compared_frames: int
    angular_velocity_correlation: float


@dataclass(frozen=True)
class PhysicsCheck:
    """T2: specific force against gravity, and against configured bounds."""

    quasi_static_frames: int
    quasi_static_mean_magnitude: float
    quasi_static_std_magnitude: float
    gravity_error_m_per_s2: float
    magnitude_breach_frames: int
    checked_frames: int

    @property
    def gravity_error_ppm(self) -> float:
        return self.gravity_error_m_per_s2 / _STANDARD_GRAVITY * 1e6


def _body_rates(orientation_wxyz: np.ndarray, dt_s: float) -> np.ndarray:
    """Body-frame angular velocity by the central log map, NaN at both ends.

    The same estimator the synthesis path uses, so a comparison never differs from the
    signal it is judging merely because the two were differentiated differently.
    """
    return quat.body_angular_velocity(orientation_wxyz, dt_s)


def solve_mounting_rotation(
    synthesised_rates: np.ndarray,
    measured_rates: np.ndarray,
    *,
    minimum_rate_rad_per_s: float,
) -> tuple[np.ndarray, int]:
    """Recover the constant rotation carrying synthesised body axes onto measured ones.

    The convention is fixed and row-major to match how the rate arrays are laid out:
    the returned ``M`` satisfies ``measured_row = synthesised_row @ M``. Equivalently, if the
    two orientation streams are related by ``R_measured = R_world . R_synthesised . M`` then
    this is that same ``M``, which is why :func:`compare_orientation` can multiply the
    synthesised quaternions by it directly.

    Only frames turning faster than ``minimum_rate_rad_per_s`` contribute, because a nearly
    stationary frame has no well-defined rotation axis and would add noise without adding
    information.
    """
    if minimum_rate_rad_per_s <= 0.0:
        raise ValidationError(
            f"minimum_rate_rad_per_s must be positive, got {minimum_rate_rad_per_s}"
        )
    usable = (
        np.isfinite(synthesised_rates).all(axis=1)
        & np.isfinite(measured_rates).all(axis=1)
        & (np.linalg.norm(synthesised_rates, axis=1) > minimum_rate_rad_per_s)
    )
    count = int(usable.sum())
    if count < 3:
        raise ValidationError(
            f"only {count} frames turn faster than {minimum_rate_rad_per_s} rad/s; "
            f"the mounting rotation is underdetermined"
        )

    # Minimise ||synthesised @ M - measured||. The shared helper returns the column-vector
    # rotation V diag(1,1,d) U^T for H = synthesised.T @ measured; the row-vector M this
    # function promises is its transpose, U diag(1,1,d) V^T.
    covariance = synthesised_rates[usable].T @ measured_rates[usable]
    return proper_rotation_from_covariance(covariance).T, count


def compare_orientation(
    synthesised_wxyz: np.ndarray,
    measured_wxyz: np.ndarray,
    *,
    dt_s: float,
    minimum_rate_rad_per_s: float,
) -> OrientationComparison:
    """T1. Returns statistics only; the measured stream never leaves this function."""
    if synthesised_wxyz.shape != measured_wxyz.shape:
        raise ValidationError(
            f"synthesised {synthesised_wxyz.shape} and measured {measured_wxyz.shape} "
            f"must have the same shape"
        )

    synthesised_rates = _body_rates(synthesised_wxyz, dt_s)
    measured_rates = _body_rates(measured_wxyz, dt_s)
    mounting, used = solve_mounting_rotation(
        synthesised_rates, measured_rates, minimum_rate_rad_per_s=minimum_rate_rad_per_s
    )

    both = np.isfinite(synthesised_rates).all(axis=1) & np.isfinite(measured_rates).all(
        axis=1
    )
    rotated = synthesised_rates[both] @ mounting
    correlation = float(
        np.corrcoef(
            np.linalg.norm(rotated, axis=1),
            np.linalg.norm(measured_rates[both], axis=1),
        )[0, 1]
    )

    # The world-frame alignment between the optical and inertial frames is also constant, so
    # it can be taken out the same way: the residual is what neither constant explains.
    finite = np.isfinite(synthesised_wxyz).all(axis=1) & np.isfinite(
        measured_wxyz
    ).all(axis=1)
    if finite.sum() < 2:
        raise ValidationError("fewer than two frames have both orientations")

    # The model is q_measured = q_world . q_synthesised . q_mounting, with both q_world and
    # q_mounting constant. q_mounting is already known from the rates, so q_world follows by
    # right-multiplying the inverse of what the mounting alone predicts -- the world offset
    # sits on the left, so its inverse comes off on the right.
    mounting_quaternion = _matrix_to_wxyz_single(mounting)
    predicted = quat.multiply(synthesised_wxyz[finite], mounting_quaternion[None])
    world_offset = quat.canonicalise_sign(
        quat.multiply(measured_wxyz[finite], quat.conjugate(predicted))
    )
    mean_offset = quat.normalise(world_offset.mean(axis=0)[None])

    corrected = quat.multiply(np.repeat(mean_offset, int(finite.sum()), 0), predicted)
    residual = quat.multiply(quat.conjugate(corrected), measured_wxyz[finite])
    residual = np.where(residual[:, :1] < 0.0, -residual, residual)
    errors = np.degrees(2.0 * np.linalg.norm(quat.log_unit(residual)[:, 1:], axis=1))

    return OrientationComparison(
        mounting_rotation=mounting,
        median_error_deg=float(np.median(errors)),
        p95_error_deg=float(np.percentile(errors, 95)),
        max_error_deg=float(errors.max()),
        compared_frames=int(finite.sum()),
        angular_velocity_correlation=correlation,
    )


def _matrix_to_wxyz_single(matrix: np.ndarray) -> np.ndarray:
    """Convert one rotation matrix to a (w, x, y, z) quaternion."""
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return np.array([w, x, y, z])


def check_physics(
    specific_force_m_per_s2: np.ndarray,
    angular_velocity_deg_per_s: np.ndarray,
    *,
    quasi_static_rate_deg_per_s: float,
    quasi_static_magnitude_tolerance_m_per_s2: float,
    magnitude_bound_m_per_s2: float,
) -> PhysicsCheck:
    """T2. Gravity agreement on quasi-static frames, plus a magnitude bound everywhere.

    A frame counts as quasi-static when it is turning slowly *and* its specific force is
    already close to one gravity; requiring both keeps a fast linear translation at constant
    orientation from being mistaken for stillness.
    """
    for name, value in (
        ("quasi_static_rate_deg_per_s", quasi_static_rate_deg_per_s),
        (
            "quasi_static_magnitude_tolerance_m_per_s2",
            quasi_static_magnitude_tolerance_m_per_s2,
        ),
        ("magnitude_bound_m_per_s2", magnitude_bound_m_per_s2),
    ):
        if value <= 0.0:
            raise ValidationError(f"{name} must be positive, got {value}")

    finite = np.isfinite(specific_force_m_per_s2).all(axis=1) & np.isfinite(
        angular_velocity_deg_per_s
    ).all(axis=1)
    if not finite.any():
        raise ValidationError("no frame has both a specific force and an angular rate")

    magnitude = np.linalg.norm(specific_force_m_per_s2[finite], axis=1)
    rate = np.linalg.norm(angular_velocity_deg_per_s[finite], axis=1)

    quasi_static = (rate < quasi_static_rate_deg_per_s) & (
        np.abs(magnitude - _STANDARD_GRAVITY)
        < quasi_static_magnitude_tolerance_m_per_s2
    )
    breaches = int((magnitude > magnitude_bound_m_per_s2).sum())

    if quasi_static.any():
        selected = magnitude[quasi_static]
        mean = float(selected.mean())
        std = float(selected.std())
        error = abs(mean - _STANDARD_GRAVITY)
    else:
        mean = float("nan")
        std = float("nan")
        error = float("nan")

    return PhysicsCheck(
        quasi_static_frames=int(quasi_static.sum()),
        quasi_static_mean_magnitude=mean,
        quasi_static_std_magnitude=std,
        gravity_error_m_per_s2=error,
        magnitude_breach_frames=breaches,
        checked_frames=int(finite.sum()),
    )
