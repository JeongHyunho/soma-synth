"""Synthesise an inertial signal at a point on a tracked rigid body.

Given a cluster's pose and the lever arm to a sensor origin, the sensor's specific force and
angular rate follow from kinematics alone. Two choices in here matter enough to state.

**The sensor's world trajectory is built first, then differentiated once.** Writing the
acceleration as ``a_c + omega_dot x r + omega x (omega x r)`` needs a derivative of angular
velocity, which is a second derivative of orientation and noisier than anything else in the
chain. Forming ``p_c + R r`` and taking its second derivative gives the same quantity with
the differentiation order of the position alone.

**Angular velocity comes from the logarithmic map, not a finite difference of R.** For a
constant rate, ``log(R(t-dt)^T R(t+dt)) / (2 dt)`` is exact, whereas differencing rotation
matrices carries an ``O(dt^2)`` error that grows with the cube of the rate. The result is
also guaranteed to be a rotation vector rather than the skew part of a matrix that has
drifted off SO(3).

Positions are low-pass filtered before differentiation because the second derivative
amplifies marker noise by ``1/dt^2``. Orientation is not filtered: a four-marker Procrustes
fit already resolves it to well under a degree, and filtering rotation matrices componentwise
would take them off the manifold. Angular velocity, being an ordinary vector, is filtered.

Cut-off, filter order, edge trim and gravity are all required arguments, never code
constants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

from soma_synth.kinematics import quaternion as quat

__all__ = [
    "VirtualImuError",
    "VirtualImuSignal",
    "angular_velocity_rad_per_s",
    "contiguous_runs",
    "sensor_world_track",
    "synthesise_virtual_imu",
]

_RAD_TO_DEG = 180.0 / np.pi


class VirtualImuError(ValueError):
    """Raised when a virtual inertial signal cannot be synthesised as configured."""


@dataclass(frozen=True)
class VirtualImuSignal:
    """One virtual sensor's channels on the source time grid.

    Frames outside a usable span -- unsolved pose, or inside a filter edge transient -- are
    NaN in every channel and false in ``valid``. Nothing is extrapolated into them.
    """

    orientation_wxyz: np.ndarray
    acceleration_m_per_s2: np.ndarray
    angular_velocity_deg_per_s: np.ndarray
    valid: np.ndarray

    def __len__(self) -> int:
        return int(self.valid.shape[0])


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``(start, stop)`` spans of consecutive true values."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise VirtualImuError(f"expected a 1-D mask, got shape {mask.shape}")
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.diff(padded)
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist())
    )


def sensor_world_track(
    translation: np.ndarray, rotation: np.ndarray, lever_arm_local: np.ndarray
) -> np.ndarray:
    """World position of the sensor origin: ``p_c + R r``."""
    lever = np.asarray(lever_arm_local, dtype=np.float64)
    if lever.shape != (3,):
        raise VirtualImuError(f"lever arm must be a 3-vector, got {lever.shape}")
    if rotation.ndim != 3 or rotation.shape[1:] != (3, 3):
        raise VirtualImuError(f"expected (frames, 3, 3) rotations, got {rotation.shape}")
    if translation.shape != rotation.shape[:1] + (3,):
        raise VirtualImuError(
            f"translation {translation.shape} does not match rotation {rotation.shape}"
        )
    return translation + np.einsum("fij,j->fi", rotation, lever)


def _matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to (w, x, y, z), sign-continuous along the series.

    SciPy's conversion already handles the near-180-degree cases where the naive trace
    formula loses precision; only the component order and the sign continuity are ours.
    """
    xyzw = Rotation.from_matrix(rotation).as_quat()
    wxyz = np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])
    return quat.canonicalise_sign(wxyz)


def angular_velocity_rad_per_s(orientation_wxyz: np.ndarray, dt_s: float) -> np.ndarray:
    """Body-frame angular velocity by the logarithmic map, exact for a constant rate."""
    if dt_s <= 0.0:
        raise VirtualImuError(f"dt_s must be positive, got {dt_s}")
    return quat.body_angular_velocity(orientation_wxyz, dt_s)


def _require_valid_cutoff(*, dt_s: float, cutoff_hz: float) -> float:
    """Validate the cut-off against Nyquist and return Nyquist."""
    if dt_s <= 0.0:
        raise VirtualImuError(f"dt_s must be positive, got {dt_s}")
    nyquist = 0.5 / dt_s
    if not 0.0 < cutoff_hz < nyquist:
        raise VirtualImuError(
            f"cutoff {cutoff_hz} Hz must lie strictly between 0 and Nyquist {nyquist} Hz"
        )
    return nyquist


def _zero_phase_lowpass(
    values: np.ndarray, *, dt_s: float, cutoff_hz: float, order: int
) -> np.ndarray:
    nyquist = _require_valid_cutoff(dt_s=dt_s, cutoff_hz=cutoff_hz)
    coefficients = butter(order, cutoff_hz / nyquist, btype="low")
    return filtfilt(coefficients[0], coefficients[1], values, axis=0)


def _minimum_filter_length(*, cutoff_hz: float, dt_s: float, order: int) -> int:
    """Shortest segment ``filtfilt`` can process at this order, from its padding rule."""
    numerator, denominator = butter(order, cutoff_hz / (0.5 / dt_s), btype="low")
    return 3 * (max(len(numerator), len(denominator)) - 1) + 1


def synthesise_virtual_imu(
    *,
    translation: np.ndarray,
    rotation_world_from_cluster: np.ndarray,
    usable: np.ndarray,
    lever_arm_local: np.ndarray,
    rotation_cluster_from_sensor: np.ndarray,
    dt_s: float,
    cutoff_hz: float,
    filter_order: int,
    edge_trim_frames: int,
    gravity_world_m_per_s2: np.ndarray,
) -> VirtualImuSignal:
    """Synthesise orientation, specific force and angular rate for one sensor site.

    Each contiguous usable span is processed on its own, so a rejected interval never leaks
    its neighbours' data across the gap through the filter.
    """
    _require_valid_cutoff(dt_s=dt_s, cutoff_hz=cutoff_hz)
    if filter_order < 1:
        raise VirtualImuError(f"filter_order must be at least 1, got {filter_order}")
    if edge_trim_frames < 0:
        raise VirtualImuError(
            f"edge_trim_frames must be non-negative, got {edge_trim_frames}"
        )
    gravity = np.asarray(gravity_world_m_per_s2, dtype=np.float64)
    if gravity.shape != (3,):
        raise VirtualImuError(f"gravity must be a 3-vector, got {gravity.shape}")
    sensor_rotation = np.asarray(rotation_cluster_from_sensor, dtype=np.float64)
    if sensor_rotation.shape != (3, 3):
        raise VirtualImuError(
            f"rotation_cluster_from_sensor must be 3x3, got {sensor_rotation.shape}"
        )

    frames = translation.shape[0]
    track = sensor_world_track(translation, rotation_world_from_cluster, lever_arm_local)

    orientation = np.full((frames, 4), np.nan)
    acceleration = np.full((frames, 3), np.nan)
    angular_velocity = np.full((frames, 3), np.nan)
    valid = np.zeros(frames, dtype=bool)

    minimum_length = _minimum_filter_length(
        cutoff_hz=cutoff_hz, dt_s=dt_s, order=filter_order
    )
    for start, stop in contiguous_runs(np.asarray(usable, dtype=bool)):
        length = stop - start
        keep_start, keep_stop = start + edge_trim_frames, stop - edge_trim_frames
        if keep_stop <= keep_start:
            continue
        # A span shorter than the filter's own padding requirement is a data condition, not
        # a misconfiguration: it is left unusable rather than filtered by some other means,
        # because substituting a different filter there would make the span's noise
        # characteristics differ silently from every other span.
        # Two frames beyond the filter minimum, because the central log map consumes one at
        # each end before the rate series is itself filtered.
        if length < minimum_length + 2:
            continue

        smoothed = _zero_phase_lowpass(
            track[start:stop], dt_s=dt_s, cutoff_hz=cutoff_hz, order=filter_order
        )
        world_acceleration = np.gradient(
            np.gradient(smoothed, dt_s, axis=0), dt_s, axis=0
        )

        world_from_sensor = np.einsum(
            "fij,jk->fik", rotation_world_from_cluster[start:stop], sensor_rotation
        )
        segment_quaternions = _matrix_to_wxyz(world_from_sensor)

        specific_force = np.einsum(
            "fji,fj->fi", world_from_sensor, world_acceleration - gravity
        )
        # The central log map leaves the first and last frame of the segment undefined, so
        # only the interior is filtered. Angular velocity is an ordinary vector, which is
        # why it may be filtered at all where a rotation matrix may not.
        rate = angular_velocity_rad_per_s(segment_quaternions, dt_s)
        interior = slice(1, length - 1)
        rate[interior] = _zero_phase_lowpass(
            rate[interior], dt_s=dt_s, cutoff_hz=cutoff_hz, order=filter_order
        )

        local = slice(keep_start - start, keep_stop - start)
        orientation[keep_start:keep_stop] = segment_quaternions[local]
        acceleration[keep_start:keep_stop] = specific_force[local]
        angular_velocity[keep_start:keep_stop] = rate[local] * _RAD_TO_DEG
        valid[keep_start:keep_stop] = np.isfinite(rate[local]).all(axis=1)

    orientation[~valid] = np.nan
    acceleration[~valid] = np.nan
    angular_velocity[~valid] = np.nan

    return VirtualImuSignal(
        orientation_wxyz=orientation,
        acceleration_m_per_s2=acceleration,
        angular_velocity_deg_per_s=angular_velocity,
        valid=valid,
    )
