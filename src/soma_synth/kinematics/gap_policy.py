"""Stage C and Stage D of the gap policy: pose-level filling, then rejection.

Once a cluster's pose has been solved wherever three markers were visible, what remains is
a small set of intervals with no pose at all. Two decisions apply to them.

**Fill the pose, not the markers.** Interpolating individual markers lets the constellation
stop being rigid mid-gap, which then feeds a physically impossible body into every later
stage. Interpolating the pose keeps rigidity exact by construction.

**Fill position and rotation with C1-continuous schemes.** The synthesised accelerometer
signal is a second derivative of position and the gyroscope a first derivative of rotation,
so a scheme that is merely continuous in value produces a step in the output. Position uses
a cubic Hermite honouring endpoint velocities; rotation uses a spherical cubic whose control
quaternions carry the measured end tangents, which makes angular velocity continuous at the
joins where plain SLERP is not.

Both schemes read the endpoint rate from the samples either side of the gap, and both must
account for how far away those samples really are. Where gaps cluster, the nearest observed
sample can sit several frames out rather than one, and where a gap reaches the start or end
of a series there is no such sample at all. Reading a multi-frame step as a single frame, or
substituting a rate of zero, puts a large and entirely synthetic transient into exactly the
channels this stage exists to keep clean.

Beyond a caller-supplied length, an interval is not filled at all. It is marked
``UNAVAILABLE`` and stays NaN, per the spec rule that missing data is never given a
plausible value.

Every threshold is a required argument; none is a code constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from soma_synth.kinematics import quaternion as quat

__all__ = [
    "FrameStatus",
    "FilledPose",
    "GapPolicyError",
    "WindowAcceptance",
    "accept_windows",
    "fill_pose_gaps",
    "find_gap_runs",
    "hermite_fill",
    "squad_fill",
]


class FrameStatus(IntEnum):
    """Provenance of a single frame's pose.

    The four values map onto the spec's provenance vocabulary: ``OBSERVED`` and
    ``RECONSTRUCTED`` are both ``source_derived`` (the second is determined by rigid
    geometry, not assumed), ``INTERPOLATED`` is ``estimated``, and ``UNAVAILABLE`` is
    ``unavailable``.
    """

    OBSERVED = 0
    RECONSTRUCTED = 1
    INTERPOLATED = 2
    UNAVAILABLE = 3


class GapPolicyError(ValueError):
    """Raised when a gap cannot be treated under the configured policy."""


@dataclass(frozen=True)
class FilledPose:
    """Pose series after Stage C, with the provenance of every frame."""

    translation: np.ndarray
    quaternion: np.ndarray
    status: np.ndarray

    @property
    def usable(self) -> np.ndarray:
        return self.status != FrameStatus.UNAVAILABLE


@dataclass(frozen=True)
class WindowAcceptance:
    """Which sliding windows survive Stage D, and why the others did not."""

    starts: np.ndarray
    accepted: np.ndarray
    estimated_fraction: np.ndarray
    contains_unavailable: np.ndarray


def find_gap_runs(solved: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` half-open index pairs for each run of unsolved frames."""
    solved = np.asarray(solved, dtype=bool)
    if solved.ndim != 1:
        raise GapPolicyError(f"expected a 1-D mask, got shape {solved.shape}")
    padded = np.concatenate([[False], ~solved, [False]]).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def hermite_fill(values: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Fill ``values[start:stop]`` with a cubic Hermite between the bracketing samples.

    Endpoint derivatives are estimated from the neighbouring valid samples, which is what
    makes the join C1: the filled segment leaves the last observed sample at the velocity
    that sample actually had, and arrives at the next one likewise.

    Where a bracket has no sample beyond it -- the series ends there, or another gap does --
    there is no velocity to honour and the chord across the gap is all the data supports.
    The chord covers the whole span, so it is divided by that span to become the per-frame
    velocity the basis expects. Both ends falling back this way reduces the cubic exactly to
    the straight line between the brackets, which is all the data supports when no endpoint
    velocity is observed.
    """
    if start <= 0 or stop >= values.shape[0]:
        raise GapPolicyError("a gap touching the series boundary has no bracket to fill from")

    left, right = start - 1, stop
    p0, p1 = values[left], values[right]
    span = float(right - left)
    chord_rate = (p1 - p0) / span
    v0 = (
        p0 - values[left - 1]
        if left - 1 >= 0 and np.all(np.isfinite(values[left - 1]))
        else chord_rate
    )
    v1 = (
        values[right + 1] - p1
        if right + 1 < values.shape[0] and np.all(np.isfinite(values[right + 1]))
        else chord_rate
    )

    t = (np.arange(start, stop, dtype=np.float64) - left) / span
    t = t[:, None]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * p0 + h10 * (v0 * span) + h01 * p1 + h11 * (v1 * span)


def _step_log(quaternions: np.ndarray, earlier: int, later: int) -> np.ndarray:
    """Half-angle rotation vector accumulated from frame ``earlier`` to frame ``later``."""
    return quat.log_unit(
        quat.multiply(quat.conjugate(quaternions[earlier][None]), quaternions[later][None])
    )


def _end_tangent(
    quaternions: np.ndarray,
    knot: int,
    *,
    outward: int,
    frame_indices: np.ndarray,
    chord_rate: np.ndarray,
) -> np.ndarray:
    """Per-frame angular tangent at ``knot``, estimated from the side away from the gap.

    A single difference across ``[knot-1, knot]`` is centred half a frame away from the
    knot, which on an accelerating rotation biases the tangent by half the angular
    acceleration times the sample interval -- small, but a systematic step at the join. The
    three-point one-sided estimate removes that first-order bias. Where the series does not
    reach far enough for three points, the two-point estimate is used, which is the most
    the data supports.

    ``quaternions`` holds only the samples the construction gathered, so adjacent slots are
    adjacent *observations* rather than adjacent frames: where gaps cluster, reaching past
    one to find an observed sample makes that step several frames long. ``frame_indices``
    carries the frame each slot came from, and the two steps are divided by their real
    lengths before the estimate is formed. Reading a five-frame step as one frame would
    report five times the true rate and throw the cubic out of the bracket at that speed.

    Where there is no observed sample beyond the knot at all -- the series begins or ends
    there -- no rate is observable and ``chord_rate``, the average rate across the gap, is
    what the data supports. Returning zero instead would assert the body was momentarily
    still, which is a claim about the motion rather than an absence of one.
    """
    near = knot + outward
    far = knot + 2 * outward
    slots = quaternions.shape[0]

    def separation(first: int, second: int) -> float:
        """Frames between two gathered slots, zero where the gather repeated a sample."""
        return abs(float(frame_indices[second] - frame_indices[first]))

    # Both steps are read forward in time whichever side of the gap the knot is on, because
    # the tangent wanted is always the forward angular rate at the knot. A zero separation
    # means the gather ran out on that side and repeated a sample; extrapolating from an
    # empty step would invent a tangent, so the estimate degrades to what remains.
    if not 0 <= near < slots:
        return chord_rate
    inner_frames = separation(knot, near)
    if inner_frames == 0.0:
        return chord_rate
    inner = _step_log(quaternions, min(knot, near), max(knot, near))
    if not 0 <= far < slots:
        return inner / inner_frames
    outer_frames = separation(near, far)
    if outer_frames == 0.0:
        return inner / inner_frames
    outer = _step_log(quaternions, min(near, far), max(near, far))

    # Derivative at the knot of the quadratic through the three samples, for spacings that
    # need not be equal. At one frame each this is the usual ``1.5 * inner - 0.5 * outer``.
    reach = inner_frames + outer_frames
    accumulated = inner + outer
    return inner * (reach / (inner_frames * outer_frames)) - accumulated * (
        inner_frames / (reach * outer_frames)
    )


def _de_casteljau(
    q0: np.ndarray, b1: np.ndarray, b2: np.ndarray, q1: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Evaluate the spherical cubic Bezier through its control quaternions."""
    p01 = quat.slerp(q0, b1, t)
    p12 = quat.slerp(b1, b2, t)
    p23 = quat.slerp(b2, q1, t)
    return quat.slerp(quat.slerp(p01, p12, t), quat.slerp(p12, p23, t), t)


def squad_fill(quaternions: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Fill ``quaternions[start:stop]`` with a spherical cubic between the brackets.

    Shoemake's SQUAD is the spherical cubic whose two inner control quaternions carry the
    end tangents, and its usual closed form for those controls assumes the knots either
    side are one sample away. Across a gap that assumption does not hold: the left bracket's
    outer neighbour sits one frame away while the right bracket sits ``span`` frames away,
    so the closed form would scale both tangents as though the spacing were uniform and
    leave a visible kink at each join.

    The control quaternions are therefore derived from the measured end angular velocities
    instead. ``b1`` advances from the left bracket by a third of the tangent it actually
    had, ``b2`` retreats from the right bracket likewise, and the curve is evaluated by de
    Casteljau. Angular velocity is then continuous at both joins by construction, at any
    spacing -- which is the property the accelerometer and gyroscope channels depend on,
    and the one plain SLERP lacks.
    """
    if start <= 0 or stop >= quaternions.shape[0]:
        raise GapPolicyError("a gap touching the series boundary has no bracket to fill from")

    frames = quaternions.shape[0]
    left, right = start - 1, stop
    finite = np.isfinite(quaternions).all(axis=1)
    if not (finite[left] and finite[right]):
        raise GapPolicyError("both brackets of a gap must be observed")

    def step_back(index: int, count: int) -> int:
        """Nearest observed sample at least ``count`` steps before ``index``.

        The outer samples that set the end tangents can themselves fall inside another gap,
        which is common where two gaps sit close together. Walking to the nearest observed
        sample keeps the tangent estimated from real data instead of propagating a NaN
        through the whole fill.
        """
        found = index
        for _ in range(count):
            candidate = found - 1
            while candidate >= 0 and not finite[candidate]:
                candidate -= 1
            if candidate < 0:
                break
            found = candidate
        return found

    def step_forward(index: int, count: int) -> int:
        found = index
        for _ in range(count):
            candidate = found + 1
            while candidate < frames and not finite[candidate]:
                candidate += 1
            if candidate >= frames:
                break
            found = candidate
        return found

    # Gather only the observed samples the construction needs, in time order, and put them
    # all in one hemisphere before any logarithm is taken.
    indices = [
        step_back(left, 2),
        step_back(left, 1),
        left,
        right,
        step_forward(right, 1),
        step_forward(right, 2),
    ]
    block = quat.canonicalise_sign(quaternions[indices])
    gathered_frames = np.asarray(indices, dtype=np.float64)
    local_left, local_right = 2, 3

    span = float(right - left)
    # The average rate across the gap, used wherever an end has no observed rate of its own.
    chord_rate = _step_log(block, local_left, local_right) / span
    tangent0 = _end_tangent(
        block,
        local_left,
        outward=-1,
        frame_indices=gathered_frames,
        chord_rate=chord_rate,
    )
    tangent1 = _end_tangent(
        block,
        local_right,
        outward=+1,
        frame_indices=gathered_frames,
        chord_rate=chord_rate,
    )
    q0, q1 = block[local_left], block[local_right]
    # Rescale from per-frame to per-unit-of-gap-parameter, then take the Bezier third.
    b1 = quat.multiply(q0[None], quat.exp_pure(tangent0 * span / 3.0))
    b2 = quat.multiply(q1[None], quat.exp_pure(-tangent1 * span / 3.0))

    t = (np.arange(start, stop, dtype=np.float64) - left) / span
    count = t.shape[0]
    return _de_casteljau(
        np.repeat(q0[None], count, 0),
        np.repeat(b1, count, 0),
        np.repeat(b2, count, 0),
        np.repeat(q1[None], count, 0),
        t,
    )


def fill_pose_gaps(
    translation: np.ndarray,
    quaternion: np.ndarray,
    solved: np.ndarray,
    *,
    max_gap_frames: int,
) -> FilledPose:
    """Apply Stage C and Stage D to one cluster's pose series.

    Runs no longer than ``max_gap_frames`` are filled and marked ``INTERPOLATED``. Longer
    runs, and any run touching either end of the series where there is nothing to
    interpolate between, are left NaN and marked ``UNAVAILABLE``.
    """
    if max_gap_frames < 0:
        raise GapPolicyError(f"max_gap_frames must be non-negative, got {max_gap_frames}")
    frames = translation.shape[0]
    if quaternion.shape[0] != frames or solved.shape[0] != frames:
        raise GapPolicyError("translation, quaternion and solved must share a frame count")

    filled_translation = translation.copy()
    filled_quaternion = quaternion.copy()
    status = np.where(
        np.asarray(solved, dtype=bool), FrameStatus.OBSERVED, FrameStatus.UNAVAILABLE
    ).astype(np.int8)

    for start, stop in find_gap_runs(solved):
        length = stop - start
        touches_boundary = start == 0 or stop == frames
        if length > max_gap_frames or touches_boundary:
            continue
        filled_translation[start:stop] = hermite_fill(translation, start, stop)
        filled_quaternion[start:stop] = squad_fill(quaternion, start, stop)
        status[start:stop] = FrameStatus.INTERPOLATED

    return FilledPose(
        translation=filled_translation, quaternion=filled_quaternion, status=status
    )


def accept_windows(
    status: np.ndarray,
    *,
    window_frames: int,
    stride_frames: int,
    max_estimated_fraction: float,
) -> WindowAcceptance:
    """Decide which sliding windows may be used.

    A window is rejected outright if it contains any unavailable frame, and rejected if the
    share of interpolated frames exceeds the configured fraction. Both bounds come from the
    caller; neither has a default.
    """
    if window_frames <= 0 or stride_frames <= 0:
        raise GapPolicyError("window_frames and stride_frames must be positive")
    if not 0.0 <= max_estimated_fraction <= 1.0:
        raise GapPolicyError(
            f"max_estimated_fraction must lie in [0, 1], got {max_estimated_fraction}"
        )

    status = np.asarray(status)
    frames = status.shape[0]
    if frames < window_frames:
        empty_int = np.empty(0, dtype=np.int64)
        empty_bool = np.empty(0, dtype=bool)
        return WindowAcceptance(
            starts=empty_int,
            accepted=empty_bool,
            estimated_fraction=np.empty(0, dtype=np.float64),
            contains_unavailable=empty_bool,
        )

    starts = np.arange(0, frames - window_frames + 1, stride_frames, dtype=np.int64)
    unavailable = (status == FrameStatus.UNAVAILABLE).astype(np.int64)
    interpolated = (status == FrameStatus.INTERPOLATED).astype(np.int64)
    cumulative_unavailable = np.concatenate([[0], np.cumsum(unavailable)])
    cumulative_interpolated = np.concatenate([[0], np.cumsum(interpolated)])

    ends = starts + window_frames
    has_unavailable = (
        cumulative_unavailable[ends] - cumulative_unavailable[starts]
    ) > 0
    fraction = (
        cumulative_interpolated[ends] - cumulative_interpolated[starts]
    ) / float(window_frames)

    accepted = ~has_unavailable & (fraction <= max_estimated_fraction)
    return WindowAcceptance(
        starts=starts,
        accepted=accepted,
        estimated_fraction=fraction,
        contains_unavailable=has_unavailable,
    )
