"""Marker stream conditioning: occlusion sentinels, spikes, and rigid-geometry violations.

GAITEX encodes an occluded marker as an exact ``(0, 0, 0)`` triplet rather than as a blank
or a NaN. Differentiating across that sentinel would read as the marker teleporting to the
laboratory origin and back, producing accelerations of order ``1e4 m/s^2``. Converting it
to NaN is therefore the first thing that must happen to a marker stream, and it is kept
separate from reading so the conversion is explicit and testable on its own.

Two further defects do *not* announce themselves as zeros and so survive that conversion:

* a momentary tracking spike, caught by an upper bound on marker speed;
* a swapped or misassigned label, caught by the constellation's own rigid geometry -- the
  distances between markers on one plate are constants, so a marker whose distances to its
  neighbours no longer match the learned reference is the one at fault.

Every threshold is a required argument. ``DETERMINISTIC_EXECUTION_CONFIG_HOLD`` forbids
fixing tolerance values as module constants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MarkerConditioning",
    "apply_occlusion_sentinel",
    "condition_cluster",
    "flag_rigid_violations",
    "flag_speed_outliers",
    "reference_pair_distances",
]


@dataclass(frozen=True)
class MarkerConditioning:
    """Conditioned positions plus the mask that says why each sample was withdrawn.

    The masks are kept apart rather than merged so that provenance can distinguish an
    occlusion the source declared from a defect this code inferred.
    """

    positions_m: np.ndarray
    sentinel_mask: np.ndarray
    speed_mask: np.ndarray
    rigid_mask: np.ndarray

    @property
    def withdrawn_mask(self) -> np.ndarray:
        """Any sample withdrawn, for whatever reason. Shape ``(frames, markers)``."""
        return self.sentinel_mask | self.speed_mask | self.rigid_mask

    @property
    def visible_count(self) -> np.ndarray:
        return np.sum(~self.withdrawn_mask, axis=1).astype(np.int64)


def apply_occlusion_sentinel(positions_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert exact ``(0, 0, 0)`` triplets to NaN.

    All three components must be exactly zero. A marker that merely passes near the origin
    on one or two axes is real data and is left alone; requiring the full triplet is what
    makes the test safe, since the probability of three independent float coordinates being
    bit-exactly zero by measurement is nil.
    """
    if positions_m.ndim != 3 or positions_m.shape[2] != 3:
        raise ValueError(
            f"expected positions shaped (frames, markers, 3), got {positions_m.shape}"
        )
    sentinel = np.all(positions_m == 0.0, axis=2)
    cleaned = positions_m.copy()
    cleaned[sentinel] = np.nan
    return cleaned, sentinel


def flag_speed_outliers(
    positions_m: np.ndarray, *, dt_s: float, max_speed_m_per_s: float
) -> np.ndarray:
    """Flag samples whose implied speed exceeds a physiological bound.

    Speed is taken against both neighbours, so an isolated one-frame spike is flagged while
    the sound samples on either side of it are not. A sample adjacent to a gap has only one
    finite neighbour and is judged on that side alone.
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    if max_speed_m_per_s <= 0.0:
        raise ValueError(f"max_speed_m_per_s must be positive, got {max_speed_m_per_s}")

    frames = positions_m.shape[0]
    flagged = np.zeros(positions_m.shape[:2], dtype=bool)
    if frames < 2:
        return flagged

    step = np.linalg.norm(np.diff(positions_m, axis=0), axis=2) / dt_s
    exceeds = step > max_speed_m_per_s

    backward = np.zeros_like(flagged)
    forward = np.zeros_like(flagged)
    backward[1:] = exceeds
    forward[:-1] = exceeds

    has_backward = np.zeros_like(flagged)
    has_forward = np.zeros_like(flagged)
    finite = np.all(np.isfinite(positions_m), axis=2)
    has_backward[1:] = finite[:-1]
    has_forward[:-1] = finite[1:]

    # An interior sample must offend on both sides; an edge sample on the side it has.
    both_sides = has_backward & has_forward
    flagged[both_sides] = (backward & forward)[both_sides]
    one_side = ~both_sides
    flagged[one_side] = (backward | forward)[one_side]
    flagged &= finite
    return flagged


def reference_pair_distances(positions_m: np.ndarray) -> np.ndarray:
    """Learn the constellation's pairwise distances from frames where all markers show.

    Returns a symmetric ``(markers, markers)`` matrix of medians. The median rather than the
    mean because a handful of contaminated frames should not move the reference the checks
    are then measured against.
    """
    if positions_m.ndim != 3 or positions_m.shape[2] != 3:
        raise ValueError(
            f"expected positions shaped (frames, markers, 3), got {positions_m.shape}"
        )
    complete = np.all(np.isfinite(positions_m), axis=(1, 2))
    if not complete.any():
        raise ValueError("no frame has every marker visible; cannot learn pair distances")

    observed = positions_m[complete]
    deltas = observed[:, :, None, :] - observed[:, None, :, :]
    distances = np.linalg.norm(deltas, axis=3)
    return np.median(distances, axis=0)


def flag_rigid_violations(
    positions_m: np.ndarray, *, reference_distances: np.ndarray, tolerance_m: float
) -> np.ndarray:
    """Flag markers whose distances to their neighbours contradict the rigid reference.

    A marker is flagged when it disagrees with the majority of its visible neighbours,
    which attributes the fault to the marker that moved rather than condemning the whole
    cluster. On an asymmetric constellation -- GAITEX's plate has six distinct edge lengths
    -- this also catches a swapped pair, because a swap changes each of the two markers'
    distance signatures.
    """
    if tolerance_m <= 0.0:
        raise ValueError(f"tolerance_m must be positive, got {tolerance_m}")
    markers = positions_m.shape[1]
    if reference_distances.shape != (markers, markers):
        raise ValueError(
            f"reference_distances must be ({markers}, {markers}), "
            f"got {reference_distances.shape}"
        )

    deltas = positions_m[:, :, None, :] - positions_m[:, None, :, :]
    distances = np.linalg.norm(deltas, axis=3)
    deviation = np.abs(distances - reference_distances[None, :, :])

    off_diagonal = ~np.eye(markers, dtype=bool)
    comparable = np.isfinite(deviation) & off_diagonal[None, :, :]
    violating = comparable & (deviation > tolerance_m)

    neighbours = comparable.sum(axis=2)
    offences = violating.sum(axis=2)
    # Strict majority, and never flag a marker with nothing to compare against.
    return (neighbours > 0) & (offences * 2 > neighbours)


def condition_cluster(
    positions_m: np.ndarray,
    *,
    dt_s: float,
    max_speed_m_per_s: float,
    rigid_tolerance_m: float,
) -> MarkerConditioning:
    """Run the full Stage A conditioning over one cluster, in order.

    The order matters: sentinels become NaN first so that neither the speed test nor the
    rigid test ever measures against the laboratory origin, and the rigid reference is
    learned only after both earlier defects have been withdrawn.
    """
    cleaned, sentinel = apply_occlusion_sentinel(positions_m)

    speed = flag_speed_outliers(
        cleaned, dt_s=dt_s, max_speed_m_per_s=max_speed_m_per_s
    )
    cleaned = cleaned.copy()
    cleaned[speed] = np.nan

    reference = reference_pair_distances(cleaned)
    rigid = flag_rigid_violations(
        cleaned, reference_distances=reference, tolerance_m=rigid_tolerance_m
    )
    cleaned = cleaned.copy()
    cleaned[rigid] = np.nan

    return MarkerConditioning(
        positions_m=cleaned,
        sentinel_mask=sentinel,
        speed_mask=speed,
        rigid_mask=rigid,
    )
