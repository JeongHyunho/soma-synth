"""Lever arm from the marker plate to the sensor origin.

A marker cluster reports the motion of the plate it is glued to, not of the sensing element
inside the case beneath it. The offset between them matters: at an angular rate of 10 rad/s
a 2 cm lever contributes a 2 m/s^2 centripetal term, which is a fifth of gravity.

GAITEX cannot be asked to supply that offset. It publishes no accelerometer, so the usual
route -- fit the lever that best reconciles synthesised acceleration with measured
acceleration -- has nothing to fit against. Its OpenSim models define no IMU frames either,
because orientation-only inverse kinematics never needs a sensor position. What the archive
*does* determine is the plate: four markers in a fixed, planar, asymmetric arrangement,
reproducible across clusters and trials to a few tenths of a millimetre.

So the lever is split into a part the data determines and a part it cannot:

* the **plate frame** -- origin, in-plane axes, and normal direction -- is estimated per
  cluster per trial from the observed constellation, with the normal's sign resolved from
  an anatomical reference rather than from the measured IMU;
* the **depth** along that normal is a property of the hardware, not of any trial. It is
  supplied by the caller together with its uncertainty, and :func:`centripetal_sensitivity`
  converts that uncertainty into the acceleration error it implies.

Because the plate geometry is a manufactured constant, a per-trial estimate that drifts
from the reference is evidence of a mounting or labelling anomaly rather than of a real
difference. :func:`check_plate_geometry` turns that into a QC signal.

No depth, offset or tolerance has a default here; they come from config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CentripetalSensitivity",
    "LeverArmError",
    "PlateFrame",
    "PlateGeometryQc",
    "centripetal_sensitivity",
    "check_plate_geometry",
    "edge_lengths",
    "estimate_plate_frame",
    "inward_reference_local",
    "lever_arm_local",
]


class LeverArmError(ValueError):
    """Raised when a plate frame or lever arm cannot be defined."""


@dataclass(frozen=True)
class PlateFrame:
    """The marker plate's own frame, expressed in cluster-local coordinates.

    ``normal_local`` points *into the body*, so a positive depth moves from the plate
    towards the sensor and the skin.
    """

    in_plane_axes: np.ndarray
    normal_local: np.ndarray
    planarity_m: float
    edge_lengths_m: np.ndarray

    def rotation_local_from_plate(self) -> np.ndarray:
        """Columns are the plate axes expressed in cluster-local coordinates."""
        return np.column_stack([self.in_plane_axes[0], self.in_plane_axes[1], self.normal_local])


@dataclass(frozen=True)
class PlateGeometryQc:
    """Whether one trial's plate matches the reference geometry."""

    max_edge_deviation_m: float
    planarity_m: float
    edge_within_tolerance: bool
    planarity_within_tolerance: bool

    @property
    def passed(self) -> bool:
        return self.edge_within_tolerance and self.planarity_within_tolerance


@dataclass(frozen=True)
class CentripetalSensitivity:
    """Acceleration error implied by an uncertain lever depth."""

    mean_m_per_s2: float
    p95_m_per_s2: float
    max_m_per_s2: float


def edge_lengths(local_shape: np.ndarray) -> np.ndarray:
    """Sorted pairwise distances of a constellation, the plate's geometric signature."""
    if local_shape.ndim != 2 or local_shape.shape[1] != 3:
        raise LeverArmError(f"expected a (markers, 3) shape, got {local_shape.shape}")
    rows, columns = np.triu_indices(local_shape.shape[0], k=1)
    deltas = local_shape[rows] - local_shape[columns]
    return np.sort(np.linalg.norm(deltas, axis=1))


def inward_reference_local(
    rotation: np.ndarray, inward_world: np.ndarray
) -> np.ndarray:
    """Average a world-frame "towards the body" direction into the cluster-local frame.

    The caller builds ``inward_world`` from anatomy -- for a shank plate, the direction from
    the plate towards the knee-ankle axis; for a foot plate, towards the calcaneus. Only the
    sign of its projection on the plate normal is used, so the vector needs a direction, not
    a magnitude.
    """
    if rotation.ndim != 3 or rotation.shape[1:] != (3, 3):
        raise LeverArmError(f"expected (frames, 3, 3) rotations, got {rotation.shape}")
    if inward_world.shape != rotation.shape[:1] + (3,):
        raise LeverArmError(
            f"expected (frames, 3) world vectors, got {inward_world.shape}"
        )

    usable = np.all(np.isfinite(rotation), axis=(1, 2)) & np.all(
        np.isfinite(inward_world), axis=1
    )
    if not usable.any():
        raise LeverArmError("no frame has both a pose and an inward reference")

    local = np.einsum("fji,fj->fi", rotation[usable], inward_world[usable])
    norms = np.linalg.norm(local, axis=1, keepdims=True)
    positive = norms[:, 0] > 0.0
    if not positive.any():
        raise LeverArmError("the inward reference is degenerate on every usable frame")
    mean = local[positive].mean(axis=0)
    magnitude = np.linalg.norm(mean)
    if magnitude == 0.0:
        raise LeverArmError("the inward reference averages to zero; direction is undefined")
    return mean / magnitude


def estimate_plate_frame(
    local_shape: np.ndarray, *, inward_reference_local: np.ndarray
) -> PlateFrame:
    """Derive the plate frame from the constellation's own geometry.

    The three singular vectors of the centred shape order the constellation's extent. For a
    planar plate the third carries almost none, which is what makes it the normal; its
    singular value is reported as ``planarity_m`` so a caller can check the plate really is
    planar before trusting the direction.
    """
    if local_shape.ndim != 2 or local_shape.shape[1] != 3:
        raise LeverArmError(f"expected a (markers, 3) shape, got {local_shape.shape}")
    if local_shape.shape[0] < 3:
        raise LeverArmError("a plate frame needs at least 3 markers")

    reference = np.asarray(inward_reference_local, dtype=np.float64)
    if reference.shape != (3,):
        raise LeverArmError(f"inward reference must be a 3-vector, got {reference.shape}")
    if not np.isfinite(reference).all() or np.linalg.norm(reference) == 0.0:
        raise LeverArmError("inward reference must be finite and non-zero")

    centred = local_shape - local_shape.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(centred, full_matrices=True)
    axes = vt
    normal = axes[2]

    projection = float(np.dot(normal, reference))
    if projection == 0.0:
        raise LeverArmError(
            "the inward reference lies in the plate plane, so the normal's sign is undecidable"
        )
    if projection < 0.0:
        normal = -normal

    # Keep the in-plane pair right-handed with the resolved normal.
    first = axes[0]
    second = np.cross(normal, first)
    planarity = float(singular_values[2]) if singular_values.shape[0] > 2 else 0.0

    return PlateFrame(
        in_plane_axes=np.stack([first, second]),
        normal_local=normal,
        planarity_m=planarity,
        edge_lengths_m=edge_lengths(local_shape),
    )


def lever_arm_local(
    plate: PlateFrame, *, depth_m: float, in_plane_offset_m: np.ndarray
) -> np.ndarray:
    """Offset from the cluster centroid to the sensor origin, in cluster-local coordinates.

    ``in_plane_offset_m`` is given in plate axes ``(first, second)``; ``depth_m`` runs along
    the inward normal. Both come from configuration -- neither is inferable from GAITEX.
    """
    offset = np.asarray(in_plane_offset_m, dtype=np.float64)
    if offset.shape != (2,):
        raise LeverArmError(f"in_plane_offset_m must be a 2-vector, got {offset.shape}")
    if not np.isfinite(offset).all() or not np.isfinite(depth_m):
        raise LeverArmError("depth and in-plane offset must be finite")
    return (
        offset[0] * plate.in_plane_axes[0]
        + offset[1] * plate.in_plane_axes[1]
        + float(depth_m) * plate.normal_local
    )


def check_plate_geometry(
    plate: PlateFrame,
    *,
    reference_edges_m: np.ndarray,
    edge_tolerance_m: float,
    planarity_tolerance_m: float,
) -> PlateGeometryQc:
    """Compare one trial's plate against the reference geometry.

    The plate is manufactured, so its edges are the same in every trial. A deviation is
    therefore evidence about the trial -- a marker mislabelled, a plate replaced, a
    constellation that is not the plate at all -- and not a measurement to be accepted.
    """
    reference = np.asarray(reference_edges_m, dtype=np.float64)
    if reference.shape != plate.edge_lengths_m.shape:
        raise LeverArmError(
            f"reference has {reference.shape} edges but the plate has "
            f"{plate.edge_lengths_m.shape}"
        )
    if edge_tolerance_m <= 0.0 or planarity_tolerance_m <= 0.0:
        raise LeverArmError("tolerances must be positive")

    deviation = float(np.abs(plate.edge_lengths_m - np.sort(reference)).max())
    return PlateGeometryQc(
        max_edge_deviation_m=deviation,
        planarity_m=plate.planarity_m,
        edge_within_tolerance=deviation <= edge_tolerance_m,
        planarity_within_tolerance=plate.planarity_m <= planarity_tolerance_m,
    )


def centripetal_sensitivity(
    angular_speed_rad_per_s: np.ndarray, *, depth_uncertainty_m: float
) -> CentripetalSensitivity:
    """Acceleration error an uncertain lever depth implies, over an observed rate series.

    The centripetal term is ``|omega|^2 * r``, so an uncertainty in ``r`` scales directly
    with the square of the angular rate the trial actually exhibited. Reporting it against
    the real rates -- rather than a nominal figure -- is what lets the uncertainty be
    carried into the channel's error budget at its actual size.
    """
    if depth_uncertainty_m < 0.0:
        raise LeverArmError("depth uncertainty cannot be negative")
    speed = np.asarray(angular_speed_rad_per_s, dtype=np.float64)
    finite = speed[np.isfinite(speed)]
    if finite.size == 0:
        raise LeverArmError("no finite angular rate samples to evaluate")

    term = finite**2 * float(depth_uncertainty_m)
    return CentripetalSensitivity(
        mean_m_per_s2=float(term.mean()),
        p95_m_per_s2=float(np.percentile(term, 95)),
        max_m_per_s2=float(term.max()),
    )
