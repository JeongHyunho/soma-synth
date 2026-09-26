"""Rigid-body pose from marker constellations.

A marker cluster rigidly attached to a body observes that body's full six-DOF pose. Three
non-collinear markers determine the pose exactly; a fourth over-determines it and turns the
fit residual into a quality signal. GAITEX mounts four coplanar markers on every IMU, so a
frame with any three of them visible still yields an exact pose and the missing marker can
be reconstructed analytically rather than interpolated.

Every tolerance is a required argument. ``DETERMINISTIC_EXECUTION_CONFIG_HOLD`` forbids
fixing threshold values as module constants, so nothing here carries
a default; callers must pass values that came from versioned config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ClusterPose",
    "MIN_MARKERS_FOR_POSE",
    "RigidBodyError",
    "flag_high_residual",
    "mean_local_shape",
    "reconstruct_markers",
    "solve_pose_series",
    "solve_single_pose",
]

MIN_MARKERS_FOR_POSE = 3
"""Three non-collinear points fix a rigid pose. Not a tunable threshold."""


class RigidBodyError(ValueError):
    """Raised when a pose cannot be defined from the given constellation."""


@dataclass(frozen=True)
class ClusterPose:
    """Per-frame rigid pose of a marker cluster.

    ``rotation`` maps local (cluster-frame) coordinates into world:
    ``world = rotation @ local + translation``. Frames without a solvable pose carry NaN in
    every field so that no caller can silently consume a fabricated pose.
    """

    rotation: np.ndarray
    translation: np.ndarray
    residual_rms_m: np.ndarray
    visible_count: np.ndarray

    @property
    def solved(self) -> np.ndarray:
        """Boolean mask of frames whose pose is defined."""
        return np.isfinite(self.residual_rms_m)

    def __len__(self) -> int:
        return int(self.rotation.shape[0])


def _validate_constellation(positions: np.ndarray) -> None:
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise RigidBodyError(
            f"expected positions shaped (frames, markers, 3), got {positions.shape}"
        )
    if positions.shape[1] < MIN_MARKERS_FOR_POSE:
        raise RigidBodyError(
            f"a rigid pose needs at least {MIN_MARKERS_FOR_POSE} markers, "
            f"got {positions.shape[1]}"
        )


def proper_rotation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """Nearest proper rotation to a cross-covariance ``H = local.T @ world``.

    Returns ``V diag(1, 1, d) U^T`` for ``H = U S V^T``, with ``d`` the sign that keeps the
    determinant positive. In the column-vector convention that is the ``R`` satisfying
    ``world = R @ local``; a caller working in row-vector form takes its transpose.

    Accepts one ``(3, 3)`` matrix or a batch ``(n, 3, 3)`` and returns the same shape. The
    determinant correction is what keeps a noisy or near-degenerate constellation from
    producing a reflection, which would silently mirror every downstream acceleration --
    and every site that solves a rotation from a covariance goes through this one place so
    that correction cannot drift between copies.
    """
    u, _, vt = np.linalg.svd(covariance)
    v = np.swapaxes(vt, -1, -2)
    ut = np.swapaxes(u, -1, -2)
    determinant = np.linalg.det(v @ ut)
    sign = np.where(determinant < 0.0, -1.0, 1.0)
    corrected_v = v.copy()
    corrected_v[..., :, 2] *= sign[..., None]
    return corrected_v @ ut


def _kabsch(local: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Rotation taking centred ``local`` onto centred ``world``, guaranteed proper."""
    return proper_rotation_from_covariance(local.T @ world)


def mean_local_shape(positions: np.ndarray) -> np.ndarray:
    """Procrustes mean shape of a constellation, centred on its own centroid.

    Only frames where every marker is visible contribute, because a partial frame cannot
    say where the missing marker sits. The result is the cluster's local geometry and is
    what later frames are matched against.
    """
    _validate_constellation(positions)
    complete = np.all(np.isfinite(positions), axis=(1, 2))
    if not complete.any():
        raise RigidBodyError("no frame has every marker visible; cannot learn a local shape")

    observed = positions[complete]
    reference = observed[0] - observed[0].mean(axis=0)
    accumulated = np.zeros_like(reference)
    for frame in observed:
        centred = frame - frame.mean(axis=0)
        # _kabsch returns R mapping reference into this frame, so the frame is carried back
        # into the reference pose by R transpose. For row-major points that is `centred @ R`,
        # not `centred @ R.T` -- applying R forward here would accumulate the shape twice
        # rotated and quietly distort the learned geometry.
        accumulated += centred @ _kabsch(reference, centred)
    shape = accumulated / observed.shape[0]
    return shape - shape.mean(axis=0)


def solve_single_pose(
    local_shape: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pose of one frame from the visible subset of a constellation.

    ``observed`` may contain NaN for occluded markers. Returns ``(rotation, translation,
    residual_rms)``; the residual is ``0.0`` when exactly three markers are visible, since
    three points determine the pose with no redundancy left to disagree.
    """
    visible = np.all(np.isfinite(observed), axis=1)
    count = int(visible.sum())
    if count < MIN_MARKERS_FOR_POSE:
        nan_rotation = np.full((3, 3), np.nan)
        return nan_rotation, np.full(3, np.nan), float("nan")

    local = local_shape[visible]
    world = observed[visible]
    local_centre = local.mean(axis=0)
    world_centre = world.mean(axis=0)
    rotation = _kabsch(local - local_centre, world - world_centre)
    translation = world_centre - rotation @ local_centre

    fitted = (local @ rotation.T) + translation
    residual = float(np.sqrt(np.mean(np.sum((world - fitted) ** 2, axis=1))))
    return rotation, translation, residual


def solve_pose_series(local_shape: np.ndarray, positions: np.ndarray) -> ClusterPose:
    """Solve the pose of every frame that has at least three visible markers."""
    _validate_constellation(positions)
    if local_shape.shape != positions.shape[1:]:
        raise RigidBodyError(
            f"local shape {local_shape.shape} does not match constellation "
            f"{positions.shape[1:]}"
        )

    frames = positions.shape[0]
    rotation = np.full((frames, 3, 3), np.nan)
    translation = np.full((frames, 3), np.nan)
    residual = np.full(frames, np.nan)
    visible = np.all(np.isfinite(positions), axis=2)
    visible_count = visible.sum(axis=1).astype(np.int64)

    # Frames sharing a visibility pattern share a marker subset, so they can go through one
    # batched SVD. In practice almost every frame has the full constellation, which makes
    # this a single batch and keeps a corpus-scale run tractable -- a per-frame Python loop
    # over 73 trials and five sites would be some nine million iterations.
    solvable = visible_count >= MIN_MARKERS_FOR_POSE
    if solvable.any():
        packed = np.packbits(visible, axis=1).view(np.uint8)
        keys = np.array(
            [bytes(row) for row in packed], dtype=object
        )
        for key in np.unique(keys[solvable]):
            group = np.flatnonzero(solvable & (keys == key))
            subset = visible[group[0]]
            local = local_shape[subset]
            world = positions[np.ix_(group, subset)]

            local_centre = local.mean(axis=0)
            world_centre = world.mean(axis=1)
            centred_local = local - local_centre
            centred_world = world - world_centre[:, None, :]

            covariance = np.einsum("mi,fmj->fij", centred_local, centred_world)
            group_rotation = proper_rotation_from_covariance(covariance)

            fitted = np.einsum("fij,mj->fmi", group_rotation, centred_local)
            group_residual = np.sqrt(
                np.mean(np.sum((centred_world - fitted) ** 2, axis=2), axis=1)
            )

            rotation[group] = group_rotation
            translation[group] = world_centre - np.einsum(
                "fij,j->fi", group_rotation, local_centre
            )
            residual[group] = group_residual

    return ClusterPose(
        rotation=rotation,
        translation=translation,
        residual_rms_m=residual,
        visible_count=visible_count,
    )


def flag_high_residual(pose: ClusterPose, *, threshold_m: float) -> np.ndarray:
    """Mark solved frames whose fit residual exceeds a caller-supplied bound.

    The frames are flagged, not removed. A large residual says the constellation stopped
    behaving rigidly, which is a provenance fact worth carrying rather than grounds for
    silently deleting the frame -- deletion would leave a gap indistinguishable from an
    occlusion and would misreport how much data the source actually supplied.
    """
    if threshold_m <= 0.0:
        raise RigidBodyError(f"threshold_m must be positive, got {threshold_m}")
    return pose.solved & (pose.residual_rms_m > threshold_m)


def reconstruct_markers(
    local_shape: np.ndarray, pose: ClusterPose, positions: np.ndarray
) -> np.ndarray:
    """Fill occluded markers from the solved pose.

    This is reconstruction, not interpolation: where the pose is known, an occluded
    marker's position follows from the cluster's rigid geometry alone and involves no
    assumption about how the body moved between frames. Frames without a pose are left NaN.
    """
    _validate_constellation(positions)
    filled = positions.copy()
    solved = pose.solved
    if not solved.any():
        return filled

    predicted = np.einsum("fij,mj->fmi", pose.rotation[solved], local_shape)
    predicted += pose.translation[solved][:, None, :]

    missing = ~np.isfinite(positions)
    frame_indices = np.flatnonzero(solved)
    target = missing[solved]
    filled_block = filled[solved]
    filled_block[target] = predicted[target]
    filled[frame_indices] = filled_block
    return filled
