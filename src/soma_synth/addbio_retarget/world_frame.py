"""Bring the AddBiomechanics world into the Z-up world the rest of the corpus uses.

PRISM and AMASS artifacts are right-handed, Z-up, gravity along -Z. AddBiomechanics models
declare `gravity 0 -9.8067 0`, so everything with a direction -- joint centres, root
translation, global orientation, ground reaction force, centre of pressure, force plate
corners -- has to be turned by the same fixed rotation, or the artifact will hold a body
walking up a wall and forces pointing sideways.

The rotation is built from the gravity the model declares rather than from a Y-up assumption,
because that assumption is exactly the sort that holds for eighteen sampled subjects and then
does not. Gravity fixes two degrees of freedom; the spin about the vertical is not recoverable
from it, is not comparable between capture labs, and is pinned to zero. The resulting constant
is written into the artifact so the transform can be undone.

Magnitude needs no reconciling: the models declare 9.80665, the same number the corpus uses,
uniformly across every study directory in the cohort. Only the axis differs, so the operation
here is a rotation and nothing else -- no rescaling, ever.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SOMA_GRAVITY_ZUP",
    "SOURCE_GRAVITY_YUP",
    "WorldFrameTransform",
    "rotation_from_gravity",
    "transform_for_gravity",
]

#: Gravity as the AddBiomechanics models declare it -- Y-up. Measured uniform across every
#: study directory in the cohort.
SOURCE_GRAVITY_YUP = (0.0, -9.80665, 0.0)

#: Gravity as PRISM and AMASS artifacts carry it -- Z-up, right-handed.
SOMA_GRAVITY_ZUP = (0.0, 0.0, -9.80665)

_TARGET_DIRECTION = np.array([0.0, 0.0, -1.0])
_FRAME_CONVENTION = "addbio_world_converted_Zup"


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_from_gravity(gravity) -> np.ndarray:
    """The smallest rotation carrying `gravity` onto -Z.

    Smallest, rather than any: it introduces no spin about the vertical, which keeps the
    constant reproducible from the model alone.
    """
    source = np.asarray(gravity, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(source))
    if norm == 0.0:
        raise ValueError("gravity has no direction; cannot orient the world frame")
    unit = source / norm

    axis = np.cross(unit, _TARGET_DIRECTION)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(unit, _TARGET_DIRECTION))
    if sine == 0.0:
        if cosine > 0.0:
            return np.eye(3)
        # gravity points straight up: turn a half circle about any perpendicular axis
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

    cross = _skew(axis)
    return np.eye(3) + cross + cross @ cross * ((1.0 - cosine) / sine**2)


def _quaternion_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = np.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k]) * 2.0
        components = [0.0, 0.0, 0.0]
        components[i] = 0.25 * scale
        components[j] = (rotation[j, i] + rotation[i, j]) / scale
        components[k] = (rotation[k, i] + rotation[i, k]) / scale
        w = (rotation[k, j] - rotation[j, k]) / scale
        x, y, z = components
    if w < 0.0:                       # one hemisphere, so the recorded constant is unique
        w, x, y, z = -w, -x, -y, -z
    return (float(w), float(x), float(y), float(z))


@dataclass(frozen=True)
class WorldFrameTransform:
    """The fixed rotation from one source's world into the corpus world, plus its provenance."""

    rotation: np.ndarray
    source_gravity: tuple[float, float, float]
    frame_convention: str = _FRAME_CONVENTION

    @property
    def quaternion_wxyz(self) -> tuple[float, float, float, float]:
        return _quaternion_wxyz(self.rotation)

    def apply(self, vectors) -> np.ndarray:
        """Rotate any array of 3-vectors, whatever its leading shape."""
        array = np.asarray(vectors, dtype=np.float64)
        if array.shape[-1] != 3:
            raise ValueError(f"expected trailing axis of 3, got {array.shape}")
        return array @ self.rotation.T

    def invert(self, vectors) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float64)
        if array.shape[-1] != 3:
            raise ValueError(f"expected trailing axis of 3, got {array.shape}")
        return array @ self.rotation


def transform_for_gravity(gravity) -> WorldFrameTransform:
    source = np.asarray(gravity, dtype=np.float64).reshape(3)
    return WorldFrameTransform(
        rotation=rotation_from_gravity(source),
        source_gravity=(float(source[0]), float(source[1]), float(source[2])),
    )
