"""Minimal unit-quaternion algebra in ``(w, x, y, z)`` order.

The spec fixes the component order as ``(w, x, y, z)`` while SciPy's ``Rotation`` uses
``(x, y, z, w)``. Rather than convert at every call site -- which is where sign and order
mistakes hide -- the few operations SQUAD needs are written directly against the spec's
order here.

``q`` and ``-q`` denote the same rotation. Interpolating across a sign flip travels the
long way round the sphere, so :func:`canonicalise_sign` is applied before any interpolation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "body_angular_velocity",
    "canonicalise_sign",
    "conjugate",
    "exp_pure",
    "log_unit",
    "multiply",
    "normalise",
    "slerp",
]


def normalise(q: np.ndarray) -> np.ndarray:
    """Scale quaternions to unit norm along the last axis."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("cannot normalise a zero quaternion")
    return q / norm


def conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate, which is the inverse for unit quaternions."""
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., 1:] *= -1.0
    return out


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, broadcasting over leading axes."""
    aw, ax, ay, az = (a[..., i] for i in range(4))
    bw, bx, by, bz = (b[..., i] for i in range(4))
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def canonicalise_sign(q: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions lie in the same hemisphere.

    Without this a source that legitimately emits ``-q`` on one frame makes any
    interpolation swing through the far side of the sphere, which would appear downstream as
    an enormous spurious angular velocity.
    """
    out = np.asarray(q, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] != 4:
        raise ValueError(f"expected a (frames, 4) quaternion series, got {out.shape}")
    for index in range(1, out.shape[0]):
        if np.dot(out[index - 1], out[index]) < 0.0:
            out[index] = -out[index]
    return out


def log_unit(q: np.ndarray) -> np.ndarray:
    """Logarithm of a unit quaternion, returned as a pure quaternion ``(0, v)``.

    ``v`` is the rotation axis scaled by half the rotation angle.
    """
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    vector = q[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1)
    scalar = np.clip(q[..., 0], -1.0, 1.0)
    angle = np.arctan2(vector_norm, scalar)

    # As the rotation vanishes the axis becomes undefined; the limit of angle/sin(angle) is
    # 1, so scaling by the angle directly gives the correct zero vector without a 0/0.
    scale = np.where(vector_norm > 1e-12, angle / np.where(vector_norm > 1e-12, vector_norm, 1.0), 0.0)
    out = np.zeros_like(q)
    out[..., 1:] = vector * scale[..., None]
    return out


def exp_pure(v: np.ndarray) -> np.ndarray:
    """Exponential of a pure quaternion ``(0, v)``, giving a unit quaternion."""
    v = np.atleast_2d(np.asarray(v, dtype=np.float64))
    vector = v[..., 1:]
    angle = np.linalg.norm(vector, axis=-1)
    out = np.zeros_like(v)
    out[..., 0] = np.cos(angle)
    scale = np.where(angle > 1e-12, np.sin(angle) / np.where(angle > 1e-12, angle, 1.0), 1.0)
    out[..., 1:] = vector * scale[..., None]
    return out


def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation between two quaternion series at fractions ``t``."""
    q0 = np.atleast_2d(np.asarray(q0, dtype=np.float64))
    q1 = np.atleast_2d(np.asarray(q1, dtype=np.float64))
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))[:, None]

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)

    # As theta goes to zero, sin((1-t)theta)/sin(theta) tends to (1-t) and sin(t.theta)/
    # sin(theta) tends to t. Those limits are evaluated directly for near-parallel inputs,
    # where the ratio would otherwise be 0/0.
    close = dot > 1.0 - 1e-9
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    safe_sin = np.where(close, 1.0, sin_theta)
    weight0 = np.where(close, 1.0 - t, np.sin((1.0 - t) * theta) / safe_sin)
    weight1 = np.where(close, t, np.sin(t * theta) / safe_sin)
    return normalise(weight0 * q0 + weight1 * q1)


def body_angular_velocity(orientation_wxyz: np.ndarray, dt_s: float) -> np.ndarray:
    """Body-frame angular velocity of a ``(frames, 4)`` series by the central log map.

    ``log(q(t-dt)^-1 q(t+dt)) / (2 dt)`` is exact for a constant rate, where differencing
    rotation matrices carries an ``O(dt^2)`` error growing with the cube of the rate. The
    first and last frame have no two-sided neighbour and are returned as NaN.

    ``q`` and ``-q`` are the same rotation but their logarithms differ by a half turn. The
    relative quaternion is taken in the positive-scalar hemisphere, which selects the
    shorter arc -- the physical one over a two-sample interval -- and makes the result
    independent of whatever sign convention the caller's series happened to carry.
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    frames = orientation_wxyz.shape[0]
    rates = np.full((frames, 3), np.nan)
    if frames < 3:
        return rates
    relative = multiply(conjugate(orientation_wxyz[:-2]), orientation_wxyz[2:])
    relative = np.where(relative[:, :1] < 0.0, -relative, relative)
    # log_unit returns the half-angle vector, so twice it is the rotation vector.
    rates[1:-1] = 2.0 * log_unit(relative)[:, 1:] / (2.0 * dt_s)
    return rates
