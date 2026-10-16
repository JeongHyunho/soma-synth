"""Newton-Euler inverse dynamics on a tree of rigid bodies driven by measured kinematics.

Each body ``b`` hangs from its joint ``b`` (SMPL convention: body index = joint index) and moves
with that joint's global rotation. For every joint the wrench the parent exerts on the child
subtree is the subtree's inertial wrench minus the external and gravity loads on it:

    F_j = sum_{b in sub(j)} m_b (a_b - g) - sum F_ext
    M_j = sum_{b in sub(j)} [(c_b - p_j) x m_b (a_b - g) + I_b alpha_b + w_b x I_b w_b]
          - sum [(r_ext - p_j) x F_ext + T_ext]

This is the internal intersegmental (parent-on-child) wrench at the joint centre ``p_j``, in the
world frame. Summed over the whole tree it is the root residual: the wrench that the measured
loads fail to explain, which is zero for consistent data.

The frame must be inertial: for treadmill data, the laboratory frame where the plates measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

__all__ = [
    "BodyKinematics",
    "ExternalLoad",
    "JointWrenches",
    "WholeBody",
    "body_kinematics",
    "joint_wrenches",
    "lowpass",
    "project_so3",
    "rotation_step_deg",
    "smooth_rotations",
    "whole_body",
    "world_angular_velocity",
]


def lowpass(x: np.ndarray, cutoff_hz: float, fs_hz: float, order: int) -> np.ndarray:
    """Zero-lag Butterworth low-pass along axis 0 (``filtfilt`` doubles the effective order)."""
    b, a = butter(order, cutoff_hz / (0.5 * fs_hz))
    return filtfilt(b, a, np.asarray(x, np.float64), axis=0)


def project_so3(R: np.ndarray) -> np.ndarray:
    """Nearest rotation (polar/SVD projection) of each 3x3 matrix; reflections are refused."""
    U, _, Vt = np.linalg.svd(R)
    d = np.linalg.det(U @ Vt)
    if np.any(d < 0):
        raise ValueError("a matrix projects to a reflection, not a rotation")
    return U @ Vt


def smooth_rotations(R: np.ndarray, cutoff_hz: float, fs_hz: float, order: int) -> np.ndarray:
    """Low-pass a rotation sequence (T,...,3,3) through sign-continuous unit quaternions.

    Filtering matrix elements can average a fast turn into a near-singular matrix that projects
    to a reflection; a quaternion path made sign-continuous and renormalised after the filter
    always returns a rotation.
    """
    R = np.asarray(R, np.float64)
    shape = R.shape
    q = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat().reshape(*shape[:-2], 4)
    flip = np.sum(q[1:] * q[:-1], axis=-1) < 0
    sign = np.where(np.cumsum(flip, axis=0) % 2 == 1, -1.0, 1.0)
    q[1:] *= sign[..., None]
    q = lowpass(q, cutoff_hz, fs_hz, order)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(shape)


def rotation_step_deg(R: np.ndarray) -> np.ndarray:
    """Angle (deg) between consecutive rotations, (T,...): 0 for the first frame."""
    R = np.asarray(R, np.float64)
    rel = np.swapaxes(R[:-1], -1, -2) @ R[1:]
    ang = np.degrees(Rotation.from_matrix(rel.reshape(-1, 3, 3)).magnitude()).reshape(rel.shape[:-2])
    return np.concatenate([np.zeros((1, *ang.shape[1:])), ang], 0)


def world_angular_velocity(R: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity (rad/s) of a rotation sequence (T,...,3,3), central difference."""
    R = np.asarray(R, np.float64)
    T = R.shape[0]
    lead = R.shape[1:-2]
    w = np.zeros((T, *lead, 3))
    if T < 2:
        return w

    def _log(A):
        return Rotation.from_matrix(A.reshape(-1, 3, 3)).as_rotvec().reshape(*A.shape[:-2], 3)

    if T >= 3:
        w[1:-1] = _log(R[2:] @ np.swapaxes(R[:-2], -1, -2)) / (2.0 * dt)
    w[0] = _log(R[1] @ np.swapaxes(R[0], -1, -2)) / dt
    w[-1] = _log(R[-1] @ np.swapaxes(R[-2], -1, -2)) / dt
    return w


@dataclass(frozen=True)
class BodyKinematics:
    """World-frame kinematics of every body, (T,B,...)."""

    rotation: np.ndarray          # (T,B,3,3) global rotation (rest axes -> world)
    joint_position: np.ndarray    # (T,B,3) joint centre the body hangs from
    com: np.ndarray               # (T,B,3)
    velocity: np.ndarray          # (T,B,3) of the COM
    acceleration: np.ndarray      # (T,B,3) of the COM
    omega: np.ndarray             # (T,B,3) rad/s
    alpha: np.ndarray             # (T,B,3) rad/s^2


def body_kinematics(rotation, joint_position, rest_joint, rest_com, dt) -> BodyKinematics:
    """COM trajectories and derivatives from (already smoothed) rotations and joint centres."""
    R = np.asarray(rotation, np.float64)
    P = np.asarray(joint_position, np.float64)
    off = np.asarray(rest_com, np.float64) - np.asarray(rest_joint, np.float64)     # (B,3)
    com = P + np.einsum("tbij,bj->tbi", R, off)
    vel = np.gradient(com, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    omega = world_angular_velocity(R, dt)
    alpha = np.gradient(omega, dt, axis=0)
    return BodyKinematics(R, P, com, vel, acc, omega, alpha)


@dataclass(frozen=True)
class ExternalLoad:
    """A measured wrench on the body, given at the world origin, applied to one body per frame."""

    force: np.ndarray             # (T,3) N, acting on the body
    moment_at_origin: np.ndarray  # (T,3) N m about the world origin
    body: np.ndarray              # (T,) int, receiving body; -1 where the load is absent


@dataclass(frozen=True)
class JointWrenches:
    force: np.ndarray             # (T,B,3) parent-on-child force at each joint; [:,0] is the root residual
    moment: np.ndarray            # (T,B,3) parent-on-child moment about the joint centre


def _inertial(mass, inertia_rest, kin: BodyKinematics, gravity):
    g = np.asarray(gravity, np.float64)
    f = mass[None, :, None] * (kin.acceleration - g)
    Iw = np.einsum("tbij,bjk,tblk->tbil", kin.rotation, inertia_rest, kin.rotation)
    Iw_w = np.einsum("tbij,tbj->tbi", Iw, kin.omega)
    n = np.einsum("tbij,tbj->tbi", Iw, kin.alpha) + np.cross(kin.omega, Iw_w)
    return f, n, Iw_w


def joint_wrenches(parents, mass, inertia_rest, kin: BodyKinematics,
                   loads: list[ExternalLoad], gravity) -> JointWrenches:
    """Parent-on-child wrench at every joint (see the module docstring)."""
    parents = np.asarray(parents, np.int64)
    mass = np.asarray(mass, np.float64)
    T, B = kin.com.shape[:2]
    f, n, _ = _inertial(mass, np.asarray(inertia_rest, np.float64), kin, gravity)
    Fsub = f.copy()
    Nsub = np.cross(kin.com, f) + n                  # about the world origin
    for ld in loads:
        b = np.asarray(ld.body, np.int64)
        on = b >= 0
        t = np.flatnonzero(on)
        np.subtract.at(Fsub, (t, b[on]), ld.force[on])
        np.subtract.at(Nsub, (t, b[on]), ld.moment_at_origin[on])
    for j in range(B - 1, 0, -1):                    # children have larger indices than parents
        p = parents[j]
        Fsub[:, p] += Fsub[:, j]
        Nsub[:, p] += Nsub[:, j]
    moment = Nsub - np.cross(kin.joint_position, Fsub)
    return JointWrenches(force=Fsub, moment=moment)


@dataclass(frozen=True)
class WholeBody:
    com: np.ndarray               # (T,3)
    com_acceleration: np.ndarray  # (T,3)
    angular_momentum: np.ndarray  # (T,3) about the whole-body COM


def whole_body(mass, inertia_rest, kin: BodyKinematics) -> WholeBody:
    mass = np.asarray(mass, np.float64)
    M = mass.sum()
    w = mass[None, :, None] / M
    com = (w * kin.com).sum(1)
    vcom = (w * kin.velocity).sum(1)
    acom = (w * kin.acceleration).sum(1)
    _, _, Iw_w = _inertial(mass, np.asarray(inertia_rest, np.float64), kin, np.zeros(3))
    r = kin.com - com[:, None]
    v = kin.velocity - vcom[:, None]
    H = (np.cross(r, mass[None, :, None] * v) + Iw_w).sum(1)
    return WholeBody(com=com, com_acceleration=acom, angular_momentum=H)
