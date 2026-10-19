"""Milestone-2 (KAIST EXO Lab) kinetics: SMPL fits on a split-belt instrumented treadmill.

The inverse dynamics runs in the treadmill laboratory frame, the inertial frame the plates
measured in (``trans_treadmill`` / ``global_orient_treadmill`` of the fits). The fits also carry
an overground reconstruction (``trans`` / ``global_orient``): the same body pose moved by a rigid
transform per frame, which adds the belt's forward travel and turns the subject round. The
kinematics of a bundle use that reconstruction, so :func:`to_frame` carries the kinetics across
with the same transform: vectors are rotated, the centre of pressure is moved with the foot.

Every number comes from the source's configuration (``configs/kinetics/milestone2_inverse_dynamics_v1.yaml``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from . import bsip, forceplate, inverse_dynamics as idyn, jcs, smpl_body

__all__ = ["TakeKinetics", "compute", "lab_to_overground", "to_frame"]

_FEET = {"left": (7, 10), "right": (8, 11)}


@dataclass(frozen=True)
class TakeKinetics:
    """One take's kinetics in the laboratory frame (vectors) and the JCS (moments, powers)."""

    joint_names: tuple[str, ...]
    grf: np.ndarray                 # (T,2,3) N, zero without contact
    cop: np.ndarray                 # (T,2,3) m on the plate surface, NaN without contact
    free_moment: np.ndarray         # (T,2) N m, NaN without contact
    foot_contact: np.ndarray        # (T,2) bool
    com_acceleration: np.ndarray    # (T,3)
    angular_momentum: np.ndarray    # (T,3)
    joint_torque: np.ndarray        # (T,17) primary JCS component
    joint_power: np.ndarray         # (T,17) primary component power
    joint_moments_jcs: np.ndarray   # (T,17,3)
    joint_power_total: np.ndarray   # (T,17)
    primary_component: np.ndarray   # (17,)
    residual_force: np.ndarray      # (T,3)
    residual_moment: np.ndarray     # (T,3)
    grf_valid: np.ndarray           # (T,)
    kinematics_valid: np.ndarray    # (T,) False around SMPL fit glitches
    cop_to_foot_distance: np.ndarray  # (T,2) m, NaN without contact
    max_rotation_step_deg: np.ndarray  # (T,)
    body_mass_force_plate_kg: float
    segment_mass_kg: np.ndarray     # (24,)
    mesh_volume_m3: float

    @property
    def valid(self) -> np.ndarray:
        return self.grf_valid & self.kinematics_valid


def compute(fit, plates, model: smpl_body.SmplModel, cfg: dict) -> TakeKinetics:
    """Kinetics of one take. ``fit`` and ``plates`` map the source's npz keys to arrays."""
    fs = float(cfg["sampling_rate_hz"])
    dt = 1.0 / fs
    for name, src in (("fit", fit), ("plates", plates)):
        if abs(float(src["fps"]) - fs) > 1e-6:
            raise ValueError(f"{name}: sampling rate {float(src['fps'])} Hz, expected {fs} Hz")
    T = np.asarray(fit["trans"]).shape[0]
    if np.asarray(plates["grf_left"]).shape[0] != T or int(plates["trim_start_idx"]) != int(fit["trim_start_idx"]):
        raise ValueError("plates and SMPL fit are not frame-aligned")
    f = cfg["filter"]
    cut, order = float(f["cutoff_hz"]), int(f["order"])
    g = np.asarray(cfg["gravity_m_s2"], np.float64)
    pl, q = cfg["plates"], cfg["quality"]

    v, J = smpl_body.shaped(model, np.asarray(fit["betas"], np.float64))
    F_raw = {s: np.asarray(plates[pl[s]["force"]], np.float64) for s in _FEET}
    mass = float(-(F_raw["left"][:, 2] + F_raw["right"][:, 2]).mean() / -g[2])
    seg = bsip.segment_inertia_from_mesh(v, model.faces, model.weights, mass,
                                         float(cfg["segments"]["voxel_size_m"]))

    R, P = smpl_body.fk(model, J, fit["global_orient_treadmill"], fit["body_pose"], fit["trans_treadmill"])
    step = idyn.rotation_step_deg(R).max(axis=1)
    glitch = step > float(q["max_rotation_step_deg"])
    pad = int(q["glitch_pad_frames"])
    kin_valid = ~np.convolve(glitch.astype(float), np.ones(2 * pad + 1), mode="same").astype(bool)
    Rf = idyn.smooth_rotations(R, cut, fs, order)
    Pf = idyn.lowpass(P, cut, fs, order)
    kin = idyn.body_kinematics(Rf, Pf, J, seg.com, dt)

    loads, feet = [], {}
    for side, (ank, toe) in _FEET.items():
        M_raw = np.asarray(plates[pl[side]["moment"]], np.float64) * float(pl["moment_scale_to_n_m"])
        load = forceplate.plate_load(idyn.lowpass(F_raw[side], cut, fs, order),
                                     idyn.lowpass(M_raw, cut, fs, order),
                                     pl[side]["origin_m"], float(pl["contact_threshold_n"]))
        body = forceplate.assign_foot_body(load, Pf[:, ank], Pf[:, toe], ank, toe)
        loads.append(idyn.ExternalLoad(load.force, load.moment_at_origin, body))
        mid = 0.5 * (Pf[:, ank] + Pf[:, toe])
        dist = np.where(load.contact, np.linalg.norm((load.cop - mid)[:, :2], axis=1), np.nan)
        feet[side] = (load, dist)

    wr = idyn.joint_wrenches(model.parents, seg.mass, seg.inertia, kin, loads, g)
    wb = idyn.whole_body(seg.mass, seg.inertia, kin)
    cal = jcs.calibrate(J)
    Mj = wr.moment[:, cal.child]
    mom = jcs.project(cal, Rf, Mj)
    om = jcs.project(cal, Rf, kin.omega[:, cal.child] - kin.omega[:, cal.parent])
    idx = np.arange(len(cal.names))
    torque = mom[:, idx, cal.primary]
    power_total = (Mj * (kin.omega[:, cal.child] - kin.omega[:, model.parents[cal.child]])).sum(-1)

    contact = np.stack([feet[s][0].contact for s in _FEET], 1)
    dist = np.stack([feet[s][1] for s in _FEET], 1)
    grf = np.stack([feet[s][0].force for s in _FEET], 1) * contact[..., None]
    return TakeKinetics(
        joint_names=cal.names,
        grf=grf,
        cop=np.stack([feet[s][0].cop for s in _FEET], 1),
        free_moment=np.stack([feet[s][0].free_moment for s in _FEET], 1),
        foot_contact=contact,
        com_acceleration=wb.com_acceleration,
        angular_momentum=wb.angular_momentum,
        joint_torque=torque,
        joint_power=torque * om[:, idx, cal.primary],
        joint_moments_jcs=mom,
        joint_power_total=power_total,
        primary_component=cal.primary.copy(),
        residual_force=wr.force[:, 0],
        residual_moment=wr.moment[:, 0],
        grf_valid=~(np.nan_to_num(dist, nan=0.0) > float(q["cop_to_foot_max_m"])).any(1),
        kinematics_valid=kin_valid,
        cop_to_foot_distance=dist,
        max_rotation_step_deg=step,
        body_mass_force_plate_kg=mass,
        segment_mass_kg=seg.mass,
        mesh_volume_m3=seg.volume,
    )


def lab_to_overground(fit, rest_pelvis: np.ndarray):
    """Per-frame rigid transform (R (T,3,3), d (T,3)) with x_overground = R x_lab + d.

    Read off the pelvis: the overground fit is the treadmill pose moved rigidly, so the
    transform that carries the pelvis frame carries every point of the body.
    """
    Ro = Rotation.from_rotvec(np.asarray(fit["global_orient"], np.float64)).as_matrix()
    Rt = Rotation.from_rotvec(np.asarray(fit["global_orient_treadmill"], np.float64)).as_matrix()
    R = Ro @ np.swapaxes(Rt, -1, -2)
    po = np.asarray(fit["trans"], np.float64) + rest_pelvis
    pt = np.asarray(fit["trans_treadmill"], np.float64) + rest_pelvis
    return R, po - np.einsum("tij,tj->ti", R, pt)


def to_frame(k: TakeKinetics, R: np.ndarray, d: np.ndarray) -> dict[str, np.ndarray]:
    """The world arrays of ``k`` in the frame x' = R x + d (R (T,3,3) or (3,3), d (T,3) or (3,))."""
    R = np.broadcast_to(np.asarray(R, np.float64), (len(k.grf), 3, 3))
    d = np.broadcast_to(np.asarray(d, np.float64), (len(k.grf), 3))

    def vec(x):
        return np.einsum("tij,t...j->t...i", R, x)

    cop = vec(k.cop) + d[:, None, :]
    return {
        "grf": vec(k.grf),
        "cop": cop,
        "com_acceleration": vec(k.com_acceleration),
        "angular_momentum": vec(k.angular_momentum),
        "residual_force": vec(k.residual_force),
        "residual_moment": vec(k.residual_moment),
    }
