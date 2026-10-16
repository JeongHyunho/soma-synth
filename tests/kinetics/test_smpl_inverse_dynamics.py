"""Analytic checks of the SMPL inverse-dynamics building blocks (soma_synth.kinetics)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from soma_synth.kinetics import bsip, forceplate, inverse_dynamics as idyn, jcs

G = np.array([0.0, 0.0, -9.80665])


def _cube(side):
    v = np.array([[x, y, z] for x in (0, side) for y in (0, side) for z in (0, side)], float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def test_voxelised_cube_has_the_cube_volume_and_inertia():
    v, f = _cube(0.2)
    w = np.ones((8, 1))
    seg = bsip.segment_inertia_from_mesh(v, f, w, total_mass=10.0, voxel_size=0.005)
    assert seg.volume == pytest.approx(0.008, rel=0.03)
    assert seg.mass[0] == pytest.approx(10.0)
    np.testing.assert_allclose(seg.com[0], [0.1, 0.1, 0.1], atol=0.003)
    np.testing.assert_allclose(np.diag(seg.inertia[0]), 10.0 * 0.04 / 6.0, rtol=0.04)


def _static_kin(R, P, rest_joint, rest_com, T=50):
    R = np.repeat(R[None], T, 0)
    P = np.repeat(P[None], T, 0)
    return idyn.body_kinematics(R, P, rest_joint, rest_com, dt=0.01)


def test_hanging_two_link_chain_matches_the_static_moments():
    # Body 0 hangs from joint 0 at the origin, body 1 from joint 1 at x = 1 m; links point along +x.
    parents = np.array([-1, 0])
    rest_joint = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    rest_com = np.array([[0.5, 0, 0], [1.5, 0, 0]])
    mass = np.array([2.0, 3.0])
    inertia = np.zeros((2, 3, 3))
    kin = _static_kin(np.stack([np.eye(3)] * 2), rest_joint, rest_joint, rest_com)
    w = idyn.joint_wrenches(parents, mass, inertia, kin, [], G)
    g = 9.80665
    # Moment the parent must supply to hold the child: about +y?  r = +x, weight = -z: r x F_req.
    np.testing.assert_allclose(w.moment[:, 1], [[0, -0.5 * 3 * g, 0]] * 50, atol=1e-9)
    np.testing.assert_allclose(w.moment[:, 0], [[0, -(0.5 * 2 + 1.5 * 3) * g, 0]] * 50, atol=1e-9)
    np.testing.assert_allclose(w.force[:, 0], [[0, 0, 5 * g]] * 50, atol=1e-9)


def test_a_body_held_up_by_its_ground_reaction_leaves_no_residual():
    parents = np.array([-1])
    rest_joint = np.zeros((1, 3))
    rest_com = np.array([[0.1, 0.2, 0.9]])
    mass = np.array([70.0])
    kin = _static_kin(np.eye(3)[None], rest_joint, rest_joint, rest_com)
    F = np.tile([0, 0, 70.0 * 9.80665], (50, 1))
    cop = np.array([0.1, 0.2, 0.0])
    load = idyn.ExternalLoad(force=F, moment_at_origin=np.cross(cop, F), body=np.zeros(50, int))
    w = idyn.joint_wrenches(parents, mass, np.zeros((1, 3, 3)), kin, [load], G)
    np.testing.assert_allclose(w.force[:, 0], 0, atol=1e-9)
    np.testing.assert_allclose(w.moment[:, 0], 0, atol=1e-9)


def test_plate_wrench_becomes_the_body_load_with_its_cop_and_free_moment():
    # The foot pushes on the plate with 700 N down and 50 N backward at a CoP 5 cm right of and
    # 10 cm ahead of the plate origin, twisting it by 3 N m: what a plate reports, about its origin.
    O = np.array([0.25, 0.0, 0.0])
    c = np.array([0.30, 0.10, 0.0])
    F_plate = np.array([[0.0, -50.0, -700.0], [0.0, 0.0, -5.0]])
    tz = np.array([3.0, 0.0])
    M_plate = np.cross(c - O, F_plate) + np.c_[0 * tz, 0 * tz, tz]
    load = forceplate.plate_load(F_plate, M_plate, O, contact_threshold_n=20.0)
    np.testing.assert_allclose(load.force[0], [0, 50, 700])
    np.testing.assert_allclose(load.cop[0], c, atol=1e-12)
    assert load.free_moment[0] == pytest.approx(-3.0)
    assert np.isnan(load.cop[1]).all() and np.isnan(load.free_moment[1])
    assert load.contact.tolist() == [True, False]
    # The world-origin moment is the body-side wrench moved from the CoP.
    np.testing.assert_allclose(load.moment_at_origin[0], np.cross(c, load.force[0]) + [0, 0, -3.0], atol=1e-9)


def test_spinning_body_angular_momentum_is_inertia_times_rate():
    T, dt, rate = 200, 0.01, 2.0
    R = Rotation.from_rotvec(np.outer(np.arange(T) * dt * rate, [0, 0, 1])).as_matrix()[:, None]
    P = np.zeros((T, 1, 3))
    I = np.diag([1.0, 2.0, 3.0])[None]
    kin = idyn.body_kinematics(R, P, np.zeros((1, 3)), np.zeros((1, 3)), dt)
    wb = idyn.whole_body(np.array([5.0]), I, kin)
    np.testing.assert_allclose(wb.angular_momentum[5:-5], [[0, 0, 3.0 * rate]] * (T - 10), atol=1e-9)


def _tpose():
    """A synthetic SMPL-like rest skeleton: +x left, +y up, +z forward (the SMPL rest axes)."""
    j = np.zeros((24, 3))
    j[0] = [0, 0.9, 0]
    j[1], j[2] = [0.09, 0.82, 0], [-0.09, 0.82, 0]
    j[4], j[5] = [0.1, 0.45, 0], [-0.1, 0.45, 0]
    j[7], j[8] = [0.1, 0.07, -0.02], [-0.1, 0.07, -0.02]
    j[10], j[11] = [0.1, 0.0, 0.12], [-0.1, 0.0, 0.12]
    j[3], j[6], j[9] = [0, 1.0, 0], [0, 1.12, 0], [0, 1.25, 0]
    j[12], j[15] = [0, 1.45, 0], [0, 1.55, 0]
    j[13], j[14] = [0.08, 1.38, 0], [-0.08, 1.38, 0]
    j[16], j[17] = [0.18, 1.4, 0], [-0.18, 1.4, 0]
    j[18], j[19] = [0.45, 1.4, 0], [-0.45, 1.4, 0]
    j[20], j[21] = [0.7, 1.4, 0], [-0.7, 1.4, 0]
    j[22], j[23] = [0.8, 1.4, 0], [-0.8, 1.4, 0]
    return j


@pytest.mark.parametrize(
    "joint, primary_sign",
    [
        ("right_hip", -1), ("left_hip", -1),          # backward swing = hip extension
        ("right_knee", +1), ("left_knee", +1),        # backward swing = knee flexion
        ("right_ankle", -1), ("left_ankle", -1),      # toes move down = plantarflexion
        ("spine3", +1), ("neck", +1),                  # top moves forward = flexion
    ],
)
def test_sagittal_primary_signs_follow_the_anatomical_motion(joint, primary_sign):
    # A rotation of the child about +x (subject left, the SMPL rest x axis) moves a point below
    # the joint (r = -y) with velocity x cross -y = -z, i.e. backward, and a point above it or
    # ahead of it (the toes) backward or down. Each joint's primary component must read that
    # motion with the anatomical sign.
    cal = jcs.calibrate(_tpose())
    i = cal.names.index(joint)
    R = np.repeat(np.eye(3)[None, None], 24, 1)
    w = np.zeros((1, len(cal.names), 3))
    w[0, i] = [1.0, 0.0, 0.0]
    comp = jcs.project(cal, R, w)[0, i, cal.primary[i]]
    assert np.sign(comp) == primary_sign


def test_every_joint_gets_a_primary_sign_and_left_secondaries_are_mirrored():
    cal = jcs.calibrate(_tpose())
    assert len(cal.names) == 17
    assert np.all(np.abs(cal.sign) == 1)
    for i, n in enumerate(cal.names):
        if n.startswith("left_"):
            others = [k for k in range(3) if k != cal.primary[i]]
            assert np.all(cal.sign[i, others] == -1)
