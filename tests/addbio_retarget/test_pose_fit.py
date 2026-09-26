"""Turning OpenSim body rotations into SMPL pose.

The transfer is one constant per bone: SMPL's rest skeleton and OpenSim's body frames differ by
a fixed rotation, and once that is pinned the whole time-varying rotation -- including the twist
about a bone's own axis, which positions can never show -- carries across unchanged.

Pinning it is where the arbitrariness lives, and the tests draw the line. A joint with several
children is fully determined. A joint with one child is determined only up to twist, so the
smallest rotation is chosen and written into `rest_alignment` for downstream to re-reference;
what must *not* happen is that choice drifting from frame to frame, which would inject rotation
that the source never had.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from soma_synth.addbio_retarget.pose_fit import (
    fit_pose,
    smpl_world_positions,
)
from soma_synth.addbio_retarget.smpl_correspondence import (
    ABSENT,
    DERIVED_LUMBAR,
    MEASURED,
    SMPL24_NAMES,
    SMPL24_PARENTS,
    SmplJointSource,
)

# A skeleton where SMPL and the source agree exactly, so the alignment constants are identity
# and any deviation the fit reports is its own doing.
BODIES = {
    0: "pelvis", 1: "femur_l", 2: "femur_r", 4: "tibia_l", 5: "tibia_r",
    7: "calcn_l", 8: "calcn_r", 10: "toes_l", 11: "toes_r",
    16: "humerus_l", 17: "humerus_r", 18: "radius_l", 19: "radius_r",
    20: "hand_l", 21: "hand_r",
}
CENTRE_JOINT = {
    0: "ground_pelvis", 1: "hip_l", 2: "hip_r", 3: "back", 4: "knee_l", 5: "knee_r",
    7: "ankle_l", 8: "ankle_r", 10: "mtp_l", 11: "mtp_r",
    16: "acromial_l", 17: "acromial_r", 18: "elbow_l", 19: "elbow_r",
    20: "wrist_l", 21: "wrist_r",
}
LUMBAR = ("lumbar_extension", "lumbar_bending", "lumbar_rotation")


def correspondence():
    table = []
    for index, name in enumerate(SMPL24_NAMES):
        if index in (3, 6, 9):
            table.append(SmplJointSource(index, name, DERIVED_LUMBAR,
                                         "back" if index == 3 else None, LUMBAR))
        elif index in BODIES:
            table.append(SmplJointSource(index, name, MEASURED,
                                         CENTRE_JOINT[index], (f"q{index}",)))
        else:
            table.append(SmplJointSource(index, name, ABSENT, None, ()))
    return tuple(table)


def rest_skeleton(seed=3):
    rng = np.random.default_rng(seed)
    joints = np.zeros((24, 3))
    for child, parent in enumerate(SMPL24_PARENTS):
        if parent >= 0:
            joints[child] = joints[parent] + rng.normal(0.0, 0.2, 3)
    return joints


def synthetic_source(j_rest, frames=6, seed=5, alignment=None):
    """Build source rotations and centres from a known SMPL pose, so truth is available.

    `alignment` optionally rotates every body frame by a fixed amount, standing in for the two
    skeletons defining their frames differently.
    """
    rng = np.random.default_rng(seed)
    local = np.tile(np.eye(3), (frames, 24, 1, 1))
    for t in range(frames):
        for j in BODIES:
            local[t, j] = Rotation.from_rotvec(rng.normal(0.0, 0.25, 3)).as_matrix()
        # The source measures the trunk as one rigid body, so a representable truth splits it
        # the way the rule does. Joints with no source stay at rest, as the artifact will hold
        # them. Anything else here would ask the fit to recover what the source never had.
        third = Rotation.from_rotvec(rng.normal(0.0, 0.1, 3)).as_matrix()
        for j in (3, 6, 9):
            local[t, j] = third
    globals_ = np.zeros((frames, 24, 3, 3))
    positions = np.zeros((frames, 24, 3))
    for t in range(frames):
        globals_[t, 0] = local[t, 0]
        for j in range(1, 24):
            p = int(SMPL24_PARENTS[j])
            globals_[t, j] = globals_[t, p] @ local[t, j]
            positions[t, j] = positions[t, p] + globals_[t, p] @ (j_rest[j] - j_rest[p])

    offset = np.eye(3) if alignment is None else alignment
    rotations = {}
    for index, body in BODIES.items():
        rotations[body] = np.array([globals_[t, index] @ offset.T for t in range(frames)])
    rotations["torso"] = np.array([globals_[t, 9] @ offset.T for t in range(frames)])

    root = rng.normal(0.0, 1.0, (frames, 3))
    centres = {
        CENTRE_JOINT[index]: positions[:, index] + root
        for index in CENTRE_JOINT
        if index in BODIES
    }
    centres["back"] = positions[:, 3] + root
    return rotations, centres, positions + root[:, None, :], local


def run(alignment=None, frames=6):
    j_rest = rest_skeleton()
    rotations, centres, truth, local = synthetic_source(j_rest, frames=frames,
                                                        alignment=alignment)
    result = fit_pose(
        rest_joints=j_rest,
        correspondence=correspondence(),
        body_of=BODIES,
        source_rotations=rotations,
        source_centres=centres,
        world_rotation=np.eye(3),
    )
    return result, truth, j_rest, local


# ---------------------------------------------------------------- shape of the answer


def test_the_result_is_shaped_the_way_the_artifact_wants():
    result, _truth, _rest, _local = run()
    assert result.pose.shape == (6, 24, 3)
    assert result.trans.shape == (6, 3)
    assert result.rest_alignment.shape == (24, 4)
    assert result.joint_provenance.shape == (24,)
    np.testing.assert_allclose(np.linalg.norm(result.rest_alignment, axis=1), 1.0, atol=1e-9)


def test_trans_is_the_world_pelvis_because_that_is_this_pipelines_convention():
    result, truth, _rest, _local = run()
    np.testing.assert_allclose(result.trans, truth[:, 0], atol=1e-9)


# ---------------------------------------------------------------- the fit itself


def test_it_reproduces_every_measured_joint_centre():
    result, truth, j_rest, _local = run()
    world = smpl_world_positions(j_rest, result.pose, result.trans)
    for index in CENTRE_JOINT:
        np.testing.assert_allclose(world[:, index], truth[:, index], atol=1e-9)


def test_a_constant_frame_difference_between_the_skeletons_is_absorbed():
    """The two skeletons need not agree on a body's axes; a fixed offset must not leak in."""
    offset = Rotation.from_rotvec([0.3, -0.7, 0.2]).as_matrix()
    result, truth, j_rest, _local = run(alignment=offset)
    world = smpl_world_positions(j_rest, result.pose, result.trans)
    for index in CENTRE_JOINT:
        np.testing.assert_allclose(world[:, index], truth[:, index], atol=1e-9)


def test_the_alignment_is_one_constant_and_not_a_per_frame_fudge():
    """Re-fitting on more frames must not move it; a drifting constant is invented motion."""
    short, _t1, _r1, _l1 = run(frames=3)
    long, _t2, _r2, _l2 = run(frames=12)
    for index in BODIES:
        assert 1.0 - abs(float(np.dot(short.rest_alignment[index],
                                      long.rest_alignment[index]))) < 1e-6


def test_twist_reaches_the_pose_even_though_no_position_shows_it():
    """Spin a bone about its own axis: positions do not move, the emitted pose must."""
    j_rest = rest_skeleton()
    rotations, centres, _truth, _local = synthetic_source(j_rest)
    axis = j_rest[4] - j_rest[1]
    axis = axis / np.linalg.norm(axis)
    spun = {k: v.copy() for k, v in rotations.items()}
    twist = Rotation.from_rotvec(axis * 0.4).as_matrix()
    spun["femur_l"] = np.array([r @ twist for r in spun["femur_l"]])

    plain = fit_pose(rest_joints=j_rest, correspondence=correspondence(), body_of=BODIES,
                     source_rotations=rotations, source_centres=centres,
                     world_rotation=np.eye(3))
    turned = fit_pose(rest_joints=j_rest, correspondence=correspondence(), body_of=BODIES,
                      source_rotations=spun, source_centres=centres,
                      world_rotation=np.eye(3))
    assert not np.allclose(plain.pose[:, 1], turned.pose[:, 1], atol=1e-6)


# ---------------------------------------------------------------- provenance and rules


def test_absent_joints_sit_at_rest_and_say_so():
    result, _truth, _rest, _local = run()
    for index in (12, 13, 14, 15, 22, 23):
        assert result.joint_provenance[index] == ABSENT
        np.testing.assert_allclose(result.pose[:, index], 0.0, atol=1e-12)


def test_the_lumbar_is_split_in_three_equal_turns_that_compose_to_the_trunk():
    result, _truth, j_rest, _local = run()
    for index in (3, 6, 9):
        assert result.joint_provenance[index] == DERIVED_LUMBAR
    first = Rotation.from_rotvec(result.pose[:, 3])
    second = Rotation.from_rotvec(result.pose[:, 6])
    third = Rotation.from_rotvec(result.pose[:, 9])
    np.testing.assert_allclose(first.as_rotvec(), second.as_rotvec(), atol=1e-9)
    np.testing.assert_allclose(second.as_rotvec(), third.as_rotvec(), atol=1e-9)


def test_a_source_missing_a_body_leaves_that_joint_absent_rather_than_guessing():
    j_rest = rest_skeleton()
    rotations, centres, _truth, _local = synthetic_source(j_rest)
    for body in ("humerus_l", "humerus_r", "radius_l", "radius_r", "hand_l", "hand_r"):
        rotations.pop(body)
    for name in ("acromial_l", "acromial_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r"):
        centres.pop(name)
    result = fit_pose(rest_joints=j_rest, correspondence=correspondence(), body_of=BODIES,
                      source_rotations=rotations, source_centres=centres,
                      world_rotation=np.eye(3))
    for index in (16, 17, 18, 19, 20, 21):
        assert result.joint_provenance[index] == ABSENT
        np.testing.assert_allclose(result.pose[:, index], 0.0, atol=1e-12)


def test_the_world_rotation_is_applied_once_and_to_everything():
    j_rest = rest_skeleton()
    rotations, centres, truth, _local = synthetic_source(j_rest)
    turn = Rotation.from_rotvec([np.pi / 2, 0, 0]).as_matrix()
    result = fit_pose(rest_joints=j_rest, correspondence=correspondence(), body_of=BODIES,
                      source_rotations=rotations, source_centres=centres,
                      world_rotation=turn)
    world = smpl_world_positions(j_rest, result.pose, result.trans)
    for index in CENTRE_JOINT:
        np.testing.assert_allclose(world[:, index], truth[:, index] @ turn.T, atol=1e-9)


def test_the_reported_error_is_the_error_it_actually_made():
    result, truth, j_rest, _local = run()
    world = smpl_world_positions(j_rest, result.pose, result.trans)
    measured = [i for i in CENTRE_JOINT]
    worst = max(
        float(np.abs(world[:, i] - truth[:, i]).max()) for i in measured
    )
    assert result.position_error_m == pytest.approx(worst, abs=1e-9) or (
        result.position_error_m >= worst - 1e-9
    )


# ---------------------------------------------------------------- leaves with their own frame


def _one_body_turned(rotations, body, degrees=30.0):
    """Give one source body a frame of its own, which nothing else in this suite does.

    Every other case here rotates *every* body by the same constant, so a per-body difference --
    the thing that actually happens when a source builds each segment's frame from its own
    anatomical axes -- has never been exercised.
    """
    turn = Rotation.from_rotvec([0.0, 0.0, np.deg2rad(degrees)]).as_matrix()
    turned = dict(rotations)
    turned[body] = np.array([frame @ turn.T for frame in rotations[body]])
    return turned, turn


def test_a_leaf_carries_its_own_frame_difference_into_the_pose():
    """A joint with no descendant to aim at inherits, and inheriting can be wrong.

    SMPL's left wrist has one child, the hand, and the hand has no centre -- so the wrist never
    gets a constant of its own and keeps the elbow's. That is right only when the two source
    bodies share a frame. Here the hand body is turned 30 degrees and nothing else is, and the
    whole 30 degrees lands in the emitted joint angle.
    """
    j_rest = rest_skeleton()
    rotations, centres, _truth, local = synthetic_source(j_rest)
    turned, _turn = _one_body_turned(rotations, "hand_l")
    result = fit_pose(
        rest_joints=j_rest, correspondence=correspondence(), body_of=BODIES,
        source_rotations=turned, source_centres=centres, world_rotation=np.eye(3),
    )
    emitted = Rotation.from_rotvec(result.pose[:, 20]).as_matrix()
    error = np.degrees(
        Rotation.from_matrix(np.swapaxes(local[:, 20], 1, 2) @ emitted).magnitude()
    )
    assert 20 not in result.alignment_children, "the wrist should have no constant of its own"
    assert error.min() > 29.0, f"expected the full 30 deg to show up, got {error}"


def test_a_reference_pose_gives_a_leaf_the_constant_it_cannot_fit():
    """The same case, with a configuration in which that joint is known to be at rest.

    A leaf has no offset to fit against, so the only way to place its body is to be told a pose
    where SMPL's angle there is zero and read the frame difference off it. That is what a static
    trial is for, and supplying one is the caller asserting the joint is neutral in it.
    """
    j_rest = rest_skeleton()
    rotations, centres, _truth, local = synthetic_source(j_rest)
    turned, turn = _one_body_turned(rotations, "hand_l")
    # the source as it would read with every SMPL joint at rest: the globals are identity, so
    # each body reports nothing but its own frame offset
    reference = {body: np.eye(3) for body in turned}
    reference["hand_l"] = turn.T
    result = fit_pose(
        rest_joints=j_rest, correspondence=correspondence(), body_of=BODIES,
        source_rotations=turned, source_centres=centres, world_rotation=np.eye(3),
        reference_rotations=reference,
    )
    emitted = Rotation.from_rotvec(result.pose[:, 20]).as_matrix()
    error = np.degrees(
        Rotation.from_matrix(np.swapaxes(local[:, 20], 1, 2) @ emitted).magnitude()
    )
    assert error.max() < 1e-6, f"the reference should remove it entirely, left {error.max()}"


def test_a_reference_pose_does_not_disturb_a_joint_that_was_already_fitted():
    """Joints with measured offsets keep them: a reference pose is the weaker evidence."""
    j_rest = rest_skeleton()
    rotations, centres, _truth, _local = synthetic_source(j_rest)
    reference = {body: np.eye(3) for body in rotations}
    plain = fit_pose(
        rest_joints=j_rest, correspondence=correspondence(), body_of=BODIES,
        source_rotations=rotations, source_centres=centres, world_rotation=np.eye(3))
    withref = fit_pose(
        rest_joints=j_rest, correspondence=correspondence(), body_of=BODIES,
        source_rotations=rotations, source_centres=centres, world_rotation=np.eye(3),
        reference_rotations=reference)
    for joint in sorted(plain.alignment_children):
        np.testing.assert_allclose(withref.rest_alignment[joint],
                                   plain.rest_alignment[joint], atol=1e-12)


def test_omitting_the_reference_leaves_every_number_exactly_as_it_was():
    """The default path stays bit-identical, or an already generated corpus is invalidated."""
    j_rest = rest_skeleton()
    rotations, centres, _truth, _local = synthetic_source(j_rest)
    kwargs = dict(rest_joints=j_rest, correspondence=correspondence(), body_of=BODIES,
                  source_rotations=rotations, source_centres=centres, world_rotation=np.eye(3))
    without = fit_pose(**kwargs)
    explicit = fit_pose(reference_rotations=None, **kwargs)
    np.testing.assert_array_equal(without.pose, explicit.pose)
    np.testing.assert_array_equal(without.trans, explicit.trans)
    np.testing.assert_array_equal(without.rest_alignment, explicit.rest_alignment)


# ---------------------------------------------------------------- placing the root

# Two hip offsets and nothing else is what a marker-driven source gives at the pelvis, and it is
# not enough for one Kabsch fit to answer two questions at once. The tests below hold the failure
# and the fix side by side: the same skeleton, the same displacement, one with `root_placement`
# and one without.

from soma_synth.addbio_retarget.pose_fit import RootPlacement  # noqa: E402


def two_hip_correspondence():
    """The same table with the trunk's centre removed, so the root has only its two hips."""
    return tuple(
        SmplJointSource(e.smpl_index, e.smpl_name, e.provenance,
                        None if e.smpl_index == 3 else e.centre_joint, e.source_coordinates)
        for e in correspondence()
    )


def displaced_source(displacement, frames=4):
    """A source whose pelvis centre sits `displacement` from where SMPL puts its root.

    That is the real disagreement: Visual3D's pelvis proximal end and SMPL's pelvis joint are
    both called the pelvis and are not the same point.
    """
    j_rest = rest_skeleton()
    rotations, centres, truth, _local = synthetic_source(j_rest, frames=frames)
    centres = dict(centres)
    centres.pop("back")
    centres["ground_pelvis"] = centres["ground_pelvis"] + np.einsum(
        "tij,j->ti", rotations["pelvis"], np.asarray(displacement, float)
    )
    return j_rest, rotations, centres, truth


def test_two_hip_offsets_and_a_displaced_pelvis_come_out_as_a_rotation():
    """The failure this exists to stop: a displacement the fit cannot express turns the body."""
    j_rest, rotations, centres, _truth = displaced_source(np.array([0.03, 0.0, 0.0]))
    result = fit_pose(rest_joints=j_rest, correspondence=two_hip_correspondence(),
                      body_of=BODIES, source_rotations=rotations, source_centres=centres,
                      world_rotation=np.eye(3))
    # nothing was translated: _root_alignment needs three targets before it fits an offset
    assert np.allclose(result.root_offset, 0.0)
    # and the pelvis is turned instead, which the source never did. How far depends on the
    # skeleton -- the lever is the hip offsets -- and on this one a 30 mm displacement buys
    # nearly 4 deg; on the HKNU cohort's real geometry the same trade costs 23.
    turned = Rotation.from_rotvec(result.pose[:, 0]).as_matrix()
    angle = np.degrees(np.linalg.norm(Rotation.from_matrix(
        np.einsum("tji,tjk->tik", rotations["pelvis"], turned)).as_rotvec(), axis=1))
    assert angle.max() > 2.0


def test_a_root_placement_puts_the_hip_centroid_where_the_source_measured_it():
    j_rest, rotations, centres, _truth = displaced_source(np.array([0.03, 0.0, 0.0]))
    placement = RootPlacement(
        rest_directions=np.stack([j_rest[1] - j_rest[2], j_rest[12] - j_rest[0]]),
        measured_directions=np.stack([j_rest[1] - j_rest[2], j_rest[12] - j_rest[0]]),
        anchor_joints=(1, 2), anchor_centres=("hip_l", "hip_r"),
    )
    result = fit_pose(rest_joints=j_rest, correspondence=two_hip_correspondence(),
                      body_of=BODIES, source_rotations=rotations, source_centres=centres,
                      world_rotation=np.eye(3), root_placement=placement)
    world = smpl_world_positions(j_rest, result.pose, result.trans)
    measured = 0.5 * (centres["hip_l"] + centres["hip_r"])
    assert np.abs(0.5 * (world[:, 1] + world[:, 2]) - measured).max() < 1e-9
    turned = Rotation.from_rotvec(result.pose[:, 0]).as_matrix()
    angle = np.degrees(np.linalg.norm(Rotation.from_matrix(
        np.einsum("tji,tjk->tik", rotations["pelvis"], turned)).as_rotvec(), axis=1))
    assert angle.max() < 1e-6


def test_the_root_placement_reads_directions_and_not_lengths():
    """A skeleton that disagrees about a bone's length must not lean the body because of it."""
    j_rest, rotations, centres, _truth = displaced_source(np.array([0.03, 0.0, 0.0]))
    base = dict(anchor_joints=(1, 2), anchor_centres=("hip_l", "hip_r"),
                rest_directions=np.stack([j_rest[1] - j_rest[2], j_rest[12] - j_rest[0]]))
    plain = RootPlacement(
        measured_directions=np.stack([j_rest[1] - j_rest[2], j_rest[12] - j_rest[0]]), **base)
    stretched = RootPlacement(measured_directions=np.stack(
        [8.0 * (j_rest[1] - j_rest[2]), 0.2 * (j_rest[12] - j_rest[0])]), **base)
    kwargs = dict(rest_joints=j_rest, correspondence=two_hip_correspondence(), body_of=BODIES,
                  source_rotations=rotations, source_centres=centres, world_rotation=np.eye(3))
    a = fit_pose(root_placement=plain, **kwargs)
    b = fit_pose(root_placement=stretched, **kwargs)
    assert np.abs(a.pose - b.pose).max() < 1e-12
    assert np.abs(a.trans - b.trans).max() < 1e-12


def test_one_direction_cannot_fix_three_axes_and_says_so():
    j_rest, rotations, centres, _truth = displaced_source(np.zeros(3))
    placement = RootPlacement(rest_directions=np.stack([j_rest[1] - j_rest[2]]),
                              measured_directions=np.stack([j_rest[1] - j_rest[2]]),
                              anchor_joints=(1, 2), anchor_centres=("hip_l", "hip_r"))
    with pytest.raises(ValueError, match="two directions"):
        fit_pose(rest_joints=j_rest, correspondence=two_hip_correspondence(), body_of=BODIES,
                 source_rotations=rotations, source_centres=centres,
                 world_rotation=np.eye(3), root_placement=placement)


def test_omitting_the_root_placement_leaves_every_number_exactly_as_it_was():
    j_rest, rotations, centres, _truth = displaced_source(np.array([0.02, -0.01, 0.0]))
    kwargs = dict(rest_joints=j_rest, correspondence=two_hip_correspondence(), body_of=BODIES,
                  source_rotations=rotations, source_centres=centres, world_rotation=np.eye(3))
    assert np.array_equal(fit_pose(**kwargs).pose, fit_pose(root_placement=None, **kwargs).pose)


# ---------------------------------------------------------------- what the lumbar calls zero


def frame_shifted_trunk(shift_degrees=20.0, frames=4):
    """A source whose trunk never moves relative to its pelvis, but whose frames differ.

    That is a frame convention, not a posture, and a lumbar rule reading it as rotation bends
    the fitted spine in every frame of every trial.
    """
    j_rest = rest_skeleton()
    rotations, centres, _truth, _local = synthetic_source(j_rest, frames=frames)
    difference = Rotation.from_rotvec(
        np.radians(shift_degrees) * np.array([1.0, 0.0, 0.0])).as_matrix()
    rotations = dict(rotations)
    rotations["torso"] = np.einsum("tij,jk->tik", rotations["pelvis"], difference)
    reference = {name: stack[0] for name, stack in rotations.items()}
    return j_rest, rotations, centres, reference


def test_the_lumbar_bends_by_the_frame_difference_when_nothing_says_what_zero_is():
    j_rest, rotations, centres, reference = frame_shifted_trunk()
    result = fit_pose(rest_joints=j_rest, correspondence=correspondence(), body_of=BODIES,
                      source_rotations=rotations, source_centres=centres,
                      world_rotation=np.eye(3), reference_rotations=reference)
    spine = np.degrees(np.linalg.norm(result.pose[:, [3, 6, 9]], axis=2))
    assert spine.min() > 5.0                 # each spine joint carries a third of the 20 deg


def test_measured_against_the_reference_a_neutral_trunk_leaves_the_spine_straight():
    j_rest, rotations, centres, reference = frame_shifted_trunk()
    result = fit_pose(rest_joints=j_rest, correspondence=correspondence(), body_of=BODIES,
                      source_rotations=rotations, source_centres=centres,
                      world_rotation=np.eye(3), reference_rotations=reference,
                      lumbar_from_reference=True)
    spine = np.degrees(np.linalg.norm(result.pose[:, [3, 6, 9]], axis=2))
    assert spine.max() < 1e-9


def test_the_trunk_s_actual_movement_still_crosses_over():
    """Neutralising the frame difference must not neutralise the motion with it.

    What crosses over is the *size* of the turn, not its axis written in the source's letters:
    the spine reproduces it in SMPL's frame, so it arrives conjugated by the constants. Asking
    for the matrix back unchanged would be asking the two skeletons to share a frame, which is
    the assumption this whole module exists to avoid.
    """
    j_rest, rotations, centres, reference = frame_shifted_trunk()
    turn = Rotation.from_rotvec(np.radians(9.0) * np.array([0.0, 1.0, 0.0])).as_matrix()
    moved = dict(rotations)
    moved["torso"] = moved["torso"].copy()
    moved["torso"][2:] = np.einsum("tij,jk->tik", moved["torso"][2:], turn)
    result = fit_pose(rest_joints=j_rest, correspondence=correspondence(), body_of=BODIES,
                      source_rotations=moved, source_centres=centres,
                      world_rotation=np.eye(3), reference_rotations=reference,
                      lumbar_from_reference=True)
    composed = np.eye(3)
    for index in (3, 6, 9):
        composed = composed @ Rotation.from_rotvec(result.pose[2, index]).as_matrix()
    moved_by = np.degrees(np.linalg.norm(Rotation.from_matrix(composed).as_rotvec()))
    assert abs(moved_by - 9.0) < 1e-9
    # and the frames that did not move are still straight, so the 9 deg is the movement and
    # not the frame difference leaking back in
    assert np.degrees(np.linalg.norm(result.pose[0, [3, 6, 9]], axis=1)).max() < 1e-9


def test_asking_for_a_reference_lumbar_without_a_reference_is_refused():
    j_rest, rotations, centres, _reference = frame_shifted_trunk()
    with pytest.raises(ValueError, match="reference configuration"):
        fit_pose(rest_joints=j_rest, correspondence=correspondence(), body_of=BODIES,
                 source_rotations=rotations, source_centres=centres,
                 world_rotation=np.eye(3), lumbar_from_reference=True)


def test_a_root_placement_anchored_below_the_root_s_children_is_refused():
    """The closed form is forward kinematics for the root's children and nothing further."""
    j_rest, rotations, centres, _truth = displaced_source(np.zeros(3))
    placement = RootPlacement(
        rest_directions=np.stack([j_rest[1] - j_rest[2], j_rest[12] - j_rest[0]]),
        measured_directions=np.stack([j_rest[1] - j_rest[2], j_rest[12] - j_rest[0]]),
        anchor_joints=(4, 5), anchor_centres=("knee_l", "knee_r"))
    with pytest.raises(ValueError, match="children of the root"):
        fit_pose(rest_joints=j_rest, correspondence=two_hip_correspondence(), body_of=BODIES,
                 source_rotations=rotations, source_centres=centres,
                 world_rotation=np.eye(3), root_placement=placement)


def test_a_root_placement_naming_a_centre_the_source_lacks_is_refused():
    j_rest, rotations, centres, _truth = displaced_source(np.zeros(3))
    placement = RootPlacement(
        rest_directions=np.stack([j_rest[1] - j_rest[2], j_rest[12] - j_rest[0]]),
        measured_directions=np.stack([j_rest[1] - j_rest[2], j_rest[12] - j_rest[0]]),
        anchor_joints=(1, 2), anchor_centres=("hip_l", "hip_middle_of_nowhere"))
    with pytest.raises(ValueError, match="does not have"):
        fit_pose(rest_joints=j_rest, correspondence=two_hip_correspondence(), body_of=BODIES,
                 source_rotations=rotations, source_centres=centres,
                 world_rotation=np.eye(3), root_placement=placement)


# ------------------------------------------------- solving the root against many centres


def _placement(j_rest, tracked=()):
    return RootPlacement(
        rest_directions=np.stack([j_rest[1] - j_rest[2], j_rest[12] - j_rest[0]]),
        measured_directions=np.stack([j_rest[1] - j_rest[2], j_rest[12] - j_rest[0]]),
        anchor_joints=(1, 2), anchor_centres=("hip_l", "hip_r"), tracked=tracked)


TRACKED = ((1, "hip_l"), (2, "hip_r"), (4, "knee_l"), (5, "knee_r"),
           (7, "ankle_l"), (8, "ankle_r"))


def test_tracking_puts_the_body_on_the_centres_it_tracks():
    """On a source the skeleton can reproduce exactly, tracking changes nothing it should."""
    j_rest, rotations, centres, truth = displaced_source(np.array([0.03, 0.0, 0.0]))
    kwargs = dict(rest_joints=j_rest, correspondence=two_hip_correspondence(), body_of=BODIES,
                  source_rotations=rotations, source_centres=centres, world_rotation=np.eye(3))
    result = fit_pose(root_placement=_placement(j_rest, TRACKED), **kwargs)
    world = smpl_world_positions(j_rest, result.pose, result.trans)
    for joint, name in TRACKED:
        assert np.abs(world[:, joint] - centres[name]).max() < 1e-9


def test_tracking_is_the_least_squares_offset_and_says_so_in_the_numbers():
    """The emitted translation is exactly the mean residual, not an iterate that got close."""
    j_rest, rotations, centres, _truth = displaced_source(np.array([0.02, -0.01, 0.005]))
    kwargs = dict(rest_joints=j_rest, correspondence=two_hip_correspondence(), body_of=BODIES,
                  source_rotations=rotations, source_centres=centres, world_rotation=np.eye(3))
    result = fit_pose(root_placement=_placement(j_rest, TRACKED), **kwargs)
    about_root = smpl_world_positions(j_rest, result.pose, np.zeros((len(result.trans), 3)))
    expected = np.mean([centres[n] - about_root[:, j] for j, n in TRACKED], axis=0)
    assert np.abs(result.trans - expected).max() < 1e-12


def test_tracking_beats_one_anchor_when_the_chain_cannot_close():
    """A source whose knee centre does not sit where a rigid thigh puts it -- which is what a
    marker-driven model gives, its segments placed independently -- is where the difference is."""
    j_rest, rotations, centres, _truth = displaced_source(np.zeros(3))
    centres = dict(centres)
    rng = np.random.default_rng(17)
    for name in ("knee_l", "knee_r", "ankle_l", "ankle_r"):
        centres[name] = centres[name] + rng.normal(0.0, 0.01, centres[name].shape)
    kwargs = dict(rest_joints=j_rest, correspondence=two_hip_correspondence(), body_of=BODIES,
                  source_rotations=rotations, source_centres=centres, world_rotation=np.eye(3))
    one = fit_pose(root_placement=_placement(j_rest), **kwargs)
    many = fit_pose(root_placement=_placement(j_rest, TRACKED), **kwargs)

    def worst(fit):
        world = smpl_world_positions(j_rest, fit.pose, fit.trans)
        return max(np.linalg.norm(world[:, j] - centres[n], axis=1).mean()
                   for j, n in TRACKED)

    assert worst(many) < worst(one)


def test_tracking_a_centre_the_source_lacks_is_refused():
    j_rest, rotations, centres, _truth = displaced_source(np.zeros(3))
    with pytest.raises(ValueError, match="tracks centres the source lacks"):
        fit_pose(rest_joints=j_rest, correspondence=two_hip_correspondence(), body_of=BODIES,
                 source_rotations=rotations, source_centres=centres, world_rotation=np.eye(3),
                 root_placement=_placement(j_rest, ((1, "hip_l"), (4, "knee_of_nowhere"))))


def test_an_empty_tracked_list_keeps_the_constant_anchor():
    j_rest, rotations, centres, _truth = displaced_source(np.array([0.03, 0.0, 0.0]))
    kwargs = dict(rest_joints=j_rest, correspondence=two_hip_correspondence(), body_of=BODIES,
                  source_rotations=rotations, source_centres=centres, world_rotation=np.eye(3))
    assert np.array_equal(fit_pose(root_placement=_placement(j_rest), **kwargs).trans,
                          fit_pose(root_placement=_placement(j_rest, ()), **kwargs).trans)
