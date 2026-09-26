"""Carry OpenSim body rotations across to SMPL pose.

Once the source's bodies are placed (osim_kinematics) the transfer is one constant per bone:

    G_smpl[j] = R_world . G_source[body(j)] . A[j]

with `A[j]` fixed for the whole subject. Because it is constant, every time-varying rotation --
including the twist about a bone's own axis, which no joint centre can show -- crosses over
untouched. A drifting `A` would instead manufacture rotation the source never had.

Fixing `A[j]` is where the choices are, and they differ by how much evidence there is:

  several children   fully determined; the offsets to each child pin all three axes
  one child          determined only up to twist about that bone. The smallest rotation is
                     taken, and the result is written to `rest_alignment` so a downstream
                     reader can re-reference it rather than inherit a silent convention
  no children        inherits its parent's constant. Leaves (toes, hands) and the spine, which
                     has no measured child offset of its own

Two rules land here. Joints the source does not measure hold rest -- an empty
value, not a plausible one. And the lumbar is split into three equal turns, which is a rule
rather than a recovery: the source measures the trunk as one rigid body, so spine1, spine2 and
spine3 are given a third of it each and compose back to exactly the measured trunk rotation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .smpl_correspondence import (
    ABSENT,
    MEASURED,
    SMPL24_NAMES,
    SMPL24_PARENTS,
    SmplJointSource,
)

__all__ = ["PoseFit", "RootPlacement", "fit_pose", "smpl_world_positions"]

_LUMBAR_SMPL = (3, 6, 9)
_TRUNK_BODY = "torso"
_ROOT = 0


@dataclass(frozen=True)
class RootPlacement:
    """Anatomical directions and a landmark the root must hit, for sources the child offsets
    cannot place.

    Supplied by the caller because only the caller knows which of its own measurements are the
    two skeletons' *same* anatomical direction; `_root_from_placement` says why the two
    questions are separated. Omitting it fits the root the original way, offset by offset.
    """

    #: (k, 3) directions in SMPL's rest frame. Lengths are ignored.
    rest_directions: np.ndarray
    #: (k, 3) the same anatomical directions, resolved in the root body's own frame.
    measured_directions: np.ndarray
    #: SMPL joints -- children of the root -- whose centroid the translation lands on target.
    anchor_joints: tuple[int, ...]
    #: the source centres whose centroid is that target. Named per frame, averaged internally.
    anchor_centres: tuple[str, ...]
    #: (SMPL joint, source centre) pairs the root is solved against *every frame*, instead of
    #: being held at the constant offset above. Empty keeps the constant.
    #:
    #: Anchoring a whole body on one landmark makes every error further down the chain
    #: accumulate from that one point, and an accumulated position error is an acceleration
    #: error, which is what a balance residual reads. Solving the root against many measured
    #: centres spreads it instead. With the rotations already fixed the solve is closed form --
    #: the offset that minimises the summed squared residual is the mean of (measured minus
    #: reconstructed) over the listed joints -- so this is the translation half of a per-frame
    #: IK, and it costs one pass. The rotations are still the source's own, untouched.
    tracked: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class PoseFit:
    pose: np.ndarray                 # (T, 24, 3) axis-angle, local to the parent
    trans: np.ndarray                # (T, 3) world pelvis position
    rest_alignment: np.ndarray       # (24, 4) quaternion (w, x, y, z), one per subject
    joint_provenance: np.ndarray     # (24,) measured / derived_lumbar / absent
    position_error_m: float
    per_joint_error_m: dict[int, float]
    alignment_children: dict[int, int]
    #: Where SMPL's root sits inside the source's pelvis body, in metres. Not zero: the two
    #: skeletons anchor the root at different points, and this is that displacement.
    root_offset: np.ndarray = None
    #: Joints whose constant came from `reference_rotations` rather than from measured offsets.
    #: Weaker evidence than a fit, so it is recorded rather than left to be inferred from the
    #: absence of an entry in `alignment_children`.
    placed_from_reference: tuple[int, ...] = ()


def smpl_world_positions(rest_joints, pose, trans) -> np.ndarray:
    """SMPL forward kinematics: `world = trans + positions_with_pelvis_at_origin`."""
    rest = np.asarray(rest_joints, dtype=np.float64)
    local = Rotation.from_rotvec(
        np.asarray(pose, dtype=np.float64).reshape(-1, 3)
    ).as_matrix().reshape(pose.shape[0], 24, 3, 3)
    frames = local.shape[0]
    globals_ = np.empty((frames, 24, 3, 3))
    positions = np.zeros((frames, 24, 3))
    globals_[:, 0] = local[:, 0]
    for joint in range(1, 24):
        parent = int(SMPL24_PARENTS[joint])
        globals_[:, joint] = globals_[:, parent] @ local[:, joint]
        positions[:, joint] = positions[:, parent] + np.einsum(
            "tij,j->ti", globals_[:, parent], rest[joint] - rest[parent]
        )
    return positions + np.asarray(trans, dtype=np.float64)[:, None, :]


def _mean_rotation(stack: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(stack).mean().as_matrix()


def _reference_frame(reference: Mapping[str, np.ndarray], body: str) -> np.ndarray:
    """One rotation per body from a reference configuration, however many frames it has."""
    stack = np.asarray(reference[body], dtype=np.float64)
    return stack if stack.ndim == 2 else _mean_rotation(stack)


def _global_rotations(
    source_rotations: Mapping[str, np.ndarray],
    alignment: np.ndarray,
    world: np.ndarray,
    available: Mapping[int, str],
    provenance: np.ndarray,
    frames: int,
    lumbar_neutral: np.ndarray | None = None,
) -> np.ndarray:
    """SMPL's global rotation per joint per frame, given one constant per bone.

    Shared by the fit and by the reference pass, so that the pose a reference configuration is
    reasoned about is built the same way as the pose that gets emitted -- including the lumbar
    split and the absent joints, which is exactly where a hand-rolled second copy would drift.
    """
    globals_ = np.tile(np.eye(3), (frames, 24, 1, 1))
    for joint in range(24):
        body = available.get(joint)
        if body is None:
            continue
        globals_[:, joint] = world @ source_rotations[body] @ alignment[joint]

    if _TRUNK_BODY in source_rotations and _ROOT in available:
        pelvis = source_rotations[available[_ROOT]]
        trunk = source_rotations[_TRUNK_BODY]
        relative = np.einsum("tji,tjk->tik", pelvis, trunk)
        if lumbar_neutral is not None:
            # The lumbar rule reads the pelvis-to-trunk *frame* difference and spends it as
            # rotation. On a source whose two segment frames are not parallel when the subject
            # stands neutral, that difference is not zero and never becomes zero, so every
            # frame of every trial carries a constant bend -- a different one per subject,
            # because it is their own model's frame convention. Measured against the standing
            # reference instead, a neutral subject gets a straight spine and only the trunk's
            # actual movement is split three ways.
            relative = relative @ lumbar_neutral.T
        third = Rotation.from_rotvec(
            Rotation.from_matrix(relative).as_rotvec() / 3.0
        ).as_matrix()
        step = np.tile(np.eye(3), (frames, 1, 1))
        for level, index in enumerate(_LUMBAR_SMPL, start=1):
            step = step @ third if level > 1 else third
            globals_[:, index] = world @ pelvis @ step @ alignment[index]

    # absent joints follow their parent exactly, which is a local rotation of identity
    for joint in range(1, 24):
        if provenance[joint] == ABSENT:
            globals_[:, joint] = globals_[:, int(SMPL24_PARENTS[joint])]
    return globals_


def _place_from_reference(
    joints: list[int],
    reference: Mapping[str, np.ndarray],
    alignment: np.ndarray,
    world: np.ndarray,
    available: Mapping[int, str],
    provenance: np.ndarray,
    lumbar_neutral: np.ndarray | None = None,
) -> list[int]:
    """Give the joints that have nothing to fit against a constant, from a reference pose.

    A leaf has no descendant offset, so no amount of position evidence can place its body: it
    keeps whatever constant its nearest fitted ancestor got, and that is right only when the two
    source bodies share a frame. OpenSim's do -- every pair measures 0.000 deg apart at zero
    coordinates -- so this never mattered until a source arrived whose foot frame sits 91 deg
    from its shank's.

    Given a configuration in which the joint is known to sit at SMPL's rest angle, the constant
    is the one making its local rotation identity there, so it is read off the *parent's* global
    in that configuration rather than off the ancestor whose constant it happened to inherit --
    those differ whenever the chain between them turns, which for the head it does, through the
    lumbar split. The globals are rebuilt after each joint because a corrected joint can be
    another's parent.

    Supplying a configuration is the caller asserting those joints are neutral in it: true of a
    head in a standing trial, not true of every joint in every trial, which is why this is a
    parameter and not something inferred.
    """
    single = {name: _reference_frame(reference, name)[None] for name in reference}
    placed = []
    for joint in sorted(joints):
        body = available.get(joint)
        if joint == _ROOT or body is None or body not in single:
            continue
        globals_ = _global_rotations(single, alignment, world, available, provenance, 1,
                                     lumbar_neutral)
        parent = int(SMPL24_PARENTS[joint])
        alignment[joint] = single[body][0].T @ world.T @ globals_[0, parent]
        placed.append(joint)
    return placed


def _alignment_from_offsets(
    rest_offsets: list[np.ndarray],
    measured_offsets: list[np.ndarray],
    base: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """The constant carrying SMPL's rest offsets onto the source's, and how many pinned it.

    `base` is where to start -- the parent's constant. It matters whenever the evidence is a
    single direction, which is most of the skeleton: one child pins two axes and leaves the
    twist about that bone free, so *something* has to choose it. Starting from scratch picks an
    arbitrary twist per bone; starting from the parent turns only as far as the measurement
    demands, so the whole chain inherits one reference from the pelvis, where three children
    determine it properly. It also stops a two-point fit -- spine3 from the shoulders -- from
    discarding the up-down information it has no evidence about.
    """
    if not rest_offsets:
        return (np.eye(3) if base is None else base), 0
    start = np.eye(3) if base is None else np.asarray(base, dtype=np.float64)
    rest = np.asarray(rest_offsets, dtype=np.float64) @ start.T
    measured = np.asarray(measured_offsets, dtype=np.float64)
    correction, _ = Rotation.align_vectors(measured, rest)
    return correction.as_matrix() @ start, len(rest_offsets)


def _unit(vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _root_from_placement(
    placement: "RootPlacement",
    rest: np.ndarray,
    centres: Mapping[str, np.ndarray],
    root_rotation: np.ndarray,
    root_centre: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """The root's constant from directions, and its translation from a landmark it must hit.

    `_root_alignment` fits both at once out of the child offsets, and that is sound only when
    there are enough of them. When there are two -- which is what a source measuring nothing in
    the pelvis but the two hips gives -- the fit is not merely weak, it is wrong in a specific
    way: the two skeletons put their pelvis origin at different points, and the only way a
    rotation can answer that displacement is to turn about the axis joining the hips. That axis
    is exactly the one the hips carry no evidence about, so nothing in the fit pushes back, and
    the body pitches. Measured on this cohort's static trials the pitch is 23 deg, and the
    lumbar rule then carries it into the trunk, the head and both arms.

    Here the two questions are asked separately and of evidence that can answer them.

      constant     from directions only, each normalised first. A direction cannot express a
                   displacement, so the failure above cannot recur, and normalising stops a
                   segment whose length the two skeletons disagree about -- the hip width, off
                   by 100 mm on this cohort -- from leaning the fit toward the longer vector.
                   Two non-parallel pairs determine all three axes.
      translation  closed form, once the constant is known: put the centroid of the named SMPL
                   joints on the centroid of the landmarks they correspond to. Both are rigid
                   in the root body, so the per-frame value is a constant and is averaged.

    That constant is what `root_offset` reports either way. When the placement also names
    `tracked` joints, `fit_pose` goes on to solve the translation per frame and emits that
    instead -- see the field's own note for why one anchor point is not enough on this source.
    """
    rest_dirs = _unit(placement.rest_directions)
    measured_dirs = _unit(placement.measured_directions)
    if len(rest_dirs) < 2:
        raise ValueError("a root placement needs at least two directions to fix all three axes")
    if len(rest_dirs) != len(measured_dirs):
        raise ValueError("a root placement's rest and measured directions must pair up")
    if root_centre not in centres:
        raise ValueError("the source has no pelvis centre; there is nothing to anchor to")
    missing = [n for n in placement.anchor_centres if n not in centres]
    if missing:
        raise ValueError(f"the root placement names centres the source does not have: {missing}")
    # The closed form below reads `p[j] = p[0] + G[0] @ (rest[j] - rest[0])`, which is forward
    # kinematics only for the root's own children. Anything further out turns at a joint in
    # between and would land somewhere else entirely.
    outside = [j for j in placement.anchor_joints if int(SMPL24_PARENTS[j]) != _ROOT]
    if outside:
        raise ValueError(f"root placement anchor joints must be children of the root: {outside}")
    rotation, _ = Rotation.align_vectors(measured_dirs, rest_dirs)
    matrix = rotation.as_matrix()

    held = np.mean([rest[j] - rest[_ROOT] for j in placement.anchor_joints], axis=0)
    target = np.mean(
        [np.asarray(centres[name], dtype=np.float64) for name in placement.anchor_centres],
        axis=0,
    )
    displacement = target - np.asarray(centres[root_centre], dtype=np.float64)
    in_body = np.einsum("tji,tj->ti", root_rotation, displacement).mean(axis=0)
    return matrix, in_body - matrix @ held, len(rest_dirs)


def _root_alignment(
    rest_offsets: list[np.ndarray], measured_offsets: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Rotation *and* offset for the root, because the two skeletons anchor it differently.

    SMPL's root sits directly above the hips; OpenSim's pelvis origin sits forward of them. No
    rotation can reconcile a displacement, and forcing one to try tips the whole body backwards
    -- the spine and both arms then leave by 300 mm while the legs look merely bad. The root is
    the one joint where a translation is free, since `trans` is an output, so it is fitted here
    and recorded as `root_offset`.
    """
    if len(rest_offsets) < 3:
        rotation, count = _alignment_from_offsets(rest_offsets, measured_offsets)
        return rotation, np.zeros(3), count
    rest = np.asarray(rest_offsets, dtype=np.float64)
    measured = np.asarray(measured_offsets, dtype=np.float64)
    rest_centre = rest.mean(axis=0)
    measured_centre = measured.mean(axis=0)
    rotation, _ = Rotation.align_vectors(measured - measured_centre, rest - rest_centre)
    matrix = rotation.as_matrix()
    return matrix, measured_centre - matrix @ rest_centre, len(rest)


def _targets_of(joint: int, provenance, centre_of, body_of) -> list[int]:
    """Joints with a centre whose position this joint's constant decides.

    Children qualify, and so does anything further out reached only through joints the source
    does not measure, since those hold rest and add no rotation of their own.

    Reaching further than that was tried and is worse. Letting the shoulders align the trunk --
    through the three spine joints and the collars -- looks compelling, because the trunk's
    orientation is otherwise never measured. But SMPL does not place spine1 at the source's
    lumbar centre; it places it by the chain from the pelvis, some 29 mm away. Aiming from a
    point the model does not occupy, and then rotating the spine that carries the shoulders, is
    circular: the shoulders moved from 130 mm to 250 mm off. The trunk keeps the pelvis's
    constant, and the residual is reported rather than optimised against itself.
    """
    found: list[int] = []
    frontier = [c for c, p in enumerate(SMPL24_PARENTS) if p == joint]
    while frontier:
        candidate = frontier.pop()
        if candidate in centre_of:
            found.append(candidate)
        elif provenance[candidate] == ABSENT:
            frontier.extend(c for c, p in enumerate(SMPL24_PARENTS) if p == candidate)
    return sorted(found)


def fit_pose(
    *,
    rest_joints,
    correspondence: tuple[SmplJointSource, ...],
    body_of: Mapping[int, str],
    source_rotations: Mapping[str, np.ndarray],
    source_centres: Mapping[str, np.ndarray],
    world_rotation,
    reference_rotations: Mapping[str, np.ndarray] | None = None,
    root_placement: RootPlacement | None = None,
    lumbar_from_reference: bool = False,
) -> PoseFit:
    """Build SMPL pose and translation for one trial.

    `body_of` maps an SMPL joint to the source body whose rotation drives it -- the *distal*
    body of the joint's chain, because that is the segment SMPL spans. `source_centres` and
    `source_rotations` are in the source's own frame; `world_rotation` turns the finished result
    into the corpus frame, applied once.

    `reference_rotations` is optional and, when given, places the joints that have no offset to
    fit against -- leaves, and anything whose descendants carry no centre. Without it those
    joints keep an ancestor's constant, which is right only when the two source bodies share a
    frame. Omitting it reproduces the earlier behaviour exactly, number for number.

    `root_placement` is optional in the same way, for a sharper reason: a source whose only root
    children with a centre are the two hips cannot tell the skeletons' disagreement about where
    the pelvis origin *is* from a rotation, and pays for it in pitch. Supplying it takes the
    root's constant from directions and its translation from a landmark it must hit. Omitting it
    also reproduces the earlier behaviour exactly.

    `lumbar_from_reference` decides what the lumbar rule treats as zero. Off, the trunk's turn
    is the raw pelvis-to-trunk frame difference, which is right only where the source's two
    segment frames coincide when the subject is neutral -- true of OpenSim, whose coordinates
    are zero at the model's neutral pose, and false of a marker-driven Visual3D model, where it
    leaves each subject a constant bend of their own. On, the same difference is measured
    against the reference configuration. Off reproduces the earlier behaviour exactly.
    """
    rest = np.asarray(rest_joints, dtype=np.float64)
    world = np.asarray(world_rotation, dtype=np.float64)
    entries = {entry.smpl_index: entry for entry in correspondence}

    available = {
        index: body_of[index]
        for index in body_of
        if body_of[index] in source_rotations
    }
    frames = len(next(iter(source_rotations.values())))

    provenance = np.array(
        [
            entries[i].provenance if (i in available or i in _LUMBAR_SMPL) else ABSENT
            for i in range(24)
        ],
        dtype=object,
    )
    # the spine is only derived if the trunk was actually measured
    if _TRUNK_BODY not in source_rotations:
        for index in _LUMBAR_SMPL:
            provenance[index] = ABSENT

    centres = {name: np.asarray(v, dtype=np.float64) for name, v in source_centres.items()}
    centre_of = {
        index: entries[index].centre_joint
        for index in range(24)
        if entries[index].centre_joint in centres
    }

    # ---- one constant per bone -------------------------------------------------------
    alignment = np.tile(np.eye(3), (24, 1, 1))
    root_offset = np.zeros(3)
    pinned_by: dict[int, int] = {}
    # Which body each joint's constant was actually fitted for. A joint that inherits a constant
    # inherits this too, and it is the only way to tell afterwards that the constant belongs to
    # a different segment than the one it is being applied to.
    held_for: dict[int, str] = {}
    #: joints with a body but no evidence to fit against, whose constant belongs to another body
    unfitted: list[int] = []
    # SMPL's tree lists every parent before its children, so one pass is tree order. Each
    # joint starts from its parent's constant *before* being refined -- deferring that to a
    # second pass leaves the spine at identity while spine3 is being fitted off it.
    for joint in range(24):
        if joint != _ROOT:
            parent = int(SMPL24_PARENTS[joint])
            alignment[joint] = alignment[parent]
            if parent in held_for:
                held_for[joint] = held_for[parent]
        body = available.get(joint)
        if body is None:
            continue
        if joint == _ROOT and root_placement is not None:
            alignment[joint], root_offset, pinned_by[joint] = _root_from_placement(
                root_placement, rest, centres, source_rotations[body],
                entries[_ROOT].centre_joint,
            )
            held_for[joint] = body
            continue
        targets = _targets_of(joint, provenance, centre_of, available)
        # Away from the root, drop children the source does not measure: SMPL's spine1 and
        # OpenSim's `back` are not the same landmark, and a rotation asked to hit both it and
        # the hips compromises on all three. At the root the free translation absorbs that
        # difference, so every child can be used there.
        if joint != _ROOT:
            targets = [t for t in targets if provenance[t] == MEASURED]

        if not targets or joint not in centre_of:
            # Nothing to fit against: no descendant offset, or no centre of its own. The joint
            # keeps its ancestor's constant -- see _targets_of for why reaching further for the
            # trunk makes the arms worse rather than better. Recorded so that a reference
            # configuration, if the caller gave one, can place it afterwards.
            if held_for.get(joint) != body:
                unfitted.append(joint)
            continue

        rotation = np.einsum("tji->tij", source_rotations[body])
        rest_points = [rest[t] - rest[joint] for t in targets]
        measured_points = [
            np.einsum("tij,tj->ti", rotation,
                      centres[centre_of[t]] - centres[centre_of[joint]]).mean(axis=0)
            for t in targets
        ]
        if joint == _ROOT:
            alignment[joint], root_offset, pinned_by[joint] = _root_alignment(
                rest_points, measured_points
            )
        else:
            alignment[joint], pinned_by[joint] = _alignment_from_offsets(
                rest_points, measured_points, alignment[int(SMPL24_PARENTS[joint])]
            )
        held_for[joint] = body

    # The trunk's neutral, if the caller asked for one. Read before anything uses the lumbar
    # rule, because the reference pass below builds a pose with it too.
    lumbar_neutral = None
    if lumbar_from_reference:
        if reference_rotations is None:
            raise ValueError("lumbar_from_reference needs a reference configuration to read")
        if _TRUNK_BODY in reference_rotations and _ROOT in available:
            lumbar_neutral = (
                _reference_frame(reference_rotations, available[_ROOT]).T
                @ _reference_frame(reference_rotations, _TRUNK_BODY)
            )

    placed_from_reference: list[int] = []
    if reference_rotations is not None and unfitted:
        placed_from_reference = _place_from_reference(
            unfitted, reference_rotations, alignment, world, available, provenance,
            lumbar_neutral,
        )

    # ---- global rotations ------------------------------------------------------------
    globals_ = _global_rotations(
        source_rotations, alignment, world, available, provenance, frames, lumbar_neutral
    )

    # ---- local rotations and translation ---------------------------------------------
    local = np.empty_like(globals_)
    local[:, 0] = globals_[:, 0]
    for joint in range(1, 24):
        parent = int(SMPL24_PARENTS[joint])
        local[:, joint] = np.einsum(
            "tji,tjk->tik", globals_[:, parent], globals_[:, joint]
        )
    pose = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(frames, 24, 3)

    root_name = entries[_ROOT].centre_joint
    if root_name not in centres:
        raise ValueError("the source has no pelvis centre; there is nothing to anchor to")
    root_body = available.get(_ROOT)
    anchored = centres[root_name]
    if root_body is not None and np.linalg.norm(root_offset):
        anchored = anchored + np.einsum(
            "tij,j->ti", source_rotations[root_body], root_offset
        )
    trans = anchored @ world.T

    if root_placement is not None and root_placement.tracked:
        # The translation half of a per-frame IK, in closed form. `smpl_world_positions` with a
        # zero translation is the body reconstructed about its own root, so what is left is one
        # offset per frame, and the least-squares one is the mean residual over the tracked
        # joints. The constant solved above is still reported as `root_offset` -- it is where
        # the body sits on average, and it is what a reader comparing the two skeletons wants --
        # but the emitted `trans` is this.
        missing = [name for _, name in root_placement.tracked if name not in centres]
        if missing:
            raise ValueError(f"the root placement tracks centres the source lacks: {missing}")
        about_root = smpl_world_positions(rest, pose, np.zeros((frames, 3)))
        residual = np.stack(
            [centres[name] @ world.T - about_root[:, joint]
             for joint, name in root_placement.tracked],
            axis=1,
        )
        trans = residual.mean(axis=1)

    # ---- what it cost ----------------------------------------------------------------
    reproduced = smpl_world_positions(rest, pose, trans)
    per_joint: dict[int, float] = {}
    for joint, name in centre_of.items():
        # The root is excluded: SMPL's root is deliberately *not* the source's pelvis origin,
        # and the distance between them is `root_offset`, not an error. Reporting it as one
        # would inflate the number with a quantity the fit chose on purpose.
        if joint == _ROOT:
            continue
        target = centres[name] @ world.T
        per_joint[joint] = float(
            np.linalg.norm(reproduced[:, joint] - target, axis=1).max()
        )
    worst = max(per_joint.values()) if per_joint else 0.0

    quaternions = Rotation.from_matrix(alignment).as_quat()
    rest_alignment = np.concatenate(
        (quaternions[:, 3:4], quaternions[:, :3]), axis=1
    )

    return PoseFit(
        pose=pose,
        trans=trans,
        rest_alignment=rest_alignment,
        joint_provenance=np.array(
            [str(p) for p in provenance], dtype=np.dtype("U16")
        ),
        position_error_m=worst,
        per_joint_error_m=per_joint,
        alignment_children=pinned_by,
        root_offset=root_offset,
        placed_from_reference=tuple(placed_from_reference),
    )


def joint_name(index: int) -> str:
    return SMPL24_NAMES[index]
