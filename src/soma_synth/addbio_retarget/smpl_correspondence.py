"""Which OpenSim joints feed which SMPL joint, and which SMPL joints nothing feeds.

The correspondence is written once, in terms of OpenSim joint names, and then *intersected with
the model the file actually declares*. A No_Arm subject is not a special case handled by a
branch: its model simply has no `acromial_l`, so SMPL's left shoulder comes out `absent` by the
same rule that leaves the neck absent in every variant. Adding a cohort with a different
skeleton degrades the same way instead of raising or, worse, silently shifting the table.

Three provenances, and the artifact carries one per joint:

  measured        an OpenSim joint the model declares supplies this rotation
  derived_lumbar  the three spine joints share the trunk's single 3-DOF `back`, split by rule
  absent          nothing in the source measures this; the rotation is left at rest as a marker

`centre_joint` is a *position* target -- the stored `world_frame_joint_centers` entry an SMPL
joint should be fitted to. It is not the same question as provenance: SMPL's spine1 has a
measured centre (the `back` joint origin) while its rotation is still only derived.
"""

from __future__ import annotations

from dataclasses import dataclass

from .osim_topology import OsimTopology

__all__ = [
    "ABSENT",
    "DERIVED_LUMBAR",
    "MEASURED",
    "SMPL24_NAMES",
    "SMPL24_PARENTS",
    "SmplJointSource",
    "correspondence_for",
]

MEASURED = "measured"
DERIVED_LUMBAR = "derived_lumbar"
ABSENT = "absent"

#: SMPL-24 kinematic tree, mirroring scripts/poc/anthro_smpl.py (kept in step by a test).
SMPL24_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
)

SMPL24_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)

#: SMPL joint -> the OpenSim joints it consumes, proximal first. The first entry also supplies
#: the position target; later entries contribute rotation only, because they sit part-way along
#: a segment SMPL spans with one joint.
_CHAIN: dict[int, tuple[str, ...]] = {
    0: ("ground_pelvis",),
    1: ("hip_l",),
    2: ("hip_r",),
    4: ("walker_knee_l",),
    5: ("walker_knee_r",),
    7: ("ankle_l", "subtalar_l"),
    8: ("ankle_r", "subtalar_r"),
    10: ("mtp_l",),
    11: ("mtp_r",),
    16: ("acromial_l",),
    17: ("acromial_r",),
    18: ("elbow_l", "radioulnar_l"),
    19: ("elbow_r", "radioulnar_r"),
    20: ("radius_hand_l",),
    21: ("radius_hand_r",),
}

#: The three SMPL spine joints all read the trunk's single `back` joint; only the lowest of
#: them has a measured centre to aim at.
_LUMBAR_JOINT = "back"
_LUMBAR_SMPL = (3, 6, 9)

#: The root's translation coordinates describe where the pelvis is, not how it is turned. They
#: belong to `trans`, and reporting them among a joint's rotations would double-count them.
_ROOT_TRANSLATION = frozenset({"pelvis_tx", "pelvis_ty", "pelvis_tz"})


@dataclass(frozen=True)
class SmplJointSource:
    smpl_index: int
    smpl_name: str
    provenance: str
    centre_joint: str | None
    source_coordinates: tuple[str, ...]
    #: The body whose rotation drives this SMPL joint: the *distal* body of its chain, since
    #: that is the segment SMPL spans. For the ankle that is the calcaneus, past the subtalar,
    #: not the talus; taking the proximal body instead loses the subtalar's rotation entirely.
    source_body: str | None = None


def _rotation_coordinates(joint) -> tuple[str, ...]:
    return tuple(c for c in joint.coordinates if c not in _ROOT_TRANSLATION)


def correspondence_for(topology: OsimTopology) -> tuple[SmplJointSource, ...]:
    """Build the per-joint table for the skeleton this particular file declares."""
    declared = {joint.name: joint for joint in topology.centre_joints}
    dependent = set(topology.dependent_coordinates)

    table: list[SmplJointSource] = []
    for index, name in enumerate(SMPL24_NAMES):
        if index in _LUMBAR_SMPL:
            lumbar = declared.get(_LUMBAR_JOINT)
            if lumbar is None:
                table.append(SmplJointSource(index, name, ABSENT, None, ()))
                continue
            table.append(
                SmplJointSource(
                    smpl_index=index,
                    smpl_name=name,
                    provenance=DERIVED_LUMBAR,
                    # the trunk has one measured origin; the upper two spine joints are
                    # interpolated and have nothing of their own to be fitted to
                    centre_joint=_LUMBAR_JOINT if index == _LUMBAR_SMPL[0] else None,
                    source_coordinates=_rotation_coordinates(lumbar),
                    source_body=lumbar.child_body,
                )
            )
            continue

        chain = [declared[n] for n in _CHAIN.get(index, ()) if n in declared]
        if not chain:
            table.append(SmplJointSource(index, name, ABSENT, None, ()))
            continue

        coordinates = tuple(
            c
            for joint in chain
            for c in _rotation_coordinates(joint)
            if c not in dependent
        )
        table.append(
            SmplJointSource(
                smpl_index=index,
                smpl_name=name,
                provenance=MEASURED,
                centre_joint=chain[0].name,
                source_coordinates=coordinates,
                source_body=chain[-1].child_body,
            )
        )
    return tuple(table)
