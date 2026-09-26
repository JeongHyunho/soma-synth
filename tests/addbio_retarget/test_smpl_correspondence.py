"""Which SMPL joint each OpenSim joint feeds, and which ones nothing feeds.

The design's correspondence table decides what the artifact claims to have measured. Written as
prose it can drift from the model; written as a lookup keyed on variant it breaks on the first
cohort whose skeleton differs. So it is derived: the table names OpenSim joints, and whichever
of them the file's own model actually declares becomes `measured`. Everything else falls to
`absent` on its own, which is correct and needs no per-variant branch.
"""

import importlib.util
import pathlib

import pytest

from soma_synth.addbio_retarget.osim_topology import parse_osim
from soma_synth.addbio_retarget.smpl_correspondence import (
    ABSENT,
    DERIVED_LUMBAR,
    MEASURED,
    SMPL24_NAMES,
    SMPL24_PARENTS,
    correspondence_for,
)

REPO = pathlib.Path(__file__).resolve().parents[2]


def _osim(joints, constraints=()):
    """Assemble a model declaring exactly the named joints, coordinates and bodies."""
    body = []
    for name, coordinates in joints:
        listed = "".join(f'<Coordinate name="{c}"/>' for c in coordinates)
        parent_body, child_body = BODIES[name]
        body.append(
            f'<CustomJoint name="{name}">'
            f"<socket_parent_frame>/bodyset/{parent_body}</socket_parent_frame>"
            f"<socket_child_frame>/bodyset/{child_body}</socket_child_frame>"
            f"<coordinates>{listed}</coordinates>"
            f"</CustomJoint>"
        )
    coupled = "".join(
        f'<CoordinateCouplerConstraint name="c{i}">'
        f"<dependent_coordinate_name>{d}</dependent_coordinate_name>"
        f"</CoordinateCouplerConstraint>"
        for i, d in enumerate(constraints)
    )
    return (
        '<?xml version="1.0"?><OpenSimDocument Version="40500"><Model name="m">'
        f"<JointSet><objects>{''.join(body)}</objects></JointSet>"
        f"<ConstraintSet><objects>{coupled}</objects></ConstraintSet>"
        "</Model></OpenSimDocument>"
    )


# The Rajagopal body chain, so the "distal body of the chain" rule has something to be wrong
# about: SMPL's foot spans ankle *and* subtalar, ending at the calcaneus rather than the talus.
BODIES = {
    "ground_pelvis": ("ground", "pelvis"),
    "hip_r": ("pelvis", "femur_r"), "walker_knee_r": ("femur_r", "tibia_r"),
    "patellofemoral_r": ("femur_r", "patella_r"), "ankle_r": ("tibia_r", "talus_r"),
    "subtalar_r": ("talus_r", "calcn_r"), "mtp_r": ("calcn_r", "toes_r"),
    "hip_l": ("pelvis", "femur_l"), "walker_knee_l": ("femur_l", "tibia_l"),
    "patellofemoral_l": ("femur_l", "patella_l"), "ankle_l": ("tibia_l", "talus_l"),
    "subtalar_l": ("talus_l", "calcn_l"), "mtp_l": ("calcn_l", "toes_l"),
    "back": ("pelvis", "torso"),
    "acromial_r": ("torso", "humerus_r"), "elbow_r": ("humerus_r", "ulna_r"),
    "radioulnar_r": ("ulna_r", "radius_r"), "radius_hand_r": ("radius_r", "hand_r"),
    "acromial_l": ("torso", "humerus_l"), "elbow_l": ("humerus_l", "ulna_l"),
    "radioulnar_l": ("ulna_l", "radius_l"), "radius_hand_l": ("radius_l", "hand_l"),
}

LOWER_BODY = [
    ("ground_pelvis", ["pelvis_tilt", "pelvis_list", "pelvis_rotation",
                       "pelvis_tx", "pelvis_ty", "pelvis_tz"]),
    ("hip_r", ["hip_flexion_r", "hip_adduction_r", "hip_rotation_r"]),
    ("walker_knee_r", ["knee_angle_r"]),
    ("patellofemoral_r", ["knee_angle_r_beta"]),
    ("ankle_r", ["ankle_angle_r"]),
    ("subtalar_r", ["subtalar_angle_r"]),
    ("mtp_r", ["mtp_angle_r"]),
    ("hip_l", ["hip_flexion_l", "hip_adduction_l", "hip_rotation_l"]),
    ("walker_knee_l", ["knee_angle_l"]),
    ("patellofemoral_l", ["knee_angle_l_beta"]),
    ("ankle_l", ["ankle_angle_l"]),
    ("subtalar_l", ["subtalar_angle_l"]),
    ("mtp_l", ["mtp_angle_l"]),
    ("back", ["lumbar_extension", "lumbar_bending", "lumbar_rotation"]),
]

ARMS = [
    ("acromial_r", ["arm_flex_r", "arm_add_r", "arm_rot_r"]),
    ("elbow_r", ["elbow_flex_r"]),
    ("radioulnar_r", ["pro_sup_r"]),
    ("radius_hand_r", ["wrist_flex_r", "wrist_dev_r"]),
    ("acromial_l", ["arm_flex_l", "arm_add_l", "arm_rot_l"]),
    ("elbow_l", ["elbow_flex_l"]),
    ("radioulnar_l", ["pro_sup_l"]),
    ("radius_hand_l", ["wrist_flex_l", "wrist_dev_l"]),
]

BETA = ["knee_angle_r_beta", "knee_angle_l_beta"]


@pytest.fixture(scope="module")
def no_arm():
    return correspondence_for(parse_osim(_osim(LOWER_BODY, BETA)))


@pytest.fixture(scope="module")
def with_arm():
    return correspondence_for(parse_osim(_osim(LOWER_BODY + ARMS, BETA)))


def test_the_smpl_tree_matches_the_one_the_synthesis_code_already_uses():
    """Two copies of the kinematic tree is one too many; this catches them diverging."""
    path = REPO / "scripts/poc/anthro_smpl.py"
    spec = importlib.util.spec_from_file_location("_anthro_smpl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert list(module.SMPL24_PARENTS) == list(SMPL24_PARENTS)
    assert list(module.JOINT24_NAMES) == list(SMPL24_NAMES)


def test_every_smpl_joint_gets_exactly_one_entry(no_arm, with_arm):
    for table in (no_arm, with_arm):
        assert [entry.smpl_index for entry in table] == list(range(24))
        assert [entry.smpl_name for entry in table] == list(SMPL24_NAMES)


def test_the_variant_counts_are_what_the_design_claims(no_arm, with_arm):
    def tally(table, kind):
        return sum(1 for entry in table if entry.provenance == kind)

    assert (tally(with_arm, MEASURED), tally(with_arm, DERIVED_LUMBAR),
            tally(with_arm, ABSENT)) == (15, 3, 6)
    assert (tally(no_arm, MEASURED), tally(no_arm, DERIVED_LUMBAR),
            tally(no_arm, ABSENT)) == (9, 3, 12)


def test_arms_are_measured_only_where_the_model_has_them(no_arm, with_arm):
    arm_joints = [16, 17, 18, 19, 20, 21]
    assert all(with_arm[i].provenance == MEASURED for i in arm_joints)
    assert all(no_arm[i].provenance == ABSENT for i in arm_joints)
    # nothing was invented for the variant that lacks them
    assert all(no_arm[i].source_coordinates == () for i in arm_joints)
    assert all(no_arm[i].centre_joint is None for i in arm_joints)


def test_the_upper_body_is_absent_in_both_variants(no_arm, with_arm):
    """OpenSim's trunk is one rigid torso; there is no neck, collar or hand to read."""
    for index in (12, 13, 14, 15, 22, 23):
        assert no_arm[index].provenance == ABSENT
        assert with_arm[index].provenance == ABSENT


def test_the_lumbar_is_shared_three_ways_and_says_so(with_arm):
    lumbar = ("lumbar_extension", "lumbar_bending", "lumbar_rotation")
    for index in (3, 6, 9):
        assert with_arm[index].provenance == DERIVED_LUMBAR
        assert with_arm[index].source_coordinates == lumbar
    # only the lowest of the three has a measured centre to aim at
    assert with_arm[3].centre_joint == "back"
    assert with_arm[6].centre_joint is None
    assert with_arm[9].centre_joint is None


def test_left_and_right_are_not_swapped(with_arm):
    assert with_arm[1].centre_joint == "hip_l"
    assert with_arm[2].centre_joint == "hip_r"
    assert with_arm[4].centre_joint == "walker_knee_l"
    assert with_arm[5].centre_joint == "walker_knee_r"
    assert with_arm[16].centre_joint == "acromial_l"
    assert with_arm[17].centre_joint == "acromial_r"
    for entry in with_arm:
        if entry.smpl_name.startswith("left"):
            assert all(not c.endswith("_r") for c in entry.source_coordinates)
        if entry.smpl_name.startswith("right"):
            assert all(not c.endswith("_l") for c in entry.source_coordinates)


def test_a_joint_that_swallows_two_opensim_joints_keeps_both_coordinates(with_arm):
    assert with_arm[7].source_coordinates == ("ankle_angle_l", "subtalar_angle_l")
    assert with_arm[7].centre_joint == "ankle_l"          # the proximal one carries the centre
    assert with_arm[18].source_coordinates == ("elbow_flex_l", "pro_sup_l")
    assert with_arm[18].centre_joint == "elbow_l"


def test_the_driving_body_is_the_distal_one_of_the_chain(with_arm, no_arm):
    """SMPL's foot spans ankle and subtalar; stopping at the talus loses the subtalar rotation."""
    assert with_arm[7].source_body == "calcn_l"       # not talus_l
    assert with_arm[8].source_body == "calcn_r"
    assert with_arm[18].source_body == "radius_l"     # not ulna_l
    assert with_arm[1].source_body == "femur_l"
    assert with_arm[4].source_body == "tibia_l"
    assert with_arm[0].source_body == "pelvis"
    assert with_arm[3].source_body == "torso"
    assert with_arm[12].source_body is None           # neck: nothing drives it
    assert no_arm[16].source_body is None             # no shoulder in this variant


def test_the_dependent_knee_coordinate_never_reaches_the_artifact(with_arm):
    every = [c for entry in with_arm for c in entry.source_coordinates]
    assert "knee_angle_r_beta" not in every
    assert "knee_angle_l_beta" not in every
    assert with_arm[5].source_coordinates == ("knee_angle_r",)


def test_pelvis_translation_is_not_reported_as_a_joint_rotation(with_arm):
    assert with_arm[0].source_coordinates == (
        "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    )


def test_each_stored_centre_feeds_at_most_one_smpl_joint(with_arm, no_arm):
    """Four centres are deliberately unused as targets: the subtalars and the radioulnars.

    They sit part-way along a segment SMPL models with a single joint, so aiming an SMPL
    joint at them would pull the skeleton toward a landmark it does not have. Their angles
    are still consumed -- as rotation, on the joint that swallows them.
    """
    targets = [e.centre_joint for e in with_arm if e.centre_joint]
    assert len(targets) == len(set(targets)), "a centre must feed at most one SMPL joint"
    assert len(targets) == 16                   # 20 stored centres - 4 intentionally unused
    assert len([e for e in no_arm if e.centre_joint]) == 10      # 12 - 2 subtalars
    unused = {"subtalar_l", "subtalar_r", "radioulnar_l", "radioulnar_r"}
    assert unused.isdisjoint(targets)


def test_a_model_missing_a_leg_joint_degrades_rather_than_raising():
    partial = [j for j in LOWER_BODY if j[0] != "mtp_l"]
    table = correspondence_for(parse_osim(_osim(partial, BETA)))
    assert table[10].provenance == ABSENT          # left_foot
    assert table[11].provenance == MEASURED        # right_foot untouched
