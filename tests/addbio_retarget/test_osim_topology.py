"""The two index maps that every downstream number depends on.

`.b3d` frames carry `pos` and `world_frame_joint_centers` as bare arrays with no names.
Getting either order wrong produces a plausible-looking skeleton that is silently wrong, so
the orders are derived from the model rather than hardcoded, and pinned here.

Both rules were established against the real cohort:
  * `pos`                        = model coordinate order minus constraint-dependent coordinates
  * `world_frame_joint_centers`  = model joint order minus joints whose coordinates are all
                                   constraint-dependent
"""

import pytest

from soma_synth.addbio_retarget.osim_topology import parse_osim

# A miniature Rajagopal: a floating pelvis, a hip, the coupled walker knee + patellofemoral
# pair, and a pin ankle. Same serialisation shape as the models embedded in .b3d headers --
# the function is a direct child carrying its concrete type as the tag, not a <function>
# wrapper.
MINI_OSIM = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40500">
  <Model name="mini">
    <JointSet>
      <objects>
        <CustomJoint name="ground_pelvis">
          <socket_parent_frame>/ground</socket_parent_frame>
          <socket_child_frame>/bodyset/pelvis</socket_child_frame>
          <coordinates>
            <Coordinate name="pelvis_tilt"/>
            <Coordinate name="pelvis_tx"/>
          </coordinates>
          <SpatialTransform>
            <TransformAxis name="rotation1">
              <coordinates>pelvis_tilt</coordinates>
              <axis>0 0 1</axis>
              <LinearFunction/>
            </TransformAxis>
            <TransformAxis name="translation1">
              <coordinates>pelvis_tx</coordinates>
              <axis>1 0 0</axis>
              <LinearFunction/>
            </TransformAxis>
          </SpatialTransform>
        </CustomJoint>
        <CustomJoint name="hip_r">
          <socket_parent_frame>/bodyset/pelvis</socket_parent_frame>
          <socket_child_frame>/bodyset/femur_r</socket_child_frame>
          <coordinates>
            <Coordinate name="hip_flexion_r"/>
          </coordinates>
          <SpatialTransform>
            <TransformAxis name="rotation1">
              <coordinates>hip_flexion_r</coordinates>
              <axis>0 0 1</axis>
              <LinearFunction/>
            </TransformAxis>
            <TransformAxis name="translation1">
              <coordinates></coordinates>
              <axis>1 0 0</axis>
              <Constant/>
            </TransformAxis>
          </SpatialTransform>
        </CustomJoint>
        <CustomJoint name="walker_knee_r">
          <socket_parent_frame>/bodyset/femur_r</socket_parent_frame>
          <socket_child_frame>/bodyset/tibia_r</socket_child_frame>
          <coordinates>
            <Coordinate name="knee_angle_r"/>
          </coordinates>
          <SpatialTransform>
            <TransformAxis name="rotation1">
              <coordinates>knee_angle_r</coordinates>
              <axis>0 0 1</axis>
              <LinearFunction/>
            </TransformAxis>
            <TransformAxis name="translation1">
              <coordinates>knee_angle_r</coordinates>
              <axis>1 0 0</axis>
              <MultiplierFunction/>
            </TransformAxis>
          </SpatialTransform>
        </CustomJoint>
        <CustomJoint name="patellofemoral_r">
          <socket_parent_frame>/bodyset/femur_r</socket_parent_frame>
          <socket_child_frame>/bodyset/patella_r</socket_child_frame>
          <coordinates>
            <Coordinate name="knee_angle_r_beta"/>
          </coordinates>
          <SpatialTransform>
            <TransformAxis name="rotation1">
              <coordinates>knee_angle_r_beta</coordinates>
              <axis>0 0 1</axis>
              <LinearFunction/>
            </TransformAxis>
          </SpatialTransform>
        </CustomJoint>
        <PinJoint name="ankle_r">
          <socket_parent_frame>/bodyset/tibia_r</socket_parent_frame>
          <socket_child_frame>/bodyset/talus_r</socket_child_frame>
          <coordinates>
            <Coordinate name="ankle_angle_r"/>
          </coordinates>
        </PinJoint>
      </objects>
    </JointSet>
    <ConstraintSet>
      <objects>
        <CoordinateCouplerConstraint name="patellofemoral_knee_angle_r_con">
          <dependent_coordinate_name>knee_angle_r_beta</dependent_coordinate_name>
          <independent_coordinate_names>knee_angle_r</independent_coordinate_names>
        </CoordinateCouplerConstraint>
      </objects>
    </ConstraintSet>
  </Model>
</OpenSimDocument>
"""


@pytest.fixture(scope="module")
def topo():
    return parse_osim(MINI_OSIM)


def test_joint_order_follows_the_model(topo):
    assert [j.name for j in topo.joints] == [
        "ground_pelvis", "hip_r", "walker_knee_r", "patellofemoral_r", "ankle_r",
    ]


def test_bodies_resolve_through_the_socket_paths(topo):
    by_name = {j.name: j for j in topo.joints}
    assert by_name["ground_pelvis"].parent_body == "ground"
    assert by_name["ground_pelvis"].child_body == "pelvis"
    assert by_name["walker_knee_r"].parent_body == "femur_r"
    assert by_name["walker_knee_r"].child_body == "tibia_r"


def test_dependent_coordinates_come_from_the_constraint_set(topo):
    assert topo.dependent_coordinates == ("knee_angle_r_beta",)


def test_pos_order_drops_the_dependent_coordinate(topo):
    assert topo.coordinate_names == (
        "pelvis_tilt", "pelvis_tx", "hip_flexion_r",
        "knee_angle_r", "knee_angle_r_beta", "ankle_angle_r",
    )
    assert topo.independent_coordinate_names == (
        "pelvis_tilt", "pelvis_tx", "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
    )


def test_joint_centre_order_drops_the_fully_dependent_joint(topo):
    assert topo.joint_centre_names == (
        "ground_pelvis", "hip_r", "walker_knee_r", "ankle_r",
    )


def test_a_driven_nonconstant_translation_axis_marks_the_joint_as_translating(topo):
    by_name = {j.name: j for j in topo.joints}
    # the walker knee slides along the femur; hip and ankle do not
    assert by_name["walker_knee_r"].translates is True
    assert by_name["hip_r"].translates is False
    assert by_name["ankle_r"].translates is False
    assert by_name["ground_pelvis"].translates is True


def test_rigidity_prediction_matches_the_topology(topo):
    """Two centres hold a constant distance iff they are fixed in a common body.

    This is the prediction the cohort measurement confirmed with zero violations; keeping it
    here means a regression in `translates` detection shows up as a rigidity claim, which is
    what the retarget actually relies on.
    """
    rigid = topo.rigid_centre_pairs()
    names = topo.joint_centre_names
    idx = {n: i for i, n in enumerate(names)}
    # hip and knee share the femur, but the knee slides along it -> not rigid
    assert not rigid[idx["hip_r"]][idx["walker_knee_r"]]
    # knee and ankle are both fixed in the tibia -> rigid
    assert rigid[idx["walker_knee_r"]][idx["ankle_r"]]
    # pelvis root and hip are both fixed in the pelvis -> rigid
    assert rigid[idx["ground_pelvis"]][idx["hip_r"]]


def test_gravity_is_read_from_the_model_not_assumed(topo):
    """Which way is up is a statement the file makes; absent, it is not guessed."""
    assert topo.gravity is None
    with_gravity = MINI_OSIM.replace(
        '<Model name="mini">', '<Model name="mini">\n    <gravity>0 -9.80665 0</gravity>'
    )
    assert parse_osim(with_gravity).gravity == pytest.approx((0.0, -9.80665, 0.0))


def test_malformed_gravity_is_reported_as_absent_rather_than_half_read():
    for bad in ("0 -9.80665", "", "up"):
        broken = MINI_OSIM.replace(
            '<Model name="mini">', f'<Model name="mini">\n    <gravity>{bad}</gravity>'
        )
        assert parse_osim(broken).gravity is None


def test_a_model_without_a_constraint_set_keeps_every_coordinate():
    stripped = MINI_OSIM.replace("knee_angle_r_beta", "knee_angle_r_beta_x")
    topo2 = parse_osim(stripped)
    # the constraint now names a coordinate that still exists, so nothing else changes;
    # removing the ConstraintSet entirely must keep all coordinates independent
    no_constraints = MINI_OSIM[: MINI_OSIM.index("<ConstraintSet>")] + "</Model>\n</OpenSimDocument>\n"
    topo3 = parse_osim(no_constraints)
    assert topo3.dependent_coordinates == ()
    assert topo3.independent_coordinate_names == topo3.coordinate_names
    assert len(topo3.joint_centre_names) == 5
    assert topo2.dependent_coordinates == ("knee_angle_r_beta_x",)
