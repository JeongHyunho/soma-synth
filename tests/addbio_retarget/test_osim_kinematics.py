"""Forward kinematics for the OpenSim model the .b3d carries, checked against hand arithmetic.

Positions do not need this -- AddBiomechanics stores the joint centres it computed. Rotations do:
a bone's twist about its own axis moves nothing, so it cannot be read off positions, and it is
exactly what a downstream IMU orientation is a relabel of.

Every piece here is a place a plausible wrong answer can hide: which order the three rotations
compose, whether a translation is expressed before or after them, whether an offset frame's
orientation is applied at all. So the fixtures are small enough to work out by hand, and a
cohort-wide check then confronts the whole thing with
`world_frame_joint_centers`, which is the same quantity computed by the software that wrote it.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from soma_synth.addbio_retarget.osim_kinematics import (
    UnsupportedJoint,
    evaluate_function,
    parse_kinematics,
)


def _document(joints: str) -> str:
    return (
        '<?xml version="1.0"?><OpenSimDocument Version="40500"><Model name="m">'
        f"<JointSet><objects>{joints}</objects></JointSet></Model></OpenSimDocument>"
    )


def _offset(name: str, body: str, translation="0 0 0", orientation="0 0 0") -> str:
    return (
        f'<PhysicalOffsetFrame name="{name}">'
        f"<socket_parent>/bodyset/{body}</socket_parent>"
        f"<translation>{translation}</translation>"
        f"<orientation>{orientation}</orientation>"
        f"</PhysicalOffsetFrame>"
    )


FREE_PELVIS = (
    '<CustomJoint name="ground_pelvis">'
    "<socket_parent_frame>ground_offset</socket_parent_frame>"
    "<socket_child_frame>pelvis_offset</socket_child_frame>"
    "<coordinates>"
    '<Coordinate name="pelvis_tilt"/><Coordinate name="pelvis_list"/>'
    '<Coordinate name="pelvis_rotation"/><Coordinate name="pelvis_tx"/>'
    '<Coordinate name="pelvis_ty"/><Coordinate name="pelvis_tz"/>'
    "</coordinates>"
    "<SpatialTransform>"
    '<TransformAxis name="rotation1"><coordinates>pelvis_tilt</coordinates>'
    "<axis>0 0 1</axis><LinearFunction><coefficients>1 0</coefficients></LinearFunction>"
    "</TransformAxis>"
    '<TransformAxis name="rotation2"><coordinates>pelvis_list</coordinates>'
    "<axis>1 0 0</axis><LinearFunction><coefficients>1 0</coefficients></LinearFunction>"
    "</TransformAxis>"
    '<TransformAxis name="rotation3"><coordinates>pelvis_rotation</coordinates>'
    "<axis>0 1 0</axis><LinearFunction><coefficients>1 0</coefficients></LinearFunction>"
    "</TransformAxis>"
    '<TransformAxis name="translation1"><coordinates>pelvis_tx</coordinates>'
    "<axis>1 0 0</axis><LinearFunction><coefficients>1 0</coefficients></LinearFunction>"
    "</TransformAxis>"
    '<TransformAxis name="translation2"><coordinates>pelvis_ty</coordinates>'
    "<axis>0 1 0</axis><LinearFunction><coefficients>1 0</coefficients></LinearFunction>"
    "</TransformAxis>"
    '<TransformAxis name="translation3"><coordinates>pelvis_tz</coordinates>'
    "<axis>0 0 1</axis><LinearFunction><coefficients>1 0</coefficients></LinearFunction>"
    "</TransformAxis>"
    "</SpatialTransform>"
    "<frames>"
    + _offset("ground_offset", "ground")
    + _offset("pelvis_offset", "pelvis")
    + "</frames></CustomJoint>"
)

PIN_HIP = (
    '<PinJoint name="hip_r">'
    "<socket_parent_frame>pelvis_offset</socket_parent_frame>"
    "<socket_child_frame>femur_r_offset</socket_child_frame>"
    '<coordinates><Coordinate name="hip_flexion_r"/></coordinates>'
    "<frames>"
    + _offset("pelvis_offset", "pelvis", translation="0.1 -0.05 0.07")
    + _offset("femur_r_offset", "femur_r")
    + "</frames></PinJoint>"
)


@pytest.fixture(scope="module")
def two_joint():
    return parse_kinematics(_document(FREE_PELVIS + PIN_HIP))


# ---------------------------------------------------------------- functions


def test_a_linear_function_is_slope_then_intercept():
    node = _parse_one('<LinearFunction><coefficients>2 0.5</coefficients></LinearFunction>')
    assert evaluate_function(node, 3.0) == pytest.approx(6.5)


def test_a_constant_ignores_its_input():
    node = _parse_one("<Constant><value>0.25</value></Constant>")
    assert evaluate_function(node, 99.0) == pytest.approx(0.25)


def test_a_spline_passes_through_its_knots():
    node = _parse_one(
        "<SimmSpline><x>0 1 2 3</x><y>0 2 1 4</y></SimmSpline>"
    )
    for x, y in zip([0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 1.0, 4.0]):
        assert evaluate_function(node, x) == pytest.approx(y, abs=1e-12)


def test_a_spline_is_smooth_between_knots_not_piecewise_linear():
    node = _parse_one("<SimmSpline><x>0 1 2 3</x><y>0 2 1 4</y></SimmSpline>")
    midpoint = evaluate_function(node, 1.5)
    assert midpoint != pytest.approx(1.5)          # the linear interpolant would be 1.5


def test_a_spline_extrapolates_straight_rather_than_curving_away():
    """Past the last knot a cubic runs off; OpenSim clamps coordinates, so stay linear."""
    node = _parse_one("<SimmSpline><x>0 1 2</x><y>0 1 2</y></SimmSpline>")
    assert evaluate_function(node, 5.0) == pytest.approx(5.0, abs=1e-9)
    assert evaluate_function(node, -3.0) == pytest.approx(-3.0, abs=1e-9)


def test_a_multiplier_scales_whatever_it_wraps():
    node = _parse_one(
        "<MultiplierFunction><function>"
        "<LinearFunction><coefficients>1 0</coefficients></LinearFunction>"
        "</function><scale>2.5</scale></MultiplierFunction>"
    )
    assert evaluate_function(node, 4.0) == pytest.approx(10.0)


def _parse_one(xml: str):
    import xml.etree.ElementTree as ET

    return ET.fromstring(xml)


def test_models_parsed_one_after_another_keep_their_own_curves():
    """A cohort sweep reads a thousand models in one process, and each has its own splines.

    Caching a compiled spline against the XML element's identity passes every single-model
    test and then fails here: once a model is collected, CPython hands its id() to the next
    one, and a subject silently inherits the previous subject's knee. Found by a sweep whose
    error jumped to most of a metre on a subject that checks out perfectly on its own.
    """
    def knee(y_values: str) -> str:
        return _document(
            FREE_PELVIS
            + '<CustomJoint name="knee">'
            "<socket_parent_frame>pelvis_offset</socket_parent_frame>"
            "<socket_child_frame>tibia_offset</socket_child_frame>"
            '<coordinates><Coordinate name="knee_angle"/></coordinates>'
            "<SpatialTransform>"
            '<TransformAxis name="translation1"><coordinates>knee_angle</coordinates>'
            f"<axis>1 0 0</axis><SimmSpline><x>0 1 2</x><y>{y_values}</y></SimmSpline>"
            "</TransformAxis>"
            "</SpatialTransform>"
            "<frames>"
            + _offset("pelvis_offset", "pelvis")
            + _offset("tibia_offset", "tibia")
            + "</frames></CustomJoint>"
        )

    first = parse_kinematics(knee("0 1 2"))
    for _ in range(50):                      # churn enough elements to recycle ids
        parse_kinematics(knee("0 5 10"))
    second = parse_kinematics(knee("0 5 10"))

    np.testing.assert_allclose(first.forward({"knee_angle": 1.0}).joint_centres["knee"],
                               [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(second.forward({"knee_angle": 1.0}).joint_centres["knee"],
                               [5.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------- structure


def test_the_joints_and_their_frames_are_read(two_joint):
    assert [j.name for j in two_joint.joints] == ["ground_pelvis", "hip_r"]
    hip = two_joint.joints[1]
    assert hip.parent_frame.body == "pelvis"
    assert hip.child_frame.body == "femur_r"
    np.testing.assert_allclose(hip.parent_frame.translation, [0.1, -0.05, 0.07])


def test_an_unsupported_joint_type_is_refused_rather_than_treated_as_a_weld():
    document = _document(
        '<GimbalJoint name="odd"><socket_parent_frame>a</socket_parent_frame>'
        "<socket_child_frame>b</socket_child_frame></GimbalJoint>"
    )
    with pytest.raises(UnsupportedJoint, match="GimbalJoint"):
        parse_kinematics(document)


# ---------------------------------------------------------------- forward kinematics


def test_the_root_translation_lands_the_pelvis_where_the_coordinates_say(two_joint):
    result = two_joint.forward(
        {"pelvis_tx": 1.0, "pelvis_ty": 2.0, "pelvis_tz": 3.0}
    )
    np.testing.assert_allclose(result.joint_centres["ground_pelvis"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result.body_rotation["pelvis"], np.eye(3), atol=1e-12)


def test_an_unmentioned_coordinate_is_zero_not_an_error(two_joint):
    result = two_joint.forward({})
    np.testing.assert_allclose(result.joint_centres["ground_pelvis"], [0.0, 0.0, 0.0])


def test_the_three_rotations_compose_in_the_order_the_model_lists_them(two_joint):
    q = {"pelvis_tilt": 0.3, "pelvis_list": -0.2, "pelvis_rotation": 0.5}
    expected = (
        Rotation.from_rotvec([0, 0, 0.3]).as_matrix()
        @ Rotation.from_rotvec([-0.2, 0, 0]).as_matrix()
        @ Rotation.from_rotvec([0, 0.5, 0]).as_matrix()
    )
    result = two_joint.forward(q)
    np.testing.assert_allclose(result.body_rotation["pelvis"], expected, atol=1e-12)


def test_a_child_joint_centre_rides_on_its_parents_rotation(two_joint):
    offset = np.array([0.1, -0.05, 0.07])
    q = {"pelvis_tilt": 0.4, "pelvis_tx": 1.0}
    result = two_joint.forward(q)
    rotation = Rotation.from_rotvec([0, 0, 0.4]).as_matrix()
    np.testing.assert_allclose(
        result.joint_centres["hip_r"], np.array([1.0, 0.0, 0.0]) + rotation @ offset,
        atol=1e-12,
    )


def test_a_pin_joint_turns_about_z_and_moves_nothing(two_joint):
    still = two_joint.forward({})
    turned = two_joint.forward({"hip_flexion_r": 0.7})
    np.testing.assert_allclose(
        still.joint_centres["hip_r"], turned.joint_centres["hip_r"], atol=1e-12
    )
    np.testing.assert_allclose(
        turned.body_rotation["femur_r"],
        Rotation.from_rotvec([0, 0, 0.7]).as_matrix(),
        atol=1e-12,
    )


def test_an_offset_frame_orientation_turns_the_child_body():
    document = _document(
        FREE_PELVIS
        + '<PinJoint name="hip_r">'
        "<socket_parent_frame>pelvis_offset</socket_parent_frame>"
        "<socket_child_frame>femur_r_offset</socket_child_frame>"
        '<coordinates><Coordinate name="hip_flexion_r"/></coordinates>'
        "<frames>"
        + _offset("pelvis_offset", "pelvis", orientation="0 0 0.5")
        + _offset("femur_r_offset", "femur_r")
        + "</frames></PinJoint>"
    )
    model = parse_kinematics(document)
    result = model.forward({})
    np.testing.assert_allclose(
        result.body_rotation["femur_r"],
        Rotation.from_euler("XYZ", [0.0, 0.0, 0.5]).as_matrix(),
        atol=1e-12,
    )


def test_forward_returns_a_centre_for_every_joint_it_parsed(two_joint):
    result = two_joint.forward({})
    assert set(result.joint_centres) == {"ground_pelvis", "hip_r"}
    assert set(result.body_rotation) >= {"pelvis", "femur_r"}


def test_a_whole_trial_at_once_gives_exactly_what_frame_by_frame_gives(two_joint):
    """The batched path exists for speed and must not become a second, subtly different model.

    Frame by frame is 1.3 ms, which is 20 hours over the cohort, so the conversion uses the
    batched path -- and then every number in the corpus depends on the two agreeing.
    """
    rng = np.random.default_rng(11)
    names = ["pelvis_tilt", "pelvis_list", "pelvis_rotation",
             "pelvis_tx", "pelvis_ty", "pelvis_tz", "hip_flexion_r"]
    block = rng.normal(0.0, 0.6, (17, len(names)))
    batched = two_joint.forward_batch(dict(zip(names, block.T)))
    for frame in range(block.shape[0]):
        one = two_joint.forward(dict(zip(names, block[frame])))
        for joint in one.joint_centres:
            np.testing.assert_allclose(
                batched.joint_centres[joint][frame], one.joint_centres[joint], atol=1e-12
            )
        for body in one.body_rotation:
            np.testing.assert_allclose(
                batched.body_rotation[body][frame], one.body_rotation[body], atol=1e-12
            )


def test_the_batched_path_agrees_on_a_spline_joint_too():
    """Splines are where a vectorised rewrite is most likely to diverge."""
    document = _document(
        FREE_PELVIS
        + '<CustomJoint name="knee">'
        "<socket_parent_frame>pelvis_offset</socket_parent_frame>"
        "<socket_child_frame>tibia_offset</socket_child_frame>"
        '<coordinates><Coordinate name="knee_angle"/></coordinates>'
        "<SpatialTransform>"
        '<TransformAxis name="rotation2"><coordinates>knee_angle</coordinates><axis>0 0 1</axis>'
        "<SimmSpline><x>0 0.5 1 1.5 2</x><y>0 0.02 0.03 0.02 -0.01</y></SimmSpline>"
        "</TransformAxis>"
        '<TransformAxis name="translation1"><coordinates>knee_angle</coordinates><axis>0 1 0</axis>'
        "<MultiplierFunction><function>"
        "<SimmSpline><x>0 0.5 1 1.5 2</x><y>0 0.001 0.0014 0.0014 0.0016</y></SimmSpline>"
        "</function><scale>1.007</scale></MultiplierFunction>"
        "</TransformAxis>"
        "</SpatialTransform>"
        "<frames>"
        + _offset("pelvis_offset", "pelvis")
        + _offset("tibia_offset", "tibia")
        + "</frames></CustomJoint>"
    )
    model = parse_kinematics(document)
    # deliberately steps outside the knots at both ends, where extrapolation applies
    angles = np.array([-0.4, 0.0, 0.25, 1.0, 1.9, 2.0, 2.6])
    batched = model.forward_batch({"knee_angle": angles})
    for frame, angle in enumerate(angles):
        one = model.forward({"knee_angle": float(angle)})
        np.testing.assert_allclose(
            batched.joint_centres["knee"][frame], one.joint_centres["knee"], atol=1e-12
        )
        np.testing.assert_allclose(
            batched.body_rotation["tibia"][frame], one.body_rotation["tibia"], atol=1e-12
        )


def test_the_batch_infers_its_length_and_refuses_ragged_input(two_joint):
    batched = two_joint.forward_batch({"pelvis_tx": np.zeros(9)})
    assert batched.joint_centres["ground_pelvis"].shape == (9, 3)
    assert batched.body_rotation["pelvis"].shape == (9, 3, 3)
    with pytest.raises(ValueError, match="length"):
        two_joint.forward_batch({"pelvis_tx": np.zeros(9), "pelvis_ty": np.zeros(8)})


def test_a_batch_with_no_coordinates_still_places_the_skeleton(two_joint):
    batched = two_joint.forward_batch({}, frames=4)
    np.testing.assert_allclose(batched.joint_centres["ground_pelvis"], np.zeros((4, 3)))
    with pytest.raises(ValueError, match="frames"):
        two_joint.forward_batch({})


def test_a_translation_axis_moves_along_the_parent_frame_not_the_child():
    """A coupled joint slides *then* turns; getting that backwards bends the knee sideways."""
    document = _document(
        FREE_PELVIS
        + '<CustomJoint name="slider">'
        "<socket_parent_frame>pelvis_offset</socket_parent_frame>"
        "<socket_child_frame>femur_r_offset</socket_child_frame>"
        '<coordinates><Coordinate name="q"/></coordinates>'
        "<SpatialTransform>"
        '<TransformAxis name="rotation1"><coordinates>q</coordinates><axis>0 0 1</axis>'
        "<LinearFunction><coefficients>1 0</coefficients></LinearFunction></TransformAxis>"
        '<TransformAxis name="translation1"><coordinates>q</coordinates><axis>1 0 0</axis>'
        "<LinearFunction><coefficients>1 0</coefficients></LinearFunction></TransformAxis>"
        "</SpatialTransform>"
        "<frames>"
        + _offset("pelvis_offset", "pelvis")
        + _offset("femur_r_offset", "femur_r")
        + "</frames></CustomJoint>"
    )
    model = parse_kinematics(document)
    result = model.forward({"q": 0.5})
    # translation is read in the parent frame, so a simultaneous rotation must not turn it
    np.testing.assert_allclose(result.joint_centres["slider"], [0.5, 0.0, 0.0], atol=1e-12)
