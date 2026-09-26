"""What one converted trial looks like on disk.

The artifact is the product of the conversion: a downstream reader uses it without reading
any of this code, so what it does and does not contain has to be legible from the file alone.
Three properties are checked. Every joint says where it came from, so `absent` is visible as a
marker rather than mistaken for a measurement. The frame rotation is recorded, so the Y-up
source can be recovered. And the ten attribution fields ride along, because the licence requires
them and a file that leaves them out cannot be published even internally.
"""

import numpy as np
import pytest

from soma_synth.addbio_retarget.artifact import (
    ARTIFACT_CLASS,
    QUALITY_GATE,
    SPEC_ID,
    SPEC_VERSION,
    build_artifact,
    write_artifact,
)
from soma_synth.addbio_retarget.smpl_correspondence import ABSENT, MEASURED

FRAMES = 5
HASHES = dict(adapter_hash="a1", config_hash="b2", code_hash="c3", model_hash="d4")


def payload(**overrides):
    base = dict(
        source_folder="Moore2015_Formatted_No_Arm",
        subject="subject12",
        variant="No_Arm",
        pass_used="dynamics",
        trial_name="t0",
        betas=np.arange(10, dtype=np.float64),
        gender="male",
        pose=np.zeros((FRAMES, 24, 3)),
        trans=np.ones((FRAMES, 3)),
        joint_provenance=np.array([MEASURED] * 18 + [ABSENT] * 6, dtype=object),
        source_coordinates=[("hip_flexion_l", "hip_adduction_l")] * 24,
        rest_alignment=np.tile([1.0, 0.0, 0.0, 0.0], (24, 1)),
        root_offset=np.array([0.01, 0.02, 0.03]),
        frame_rotation_source_to_world=np.array([0.7071, 0.7071, 0.0, 0.0]),
        timestamps_s=np.arange(FRAMES) * 0.005,
        source_rate_hz=200.0,
        position_error_m=0.031,
        **HASHES,
    )
    base.update(overrides)
    return build_artifact(**base)


def test_it_holds_every_field_the_artifact_defines():
    built = payload()
    for key in (
        "betas", "gender", "pose", "trans", "joint_provenance", "source_dof_names",
        "rest_alignment", "lumbar_distribution", "frame_rotation_source_to_world",
        "timestamps_s", "source_rate_hz", "variant", "pass_used",
        "wrap_repaired_coordinates",
    ):
        assert key in built, key


def test_the_wrap_repair_is_disclosed_even_when_nothing_was_repaired():
    """An empty string says "checked, nothing to do"; a missing key would say nothing at all."""
    assert str(payload()["wrap_repaired_coordinates"]) == ""
    named = payload(wrap_repaired_coordinates=["pelvis_rotation", "arm_rot_r"])
    assert str(named["wrap_repaired_coordinates"]) == "pelvis_rotation,arm_rot_r"


def test_the_shapes_are_the_ones_downstream_will_index():
    built = payload()
    assert built["betas"].shape == (10,)
    assert built["pose"].shape == (FRAMES, 24, 3)
    assert built["trans"].shape == (FRAMES, 3)
    assert built["joint_provenance"].shape == (24,)
    assert built["rest_alignment"].shape == (24, 4)
    assert built["frame_rotation_source_to_world"].shape == (4,)
    assert built["timestamps_s"].shape == (FRAMES,)
    assert built["root_offset"].shape == (3,)


def test_poses_are_float32_and_time_stays_float64():
    built = payload()
    assert built["pose"].dtype == np.float32
    assert built["trans"].dtype == np.float32
    assert built["betas"].dtype == np.float32
    # a 0.005 s step over a long trial loses frames at float32; time keeps its precision
    assert built["timestamps_s"].dtype == np.float64


def test_the_identity_is_addbio_smpl24_raw():
    built = payload()
    assert str(built["spec_id"]) == SPEC_ID == "addbio_smpl24_raw"
    assert str(built["spec_version"]) == SPEC_VERSION == "retarget-v2"


def test_it_does_not_borrow_the_governed_pipelines_vocabulary():
    """`contract_version` and `canonical` belong to the governed pipeline; this is not that."""
    built = payload()
    for key, value in built.items():
        assert "contract_version" not in key
        assert "canonical" not in key
        if isinstance(value, str):
            assert "canonical" not in value


def test_it_says_it_is_not_a_candidate_and_has_not_been_graded():
    built = payload()
    assert str(built["artifact_class"]) == ARTIFACT_CLASS == "experimental_non_candidate"
    assert str(built["quality_gate"]) == QUALITY_GATE == "NOT_EVALUATED"
    assert str(built["distribution_scope"]) == "internal_only"


def test_the_ten_attribution_fields_ride_along():
    built = payload()
    assert str(built["stable_source_id"]) == "ab_moore_2015"
    assert "25945311" in str(built["publication"])
    assert str(built["license_id"]) == "CC-BY-4.0"
    assert str(built["changes_made"]) == "true"
    assert "adapter=a1" in str(built["content_hashes"])


def test_a_source_it_cannot_attribute_is_refused():
    from soma_synth.addbio_retarget.attribution import AttributionSourceUnknown

    with pytest.raises(AttributionSourceUnknown):
        payload(source_folder="Nobody2030_Formatted_No_Arm")


def test_absent_joints_are_visible_as_absent_and_hold_rest():
    built = payload()
    provenance = built["joint_provenance"]
    assert (provenance == ABSENT).sum() == 6
    absent = np.where(provenance == ABSENT)[0]
    np.testing.assert_allclose(built["pose"][:, absent], 0.0)


def test_the_lumbar_rule_is_named_not_implied():
    built = payload()
    assert str(built["lumbar_distribution"]) == "equal_thirds_v1"


def test_a_pose_that_is_not_24_joints_is_refused():
    with pytest.raises(ValueError, match="24"):
        payload(pose=np.zeros((FRAMES, 22, 3)))


def test_frames_that_disagree_between_pose_and_time_are_refused():
    with pytest.raises(ValueError, match="frames"):
        payload(timestamps_s=np.arange(FRAMES + 1) * 0.005)


def test_it_round_trips_through_a_file(tmp_path):
    built = payload()
    path = tmp_path / "trial.npz"
    write_artifact(path, built)
    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(built)
        np.testing.assert_allclose(loaded["pose"], built["pose"])
        assert str(loaded["stable_source_id"]) == "ab_moore_2015"


def test_the_written_file_never_needs_pickle_to_read():
    """A reader that must allow_pickle cannot be trusted; keep every value a plain array."""
    built = payload()
    for key, value in built.items():
        assert isinstance(value, np.ndarray), key
        assert value.dtype != np.dtype("O"), key
