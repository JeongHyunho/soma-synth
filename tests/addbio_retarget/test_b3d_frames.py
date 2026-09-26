"""Frame addressing in .b3d, exercised against a file this test builds itself.

The payload is a flat run of fixed-width records with no index: a trial's frame is found by
adding up every preceding trial's `length * (raw_sensor_frame_size + passes *
processing_pass_frame_size)`. Trials differ in both length *and* pass count, so an arithmetic
slip lands on a neighbouring frame and still decodes -- it returns a real pose, just the wrong
one. Every trial here is given a distinct pass count and a distinct value pattern so that
landing on the wrong record fails loudly.
"""

import struct

import numpy as np
import pytest

from soma_synth.addbio_retarget.b3d_frames import (
    B3DLayoutError,
    open_b3d,
    read_pass_frames,
)
from soma_synth.addbio_retarget.b3d_frames import _protobuf as pb

NUM_DOFS = 3
NUM_CENTRES = 2
TRIALS = (
    # (length, pass types)
    (4, ("kinematics", "lowPassFilter", "dynamics")),
    (3, ("kinematics", "dynamics")),
    (2, ("kinematics", "lowPassFilter", "dynamics")),
)


def _sensor_frame() -> bytes:
    frame = pb.SubjectOnDiskSensorFrame()
    frame.marker_obs.extend([0.0, 0.0, 0.0])
    return frame.SerializeToString()


def _pass_frame(trial: int, frame: int, which: int) -> bytes:
    """A frame whose numbers encode exactly where it lives."""
    payload = pb.SubjectOnDiskProcessingPassFrame()
    tag = 100 * trial + 10 * frame + which
    payload.pos.extend([tag + 0.1, tag + 0.2, tag + 0.3])
    payload.world_frame_joint_centers.extend(
        [tag + 1.0, tag + 2.0, tag + 3.0, tag + 4.0, tag + 5.0, tag + 6.0]
    )
    payload.ground_contact_force.extend([tag, 0.0, 0.0])
    payload.ground_contact_center_of_pressure.extend([0.0, tag, 0.0])
    return payload.SerializeToString()


@pytest.fixture(scope="module")
def synthetic_b3d(tmp_path_factory):
    header = pb.SubjectOnDiskHeader()
    header.num_dofs = NUM_DOFS
    header.version = 4
    header.biological_sex = "female"
    header.height_m = 1.7
    header.mass_kg = 62.0
    for name in ("kinematics", "lowPassFilter", "dynamics"):
        entry = header.passes.add()
        entry.pass_type = pb.ProcessingPassType.Value(name)
        entry.model_osim_text = _MINI_OSIM
    header.ground_contact_body.extend(["calcn_r", "calcn_l"])

    sensor = _sensor_frame()
    header.raw_sensor_frame_size = len(sensor)
    header.processing_pass_frame_size = len(_pass_frame(0, 0, 0))

    for length, passes in TRIALS:
        trial = header.trial_header.add()
        trial.trial_length = length
        trial.trial_timestep = 0.005
        for name in passes:
            ph = trial.processing_pass_header.add()
            ph.type = pb.ProcessingPassType.Value(name)

    blob = header.SerializeToString()
    body = bytearray()
    body += struct.pack("<q", len(blob))
    body += blob
    for trial_index, (length, passes) in enumerate(TRIALS):
        for frame_index in range(length):
            body += sensor
            for pass_index in range(len(passes)):
                record = _pass_frame(trial_index, frame_index, pass_index)
                assert len(record) == header.processing_pass_frame_size
                body += record

    path = tmp_path_factory.mktemp("b3d") / "synthetic.b3d"
    path.write_bytes(bytes(body))
    return path


_MINI_OSIM = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40500">
  <Model name="mini">
    <JointSet>
      <objects>
        <CustomJoint name="ground_pelvis">
          <socket_parent_frame>/ground</socket_parent_frame>
          <socket_child_frame>/bodyset/pelvis</socket_child_frame>
          <coordinates><Coordinate name="pelvis_tilt"/><Coordinate name="pelvis_tx"/></coordinates>
        </CustomJoint>
        <PinJoint name="hip_r">
          <socket_parent_frame>/bodyset/pelvis</socket_parent_frame>
          <socket_child_frame>/bodyset/femur_r</socket_child_frame>
          <coordinates><Coordinate name="hip_flexion_r"/></coordinates>
        </PinJoint>
      </objects>
    </JointSet>
  </Model>
</OpenSimDocument>
"""


def test_layout_accounts_for_every_byte(synthetic_b3d):
    b3d = open_b3d(synthetic_b3d)
    assert [t.length for t in b3d.trials] == [4, 3, 2]
    assert [t.pass_count for t in b3d.trials] == [3, 2, 3]
    assert b3d.payload_offset + sum(
        t.length * t.frame_size for t in b3d.trials
    ) == synthetic_b3d.stat().st_size


def test_the_dynamics_pass_is_found_per_trial_not_per_file(synthetic_b3d):
    b3d = open_b3d(synthetic_b3d)
    # trial 1 has no lowPassFilter, so dynamics sits at a different index there
    assert [t.dynamics_pass_index for t in b3d.trials] == [2, 1, 2]


def test_every_frame_of_every_trial_addresses_its_own_record(synthetic_b3d):
    b3d = open_b3d(synthetic_b3d)
    for trial_index, trial in enumerate(b3d.trials):
        for pass_index in range(trial.pass_count):
            frames = read_pass_frames(
                b3d, trial_index, range(trial.length), pass_index=pass_index
            )
            for frame_index in range(trial.length):
                tag = 100 * trial_index + 10 * frame_index + pass_index
                assert frames.pos[frame_index].tolist() == pytest.approx(
                    [tag + 0.1, tag + 0.2, tag + 0.3]
                )
                np.testing.assert_allclose(
                    frames.joint_centres[frame_index],
                    [[tag + 1.0, tag + 2.0, tag + 3.0], [tag + 4.0, tag + 5.0, tag + 6.0]],
                )


def test_joint_centres_come_back_shaped_as_points(synthetic_b3d):
    b3d = open_b3d(synthetic_b3d)
    frames = read_pass_frames(b3d, 0, [0, 2])
    assert frames.joint_centres.shape == (2, NUM_CENTRES, 3)
    assert frames.pos.shape == (2, NUM_DOFS)
    assert frames.ground_contact_force.shape == (2, 1, 3)


def test_defaulting_to_dynamics_reads_the_dynamics_record(synthetic_b3d):
    b3d = open_b3d(synthetic_b3d)
    default = read_pass_frames(b3d, 1, [0])
    explicit = read_pass_frames(b3d, 1, [0], pass_index=1)
    assert default.pos.tolist() == explicit.pos.tolist()
    assert default.pass_index == 1


def test_frames_carry_their_own_timestamps(synthetic_b3d):
    b3d = open_b3d(synthetic_b3d)
    frames = read_pass_frames(b3d, 0, [0, 1, 3])
    assert frames.timestamps_s.tolist() == pytest.approx([0.0, 0.005, 0.015])
    assert frames.source_rate_hz == pytest.approx(200.0)


def test_reading_past_the_end_of_a_trial_is_refused(synthetic_b3d):
    b3d = open_b3d(synthetic_b3d)
    with pytest.raises(B3DLayoutError):
        read_pass_frames(b3d, 1, [3])          # trial 1 holds 3 frames: 0..2
    with pytest.raises(B3DLayoutError):
        read_pass_frames(b3d, 0, [0], pass_index=3)
    with pytest.raises(B3DLayoutError):
        read_pass_frames(b3d, 9, [0])


def test_a_truncated_file_is_refused_rather_than_read_short(synthetic_b3d, tmp_path):
    truncated = tmp_path / "short.b3d"
    truncated.write_bytes(synthetic_b3d.read_bytes()[:-40])
    with pytest.raises(B3DLayoutError):
        open_b3d(truncated)


def test_the_model_is_parsed_from_the_dynamics_pass(synthetic_b3d):
    b3d = open_b3d(synthetic_b3d)
    assert b3d.topology.independent_coordinate_names == (
        "pelvis_tilt", "pelvis_tx", "hip_flexion_r",
    )
    assert b3d.topology.joint_centre_names == ("ground_pelvis", "hip_r")
    assert b3d.subject.biological_sex == "female"
    assert b3d.subject.height_m == pytest.approx(1.7)


def test_declared_dof_count_must_match_the_model(synthetic_b3d, tmp_path):
    """A header whose num_dofs disagrees with its own model means one of them is not ours."""
    b3d = open_b3d(synthetic_b3d)
    assert b3d.header.num_dofs == len(b3d.topology.independent_coordinate_names)


def test_a_model_without_gravity_refuses_to_hand_out_a_world_frame(synthetic_b3d):
    """Rotating a source whose up axis is unstated would tip the whole artifact silently."""
    b3d = open_b3d(synthetic_b3d)
    with pytest.raises(B3DLayoutError, match="gravity"):
        _ = b3d.world_frame


def test_pos_and_centres_are_read_as_float64(synthetic_b3d):
    b3d = open_b3d(synthetic_b3d)
    frames = read_pass_frames(b3d, 2, [0])
    assert frames.pos.dtype == np.float64
    assert frames.joint_centres.dtype == np.float64
