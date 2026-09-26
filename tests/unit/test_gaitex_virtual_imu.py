"""Virtual inertial signal synthesis."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from soma_synth.sensors import virtual_imu as vi

DT_S = 0.01
GRAVITY = np.array([0.0, 0.0, -9.80665])
CUTOFF_HZ = 12.0
ORDER = 4
TRIM = 17


def synth(**overrides):
    defaults = dict(
        dt_s=DT_S,
        cutoff_hz=CUTOFF_HZ,
        filter_order=ORDER,
        edge_trim_frames=TRIM,
        gravity_world_m_per_s2=GRAVITY,
        rotation_cluster_from_sensor=np.eye(3),
        lever_arm_local=np.zeros(3),
    )
    defaults.update(overrides)
    return vi.synthesise_virtual_imu(**defaults)


def stationary(frames: int):
    translation = np.tile(np.array([0.3, -0.2, 1.0]), (frames, 1))
    rotation = np.repeat(np.eye(3)[None], frames, axis=0)
    return translation, rotation, np.ones(frames, dtype=bool)


class TestSensorTrack:
    def test_a_zero_lever_leaves_the_cluster_centroid_alone(self) -> None:
        translation, rotation, _ = stationary(10)
        track = vi.sensor_world_track(translation, rotation, np.zeros(3))
        np.testing.assert_allclose(track, translation)

    def test_the_lever_rotates_with_the_body(self) -> None:
        frames = 4
        translation = np.zeros((frames, 3))
        rotation = Rotation.from_euler(
            "z", [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        ).as_matrix()

        track = vi.sensor_world_track(translation, rotation, np.array([0.02, 0.0, 0.0]))

        np.testing.assert_allclose(track[0], [0.02, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(track[1], [0.0, 0.02, 0.0], atol=1e-12)
        np.testing.assert_allclose(track[2], [-0.02, 0.0, 0.0], atol=1e-12)

    def test_a_wrong_lever_rank_is_refused(self) -> None:
        translation, rotation, _ = stationary(5)
        with pytest.raises(vi.VirtualImuError, match="3-vector"):
            vi.sensor_world_track(translation, rotation, np.zeros(2))


class TestGravityOnly:
    def test_a_stationary_sensor_reads_exactly_one_gravity(self) -> None:
        translation, rotation, usable = stationary(400)

        signal = synth(
            translation=translation,
            rotation_world_from_cluster=rotation,
            usable=usable,
        )

        magnitude = np.linalg.norm(signal.acceleration_m_per_s2[signal.valid], axis=1)
        assert signal.valid.sum() > 300
        assert np.abs(magnitude - 9.80665).max() < 1e-6

    def test_gravity_appears_along_the_sensor_up_axis(self) -> None:
        translation, rotation, usable = stationary(400)

        signal = synth(
            translation=translation,
            rotation_world_from_cluster=rotation,
            usable=usable,
        )

        reading = signal.acceleration_m_per_s2[signal.valid]
        np.testing.assert_allclose(
            reading, np.tile([0.0, 0.0, 9.80665], (reading.shape[0], 1)), atol=1e-6
        )

    def test_a_tilted_sensor_redistributes_gravity(self) -> None:
        frames = 400
        translation = np.zeros((frames, 3))
        tilt = Rotation.from_euler("y", np.pi / 6).as_matrix()
        rotation = np.repeat(tilt[None], frames, axis=0)

        signal = synth(
            translation=translation,
            rotation_world_from_cluster=rotation,
            usable=np.ones(frames, dtype=bool),
        )

        reading = signal.acceleration_m_per_s2[signal.valid][0]
        assert np.linalg.norm(reading) == pytest.approx(9.80665, abs=1e-6)
        assert reading[0] == pytest.approx(-9.80665 * np.sin(np.pi / 6), abs=1e-6)
        assert reading[2] == pytest.approx(9.80665 * np.cos(np.pi / 6), abs=1e-6)

    def test_gravity_is_taken_from_the_caller(self) -> None:
        translation, rotation, usable = stationary(400)

        signal = synth(
            translation=translation,
            rotation_world_from_cluster=rotation,
            usable=usable,
            gravity_world_m_per_s2=np.array([0.0, 0.0, -1.0]),
        )

        magnitude = np.linalg.norm(signal.acceleration_m_per_s2[signal.valid], axis=1)
        assert np.abs(magnitude - 1.0).max() < 1e-6


class TestAngularVelocity:
    def test_a_constant_rate_is_recovered_exactly(self) -> None:
        frames = 400
        rate = np.array([0.0, 0.0, 1.7])  # rad/s about the sensor z axis
        angles = np.arange(frames) * DT_S * rate[2]
        rotation = Rotation.from_euler("z", angles).as_matrix()

        signal = synth(
            translation=np.zeros((frames, 3)),
            rotation_world_from_cluster=rotation,
            usable=np.ones(frames, dtype=bool),
        )

        observed = np.radians(signal.angular_velocity_deg_per_s[signal.valid])
        assert np.abs(observed - rate).max() < 1e-6

    def test_the_channel_is_reported_in_degrees_per_second(self) -> None:
        frames = 400
        rotation = Rotation.from_euler(
            "z", np.arange(frames) * DT_S * 1.0
        ).as_matrix()

        signal = synth(
            translation=np.zeros((frames, 3)),
            rotation_world_from_cluster=rotation,
            usable=np.ones(frames, dtype=bool),
        )

        observed = signal.angular_velocity_deg_per_s[signal.valid]
        assert np.abs(observed[:, 2] - np.degrees(1.0)).max() < 1e-6

    def test_the_log_map_beats_differencing_rotation_matrices(self) -> None:
        """Why the log map: matrix differencing carries an O(dt^2) rate-cubed error."""
        frames = 200
        rate = 6.0
        angles = np.arange(frames) * DT_S * rate
        matrices = Rotation.from_euler("z", angles).as_matrix()
        xyzw = Rotation.from_matrix(matrices).as_quat()
        wxyz = np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])

        by_log = vi.angular_velocity_rad_per_s(wxyz, DT_S)[1:-1, 2]
        derivative = (matrices[2:] - matrices[:-2]) / (2.0 * DT_S)
        skew = np.einsum("fji,fjk->fik", matrices[1:-1], derivative)
        by_difference = (skew[:, 1, 0] - skew[:, 0, 1]) / 2.0

        assert np.abs(by_log - rate).max() < 1e-9
        assert np.abs(by_difference - rate).max() > 1e-4


class TestLeverArmEffect:
    def test_a_lever_on_a_spinning_body_produces_centripetal_acceleration(self) -> None:
        frames = 600
        rate = 5.0
        radius = 0.02
        angles = np.arange(frames) * DT_S * rate
        rotation = Rotation.from_euler("z", angles).as_matrix()

        signal = synth(
            translation=np.zeros((frames, 3)),
            rotation_world_from_cluster=rotation,
            usable=np.ones(frames, dtype=bool),
            lever_arm_local=np.array([radius, 0.0, 0.0]),
        )

        reading = signal.acceleration_m_per_s2[signal.valid]
        # True acceleration points at the axis, so specific force -- which is a - g --
        # carries -omega^2 r along the sensor's own x, with gravity still on z.
        assert reading[:, 0].mean() == pytest.approx(-(rate**2) * radius, rel=1e-3)
        assert reading[:, 2].mean() == pytest.approx(9.80665, abs=1e-3)

    def test_a_zero_lever_shows_no_centripetal_term(self) -> None:
        frames = 600
        angles = np.arange(frames) * DT_S * 5.0
        rotation = Rotation.from_euler("z", angles).as_matrix()

        signal = synth(
            translation=np.zeros((frames, 3)),
            rotation_world_from_cluster=rotation,
            usable=np.ones(frames, dtype=bool),
        )

        reading = signal.acceleration_m_per_s2[signal.valid]
        assert np.abs(reading[:, 0]).max() < 1e-5


class TestSegmentationAndTrim:
    def test_edges_of_each_usable_span_are_trimmed(self) -> None:
        frames = 600
        translation, rotation, _ = stationary(frames)
        usable = np.ones(frames, dtype=bool)
        usable[250:300] = False

        signal = synth(
            translation=translation,
            rotation_world_from_cluster=rotation,
            usable=usable,
            edge_trim_frames=17,
        )

        assert not signal.valid[:17].any()
        assert signal.valid[17]
        assert not signal.valid[250:300].any()
        assert not signal.valid[233:250].any()
        assert not signal.valid[300:317].any()
        assert signal.valid[317]

    def test_unusable_frames_carry_nan_in_every_channel(self) -> None:
        frames = 400
        translation, rotation, _ = stationary(frames)
        usable = np.ones(frames, dtype=bool)
        usable[100:150] = False

        signal = synth(
            translation=translation,
            rotation_world_from_cluster=rotation,
            usable=usable,
        )

        invalid = ~signal.valid
        assert np.isnan(signal.acceleration_m_per_s2[invalid]).all()
        assert np.isnan(signal.angular_velocity_deg_per_s[invalid]).all()
        assert np.isnan(signal.orientation_wxyz[invalid]).all()

    def test_a_span_too_short_to_filter_is_dropped_rather_than_approximated(self) -> None:
        frames = 400
        translation, rotation, _ = stationary(frames)
        usable = np.zeros(frames, dtype=bool)
        usable[10:14] = True

        signal = synth(
            translation=translation,
            rotation_world_from_cluster=rotation,
            usable=usable,
            edge_trim_frames=0,
        )

        assert not signal.valid.any()

    def test_contiguous_runs_are_half_open(self) -> None:
        mask = np.array([0, 1, 1, 0, 1], dtype=bool)
        assert vi.contiguous_runs(mask) == [(1, 3), (4, 5)]


class TestOrientationChannel:
    def test_orientation_is_returned_w_first_and_unit_norm(self) -> None:
        frames = 400
        rotation = Rotation.from_euler(
            "xyz", np.column_stack([0.01 * np.arange(frames)] * 3)
        ).as_matrix()

        signal = synth(
            translation=np.zeros((frames, 3)),
            rotation_world_from_cluster=rotation,
            usable=np.ones(frames, dtype=bool),
        )

        quaternions = signal.orientation_wxyz[signal.valid]
        np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-12)
        # Identity at frame 0 would be (1, 0, 0, 0); the first valid frame is close to it.
        assert quaternions[0, 0] > 0.9

    def test_the_sensor_mounting_rotation_is_applied(self) -> None:
        frames = 400
        translation, rotation, usable = stationary(frames)
        mounting = Rotation.from_euler("x", np.pi / 2).as_matrix()

        signal = synth(
            translation=translation,
            rotation_world_from_cluster=rotation,
            usable=usable,
            rotation_cluster_from_sensor=mounting,
        )

        reading = signal.acceleration_m_per_s2[signal.valid][0]
        # With B a quarter turn about x, B^T maps world +z onto sensor +y, so the whole of
        # gravity moves off the sensor's z axis and onto its y.
        assert reading[1] == pytest.approx(9.80665, abs=1e-6)
        assert abs(reading[2]) < 1e-6


class TestConfigurationIsRequired:
    @pytest.mark.parametrize(
        "missing",
        ["dt_s", "cutoff_hz", "filter_order", "edge_trim_frames", "gravity_world_m_per_s2"],
    )
    def test_every_execution_constant_must_be_supplied(self, missing: str) -> None:
        translation, rotation, usable = stationary(200)
        kwargs = dict(
            translation=translation,
            rotation_world_from_cluster=rotation,
            usable=usable,
            lever_arm_local=np.zeros(3),
            rotation_cluster_from_sensor=np.eye(3),
            dt_s=DT_S,
            cutoff_hz=CUTOFF_HZ,
            filter_order=ORDER,
            edge_trim_frames=TRIM,
            gravity_world_m_per_s2=GRAVITY,
        )
        del kwargs[missing]
        with pytest.raises(TypeError):
            vi.synthesise_virtual_imu(**kwargs)  # type: ignore[arg-type]

    def test_a_cutoff_at_or_above_nyquist_is_refused(self) -> None:
        translation, rotation, usable = stationary(400)
        with pytest.raises(vi.VirtualImuError, match="Nyquist"):
            synth(
                translation=translation,
                rotation_world_from_cluster=rotation,
                usable=usable,
                cutoff_hz=50.0,
            )
