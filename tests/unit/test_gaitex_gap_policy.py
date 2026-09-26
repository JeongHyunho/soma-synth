"""Pose-level gap filling and window rejection."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from soma_synth.kinematics import gap_policy as gp
from soma_synth.kinematics import quaternion as quat

DT_S = 0.01


def smooth_positions(frames: int) -> np.ndarray:
    t = np.arange(frames) * DT_S
    return np.column_stack([np.sin(2.0 * t), 0.5 * t**2, 1.0 + 0.3 * np.cos(3.0 * t)])


def constant_velocity_positions(frames: int) -> np.ndarray:
    """Straight line at a fixed speed, whose true acceleration is exactly zero."""
    t = np.arange(frames) * DT_S
    return np.column_stack([t, np.zeros(frames), np.zeros(frames)])


def constant_rate_rotation(frames: int, rate_rad_per_s: float = 3.0) -> np.ndarray:
    """Fixed axis at a fixed angular rate, in (w, x, y, z)."""
    t = np.arange(frames) * DT_S
    rotations = Rotation.from_rotvec(
        (rate_rad_per_s * t)[:, None] * np.array([0.0, 0.0, 1.0])[None, :]
    )
    xyzw = rotations.as_quat()
    return np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])


def accelerating_rotation(frames: int) -> np.ndarray:
    """Rotation whose angular velocity grows with time, in (w, x, y, z)."""
    t = np.arange(frames) * DT_S
    angle = 0.5 * t + 1.5 * t**2  # non-constant angular rate
    axis = np.array([0.0, 0.0, 1.0])
    rotations = Rotation.from_rotvec(angle[:, None] * axis[None, :])
    xyzw = rotations.as_quat()
    return np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])


def angular_speed(quaternions: np.ndarray) -> np.ndarray:
    """Central-difference angular speed in rad/s from a (w, x, y, z) series."""
    canonical = quat.canonicalise_sign(quaternions)
    relative = quat.multiply(quat.conjugate(canonical[:-2]), canonical[2:])
    half_angle = np.linalg.norm(quat.log_unit(relative)[:, 1:], axis=1)
    return 2.0 * half_angle / (2.0 * DT_S)


def slerp_fill(quaternions: np.ndarray, start: int, stop: int) -> np.ndarray:
    """A deliberately plain SLERP fill, used only to contrast against SQUAD."""
    left, right = start - 1, stop
    span = float(right - left)
    t = (np.arange(start, stop, dtype=np.float64) - left) / span
    count = t.shape[0]
    return quat.slerp(
        np.repeat(quaternions[left][None], count, 0),
        np.repeat(quaternions[right][None], count, 0),
        t,
    )


class TestGapRunDetection:
    def test_runs_are_reported_as_half_open_intervals(self) -> None:
        solved = np.array([1, 1, 0, 0, 0, 1, 1, 0, 1], dtype=bool)
        assert gp.find_gap_runs(solved) == [(2, 5), (7, 8)]

    def test_a_fully_solved_series_has_no_runs(self) -> None:
        assert gp.find_gap_runs(np.ones(10, dtype=bool)) == []

    def test_runs_at_both_boundaries_are_reported(self) -> None:
        solved = np.array([0, 1, 1, 0], dtype=bool)
        assert gp.find_gap_runs(solved) == [(0, 1), (3, 4)]


class TestHermitePositionFill:
    def test_filled_positions_track_the_true_trajectory(self) -> None:
        truth = smooth_positions(60)
        observed = truth.copy()
        observed[30:34] = np.nan

        filled = gp.hermite_fill(observed, 30, 34)
        error = np.abs(filled - truth[30:34]).max()

        # Sub-millimetre over a 40 ms gap, and materially better than the straight line
        # between the same two brackets, which is what makes the cubic worth its cost.
        linear = np.linspace(truth[29], truth[34], 6)[1:-1]
        linear_error = np.abs(linear - truth[30:34]).max()
        assert error < 1e-3
        assert error < linear_error / 3.0

    def test_velocity_is_continuous_across_the_join(self) -> None:
        truth = smooth_positions(60)
        observed = truth.copy()
        observed[30:34] = np.nan
        observed[30:34] = gp.hermite_fill(observed, 30, 34)

        velocity = np.diff(observed, axis=0) / DT_S
        jump_in = np.linalg.norm(velocity[30] - velocity[29])
        jump_out = np.linalg.norm(velocity[34] - velocity[33])
        typical = np.median(np.linalg.norm(np.diff(velocity, axis=0), axis=1))

        assert jump_in < 5.0 * typical
        assert jump_out < 5.0 * typical

    def test_a_gap_touching_the_boundary_is_refused(self) -> None:
        values = smooth_positions(10)
        with pytest.raises(gp.GapPolicyError, match="boundary"):
            gp.hermite_fill(values, 0, 3)
        with pytest.raises(gp.GapPolicyError, match="boundary"):
            gp.hermite_fill(values, 7, 10)

    def test_a_bracket_without_an_outer_neighbour_degrades_to_a_straight_line(self) -> None:
        """With no sample beyond the bracket there is no velocity to honour, only a chord.

        The chord spans the whole gap, so it must be divided by that span before it is used
        as a per-frame velocity. Feeding the undivided chord in makes the cubic overshoot by
        the span again, which on a straight constant-speed path invents both an excursion
        and a reversal where the truth is a plain line.
        """
        truth = constant_velocity_positions(40)
        observed = truth.copy()
        observed[1:9] = np.nan  # left bracket is frame 0, which has nothing before it

        filled = gp.hermite_fill(observed, 1, 9)

        np.testing.assert_allclose(filled, truth[1:9], atol=1e-12)
        # Straight line at constant speed: no reversal and no acceleration of its own.
        rebuilt = observed.copy()
        rebuilt[1:9] = filled
        step = np.diff(rebuilt[:10, 0])
        assert (step > 0).all()
        acceleration = np.diff(rebuilt[:10, 0], n=2) / DT_S**2
        assert np.abs(acceleration).max() < 1e-6


class TestSquadRotationFill:
    def test_squad_tracks_the_true_rotation(self) -> None:
        truth = accelerating_rotation(60)
        filled = gp.squad_fill(truth, 30, 34)

        error = quat.multiply(quat.conjugate(truth[30:34]), filled)
        angle = 2.0 * np.linalg.norm(quat.log_unit(error)[:, 1:], axis=1)
        assert np.degrees(angle).max() < 0.05

    def test_squad_beats_slerp_on_angular_velocity_continuity(self) -> None:
        """Why SQUAD: SLERP holds angular velocity constant.

        The comparison is against the true angular speed rather than against the previous
        sample, because this trajectory is genuinely accelerating -- consecutive samples
        differ by roughly ``3 rad/s^2 * 0.01 s`` even for a perfect fill, which would swamp
        the discontinuity being looked for.
        """
        truth = accelerating_rotation(80)
        start, stop = 40, 50

        with_squad = truth.copy()
        with_squad[start:stop] = gp.squad_fill(truth, start, stop)
        with_slerp = truth.copy()
        with_slerp[start:stop] = slerp_fill(truth, start, stop)

        speed_truth = angular_speed(truth)
        # Speed sample i is centred on frame i+1, so the filled span covers these indices.
        span = slice(start - 2, stop)
        error_squad = np.abs(angular_speed(with_squad)[span] - speed_truth[span]).max()
        error_slerp = np.abs(angular_speed(with_slerp)[span] - speed_truth[span]).max()

        assert error_squad < 1e-3
        assert error_slerp > 20.0 * error_squad

    def test_a_second_gap_nearby_does_not_corrupt_the_end_tangents(self) -> None:
        """Reaching past one gap to find a sample makes that step several frames long.

        The tangent at a bracket is estimated from the nearest observed samples on the far
        side, which where gaps cluster can sit five frames away rather than one. Read as a
        one-frame step it reports several times the true rate, and the cubic then leaves the
        bracket far too fast. A constant-rate rotation exposes it: the correct fill is exact,
        so any departure is the estimator's own.
        """
        truth = constant_rate_rotation(60)
        observed = truth.copy()
        observed[25:35] = np.nan  # the gap being filled
        observed[36:40] = np.nan  # a second gap, one observed frame later

        filled = gp.squad_fill(observed, 25, 35)

        error = quat.multiply(quat.conjugate(truth[25:35]), filled)
        angle = 2.0 * np.linalg.norm(quat.log_unit(error)[:, 1:], axis=1)
        assert np.degrees(angle).max() < 0.05

    def test_a_neighbouring_gap_leaves_the_angular_rate_intact(self) -> None:
        """The consequence that reaches the gyroscope channel: a rate that should be flat."""
        truth = constant_rate_rotation(60)
        observed = truth.copy()
        observed[25:35] = np.nan
        observed[36:40] = np.nan
        solved = np.isfinite(observed).all(axis=1)

        result = gp.fill_pose_gaps(
            np.zeros((60, 3)), observed, solved, max_gap_frames=20
        )
        speed = angular_speed(result.quaternion)

        # Every filled frame carries the same 3 rad/s the observed ones do.
        assert np.abs(speed[24:40] - 3.0).max() < 1e-3

    def test_a_bracket_with_nothing_beyond_it_uses_the_chord_rate(self) -> None:
        """A bracket at the very start observes no rate of its own, but it is not still.

        Falling back to zero would have the cubic leave the bracket motionless and then
        catch up, which on a body turning steadily is a claim about the motion rather than
        an admission that none was measured. The average rate across the gap is what the
        data supports, and on a constant rate it is also the right answer.
        """
        truth = constant_rate_rotation(60)
        observed = truth.copy()
        observed[1:4] = np.nan  # left bracket is frame 0: nothing precedes it

        filled = gp.squad_fill(observed, 1, 4)

        error = quat.multiply(quat.conjugate(truth[1:4]), filled)
        angle = 2.0 * np.linalg.norm(quat.log_unit(error)[:, 1:], axis=1)
        assert np.degrees(angle).max() < 0.05

    def test_a_sign_flipped_input_does_not_take_the_long_way_round(self) -> None:
        truth = accelerating_rotation(60)
        flipped = truth.copy()
        flipped[34:] *= -1.0  # same rotations, opposite quaternion hemisphere

        filled = gp.squad_fill(flipped, 30, 34)
        error = quat.multiply(quat.conjugate(truth[30:34]), filled)
        angle = 2.0 * np.linalg.norm(quat.log_unit(error)[:, 1:], axis=1)
        # Modulo the hemisphere, the filled rotations must still match the truth.
        angle = np.minimum(angle, 2.0 * np.pi - angle)
        assert np.degrees(angle).max() < 0.05


class TestFillPolicy:
    def _series(self, frames: int, gaps: list[tuple[int, int]]):
        translation = smooth_positions(frames)
        quaternions = accelerating_rotation(frames)
        solved = np.ones(frames, dtype=bool)
        for start, stop in gaps:
            solved[start:stop] = False
            translation[start:stop] = np.nan
            quaternions[start:stop] = np.nan
        return translation, quaternions, solved

    def test_short_gaps_are_filled_and_marked_interpolated(self) -> None:
        translation, quaternions, solved = self._series(80, [(30, 34)])

        result = gp.fill_pose_gaps(translation, quaternions, solved, max_gap_frames=20)

        assert np.isfinite(result.translation[30:34]).all()
        assert np.isfinite(result.quaternion[30:34]).all()
        assert (result.status[30:34] == gp.FrameStatus.INTERPOLATED).all()
        assert (result.status[:30] == gp.FrameStatus.OBSERVED).all()

    def test_long_gaps_are_left_nan_and_marked_unavailable(self) -> None:
        translation, quaternions, solved = self._series(120, [(40, 80)])

        result = gp.fill_pose_gaps(translation, quaternions, solved, max_gap_frames=20)

        assert np.isnan(result.translation[40:80]).all()
        assert np.isnan(result.quaternion[40:80]).all()
        assert (result.status[40:80] == gp.FrameStatus.UNAVAILABLE).all()
        assert not result.usable[40:80].any()

    def test_a_gap_exactly_at_the_limit_is_filled_and_one_beyond_is_not(self) -> None:
        translation, quaternions, solved = self._series(120, [(30, 50), (70, 91)])

        result = gp.fill_pose_gaps(translation, quaternions, solved, max_gap_frames=20)

        assert (result.status[30:50] == gp.FrameStatus.INTERPOLATED).all()
        assert (result.status[70:91] == gp.FrameStatus.UNAVAILABLE).all()

    def test_a_gap_at_the_series_boundary_is_never_filled(self) -> None:
        translation, quaternions, solved = self._series(60, [(0, 3), (57, 60)])

        result = gp.fill_pose_gaps(translation, quaternions, solved, max_gap_frames=20)

        assert (result.status[0:3] == gp.FrameStatus.UNAVAILABLE).all()
        assert (result.status[57:60] == gp.FrameStatus.UNAVAILABLE).all()

    def test_a_negative_limit_is_refused(self) -> None:
        translation, quaternions, solved = self._series(20, [])
        with pytest.raises(gp.GapPolicyError, match="non-negative"):
            gp.fill_pose_gaps(translation, quaternions, solved, max_gap_frames=-1)


class TestWindowAcceptance:
    def test_a_clean_series_accepts_every_window(self) -> None:
        status = np.full(3000, gp.FrameStatus.OBSERVED, dtype=np.int8)

        result = gp.accept_windows(
            status, window_frames=1000, stride_frames=500, max_estimated_fraction=0.02
        )

        assert result.accepted.all()
        assert result.starts.tolist() == [0, 500, 1000, 1500, 2000]

    def test_any_unavailable_frame_rejects_its_windows(self) -> None:
        status = np.full(3000, gp.FrameStatus.OBSERVED, dtype=np.int8)
        status[1200] = gp.FrameStatus.UNAVAILABLE

        result = gp.accept_windows(
            status, window_frames=1000, stride_frames=500, max_estimated_fraction=0.02
        )

        rejected = result.starts[~result.accepted].tolist()
        assert rejected == [500, 1000]
        assert result.contains_unavailable[result.starts == 500][0]

    def test_the_estimated_fraction_bound_is_applied(self) -> None:
        status = np.full(2000, gp.FrameStatus.OBSERVED, dtype=np.int8)
        status[0:25] = gp.FrameStatus.INTERPOLATED  # 2.5% of a 1000-frame window

        result = gp.accept_windows(
            status, window_frames=1000, stride_frames=500, max_estimated_fraction=0.02
        )

        assert not result.accepted[0]
        assert result.estimated_fraction[0] == pytest.approx(0.025)
        assert result.accepted[-1]

    def test_a_series_shorter_than_one_window_yields_nothing(self) -> None:
        status = np.full(100, gp.FrameStatus.OBSERVED, dtype=np.int8)

        result = gp.accept_windows(
            status, window_frames=1000, stride_frames=500, max_estimated_fraction=0.02
        )

        assert result.starts.size == 0
        assert result.accepted.size == 0

    def test_policy_parameters_are_validated(self) -> None:
        status = np.full(100, gp.FrameStatus.OBSERVED, dtype=np.int8)
        with pytest.raises(gp.GapPolicyError, match="must be positive"):
            gp.accept_windows(
                status, window_frames=0, stride_frames=5, max_estimated_fraction=0.02
            )
        with pytest.raises(gp.GapPolicyError, match="\\[0, 1\\]"):
            gp.accept_windows(
                status, window_frames=10, stride_frames=5, max_estimated_fraction=1.5
            )
