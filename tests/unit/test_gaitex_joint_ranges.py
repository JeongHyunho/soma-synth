"""The gate that would have caught a knee turning through a full revolution.

GAITEX publishes trials whose own inverse kinematics failed. Nothing downstream noticed: the
retarget places a limb wherever it is told, and the unified8 L2 band bounds gyroscope magnitude,
which a joint rotating smoothly through 360 degrees passes. An earlier units guard checked two
trials against physiology and both happened to be clean.

These tests are mostly synthetic on purpose -- the gate must be checkable without the source
data, which is where earlier coverage went missing.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from soma_synth.gaitex_retarget import joint_ranges as jr

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = REPOSITORY_ROOT / "configs" / "datasets" / "gaitex_joint_ranges_v1.yaml"

NAMES = ["pelvis_tx", "knee_angle_r", "knee_angle_l", "ankle_angle_r"]


def limits(spans: dict[str, float] | None = None, **bounds) -> jr.JointRangeLimits:
    return jr.JointRangeLimits(
        config_id="test", version="0", limits_deg=dict(bounds),
        span_limits_deg=dict(spans or {}))


def radians(rows: list[list[float]]) -> np.ndarray:
    """Degrees in, radians out -- except pelvis_tx, which is metres and never converted."""
    values = np.asarray(rows, dtype=np.float64)
    values[:, 1:] = np.deg2rad(values[:, 1:])
    return values


class TestTheShippedConfig:
    """The bounds are data, so the file is part of the behaviour and is tested as such."""

    def test_the_config_parses_and_bounds_the_joints_that_failed(self) -> None:
        loaded = jr.load_limits(CONFIG)
        assert loaded.config_id == "gaitex_joint_ranges_v1"
        # The coordinates found broken must actually be bounded.
        for name in ("knee_angle_r", "knee_angle_l", "ankle_angle_r", "lumbar_extension"):
            assert name in loaded.limits_deg, name
        for name, (low, high) in loaded.limits_deg.items():
            assert low < high, name

    def test_the_config_declares_that_it_authorises_nothing(self) -> None:
        """A threshold file in this repository is not a permission."""
        config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
        assert config["activation_state"] == "non_authorizing"
        assert config["distribution_scope"] == "internal_only"

    def test_a_walking_knee_passes_and_a_revolving_one_does_not(self) -> None:
        """Calibration, against the two cases that matter, using the shipped bounds."""
        loaded = jr.load_limits(CONFIG)
        low, high = loaded.limits_deg["knee_angle_r"]
        assert low <= 5.0 and 65.0 <= high        # ordinary gait sits inside
        assert not (low <= -180.0) and not (180.0 <= high)   # katee/gwo does not


class TestMeasuring:
    def test_a_coordinate_inside_its_bounds_reports_nothing(self) -> None:
        values = radians([[0.9, 20.0, 20.0, 5.0], [0.9, 40.0, 40.0, -5.0]])
        report = jr.measure(values, NAMES, limits(knee_angle_r=(-15.0, 155.0)))
        assert report.clean
        assert report.frames_outside == 0
        assert report.worst is None
        assert report.per_coordinate["knee_angle_r"]["frames_outside"] == 0

    def test_a_knee_through_a_revolution_is_counted(self) -> None:
        values = radians([[0.9, 20.0, 0.0, 0.0],
                          [0.9, -180.0, 0.0, 0.0],
                          [0.9, 180.0, 0.0, 0.0]])
        report = jr.measure(values, NAMES, limits(knee_angle_r=(-15.0, 155.0)))
        assert report.frames_outside == 2
        assert report.worst == "knee_angle_r"
        entry = report.per_coordinate["knee_angle_r"]
        assert entry["min_deg"] == pytest.approx(-180.0)
        assert entry["max_deg"] == pytest.approx(180.0)
        assert entry["fraction_outside"] == pytest.approx(2 / 3)

    def test_one_frame_outside_two_joints_is_one_frame_not_two(self) -> None:
        """`frames_outside_any` is a frame count, so it must not double-count a bad frame."""
        values = radians([[0.9, 200.0, 200.0, 0.0], [0.9, 20.0, 20.0, 0.0]])
        report = jr.measure(
            values, NAMES,
            limits(knee_angle_r=(-15.0, 155.0), knee_angle_l=(-15.0, 155.0)))
        assert report.frames_outside == 1
        assert report.per_coordinate["knee_angle_r"]["frames_outside"] == 1
        assert report.per_coordinate["knee_angle_l"]["frames_outside"] == 1

    def test_the_bounds_are_inclusive_so_a_joint_exactly_at_the_limit_passes(self) -> None:
        values = radians([[0.9, 155.0, -15.0, 0.0]])
        report = jr.measure(
            values, NAMES,
            limits(knee_angle_r=(-15.0, 155.0), knee_angle_l=(-15.0, 155.0)))
        assert report.clean

    def test_a_bounded_coordinate_the_model_does_not_have_is_reported_absent(self) -> None:
        """Checked-and-clean must be distinguishable from never-looked-at."""
        values = radians([[0.9, 20.0, 20.0, 0.0]])
        report = jr.measure(values, NAMES, limits(hip_flexion_r=(-40.0, 140.0)))
        assert report.coordinates_absent == ("hip_flexion_r",)
        assert report.coordinates_checked == ()
        assert report.clean
        assert report.as_manifest_block()["coordinates_absent_from_model"] == ["hip_flexion_r"]

    def test_the_mask_restricts_the_measurement_to_the_frames_a_take_keeps(self) -> None:
        """A trial's broken frames must not be attributed to a take that excluded them."""
        values = radians([[0.9, 900.0, 0.0, 0.0], [0.9, 20.0, 0.0, 0.0]])
        mask = np.array([False, True])
        report = jr.measure(values, NAMES, limits(knee_angle_r=(-15.0, 155.0)), frame_mask=mask)
        assert report.frames_checked == 1
        assert report.clean

    def test_the_translational_coordinate_is_left_alone(self) -> None:
        """pelvis_tx is metres. Bounding it in degrees would be a category error, so the gate
        simply does not name it, and this pins that it stays unnamed."""
        loaded = jr.load_limits(CONFIG)
        for name in ("pelvis_tx", "pelvis_ty", "pelvis_tz"):
            assert name not in loaded.limits_deg

    def test_a_shape_that_does_not_match_the_names_is_refused(self) -> None:
        with pytest.raises(ValueError):
            jr.measure(np.zeros((3, 2)), NAMES, limits(knee_angle_r=(-15.0, 155.0)))

    def test_an_empty_limit_table_is_refused_rather_than_passing_everything(self) -> None:
        """A gate that checks nothing must not look like a gate that found nothing."""
        empty = pathlib.Path(__file__).parent / "_empty_limits.json"
        empty.write_text(json.dumps({"config_id": "x", "version": "0", "limits_deg": {}}),
                         encoding="utf-8")
        try:
            with pytest.raises(ValueError):
                jr.load_limits(empty)
        finally:
            empty.unlink()


class TestTheManifestBlock:
    def test_the_block_says_it_dropped_nothing(self) -> None:
        """The policy is measure-and-record, and the artifact has to state which it was."""
        values = radians([[0.9, 900.0, 0.0, 0.0]])
        block = jr.measure(values, NAMES, limits(knee_angle_r=(-15.0, 155.0))).as_manifest_block()
        assert "no frame was dropped" in block["policy"]
        assert block["config_id"] == "gaitex_joint_ranges_v1"
        assert block["frames_outside_any"] == 1
        assert block["fraction_outside_any"] == pytest.approx(1.0)

    def test_an_empty_take_does_not_divide_by_zero(self) -> None:
        values = radians([[0.9, 20.0, 0.0, 0.0]])
        block = jr.measure(
            values, NAMES, limits(knee_angle_r=(-15.0, 155.0)),
            frame_mask=np.array([False]),
        ).as_manifest_block()
        assert block["frames_checked"] == 0
        assert block["fraction_outside_any"] == 0.0


class TestSpanSeparatesBrokenDataFromPosture:
    """The distinction the first version of this gate did not make.

    elodie/rd holds its lumbar between -77.5 and -36.8 degrees: outside the bound on 97 per cent
    of frames, but a span of 41 degrees, which is ordinary trunk motion at an offset. katee/gwo
    sweeps a knee from -180 to +180: a span of 360, which no knee does. Both leave the value
    bound; only one is broken data, and a gate that cannot tell them apart gets ignored.
    """

    SPANS = {"knee_angle_r": 170.0, "lumbar_extension": 120.0}

    def test_a_sustained_offset_is_not_called_a_failed_solve(self) -> None:
        values = radians([[0.9, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0]])
        values[:, 1] = np.deg2rad([-77.5, -36.8])          # the elodie/rd shape
        report = jr.measure(
            values, NAMES, limits(self.SPANS, knee_angle_r=(-15.0, 155.0)))
        assert report.frames_outside == 2                  # the level is outside
        assert not report.solve_came_apart                 # but the span is 41 degrees
        assert report.per_coordinate["knee_angle_r"]["span_deg"] == pytest.approx(40.7)

    def test_a_joint_traversing_more_than_its_range_is_called_broken(self) -> None:
        values = radians([[0.9, -180.0, 0.0, 0.0], [0.9, 180.0, 0.0, 0.0]])
        report = jr.measure(
            values, NAMES, limits(self.SPANS, knee_angle_r=(-15.0, 155.0)))
        assert report.solve_came_apart
        assert report.failed_solve == ("knee_angle_r",)
        assert report.per_coordinate["knee_angle_r"]["span_exceeds_limit"] is True

    def test_a_coordinate_with_no_declared_span_is_never_called_broken(self) -> None:
        """Absence of a bound must not read as a passed check."""
        values = radians([[0.9, -180.0, 0.0, 0.0], [0.9, 180.0, 0.0, 0.0]])
        report = jr.measure(values, NAMES, limits({}, knee_angle_r=(-15.0, 155.0)))
        assert not report.solve_came_apart
        assert report.per_coordinate["knee_angle_r"]["span_limit_deg"] is None

    def test_the_manifest_block_tells_a_consumer_which_field_to_filter_on(self) -> None:
        values = radians([[0.9, -180.0, 0.0, 0.0], [0.9, 180.0, 0.0, 0.0]])
        block = jr.measure(
            values, NAMES, limits(self.SPANS, knee_angle_r=(-15.0, 155.0))).as_manifest_block()
        assert block["solve_came_apart"] is True
        assert block["coordinates_whose_span_is_impossible"] == ["knee_angle_r"]
        assert "solve_came_apart" in block["how_to_read_this"]

    def test_the_shipped_config_declares_a_span_for_every_bounded_joint(self) -> None:
        loaded = jr.load_limits(CONFIG)
        assert set(loaded.span_limits_deg) == set(loaded.limits_deg)
        for name, span in loaded.span_limits_deg.items():
            low, high = loaded.limits_deg[name]
            # A span limit tighter than the value bound would make the two contradict.
            assert span >= (high - low) * 0.5, name
