"""The two GAITEX corrections that are not about reading files.

One says a joint is measured only where a sensor measured it. The other pools a subject's
disagreeing skeletons instead of picking a favourite. Both are pure functions over small
structures, so most of this runs without the data; the two that check a published trial
skip when the GAITEX source folder does not resolve (``SOMA_SOURCE_ROOT``, else
``<SOMA_DATA_ROOT>/extracted``).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from soma_synth.addbio_retarget.smpl_correspondence import (
    ABSENT,
    DERIVED_LUMBAR,
    MEASURED,
    SmplJointSource,
    correspondence_for,
)
from soma_synth.gaitex_retarget import (
    knee_bracketing_pairs,
    open_trial,
    pool_segment_lengths,
    restrict_to_driven,
    undriven_summary,
)
from soma_synth.pipeline import paths


def _gaitex_root() -> pathlib.Path | None:
    """The GAITEX source folder, or None when neither variable resolves."""
    try:
        return paths.source_dir("gaitex")
    except paths.PathConfigError:
        return None


GAITEX_ROOT = _gaitex_root()
needs_gaitex_source = pytest.mark.skipif(
    GAITEX_ROOT is None or not GAITEX_ROOT.is_dir(),
    reason="SOMA_DATA_ROOT / SOMA_SOURCE_ROOT unset, or the GAITEX source is not under it"
)


def entry(index: int, name: str, provenance: str, body: str | None) -> SmplJointSource:
    return SmplJointSource(
        smpl_index=index,
        smpl_name=name,
        provenance=provenance,
        centre_joint=f"{name}_joint" if provenance != ABSENT else None,
        source_coordinates=(f"{name}_angle",) if provenance != ABSENT else (),
        source_body=body,
    )


class TestRestrictingToTheDrivenSet:
    def test_a_joint_whose_body_wears_a_sensor_is_left_alone(self) -> None:
        table = (entry(4, "left_knee", MEASURED, "tibia_l"),)
        assert restrict_to_driven(table, {"tibia_l"}) == table

    def test_a_joint_whose_body_wears_nothing_becomes_absent(self) -> None:
        table = (entry(16, "left_shoulder", MEASURED, "humerus_l"),)
        (restricted,) = restrict_to_driven(table, {"tibia_l"})
        assert restricted.provenance == ABSENT

    def test_a_downgraded_joint_keeps_nothing_to_be_fitted_to(self) -> None:
        """Absent must mean absent. A leftover centre would still pull the fit toward a joint
        the source never observed, which is the failure the downgrade exists to prevent."""
        table = (entry(20, "left_wrist", MEASURED, "hand_l"),)
        (restricted,) = restrict_to_driven(table, set())
        assert restricted.centre_joint is None
        assert restricted.source_coordinates == ()
        assert restricted.source_body is None

    def test_a_derived_lumbar_joint_survives_when_the_trunk_is_driven(self) -> None:
        table = (entry(3, "spine1", DERIVED_LUMBAR, "torso"),)
        (restricted,) = restrict_to_driven(table, {"torso"})
        assert restricted.provenance == DERIVED_LUMBAR

    def test_a_joint_the_skeleton_never_declared_is_not_reported_as_withheld(self) -> None:
        """Only a downgrade is news. A joint that was already absent was not taken away."""
        table = (entry(12, "neck", ABSENT, None),)
        assert undriven_summary(table, restrict_to_driven(table, set())) == {}

    def test_the_summary_names_the_body_that_carried_no_sensor(self) -> None:
        table = (entry(17, "right_shoulder", MEASURED, "humerus_r"),)
        summary = undriven_summary(table, restrict_to_driven(table, set()))
        assert "humerus_r" in summary["right_shoulder"]


@pytest.fixture(scope="class")
def restricted():
    """One published trial, opened once for the class that reads it. A module-level function:
    a class-scoped fixture defined as an instance method is deprecated (PytestRemovedIn10Warning)."""
    source = open_trial(GAITEX_ROOT, "austra", "gwo")
    table = correspondence_for(source.topology)
    return source, table, restrict_to_driven(table, source.driven_bodies)


@needs_gaitex_source
class TestAPublishedTrial:
    def test_the_driven_set_is_the_nine_bodies_the_metadata_names(self, restricted) -> None:
        source, _, _ = restricted
        assert set(source.driven_bodies) == {
            "pelvis", "torso", "femur_l", "femur_r", "tibia_l", "tibia_r",
            "calcn_l", "calcn_r", "toes_r",
        }

    def test_no_arm_joint_is_claimed_as_measured(self, restricted) -> None:
        """GAITEX carries no sensor above the sternum, so the shoulders, elbows and wrists are
        the solver's relaxation. Shipping them as measured would put a falsehood in every take."""
        _, _, table = restricted
        arms = [e for e in table if e.smpl_index in (16, 17, 18, 19, 20, 21)]
        assert [e.provenance for e in arms] == [ABSENT] * 6

    def test_the_right_toe_is_driven_and_the_left_is_not(self, restricted) -> None:
        """The asymmetry is the source's, not a special case: one unit sits on toes_r."""
        _, _, table = restricted
        by_index = {e.smpl_index: e for e in table}
        assert by_index[11].provenance == MEASURED
        assert by_index[10].provenance == ABSENT

    def test_the_legs_and_spine_survive_intact(self, restricted) -> None:
        _, _, table = restricted
        kept = [e.smpl_index for e in table if e.provenance != ABSENT]
        assert kept == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]


class TestPoolingASubjectsTrials:
    def test_the_pooled_length_is_the_median_of_what_the_trials_offered(self) -> None:
        pooled = pool_segment_lengths([{(1, 4): 0.44}, {(1, 4): 0.45}, {(1, 4): 0.47}])
        assert pooled.lengths_m[(1, 4)] == pytest.approx(0.45)

    def test_the_pooled_length_lies_inside_the_range_the_trials_spanned(self) -> None:
        values = [0.4441, 0.4479, 0.4523, 0.4666]
        pooled = pool_segment_lengths([{(2, 5): v} for v in values])
        assert min(values) <= pooled.lengths_m[(2, 5)] <= max(values)

    def test_the_spread_the_pooling_absorbed_is_reported_not_discarded(self) -> None:
        """A femur of 0.45 m whose source offered values 22 mm apart is a different number from
        one every trial agreed on, and a consumer has to be able to tell them apart."""
        pooled = pool_segment_lengths([{(2, 5): 0.4441}, {(2, 5): 0.4666}])
        assert pooled.spread_m[(2, 5)] == pytest.approx(0.0225, abs=1e-6)

    def test_a_segment_only_some_trials_measured_is_pooled_over_those(self) -> None:
        pooled = pool_segment_lengths([{(1, 4): 0.44}, {}, {(1, 4): 0.46}])
        assert pooled.lengths_m[(1, 4)] == pytest.approx(0.45)
        assert pooled.trials_pooled == 3

    def test_a_non_finite_length_does_not_enter_the_pool(self) -> None:
        pooled = pool_segment_lengths([{(1, 4): 0.44}, {(1, 4): float("nan")}])
        assert pooled.lengths_m[(1, 4)] == pytest.approx(0.44)

    def test_pooling_nothing_is_refused_rather_than_returning_an_empty_body(self) -> None:
        with pytest.raises(ValueError):
            pool_segment_lengths([])


class TestTheKneeBracket:
    def test_the_hip_to_ankle_span_is_added_on_each_side(self) -> None:
        pairs = {
            (1, 4): ("hip_l", "walker_knee_l"), (4, 7): ("walker_knee_l", "ankle_l"),
            (2, 5): ("hip_r", "walker_knee_r"), (5, 8): ("walker_knee_r", "ankle_r"),
        }
        extended = knee_bracketing_pairs(pairs)
        assert extended[(1, 7)] == ("hip_l", "ankle_l")
        assert extended[(2, 8)] == ("hip_r", "ankle_r")

    def test_the_two_parts_are_kept_beside_the_span(self) -> None:
        """Dropping them would leave the knee free to sit anywhere along a limb of the right
        length, which trades one error for another."""
        pairs = {(1, 4): ("hip_l", "walker_knee_l"), (4, 7): ("walker_knee_l", "ankle_l")}
        extended = knee_bracketing_pairs(pairs)
        assert extended[(1, 4)] == pairs[(1, 4)]
        assert extended[(4, 7)] == pairs[(4, 7)]

    def test_a_side_missing_one_end_gets_no_span(self) -> None:
        pairs = {(1, 4): ("hip_l", "walker_knee_l")}
        assert (1, 7) not in knee_bracketing_pairs(pairs)

    def test_the_input_is_not_mutated(self) -> None:
        pairs = {(1, 4): ("hip_l", "walker_knee_l"), (4, 7): ("walker_knee_l", "ankle_l")}
        before = dict(pairs)
        knee_bracketing_pairs(pairs)
        assert pairs == before


@needs_gaitex_source
def test_the_bracket_is_more_stable_than_its_parts_on_a_published_subject() -> None:
    """The measurement the pooling rests on, checked on the data rather than quoted.

    The femur and tibia trade millimetres because the knee centre wanders, so the span that
    brackets the knee should vary less than either part. Measured across the cohort the relative
    spread roughly halves; here it is asserted for one subject on the right side.
    """
    from soma_synth.addbio_retarget.osim_kinematics import parse_kinematics

    lengths = {"femur": [], "tibia": [], "span": []}
    for condition in sorted(p.name for p in (GAITEX_ROOT / "austra").iterdir() if p.is_dir()):
        source = open_trial(GAITEX_ROOT, "austra", condition)
        centres = parse_kinematics(source.osim_text).forward_batch({}, frames=1).joint_centres
        hip, knee, ankle = centres["hip_r"][0], centres["walker_knee_r"][0], centres["ankle_r"][0]
        lengths["femur"].append(float(np.linalg.norm(hip - knee)))
        lengths["tibia"].append(float(np.linalg.norm(knee - ankle)))
        lengths["span"].append(lengths["femur"][-1] + lengths["tibia"][-1])

    def relative_spread(values: list[float]) -> float:
        return (max(values) - min(values)) / float(np.median(values))

    assert relative_spread(lengths["span"]) < max(
        relative_spread(lengths["femur"]), relative_spread(lengths["tibia"])
    )
