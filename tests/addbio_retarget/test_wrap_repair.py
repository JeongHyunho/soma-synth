"""The wrap repair, exercised against a defect this test manufactures itself.

The source's mistake is reproducible in four lines: take a heading that crosses the branch cut,
wrap it, and low-pass filter the wrapped form. The filter sweeps the value through a whole turn
and pushes it past |pi|, which no wrapped angle can reach. So the fixture here does not describe
the defect, it commits it -- and then the repair has to undo it back to the signal the source
would have produced had it filtered the angle as an angle.

Two things are asserted that a looser test would miss. The fixture is checked to actually break
before the repair is asked to fix it, so the test cannot pass by repairing nothing; and a column
that never wraps is required to come back *bit-identical*, not merely close, because the repair
runs over every trial in the cohort and most of them are not broken.
"""

import numpy as np
import pytest
from scipy.signal import butter, filtfilt

from soma_synth.addbio_retarget.wrap_repair import (
    FilterSpec,
    WrapRepairError,
    repair_positions,
    wrap_counter,
)

RATE_HZ = 250.0
SPEC = FilterSpec(order=2, cutoff_hz=30.0, rate_hz=RATE_HZ)


def _heading(frames: int = 800) -> np.ndarray:
    """A continuous heading that walks across +pi and back, starting inside the branch."""
    t = np.arange(frames) / RATE_HZ
    return 2.9 + 0.9 * np.sin(2.0 * np.pi * 0.7 * t) + 0.25 * t


def _wrap(angles: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * angles))


def _still(frames: int = 800) -> np.ndarray:
    """A coordinate that never approaches the branch cut, so nothing should touch it."""
    t = np.arange(frames) / RATE_HZ
    return 0.4 * np.sin(2.0 * np.pi * 1.3 * t) + 0.03 * np.cos(2.0 * np.pi * 11.0 * t)


def _source_filter(series: np.ndarray) -> np.ndarray:
    numerator, denominator = butter(SPEC.order, SPEC.normalised_cutoff, btype="low")
    return filtfilt(numerator, denominator, series)


def _passes():
    """(kinematics, filtered) for a two-coordinate trial, one wrapping and one not."""
    heading, still = _heading(), _still()
    kinematics = np.column_stack([_wrap(heading), still])
    filtered = np.column_stack([_source_filter(_wrap(heading)), _source_filter(still)])
    return kinematics, filtered, heading


def test_wrap_counter_splits_the_angle_exactly():
    wrapped = _wrap(_heading())
    counter = wrap_counter(wrapped)
    assert np.count_nonzero(np.diff(counter)) > 0, "fixture never crosses the branch cut"
    np.testing.assert_array_equal(counter, np.rint(counter))
    np.testing.assert_allclose(
        wrapped, np.unwrap(wrapped) + 2.0 * np.pi * counter, rtol=0, atol=1e-12
    )


def test_the_fixture_really_commits_the_defect():
    """Without this the repair could be exercised against a signal that was never broken."""
    _, filtered, _ = _passes()
    corrupted = filtered[:, 0]
    assert np.abs(corrupted).max() > np.pi, "a wrapped angle that never escapes |pi| is not broken"
    corrupted_step = np.abs(np.diff(corrupted)).max()
    clean_step = np.abs(np.diff(_source_filter(_heading()))).max()
    assert corrupted_step > 20 * clean_step, (
        f"the filter barely noticed the wrap: {corrupted_step:.4f} vs {clean_step:.4f} rad"
    )


def test_repair_recovers_the_signal_the_source_should_have_written():
    kinematics, filtered, heading = _passes()
    result = repair_positions(kinematics, filtered, SPEC)

    assert result.repaired_indices == (0,)
    assert result.wrap_events[0] == np.count_nonzero(np.diff(wrap_counter(kinematics[:, 0])))

    # unwrap rebuilds the heading exactly here because it starts inside (-pi, pi]
    np.testing.assert_allclose(
        result.pos[:, 0], _source_filter(heading), rtol=0, atol=1e-9
    )
    assert np.abs(result.pos[:, 0]).max() > np.pi, (
        "the true heading does leave (-pi, pi]; the repair must not re-wrap it"
    )


def test_a_column_that_never_wraps_is_returned_bit_identical():
    kinematics, filtered, _ = _passes()
    result = repair_positions(kinematics, filtered, SPEC)
    assert np.array_equal(result.pos[:, 1], filtered[:, 1])


def test_a_trial_with_no_wrap_anywhere_is_returned_bit_identical():
    still = np.column_stack([_still(), _still() * 0.5])
    filtered = np.column_stack([_source_filter(still[:, 0]), _source_filter(still[:, 1])])
    result = repair_positions(still, filtered, SPEC)
    assert not result.repaired
    assert np.array_equal(result.pos, filtered)


def test_cutoff_at_or_above_nyquist_is_refused():
    kinematics, filtered, _ = _passes()
    with pytest.raises(WrapRepairError, match="Nyquist"):
        repair_positions(kinematics, filtered, FilterSpec(2, 30.0, 50.0))


def test_a_wrapping_trial_too_short_to_filter_is_refused_not_waved_through():
    short = np.array([[-3.10], [-3.13], [3.13], [3.10]])
    with pytest.raises(WrapRepairError, match="fewer than"):
        repair_positions(short, short.copy(), SPEC)


def test_a_short_trial_with_no_wrap_is_left_alone():
    short = np.array([[0.10], [0.11], [0.12], [0.13]])
    result = repair_positions(short, short.copy(), SPEC)
    assert not result.repaired
    assert np.array_equal(result.pos, short)


def test_passes_of_different_shapes_are_refused():
    with pytest.raises(WrapRepairError, match="shape"):
        repair_positions(np.zeros((10, 2)), np.zeros((10, 3)), SPEC)
