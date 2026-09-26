"""The real-data tests' skip reasons tell an unset SOMA_DATA_ROOT from a PC without the data.

Both used to read "SOMA_DATA_ROOT unset or ... source missing", so a run that lost the data tests
to a shell without the variable looked the same as a run on a PC that never had the sources.
"""

from __future__ import annotations

import data_root_skips as skips
import pytest

from soma_synth.pipeline import paths


@pytest.fixture
def no_root(monkeypatch):
    monkeypatch.delenv(paths.DATA_ROOT_ENV, raising=False)
    return monkeypatch


def test_unset_says_the_test_did_not_run(no_root):
    root, problem = skips.resolve()
    assert root is None
    reason = skips.skip_reason("the PRISM take x.pkl", (root, problem))
    assert reason.startswith("SOMA_DATA_ROOT unset:")
    assert "did not run" in reason and "the PRISM take x.pkl" in reason


def test_blank_counts_as_unset(no_root):
    no_root.setenv(paths.DATA_ROOT_ENV, "   ")
    assert skips.skip_reason("x", skips.resolve()).startswith("SOMA_DATA_ROOT unset:")


def test_set_but_unusable_is_its_own_reason(no_root, tmp_path):
    no_root.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "absent"))
    reason = skips.skip_reason("x", skips.resolve())
    assert reason.startswith("SOMA_DATA_ROOT unusable:")
    assert "does not exist" in reason


def test_set_with_the_data_missing_names_the_root(no_root, tmp_path):
    no_root.setenv(paths.DATA_ROOT_ENV, str(tmp_path))
    status = skips.resolve()
    assert status == (tmp_path, None)
    reason = skips.skip_reason("the AMASS source extracted/amass/KIT", status)
    assert reason.startswith("SOMA_DATA_ROOT set, data missing:")
    assert "extracted/amass/KIT" in reason and str(tmp_path) in reason


def test_the_three_reasons_are_distinct(no_root, tmp_path):
    unset = skips.skip_reason("x", skips.resolve())
    no_root.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "absent"))
    unusable = skips.skip_reason("x", skips.resolve())
    no_root.setenv(paths.DATA_ROOT_ENV, str(tmp_path))
    missing = skips.skip_reason("x", skips.resolve())
    prefixes = {reason.split(":", 1)[0] for reason in (unset, unusable, missing)}
    assert len(prefixes) == 3


def test_an_unresolved_root_holds_nothing():
    """What the data tests probe under it must not exist, so they skip rather than error."""
    assert not (skips.UNRESOLVED / "extracted" / "amass" / "KIT").exists()
    assert not (skips.UNRESOLVED / "extracted/prism/subj001/take002.pkl").exists()


def test_the_marker_skips_only_when_absent():
    assert skips.needs_data(True, "x").args == (False,)
    marker = skips.needs_data(False, "x")
    assert marker.args == (True,)
    assert marker.kwargs["reason"] == skips.skip_reason("x")
