"""A run's validation report does not warn about files the same run writes afterwards.

The validate step judges conformance before it writes VALIDATION_REPORT.json and
validation_ledger.json, and before the readme step writes README.md, so every freshly built
bundle's report carried three WARNs "missing ... not distributable" although the files were there
when the run ended. The runner now tells validation which of them are still to come; validate on
its own, which writes none of them, still warns. README.md is to come only after a validation that
passed: a failed one stops the run before the readme step, and its report keeps that warning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soma_synth import cli
from soma_synth.pipeline import runner as runner_mod
from soma_synth.validation import checks
from soma_synth.validation.runner import validate

from test_validation_checks import make_dataset

DISTRIBUTION = ("README.md", "VALIDATION_REPORT.json", "validation_ledger.json")


def _not_distributable(findings) -> list[str]:
    return sorted(f["message"].split(";")[0].removeprefix("missing ") for f in findings
                  if f["severity"] == "WARN" and "not distributable" in f["message"])


def _report(bundle: Path) -> dict:
    return json.loads((bundle / "VALIDATION_REPORT.json").read_text(encoding="utf-8"))


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    for name in ("SOMA_DATA_ROOT", "SOMA_SOURCE_ROOT", "SOMA_BODY_MODEL_DIR"):
        monkeypatch.delenv(name, raising=False)
    return make_dataset(tmp_path / "data" / "tmp" / "amass_x", n=1)


@pytest.mark.parametrize("stop_after, warned", [
    ("readme", []),                     # the readme step writes README.md after the validation
    ("validate", ["README.md"]),        # nothing in this run writes it
    ("register", []),
])
def test_a_run_does_not_warn_about_what_it_writes_afterwards(bundle, tmp_path, stop_after,
                                                             warned):
    root = tmp_path / "data"
    result = runner_mod.PipelineRunner(source_name="amass", data_root=root,
                                       dataset_dir=bundle).run(
        start_at="validate", stop_after=stop_after, catalog_dir=root / "catalog",
        log=lambda message: None)
    assert result.ok, result.summary()
    report = _report(bundle)
    assert _not_distributable(report["findings"]) == warned
    assert report["counts"]["WARN"] == len([f for f in report["findings"]
                                            if f["severity"] == "WARN"])
    for name in DISTRIBUTION:
        assert (bundle / name).is_file() is (name != "README.md" or stop_after != "validate")


@pytest.mark.parametrize("stop_after", ["readme", "register"])
def test_a_run_whose_validation_fails_still_warns_about_the_readme_it_never_writes(
        bundle, tmp_path, stop_after):
    """The validate step excused README.md whenever the plan had the readme step, but a failed
    validation stops the run before that step: the bundle was left without README.md and its
    report said nothing. The warning stays in a report that failed; the report and the ledger,
    which the step writes whatever the outcome, are still not warned about."""
    (bundle / "amass-take000" / "anthro_reference.npz").unlink()
    root = tmp_path / "data"
    result = runner_mod.PipelineRunner(source_name="amass", data_root=root,
                                       dataset_dir=bundle).run(
        start_at="validate", stop_after=stop_after, catalog_dir=root / "catalog",
        log=lambda message: None)
    assert [(o.step, o.status) for o in result.outcomes] == [("validate", "failed")]
    report = _report(bundle)
    assert report["ok"] is False
    assert _not_distributable(report["findings"]) == ["README.md"]
    assert report["counts"]["WARN"] == len([f for f in report["findings"]
                                            if f["severity"] == "WARN"])
    assert not (bundle / "README.md").exists()
    assert not (root / "catalog").exists()


def test_the_withdrawn_warning_is_the_one_conformance_gives(bundle):
    """The runner finds the README.md warning by equality with checks.missing_distribution_file:
    the finding the conformance judgement gives for a missing README.md."""
    findings = checks.check_dataset_profile_conformance(bundle)
    assert checks.missing_distribution_file("README.md") in findings
    assert checks.missing_distribution_file("VALIDATION_REPORT.json") in findings


def test_validate_on_its_own_still_warns_about_what_the_bundle_lacks(bundle, capsys):
    report = bundle.parent / "report.json"
    assert cli.main(["validate", str(bundle), "--level", "L2", "--report", str(report)]) == 0
    assert _not_distributable(json.loads(report.read_text(encoding="utf-8"))["findings"]) == \
        sorted(DISTRIBUTION)
    # its own report and ledger written into the bundle are not missing; README.md still is
    assert cli.main(["validate", str(bundle), "--level", "L2",
                     "--report", str(bundle / "VALIDATION_REPORT.json"),
                     "--ledger", str(bundle / "validation_ledger.json")]) == 0
    capsys.readouterr()
    assert _not_distributable(_report(bundle)["findings"]) == ["README.md"]


def test_pending_files_excuse_only_the_distribution_files(bundle):
    """A pending name that is a file the generator owes (KNOWN_LIMITATIONS.json) is still a FAIL
    when it is missing: only the files a caller writes after validating can be pending."""
    (bundle / "KNOWN_LIMITATIONS.json").unlink()
    findings = checks.check_dataset_profile_conformance(
        bundle, pending_files=["KNOWN_LIMITATIONS.json", "README.md"])
    messages = [(f.severity, f.message) for f in findings]
    assert ("FAIL", "missing bundle file KNOWN_LIMITATIONS.json") in messages
    assert not any("README.md" in message for _, message in messages)
    assert sum("not distributable" in message for _, message in messages) == 2
    report = validate(bundle, level="L0", conformance=True, pending_files=DISTRIBUTION)
    assert not any("not distributable" in f.message for f in report.findings)
