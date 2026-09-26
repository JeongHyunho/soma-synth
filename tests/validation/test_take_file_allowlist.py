"""A take directory may hold only what the registry declares.

The registry declares `take_deliverables`, and validation reads it: a companion file passes
because a rule allows it, not because no rule mentions it. These pin that rule.
"""

from __future__ import annotations

import json
from pathlib import Path

from soma_synth.contracts import dataset_profiles
from soma_synth.validation.checks import check_take_profile_conformance

REGISTRY = dataset_profiles.default_registry()


def _take(tmp_path: Path, source: str, files: list[str]) -> Path:
    take = tmp_path / f"{source}-take000"
    take.mkdir(parents=True)
    (take / "manifest.json").write_text(
        json.dumps({"source": {"source_name": source}}), encoding="utf-8")
    for name in files:
        (take / name).write_bytes(b"")
    return take


def _universal() -> list[str]:
    return sorted(REGISTRY.universal.take_deliverables - {"manifest.json"})


def test_the_registry_no_longer_carries_the_field_nothing_read():
    profile = REGISTRY.profile_for("prism")
    assert not hasattr(profile, "additive_small_npz_keys")


def test_universal_deliverables_alone_never_fail_for_any_source(tmp_path: Path):
    for source in REGISTRY.profiles:
        take = _take(tmp_path / source, source, _universal())
        findings = check_take_profile_conformance(take, "t0")
        assert not [f for f in findings if f.severity == "FAIL"], (source, findings)


def test_a_file_nobody_declared_fails(tmp_path: Path):
    take = _take(tmp_path, "gaitex", _universal() + ["notes.txt"])
    findings = check_take_profile_conformance(take, "t0")
    fails = [f for f in findings if f.severity == "FAIL"]
    assert len(fails) == 1
    assert "undeclared file 'notes.txt'" in fails[0].message


def test_prisms_declared_companions_are_allowed_and_only_for_prism(tmp_path: Path):
    files = _universal() + ["development_reference.npz", "heading_synth_reference.npz"]
    assert check_take_profile_conformance(_take(tmp_path / "p", "prism", files), "t0") == []

    findings = check_take_profile_conformance(_take(tmp_path / "h", "hknu", files), "t0")
    undeclared = {f.message.split("'")[1] for f in findings if f.severity == "FAIL"}
    assert undeclared == {"development_reference.npz", "heading_synth_reference.npz"}


def test_a_declared_companion_that_is_absent_is_a_warn_not_a_fail(tmp_path: Path):
    """The bundle offers the companion; a take does not owe it."""
    take = _take(tmp_path, "prism", _universal() + ["development_reference.npz"])
    findings = check_take_profile_conformance(take, "t0")
    assert [f.severity for f in findings] == ["WARN"]
    assert "heading_synth_reference.npz" in findings[0].message


def test_the_allowlist_is_the_union_the_registry_promises():
    allowed = REGISTRY.allowed_take_files("prism")
    assert REGISTRY.universal.take_deliverables <= allowed
    assert "heading_synth_reference.npz" in allowed
    assert "heading_synth_reference.npz" not in REGISTRY.allowed_take_files("amass")
