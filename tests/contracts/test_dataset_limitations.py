"""The known-limitations file has one shape for every bundle, and the bundle copy cannot drift
from the repository copy without a finding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from soma_synth.contracts import dataset_limitations as dl

GOOD = {
    "id": "prism.insole_heading_unreferenced",
    "kind": "hardware_limit",
    "severity": "caution",
    "scope": {"level": "site", "sites": ["foot_l", "foot_r"]},
    "statement": "The insoles are six-axis; their heading is the gyro integral and drifts.",
    "consequence": "Do not use the measured insole heading as an absolute reference.",
    "evidence": "small_reference.npz: imu_orientation_absolute_heading = [T,T,T,T,T,T,F,F]",
    "status": "mitigated",
    "since": "2026-09-09",
    "mitigation": "heading_synth_reference.npz carries a heading transferred from the shank.",
}


def _config(tmp_path: Path, bundles: dict) -> Path:
    path = tmp_path / "known_limitations_v1.yaml"
    path.write_text(yaml.safe_dump({"schema": dl.SCHEMA, "bundles": bundles},
                                   allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_a_complete_entry_has_no_problems():
    assert dl.problems_in_entry(GOOD, "e") == []


@pytest.mark.parametrize("field", ["id", "kind", "severity", "statement", "consequence",
                                   "evidence", "status", "since"])
def test_each_required_field_is_required(field):
    bad = {**GOOD, field: ""}
    assert any(field in p for p in dl.problems_in_entry(bad, "e"))


def test_enums_are_closed():
    for field, value in (("kind", "bug"), ("severity", "high"), ("status", "wontfix")):
        assert dl.problems_in_entry({**GOOD, field: value}, "e"), field


def test_a_mitigated_entry_must_say_how_and_a_fixed_one_where():
    assert any("mitigation" in p for p in dl.problems_in_entry({**GOOD, "mitigation": ""}, "e"))
    fixed = {**GOOD, "status": "fixed", "fixed_in": ""}
    assert any("fixed_in" in p for p in dl.problems_in_entry(fixed, "e"))


def test_a_site_scope_must_name_sites_and_an_unknown_scope_field_is_refused():
    assert any("sites" in p for p in dl.problems_in_entry(
        {**GOOD, "scope": {"level": "site"}}, "e"))
    assert any("unknown scope field" in p for p in dl.problems_in_entry(
        {**GOOD, "scope": {"level": "bundle", "channels": ["x"]}}, "e"))


def test_an_unknown_top_level_field_is_refused():
    """A third shape must not grow by accretion; that is how source_attribution drifted."""
    assert any("unknown field" in p for p in dl.problems_in_entry({**GOOD, "notes": "x"}, "e"))


def test_duplicate_ids_in_one_bundle_are_refused(tmp_path: Path):
    with pytest.raises(dl.LimitationsError, match="duplicate id"):
        dl.load_config(_config(tmp_path, {"b": [GOOD, GOOD]}))


def test_render_is_sorted_and_round_trips_through_the_file_check(tmp_path: Path):
    other = {**GOOD, "id": "prism.a_first", "status": "open"}
    del other["mitigation"]
    config = dl.load_config(_config(tmp_path, {"prism_faithful_full_v1": [GOOD, other]}))
    rendered = dl.render("prism_faithful_full_v1", config.bundles["prism_faithful_full_v1"],
                         generated_utc="2026-09-14T00:00:00Z")
    assert [e["id"] for e in rendered["entries"]] == ["prism.a_first",
                                                      "prism.insole_heading_unreferenced"]
    path = tmp_path / dl.FILE_NAME
    path.write_text(json.dumps(rendered), encoding="utf-8")
    assert dl.problems_in_file(path, "prism_faithful_full_v1", config) == []


def test_a_bundle_copy_that_differs_from_the_config_is_a_finding(tmp_path: Path):
    config = dl.load_config(_config(tmp_path, {"b1": [GOOD]}))
    rendered = dl.render("b1", config.bundles["b1"], generated_utc="2026-09-14T00:00:00Z")
    rendered["entries"][0]["severity"] = "info"          # someone softened it by hand
    path = tmp_path / dl.FILE_NAME
    path.write_text(json.dumps(rendered), encoding="utf-8")
    problems = dl.problems_in_file(path, "b1", config)
    assert any("differs from" in p for p in problems)


def test_a_bundle_copy_naming_the_wrong_bundle_is_a_finding(tmp_path: Path):
    config = dl.load_config(_config(tmp_path, {"b1": [GOOD]}))
    rendered = dl.render("b1", config.bundles["b1"], generated_utc="2026-09-14T00:00:00Z")
    path = tmp_path / dl.FILE_NAME
    path.write_text(json.dumps(rendered), encoding="utf-8")
    assert any("names bundle" in p for p in dl.problems_in_file(path, "b2", config))


def test_the_shipped_config_parses_and_covers_every_registered_bundle():
    """The repository's own config must load, and name each bundle the registry knows about."""
    config = dl.load_config()
    expected = {"prism_faithful_full", "hknu_unified8", "gaitex_unified8",
                "amass_faithful_full", "addbio_unified8"}
    assert expected <= set(config.bundles)
    for name, entries in config.bundles.items():
        assert entries, f"{name} declares no limitations; every bundle has at least rights"
        assert any(e.kind == "rights" for e in entries), f"{name} lacks a rights entry"
