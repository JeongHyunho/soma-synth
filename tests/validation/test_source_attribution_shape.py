"""`source_attribution` has a shape now, and the shipped bundles have to fit it.

ADR-0040 D1 made the field universal without saying what goes in it, and by the time anyone
looked the two bundles that had one had grown different shapes -- gaitex eight keys, addbio six.
Nothing objected, because nothing was checking. These tests are the objection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soma_synth.contracts import dataset_profiles
from soma_synth.pipeline import paths
from soma_synth.validation.checks import _check_source_attribution_shape

REGISTRY = dataset_profiles.default_registry()

#: The five keys gaitex and addbio already agreed on, before anyone wrote them down.
EXPECTED_UNIVERSAL = {
    "dataset_citation",
    "publication",
    "license_id",
    "stable_source_id",
    "changes_made",
}

#: What the two bundles that shipped an attribution actually carry. If a future change to the
#: registry stops accounting for one of these, the bundle on disk becomes non-conformant while
#: nothing about it changed -- so the observed shapes are pinned here, not paraphrased.
SHIPPED = {
    "gaitex": EXPECTED_UNIVERSAL | {"source_name", "source_version", "attribution_note"},
    "addbiomechanics": EXPECTED_UNIVERSAL | {"retarget_content_hashes"},
    "hknu": EXPECTED_UNIVERSAL | {"source_name", "source_version", "attribution_note"},
    # amass and prism carry no source_version: neither licence record fingerprints one, and
    # writing an unrecorded version would be the mistake the whole backfill was shaped to avoid.
    "amass": EXPECTED_UNIVERSAL | {"source_name", "attribution_note"},
    "prism": EXPECTED_UNIVERSAL | {"source_name", "attribution_note"},
}


def _manifest(source: str, attribution) -> dict:
    return {"source": {"source_name": source}, "source_attribution": attribution}


def test_the_universal_set_is_the_intersection_of_the_two_bundles_that_had_one():
    assert set(REGISTRY.universal.source_attribution_keys) == EXPECTED_UNIVERSAL


@pytest.mark.parametrize("source", sorted(SHIPPED))
def test_every_key_a_shipped_bundle_carries_is_accounted_for(source):
    assert SHIPPED[source] <= REGISTRY.allowed_source_attribution_keys(source)


@pytest.mark.parametrize("source", sorted(SHIPPED))
def test_a_shipped_shape_passes_the_check(source):
    manifest = _manifest(source, {k: "x" for k in SHIPPED[source]})
    profile = REGISTRY.profile_for(source)
    assert _check_source_attribution_shape(manifest, "t0", profile) == []


def test_a_missing_required_key_fails():
    """An attribution naming a licence but not what was changed does not discharge the
    obligation it exists to record."""
    attribution = {k: "x" for k in EXPECTED_UNIVERSAL - {"changes_made"}}
    findings = _check_source_attribution_shape(
        _manifest("gaitex", attribution), "t0", REGISTRY.profile_for("gaitex")
    )
    assert [f.severity for f in findings] == ["FAIL"]
    assert "changes_made" in findings[0].message


def test_a_key_no_profile_declares_fails():
    """The drift this file exists to stop: a new bundle inventing a third shape in silence."""
    attribution = {k: "x" for k in EXPECTED_UNIVERSAL} | {"invented_key": "x"}
    findings = _check_source_attribution_shape(
        _manifest("gaitex", attribution), "t0", REGISTRY.profile_for("gaitex")
    )
    assert [f.severity for f in findings] == ["FAIL"]
    assert "invented_key" in findings[0].message


def test_one_bundles_additive_key_is_not_allowed_in_another():
    """addbio's retarget hashes are an addbio fact. Declared per source, not globally, or the
    registry stops being a whitelist."""
    attribution = {k: "x" for k in EXPECTED_UNIVERSAL} | {"retarget_content_hashes": "x"}
    findings = _check_source_attribution_shape(
        _manifest("gaitex", attribution), "t0", REGISTRY.profile_for("gaitex")
    )
    assert len(findings) == 1 and "retarget_content_hashes" in findings[0].message


def test_an_absent_field_is_not_reported_here():
    """Absence belongs to the manifest_keys rule, which knows about waivers. Reporting it twice
    would turn one gap into two findings and make the waived case a FAIL again."""
    assert _check_source_attribution_shape({"source": {"source_name": "amass"}}, "t0", None) == []


def test_a_non_object_attribution_is_reported_rather_than_crashing():
    findings = _check_source_attribution_shape(
        _manifest("gaitex", "CC-BY-4.0"), "t0", REGISTRY.profile_for("gaitex")
    )
    assert len(findings) == 1 and "not an object" in findings[0].message


def test_an_empty_registry_key_list_is_refused_at_load(tmp_path: Path):
    """A registry that pins no keys would pass everything, which is worse than not checking."""
    source = REGISTRY.path.read_text(encoding="utf-8")
    broken = source.replace(
        "  source_attribution_keys:\n    - dataset_citation",
        "  source_attribution_keys: []\n  _unused:\n    - dataset_citation",
        1,
    )
    path = tmp_path / "registry.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(dataset_profiles.ProfileError, match="source_attribution_keys"):
        dataset_profiles.load(path)


# ------------------------------------------------------------------- the bundles on disk
def _bundle_root() -> Path | None:
    """``<SOMA_DATA_ROOT>/runs/experimental_generation_poc_demo``, or None when it is unset."""
    try:
        return paths.data_root() / "runs" / "experimental_generation_poc_demo"
    except paths.PathConfigError:
        return None


BUNDLE_ROOT = _bundle_root()
# Lineage names, no version suffix (docs/guides/RETENTION_RULES.md 1.1); a lineage that has
# been retired and not regenerated is skipped, not failed.
ON_DISK = {
    "gaitex": "gaitex_unified8",
    "addbiomechanics": "addbio_unified8",
    "hknu": "hknu_unified8",
    "amass": "amass_faithful_full",
    "prism": "prism_faithful_full",
}


@pytest.mark.parametrize("source,bundle", sorted(ON_DISK.items()))
def test_the_bundle_on_disk_matches_the_shape_the_registry_declares(source, bundle):
    """Reads one take, not the corpus: the shape is per-bundle, so the first take decides it."""
    if BUNDLE_ROOT is None:
        pytest.skip("SOMA_DATA_ROOT unset: the bundles on disk were not read")
    root = BUNDLE_ROOT / bundle
    if not root.is_dir():
        pytest.skip(f"{bundle} not present under SOMA_DATA_ROOT")
    take = next(
        (p for p in sorted(root.iterdir()) if (p / "manifest.json").is_file()), None
    )
    if take is None:
        pytest.skip(f"{bundle} has no takes")
    manifest = json.loads((take / "manifest.json").read_text(encoding="utf-8"))
    assert _check_source_attribution_shape(manifest, take.name, REGISTRY.profile_for(source)) == []
