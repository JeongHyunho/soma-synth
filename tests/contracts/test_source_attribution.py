"""The attribution registry as a generator reads it: resolved sources only, the shared keys always."""

from __future__ import annotations

from pathlib import Path

import pytest

from soma_synth.contracts import dataset_profiles
from soma_synth.contracts import source_attribution as attribution

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_repository_registry_loads_and_every_resolved_source_carries_the_shared_keys():
    config = attribution.load()
    assert config.required_keys == ("dataset_citation", "publication", "license_id",
                                    "stable_source_id", "changes_made")
    assert "hknu" in config.resolved
    for source, value in config.resolved.items():
        for key in config.required_keys:
            assert value[key], f"{source}: {key} is empty"


def test_hknu_resolves_to_the_block_the_profile_registry_allows():
    """What the generator writes must be what the validator lets through."""
    value = attribution.resolved_for("hknu")
    allowed = dataset_profiles.load().allowed_source_attribution_keys("hknu")
    assert set(value) <= set(allowed), sorted(set(value) - set(allowed))
    assert value["license_id"] == "CC-BY-4.0"
    assert "hknu_smpl24_paired" in value["changes_made"]
    assert "_v1" not in value["changes_made"]


def test_a_pending_source_cannot_be_written(tmp_path: Path):
    registry = tmp_path / "attr.yaml"
    registry.write_text(
        "schema: source_attribution_v1\n"
        "required_keys: [dataset_citation, publication, license_id, stable_source_id, changes_made]\n"
        "sources:\n"
        "  x:\n"
        "    status: pending_user_confirmation\n"
        "    blocked_on: the licence text is unread\n",
        encoding="utf-8",
    )
    with pytest.raises(attribution.AttributionError, match="awaits the data owner's confirmation"):
        attribution.resolved_for("x", registry)


def test_an_unknown_source_and_a_resolved_block_missing_a_shared_key_are_refused(tmp_path: Path):
    registry = tmp_path / "attr.yaml"
    registry.write_text(
        "schema: source_attribution_v1\n"
        "required_keys: [dataset_citation, publication]\n"
        "sources:\n"
        "  y:\n"
        "    status: resolved\n"
        "    value:\n"
        "      dataset_citation: a\n",
        encoding="utf-8",
    )
    with pytest.raises(attribution.AttributionError, match="missing publication"):
        attribution.load(registry)
    registry.write_text(
        "schema: source_attribution_v1\nrequired_keys: [dataset_citation]\nsources:\n"
        "  y:\n    status: resolved\n    value:\n      dataset_citation: a\n",
        encoding="utf-8",
    )
    with pytest.raises(attribution.AttributionError, match="no entry"):
        attribution.resolved_for("z", registry)


def test_the_backfill_tool_reads_through_the_same_module():
    """One reader for one file: the backfill and the generators cannot disagree about it."""
    import importlib.util
    import sys

    path = REPO_ROOT / "scripts" / "backfill_manifest_fields.py"
    spec = importlib.util.spec_from_file_location("backfill_manifest_fields_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["backfill_manifest_fields_under_test"] = module
    spec.loader.exec_module(module)
    registry = module.load_attribution()
    assert registry.resolved["hknu"] == attribution.resolved_for("hknu")
    assert registry.required_keys == attribution.load().required_keys
