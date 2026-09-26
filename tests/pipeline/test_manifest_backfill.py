"""The backfill's two promises: it derives nothing it cannot witness, and it can be undone.

The dangerous failure here is not a crash. It is a manifest that quietly gains a field saying
something untrue about the data -- "no resampling" on a take that was resampled, an up axis read
from one witness while the other says otherwise. Most of what follows tests refusal.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "backfill_manifest_fields.py"


def _load():
    spec = importlib.util.spec_from_file_location("backfill_manifest_fields", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves annotations through sys.modules, and a module
    # that is not there yet fails on the first frozen dataclass rather than on import.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bf = _load()


PRISM_MANIFEST = {
    "spec_id": "qmd_unified8_smpl18",
    "source": {"source_name": "prism", "native_rate_hz": 100, "native_frames_total": 13160},
    "global_frame": {"convention": "PRISM world, right-handed, Z-up, gravity along -Z"},
    "canonicalization_config": {
        "target_rate_hz": 100,
        "frame_count": 13160,
        "gravity_prism_world_m_s2": [0.0, 0.0, -9.80665],
    },
    "window": {"frame_start": 0, "frame_end_exclusive": 13160, "frame_count": 13160},
}


def _manifest(**overrides):
    manifest = json.loads(json.dumps(PRISM_MANIFEST))
    for dotted, value in overrides.items():
        head, _, tail = dotted.partition(".")
        if tail:
            manifest[head][tail] = value
        else:
            manifest[head] = value
    return manifest


# -------------------------------------------------------------------------- serialisation
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("trailing", [False, True])
def test_every_style_the_five_bundles_use_round_trips(newline, trailing):
    """PRISM writes CRLF with no trailing newline, AMASS LF with one. Reading the style from
    the file rather than assuming it is what keeps the edit to the keys it means to add."""
    style = bf.Style(newline=newline, trailing_newline=trailing)
    raw = style.dump(PRISM_MANIFEST)
    detected = bf.detect_style(raw, PRISM_MANIFEST)
    assert detected == style


def test_a_manifest_that_does_not_round_trip_is_refused_not_reformatted(tmp_path: Path):
    """Without the byte-exact proof there is no way to promise a write added only what it meant
    to, so the take is left alone."""
    take = tmp_path / "take"
    take.mkdir()
    # Four-space indent is not a style this tool emits, so no round-trip exists.
    (take / "manifest.json").write_text(
        json.dumps(PRISM_MANIFEST, indent=4), encoding="utf-8", newline=""
    )
    plan = bf.plan_bundle(tmp_path, bf.load_attribution())

    assert len(plan.takes) == 1
    assert "round-trip" in plan.takes[0].blocked
    assert plan.takes[0].additions == {}


# -------------------------------------------------------------------------------- up_axis
def test_up_axis_needs_both_witnesses_to_agree():
    assert bf.derive_up_axis(_manifest()).value == "z"


def test_up_axis_is_refused_when_the_witnesses_disagree():
    """The prose convention and the gravity vector are written by different parts of the
    generator. Trusting whichever was read first would hide a real defect."""
    result = bf.derive_up_axis(
        _manifest(**{"canonicalization_config.gravity_prism_world_m_s2": [0.0, -9.80665, 0.0]})
    )
    assert not result.ok
    assert "disagree" in result.skip_reason


def test_up_axis_is_refused_on_a_single_witness():
    result = bf.derive_up_axis(_manifest(global_frame={}))
    assert not result.ok
    assert "only one witness" in result.skip_reason


def test_gravity_pointing_up_is_not_read_as_an_up_axis():
    """A positive gravity component is a different frame convention, not a sign to flip."""
    assert bf._axis_from_gravity([0.0, 0.0, 9.80665]) is None


def test_a_tilted_gravity_vector_witnesses_nothing():
    assert bf._axis_from_gravity([0.1, 0.0, -9.8]) is None


# ------------------------------------------------------------------------------- resample
def test_resample_records_a_noop_when_all_four_numbers_agree():
    value = bf.derive_resample(_manifest()).value
    assert value["source_fps"] == 100.0 and value["target_fps"] == 100
    assert value["frames_in"] == value["frames_out"] == 13160
    assert value["noop"] is True and value["direction"] == "none"


def test_resample_is_refused_when_the_rates_differ():
    """Writing 'no resampling' onto a take that was resampled is the worst thing this can do."""
    result = bf.derive_resample(_manifest(**{"source.native_rate_hz": 120}))
    assert not result.ok
    assert "resampled" in result.skip_reason


def test_resample_is_refused_when_the_window_crops_the_source():
    """Equal rates with unequal counts means the take was cropped; frames_in is then not the
    source count, and inferring it would put a wrong ratio in the record."""
    result = bf.derive_resample(
        _manifest(**{"window.frame_count": 9000, "window.frame_end_exclusive": 9000})
    )
    assert not result.ok
    assert "disagree" in result.skip_reason


def test_resample_keeps_the_recorded_rate_rather_than_a_default():
    """The value comes from the manifest; the float is a representation change, not a new number."""
    value = bf.derive_resample(
        _manifest(**{"source.native_rate_hz": 60, "canonicalization_config.target_rate_hz": 60})
    ).value
    assert value["source_fps"] == 60.0


# ---------------------------------------------------------------------------- attribution
def test_the_shipped_registry_resolves_the_three_sources_that_lacked_an_attribution():
    """amass and prism record two interpretations confirmed on 2026-09-09: license_id names a
    bundled text or a form scope rather than an SPDX identifier, and publication is
    NOT_ESTABLISHED."""
    registry = bf.load_attribution()
    assert set(registry.resolved) == {"amass", "hknu", "prism"}
    assert registry.pending == {}


@pytest.mark.parametrize("source", ["amass", "prism"])
def test_the_confirmed_sources_still_record_what_was_open_about_them(source):
    """An approved interpretation is still an interpretation. Dropping the note would leave a
    reader unable to tell a confirmed judgement from a fact read off a DOI."""
    raw = bf.load_attribution().path.read_text(encoding="utf-8")
    assert "was_blocked_on" in raw
    value = bf.load_attribution().resolved[source]
    assert value["publication"] == "NOT_ESTABLISHED"
    assert "SPDX" not in value["license_id"] and "CC-BY" not in value["license_id"]


def test_a_source_whose_interpretation_is_pending_is_refused_not_guessed(tmp_path: Path):
    """The mechanism that keeps an unconfirmed source out. It must keep working for the next
    source still awaiting confirmation, or that source is written silently."""
    path = tmp_path / "attr.yaml"
    path.write_text(
        "schema: source_attribution_v1\n"
        "required_keys: [dataset_citation]\n"
        "sources:\n"
        "  newsource:\n"
        "    status: pending_user_confirmation\n"
        "    blocked_on: the licence record leaves the interpretation to the data owner\n",
        encoding="utf-8",
    )
    result = bf.load_attribution(path).value_for("newsource")
    assert not result.ok
    assert "awaits the data owner's confirmation" in result.skip_reason
    assert "leaves the interpretation to the data owner" in result.skip_reason


@pytest.mark.parametrize("source", ["amass", "hknu", "prism"])
def test_a_resolved_attribution_carries_every_required_key(source):
    registry = bf.load_attribution()
    for key in registry.required_keys:
        assert registry.resolved[source].get(key), key


def test_an_incomplete_resolved_attribution_is_rejected_at_load(tmp_path: Path):
    path = tmp_path / "attr.yaml"
    path.write_text(
        "schema: source_attribution_v1\n"
        "required_keys: [dataset_citation, license_id]\n"
        "sources:\n"
        "  hknu:\n"
        "    status: resolved\n"
        "    value:\n"
        "      dataset_citation: something\n",
        encoding="utf-8",
    )
    with pytest.raises(bf.BackfillError, match="license_id"):
        bf.load_attribution(path)


def test_an_unknown_source_is_not_given_a_default_attribution():
    result = bf.load_attribution().value_for("no_such_source")
    assert not result.ok


# ---------------------------------------------------------------------- apply and rollback
def _bundle(tmp_path: Path, manifest, *, style=None) -> Path:
    style = style or bf.Style(newline="\r\n", trailing_newline=False)
    take = tmp_path / "prism-subj001-take002"
    take.mkdir(parents=True)
    (take / "manifest.json").write_bytes(style.dump(manifest))
    return tmp_path


def test_applying_adds_only_the_planned_keys_and_leaves_the_rest_byte_identical(tmp_path: Path):
    bundle = _bundle(tmp_path, PRISM_MANIFEST)
    plan = bf.plan_bundle(bundle, bf.load_attribution())
    take = plan.takes[0]
    assert sorted(take.additions) == ["resample", "source_attribution", "up_axis"]

    entry = bf.apply_take(take)
    after = json.loads((bundle / take.take_id / "manifest.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in after.items() if k not in take.additions} == PRISM_MANIFEST
    assert entry["added_keys"] == ["resample", "source_attribution", "up_axis"]
    assert entry["sha256_before"] != entry["sha256_after"]


def test_applying_twice_is_a_noop_because_an_existing_key_is_never_overwritten(tmp_path: Path):
    bundle = _bundle(tmp_path, PRISM_MANIFEST)
    registry = bf.load_attribution()
    bf.apply_take(bf.plan_bundle(bundle, registry).takes[0])
    second = bf.plan_bundle(bundle, registry).takes[0]
    assert second.additions == {}


def test_rollback_restores_the_recorded_preimage_exactly(tmp_path: Path):
    bundle = _bundle(tmp_path, PRISM_MANIFEST)
    manifest_path = bundle / "prism-subj001-take002" / "manifest.json"
    original = manifest_path.read_bytes()

    plan = bf.plan_bundle(bundle, bf.load_attribution())
    journal = tmp_path / "journal.jsonl"
    journal.write_text(
        json.dumps(bf.apply_take(plan.takes[0])) + "\n", encoding="utf-8", newline="\n"
    )
    assert manifest_path.read_bytes() != original

    assert bf.rollback(journal, log=lambda _m: None) == 0
    assert manifest_path.read_bytes() == original


def test_rollback_refuses_a_file_that_changed_after_the_journal_was_written(tmp_path: Path):
    """The journal describes one specific pre-image. If the file on disk is not the one it
    wrote, restoring it would discard whoever edited it since."""
    bundle = _bundle(tmp_path, PRISM_MANIFEST)
    manifest_path = bundle / "prism-subj001-take002" / "manifest.json"
    plan = bf.plan_bundle(bundle, bf.load_attribution())
    journal = tmp_path / "journal.jsonl"
    journal.write_text(
        json.dumps(bf.apply_take(plan.takes[0])) + "\n", encoding="utf-8", newline="\n"
    )

    edited = json.loads(manifest_path.read_text(encoding="utf-8"))
    edited["someone_elses_edit"] = True
    manifest_path.write_bytes(bf.Style("\r\n", False).dump(edited))

    messages: list[str] = []
    assert bf.rollback(journal, log=messages.append) == 1
    assert any("CHANGED SINCE" in m for m in messages)
    assert "someone_elses_edit" in manifest_path.read_text(encoding="utf-8")


def test_a_superseded_journal_refuses_wholesale_and_says_why(tmp_path: Path):
    """PRISM was backfilled twice. Undoing the first journal after the second ran would drop the
    second pass's work, so every entry is refused -- and the reason has to be findable, or the
    operator reads 150 CHANGED SINCE lines and guesses."""
    bundle = _bundle(tmp_path, PRISM_MANIFEST)
    manifest_path = bundle / "prism-subj001-take002" / "manifest.json"
    registry = bf.load_attribution()

    first = tmp_path / "first.jsonl"
    plan = bf.plan_bundle(bundle, registry)
    # Write only up_axis in the first pass, so the second has something left to add.
    take = plan.takes[0]
    take.additions = {"up_axis": take.additions["up_axis"]}
    first.write_text(json.dumps(bf.apply_take(take)) + "\n", encoding="utf-8", newline="\n")

    second_take = bf.plan_bundle(bundle, registry).takes[0]
    bf.apply_take(second_take)

    messages: list[str] = []
    assert bf.rollback(first, log=messages.append) == 1
    assert any("superseded by a later pass" in m for m in messages)
    # The second pass's keys survive the refusal.
    assert "source_attribution" in json.loads(manifest_path.read_text(encoding="utf-8"))


def test_a_bundle_the_tool_has_no_deriver_for_is_reported_not_guessed(tmp_path: Path):
    manifest = _manifest(**{"source.source_name": "gaitex"})
    del manifest["global_frame"]
    bundle = _bundle(tmp_path, manifest)
    plan = bf.plan_bundle(bundle, bf.load_attribution())
    assert plan.takes[0].additions == {}
    assert "no deriver for gaitex.up_axis" in plan.takes[0].skips["up_axis"]
