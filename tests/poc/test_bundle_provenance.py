"""A published bundle has to name the corpus it actually came from, and count what it actually holds.

Both of these shipped wrong once. The generator hardcoded `hknu_smpl24_paired_v1` into every
manifest while there was one paired corpus; when a second one existed, the v2 bundle's manifests
still pointed at v1 -- and the difference between those two corpora is how the body was fitted, so
the pointer sent anyone tracing a channel to a different body. Separately, the generator writes the
root description before the discontinuity scan has excluded anything, so the description said
`287 ok, 0 excluded` beside an INDEX that said `280 ok, 7 excluded`.

Neither is caught by L0-L4: the validator checks the takes the catalog calls ok, not whether the
prose beside them is true. So they are checked here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scan():
    return _load(REPO / "scripts" / "diagnostics" / "pose_discontinuity_scan.py",
                 "pose_discontinuity_scan")


@pytest.fixture(scope="module")
def generator():
    sys.path.insert(0, str(REPO / "src"))
    return _load(REPO / "scripts" / "poc" / "generate_hknu_unified8.py",
                 "generate_hknu_unified8")


DESCRIPTION = """# HKNU FullBody - synthetic IMU reference

**INTERNAL-ONLY.** Not canonical output.

- `spec_id` / `spec_version`: `qmd_unified8_smpl18` / `faithful-v2`
- source: `hknu`
- takes: 287 ok, 0 failed, 0 excluded of 287
- generated: 2000-01-01T00:00:00+00:00

## Per-take deliverables
"""


def test_the_description_is_made_to_agree_with_the_index_it_ships_beside(scan, tmp_path):
    (tmp_path / "DATA_DESCRIPTION_EN.md").write_text(DESCRIPTION, encoding="utf-8")

    scan.retell_counts(tmp_path, {"total": 287, "ok": 280, "failed": 0, "excluded": 7})

    text = (tmp_path / "DATA_DESCRIPTION_EN.md").read_text(encoding="utf-8")
    assert "- takes: 280 ok, 0 failed, 7 excluded of 287" in text
    assert "287 ok" not in text
    # nothing else moved
    assert "**INTERNAL-ONLY.** Not canonical output." in text
    assert text.endswith("## Per-take deliverables\n")


def test_both_descriptions_are_updated_when_a_bundle_carries_the_short_companion(scan, tmp_path):
    (tmp_path / "DATA_DESCRIPTION_EN.md").write_text(DESCRIPTION, encoding="utf-8")
    (tmp_path / "DATASET_DESCRIPTION_EN.md").write_text(DESCRIPTION, encoding="utf-8")

    scan.retell_counts(tmp_path, {"total": 287, "ok": 280, "failed": 0, "excluded": 7})

    for name in ("DATA_DESCRIPTION_EN.md", "DATASET_DESCRIPTION_EN.md"):
        assert "280 ok, 0 failed, 7 excluded" in (tmp_path / name).read_text(encoding="utf-8")


def test_a_description_someone_rewrote_by_hand_is_left_alone(scan, tmp_path, capsys):
    hand_written = "# HKNU\n\n- takes: 280 ok and 7 excluded, said in my own words\n"
    (tmp_path / "DATA_DESCRIPTION_EN.md").write_text(hand_written, encoding="utf-8")

    scan.retell_counts(tmp_path, {"total": 287, "ok": 280, "failed": 0, "excluded": 7})

    assert (tmp_path / "DATA_DESCRIPTION_EN.md").read_text(encoding="utf-8") == hand_written
    assert "left untouched" in capsys.readouterr().out


def test_the_manifest_names_the_corpus_the_bundle_was_published_from(generator, tmp_path):
    """The bug this replaces: v2 takes carrying `hknu_smpl24_paired_v1` in every manifest."""
    corpus = tmp_path / "runs" / "experimental_generation_poc_demo" / "hknu_smpl24_paired_v2"
    take = corpus / "S01" / "S01_Walking_4KPH_0.npz"
    take.parent.mkdir(parents=True)
    take.write_bytes(b"not a real npz; only its digest is read here")

    identity = generator.identity_for(
        take, "S01", "Walking_4KPH_0", 1000, 100.0,
        {"height_m": 1.75, "mass_kg": 70.0}, root=tmp_path, paired=corpus)

    counterpart = identity.extra_manifest["paired_measured_counterpart"]
    assert counterpart["dataset"] == "hknu_smpl24_paired_v2"
    assert counterpart["relative_path"] == (
        "runs/experimental_generation_poc_demo/hknu_smpl24_paired_v2")
    assert "hknu_smpl24_paired_v1" not in json.dumps(counterpart)
    # the prose that names it travels with the same value
    assert "hknu_smpl24_paired_v2" in identity.source_note
    assert "hknu_smpl24_paired_v2" in identity.site_provenance["back_T4"]
    assert any("hknu_smpl24_paired_v2" in line for line in identity.unavailable)


def test_the_corpus_name_is_read_from_the_corpus_and_not_from_a_default(generator, tmp_path):
    corpus = tmp_path / "runs" / "experimental_generation_poc_demo" / "hknu_smpl24_paired_v9_trial"
    take = corpus / "S02" / "S02_Running_8KPH.npz"
    take.parent.mkdir(parents=True)
    take.write_bytes(b"digest only")

    identity = generator.identity_for(
        take, "S02", "Running_8KPH", 500, 100.0,
        {"height_m": 1.68, "mass_kg": 62.0}, root=tmp_path, paired=corpus)

    assert identity.extra_manifest["paired_measured_counterpart"]["dataset"] == (
        "hknu_smpl24_paired_v9_trial")
