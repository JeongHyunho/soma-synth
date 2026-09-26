"""The harness verdict: hash_outputs.py + compare_hashes.py on synthetic scratch runs.

The gate separates the declared differences (a corpus digest, the pair_id that follows it, the
scratch-location prefix) from everything else. These build two tiny harness runs that differ
in one controlled way and check the category each difference lands in, end to end through both
scripts. No generator runs and nothing outside tmp_path is read or written.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "harness"
POC = Path("runs") / "experimental_generation_poc_demo"
REL = "gaitex_austra_gwo"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"harness_{name}", HARNESS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hash_outputs = _load("hash_outputs")
compare_hashes = _load("compare_hashes")
run_harness = _load("run_harness")


# ------------------------------------------------------------------------ fixtures
CONTENT_HASHES = "adapter=b133211b389262e4;config=none;code={code};model={model}"


def _write_run(data_root: Path, run: str, *, corpus: str = "gaitex_smpl24",
               retarget_sha: str = "a" * 64, pair_id: str = "p" * 64, frames: int = 100,
               asset_id: str | None = None, spec_id: str = "qmd_unified8_smpl18",
               settings_hash: str = "s1", content_hashes: str | None = None) -> Path:
    """One gaitex_unified8 take under <data_root>/tmp/harness/<run>, named as a generator names it.

    `content_hashes` adds the manifest copy of the corpus value (as the AddBio bundle carries it)."""
    scratch = data_root / "tmp" / "harness" / run
    bundle = scratch / POC / "gaitex_unified8"
    take = bundle / REL
    take.mkdir(parents=True)
    rel_path = f"tmp/harness/{run}/{POC.as_posix()}/{corpus}/austra/gwo.npz"
    manifest = {
        "generated_utc": f"2026-09-25T00:00:0{len(run) % 10}+00:00",
        "identity": {"pair_id": pair_id},
        "source": {"relative_path": rel_path, "source_asset_sha256": retarget_sha},
        "small_synthesis": {"inputs": {
            "retarget_npz": {"relative_path": rel_path, "sha256": retarget_sha, "frames": frames},
            "note": "the marker and imu files are the small's own inputs"}},
        "anthro_reconstruction": {"reduced_model_fit": {"settings": {"hash": settings_hash}}},
    }
    if content_hashes is not None:
        manifest["source_attribution"] = {"retarget_content_hashes": content_hashes}
    (take / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8",
                                        newline="\n")
    np.savez(take / "anthro_reference.npz",
             source_asset_id=np.array(asset_id if asset_id is not None else rel_path),
             pair_id=np.array(pair_id), betas=np.arange(4, dtype=np.float32))
    index = {"spec_id": spec_id, "source": "gaitex", "generated_utc": "2026-09-25T00:00:00+00:00",
             "counts": {"total": 1, "ok": 1},
             "takes": [{"rel": REL, "take_id": REL, "subject_id": "gaitex_austra",
                        "pair_id": pair_id, "relative_path": rel_path, "status": "ok"}]}
    (bundle / "INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8", newline="\n")
    fit = {"schema": "reduced_model_fit", "source": "gaitex", "group_count": 1, "take_count": 1,
           "settings": {"hash": settings_hash},
           "groups": {"gaitex_austra": {"settings": {"hash": settings_hash},
                                         "constants_wxyz": [[1.0, 0.0, 0.0, 0.0]],
                                         "residual_fitted_m": 0.005}},
           "takes": {REL: "gaitex_austra"}, "unfitted": {}}
    (bundle / "reduced_model_fit.json").write_text(json.dumps(fit), encoding="utf-8", newline="\n")
    return scratch


def _write_corpus(scratch: Path, content_hashes: str, *, extra: tuple[str, ...] = ()) -> Path:
    """The gaitex_smpl24 corpus the take above names: one trial npz and its run record."""
    corpus = scratch / POC / "gaitex_smpl24"
    (corpus / "austra").mkdir(parents=True)
    np.savez(corpus / "austra" / "gwo.npz", content_hashes=np.array(content_hashes),
             pose=np.zeros((3, 4), np.float32))
    (corpus / "_run.json").write_text(json.dumps({"subjects": 1}), encoding="utf-8", newline="\n")
    for name in extra:
        (corpus / name).write_text(json.dumps({"extra": name}), encoding="utf-8", newline="\n")
    return corpus


def _run_with_corpus(data_root: Path, run: str, *, code: str = "c1", model: str = "m1",
                     extra: tuple[str, ...] = ()) -> Path:
    value = CONTENT_HASHES.format(code=code, model=model)
    scratch = _write_run(data_root, run, content_hashes=value)
    _write_corpus(scratch, value, extra=extra)
    return scratch


def _hash(data_root: Path, scratch: Path, out: Path) -> Path:
    assert hash_outputs.main(["--scratch", str(scratch), "--data-root", str(data_root),
                              "--out", str(out)]) == 0
    return out


def _compare(a: Path, b: Path, out: Path, *normalize: str) -> tuple[int, list[dict]]:
    argv = [str(a), str(b), "--out", str(out)]
    for spec in normalize:
        argv += ["--normalize-prefix", spec]
    code = compare_hashes.main(argv)
    return code, json.loads(out.read_text(encoding="utf-8"))["items"]


def _found(items: list[dict]) -> set[tuple[str, str]]:
    return {(item["category"], item["what"]) for item in items}


def _as_format1(path: Path) -> Path:
    """The same hashes without the format-2 additions, as an older reference file carries them."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc.pop("format", None)
    for bundle in doc["bundles"].values():
        bundle.pop("index_head", None)
        bundle.pop("index_order", None)
        (bundle.get("fit") or {}).pop("head", None)
        (bundle.get("fit") or {}).pop("complete", None)
        for take in bundle["takes"].values():
            for record in take["files"].values():
                record.pop("deep_subtrees", None)
                record.pop("norm_deep_subtrees", None)
    old = path.with_name(path.stem + "_format1.json")
    old.write_text(json.dumps(doc), encoding="utf-8")
    return old


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    return root


@pytest.fixture
def reference(data_root, tmp_path):
    """The reference run every case is compared against."""
    return _hash(data_root, _write_run(data_root, "h1"), tmp_path / "h1.json")


# ------------------------------------------------------------------------ hash_outputs
def test_the_depth_four_layer_is_kept_apart_from_the_depth_three_one():
    doc = {"a": {"b": {"c": {"d": 1, "e": 2}}}, "f": 3}
    shallow = hash_outputs.subtrees(doc)
    deep = hash_outputs.subtrees(doc, hash_outputs.DEEP_LEVEL, start=hash_outputs.DEEP_LEVEL)
    assert set(shallow) == {"a", "a.b", "a.b.c", "f"}
    assert set(deep) == {"a.b.c.d", "a.b.c.e"}


def test_a_run_records_the_index_head_order_and_the_split_composite(data_root, reference):
    doc = json.loads(reference.read_text(encoding="utf-8"))
    bundle = doc["bundles"]["gaitex_unified8"]
    assert doc["format"] == 2
    assert bundle["index_order"] == [REL]
    assert bundle["index_head"]["generated_utc"] == "<masked>"
    assert "takes" not in bundle["index_head"]
    manifest = bundle["takes"][REL]["files"]["manifest.json"]
    assert "small_synthesis.inputs.retarget_npz.relative_path" in manifest["deep_subtrees"]
    assert "small_synthesis.inputs.retarget_npz.relative_path" in manifest["norm_deep_subtrees"]
    assert bundle["fit"]["complete"] is True and bundle["fit"]["head"]["schema"] == "reduced_model_fit"


# ------------------------------------------------------------------------ the prefix
def test_two_runs_that_differ_only_by_location_are_clean(data_root, tmp_path, reference):
    h2 = _hash(data_root, _write_run(data_root, "h2_r1"), tmp_path / "h2.json")
    code, items = _compare(reference, h2, tmp_path / "d.json", "auto")
    assert (code, items) == (0, [])


def test_without_normalization_the_location_is_declared_c5_and_nothing_else(data_root, tmp_path,
                                                                            reference):
    h2 = _hash(data_root, _write_run(data_root, "h2_r1"), tmp_path / "h2.json")
    code, items = _compare(reference, h2, tmp_path / "d.json")
    assert code == 0
    assert {item["category"] for item in items} == {"C5"}
    assert ("C5", "json small_synthesis.inputs.retarget_npz.relative_path") in _found(items)
    assert ("C5", "relative_path") in _found(items)            # the INDEX entry
    assert ("C5", "member source_asset_id") in _found(items)


@pytest.mark.parametrize("normalize", [("auto",), ()])
def test_a_relative_path_changed_beyond_the_prefix_is_undeclared(data_root, tmp_path, reference,
                                                                 normalize):
    """Both sides normalized, and the path still differs (a folder renamed). That
    is a regression whatever the field is called, never C5."""
    h2 = _hash(data_root, _write_run(data_root, "h2_r1", corpus="GAITEX_SMPL24"),
               tmp_path / "h2.json")
    code, items = _compare(reference, h2, tmp_path / "d.json", *normalize)
    assert code == 1
    found = _found(items)
    assert ("XX", "json source.relative_path") in found
    assert ("XX", "json small_synthesis.inputs.retarget_npz.relative_path") in found
    assert ("XX", "relative_path") in found                    # the INDEX entry
    assert ("XX", "member source_asset_id") in found
    assert not any(cat == "C5" for cat, _ in found)


def test_an_index_entry_that_merely_contains_the_scratch_folder_is_not_c5(data_root, tmp_path,
                                                                          reference):
    """Both values contain tmp/harness/, which used to be enough for C5."""
    scratch = _write_run(data_root, "h2_r1")
    index_path = scratch / POC / "gaitex_unified8" / "INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["takes"][0]["relative_path"] = index["takes"][0]["relative_path"].replace("gwo", "ng")
    index_path.write_text(json.dumps(index), encoding="utf-8")
    h2 = _hash(data_root, scratch, tmp_path / "h2.json")
    code, items = _compare(reference, h2, tmp_path / "d.json")
    assert code == 1
    assert ("XX", "relative_path") in _found(items)


# ------------------------------------------------------------------------ the corpus digest
def test_a_corpus_digest_is_c2_and_split_from_the_path_beside_it(data_root, tmp_path, reference):
    h2 = _hash(data_root, _write_run(data_root, "h2_r1", retarget_sha="b" * 64, pair_id="q" * 64),
               tmp_path / "h2.json")
    code, items = _compare(reference, h2, tmp_path / "d.json", "auto")
    assert code == 0, items
    found = _found(items)
    assert ("C2", "json small_synthesis.inputs.retarget_npz.sha256") in found
    assert ("C2", "json source.source_asset_sha256") in found
    assert ("C3", "json identity.pair_id") in found
    assert ("C3", "member pair_id") in found
    assert ("C3", "pair_id") in found                           # the INDEX entry
    assert {cat for cat, _ in found} == {"C2", "C3"}


def test_a_path_change_hidden_behind_a_corpus_digest_change_is_found(data_root, tmp_path,
                                                                     reference):
    h2 = _hash(data_root, _write_run(data_root, "h2_r1", retarget_sha="b" * 64, frames=99),
               tmp_path / "h2.json")
    code, items = _compare(reference, h2, tmp_path / "d.json", "auto")
    assert code == 1
    assert ("XX", "json small_synthesis.inputs.retarget_npz.frames") in _found(items)
    assert ("C2", "json small_synthesis.inputs.retarget_npz.sha256") in _found(items)


def test_against_a_format1_file_the_composite_is_unlocalized_not_c2(data_root, tmp_path,
                                                                    reference):
    """A format-1 reference file has no depth-4 layer: its retarget_npz digest covers the rewritten
    relative_path and the C2 sha256 at once, so a difference there cannot be called declared."""
    h2 = _hash(data_root, _write_run(data_root, "h2_r1", retarget_sha="b" * 64),
               tmp_path / "h2.json")
    code, items = _compare(_as_format1(reference), h2, tmp_path / "d.json", "auto")
    assert code == 1
    assert ("UL", "json small_synthesis.inputs.retarget_npz") in _found(items)
    assert ("C2", "json source.source_asset_sha256") in _found(items)


def test_a_format1_composite_that_differs_only_by_the_prefix_is_c5(data_root, tmp_path, reference):
    h2 = _hash(data_root, _write_run(data_root, "h2_r1"), tmp_path / "h2.json")
    code, items = _compare(_as_format1(reference), h2, tmp_path / "d.json")
    assert code == 0
    assert ("C5", "json small_synthesis.inputs.retarget_npz") in _found(items)


# ------------------------------------------------------------------------ INDEX.json
def test_an_index_that_differs_only_in_its_entries_is_located(data_root, tmp_path, reference):
    """Format 2 records the head and the order, so the file-level difference is never UL."""
    h2 = _hash(data_root, _write_run(data_root, "h2_r1", pair_id="q" * 64), tmp_path / "h2.json")
    code, items = _compare(reference, h2, tmp_path / "d.json", "auto")
    assert code == 0
    assert not any(item["where"] == "<root>/INDEX.json" for item in items)


def test_a_changed_index_head_is_undeclared(data_root, tmp_path, reference):
    h2 = _hash(data_root, _write_run(data_root, "h2_r1", spec_id="other_spec"), tmp_path / "h2.json")
    code, items = _compare(reference, h2, tmp_path / "d.json", "auto")
    assert code == 1
    assert ("XX", "head spec_id") in _found(items)


def test_a_format1_index_is_verified_by_rebuilding_it(data_root, tmp_path, reference):
    """Against a format-1 file, the format-2 head and order plus the format-1 side's own entries
    must reproduce its masked hash; then only the entries differ, and those are compared."""
    h2 = _hash(data_root, _write_run(data_root, "h2_r1", pair_id="q" * 64), tmp_path / "h2.json")
    code, items = _compare(_as_format1(reference), h2, tmp_path / "d.json", "auto")
    assert code == 0, items                     # C3 only: the pair_ids, in the entries and files
    assert not any(item["where"] == "<root>/INDEX.json" for item in items), items
    summary = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))["summary"]
    assert summary["gaitex_unified8"]["index_head_and_order_verified_by_rebuild"] == 1


def test_a_format1_index_with_another_head_stays_unlocalized(data_root, tmp_path, reference):
    h2 = _hash(data_root, _write_run(data_root, "h2_r1", spec_id="other_spec", pair_id="q" * 64),
               tmp_path / "h2.json")
    code, items = _compare(_as_format1(reference), h2, tmp_path / "d.json", "auto")
    assert code == 1
    assert ("UL", "json") in {(i["category"], i["what"]) for i in items
                              if i["where"] == "<root>/INDEX.json"}


# ------------------------------------------------------------------------ content_hashes tokens
def test_content_hashes_that_differ_in_code_only_are_c2(data_root, tmp_path):
    h1 = _hash(data_root, _run_with_corpus(data_root, "h1", code="c1"), tmp_path / "h1.json")
    h2 = _hash(data_root, _run_with_corpus(data_root, "h2_r1", code="c2"), tmp_path / "h2.json")
    manifest = json.loads(h2.read_text(encoding="utf-8"))["bundles"]["gaitex_unified8"]["takes"][
        REL]["files"]["manifest.json"]
    assert manifest["values"] == {"source_attribution.retarget_content_hashes":
                                  CONTENT_HASHES.format(code="c2", model="m1")}
    code, items = _compare(h1, h2, tmp_path / "d.json", "auto")
    assert code == 0, items
    assert _found(items) == {("C2", "member content_hashes"),
                             ("C2", "json source_attribution.retarget_content_hashes")}


@pytest.mark.parametrize("change", [{"model": "m2"}, {"code": "c2", "model": "m2"}])
def test_content_hashes_with_another_token_changed_are_undeclared(data_root, tmp_path, change):
    """A different body model (or source file, or config) is not the declared code= change,
    whatever the field is called."""
    h1 = _hash(data_root, _run_with_corpus(data_root, "h1"), tmp_path / "h1.json")
    h2 = _hash(data_root, _run_with_corpus(data_root, "h2_r1", **change), tmp_path / "h2.json")
    code, items = _compare(h1, h2, tmp_path / "d.json", "auto")
    assert code == 1
    assert _found(items) == {("XX", "member content_hashes"),
                             ("XX", "json source_attribution.retarget_content_hashes")}
    assert any("model: A=m1 B=m2" in item["detail"] for item in items)


def _drop_manifest_values(path: Path) -> Path:
    """The hash file as hash_outputs.py wrote it before it recorded `values` (older reference)."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    for take in doc["bundles"]["gaitex_unified8"]["takes"].values():
        take["files"]["manifest.json"].pop("values", None)
    old = path.with_name(path.stem + "_novalues.json")
    old.write_text(json.dumps(doc), encoding="utf-8")
    return old


@pytest.mark.parametrize("change, category", [({"code": "c2"}, "C2"), ({"model": "m2"}, "UL")])
def test_a_manifest_without_values_follows_its_own_corpus_file(data_root, tmp_path, change,
                                                                category):
    """Without the recorded value the manifest copy is C2 only when the take's corpus file (its
    source_asset_id) showed a code-only difference; a corpus file that differed otherwise leaves it
    unlocalized (the corpus member itself is XX)."""
    h1 = _hash(data_root, _run_with_corpus(data_root, "h1"), tmp_path / "h1.json")
    h2 = _hash(data_root, _run_with_corpus(data_root, "h2_r1", **change), tmp_path / "h2.json")
    _code, items = _compare(_drop_manifest_values(h1), h2, tmp_path / "d.json", "auto")
    assert (category, "json source_attribution.retarget_content_hashes") in _found(items)


def test_a_manifest_without_values_or_corpus_evidence_is_unlocalized(data_root, tmp_path):
    h1 = _hash(data_root, _run_with_corpus(data_root, "h1"), tmp_path / "h1.json")
    h2 = _hash(data_root, _run_with_corpus(data_root, "h2_r1", code="c2"), tmp_path / "h2.json")
    doc = json.loads(_drop_manifest_values(h1).read_text(encoding="utf-8"))
    doc["corpora"] = {}
    bare = tmp_path / "h1_bare.json"
    bare.write_text(json.dumps(doc), encoding="utf-8")
    code, items = _compare(bare, h2, tmp_path / "d.json", "auto")
    assert code == 1
    assert ("UL", "json source_attribution.retarget_content_hashes") in _found(items)


def test_a_corpus_value_nobody_recorded_is_unlocalized(data_root, tmp_path):
    """An ARRAY_SHA256 reference records no value, so which token moved cannot be said."""
    h1 = _hash(data_root, _run_with_corpus(data_root, "h1"), tmp_path / "h1.json")
    h2 = _hash(data_root, _run_with_corpus(data_root, "h2_r1", code="c2"), tmp_path / "h2.json")
    files = json.loads(h1.read_text(encoding="utf-8"))["corpora"]["gaitex_smpl24"]["files"]
    reference = {"files": {rel: {name: {k: v for k, v in member.items() if k != "value"}
                                 for name, member in rec["members"].items()}
                           for rel, rec in files.items() if "members" in rec}}
    ref_path = tmp_path / "ARRAY_SHA256.json"
    ref_path.write_text(json.dumps(reference), encoding="utf-8")
    code = compare_hashes.main([str(h1), str(h2), "--normalize-prefix", "auto",
                                "--a-corpus", f"gaitex_smpl24={ref_path}",
                                "--out", str(tmp_path / "d.json")])
    items = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))["items"]
    assert code == 1
    assert ("UL", "member content_hashes") in _found(items)
    assert not any(item["what"] == "file" for item in items)   # _run.json: counted, not judged


# ------------------------------------------------------------------------ corpus file sets
def test_a_corpus_file_only_one_harness_run_wrote_is_undeclared(data_root, tmp_path):
    h1 = _hash(data_root, _run_with_corpus(data_root, "h1"), tmp_path / "h1.json")
    h2 = _hash(data_root, _run_with_corpus(data_root, "h2_r1", extra=("SUMMARY.json",)),
               tmp_path / "h2.json")
    code, items = _compare(h1, h2, tmp_path / "d.json", "auto")
    assert code == 1
    assert [(i["category"], i["where"], i["detail"]) for i in items] == [
        ("XX", "SUMMARY.json", "only in B")]


# ------------------------------------------------------------------------ reduced_model_fit.json
def test_the_fit_file_is_compared_through_its_summary(data_root, tmp_path, reference):
    """Only the smpl18 settings hash moved: C1 in the summary and the manifests, no UL."""
    h2 = _hash(data_root, _write_run(data_root, "h2_r1", settings_hash="s2"), tmp_path / "h2.json")
    code, items = _compare(reference, h2, tmp_path / "d.json", "auto")
    assert code == 0, items
    assert {item["category"] for item in items} == {"C1"}
    assert ("C1", "settings") in _found(items)


# ------------------------------------------------------------------------ run_harness
def _driver(tmp_path: Path, record_dir: Path | None):
    import argparse

    data_root = tmp_path / "data"
    (data_root / "tmp" / "harness").mkdir(parents=True)
    args = argparse.Namespace(code_root=str(REPO), scratch=str(data_root / "tmp" / "harness" / "r"),
                              data_root=str(data_root), python=sys.executable, sources="gaitex",
                              record_dir=str(record_dir) if record_dir else None)
    return run_harness.Driver(args), data_root


def test_the_driver_removes_every_location_variable_from_the_children(tmp_path, monkeypatch):
    for name in run_harness.UNPINNED_VARS:
        monkeypatch.setenv(name, str(tmp_path / name))
    driver, data_root = _driver(tmp_path, None)
    env = driver.env()
    assert not set(run_harness.UNPINNED_VARS) & set(env)
    assert env["SOMA_DATA_ROOT"] == str(data_root.resolve())
    assert {"SOMA_SOURCE_ROOT", "SOMA_BODY_MODEL_DIR", "SOMA_POC_OUT_DIR"} <= set(run_harness.UNPINNED_VARS)


def test_a_record_dir_in_the_production_tree_is_refused(tmp_path):
    driver, data_root = _driver(tmp_path, tmp_path / "data" / POC / "logs")
    with pytest.raises(run_harness.HarnessError, match="record-dir"):
        driver.check_layout()
    assert not (data_root / POC).exists()


@pytest.mark.parametrize("where", ["inside_scratch", "outside_data_root"])
def test_a_record_dir_in_the_scratch_or_off_the_data_root_is_accepted(tmp_path, where):
    record = (tmp_path / "data" / "tmp" / "harness" / "r" / "_harness"
              if where == "inside_scratch" else tmp_path / "records")
    driver, _data_root = _driver(tmp_path, record)
    driver.check_layout()


@pytest.mark.parametrize("scratch", [
    Path("tmp") / "harness",                                  # the harness folder itself
    Path("tmp") / "elsewhere" / "r",                          # beside it
    Path(POC) / "gaitex_unified8",                            # a production bundle
])
def test_a_scratch_directory_outside_a_harness_run_folder_is_refused(tmp_path, scratch):
    """The authorization records allow the sample runs under <data root>/tmp/harness/<run> only."""
    driver, data_root = _driver(tmp_path, None)
    driver.scratch = (data_root / scratch).resolve()
    with pytest.raises(run_harness.HarnessError, match="not a run directory"):
        driver.check_layout()


def test_a_scratch_directory_off_the_data_root_is_refused(tmp_path):
    driver, _data_root = _driver(tmp_path, None)
    driver.scratch = (tmp_path / "scratch" / "r").resolve()
    with pytest.raises(run_harness.HarnessError, match="not a run directory"):
        driver.check_layout()


def test_the_driver_finds_the_package_of_its_code_root(tmp_path):
    """A code root holds the package as soma_synth or as soma_synthetic_imu. The layout check and
    the preflight import whichever it is."""
    driver, _data_root = _driver(tmp_path, None)
    assert driver.package == "soma_synth"
    old = tmp_path / "old_checkout"
    (old / "src" / "soma_synthetic_imu").mkdir(parents=True)
    import argparse

    args = argparse.Namespace(code_root=str(old), scratch=str(tmp_path / "data" / "tmp" / "harness" / "r"),
                              data_root=str(tmp_path / "data"), python=sys.executable,
                              sources="gaitex", record_dir=None)
    assert run_harness.Driver(args).package == "soma_synthetic_imu"


def test_the_authorization_records_are_read_from_the_decisions_dir_given(tmp_path):
    """soma-synth carries no decision records: the parent project's research/decisions is handed
    in, read in place, and a missing record is refused with a pointer to --decisions-dir."""
    decisions = tmp_path / "parent" / "research" / "decisions"
    decisions.mkdir(parents=True)
    with pytest.raises(run_harness.HarnessError, match="--decisions-dir"):
        run_harness.authorization_for("gaitex", decisions)
    record = decisions / run_harness.AUTH_TEMPLATE.format(source="gaitex")
    record.write_text("source_name: gaitex\n<SOMA_DATA_ROOT>/tmp/harness\n"
                      "releases_holds: false\nexperimental_non_candidate\n", encoding="utf-8")
    assert run_harness.authorization_for("gaitex", decisions) == record
    # either separator names the scratch folder
    record.write_text("source_name: gaitex\n<SOMA_DATA_ROOT>\\tmp\\harness\n"
                      "releases_holds: false\nexperimental_non_candidate\n", encoding="utf-8")
    assert run_harness.authorization_for("gaitex", decisions) == record
    record.write_text("source_name: gaitex\n<SOMA_DATA_ROOT>/tmp/elsewhere\n"
                      "releases_holds: false\nexperimental_non_candidate\n", encoding="utf-8")
    with pytest.raises(run_harness.HarnessError, match="tmp/harness"):
        run_harness.authorization_for("gaitex", decisions)
    record.write_text("source_name: gaitex\n<SOMA_DATA_ROOT>/tmp/harness\n"
                      "releases_holds: false\nexperimental_non_candidate\n", encoding="utf-8")
    import argparse

    args = argparse.Namespace(code_root=str(REPO), scratch=str(tmp_path / "data" / "tmp" / "harness" / "r"),
                              data_root=str(tmp_path / "data"), python=sys.executable,
                              sources="gaitex", record_dir=None, decisions_dir=str(decisions))
    driver = run_harness.Driver(args)
    assert driver.decisions == decisions.resolve()
    assert driver.display(record) == "research/decisions/" + record.name
    assert not (REPO / "research").exists()


def _production(data_root: Path) -> Path:
    """A data root with one production bundle: an INDEX, one sample take and one other take."""
    poc = data_root / POC
    bundle = poc / "gaitex_unified8"
    for rel in (REL, "gaitex_other_gwo"):
        (bundle / rel).mkdir(parents=True)
        (bundle / rel / "manifest.json").write_text("{}", encoding="utf-8")
    index = {"takes": [{"rel": REL, "subject_id": "gaitex_austra", "take_id": REL},
                       {"rel": "gaitex_other_gwo", "subject_id": "gaitex_other",
                        "take_id": "gaitex_other_gwo"}]}
    (bundle / "INDEX.json").write_text(json.dumps(index), encoding="utf-8")
    (poc / "_superseded" / "gaitex_smpl24_2026-09-15").mkdir(parents=True)
    (poc / "_superseded" / "gaitex_smpl24_2026-09-15" / "SUMMARY.json").write_text(
        "{}", encoding="utf-8")
    (data_root / "extracted").mkdir(parents=True, exist_ok=True)
    return bundle


def _touch(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))


@pytest.mark.parametrize("changed, section", [
    ("gaitex_unified8/INDEX.json", "production_bundle_roots"),
    (f"gaitex_unified8/{REL}/manifest.json", "production_sample_takes"),
    ("_superseded/gaitex_smpl24_2026-09-15/SUMMARY.json", "evidence_folders"),
])
def test_the_snapshots_report_a_touched_production_file(tmp_path, changed, section):
    """Before this check covered only the bundle roots, a manifest rewritten inside a production
    take changed nothing it looked at."""
    data_root = tmp_path / "data"
    _production(data_root)
    before = run_harness.snapshot(data_root, [])
    assert f"gaitex_unified8/{REL}/manifest.json" in before["production_sample_takes"]
    assert "gaitex_unified8/gaitex_other_gwo/manifest.json" not in before["production_sample_takes"]
    assert run_harness.diff_snapshots(before, run_harness.snapshot(data_root, [])) == []
    _touch(data_root / POC / changed)
    problems = run_harness.diff_snapshots(before, run_harness.snapshot(data_root, []))
    assert problems and all(line.startswith(section + ":") for line in problems), problems


def test_the_default_decisions_folder_is_the_parent_projects_when_mounted(tmp_path):
    """Mounted at <parent>/packages/soma-synth beside the parent's evaluator and decision records,
    the driver reads the parent's research/decisions; a standalone checkout has no such default."""
    parent = tmp_path / "parent"
    mounted = parent / "packages" / "soma-synth"
    mounted.mkdir(parents=True)
    (parent / "configs" / "evaluators").mkdir(parents=True)
    (parent / "research" / "decisions").mkdir(parents=True)
    assert run_harness.default_decisions(mounted) == parent / "research" / "decisions"

    standalone = tmp_path / "soma-synth"
    standalone.mkdir()
    assert run_harness.default_decisions(standalone) == standalone / "research" / "decisions"

    (parent / "configs" / "evaluators").rmdir()
    assert run_harness.default_decisions(mounted) == mounted / "research" / "decisions"
