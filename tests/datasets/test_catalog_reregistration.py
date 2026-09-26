"""Registering a dataset that is already in the catalog reconciles it instead of appending.

A bundle regenerated in place keeps its asset ids and paths and changes its bytes; an appending
registration added a new entry beside each earlier one, whose digest no longer matched the file,
and both stayed live. A registration now marks every earlier live entry of the dataset that is not
the asset as registered superseded, by the convention existing catalogs already follow, in one journalled write; one that would change nothing writes nothing.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from soma_synth import cli
from soma_synth.datasets import catalog

from test_validation_checks import make_dataset

POC = "runs/experimental_generation_poc_demo"


def _register(catdir: Path, dataset: Path, root: Path, **kwargs) -> catalog.Registration:
    return catalog.register_dataset(catdir, dataset, source="gaitex", spec_version="faithful-v2",
                                    data_root=root, **kwargs)


def _bundle(root: Path, name: str = "gaitex_unified8", n: int = 2) -> Path:
    return make_dataset(root / POC / name, source="gaitex", n=n)


def _regenerate(dataset: Path, takes=None) -> None:
    """New bytes under the same take folders and names, as a regeneration in place leaves them."""
    for take in sorted(p for p in dataset.iterdir() if p.is_dir()):
        if takes is not None and take.name not in takes:
            continue
        manifest = take / "manifest.json"
        manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with open(take / "anthro_reference.npz", "ab") as handle:
            handle.write(b"\0")


def _live(document: dict) -> list[dict]:
    return [a for a in document["assets"] if not a.get("superseded")]


def _live_matches_the_files(document: dict, root: Path) -> bool:
    return all(catalog.sha256_file(root / a["relative_path"]) == a["sha256"]
               for a in _live(document))


def test_registering_the_same_bundle_again_adds_nothing_and_writes_nothing(tmp_path):
    dataset = _bundle(tmp_path)
    catdir = tmp_path / "catalog"
    acp = catdir / "asset_catalog.json"
    first = _register(catdir, dataset, tmp_path)
    assert (first.added, first.unchanged, first.superseded) == (len(first), 0, 0)
    before = acp.read_bytes()

    again = _register(catdir, dataset, tmp_path)
    assert list(again) == list(first)                       # the assets, as it always returned
    assert (again.added, again.unchanged, again.superseded) == (0, len(first), 0)
    assert acp.read_bytes() == before                       # the file untouched
    assert len(catalog.read_journal(acp)) == 1              # and no record of a write that was not made
    assert len(catalog.read_journal(catdir / "gaitex" / "source_manifest.json")) == 1
    assert "0 added" in again.summary() and "0 earlier entries superseded" in again.summary()


def test_a_regenerated_bundle_supersedes_its_earlier_entries_in_place(tmp_path):
    dataset = _bundle(tmp_path)
    catdir = tmp_path / "catalog"
    acp = catdir / "asset_catalog.json"
    first = _register(catdir, dataset, tmp_path)
    after_first = acp.read_bytes()
    _regenerate(dataset, takes={"gaitex-take000"})

    again = _register(catdir, dataset, tmp_path,
                      registered_by={"tool": "soma-synth run", "run_id": "run_0123456789ab"})
    changed = [a for a in again if a not in first]
    assert {a["role"] for a in changed} == {"manifest", "anthro"}
    assert (again.added, again.superseded) == (len(changed), len(changed))
    assert again.unchanged == len(first) - len(changed)

    document = catalog.load_asset_catalog(catdir)
    assert len(document["assets"]) == len(first) + len(changed)
    old = {a["asset_id"]: a for a in first}
    for index, entry in enumerate(document["assets"][:len(first)]):
        replaced_by = next((a for a in changed if a["asset_id"] == entry["asset_id"]), None)
        if replaced_by is None:
            assert entry == old[entry["asset_id"]]           # untouched, at its index
            continue
        # the catalog convention for an entry a later registration replaced
        assert entry == {**old[entry["asset_id"]], "superseded": True,
                         "supersession_reason": "registration_replaced",
                         "superseded_by": replaced_by["asset_id"],
                         "superseded_by_sha256": replaced_by["sha256"]}
    assert document["assets"][len(first):] == changed
    ids = Counter(a["asset_id"] for a in _live(document))
    assert set(ids.values()) == {1} and _live_matches_the_files(document, tmp_path)

    # one journalled write: each replaced entry with its old value, each new one added
    record = catalog.read_journal(acp)[-1]
    assert record["sequence"] == 2
    assert record["counts"] == {"added": len(changed), "replaced": len(changed), "removed": 0}
    assert record["registered_by"] == {"tool": "soma-synth run", "run_id": "run_0123456789ab"}
    assert {k: record["operation"][k] for k in ("added", "unchanged", "superseded")} == {
        "added": len(changed), "unchanged": len(first) - len(changed), "superseded": len(changed)}
    replaced = [c for c in record["changes"] if c["change"] == "replaced"]
    assert [c["old"] for c in replaced] == [old[c["old"]["asset_id"]] for c in replaced]
    assert all(c["new"]["superseded"] is True for c in replaced)
    # and the earlier catalog is there to be rebuilt
    rebuilt, exact = catalog.catalog_as_of(acp, 1)
    assert exact and catalog.serialise(rebuilt) == after_first


def test_an_append_only_catalog_with_stale_live_entries_is_reconciled_in_one_write(tmp_path):
    """A catalog an append-only re-registration produced: the earlier entries of gaitex_unified8
    (a reconciliation's ``registration_evidence`` beside them, digests of the files that were
    replaced) and the new ones appended after them, both live; the older generation's
    folder already superseded. Registering the bundle again supersedes the stale entries, adds
    nothing, and leaves every other entry and field as it was."""
    dataset = _bundle(tmp_path, n=3)
    old_generation = [dict(a) for a in catalog.register_dataset(
        tmp_path / "scratch", dataset, source="gaitex", spec_version="faithful-v2",
        data_root=tmp_path)]
    _regenerate(dataset)
    new_generation = [dict(a) for a in catalog.register_dataset(
        tmp_path / "scratch2", dataset, source="gaitex", spec_version="faithful-v2",
        data_root=tmp_path)]
    folder_v1 = [{**a, "relative_path": a["relative_path"].replace("gaitex_unified8/",
                                                                   "gaitex_unified8_v1/"),
                  "superseded": True, "supersession_reason": "metadata_reconciliation",
                  "superseded_by": a["asset_id"], "superseded_by_sha256": a["sha256"],
                  "supersession_evidence": "latest_appended_registration_metadata_only"}
                 for a in old_generation]
    evidence = {"method": "latest_appended_registration_metadata_only",
                "payload_integrity": "not_rehashed",
                "index_relative_path": f"{POC}/gaitex_unified8/INDEX.json"}
    stale = [{**a, "registration_evidence": evidence} for a in old_generation]
    other = {"asset_id": "qmd_unified8_smpl18:faithful-v2:hknu:S01_T01:small", "role": "small",
             "relative_path": f"{POC}/hknu_unified8/S01_T01/small_reference.npz",
             "sha256": "0" * 64, "bytes": 1, "source": "hknu", "spec_id": "qmd_unified8_smpl18",
             "spec_version": "faithful-v2"}
    legacy = {"schema": catalog.SCHEMA_ASSET, "artifact_class": "experimental_non_candidate",
             "quality_gate": "NOT_EVALUATED", "reconciliations": [{"superseded_count": 3}],
             "assets": [*folder_v1, other, *stale, *new_generation]}
    catdir = tmp_path / "catalog"
    acp = catdir / "asset_catalog.json"
    acp.parent.mkdir(parents=True)
    acp.write_bytes(catalog.serialise(legacy))              # written before the journal existed
    live = Counter(a["asset_id"] for a in _live(legacy))
    assert sum(1 for count in live.values() if count == 2) == len(new_generation)

    result = _register(catdir, dataset, tmp_path)
    assert (result.added, result.unchanged, result.superseded) == (0, len(new_generation),
                                                                   len(stale))
    [record] = catalog.read_journal(acp)                   # one journalled write
    assert record["counts"] == {"added": 0, "replaced": len(stale), "removed": 0}
    assert record["before"]["sha256"] == hashlib.sha256(catalog.serialise(legacy)).hexdigest()
    first_stale = len(folder_v1) + 1
    assert [c["key"] for c in record["changes"]] == [
        f"/assets/{index}" for index in range(first_stale, first_stale + len(stale))]

    document = catalog.load_asset_catalog(catdir)
    assert {k: v for k, v in document.items() if k != "assets"} == {
        k: v for k, v in legacy.items() if k != "assets"}
    assert document["assets"][:first_stale] == legacy["assets"][:first_stale]
    assert document["assets"][first_stale + len(stale):] == new_generation
    for entry, before, current in zip(document["assets"][first_stale:first_stale + len(stale)],
                                      stale, new_generation):
        assert entry == {**before, "superseded": True,
                         "supersession_reason": "registration_replaced",
                         "superseded_by": current["asset_id"],
                         "superseded_by_sha256": current["sha256"]}
    assert set(Counter(a["asset_id"] for a in _live(document)).values()) == {1}
    assert all(catalog.sha256_file(tmp_path / a["relative_path"]) == a["sha256"]
               for a in _live(document) if a["source"] == "gaitex")

    settled = acp.read_bytes()
    _register(catdir, dataset, tmp_path)                   # and again: nothing left to do
    assert acp.read_bytes() == settled and len(catalog.read_journal(acp)) == 1


def test_other_folders_and_other_lineages_are_left_alone(tmp_path):
    """Asset ids do not name a folder: a scratch copy of the same takes is another dataset, and so
    is the same folder registered under another spec version (supersede_spec_version's job)."""
    production = _bundle(tmp_path)
    scratch = make_dataset(tmp_path / "tmp" / "gaitex_unified8", source="gaitex", n=2)
    _regenerate(scratch)
    catdir = tmp_path / "catalog"
    first = _register(catdir, production, tmp_path)
    second = _register(catdir, scratch, tmp_path)
    assert second.superseded == 0 and second.added == len(second)
    other = catalog.register_dataset(catdir, production, source="gaitex",
                                     spec_version="faithful-v3", data_root=tmp_path)
    assert other.superseded == 0 and other.added == len(other)
    document = catalog.load_asset_catalog(catdir)
    assert not any(a.get("superseded") for a in document["assets"])
    assert len(document["assets"]) == len(first) + len(second) + len(other)


def test_an_entry_at_the_same_path_under_another_take_id_is_superseded(tmp_path):
    """The same file registered under a new take id (a renamed INDEX entry): its earlier entry is
    matched by its relative path within the lineage."""
    dataset = _bundle(tmp_path, n=1)
    catdir = tmp_path / "catalog"
    first = _register(catdir, dataset, tmp_path)
    index = json.loads((dataset / "INDEX.json").read_text(encoding="utf-8"))
    index["takes"][0]["take_id"] = "gaitex-renamed"
    (dataset / "INDEX.json").write_text(json.dumps(index), encoding="utf-8")
    again = _register(catdir, dataset, tmp_path)
    assert (again.added, again.unchanged, again.superseded) == (len(first), 0, len(first))
    document = catalog.load_asset_catalog(catdir)
    assert [a["superseded_by"] for a in document["assets"][:len(first)]] == [
        a["asset_id"] for a in again]
    assert all(":gaitex-renamed:" in a["asset_id"] for a in _live(document))


def test_duplicate_live_entries_of_one_asset_are_reduced_to_the_newest(tmp_path):
    """The same registration appended twice by the old code: the newest identical entry stays, the
    older one is superseded by it."""
    dataset = _bundle(tmp_path, n=1)
    catdir = tmp_path / "catalog"
    first = _register(catdir, dataset, tmp_path)
    acp = catdir / "asset_catalog.json"
    document = catalog.load_asset_catalog(catdir)
    document["assets"] = [*document["assets"], *[dict(a) for a in first]]
    with catalog.catalog_lock(catdir):
        catalog.write_catalog_document(acp, document)
    again = _register(catdir, dataset, tmp_path)
    assert (again.added, again.unchanged, again.superseded) == (0, len(first), len(first))
    document = catalog.load_asset_catalog(catdir)
    assert all(a["superseded"] and a["superseded_by_sha256"] == a["sha256"]
               for a in document["assets"][:len(first)])
    assert document["assets"][len(first):] == list(first)


def test_the_cli_says_what_a_registration_did(tmp_path, capsys):
    dataset = _bundle(tmp_path)
    argv = ["register", str(dataset), "--catalog", str(tmp_path / "catalog"), "--source", "gaitex",
            "--data-root", str(tmp_path)]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "added" in out and "0 earlier entries superseded" in out
    _regenerate(dataset, takes={"gaitex-take001"})
    assert cli.main(argv) == 0
    assert "2 earlier entries superseded" in capsys.readouterr().out


def _small(take: str, sha: str, folder: str = f"{POC}/gaitex_unified8") -> dict:
    return {"asset_id": f"qmd_unified8_smpl18:faithful-v2:gaitex:{take}:small", "role": "small",
            "relative_path": f"{folder}/{take}/small_reference.npz", "sha256": sha, "bytes": 1,
            "source": "gaitex", "spec_id": "qmd_unified8_smpl18", "spec_version": "faithful-v2"}


def test_merge_registration_drops_the_entries_no_registered_asset_stands_for():
    """Registering [t1 changed] over [t1, t2] left t2 live: its take was no longer in the bundle,
    and after an in-place replacement its file was gone. It is marked superseded now, reason
    registration_dropped, superseded_by null; only entries under the folder and of the lineage."""
    folder = f"{POC}/gaitex_unified8"
    t1, t2, new_t1 = _small("t1", "a" * 64), _small("t2", "b" * 64), _small("t1", "c" * 64)
    elsewhere = _small("t2", "b" * 64, folder=f"{POC}/gaitex_unified8_sample")
    other_version = {**t2, "asset_id": t2["asset_id"].replace("faithful-v2", "faithful-v3"),
                     "spec_version": "faithful-v3"}
    merged, added, unchanged, superseded, dropped = catalog.merge_registration(
        [t1, t2, elsewhere, other_version], [new_t1], folder)
    assert (added, unchanged, superseded, dropped) == (1, 0, 1, 1)
    assert merged == [
        {**t1, "superseded": True, "supersession_reason": "registration_replaced",
         "superseded_by": new_t1["asset_id"], "superseded_by_sha256": new_t1["sha256"]},
        {**t2, "superseded": True, "supersession_reason": "registration_dropped",
         "superseded_by": None},
        elsewhere, other_version, new_t1]
    lineage = ("gaitex", "qmd_unified8_smpl18", "faithful-v2")
    # a registration with no assets drops the whole folder of its lineage when it names the lineage
    assert catalog.merge_registration([t1, t2], [], folder, lineage=lineage)[4] == 2
    # and nothing without one to go by, or when the folder is the data root itself
    assert catalog.merge_registration([t1, t2], [], folder)[4] == 0
    for root in ("", "."):
        assert catalog.merge_registration([t1, t2], [], root, lineage=lineage)[4] == 0


def test_a_take_or_file_the_bundle_no_longer_has_is_dropped_in_the_same_write(tmp_path):
    """A take no longer ok in the regenerated INDEX, and a role file of another take that is gone:
    their live entries are marked registration_dropped in the registration's one journalled write.
    The same takes in a scratch folder, and the folder under another spec version, stay live."""
    dataset = _bundle(tmp_path, n=3)
    scratch = make_dataset(tmp_path / "tmp" / "gaitex_unified8", source="gaitex", n=3)
    catdir = tmp_path / "catalog"
    acp = catdir / "asset_catalog.json"
    first = _register(catdir, dataset, tmp_path)
    kept_elsewhere = [dict(a) for a in _register(catdir, scratch, tmp_path)]
    kept_elsewhere += [dict(a) for a in catalog.register_dataset(
        catdir, dataset, source="gaitex", spec_version="faithful-v3", data_root=tmp_path)]
    index = json.loads((dataset / "INDEX.json").read_text(encoding="utf-8"))
    index["takes"][0]["status"] = "failed"
    (dataset / "INDEX.json").write_text(json.dumps(index), encoding="utf-8")
    (dataset / "gaitex-take001" / "anthro_reference.npz").unlink()
    gone = [a for a in first if ":gaitex-take000:" in a["asset_id"]
            or a["asset_id"].endswith(":gaitex-take001:anthro")]
    assert len(gone) > 1

    again = _register(catdir, dataset, tmp_path,
                      registered_by={"tool": "soma-synth run", "run_id": "run_0123456789ab"})
    assert (again.added, again.unchanged, again.superseded, again.dropped) == (
        0, len(first) - len(gone), 0, len(gone))
    assert f"{len(gone)} dropped (no longer in the bundle)" in again.summary()
    document = catalog.load_asset_catalog(catdir)
    assert len(document["assets"]) == len(first) + len(kept_elsewhere)
    gone_ids = {a["asset_id"] for a in gone}
    for entry, before in zip(document["assets"], first):
        if before["asset_id"] in gone_ids:
            assert entry == {**before, "superseded": True,
                             "supersession_reason": "registration_dropped", "superseded_by": None}
        else:
            assert entry == before
    assert document["assets"][len(first):] == kept_elsewhere
    live_here = [a for a in _live(document) if a["spec_version"] == "faithful-v2"
                 and a["relative_path"].startswith(f"{POC}/gaitex_unified8/")]
    assert sorted(a["asset_id"] for a in live_here) == sorted(a["asset_id"] for a in again)

    record = catalog.read_journal(acp)[-1]
    assert record["counts"] == {"added": 0, "replaced": len(gone), "removed": 0}
    assert record["operation"]["dropped"] == len(gone)
    assert record["registered_by"] == {"tool": "soma-synth run", "run_id": "run_0123456789ab"}
    settled = acp.read_bytes()
    assert _register(catdir, dataset, tmp_path).dropped == 0       # and again: nothing to do
    assert acp.read_bytes() == settled


def test_register_takes_its_data_root_from_soma_data_root(tmp_path, monkeypatch, capsys):
    """`register` without --data-root made the bundle's parent, the lineage container, the data
    root: its paths ('gaitex_unified8/...') matched none of the entries `run` had registered
    ('runs/experimental_generation_poc_demo/gaitex_unified8/...'), so registering the bundle again
    appended every asset a second time and superseded nothing. It takes SOMA_DATA_ROOT now, as
    `pipeline` and `run` do, so the reconciliation command needs no --data-root on a PC that has
    the variable set."""
    dataset = _bundle(tmp_path)
    catdir = tmp_path / "experimental" / "catalog"
    first = _register(catdir, dataset, tmp_path)           # as `run` registered it
    _regenerate(dataset, takes={"gaitex-take000"})
    monkeypatch.setenv("SOMA_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("SOMA_SOURCE_ROOT", raising=False)
    argv = ["register", str(dataset), "--catalog", str(catdir), "--source", "gaitex"]
    assert cli.main(argv) == 0
    assert "2 earlier entries superseded" in capsys.readouterr().out
    document = catalog.load_asset_catalog(catdir)
    assert all(a["relative_path"].startswith(f"{POC}/gaitex_unified8/")
               for a in document["assets"])
    assert len(document["assets"]) == len(first) + 2
    assert set(Counter(a["asset_id"] for a in _live(document)).values()) == {1}
    assert _live_matches_the_files(document, tmp_path)


def test_register_refuses_without_a_data_root_that_holds_the_bundle(tmp_path, monkeypatch,
                                                                     capsys):
    """No --data-root and no SOMA_DATA_ROOT, or a data root that does not hold the bundle: exit 2
    and no catalog, from `register` and from `pipeline`'s register step. The library call takes
    SOMA_DATA_ROOT too, and raises without it."""
    from soma_synth.pipeline.paths import PathConfigError

    root = tmp_path / "data"
    dataset = _bundle(root)
    catdir = root / "experimental" / "catalog"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.delenv("SOMA_DATA_ROOT", raising=False)
    monkeypatch.delenv("SOMA_SOURCE_ROOT", raising=False)
    argv = ["register", str(dataset), "--catalog", str(catdir), "--source", "gaitex"]
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("register: ") and "SOMA_DATA_ROOT" in err and "--data-root" in err
    for data_root in (elsewhere, dataset):                 # not above the bundle; the bundle itself
        assert cli.main([*argv, "--data-root", str(data_root)]) == 2
        assert "is not inside the data root" in capsys.readouterr().err
        assert cli.main(["pipeline", str(dataset), "--source", "gaitex", "--start-at", "register",
                         "--data-root", str(data_root), "--catalog", str(catdir)]) == 2
        assert "is not inside the data root" in capsys.readouterr().err
    monkeypatch.setenv("SOMA_DATA_ROOT", str(elsewhere))
    assert cli.main(argv) == 2
    assert "is not inside the data root" in capsys.readouterr().err
    assert not catdir.exists()

    monkeypatch.delenv("SOMA_DATA_ROOT")
    with pytest.raises(PathConfigError):
        catalog.register_dataset(catdir, dataset, source="gaitex", spec_version="faithful-v2")
    assert not catdir.exists()
    monkeypatch.setenv("SOMA_DATA_ROOT", str(root))
    assets = catalog.register_dataset(catdir, dataset, source="gaitex",
                                      spec_version="faithful-v2")
    assert assets and all(a["relative_path"].startswith(f"{POC}/gaitex_unified8/")
                          for a in assets)
