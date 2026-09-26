"""How the catalog files are written: atomically, under the catalog's writer lock, journalled.

An ``asset_catalog.json`` can reach hundreds of MB and is rewritten by every registration. A
registration that died half-way through the old in-place write left a truncated catalog, two
concurrent ones lost one of their entries, and every earlier version was gone for good. A full copy
per registration would keep them at that size each time; the journal keeps only what each write changed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from soma_synth.datasets import catalog

from test_validation_checks import make_dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _journal(path: Path) -> list[dict]:
    return catalog.read_journal(path)


def _history_files(folder: Path) -> list[Path]:
    history = folder / catalog.HISTORY_DIR
    return sorted(p for p in history.rglob("*") if p.is_file()) if history.is_dir() else []


def _register(catdir, dataset, root, **kwargs):
    return catalog.register_dataset(catdir, dataset, source="amass", spec_version="faithful-v2",
                                    data_root=root, **kwargs)


@pytest.fixture
def two_bundles(tmp_path):
    return (make_dataset(tmp_path / "amass_x", n=1), make_dataset(tmp_path / "amass_y", n=1))


def test_a_registration_journals_what_it_changed_never_a_copy(tmp_path, two_bundles):
    first, second = two_bundles
    catdir = tmp_path / "catalog"
    acp = catdir / "asset_catalog.json"
    added = _register(catdir, first, tmp_path)
    [created] = _journal(acp)
    assert created["sequence"] == 1 and created["before"] is None
    assert created["after"] == {"sha256": _sha(acp), "bytes": acp.stat().st_size}
    assert [c["new"] for c in created["changes"] if c["key"].startswith("/assets/")] == added
    before = _sha(acp)

    more = _register(catdir, second, tmp_path, registered_by={"tool": "soma-synth run",
                                                              "run_id": "run_0123456789ab"})
    record = _journal(acp)[-1]
    assert record["schema"] == catalog.SCHEMA_JOURNAL and record["catalog_file"] == "asset_catalog.json"
    assert record["sequence"] == 2
    assert datetime.fromisoformat(record["utc"]).tzinfo == UTC      # "...Z": UTC
    assert record["registered_by"] == {"tool": "soma-synth run", "run_id": "run_0123456789ab"}
    assert record["operation"]["name"] == "register_dataset"
    assert record["operation"]["dataset"] == "amass_y"            # a logical id, not a path
    assert record["before"]["sha256"] == before and record["after"]["sha256"] == _sha(acp)
    # exactly the entries this registration added, at the indices they took, and nothing else
    assert [c["change"] for c in record["changes"]] == ["added"] * len(more)
    assert [c["key"] for c in record["changes"]] == [
        f"/assets/{index}" for index in range(len(added), len(added) + len(more))]
    assert [c["old"] for c in record["changes"]] == [None] * len(more)
    assert [c["new"] for c in record["changes"]] == more
    assert record["counts"] == {"added": len(more), "replaced": 0, "removed": 0}
    # no earlier entry is lost: the file holds both registrations
    paths = [a["relative_path"] for a in catalog.load_asset_catalog(catdir)["assets"]]
    assert any(p.startswith("amass_x/") for p in paths) and any(p.startswith("amass_y/") for p in paths)
    # the only history is the two journals: no copy of either catalog file
    names = {p.relative_to(catdir).as_posix().rsplit("/", 1)[0] for p in _history_files(catdir)}
    assert names == {"_history/asset_catalog.journal"}
    # the second bundle's source manifest reads as the first's (one take each): a write that would
    # change nothing is not made, so its journal has the first registration's record only
    assert len(_journal(catdir / "amass" / "source_manifest.json")) == 1
    story = catalog.journal_story(acp)
    assert story["consistent"] and [r["applied"] for r in story["records"]] == [True, True]


def test_the_bytes_written_are_the_ones_always_written(tmp_path):
    target = tmp_path / "catalog" / "asset_catalog.json"
    payload = {"schema": catalog.SCHEMA_ASSET, "note": "é", "assets": []}
    catalog.write_catalog_document(target, payload)
    assert target.read_bytes() == (json.dumps(payload, indent=2, ensure_ascii=False)
                                   + "\n").encode("utf-8")


def _big_catalog(path: Path, count: int) -> None:
    """A catalog written before the journal existed, so it has none."""
    assets = [{"asset_id": f"qmd:faithful-v2:amass:take{index:06d}:{role}", "role": role,
               "relative_path": f"runs/experimental_generation_poc_demo/amass_faithful_full/"
                                f"take{index:06d}/{role}_reference.npz",
               "sha256": hashlib.sha256(f"{index}{role}".encode()).hexdigest(), "bytes": 123456,
               "source": "amass", "spec_id": "qmd_unified8_smpl18", "spec_version": "faithful-v1"}
              for index in range(count) for role in ("small",)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(catalog.serialise({"schema": catalog.SCHEMA_ASSET, "assets": assets}))


def test_journal_growth_is_proportional_to_the_changed_entries(tmp_path):
    """One small registration adds one small record, whether the catalog holds 50 assets or 20,000;
    so does flagging one asset superseded."""
    one = make_dataset(tmp_path / "amass_one", n=1)
    sizes = {}
    for count in (50, 20_000):
        catdir = tmp_path / f"catalog_{count}"
        acp = catdir / "asset_catalog.json"
        _big_catalog(acp, count)
        _register(catdir, one, tmp_path)
        [record_file] = catalog.journal_records(acp)
        sizes[count] = record_file.stat().st_size
        record = catalog.read_journal(acp)[0]
        assert record["counts"]["added"] == len(record["changes"]) and \
            record["counts"]["replaced"] == record["counts"]["removed"] == 0
    assert (tmp_path / "catalog_20000" / "asset_catalog.json").stat().st_size > 5_000_000
    assert sizes[20_000] < 4096
    assert abs(sizes[20_000] - sizes[50]) < 64          # the index digits, the digests

    # flag a single asset: one replaced entry, old and new
    acp = tmp_path / "catalog_20000" / "asset_catalog.json"
    document = json.loads(acp.read_text(encoding="utf-8"))
    document["assets"] = list(document["assets"])
    document["assets"][123] = {**document["assets"][123], "superseded": True}
    with catalog.catalog_lock(acp.parent):
        catalog.write_catalog_document(acp, document)
    record_file = catalog.journal_records(acp)[-1]
    assert record_file.stat().st_size < 2048
    [change] = catalog.read_journal(acp)[-1]["changes"]
    assert change["key"] == "/assets/123" and change["change"] == "replaced"
    assert "superseded" not in change["old"] and change["new"]["superseded"] is True


def test_an_earlier_entry_is_recoverable_after_it_is_replaced(tmp_path, two_bundles):
    first, second = two_bundles
    catdir = tmp_path / "catalog"
    acp = catdir / "asset_catalog.json"
    originals = _register(catdir, first, tmp_path)
    after_first = acp.read_bytes()

    flagged = catalog.supersede_spec_version(catdir, "faithful-v2", "faithful-v3")
    assert flagged == len(originals)
    assert all(a.get("superseded") for a in catalog.load_asset_catalog(catdir)["assets"])
    record = _journal(acp)[-1]
    assert record["operation"] == {"name": "supersede_spec_version", "spec_version": "faithful-v2",
                                   "superseded_by": "faithful-v3", "assets": len(originals)}
    # the journal holds each entry as it was before
    assert [c["old"] for c in record["changes"]] == originals
    assert [c["key"] for c in record["changes"]] == [f"/assets/{i}" for i in range(len(originals))]

    _register(catdir, second, tmp_path)                   # and later writes do not bury it
    document, exact = catalog.catalog_as_of(acp, 1)
    assert exact and catalog.serialise(document) == after_first
    assert document["assets"] == originals
    document, exact = catalog.catalog_as_of(acp, 2)
    assert exact and all(a.get("superseded") for a in document["assets"])
    assert catalog.catalog_as_of(acp, 0) == (None, True)  # no file before the first write
    current, exact = catalog.catalog_as_of(acp, 3)
    assert exact and current == catalog.load_asset_catalog(catdir)


class _Crash(BaseException):
    """The process dying: nothing after it runs (a BaseException, so no handler takes it)."""


def test_a_crash_between_the_journal_and_the_replace_tells_a_consistent_story(tmp_path, two_bundles,
                                                                               monkeypatch):
    """The record goes in before the replacement. A write that dies in between leaves the catalog as
    it was and a record whose "after" the file never had; the next write chains to the file as it
    is, and the story marks the dead one unapplied."""
    first, second = two_bundles
    catdir = tmp_path / "catalog"
    acp = catdir / "asset_catalog.json"
    _register(catdir, first, tmp_path)
    before = acp.read_bytes()
    replace = catalog._replace

    def die_on_the_asset_catalog(source, target):
        if Path(target).name == "asset_catalog.json":
            raise _Crash
        replace(source, target)

    monkeypatch.setattr(catalog, "_replace", die_on_the_asset_catalog)
    with pytest.raises(_Crash):
        _register(catdir, second, tmp_path)
    monkeypatch.undo()
    assert acp.read_bytes() == before                     # the catalog is as it was
    first_record, dead = _journal(acp)
    assert dead["before"]["sha256"] == first_record["after"]["sha256"] == _sha(acp)
    assert dead["after"]["sha256"] != _sha(acp)
    story = catalog.journal_story(acp)
    assert story["consistent"] and [r["applied"] for r in story["records"]] == [True, False]
    with pytest.raises(catalog.JournalError, match="never applied"):
        catalog.catalog_as_of(acp, 2)

    # the next registration goes on from the file as it is
    third = make_dataset(tmp_path / "amass_z", n=1)
    _register(catdir, third, tmp_path)
    *_, last = _journal(acp)
    assert last["sequence"] == 3 and last["before"]["sha256"] == dead["before"]["sha256"]
    assert last["changes"][0]["key"] == dead["changes"][0]["key"]   # the same index, reused
    story = catalog.journal_story(acp)
    assert story["consistent"] and [r["applied"] for r in story["records"]] == [True, False, True]
    document, exact = catalog.catalog_as_of(acp, 1)
    assert exact and catalog.serialise(document) == before
    paths = [a["relative_path"] for a in catalog.load_asset_catalog(catdir)["assets"]]
    assert not any(p.startswith("amass_y/") for p in paths)       # never registered: run it again
    assert list(catdir.rglob("*.tmp")) == []


def test_a_change_outside_the_journal_is_reported(tmp_path, two_bundles):
    first, second = two_bundles
    catdir = tmp_path / "catalog"
    acp = catdir / "asset_catalog.json"
    _register(catdir, first, tmp_path)
    document = json.loads(acp.read_text(encoding="utf-8"))
    document["note"] = "edited by hand"
    acp.write_bytes(catalog.serialise(document))          # another writer, no journal
    story = catalog.journal_story(acp)
    assert not story["consistent"] and story["records"][0]["applied"] is None
    _register(catdir, second, tmp_path)
    assert _journal(acp)[-1]["before"]["sha256"] == hashlib.sha256(
        catalog.serialise(document)).hexdigest()
    with pytest.raises(catalog.JournalError, match="cannot be explained"):
        catalog.catalog_as_of(acp, 1)


def test_a_failed_replace_leaves_the_catalog_as_it_was(tmp_path, two_bundles, monkeypatch):
    first, _ = two_bundles
    second = make_dataset(tmp_path / "amass_w", n=2)     # its source manifest differs too
    catdir = tmp_path / "catalog"
    _register(catdir, first, tmp_path)
    acp = catdir / "asset_catalog.json"
    before = acp.read_bytes()

    def refuse(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(catalog.os, "replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        _register(catdir, second, tmp_path)
    monkeypatch.undo()
    assert acp.read_bytes() == before
    assert [p.name for p in catdir.rglob("*.tmp")] == []
    # the record it wrote first stays, recognisably unapplied
    assert [r["applied"] for r in catalog.journal_story(acp)["records"]] == [True]
    manifest = catdir / "amass" / "source_manifest.json"
    assert [r["applied"] for r in catalog.journal_story(manifest)["records"]] == [True, False]


def test_a_failure_while_serialising_writes_nothing(tmp_path):
    target = tmp_path / "catalog" / "asset_catalog.json"
    catalog.write_catalog_document(target, {"assets": [1]})
    before = target.read_bytes()
    with pytest.raises(TypeError):
        catalog.write_catalog_document(target, {"assets": [object()]})
    assert target.read_bytes() == before
    assert len(catalog.journal_records(target)) == 1
    assert list(target.parent.rglob("*.tmp")) == []


def test_journal_records_are_never_overwritten(tmp_path, monkeypatch):
    """A record takes the next sequence and a name nothing had; a name found taken moves it on,
    and no record file is ever rewritten."""
    target = tmp_path / "c" / "asset_catalog.json"
    catalog.write_catalog_document(target, {"assets": []})
    [first] = catalog.journal_records(target)
    assert catalog._RECORD_NAME.match(first.name) and first.name.startswith("00000001.")
    kept = first.read_bytes()

    fixed = datetime(2026, 9, 26, 1, 2, 3, 456789, tzinfo=UTC)

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(catalog, "datetime", _Clock)
    squatter = catalog.journal_dir(target) / "00000002.20260926T010203.456789Z.json.gz"
    link = catalog._link_new

    def racing(temporary, name):
        if not squatter.exists():               # another writer takes the name meanwhile
            squatter.write_bytes(gzip.compress(b'{"sequence": 2}'))
        link(temporary, name)

    monkeypatch.setattr(catalog, "_link_new", racing)
    catalog.write_catalog_document(target, {"assets": [1]})
    names = [p.name for p in catalog.journal_records(target)]
    assert names == [first.name, squatter.name, "00000003.20260926T010203.456789Z.json.gz"]
    assert first.read_bytes() == kept and squatter.read_bytes() == gzip.compress(b'{"sequence": 2}')
    assert list(target.parent.rglob("*.tmp")) == []


def test_a_lock_file_another_writer_left_is_used_and_kept(tmp_path, two_bundles):
    """A catalog folder can carry a 0-byte .catalog.lock another writer made: it is locked and
    unlocked, never truncated, rewritten or deleted."""
    first, _ = two_bundles
    catdir = tmp_path / "catalog"
    catdir.mkdir()
    lock = catdir / catalog.LOCK_NAME
    lock.write_bytes(b"")
    stamp = lock.stat().st_mtime_ns
    _register(catdir, first, tmp_path)
    assert lock.is_file() and lock.read_bytes() == b"" and lock.stat().st_mtime_ns == stamp


def test_a_lock_this_writer_created_is_not_deleted_either(tmp_path, two_bundles):
    first, _ = two_bundles
    catdir = tmp_path / "catalog"
    _register(catdir, first, tmp_path)
    assert (catdir / catalog.LOCK_NAME).is_file()


def test_a_held_lock_refuses_another_writer_and_changes_nothing(tmp_path, two_bundles):
    first, second = two_bundles
    catdir = tmp_path / "catalog"
    _register(catdir, first, tmp_path)
    acp = catdir / "asset_catalog.json"
    before = acp.read_bytes()
    with catalog.catalog_lock(catdir):
        with pytest.raises(catalog.CatalogLocked, match="locked by another writer"):
            _register(catdir, second, tmp_path, lock_timeout=0.3)
        with pytest.raises(catalog.CatalogLocked):
            catalog.supersede_spec_version(catdir, "faithful-v2", "x", lock_timeout=0.0)
    assert acp.read_bytes() == before
    assert len(catalog.journal_records(acp)) == 1
    # released: the next writer gets in
    _register(catdir, second, tmp_path, lock_timeout=5)


def test_concurrent_writers_lose_nothing(tmp_path):
    """Several writers at once, each appending its own asset: every one of them is in the final
    catalog (the lock serialises the read-modify-write), and the journal has one record per write,
    each adding exactly that asset."""
    catdir = tmp_path / "catalog"
    acp = catdir / "asset_catalog.json"
    errors: list[BaseException] = []

    def writer(number: int) -> None:
        try:
            for round_ in range(3):
                with catalog.catalog_lock(catdir, timeout=60):
                    previous = catalog.read_catalog_file(acp)
                    ac = previous.document or {"assets": []}
                    catalog.write_catalog_document(
                        acp, {**ac, "assets": [*ac["assets"], {"asset_id": f"w{number}-r{round_}"}]},
                        previous, registered_by={"tool": f"writer {number}"})
        except BaseException as error:        # noqa: BLE001 - reported below
            errors.append(error)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)
    assert not errors, errors
    ids = sorted(a["asset_id"] for a in catalog.load_asset_catalog(catdir)["assets"])
    assert ids == sorted(f"w{n}-r{r}" for n in range(4) for r in range(3))
    records = catalog.read_journal(acp)
    assert [r["sequence"] for r in records] == list(range(1, 13))
    added = [c["new"]["asset_id"] for r in records for c in r["changes"]
             if c["key"].startswith("/assets/")]
    assert sorted(added) == ids                          # each asset added once, by one record
    story = catalog.journal_story(acp)
    assert story["consistent"] and all(r["applied"] for r in story["records"])
    for sequence in range(1, 13):
        document, exact = catalog.catalog_as_of(acp, sequence)
        assert exact and len(document["assets"]) == sequence


def test_relative_paths_are_logical_ids(tmp_path, two_bundles):
    first, _ = two_bundles
    catdir = tmp_path / "catalog"
    _register(catdir, first, tmp_path)
    for asset in catalog.load_asset_catalog(catdir)["assets"]:
        assert "\\" not in asset["relative_path"] and not os.path.isabs(asset["relative_path"])


@pytest.mark.parametrize("before, after", [
    (None, {"a": 1, "l": [1, 2]}),
    ({"a": 1, "l": [1, 2, 3], "gone": {"x": 1}}, {"a": 2, "l": [1, 9]}),
    ({"l": [1]}, {"l": [1, 2, 3], "new": "n"}),
    ({"a/b": 1, "t~": [0]}, {"a/b": 2, "t~": []}),
    ({"a": True}, {"a": 1}),                               # a change of type is a change
])
def test_the_changes_taken_back_give_the_document_before(before, after):
    changes = catalog.catalog_changes(before, after)
    import copy

    rebuilt = catalog._take_back(copy.deepcopy(after), changes)
    if before is None:
        assert rebuilt == {}
    else:
        assert rebuilt == before and all(type(rebuilt[k]) is type(before[k]) for k in before)
