"""Experimental-catalog registration + incremental verification ledger.

Writes the experimental catalog, ``<SOMA_DATA_ROOT>/experimental/catalog/`` by default (asset entries with
paths relative to the data root, no payload copies), and a per-take validation ledger so dataset expansion does not re-verify from scratch: a take is trusted
iff its content digest + spec_version + generator-code digest + validator-code digest all match a prior
PASS. Lineage is keyed by spec_id/spec_version (never contract_version / canonical).

**How a catalog file is written** (``asset_catalog.json``, ``<source>/source_manifest.json``). Every
writer of a catalog folder holds its lock, ``<catalog>/.catalog.lock`` (:func:`catalog_lock`), across the
read-modify-write, so two registrations never lose each other's entries. The lock is an operating-system
lock on that file (``msvcrt.locking`` of its first byte on Windows, ``fcntl.flock`` elsewhere), not its
presence: the file is created when missing and never deleted, by this module or on its behalf -- a lock
file another writer made stays exactly as it is. The new content goes to a temporary file in the same
folder, is flushed and fsynced, and replaces the file with ``os.replace``, so a reader sees the old file or
the new one and never half of either.

**The change journal.** Every write first appends one record to the file's journal,
``_history/<stem>.journal/`` beside it (:func:`journal_dir`): one small gzip-compressed JSON file per
write, ``<sequence>.<UTC timestamp>Z.json.gz``, created under a temporary name and linked into place
without ever overwriting a name, never rewritten and never pruned. A record holds when it was written,
what wrote it (``registered_by``: the tool, and the run id when the runner registers), the operation,
the SHA-256 and size of the file before and after, and one change per entry the write added, replaced
or removed -- its key (a JSON Pointer: ``/assets/<index>`` for an asset, ``/<field>`` for any other
field), the old entry (null when added) and the new one (null when removed). Nothing else: the journal
grows with what a write changes, never with the size of the catalog (an ``asset_catalog.json``
can reach hundreds of MB). The current file and its journal hold every earlier entry;
:func:`catalog_as_of` rebuilds the file as it was after any journalled write.

The order is: new content fsynced to its temporary file; the journal record fsynced and linked in; then
``os.replace``. A write that dies before the record is in changes nothing. One that dies between the
record and the replacement leaves a record whose "after" digest the file never had: its "before" digest
is the file's, and so is the next record's "before", which is how :func:`journal_story` tells an
unapplied record from an applied one. The reverse order could lose a replaced entry for good (file
replaced, process gone before its old value reached the journal); this one never loses anything.

**Registering a dataset again** (:func:`register_dataset`). A bundle regenerated in place keeps its
asset ids and paths and changes its bytes, so appending its assets would leave two live entries for
each id; a registration reconciles instead. An earlier live entry of the same dataset -- the same
asset id under the same bundle folder, or the same relative path in the same lineage -- is kept
when it is the asset exactly as
registered (:data:`ASSET_FIELDS`; fields an earlier writer added beside them do not count), and
otherwise marked superseded in place by the convention the catalog already follows
(``superseded``, ``supersession_reason`` :data:`REGISTRATION_REPLACED`, ``superseded_by`` the new
entry's asset id, ``superseded_by_sha256`` its digest): one "replaced" change of the write's journal
record each, which names the run. Only the assets no kept entry stands for are appended. A live
entry under the bundle folder, of the registration's lineage, that no registered asset stands for
-- a take no longer ``ok`` in the new INDEX, a take folder or a role file the bundle no longer has --
is marked superseded as well (``supersession_reason`` :data:`REGISTRATION_DROPPED`,
``superseded_by`` null): after an in-place replacement its file is gone or no longer part of the
dataset. A registration that would change nothing writes nothing: no journal record, the file
untouched.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import inspect
import json
import os
import re
import tempfile
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePath

from soma_synth.contracts import qmd_unified8_smpl18_spec as spec
from soma_synth.validation import checks as _checks
from soma_synth.validation import npz_io as _npz
from soma_synth.validation import report as _report

SCHEMA_ASSET = "soma_experimental_asset_catalog_v1"
SCHEMA_LEDGER = "soma_experimental_validation_ledger_v1"
SCHEMA_SOURCE = "soma_experimental_source_manifest_v1"
SCHEMA_JOURNAL = "soma_experimental_catalog_journal_v1"

#: The catalog folder's writer lock (see the module docstring). The lock file name is shared with
#: other catalog writers.
LOCK_NAME = ".catalog.lock"
#: The folder beside a catalog file that holds its change journal (``<stem>.journal/``).
HISTORY_DIR = "_history"
JOURNAL_SUFFIX = ".journal"
#: A journal record's name: ``<sequence, 8 digits>.<UTC timestamp>Z.json.gz``.
_RECORD_NAME = re.compile(r"^(\d{8})\.(\d{8}T\d{6}\.\d{6})Z\.json\.gz$")
#: How long a writer waits for the lock before giving up (a registration of AMASS rewrites a catalog of
#: several hundred MB, which takes a minute or two).
DEFAULT_LOCK_TIMEOUT_S = 900.0
_LOCK_POLL_S = 0.25
#: How long ``os.replace`` is retried on Windows, where a reader holding the file open makes it fail.
_REPLACE_RETRY_S = 30.0


class CatalogLocked(RuntimeError):
    """Another writer held the catalog folder's lock for longer than the writer would wait."""


ROLE_OF_FILE = {
    "small_reference.npz": "small",
    "large_reference.npz": "large",
    "anthro_reference.npz": "anthro",
    "smpl_root_translation.npz": "root_translation",
    "development_reference.npz": "development",
    "manifest.json": "manifest",
}

#: What a registration writes of an asset. Two entries that agree on all of these are the same asset,
#: whatever an earlier writer added beside them (``registration_evidence``, ``metadata``).
ASSET_FIELDS = ("asset_id", "role", "relative_path", "sha256", "bytes", "source", "spec_id",
                "spec_version")
#: The ``supersession_reason`` of an entry a later registration of its dataset replaced: the
#: value other catalog writers use for such entries.
REGISTRATION_REPLACED = "registration_replaced"
#: The ``supersession_reason`` of an entry of a dataset's folder and lineage that a later
#: registration of it no longer has an asset for (``superseded_by`` null; the journal record of the
#: write names the run).
REGISTRATION_DROPPED = "registration_dropped"


# ---------------------------------------------------------------- digests
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def take_digest(take_dir, filenames) -> str:
    """Ordered concat (⊕) of each present deliverable's sha256 hex, separated by '|', then hashed."""
    take_dir = Path(take_dir)
    parts = [sha256_file(take_dir / name) for name in sorted(filenames) if (take_dir / name).exists()]
    return hashlib.sha256("|".join(parts).encode("ascii")).hexdigest()


def module_digest(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def default_validator_digest() -> str:
    """Digest of the spec + validation code, so a validator change invalidates prior receipts."""
    return module_digest([inspect.getfile(m) for m in (spec, _checks, _npz, _report)])


@dataclass(frozen=True)
class FreshnessKey:
    take_digest: str
    spec_version: str
    generator_digest: str
    validator_digest: str


# ---------------------------------------------------------------- ledger
class ValidationLedger:
    def __init__(self, path) -> None:
        self.path = Path(path)
        self.takes: dict[str, dict] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.takes = data.get("takes", {})

    def is_fresh(self, take_id: str, key: FreshnessKey) -> bool:
        entry = self.takes.get(take_id)
        return bool(entry and entry.get("status") == "PASS" and entry.get("key") == asdict(key))

    def record(self, take_id: str, key: FreshnessKey, status: str) -> None:
        self.takes[take_id] = {"key": asdict(key), "status": status}

    def save(self) -> None:
        payload = {"schema": SCHEMA_LEDGER, "spec_version": spec.SPEC_VERSION, "takes": self.takes}
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- asset catalog
def asset_id(spec_id: str, spec_version: str, source: str, take_id: str, role: str) -> str:
    return f"{spec_id}:{spec_version}:{source}:{take_id}:{role}"


def load_asset_catalog(catalog_dir) -> dict:
    return json.loads((Path(catalog_dir) / "asset_catalog.json").read_text(encoding="utf-8"))


class Registration(list):
    """What :func:`register_dataset` registered: the dataset's assets, a list as it always returned,
    and what the registration did with them -- ``added`` (appended), ``unchanged`` (an identical
    live entry of the dataset was already there and is kept), ``superseded`` (earlier live entries
    of the dataset it marked superseded by one of its assets), ``dropped`` (earlier live entries of
    the dataset's folder and lineage it marked superseded because none of its assets stands for
    them)."""

    def __init__(self, assets=(), *, added: int = 0, unchanged: int = 0,
                 superseded: int = 0, dropped: int = 0) -> None:
        super().__init__(assets)
        self.added = added
        self.unchanged = unchanged
        self.superseded = superseded
        self.dropped = dropped

    def summary(self) -> str:
        return (f"{self.added} added, {self.unchanged} already registered, "
                f"{self.superseded} earlier entries superseded, "
                f"{self.dropped} dropped (no longer in the bundle)")


def _lineage(entry: Mapping[str, object]) -> tuple:
    return entry.get("source"), entry.get("spec_id"), entry.get("spec_version")


def _path_key(relative_path: object) -> str:
    """A relative path as registrations compare it: case-folded where the file system is (Windows),
    so a dataset folder typed in another case is still the same folder."""
    text = str(relative_path)
    return text.casefold() if os.name == "nt" else text


def _same_asset(entry: Mapping[str, object], asset: Mapping[str, object]) -> bool:
    return all(_same(entry.get(field), asset[field]) for field in ASSET_FIELDS)


def merge_registration(existing, assets, folder: str, lineage: tuple | None = None
                       ) -> tuple[list, int, int, int, int]:
    """The catalog's asset list after registering ``assets`` (one dataset's, from the bundle folder
    ``folder``, data-root-relative with forward slashes) over ``existing``, and how many assets were
    added, found already registered, earlier entries superseded, and earlier entries dropped.

    An earlier live entry (not ``superseded``) is the dataset's when it has the asset id of one of
    ``assets`` and lies under ``folder``, or has the relative path of one of them and its lineage
    (source, spec_id, spec_version). Walking from the newest, the first such entry that is the asset
    exactly as registered (:data:`ASSET_FIELDS`) is kept and the asset is not added again; every other
    one is replaced, at its index, by a copy marked ``superseded`` (reason
    :data:`REGISTRATION_REPLACED`, ``superseded_by`` / ``superseded_by_sha256`` the asset that
    replaces it).

    A live entry under ``folder`` of the registration's ``lineage`` that none of ``assets`` stands
    for is dropped: replaced, at its index, by a copy marked ``superseded`` with reason
    :data:`REGISTRATION_DROPPED` and ``superseded_by`` null. It is an asset the bundle no longer
    has -- a take no longer ``ok`` in its INDEX, a take folder or a role file that is gone -- and
    after an in-place replacement its file is gone too. ``lineage`` defaults to that of ``assets``;
    with no assets and no ``lineage``, or with ``folder`` the data root itself (``""``, ``"."``),
    nothing is dropped.

    Entries of other folders, other lineages or no match are left as they are, and no entry is
    removed or moved: the journal compares the asset list index by index. The entries are never
    edited in place; ``existing`` is not changed.
    """
    by_id: dict[str, Mapping[str, object]] = {}
    by_path: dict[tuple, Mapping[str, object]] = {}
    for asset in assets:
        if asset["asset_id"] in by_id:
            raise ValueError(f"asset id {asset['asset_id']} twice in one registration")
        by_id[asset["asset_id"]] = asset
        by_path[(_lineage(asset), _path_key(asset["relative_path"]))] = asset
    if lineage is None and assets:
        lineage = _lineage(assets[0])
    prefix = "" if folder in ("", ".") else _path_key(folder).rstrip("/") + "/"
    merged = list(existing)
    kept: set[str] = set()
    superseded = dropped = 0
    for index in range(len(merged) - 1, -1, -1):
        entry = merged[index]
        if not isinstance(entry, dict) or entry.get("superseded"):
            continue
        path = _path_key(entry.get("relative_path", ""))
        asset = by_id.get(entry.get("asset_id")) if path.startswith(prefix) else None
        if asset is None:
            asset = by_path.get((_lineage(entry), path))
        if asset is None:
            if prefix and lineage is not None and path.startswith(prefix) \
                    and _lineage(entry) == tuple(lineage):
                merged[index] = {**entry, "superseded": True,
                                 "supersession_reason": REGISTRATION_DROPPED,
                                 "superseded_by": None}
                dropped += 1
            continue
        if asset["asset_id"] not in kept and _same_asset(entry, asset):
            kept.add(asset["asset_id"])
            continue
        merged[index] = {**entry, "superseded": True, "supersession_reason": REGISTRATION_REPLACED,
                         "superseded_by": asset["asset_id"], "superseded_by_sha256": asset["sha256"]}
        superseded += 1
    added = [asset for asset in assets if asset["asset_id"] not in kept]
    merged.extend(added)
    return merged, len(added), len(kept), superseded, dropped


def register_dataset(
    catalog_dir,
    dataset_dir,
    *,
    source: str,
    spec_version: str,
    spec_id: str = spec.SPEC_ID,
    data_root=None,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT_S,
    registered_by: Mapping[str, object] | None = None,
) -> Registration:
    """Register this dataset's per-take assets in asset_catalog.json (v1) and catalog/<source>/.

    Paths are stored relative to ``data_root`` — no absolute paths, no copies. It defaults to
    SOMA_DATA_ROOT (``pipeline.paths.data_root``, which raises ``PathConfigError`` when it is not
    set), as for ``run`` and ``pipeline``. The bundle's parent is the lineage container, not a
    data root: paths relative to it would match none of the dataset's earlier entries.
    The assets are hashed first, without the lock; the catalog files are then read, reconciled and
    replaced under it (:func:`catalog_lock`, :func:`write_catalog_document`), each write journalled.
    Registering a dataset that is already there reconciles it (:func:`merge_registration`): earlier
    live entries of the dataset that are not the asset as registered now are marked superseded, so
    are those of its folder and lineage that it no longer has an asset for, and only the assets not
    already there are appended; a catalog file the registration would not change
    is not written at all. ``registered_by`` names what registered (the runner passes its run id; see
    :func:`registered_by_record`). ``lock_timeout`` is how long to wait for another writer
    (:class:`CatalogLocked` after it). Returns the assets, with the counts (:class:`Registration`).
    """
    catalog_dir = Path(catalog_dir)
    dataset_dir = Path(dataset_dir)
    if data_root is None:
        from soma_synth.pipeline import paths as _paths

        data_root = _paths.data_root()
    data_root = Path(data_root)
    # the folder every asset path starts with, spelled as they are (forward slashes whatever the
    # platform); a registration reconciles earlier entries under it only
    try:
        folder = PurePath(os.path.relpath(dataset_dir, data_root)).as_posix()
    except ValueError:                     # another drive on Windows: no asset path starts with it
        folder = dataset_dir.absolute().as_posix()

    idx = _checks.read_index(dataset_dir)
    assets: list[dict] = []
    for t in idx["takes"]:
        if t["status"] != "ok":
            continue
        td = dataset_dir / t["rel"]
        if not td.is_dir():
            continue
        _sm, src, caps, _f = _checks.take_capabilities(td)
        for fname in spec.expected_deliverables(caps):
            p = td / fname
            if not p.exists():
                continue
            role = ROLE_OF_FILE.get(fname, "other")
            # a logical id: forward slashes whatever the platform, never os.sep
            rel = PurePath(os.path.relpath(p, data_root)).as_posix()
            assets.append(
                {
                    "asset_id": asset_id(spec_id, spec_version, source, t["take_id"], role),
                    "role": role,
                    "relative_path": rel,
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                    "source": source,
                    "spec_id": spec_id,
                    "spec_version": spec_version,
                }
            )

    (catalog_dir / source).mkdir(parents=True, exist_ok=True)
    take_ids = sorted({a["asset_id"].rsplit(":", 1)[0] for a in assets})
    by = registered_by_record(registered_by)
    operation = {"name": "register_dataset", "source": source, "spec_id": spec_id,
                 "spec_version": spec_version, "dataset": _logical(dataset_dir, data_root),
                 "assets": len(assets)}
    with catalog_lock(catalog_dir, timeout=lock_timeout):
        acp = catalog_dir / "asset_catalog.json"
        previous = read_catalog_file(acp)
        if previous.document is not None:
            ac = previous.document
        else:
            ac = {
                "schema": SCHEMA_ASSET,
                "artifact_class": "experimental_non_candidate",
                "distribution_scope": "internal_only",
                "root_handle": "SOMA_DATA_ROOT",
                "note": "paths are SOMA_DATA_ROOT-relative; no absolute paths, no payload copies.",
                "assets": [],
            }
        # a new document beside the one read, never an edit of it: the journal compares the two
        merged, added, unchanged, superseded, dropped = merge_registration(
            ac.get("assets", []), assets, folder, lineage=(source, spec_id, spec_version))
        operation.update(added=added, unchanged=unchanged, superseded=superseded, dropped=dropped)

        manifest_path = catalog_dir / source / "source_manifest.json"
        manifest = {
            "schema": SCHEMA_SOURCE,
            "source": source,
            "spec_id": spec_id,
            "spec_version": spec_version,
            "artifact_class": "experimental_non_candidate",
            "distribution_scope": "internal_only",
            "generated_take_count": len(take_ids),
        }
        manifest_before = read_catalog_file(manifest_path)
        if manifest_before.document != manifest:
            write_catalog_document(manifest_path, manifest, manifest_before, registered_by=by,
                                   operation=operation)

        if previous.document is None or added or superseded or dropped:
            new = dict(ac)
            new["assets"] = merged
            write_catalog_document(acp, new, previous, registered_by=by, operation=operation)
    return Registration(assets, added=added, unchanged=unchanged, superseded=superseded,
                        dropped=dropped)


def supersede_spec_version(catalog_dir, spec_version: str, superseded_by: str, *,
                           lock_timeout: float = DEFAULT_LOCK_TIMEOUT_S,
                           registered_by: Mapping[str, object] | None = None) -> int:
    """Flag every asset of `spec_version` as superseded_by another version (e.g. faithful-v1 -> v2).
    Each flagged asset is one "replaced" change of the write's journal record, the old entry in it."""
    acp = Path(catalog_dir) / "asset_catalog.json"
    with catalog_lock(catalog_dir, timeout=lock_timeout):
        previous = read_catalog_file(acp)
        if previous.document is None:
            raise FileNotFoundError(f"no catalog file {acp}")
        ac = previous.document
        n = 0
        flagged = []
        for a in ac.get("assets", []):
            if a.get("spec_version") == spec_version and not a.get("superseded"):
                a = {**a, "superseded": True, "superseded_by": superseded_by}
                n += 1
            flagged.append(a)
        new = {**ac, "assets": flagged}
        write_catalog_document(
            acp, new, previous, registered_by=registered_by_record(registered_by),
            operation={"name": "supersede_spec_version", "spec_version": spec_version,
                       "superseded_by": superseded_by, "assets": n})
    return n


def _logical(path, data_root) -> str:
    """``path`` relative to ``data_root`` with forward slashes; its bare name when it is elsewhere."""
    try:
        relative = os.path.relpath(Path(path), Path(data_root))
    except ValueError:                     # another drive on Windows
        return Path(path).name
    return Path(path).name if relative.startswith("..") else PurePath(relative).as_posix()


# ---------------------------------------------------------------- writing a catalog file
def _lock_byte(fd: int) -> None:
    """Take the OS lock on ``fd`` without waiting; OSError when another holder has it."""
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_byte(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def catalog_lock(catalog_dir, *, timeout: float = DEFAULT_LOCK_TIMEOUT_S) -> Iterator[Path]:
    """Hold the catalog folder's writer lock, ``<catalog_dir>/.catalog.lock``, for the ``with`` block.

    The file is opened (created when missing, never truncated) and locked by the operating system; a
    writer that finds it locked -- or, on Windows, cannot open it because its holder opened it
    exclusively -- retries every quarter second for ``timeout`` seconds and then raises
    :class:`CatalogLocked`. The lock goes when the block ends or the process dies; the file stays. It is
    never deleted: it may be another writer's, and deleting a lock file someone is about to lock lets two
    writers in at once.
    """
    folder = Path(catalog_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / LOCK_NAME
    deadline = time.monotonic() + max(0.0, float(timeout))
    fd = None
    while True:
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0))
            _lock_byte(fd)
            break
        except OSError as error:
            if fd is not None:
                os.close(fd)
                fd = None
            if time.monotonic() >= deadline:
                raise CatalogLocked(
                    f"the catalog {folder} is locked by another writer ({path}: "
                    f"{type(error).__name__}: {error}); waited {timeout:.0f} s. Let that registration "
                    "finish and run again. The lock is held by a running process and goes when it "
                    f"ends; {LOCK_NAME} itself is never deleted"
                ) from None
            time.sleep(_LOCK_POLL_S)
    try:
        yield path
    finally:
        try:
            _unlock_byte(fd)
        finally:
            os.close(fd)


# ---------------------------------------------------------------- reading a catalog file
class JournalError(RuntimeError):
    """A catalog file's journal cannot tell the story asked of it."""


@dataclass(frozen=True)
class CatalogRead:
    """A catalog file as a writer read it (under the lock): the parsed content, None when there is
    no file, and the SHA-256 and size of the bytes it was parsed from."""

    document: object
    sha256: str | None
    size: int | None


def read_catalog_file(path) -> CatalogRead:
    """Read the catalog file ``path`` once: its content, digest and size."""
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError:
        return CatalogRead(None, None, None)
    return CatalogRead(json.loads(data.decode("utf-8")), hashlib.sha256(data).hexdigest(), len(data))


def serialise(document) -> bytes:
    """A catalog file's bytes, as they were always written (indent 2, UTF-8, LF)."""
    return (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def registered_by_record(registered_by: Mapping[str, object] | None = None) -> dict:
    """What a journal record names as its writer: ``tool`` and ``run_id`` (the run's record folder,
    ``run_<identity>``, when the runner registers), as ``registered_by`` gives them, else this
    module's own name and no run."""
    record: dict[str, object] = {"tool": "soma_synth.datasets.catalog", "run_id": None}
    record.update({str(key): value for key, value in dict(registered_by or {}).items()})
    return record


# ---------------------------------------------------------------- what a write changed
def _pointer(*parts) -> str:
    """A JSON Pointer (RFC 6901) to ``parts``: ``/assets/12``, ``/note``."""
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def _pointer_parts(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise JournalError(f"not a JSON Pointer: {pointer!r}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _same(a, b) -> bool:
    return a is b or (type(a) is type(b) and a == b)


def _change(key: str, kind: str, old, new) -> dict:
    return {"key": key, "change": kind, "old": old, "new": new}


def catalog_changes(before, after) -> list[dict]:
    """The changes that turn the catalog document ``before`` (None: no file) into ``after``, as a
    journal record holds them: ``{"key", "change": added|replaced|removed, "old", "new"}``.

    A top-level field is one entry (``/<field>``). The items of a list field are entries of their own
    (``/<field>/<index>``), compared index by index: the catalog's writers append assets and change
    them in place, so an index names the same asset before and after, and a registration that adds
    one asset to a catalog of a million is one change. The list is in patch order (a list's tail
    removed from its end, added in index order), so taking the changes back last-first restores
    ``before`` (:func:`catalog_as_of`)."""
    if not isinstance(after, dict) or not isinstance(before, (dict, type(None))):
        if _same(before, after):
            return []
        return [_change("", "added" if before is None else "replaced", before, after)]
    before = {} if before is None else before
    changes = [_change(_pointer(key), "removed", old, None)
               for key, old in before.items() if key not in after]
    for key, new in after.items():
        if key not in before:
            if isinstance(new, list):
                changes.append(_change(_pointer(key), "added", None, []))
                changes += [_change(_pointer(key, index), "added", None, item)
                            for index, item in enumerate(new)]
            else:
                changes.append(_change(_pointer(key), "added", None, new))
            continue
        old = before[key]
        if isinstance(old, list) and isinstance(new, list):
            common = min(len(old), len(new))
            changes += [_change(_pointer(key, index), "replaced", old[index], new[index])
                        for index in range(common) if not _same(old[index], new[index])]
            changes += [_change(_pointer(key, index), "removed", old[index], None)
                        for index in range(len(old) - 1, common - 1, -1)]
            changes += [_change(_pointer(key, index), "added", None, new[index])
                        for index in range(common, len(new))]
        elif not _same(old, new):
            changes.append(_change(_pointer(key), "replaced", old, new))
    return changes


def _take_back(document, changes):
    """``document`` with ``changes`` (one record's) undone, last first. Refuses a change that does not
    fit the document, which would mean the record is not the one that produced it."""
    for change in reversed(changes):
        parts, kind, old = _pointer_parts(change["key"]), change["change"], change["old"]
        if not parts:
            document = old
            continue
        parent = document
        try:
            for part in parts[:-1]:
                parent = parent[int(part)] if isinstance(parent, list) else parent[part]
            last = parts[-1]
            if isinstance(parent, list):
                index = int(last)
                if kind == "added":
                    if index != len(parent) - 1:
                        raise JournalError(f"{change['key']} is not the last item")
                    del parent[index]
                elif kind == "removed":
                    if index != len(parent):
                        raise JournalError(f"{change['key']} does not follow the last item")
                    parent.append(old)
                else:
                    parent[index] = old
            elif kind == "added":
                del parent[last]
            else:
                parent[last] = old
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise JournalError(f"cannot take back {kind} {change['key']}: {error}") from None
    return document


# ---------------------------------------------------------------- the journal
def journal_dir(path) -> Path:
    """The change journal of the catalog file ``path``: ``<its folder>/_history/<stem>.journal/``."""
    target = Path(path)
    return target.parent / HISTORY_DIR / f"{target.stem}{JOURNAL_SUFFIX}"


def journal_records(path) -> list[Path]:
    """The record files of ``path``'s journal in sequence order (a temporary name is no record)."""
    folder = journal_dir(path)
    if not folder.is_dir():
        return []
    found = []
    for entry in folder.iterdir():
        match = _RECORD_NAME.match(entry.name)
        if match and entry.is_file():
            found.append((int(match.group(1)), entry.name, entry))
    return [entry for *_, entry in sorted(found)]


def read_journal(path) -> list[dict]:
    """Every record of ``path``'s journal, in sequence order."""
    return [json.loads(gzip.decompress(entry.read_bytes()).decode("utf-8"))
            for entry in journal_records(path)]


def _link_new(temporary: Path, target: Path) -> None:
    """Give ``temporary`` the name ``target``, raising FileExistsError rather than replacing a file
    (Windows' rename refuses an existing name; elsewhere a hard link does, then the temporary
    name goes)."""
    if os.name == "nt":
        os.rename(temporary, target)
        return
    try:
        os.link(temporary, target)
    except FileExistsError:
        raise
    except OSError:
        if target.exists():
            raise FileExistsError(target) from None
        os.rename(temporary, target)
        return
    os.unlink(temporary)


def _append_journal(path: Path, record: Mapping[str, object]) -> Path:
    """Add ``record`` to ``path``'s journal as the next sequence and return its file: written to a
    temporary name, fsynced, then linked in under a name nothing had (never an overwrite)."""
    folder = journal_dir(path)
    folder.mkdir(parents=True, exist_ok=True)
    while True:
        existing = journal_records(path)
        sequence = int(_RECORD_NAME.match(existing[-1].name).group(1)) + 1 if existing else 1
        now = datetime.now(UTC)
        body = {"schema": SCHEMA_JOURNAL, "catalog_file": path.name, "sequence": sequence,
                "utc": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), **record}
        name = f"{sequence:08d}.{now.strftime('%Y%m%dT%H%M%S.%f')}Z.json.gz"
        data = gzip.compress(json.dumps(body, ensure_ascii=False, separators=(",", ":"))
                             .encode("utf-8"), mtime=0)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=folder)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            _link_new(Path(temporary), folder / name)
        except FileExistsError:
            Path(temporary).unlink(missing_ok=True)
            continue                          # the name was taken meanwhile: the next sequence
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        _fsync_folder(folder)
        return folder / name


def _file_digest(path: Path) -> tuple[str, int] | None:
    try:
        return sha256_file(path), path.stat().st_size
    except FileNotFoundError:
        return None


def journal_story(path) -> dict:
    """Which journalled writes reached the catalog file ``path``.

    Walks the records from the newest back, starting from the file's current digest: a record whose
    "after" digest is the state reached was applied, and the state before it is its "before"; one
    whose "after" is not but whose "before" is was never applied (its write died, or failed, between
    the record and the replacement); one that neither explains is marked unexplained (``applied``
    None) -- the file changed outside the journal -- and the walk goes on from its "before".
    """
    target = Path(path)
    current = _file_digest(target)
    state = current[0] if current else None
    story: list[dict] = []
    consistent = True
    for record in reversed(read_journal(target)):
        before = (record.get("before") or {}).get("sha256")
        after = (record.get("after") or {}).get("sha256")
        if after == state:
            applied, state = True, before
        elif before == state:
            applied = False
        else:
            applied, state, consistent = None, before, False
        story.append({"sequence": record.get("sequence"), "utc": record.get("utc"),
                      "applied": applied, "registered_by": record.get("registered_by"),
                      "operation": (record.get("operation") or {}).get("name"),
                      "counts": record.get("counts")})
    story.reverse()
    return {"file": ({"sha256": current[0], "bytes": current[1]} if current else None),
            "consistent": consistent, "records": story}


def catalog_as_of(path, sequence: int) -> tuple[object, bool]:
    """The catalog file ``path`` as it was right after journal record ``sequence`` was applied
    (``0``: as it was before the first journalled write; None when there was no file then).

    Rebuilt from the current file by taking back the changes of every later applied record, newest
    first. Returns ``(document, bytes_match)``: whether the rebuilt document serialises to exactly
    the bytes the journal says the file had. JournalError when record ``sequence`` was never
    applied, is not in the journal, or a later record cannot be explained.
    """
    target = Path(path)
    records = read_journal(target)
    story = {entry["sequence"]: entry["applied"] for entry in journal_story(target)["records"]}
    if sequence == 0:
        expected = (records[0].get("before") or {}).get("sha256") if records else None
        if not records:
            current = read_catalog_file(target)
            return current.document, True
    else:
        chosen = next((record for record in records if record.get("sequence") == sequence), None)
        if chosen is None:
            raise JournalError(f"no record {sequence} in {journal_dir(target)}")
        if story.get(sequence) is not True:
            raise JournalError(f"record {sequence} of {journal_dir(target)} was never applied"
                               if story.get(sequence) is False else
                               f"record {sequence} of {journal_dir(target)} cannot be explained")
        expected = chosen["after"]["sha256"]
    document = read_catalog_file(target).document
    for record in reversed(records):
        if record.get("sequence", 0) <= sequence:
            break
        applied = story.get(record.get("sequence"))
        if applied is None:
            raise JournalError(f"record {record.get('sequence')} of {journal_dir(target)} cannot be "
                               "explained: the file changed outside the journal")
        if applied:
            document = _take_back(document, record.get("changes") or [])
    if expected is None:
        return None, True
    return document, hashlib.sha256(serialise(document)).hexdigest() == expected


# ---------------------------------------------------------------- writing a catalog file
def _replace(source: str, target: Path) -> None:
    """``os.replace``, retried on Windows while a reader holds ``target`` open."""
    deadline = time.monotonic() + _REPLACE_RETRY_S
    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or time.monotonic() >= deadline:
                raise
            time.sleep(0.2)


def _fsync_folder(folder: Path) -> None:
    """Make a rename durable where the platform allows (not on Windows, which cannot open a folder)."""
    if os.name == "nt":
        return
    try:
        fd = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_catalog_file(path, data: bytes, record: Mapping[str, object]) -> Path:
    """Replace the catalog file ``path`` with ``data`` atomically, journalling ``record`` first.

    ``data`` goes to a temporary file in the same folder and is flushed and fsynced; ``record`` (what
    the write changes, :func:`write_catalog_document`) is appended to the journal with its sequence,
    its time and the digest of ``data`` (:func:`journal_dir`); then the temporary file replaces
    ``path`` with ``os.replace``. A failure before the record is in leaves ``path`` and the journal as
    they were; one after it leaves ``path`` as it was and the record unapplied (it stays: the
    journal is never pruned, and :func:`journal_story` tells). The temporary file never survives a
    failure the process sees. Returns the record's file. The caller holds :func:`catalog_lock`.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp",
                                             dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        after = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        body: dict[str, object] = {}
        for key, value in record.items():             # "after" read beside "before"
            body[key] = value
            if key == "before":
                body["after"] = after
        body.setdefault("after", after)
        entry = _append_journal(target, body)
        _replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    _fsync_folder(target.parent)
    return entry


def write_catalog_document(path, document, previous: CatalogRead | None = None, *,
                           registered_by: Mapping[str, object] | None = None,
                           operation: Mapping[str, object] | None = None) -> Path:
    """Write ``document`` as the catalog file ``path`` (:func:`serialise`), journalling what changed
    since ``previous`` -- the file as this writer read it under the lock (read now when not given)
    -- with ``registered_by`` and ``operation`` (:func:`write_catalog_file`). Returns the record's
    file. The caller holds :func:`catalog_lock`."""
    target = Path(path)
    if previous is None:
        previous = read_catalog_file(target)
    data = serialise(document)
    changes = catalog_changes(previous.document, document)
    record = {
        "registered_by": registered_by_record(registered_by),
        "operation": dict(operation or {"name": "write"}),
        "before": ({"sha256": previous.sha256, "bytes": previous.size}
                   if previous.sha256 is not None else None),
        "counts": {kind: sum(1 for change in changes if change["change"] == kind)
                   for kind in ("added", "replaced", "removed")},
        "changes": changes,
    }
    return write_catalog_file(target, data, record)
