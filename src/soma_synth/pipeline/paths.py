"""Where the data plane is on this machine, read from the environment when asked, never before.

Every location the generators use comes from three environment variables; none is a literal,
because a literal is one machine's layout, not the pipeline's:

``SOMA_DATA_ROOT``
    The output root. Bundles live under ``runs/experimental_generation_poc_demo/<lineage>``
    beneath it. Required: there is no default, because a default is one machine's layout.
``SOMA_SOURCE_ROOT``
    The folder that holds one folder per source (:data:`SOURCE_FOLDERS`). Defaults to
    ``<SOMA_DATA_ROOT>/extracted``.
``SOMA_BODY_MODEL_DIR``
    The folder holding ``SMPL_{MALE,FEMALE,NEUTRAL}_clean.npz``. Defaults to
    ``<SOMA_DATA_ROOT>/body_models/smpl``. Honoured (ADR-0041): the runner resolves the
    folder through :func:`body_model_dir` too and hashes the one it resolved
    (PIPELINE_GOVERNANCE §10.2 item 6), so a run that loads its models from elsewhere records
    those models rather than the default folder's.

A fourth variable, ``SOMA_SHARED_DRIVE_NAMES``, names no location the generators use: it lists the
folder names of team shared drives, which output written inside draws a warning
(:func:`on_shared_drive`, :func:`warn_if_synced`).

Nothing here reads the environment at import time. A module that imports this one stays
importable on a machine where none of the three is set, and the first call that needs a location
raises :class:`PathConfigError` naming the variable.

A manifest names a source file by its logical id, ``extracted/<folder>/<path>``
(:func:`logical_source_id`), whatever folder the source was physically read from, so a bundle made
from a relocated source tree records the same strings as one made where the sources were first
extracted.
"""

from __future__ import annotations

import os
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import Path, PurePath, PurePosixPath

__all__ = [
    "BODY_MODELS",
    "BODY_MODEL_DIR_ENV",
    "DATA_ROOT_ENV",
    "EVIDENCE_FOLDERS",
    "EXTRACTED",
    "POC_DEMO",
    "PROTECTED_DATA_FILES",
    "RAW_ARCHIVES",
    "RUN_RECORDS",
    "SHARED_DRIVE_NAMES_ENV",
    "SOURCE_FOLDERS",
    "SOURCE_ROOT_ENV",
    "OutputLocationRefused",
    "PathConfigError",
    "PathOutsideRootError",
    "body_model_dir",
    "body_model_override",
    "check_bundle_dir",
    "check_output_dir",
    "check_readme_root",
    "check_record_dir",
    "check_record_file",
    "data_relative",
    "data_root",
    "evidence_folder",
    "existing_source_dir",
    "is_default_lineage",
    "lineage_dir",
    "logical_source_id",
    "on_shared_drive",
    "refuse_partial_run_into_lineage",
    "shared_drive_names",
    "source_dir",
    "source_folder",
    "source_root",
    "source_root_override",
    "synced_location",
    "warn_if_synced",
]

DATA_ROOT_ENV = "SOMA_DATA_ROOT"
SOURCE_ROOT_ENV = "SOMA_SOURCE_ROOT"
BODY_MODEL_DIR_ENV = "SOMA_BODY_MODEL_DIR"
SHARED_DRIVE_NAMES_ENV = "SOMA_SHARED_DRIVE_NAMES"

#: The data-root-relative folder the sources were extracted into, and the first segment of every
#: logical source id.
EXTRACTED = "extracted"
#: Where the experimental bundles live under the data root, one directory per lineage.
POC_DEMO = "runs/experimental_generation_poc_demo"
#: Where the clean SMPL body models live under the data root unless SOMA_BODY_MODEL_DIR says
#: otherwise.
BODY_MODELS = "body_models/smpl"

#: Source name -> the folder holding it under the source root. Two differ from the source name.
SOURCE_FOLDERS = {
    "amass": "amass",
    "prism": "prism",
    "gaitex": "gaitex",
    "hknu": "hknu_fullbody",
    "addbiomechanics": "addbiomechanics",
}

#: Where the source archives were downloaded to under the data root. Like extracted/, retention
#: rule 1.4 forbids deleting it and the pipeline never writes into it. The retention rules are
#: docs/guides/RETENTION_RULES.md; every "retention rule 1.x" in this package means that list.
RAW_ARCHIVES = "raw_archives"
#: Evidence and run records beside the lineage directories (never deleted, retention rule 1.4):
#: never a bundle.
EVIDENCE_FOLDERS = ("_superseded", "_runs", "_manifest_backfill")
#: The evidence folder run records go in (``<lineage container>/_runs``). A run record or a
#: catalog may be written there (:func:`check_record_dir`); a bundle may not.
RUN_RECORDS = "_runs"
#: Files of the data plane that nothing overwrites (retention rule 1.4), relative to a data root:
#: its README and the optional bookkeeping files a data root may keep beside it (an index of the
#: data root and an inventory of its source archives). Absent ones are simply not there to protect.
PROTECTED_DATA_FILES = ("README.md", "MASTER.md", "state/local_archive_inventory.json")

#: A path component that marks a cloud-synchronised folder. Output there draws a warning
#: (:func:`warn_if_synced`): the sync client can rewrite files underneath a running generator and
#: copies internal-only data off the PC.
#: Dropbox, OneDrive, Google Drive (the desktop client's top-level folders in English and Korean:
#: "My Drive" / "내 드라이브", "Shared drives" / "공유 드라이브", "Other computers" /
#: "다른 컴퓨터", and the older "Google Drive" folder), Synology Drive ("SynologyDrive") and
#: iCloud Drive (Windows: ``iCloudDrive``; macOS: ``com~apple~CloudDocs``). These are the folder
#: names the clients create on Windows, macOS and Linux alike (``~/Dropbox``, ``~/OneDrive*``,
#: ``~/Google Drive``, ``/Volumes/GoogleDrive*``), so they are matched on every platform.
_SYNCED_FOLDER_EXACT = ("dropbox", "my drive", "shared drives", "other computers",
                        "내 드라이브", "공유 드라이브", "다른 컴퓨터", "iclouddrive",
                        "com~apple~clouddocs")
_SYNCED_FOLDER_PREFIXES = ("dropbox (", "onedrive", "google drive", "googledrive", "synologydrive",
                           "synology drive")
#: POSIX only: two consecutive components that mark a synchronised tree on macOS, whose File
#: Provider clients (Dropbox, OneDrive, Google Drive, Box, ...) all live under
#: ``~/Library/CloudStorage/<client>-<account>`` and whose iCloud Drive is
#: ``~/Library/Mobile Documents``. A pair, so a folder merely named ``CloudStorage`` elsewhere is
#: not refused.
_SYNCED_FOLDER_PAIRS_POSIX = (("library", "cloudstorage"), ("library", "mobile documents"))
#: POSIX only: GNOME's online accounts mount Google Drive through gvfs as
#: ``/run/user/<uid>/gvfs/google-drive:host=<domain>,user=<name>``.
_SYNCED_FOLDER_PREFIXES_POSIX = ("google-drive:",)


class PathConfigError(RuntimeError):
    """A location the pipeline needs is not configured, or is configured wrongly."""


class PathOutsideRootError(PathConfigError, ValueError):
    """A path that has to sit under a root does not.

    Also a ``ValueError``, which is what ``Path.relative_to`` raised where this is now used.
    """


class OutputLocationRefused(PathConfigError):
    """An output directory points somewhere a generator must never write."""


def _env_path(name: str) -> Path | None:
    """The variable as a path, None when unset or blank; a relative value is refused."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    path = Path(value.strip())
    if not path.is_absolute():
        raise PathConfigError(
            f"{name}={value!r} is not an absolute path; set it to the full path of the folder"
        )
    return path


def data_root() -> Path:
    """``SOMA_DATA_ROOT``: the output root on this machine. Must be set, absolute and existing."""
    path = _env_path(DATA_ROOT_ENV)
    if path is None:
        raise PathConfigError(
            f"{DATA_ROOT_ENV} is not set. Set it to the local data root (the folder that holds "
            f"{POC_DEMO}/ and, unless {SOURCE_ROOT_ENV} says otherwise, {EXTRACTED}/); "
            "there is no default"
        )
    if not path.is_dir():
        raise PathConfigError(f"{DATA_ROOT_ENV}={path} does not exist or is not a directory")
    return path


def source_root_override() -> Path | None:
    """``SOMA_SOURCE_ROOT`` as a path, or None when it is unset; a relative value is refused.

    The runner records the source root its generators will see (``SOMA_SOURCE_ROOT``, else
    ``<its data root>/extracted``) without requiring it to exist, which :func:`source_root` does.
    """
    return _env_path(SOURCE_ROOT_ENV)


def source_root() -> Path:
    """``SOMA_SOURCE_ROOT``, else ``<SOMA_DATA_ROOT>/extracted``. Must exist."""
    path = _env_path(SOURCE_ROOT_ENV)
    named = SOURCE_ROOT_ENV
    if path is None:
        path = data_root() / EXTRACTED
        named = f"{DATA_ROOT_ENV}/{EXTRACTED} ({SOURCE_ROOT_ENV} is unset)"
    if not path.is_dir():
        raise PathConfigError(f"the source root {path} ({named}) does not exist")
    return path


def _same_folder(a: Path, b: Path) -> bool:
    return a == b or os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))


def body_model_override() -> Path | None:
    """``SOMA_BODY_MODEL_DIR`` as a path, or None when it is unset; a relative value is refused."""
    return _env_path(BODY_MODEL_DIR_ENV)


def body_model_dir(root: str | os.PathLike[str] | None = None) -> Path:
    """The folder the body models are loaded from: ``SOMA_BODY_MODEL_DIR``, else
    ``<root>/body_models/smpl`` with ``root`` defaulting to ``SOMA_DATA_ROOT``.

    The runner resolves the folder through this function as well (``pipeline.body_models``,
    with ``root`` its data root, which it also hands its generators as ``SOMA_DATA_ROOT``) and
    hashes the folder it resolved (§10.2 item 6), so the record names the models the run loaded,
    wherever they are.

    Not checked for existence here: the model a gender selects is opened by the caller, whose
    error names the missing file.
    """
    override = body_model_override()
    if override is not None:
        return override
    base = Path(root) if root is not None else data_root()
    return base / BODY_MODELS


def source_folder(source: str) -> str:
    """The folder name a source lives in under the source root."""
    try:
        return SOURCE_FOLDERS[source]
    except KeyError:
        raise PathConfigError(
            f"unknown source {source!r}; known sources: {', '.join(sorted(SOURCE_FOLDERS))}"
        ) from None


def source_dir(source: str) -> Path:
    """``<source root>/<folder>`` for a source. Not checked for existence; the reader says."""
    folder = source_folder(source)
    return source_root() / folder


def existing_source_dir(source: str, path: str | os.PathLike[str] | None = None) -> Path:
    """The folder a source is read from, refused by name unless it is a directory.

    ``path`` is the folder a generator's flag names; without it, :func:`source_dir`. A batch
    generator lists or globs its source folder, and a missing folder lists as empty: without this
    check the run selects nothing and still rewrites its bundle's INDEX.json, fit record and
    description over zero takes.
    """
    name = source_folder(source)
    if path is None:
        folder = source_dir(source)
        named = (f"{SOURCE_ROOT_ENV}/{name}" if _env_path(SOURCE_ROOT_ENV) is not None
                 else f"{DATA_ROOT_ENV}/{EXTRACTED}/{name}")
    else:
        folder = Path(path)
        named = "as given"
    if not folder.is_dir():
        raise PathConfigError(
            f"the {source} source folder {folder} ({named}) does not exist or is not a directory"
        )
    return folder


def lineage_dir(name: str, root: str | os.PathLike[str] | None = None) -> Path:
    """``<root>/runs/experimental_generation_poc_demo/<name>``; ``root`` defaults to the data root."""
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise PathConfigError(f"{name!r} is not a lineage directory name")
    base = Path(root) if root is not None else data_root()
    return base / POC_DEMO / name


def is_default_lineage(path: str | os.PathLike[str], name: str) -> bool:
    """Whether ``path`` is the lineage directory ``name`` under the configured ``SOMA_DATA_ROOT``.

    Compared as written and then resolved, case-insensitively where the platform is. False when
    ``SOMA_DATA_ROOT`` is unset, since there is no default lineage directory then; a value that is
    set but wrong raises, as everywhere in this module.
    """
    if _env_path(DATA_ROOT_ENV) is None:
        return False
    return _same_folder(Path(path), lineage_dir(name))


def refuse_partial_run_into_lineage(
    path: str | os.PathLike[str], name: str, selection: Iterable[str], flag: str, note: str = ""
) -> None:
    """Refuse a run that selects part of a source when its output is the production lineage.

    ``selection`` is the selection flags the run was given as they read on the command line
    (``["--groups CMU/01", "--limit 5"]``); empty means a full run, which may write there. ``flag``
    is the option that moves the output, named in the refusal with a scratch location under
    ``<SOMA_DATA_ROOT>/tmp``, and ``note`` is appended to it.

    A partial run into the production directory rewrites the bundle's INDEX.json, fit record and
    description over the selection alone and exits 0: the production bundle then describes a
    sample, and every take outside it is dropped from the index its readers go by.
    """
    chosen = [flag_text for flag_text in selection if flag_text]
    if not chosen or not is_default_lineage(path, name):
        return
    scratch = data_root() / "tmp" / name
    raise OutputLocationRefused(
        f"refusing to write {path}: {' '.join(chosen)} selects part of the source, and {path} is "
        f"the production lineage directory {DATA_ROOT_ENV}/{POC_DEMO}/{name}, whose INDEX.json, "
        "fit record and description a partial run would rewrite over the selection alone. Pass "
        f"{flag} a scratch location instead, e.g. {flag} {scratch}" + (f" {note}" if note else "")
    )


def _relative_posix(path: str | os.PathLike[str], base: Path, what: str) -> str:
    """``path`` relative to ``base`` with forward slashes; lexical first, then resolved.

    The lexical comparison is the one the generators made before (``Path.relative_to``), so a path
    that was accepted then gives the same string now. Resolving both sides is the fallback for a
    path spelled differently from its root (relative, or through a link).
    """
    path = Path(path)
    try:
        relative = path.relative_to(base)
    except ValueError:
        try:
            relative = path.resolve().relative_to(base.resolve())
        except ValueError:
            raise PathOutsideRootError(f"{path} is not inside {what} ({base})") from None
    if not relative.parts:
        raise PathOutsideRootError(f"{path} is {what} itself, not a path inside it")
    return relative.as_posix()


def data_relative(path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> str:
    """``path`` relative to the data root (or ``root``), forward slashes; refused if outside."""
    base = Path(root) if root is not None else data_root()
    return _relative_posix(path, base, "the data root")


def logical_source_id(
    source: str, path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None
) -> str:
    """The id a manifest records for a source file: ``extracted/<folder>/<relative path>``.

    ``root`` is the directory that was read as this source's folder (``--extracted``, say); it
    defaults to :func:`source_dir`. The id does not depend on where that directory physically is.
    """
    folder = source_folder(source)
    base = Path(root) if root is not None else source_dir(source)
    return f"{EXTRACTED}/{folder}/{_relative_posix(path, base, f'the {source} source folder')}"


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


#: The label a run's own data root is named by in a refusal (``root=`` below).
_RUN_ROOT = "the run's data root"


def _guarded_roots(root: str | os.PathLike[str] | None = None) -> list[tuple[Path, str]]:
    """[(data root, how it is named), ...] whose folders are guarded: the run's own ``root``
    and ``SOMA_DATA_ROOT`` when it names another folder, or ``SOMA_DATA_ROOT`` alone without a
    run root (see :func:`_configured_locations`)."""
    roots: list[tuple[Path, str]] = []
    if root is not None:
        roots.append((Path(root), _RUN_ROOT))
        configured = _env_path(DATA_ROOT_ENV)
        if configured is not None and not _same_folder(configured, Path(root)):
            roots.append((configured, DATA_ROOT_ENV))
    elif _env_path(DATA_ROOT_ENV) is not None:
        roots.append((data_root(), DATA_ROOT_ENV))
    return roots


def _configured_locations(
    root: str | os.PathLike[str] | None = None, *, allow_run_records: bool = False
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    """([(label, read-only folder), ...], [(label, container folder), ...]) as far as the
    environment (and ``root``) configures them.

    Read-only folders are refused with everything beneath them; containers are refused only as
    themselves (a bundle goes in its own directory beneath one).

    A variable that is unset configures nothing and is skipped: a run that names every location on
    its command line needs no environment. The folders under a configured data root are guarded
    whether or not they exist yet. A value that is set but wrong (a relative path, a folder that
    does not exist) raises, because that is a mistake to report rather than an absence, and a
    guard that quietly switched itself off would be worse than none.

    ``root`` is the data root of a run that names its own (the runner's ``--data-root``, which it
    hands its generators as ``SOMA_DATA_ROOT``). Its folders are guarded as the configured data
    root's are, and so are those of ``SOMA_DATA_ROOT`` when that names another folder: the runner
    writes before any generator's own guard runs, and a generator handed the run's root guards
    that root only. With ``root`` a variable is refused only when it is malformed (relative), not
    when its folder is missing: the runner records the locations its generators will see without
    requiring them to exist, and guarding a folder that does not exist switches nothing off.
    Without ``root`` the behaviour is the generators' own, unchanged.

    ``allow_run_records`` leaves out the ``_runs`` evidence folders, where run records belong.
    """
    for name in (DATA_ROOT_ENV, SOURCE_ROOT_ENV, BODY_MODEL_DIR_ENV):
        _env_path(name)
    roots = _guarded_roots(root)
    override = body_model_override()
    explicit_sources = _env_path(SOURCE_ROOT_ENV)
    read_only: list[tuple[str, Path]] = []
    containers: list[tuple[str, Path]] = []
    if explicit_sources is not None:
        # Only the folders the generators read: another folder under a shared source root (a
        # data root placed beside the sources, say) is not theirs to protect.
        sources = source_root() if root is None else explicit_sources
        containers.append(("the source root", sources))
        read_only += [(f"the source root's {folder} folder", sources / folder)
                      for folder in SOURCE_FOLDERS.values()]
    for index, (base, named) in enumerate(roots):
        # Guarded whatever SOMA_SOURCE_ROOT says: the extracted sources and their archives under
        # a data root are never written (retention rule 1.4 forbids deleting them) even when a run
        # reads a copy elsewhere. The first root is the one the generators read their sources
        # under by default.
        read_only.append(("the source root" if explicit_sources is None and index == 0
                          else f"{named}/{EXTRACTED}", base / EXTRACTED))
        read_only.append((f"{named}/{RAW_ARCHIVES}", base / RAW_ARCHIVES))
        read_only += [(f"the evidence folder {POC_DEMO}/{name}", base / POC_DEMO / name)
                      for name in EVIDENCE_FOLDERS
                      if not (allow_run_records and name == RUN_RECORDS)]
        # the default folder stays guarded when SOMA_BODY_MODEL_DIR moves the models elsewhere:
        # it still holds the data root's models
        read_only.append((f"the body model directory {named}/{BODY_MODELS}",
                          base / BODY_MODELS))
        containers += [("the data root" if named == DATA_ROOT_ENV else named, base),
                       (f"the runs folder {named}/runs", base / "runs"),
                       (f"the lineage container {POC_DEMO}", base / POC_DEMO)]
    if override is not None:
        read_only.append((f"the body model directory {BODY_MODEL_DIR_ENV}", override))
    return read_only, containers


def _folded(part: str) -> str:
    """A path component as the synced-folder markers compare it: NFC (macOS hands out decomposed
    Korean names), case-folded. The markers are folder names a sync client creates, matched
    case-insensitively on every platform: macOS volumes are case-insensitive by default, and a
    folder named ``dropbox`` on Linux is no less likely to be synchronised. This is a name match,
    never a comparison of two paths (those go through ``os.path.normcase``, which is the identity
    on POSIX)."""
    return unicodedata.normalize("NFC", part).casefold()


def synced_location(path: str | os.PathLike[str]) -> str | None:
    """The part of ``path`` that marks a cloud-synchronised folder, or None. Pure: nothing on disk
    is read, so a :class:`pathlib.PurePosixPath` or :class:`pathlib.PureWindowsPath` can be asked
    about on any platform (the rules follow the path's flavour, not the running system's).

    On every flavour: a component named like a sync client's folder (:data:`_SYNCED_FOLDER_EXACT`,
    :data:`_SYNCED_FOLDER_PREFIXES`). On a POSIX path also: ``Library/CloudStorage`` and
    ``Library/Mobile Documents`` (macOS) and a gvfs ``google-drive:`` mount (Linux). The anchor
    (a drive, ``/``) is never a marker.
    """
    pure = path if isinstance(path, PurePath) else PurePath(os.fspath(path))
    parts = pure.parts[1:] if pure.anchor else pure.parts
    folded = [_folded(part) for part in parts]
    posix = isinstance(pure, PurePosixPath)
    for index, (part, name) in enumerate(zip(parts, folded)):
        if name in _SYNCED_FOLDER_EXACT or name.startswith(_SYNCED_FOLDER_PREFIXES):
            return part
        if not posix:
            continue
        if name.startswith(_SYNCED_FOLDER_PREFIXES_POSIX):
            return part
        if index and (folded[index - 1], name) in _SYNCED_FOLDER_PAIRS_POSIX:
            return f"{parts[index - 1]}/{part}"
    return None


def shared_drive_names() -> tuple[str, ...]:
    """``SOMA_SHARED_DRIVE_NAMES``: the folder names of team shared drives, comma-separated, as
    :func:`_folded` compares them. Empty when unset or blank, so only the sync-client folders of
    :func:`synced_location` draw a warning then. Read when called, never at import."""
    value = os.environ.get(SHARED_DRIVE_NAMES_ENV) or ""
    return tuple(dict.fromkeys(_folded(name.strip()) for name in value.split(",") if name.strip()))


def on_shared_drive(path: str | os.PathLike[str]) -> bool:
    """Whether a component of ``path`` is a team shared drive named in ``SOMA_SHARED_DRIVE_NAMES``
    (:func:`shared_drive_names`), compared like the synced-folder markers. Nothing on disk is read,
    like :func:`synced_location`; the environment is."""
    names = shared_drive_names()
    if not names:
        return False
    pure = path if isinstance(path, PurePath) else PurePath(os.fspath(path))
    parts = pure.parts[1:] if pure.anchor else pure.parts
    return any(_folded(part) in names for part in parts)


#: The paths :func:`warn_if_synced` has warned about in this process, so each is named once.
_WARNED_SYNCED: set[str] = set()


def warn_if_synced(path: str | os.PathLike[str]) -> Path:
    """Warn about a path inside a cloud-synchronised folder or on a team shared drive; return it.

    Writing there is allowed (owner decision, 2026-09-30) but not recommended, so the path is
    never refused: one warning line goes to stderr, once per distinct path in a process. A sync
    client can lock, delay or partially upload a file while it is written, and a bundle written
    into a synchronised or shared folder may be shared beyond this PC, which needs separate
    approval. The path is judged as written (made absolute) and as resolved, so a link into a
    synchronised folder is noticed too (:func:`synced_location`, :func:`on_shared_drive`). A
    plain line rather than :mod:`warnings`, which a caller may have turned into errors.

    It needs no environment, so the runner applies it (through :func:`check_output_dir` and
    :func:`check_record_dir`) to the locations it writes itself before it opens a run record.
    """
    target = Path(path)
    where = None
    for spelled in (target.absolute(), target.resolve()):
        marker = synced_location(spelled)
        if marker is not None:
            where = f"inside the cloud-synchronised folder {marker!r}"
            break
        if on_shared_drive(spelled):
            where = f"on a team shared drive ({SHARED_DRIVE_NAMES_ENV})"
            break
    if where is not None:
        key = os.path.normcase(str(target.absolute()))
        if key not in _WARNED_SYNCED:
            _WARNED_SYNCED.add(key)
            print(f"warning: {target} is {where}; writing there is allowed but not recommended: "
                  "sync clients can lock, delay or partially upload files while they are written, "
                  "and bundles written there may be shared beyond this PC (sharing needs separate "
                  f"approval). A local disk under {DATA_ROOT_ENV} is recommended",
                  file=sys.stderr, flush=True)
    return target


def check_output_dir(
    path: str | os.PathLike[str], inputs: Iterable[str | os.PathLike[str] | None] = (),
    *, root: str | os.PathLike[str] | None = None,
) -> Path:
    """Refuse an output directory a generator must never write into; return it unchanged.

    Refused:

    * anything inside a folder the run reads: each of ``inputs`` (the source, corpus or workbook
      folders the generator resolved, from its flags or its defaults) and, when the environment
      configures them, the per-source folders under ``SOMA_SOURCE_ROOT``, and under
      ``SOMA_DATA_ROOT`` its ``extracted`` and ``raw_archives`` folders (whatever
      ``SOMA_SOURCE_ROOT`` says), the evidence folders ``_superseded``, ``_runs`` and
      ``_manifest_backfill`` beside the lineage directories, and the body-model directory;
    * any folder holding one of ``inputs`` (its parent or a folder further up): a generator writes
      its INDEX.json, fit record and take directories at the top of its output, so they would
      land beside the input, or over it when a take directory shares its name;
    * the containers themselves: the data root, ``<data root>/runs``,
      ``<data root>/runs/experimental_generation_poc_demo`` (which holds one directory per lineage
      and is never a bundle) and
      ``SOMA_SOURCE_ROOT``.

    Not refused but warned about (:func:`warn_if_synced`): anything inside a cloud-synchronised
    folder (Dropbox, OneDrive, Google Drive in English or Korean, Synology Drive, iCloud Drive; on
    macOS anything under ``~/Library/CloudStorage``, on Linux a gvfs Google Drive mount:
    :func:`synced_location`) or on a team shared drive named in ``SOMA_SHARED_DRIVE_NAMES``
    (:func:`on_shared_drive`).

    The environment is consulted only where it is configured, so a run that passes every location
    explicitly is guarded by those locations alone. A variable that is set but names no folder is
    an error, never a guard that silently does nothing.

    ``root`` adds a data root the run names itself, guarded like ``SOMA_DATA_ROOT``
    (:func:`_configured_locations`); the runner passes its own, because it writes into the
    bundle and corpus directories before a generator's guard runs, or without a generator.
    """
    target = warn_if_synced(path)
    resolved = target.resolve()
    read_only, containers = _configured_locations(root)
    explicit = [Path(p) for p in inputs if p is not None]
    read_only += [("a folder this run reads", folder) for folder in explicit]
    for label, folder in read_only:
        if _is_within(resolved, folder.resolve()):
            raise OutputLocationRefused(
                f"refusing to write {target}: it is inside {label} ({folder}), which is read-only"
            )
    for folder in explicit:
        if resolved in folder.resolve().parents:
            raise OutputLocationRefused(
                f"refusing to write {target}: it is the folder holding {folder}, a folder this run "
                "reads; the bundle's files would be written beside it. Put the output in a folder "
                "of its own"
            )
    for label, folder in containers:
        if resolved == folder.resolve():
            raise OutputLocationRefused(
                f"refusing to write {target}: it is {label} itself ({folder}); a bundle goes "
                "in its own lineage directory beneath it"
            )
    return target


def evidence_folder(path: str | os.PathLike[str]) -> str | None:
    """The evidence folder ``path`` is or lies in: one named like :data:`EVIDENCE_FOLDERS` directly
    under a ``runs/experimental_generation_poc_demo``, under any root, as written (made absolute)
    or resolved; its name as spelled there, or None. Nothing is read but the links resolved.

    The guards above know the evidence folders of the data roots the environment (or a run) names;
    this one needs none, so a command that is handed a folder inside another data root's
    ``_superseded`` still sees what it is."""
    container = [os.path.normcase(part) for part in PurePath(POC_DEMO).parts]
    evidence = {os.path.normcase(name) for name in EVIDENCE_FOLDERS}
    for candidate in (Path(path).absolute(), Path(path).resolve()):
        parts = list(candidate.parts)
        folded = [os.path.normcase(part) for part in parts]
        for start in range(len(folded) - len(container)):
            below = start + len(container)
            if folded[start:below] == container and folded[below] in evidence:
                return parts[below]
    return None


def check_bundle_dir(
    path: str | os.PathLike[str], *, root: str | os.PathLike[str] | None = None,
    writes: bool = True, action: str = "write",
) -> Path:
    """Refuse a bundle directory that ``validate`` (writing its report or ledger into it),
    ``readme`` or ``register`` must not take; return it.

    Refused wherever the data root is: a directory in an evidence folder (:func:`evidence_folder`).
    ``_superseded`` keeps the small files of generations that were replaced -- their INDEX.json,
    README.md, validation report -- and ``_runs`` and ``_manifest_backfill`` keep records (retention
    rules 1.3(b) and 1.4): a README rendered there overwrites the kept one, and a registration files
    the evidence as live data. Under the data roots the environment and ``root`` name, and
    ``SOMA_SOURCE_ROOT``: a directory inside ``extracted``, ``raw_archives``, the source folders or
    the body-model directories (:func:`check_output_dir`'s read-only folders). With ``writes`` (the
    command writes into the bundle) everything else :func:`check_output_dir` refuses as well: the
    containers themselves (a cloud-synchronised folder or a team shared drive draws only a
    warning, :func:`warn_if_synced`). ``action`` is
    the verb a refusal names ("register" for a command that only reads the bundle).
    """
    target = Path(path)
    found = evidence_folder(target)
    if found is not None:
        raise OutputLocationRefused(
            f"refusing to {action} {target}: it is inside the evidence folder {POC_DEMO}/{found}, "
            "which keeps the files of earlier generations and the run records (retention rules "
            "1.3(b), 1.4); nothing there is rewritten or registered. Work on the lineage directory "
            "the evidence is for, or on a scratch copy"
        )
    if writes:
        return check_output_dir(target, root=root)
    resolved = target.resolve()
    read_only, _ = _configured_locations(root)
    for label, folder in read_only:
        if _is_within(resolved, folder.resolve()):
            raise OutputLocationRefused(
                f"refusing to {action} {target}: it is inside {label} ({folder}), which is "
                "read-only and holds no bundle"
            )
    return target


def check_record_dir(
    path: str | os.PathLike[str], *, root: str | os.PathLike[str] | None = None
) -> Path:
    """Refuse a folder for run records or a catalog that a run must not write into; return it.

    Narrower than :func:`check_output_dir`, whose refusals are for a bundle. A cloud-synchronised
    folder or a team shared drive draws a warning (:func:`warn_if_synced`). Refused: anything inside
    the read-only folders -- ``extracted``, ``raw_archives``, the source folders under
    ``SOMA_SOURCE_ROOT``, the body-model directories -- and the evidence folders ``_superseded``
    and ``_manifest_backfill``; the containers themselves (the data root, ``runs``,
    ``runs/experimental_generation_poc_demo``, ``SOMA_SOURCE_ROOT``), whose top level would
    fill with ``run_<identity>`` folders or catalog files; and anything inside a lineage
    directory ``runs/experimental_generation_poc_demo/<lineage>``, which holds a bundle or a
    corpus. The ``_runs`` evidence folder beside the lineage directories is where run records go
    (``<lineage container>/_runs`` is the runner's default); the catalog is
    ``<data root>/experimental/catalog``.
    """
    target = warn_if_synced(path)
    resolved = target.resolve()
    read_only, containers = _configured_locations(root, allow_run_records=True)
    for label, folder in read_only:
        if _is_within(resolved, folder.resolve()):
            raise OutputLocationRefused(
                f"refusing to write {target}: it is inside {label} ({folder}), which is read-only"
            )
    for label, folder in containers:
        if resolved == folder.resolve():
            raise OutputLocationRefused(
                f"refusing to write {target}: it is {label} itself ({folder}); run records go in "
                f"{POC_DEMO}/{RUN_RECORDS} and the catalog in <data root>/experimental/catalog"
            )
    for base, named in _guarded_roots(root):
        lineages = (base / POC_DEMO).resolve()
        if lineages not in resolved.parents:
            continue
        lineage = resolved.relative_to(lineages).parts[0]
        if os.path.normcase(lineage) != os.path.normcase(RUN_RECORDS):
            raise OutputLocationRefused(
                f"refusing to write {target}: it is inside the lineage directory "
                f"{named}/{POC_DEMO}/{lineage}, which holds a bundle or a corpus; run records go "
                f"in {POC_DEMO}/{RUN_RECORDS} and the catalog in <data root>/experimental/catalog"
            )
    return target


def check_record_file(
    path: str | os.PathLike[str], *, root: str | os.PathLike[str] | None = None
) -> Path:
    """Refuse a file a record (a validation report or ledger written outside its bundle) must
    not be written to; return it.

    Its folder gets :func:`check_record_dir`, and the data plane's own files
    (:data:`PROTECTED_DATA_FILES` under a guarded data root: its ``README.md`` and, where the
    data root keeps them, the optional bookkeeping files ``MASTER.md`` and
    ``state/local_archive_inventory.json``) are never overwritten (retention rule 1.4).
    """
    target = Path(path)
    check_record_dir(target.parent, root=root)
    resolved = target.resolve()
    for base, named in _guarded_roots(root):
        for relative in PROTECTED_DATA_FILES:
            if resolved == (base / relative).resolve():
                raise OutputLocationRefused(
                    f"refusing to write {target}: it is {named}/{relative}, which is never "
                    "overwritten (retention rule 1.4)"
                )
    return target


def check_readme_root(
    path: str | os.PathLike[str], *, root: str | os.PathLike[str] | None = None
) -> Path:
    """Refuse a folder a top-level ``README.md`` must not be written into; return it.

    ``soma-synth readme --top-level-root`` and ``soma-synth pipeline --top-level-root`` write
    ``<folder>/README.md`` across the bundles given. A cloud-synchronised folder or a team shared
    drive draws a warning (:func:`warn_if_synced`). Refused: a data root itself (the run's and
    ``SOMA_DATA_ROOT``: its ``README.md`` is the data plane's own and is never overwritten,
    retention rule 1.4) and ``SOMA_SOURCE_ROOT``; and anything inside the read-only folders
    (``extracted``, ``raw_archives``, the source folders, the body-model directories) or the
    evidence folders (``_superseded``, ``_runs``, ``_manifest_backfill``). The lineage container
    ``runs/experimental_generation_poc_demo`` and any folder of one's own stay allowed.
    """
    target = warn_if_synced(path)
    resolved = target.resolve()
    read_only, _ = _configured_locations(root)
    for label, folder in read_only:
        if _is_within(resolved, folder.resolve()):
            raise OutputLocationRefused(
                f"refusing to write {target / 'README.md'}: {target} is inside {label} "
                f"({folder}), which is read-only"
            )
    roots = [(base, "the data root" if named == DATA_ROOT_ENV else named)
             for base, named in _guarded_roots(root)]
    explicit_sources = _env_path(SOURCE_ROOT_ENV)
    if explicit_sources is not None:
        roots.append((explicit_sources, SOURCE_ROOT_ENV))
    for base, named in roots:
        if resolved == base.resolve():
            raise OutputLocationRefused(
                f"refusing to write {target / 'README.md'}: {target} is {named} itself, whose "
                "README.md is never overwritten (retention rule 1.4); write the top-level README "
                f"into the lineage container ({POC_DEMO}) or a folder of its own"
            )
    return target
