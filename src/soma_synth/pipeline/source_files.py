"""The source files as files: list them, hash them, and stage them onto a PC from a list.

The five source folders (:data:`paths.SOURCE_FOLDERS`: ``amass``, ``prism``, ``gaitex``,
``hknu_fullbody``, ``addbiomechanics``) arrive on a teammate's PC by copying from a folder that
holds the five source folders (for example a read-only shared copy). Two commands carry that, and
the run record's source tracing (``source_trace.py``) lists files the same way:

``soma-synth hash-sources`` (:func:`hash_sources`)
    Writes a ``SHA256SUMS`` list of every file of the named source folders, in GNU coreutils'
    format (``<sha256>  <folder>/<relative posix path>``, one line per file, sorted by name), so
    ``sha256sum -c`` can check it too. It only reads the sources, and refuses a list written into
    the source root, into ``extracted/`` or ``raw_archives/`` of the data root, over one of the data
    root's own files, or into a synchronised folder.
``soma-synth stage-sources`` (:func:`stage_sources`)
    Copies every listed file that is missing under the source root from a folder that holds the
    source folders (only read; it may be a synchronised or shared folder): each copy goes to a
    temporary name in its destination folder, is checked against the list's SHA-256 and only then
    renamed into place.
    A file already there with the listed hash is skipped. A file already there with another hash is
    a conflict: the command reports every conflict, copies nothing and exits 3. Nothing is ever
    deleted or overwritten. ``--verify-only`` copies nothing and reports what is missing or differs.

Operating-system metadata files (:data:`OS_METADATA_NAMES`, ``._*``) are not source files and are
neither listed nor traced: a Finder or Explorer visit must not change a folder's list.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from soma_synth.pipeline import paths as paths_mod

#: Files an operating system or a sync client leaves in any folder; never a source file.
OS_METADATA_NAMES = frozenset({".ds_store", "thumbs.db", "desktop.ini", ".localized"})
#: macOS AppleDouble companions (``._<name>``) on non-Apple volumes.
_APPLEDOUBLE_PREFIX = "._"
#: The chunk every hash streams in.
CHUNK = 1 << 20
#: The temporary name a staged copy has in its destination folder until it is verified:
#: ``.<name>.<random>.staging`` (:func:`is_staging_leftover`).
STAGING_SUFFIX = ".staging"
_STAGING_NAME = re.compile(r"^\..+\.[A-Za-z0-9_]+" + re.escape(STAGING_SUFFIX) + r"$")

_SUMS_LINE = re.compile(r"^(\\?)([0-9a-fA-F]{64}) [ *](.+)$")

#: Exit codes of the two commands (``cli.py``): 1 is an incomplete result (something missing or
#: different), 2 a location or list that cannot be resolved, 3 a refusal.
EXIT_INCOMPLETE = 1
EXIT_CONFIG = 2
EXIT_REFUSED = 3


class SourceFilesError(paths_mod.PathConfigError):
    """A source folder, a list or an argument the command cannot use."""


def is_metadata(name: str) -> bool:
    """Whether a file name is an operating system's or a sync client's, not a source file."""
    return name.casefold() in OS_METADATA_NAMES or name.startswith(_APPLEDOUBLE_PREFIX)


def is_staging_leftover(name: str) -> bool:
    """Whether a file name is a copy ``stage-sources`` had not yet verified and renamed
    (``.<name>.<random>.staging``): what an interrupted staging leaves. Never a source file."""
    return bool(_STAGING_NAME.match(name))


def iter_source_files(folder: Path) -> Iterator[Path]:
    """Every file under ``folder``, in name order (folders walked in sorted order, links to folders
    not followed), leaving out :func:`is_metadata` names and staging leftovers
    (:func:`is_staging_leftover`). The one enumeration the list, the staging and the run record's
    tracing share."""
    folder = Path(folder)
    for current, folders, files in os.walk(folder, followlinks=False):
        folders.sort()
        for name in sorted(files):
            if not is_metadata(name) and not is_staging_leftover(name):
                path = Path(current) / name
                if path.is_file():
                    yield path


def staging_leftovers(folder: Path) -> list[Path]:
    """The staging leftovers under ``folder`` (:func:`is_staging_leftover`), in name order. Only
    reported: this tool never deletes them."""
    found: list[Path] = []
    for current, folders, files in os.walk(Path(folder), followlinks=False):
        folders.sort()
        found += [Path(current) / name for name in sorted(files) if is_staging_leftover(name)]
    return found


def relative_name(path: Path, folder: Path) -> str:
    """``path`` relative to ``folder`` with forward slashes (a logical id's tail)."""
    return Path(path).relative_to(folder).as_posix()


def sha256_of(path: Path | str) -> str:
    """The SHA-256 of a whole file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------- the SHA256SUMS format
def format_sums_line(digest: str, name: str) -> str:
    """One line of GNU coreutils' ``sha256sum`` output (text mode), without its newline. A name
    holding a backslash, a newline or a carriage return is escaped as coreutils escapes it: the line
    starts with a backslash and those characters are written ``\\\\``, ``\\n``, ``\\r``."""
    if any(char in name for char in "\\\n\r"):
        escaped = name.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
        return f"\\{digest}  {escaped}"
    return f"{digest}  {name}"


def _unescape(name: str) -> str:
    out, index = [], 0
    while index < len(name):
        char = name[index]
        if char == "\\" and index + 1 < len(name):
            nxt = name[index + 1]
            out.append({"\\": "\\", "n": "\n", "r": "\r"}.get(nxt, "\\" + nxt))
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def parse_sums_line(line: str) -> tuple[str, str]:
    """(sha256 in lower case, name) of one ``sha256sum`` line (text ``"  "`` or binary ``" *"``
    mode, escaped or not). SourceFilesError when it is not one."""
    match = _SUMS_LINE.match(line.rstrip("\r\n"))
    if not match:
        raise SourceFilesError(f"not a sha256sum line: {line.rstrip()!r}")
    escaped, digest, name = match.groups()
    return digest.lower(), (_unescape(name) if escaped else name)


def read_sums(path: Path | str) -> list[tuple[str, str]]:
    """Every (sha256, name) of a ``SHA256SUMS`` file; blank lines are skipped, a name listed twice
    (with the same hash or not) is an error."""
    entries: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                digest, name = parse_sums_line(line)
            except SourceFilesError as error:
                raise SourceFilesError(f"{path}:{number}: {error}") from None
            if name in seen:
                raise SourceFilesError(f"{path}:{number}: {name!r} is listed twice")
            seen[name] = digest
            entries.append((digest, name))
    return entries


def split_listed_name(name: str) -> tuple[str, PurePosixPath]:
    """(source folder, path inside it) of a listed name ``<folder>/<relative path>``, refusing a
    name that is absolute, climbs out (``..``), names a drive, is empty, or names a folder that is
    none of the five source folders."""
    if "\\" in name or ":" in name.split("/", 1)[0]:
        raise SourceFilesError(f"{name!r}: a listed name is <folder>/<relative posix path>")
    pure = PurePosixPath(name)
    parts = pure.parts
    if (pure.is_absolute() or len(parts) < 2 or any(part in ("", ".", "..") for part in parts)
            or name != pure.as_posix()):
        raise SourceFilesError(f"{name!r}: a listed name is <folder>/<relative posix path> inside "
                               "one source folder")
    folder = parts[0]
    if folder not in paths_mod.SOURCE_FOLDERS.values():
        raise SourceFilesError(f"{name!r}: {folder!r} is not a source folder "
                               f"({', '.join(sorted(paths_mod.SOURCE_FOLDERS.values()))})")
    return folder, PurePosixPath(*parts[1:])


def resolve_folders(names: Iterable[str] | None) -> list[str]:
    """The source folders a ``--sources a,b`` names (a folder name or a source name: ``hknu`` is
    ``hknu_fullbody``), in the order given; all five when ``names`` is None or empty."""
    folders = list(paths_mod.SOURCE_FOLDERS.values())
    if not names:
        return sorted(folders)
    chosen: list[str] = []
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        folder = paths_mod.SOURCE_FOLDERS.get(name, name)
        if folder not in folders:
            known = sorted(set(folders) | set(paths_mod.SOURCE_FOLDERS))
            raise SourceFilesError(f"unknown source {name!r}; known: {', '.join(known)}")
        if folder not in chosen:
            chosen.append(folder)
    if not chosen:
        raise SourceFilesError("--sources names no source")
    return chosen


def parse_sources_option(text: str | None) -> list[str]:
    """``--sources a,b`` (commas or spaces) as source folders (:func:`resolve_folders`)."""
    return resolve_folders(re.split(r"[,\s]+", text) if text else None)


# ---------------------------------------------------------------- location guards
def _inside(path: Path, folder: Path) -> bool:
    resolved, base = Path(path).resolve(), Path(folder).resolve()
    return resolved == base or base in resolved.parents


#: The file every bundle and corpus holds at its top: a folder holding one is a bundle's (or a
#: corpus') own, and nothing but its generator writes there.
BUNDLE_INDEX = "INDEX.json"


def is_sums_list(path: Path | str) -> bool:
    """Whether ``path`` reads as a ``SHA256SUMS`` list :func:`hash_sources` writes: at least one
    line, every line a ``sha256sum`` line naming ``<source folder>/<relative path>``, the names
    unique and sorted, UTF-8, ending with a newline. Reading stops at the first line that is not."""
    names: list[str] = []
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            for line in handle:
                if not line.endswith("\n"):
                    return False
                _, name = parse_sums_line(line)
                split_listed_name(name)
                if names and name <= names[-1]:
                    return False
                names.append(name)
    except (OSError, UnicodeDecodeError, SourceFilesError):
        return False
    return bool(names)


def _lineage_container_parts() -> tuple[str, ...]:
    return tuple(part.casefold() for part in PurePosixPath(paths_mod.POC_DEMO).parts)


def check_sums_output(out: Path | str, source_root: Path | str, *, force: bool = False) -> Path:
    """Refuse a ``SHA256SUMS`` destination the list must not be written to; return it.

    Refused: a synchronised folder or a team shared drive (``paths.check_not_synced``); anything
    inside the source root the list describes; anything inside ``extracted/`` or ``raw_archives/``
    of ``SOMA_DATA_ROOT`` when it is set (retention rule 1.4: they are never written); anything
    inside an evidence folder (``_superseded``, ``_runs``, ``_manifest_backfill``, wherever it is),
    inside the lineage container ``runs/experimental_generation_poc_demo`` (of any data root: the
    path is matched by its folder names) or inside a bundle or corpus (a folder above it holds
    ``INDEX.json``); the data root's own files (:data:`paths.PROTECTED_DATA_FILES`: its
    ``README.md`` and the optional bookkeeping files a data root may keep); an existing folder;
    and, unless ``force``, an existing file that is not a list this command wrote
    (:func:`is_sums_list`).
    """
    target = paths_mod.check_not_synced(out)
    if target.is_dir():
        raise paths_mod.OutputLocationRefused(f"refusing to write {target}: it is a folder; name "
                                              "the list file")
    guarded = [(Path(source_root), "the source root it lists")]
    configured = paths_mod._env_path(paths_mod.DATA_ROOT_ENV)
    if configured is not None:
        guarded += [(configured / paths_mod.EXTRACTED, f"{paths_mod.DATA_ROOT_ENV}/extracted"),
                    (configured / paths_mod.RAW_ARCHIVES, f"{paths_mod.DATA_ROOT_ENV}/raw_archives"),
                    (configured / paths_mod.POC_DEMO,
                     f"the lineage container {paths_mod.DATA_ROOT_ENV}/{paths_mod.POC_DEMO}")]
    for folder, label in guarded:
        if _inside(target, folder):
            raise paths_mod.OutputLocationRefused(
                f"refusing to write {target}: it is inside {label} ({folder}), which the list "
                "must stay out of; write the list outside it")
    resolved = target.resolve()
    folders = resolved.parent.parts[1:]
    evidence = {name.casefold() for name in paths_mod.EVIDENCE_FOLDERS}
    for part in folders:
        if part.casefold() in evidence:
            raise paths_mod.OutputLocationRefused(
                f"refusing to write {target}: it is inside the evidence folder {part!r}, which "
                "holds records, never a list of sources")
    container = _lineage_container_parts()
    folded = [part.casefold() for part in folders]
    for index in range(len(folded) - len(container) + 1):
        if tuple(folded[index:index + len(container)]) == container:
            raise paths_mod.OutputLocationRefused(
                f"refusing to write {target}: it is inside the lineage container "
                f"{paths_mod.POC_DEMO}, which holds bundles and corpora only")
    for folder in (resolved.parent, *resolved.parent.parents):
        if (folder / BUNDLE_INDEX).is_file():
            raise paths_mod.OutputLocationRefused(
                f"refusing to write {target}: it is inside the bundle or corpus {folder} (it holds "
                f"{BUNDLE_INDEX}), which only its generator writes")
    if configured is not None:
        for relative in paths_mod.PROTECTED_DATA_FILES:
            if resolved == (configured / relative).resolve():
                raise paths_mod.OutputLocationRefused(
                    f"refusing to write {target}: it is {paths_mod.DATA_ROOT_ENV}/{relative}, "
                    "which is never overwritten (retention rule 1.4)")
    if target.exists() and not force and not is_sums_list(target):
        raise paths_mod.OutputLocationRefused(
            f"refusing to overwrite {target}: it is not a SHA256SUMS list hash-sources wrote; "
            "name another file, or pass --force to replace it")
    return target


def check_staging_root(to: Path | str, from_root: Path | str | None = None) -> Path:
    """Refuse a destination source root that staging must not write into; return it.

    The guards a source root gets: not a synchronised folder or a team shared drive; not inside an
    evidence folder (``_superseded``, ``_runs``, ``_manifest_backfill``, wherever it is); not the
    data root itself, nor inside its ``runs``, ``raw_archives`` or body-model folder (when
    ``SOMA_DATA_ROOT`` is set); and not inside, or holding, the folder it copies from.
    """
    target = paths_mod.check_not_synced(to)
    resolved = target.resolve()
    evidence = {name.casefold() for name in paths_mod.EVIDENCE_FOLDERS}
    for part in resolved.parts[1:]:
        if part.casefold() in evidence:
            raise paths_mod.OutputLocationRefused(
                f"refusing to stage into {target}: it is inside the evidence folder {part!r}, "
                "which holds records, never sources")
    configured = paths_mod._env_path(paths_mod.DATA_ROOT_ENV)
    if configured is not None:
        if resolved == configured.resolve():
            raise paths_mod.OutputLocationRefused(
                f"refusing to stage into {target}: it is {paths_mod.DATA_ROOT_ENV} itself; the "
                f"sources go in its {paths_mod.EXTRACTED}/ folder (or SOMA_SOURCE_ROOT)")
        for relative, label in (("runs", "the runs folder"),
                                (paths_mod.RAW_ARCHIVES, "raw_archives"),
                                (paths_mod.BODY_MODELS, "the body-model folder")):
            if _inside(resolved, configured / relative):
                raise paths_mod.OutputLocationRefused(
                    f"refusing to stage into {target}: it is inside {label} of "
                    f"{paths_mod.DATA_ROOT_ENV} ({configured / relative})")
    if from_root is not None and (_inside(resolved, from_root) or _inside(from_root, resolved)):
        raise paths_mod.OutputLocationRefused(
            f"refusing to stage into {target}: it overlaps the folder it copies from ({from_root})")
    return target


# ---------------------------------------------------------------- hash-sources
@dataclass
class HashResult:
    out: Path
    folders: list[str]
    files: int = 0
    bytes: int = 0
    seconds: float = 0.0
    per_folder: dict[str, tuple[int, int]] = field(default_factory=dict)


def hash_sources(source_root: Path | str, folders: Sequence[str], out: Path | str, *,
                 force: bool = False, log: Callable[[str], None] = print) -> HashResult:
    """Write the ``SHA256SUMS`` of ``folders`` under ``source_root`` to ``out`` (atomically: a
    temporary file beside it, then ``os.replace``). Reads the sources only. ``out`` gets
    :func:`check_sums_output` (``force``: replace an existing file that is not a list)."""
    root = Path(source_root)
    target = check_sums_output(out, root, force=force)
    missing = [folder for folder in folders if not (root / folder).is_dir()]
    if missing:
        raise SourceFilesError(f"no source folder {', '.join(missing)} under {root}")
    started = time.monotonic()
    result = HashResult(out=target, folders=list(folders))
    lines: list[tuple[str, str]] = []
    for folder in folders:
        base = root / folder
        count = size = 0
        for path in iter_source_files(base):
            name = f"{folder}/{relative_name(path, base)}"
            lines.append((name, format_sums_line(sha256_of(path), name)))
            count += 1
            size += path.stat().st_size
        result.per_folder[folder] = (count, size)
        result.files += count
        result.bytes += size
        log(f"  {folder}: {count} files, {size / 1e9:.2f} GB")
    body = "".join(line + "\n" for _, line in sorted(lines))
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp",
                                             dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    result.seconds = time.monotonic() - started
    return result


# ---------------------------------------------------------------- stage-sources
@dataclass
class StageResult:
    listed: int = 0
    present: list[str] = field(default_factory=list)
    copied: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)          # at the destination (verify-only)
    conflicts: list[str] = field(default_factory=list)        # destination differs from the list
    unavailable: list[str] = field(default_factory=list)      # not in --from
    corrupt: list[str] = field(default_factory=list)          # --from's copy differs from the list
    #: staging leftovers under the destination folders (relative to the destination root): what an
    #: interrupted run left; reported, never deleted, no bearing on the exit code
    leftovers: list[str] = field(default_factory=list)
    bytes_copied: int = 0
    seconds: float = 0.0
    verify_only: bool = False

    @property
    def exit_code(self) -> int:
        if self.conflicts and not self.verify_only:
            return EXIT_REFUSED
        if self.missing or self.conflicts or self.unavailable or self.corrupt:
            return EXIT_INCOMPLETE
        return 0


def _rename_without_overwriting(temporary: Path, target: Path) -> None:
    """Move ``temporary`` to ``target``, raising FileExistsError rather than replacing a file.

    Windows' rename refuses an existing target by itself. Elsewhere a hard link is made first
    (it fails when the name is taken) and the temporary name removed; a volume without hard
    links falls back to a check and a rename."""
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


def _stage_one(source: Path, target: Path, expected: str) -> tuple[str, int]:
    """Copy ``source`` to ``target`` through a verified temporary file. ("copied", bytes),
    ("corrupt", 0) when the copy's hash is not the listed one, or ("exists", 0) when ``target``
    appeared meanwhile; the temporary file never survives."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=STAGING_SUFFIX,
                                        dir=target.parent)
    temporary = Path(name)
    try:
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(descriptor, "wb") as out, open(source, "rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK), b""):
                digest.update(chunk)
                out.write(chunk)
                size += len(chunk)
            out.flush()
            os.fsync(out.fileno())
        if digest.hexdigest() != expected:
            temporary.unlink()
            return "corrupt", 0
        try:
            shutil.copystat(source, temporary)
        except OSError:
            pass                               # times are a courtesy, the content is the point
        try:
            _rename_without_overwriting(temporary, target)
        except FileExistsError:
            temporary.unlink(missing_ok=True)
            return "exists", 0
        return "copied", size
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def stage_sources(to_root: Path | str, sums: Path | str, *, from_root: Path | str | None = None,
                  folders: Sequence[str] | None = None, verify_only: bool = False,
                  log: Callable[[str], None] = print) -> StageResult:
    """Make ``to_root`` hold every file of ``sums`` (in ``folders``, default all listed).

    ``verify_only``: copy nothing; report what is missing or differs under ``to_root``.
    Otherwise every listed destination file that exists is hashed first: one with another hash
    is a conflict, and with any conflict the command copies nothing (exit 3). Then every missing
    file is copied from ``from_root`` through a temporary name, verified and renamed; one absent
    from ``from_root`` or whose copy differs from the list is reported (exit 1). Nothing is
    deleted or overwritten.
    """
    started = time.monotonic()
    root = Path(to_root)
    if not Path(sums).is_file():
        raise SourceFilesError(f"--sums {sums} is not a file")
    if not verify_only:
        if from_root is None:
            raise SourceFilesError("--from is required unless --verify-only")
        if not Path(from_root).is_dir():
            raise SourceFilesError(f"--from {from_root} does not exist or is not a folder")
        check_staging_root(root, from_root)
    entries = read_sums(sums)
    wanted = set(folders) if folders else None
    plan: list[tuple[str, str, Path, PurePosixPath, str]] = []
    for digest, name in entries:
        folder, inner = split_listed_name(name)
        if wanted is not None and folder not in wanted:
            continue
        plan.append((digest, name, root / folder / Path(*inner.parts), inner, folder))
    result = StageResult(listed=len(plan), verify_only=verify_only)
    if wanted is not None:
        absent = sorted(wanted - {folder for *_, folder in plan})
        if absent:
            log(f"  the list names no file of {', '.join(absent)}")
    # an interrupted staging leaves its unverified copies under their temporary names: they are
    # never source files (iter_source_files skips them) and this tool never deletes anything
    for folder in sorted(wanted if wanted is not None else {folder for *_, folder in plan}):
        if (root / folder).is_dir():
            for path in staging_leftovers(root / folder):
                name = relative_name(path, root)
                result.leftovers.append(name)
                log(f"  LEFTOVER  {name}: an unverified copy an interrupted stage-sources left; "
                    "not a source file and never deleted by this tool. Remove it by hand")

    to_copy = []
    for digest, name, target, inner, folder in plan:
        if target.is_file():
            if sha256_of(target) == digest:
                result.present.append(name)
            else:
                result.conflicts.append(name)
                log(f"  DIFFERS  {name}: {target} exists with another SHA-256")
        elif target.exists():
            result.conflicts.append(name)
            log(f"  DIFFERS  {name}: {target} exists and is not a file")
        elif verify_only:
            result.missing.append(name)
            log(f"  MISSING  {name}")
        else:
            to_copy.append((digest, name, target, inner, folder))
    if verify_only or result.conflicts:
        if result.conflicts and not verify_only:
            log(f"  refusing to stage: {len(result.conflicts)} destination file(s) differ from the "
                "list; nothing was copied. Nothing is overwritten: find out which copy is right")
        result.seconds = time.monotonic() - started
        return result

    source_root = Path(from_root)                      # type: ignore[arg-type]
    for index, (digest, name, target, inner, folder) in enumerate(to_copy, 1):
        source = source_root / folder / Path(*inner.parts)
        if not source.is_file():
            result.unavailable.append(name)
            log(f"  NOT IN --from  {name}")
            continue
        outcome, size = _stage_one(source, target, digest)
        if outcome == "copied":
            result.copied.append(name)
            result.bytes_copied += size
        elif outcome == "corrupt":
            result.corrupt.append(name)
            log(f"  CORRUPT  {name}: the copy from {source} does not have the listed SHA-256; "
                "nothing was written (is the drive still syncing?)")
        else:
            if sha256_of(target) == digest:
                result.present.append(name)
            else:
                result.conflicts.append(name)
                log(f"  DIFFERS  {name}: {target} appeared during staging with another SHA-256")
                break
        if index % 200 == 0:
            log(f"  {index}/{len(to_copy)} copied or checked")
    result.seconds = time.monotonic() - started
    return result
