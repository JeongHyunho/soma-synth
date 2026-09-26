"""The source files a run's corpus and bundle stages could read, with their full SHA-256.

A bundle built from a retarget corpus (``hknu``, ``gaitex``, ``addbiomechanics``) names the corpus
file as its source, and the corpora record their sources weakly: HKNU by subject and trial name,
GAITEX and AddBiomechanics by the SHA-256 of a file's first MiB. None of that may change (it feeds
the npz members, the manifests and ``pair_id``). So the run record closes the gap instead: the
runner adds a ``source_files`` block to the run's ``source_manifest.json`` (:func:`trace`) holding,
for every source file the run's stages could read, its logical id ``extracted/<folder>/<path>``,
its size and the SHA-256 of the whole file.

What a stage could read is its source folder under the run's source root (``SOMA_SOURCE_ROOT``,
else ``<data root>/extracted``), narrowed to the selected subjects when the run's arguments select
some (:data:`SCOPES`):

* the corpus stage, only when this run builds the corpus (a reused corpus was traced by the run
  that built it; its fingerprint is in this record): ``hknu`` and ``gaitex`` by ``--subjects``,
  ``addbiomechanics`` by ``--only <study>/<subject>`` and ``--per-study N`` (the generator's pick:
  the first N non-empty ``.b3d`` of each study in its order, and still the first one when N is 0
  or negative, since it takes a file before it compares the count with N);
* the bundle stage: ``hknu`` reads its workbook from the HKNU folder and ``gaitex`` the marker and
  IMU files of its takes, both narrowed by the bundle's ``--subjects``; the ``addbiomechanics``
  bundle reads the corpus only.

A file no subject owns (a workbook, a README, a provenance folder) is kept whatever the selection.
Every file is hashed once however many stages could read it, streamed in 1 MiB chunks
(``source_files.sha256_of``), in the order ``source_files.iter_source_files`` lists them, which is
the order ``soma-synth hash-sources`` writes.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from soma_synth.pipeline import paths as paths_mod
from soma_synth.pipeline import source_files as files_mod

SCHEMA = "source_files_v1"
ALGORITHM = "sha256 of the whole file, streamed in 1 MiB chunks"
AGGREGATE = "sha256 over the sorted lines '<relative_path>\\t<sha256>', joined by '\\n'"


@dataclass(frozen=True)
class StageScope:
    """What one stage of a source can read under its source folder."""

    #: the flag that selects subjects (None: the stage reads no source file at all)
    subject_flag: str | None
    #: how the flag takes its values: "star" (``nargs="*"``, the last one wins) or "append"
    kind: str = "star"
    #: addbiomechanics' corpus: the per-study cap flag
    cap_flag: str | None = None


#: source name -> {stage: scope}. The three sources with a corpus stage; the flags are the
#: generators' own (``--subjects`` of the four hknu/gaitex entrypoints, ``--only``/``--per-study``
#: of ``generate_addbio_smpl24.py``).
SCOPES: dict[str, dict[str, StageScope]] = {
    "hknu": {"corpus": StageScope("--subjects"), "bundle": StageScope("--subjects")},
    "gaitex": {"corpus": StageScope("--subjects"), "bundle": StageScope("--subjects")},
    "addbiomechanics": {"corpus": StageScope("--only", kind="append", cap_flag="--per-study"),
                        "bundle": StageScope(None)},
}


# ---------------------------------------------------------------- reading the selection
def _declared_flag(name: str, declared: Sequence[str]) -> str | None:
    """The flag argparse takes ``name`` for among ``declared`` (itself, or the one flag it
    abbreviates); None when it names none or several."""
    if name in declared:
        return name
    matches = [flag for flag in declared if flag.startswith(name)]
    return matches[0] if len(matches) == 1 else None


#: argparse's own test for an argument that is a negative number, not an option (a parser declares
#: no option that looks like one, so such a token is a value).
_NEGATIVE_NUMBER = re.compile(r"^-\d+$|^-\d*\.\d+$")


def flag_values(args: Sequence[str], flag: str, declared: Sequence[str], kind: str = "star"
                ) -> list[str] | None:
    """The values ``args`` give ``flag`` as the entrypoint's argparse would read them, or None
    when the flag is not given. ``--flag=v`` and argparse abbreviations count; with ``kind``
    "star" a later occurrence replaces an earlier one, with "append" they add up, and "single"
    (one value, ``--per-study -1`` included: argparse reads a negative number as a value) keeps
    the last."""
    declared = tuple(declared) if flag in declared else (*declared, flag)
    values: list[str] | None = None
    index = 0
    tokens = [str(arg) for arg in args]
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token.startswith("--") or len(token) <= 2:
            continue
        name, has_value, inline = token.partition("=")
        if _declared_flag(name, declared) != flag:
            continue
        given: list[str] = []
        if has_value:
            given = [inline]
        elif kind == "append":
            if index < len(tokens):
                given = [tokens[index]]
                index += 1
        elif kind == "single":
            if index < len(tokens) and (not tokens[index].startswith("-")
                                        or _NEGATIVE_NUMBER.match(tokens[index])):
                given = [tokens[index]]
                index += 1
        else:
            while index < len(tokens) and not tokens[index].startswith("-"):
                given.append(tokens[index])
                index += 1
        values = given if (kind in ("star", "single") or values is None) else values + given
    return values


# ---------------------------------------------------------------- who owns a file
def _hknu_subjects(folder: Path) -> set[str]:
    processed = folder / "Dataset_Processed"
    return ({p.name for p in processed.iterdir() if p.is_dir()} if processed.is_dir() else set())


def _gaitex_subjects(folder: Path) -> set[str]:
    from soma_synth.gaitex_retarget.gaitex_frames import discover_subjects

    return set(discover_subjects(folder))


def _owner(source: str, parts: tuple[str, ...], subjects: set[str]) -> object:
    """The subject a file belongs to (a name, or ``(study, subject)`` for addbiomechanics), or
    None for a file no subject owns."""
    folders = parts[:-1]
    if source == "hknu":
        return next((part for part in folders if part in subjects), None)
    if source == "gaitex":
        return folders[0] if folders and folders[0] in subjects else None
    if source == "addbiomechanics":
        # <split>/<variant>/<study>/.../<subject>/<file>: the generator's --only names the study
        # folder and the folder holding the .b3d
        return (parts[2], parts[-2]) if len(parts) >= 5 else None
    raise ValueError(source)


def _addbio_study(parts: tuple[str, ...]) -> tuple[str, str, str] | None:
    return (parts[0], parts[1], parts[2]) if len(parts) >= 5 else None


def _addbio_capped(files: list[tuple[Path, tuple[str, ...]]], cap: int,
                   only: set[tuple[str, str]] | None) -> set[Path]:
    """The ``.b3d`` files ``generate_addbio_smpl24.payload_files`` picks under ``--per-study cap``,
    by the generator's own rule: per (split, variant, study), walking its non-empty ones in sorted
    order (after ``--only``), take the file, then stop once ``taken >= cap``. So a cap of 0 or less
    still takes the first file of each study, as the generator does. Every file of the list is
    under one source folder, listed in name order."""
    taken: dict[tuple[str, str, str], int] = {}
    stopped: set[tuple[str, str, str]] = set()
    picked: set[Path] = set()
    # Path order, as the generator's sorted() walks: case-insensitive on Windows, not elsewhere
    for path, parts in sorted(files, key=lambda item: item[0]):
        study = _addbio_study(parts)
        if study is None or path.suffix != ".b3d" or path.stat().st_size == 0:
            continue
        if only is not None and (parts[2], parts[-2]) not in only:
            continue
        if study in stopped:
            continue
        picked.add(path)
        taken[study] = taken.get(study, 0) + 1
        if taken[study] >= cap:
            stopped.add(study)
    return picked


# ---------------------------------------------------------------- the trace
def stages_of(source: str, *, corpus_built: bool) -> list[str]:
    """The stages of this run that could read ``source``'s folder."""
    scopes = SCOPES.get(source, {})
    stages = []
    if corpus_built and "corpus" in scopes:
        stages.append("corpus")
    if scopes.get("bundle") is not None and scopes["bundle"].subject_flag is not None:
        stages.append("bundle")
    return stages


def trace(source: str, source_root: Path | str, *, corpus_built: bool,
          corpus_args: Sequence[str] = (), bundle_args: Sequence[str] = (),
          corpus_declared: Sequence[str] = (), bundle_declared: Sequence[str] = (),
          log: Callable[[str], None] = print) -> dict[str, object]:
    """The ``source_files`` block of a run's ``source_manifest.json`` (see the module docstring).

    ``corpus_declared`` / ``bundle_declared`` are the entrypoints' declared flags (the registry's
    required and optional args), against which an abbreviation is resolved as argparse would.
    """
    started = time.monotonic()
    folder_name = paths_mod.source_folder(source)
    folder = Path(source_root) / folder_name
    logical_folder = f"{paths_mod.EXTRACTED}/{folder_name}"
    scopes = SCOPES.get(source)
    block: dict[str, object] = {
        "schema": SCHEMA,
        "note": ("Every source file this run's corpus and bundle stages could read, with the "
                 "SHA-256 of the whole file: the corpus records its sources by name or by a "
                 "first-MiB digest only, so the full hashes live here (ADR-0041; "
                 "GENERATION_PIPELINE_STANDARD section 4.1)."),
        "source": source,
        "folder": logical_folder,
        "algorithm": ALGORITHM,
        "aggregate_algorithm": AGGREGATE,
    }
    if scopes is None:
        return {**block, "status": "not_recorded",
                "reason": f"{source} has no corpus stage; its take manifests name the source files "
                          "with their full SHA-256 (the references above)"}
    stage_names = stages_of(source, corpus_built=corpus_built)
    selections: dict[str, dict[str, object]] = {}
    for stage in ("corpus", "bundle"):
        scope = scopes.get(stage)
        args, declared = ((corpus_args, corpus_declared) if stage == "corpus"
                          else (bundle_args, bundle_declared))
        entry: dict[str, object] = {"stage": stage, "reads_source": stage in stage_names}
        if stage == "corpus":
            entry["built_in_run"] = bool(corpus_built)
        if stage in stage_names and scope is not None and scope.subject_flag is not None:
            subjects = flag_values(args, scope.subject_flag, declared, scope.kind)
            cap = flag_values(args, scope.cap_flag, declared, "single") if scope.cap_flag else None
            entry["selection"] = ({scope.subject_flag: subjects} if subjects is not None else {})
            if cap:
                entry["selection"][scope.cap_flag] = cap[-1]             # type: ignore[index]
            entry["narrowed"] = bool(subjects) or bool(cap)
        selections[stage] = entry
    block["stages"] = list(selections.values())

    if not stage_names:
        return {**block, "count": 0, "bytes": 0, "seconds": 0.0, "files": [],
                "aggregate_sha256": hashlib.sha256(b"").hexdigest(),
                "unreadable": []}
    if not folder.is_dir():
        return {**block, "status": "unavailable",
                "reason": f"the source folder {logical_folder} is not under the run's source root"}

    subjects_known = (_hknu_subjects(folder) if source == "hknu"
                      else _gaitex_subjects(folder) if source == "gaitex" else set())
    listed = [(path, Path(path).relative_to(folder).parts)
              for path in files_mod.iter_source_files(folder)]
    readers: dict[Path, list[str]] = {}
    for stage in stage_names:
        entry = selections[stage]
        selection = entry.get("selection") or {}
        scope = scopes[stage]
        wanted = selection.get(scope.subject_flag) if isinstance(selection, dict) else None
        if source == "addbiomechanics" and wanted is not None:
            owners = set()
            for value in wanted:
                pieces = str(value).replace("\\", "/").split("/")
                if len(pieces) == 2 and all(pieces):
                    owners.add((pieces[0], pieces[1]))
            wanted_set: set | None = owners
        else:
            wanted_set = set(wanted) if wanted else None
        capped = None
        if scope.cap_flag and isinstance(selection, dict) and selection.get(scope.cap_flag):
            try:
                capped = _addbio_capped(listed, int(selection[scope.cap_flag]), wanted_set)
            except ValueError:
                capped = None                  # not a number: the generator refused it anyway
        for path, parts in listed:
            owner = _owner(source, parts, subjects_known)
            if capped is not None and source == "addbiomechanics" and path.suffix == ".b3d" \
                    and _addbio_study(parts) is not None:
                if path not in capped:
                    continue
            elif owner is not None and wanted_set is not None and owner not in wanted_set:
                continue
            readers.setdefault(path, []).append(stage)

    files: list[dict[str, object]] = []
    unreadable: list[dict[str, str]] = []
    total = 0
    for path, _parts in listed:
        if path not in readers:
            continue
        relative = f"{logical_folder}/{files_mod.relative_name(path, folder)}"
        try:
            size = path.stat().st_size
            digest = files_mod.sha256_of(path)
        except OSError as error:
            unreadable.append({"relative_path": relative,
                               "error": f"{type(error).__name__}: {error}"})
            continue
        total += size
        files.append({"relative_path": relative, "sha256": digest, "bytes": size,
                      "stages": sorted(readers[path])})
    lines = "\n".join(f"{entry['relative_path']}\t{entry['sha256']}"
                      for entry in sorted(files, key=lambda e: str(e["relative_path"])))
    seconds = round(time.monotonic() - started, 3)
    log(f"    source files: {len(files)} files of {logical_folder}, {total / 1e9:.2f} GB, "
        f"hashed in {seconds:.1f} s" + (f"; {len(unreadable)} unreadable" if unreadable else ""))
    return {
        **block,
        "count": len(files),
        "bytes": total,
        "seconds": seconds,
        "aggregate_sha256": hashlib.sha256(lines.encode("utf-8")).hexdigest(),
        "files": files,
        "unreadable": unreadable,
    }
