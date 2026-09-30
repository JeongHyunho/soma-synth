"""Run the pipeline steps around the existing generators, and leave the record §10.1 requires.

The runner wraps the generator scripts rather than replacing them. Around them it adds a run
identity, the seven-artifact run directory, the provenance §10.2 asks for, and one place where
skips and exclusions are counted rather than happening silently.

**Which generation a run may do** (ADR-0041, 2026-09-25). Two bases, each recorded in the run
manifest's ``generation_decision``:

* an owner's authorization record under the parent project's ``research/decisions/``, handed in
  as ``authorized_by``: read and checked against the source by ``gates.py``, and copied into
  the run. It is read from ``gates.GOVERNANCE_ROOT``, which exists only when
  soma-synth is mounted in the parent project as ``packages/soma-synth``; otherwise the record is
  refused;
* otherwise the registry's ``generation_policy`` (``source_pipelines_v2.yaml``), the standing
  decision that lets internal users make ``experimental_non_candidate`` / ``internal_only``
  bundles on their own machines without a record per run. It releases no hold, and it applies
  the same conditions on every machine. A registry without a policy (v1) falls back to the
  gates, which refuse while the holds stand.

Either way the runner makes one class only: a source whose registry entry declares another class
is refused before anything runs, and a bundle whose INDEX.json (or a corpus whose fingerprint
file) says another class or scope fails the generate step. The refusal in ``gates.py`` is the
parent repository's hold check.

**Every build is fresh; nothing is resumed.** The bundle entrypoint and the corpus entrypoint
always write into an empty directory. A bundle directory that exists and holds anything -- a
finished bundle, an unfinished one, files this runner never wrote -- is refused unless
``replace_existing`` is given (``--replace-existing``), and so is a corpus directory the run has
to build into, unless ``rebuild_corpus`` and ``replace_existing`` are both given. The refusal
comes before anything runs and names what the directory holds. A finished corpus (its fingerprint
file, no ``.generating``) that no rebuild is asked of is reused: only read. A run that was killed
part-way is therefore built again from the start, however long it had run; that is the declared
price of never mixing takes of two runs, or of two corpora, in one bundle.

**A replacement moves the whole directory aside.** ``os.replace`` renames it on the same volume to
``<dir>.replaced-<run>`` (nothing is copied), the run registers that at once, copies the small
evidence files (:data:`REPLACED_EVIDENCE_FILES`) into its record's ``replaced/`` and builds into an
empty directory. When the generate step succeeds the aside directory is deleted, unless it is a
production directory -- any directory directly under a ``runs/experimental_generation_poc_demo``,
under any root -- whose payload is deleted only with the data owner's approval (retention rule 1.5): it
is left in place and the step's outcome names it. When the step fails, or dies on an exception
(one raised while the evidence is copied included), nothing is deleted: the fresh output is
renamed to ``<dir>.failed-<run>`` (or removed when it holds only the marker) and the aside
directory is renamed back, which restores the previous state exactly. While a
``<dir>.replaced-*`` or ``<dir>.failed-*`` stands beside it, a directory is not replaced again;
the refusal says how to restore or remove it. A production directory that holds a finished
generation is replaced in the order retention rule 1.2 sets, so ``replace_existing`` on one is
refused unless a ``_superseded/<name>_*`` folder beside it holds a copy of that very INDEX.json
or fingerprint and its ``SUPERSEDED.md``.

**One generate step per directory at a time.** Before it looks at them, the step takes a lock
beside the bundle directory and, when the source has a corpus stage, the corpus directory (built
or only read; not one in a folder no run may write, such as an evidence folder):
``<dir>.lock``, created with ``O_CREAT | O_EXCL`` and recording the process, the host, the start
time and the run identity. It is removed when the step ends, however it ends. A run that finds one
is refused with its owner and age; a lock left by a killed run is removed by hand, never by a run.

**A bundle is unfinished until the whole generate step is.** Its INDEX.json is written by the
generator, before the post-steps, ``KNOWN_LIMITATIONS.json`` and ``source_manifest.json``. So a
fresh build writes ``.generating`` into the directory first and removes it only when all of them
are done (the corpus stage does the same in the corpus directory, until its fingerprint and class
check). The marker is not read to resume anything: validate, readme, register, push and
``soma-synth pipeline`` refuse a bundle holding it, no run reuses a corpus holding it, and a
generate step over it needs ``--replace-existing`` like over any other directory that is not
empty.

**What a run will not do at all.** Sample arguments (the registry's ``selection_args``) aimed at
a production lineage directory; a production bundle built from any corpus but its own corpus
lineage; a sample run (a bundle outside the production lineage, or any selection argument)
building a corpus inside a production corpus lineage; passthrough arguments that set a flag the
runner controls (an output, the corpus handed to the bundle, a source location), or that would
reach no entrypoint (corpus arguments for a source without a corpus stage, any passthrough with
an explicit generate command); writing inside a folder the pipeline never writes into (``extracted``, ``raw_archives``, the
source folders, the body-model directories, ``_superseded``, ``_manifest_backfill``, and for a
bundle or corpus also ``_runs``; the sources and evidence folders are the ones retention rule
1.4 keeps from deletion) under the run's data root or ``SOMA_DATA_ROOT``; a bundle or
corpus directory directly under any ``runs/experimental_generation_poc_demo`` that is not this
source's own bundle or corpus lineage (another registry-declared lineage, or a production
directory nothing declares); and a bundle or corpus directory that is itself an aside
(``*.replaced-*``) or a failed replacement's output (``*.failed-*``). The last four are refused
before the run record is opened, since the record would be written there or beside it.

**The generate step** of a registry v2 source is up to three things in order: the retarget-corpus
stage (hknu, gaitex, addbiomechanics build their bundle from an SMPL-24 corpus; the runner builds
it from the extracted sources when its directory is missing or empty, or a rebuild is asked for),
the bundle entrypoint (given the corpus through the registry's flag), and the post-steps (PRISM's
insole heading). After them the runner renders ``KNOWN_LIMITATIONS.json`` from the repository's
limitations config, because the validator requires it of every bundle and a generator that had to
know about it would be a generator that could forget, and writes ``source_manifest.json``: the
source files the bundle's take manifests name, for tracing a licence withdrawal, and (its
``source_files`` block, ``source_trace.py``) every source file the run's corpus and bundle stages
could read under the source root, with the SHA-256 of the whole file -- the corpora themselves
name their sources only by name or by a first-MiB digest.

The runner hands its generators its data root as ``SOMA_DATA_ROOT``, so the body models they load
(``paths.body_model_dir``) are the ones the run record hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

from soma_synth.contracts import dataset_limitations as limitations_mod
from soma_synth.pipeline import body_models as body_models_mod
from soma_synth.pipeline import gates as gates_mod
from soma_synth.pipeline import generate as generate_mod
from soma_synth.pipeline import paths as paths_mod
from soma_synth.pipeline import provenance as provenance_mod
from soma_synth.pipeline import run_identity as run_identity_mod
from soma_synth.pipeline import source_trace as source_trace_mod
from soma_synth.pipeline import stages as stages_mod
from soma_synth.pipeline.run_directory import RunDirectory

#: Where the registry's repository-relative scripts resolve from. The decision records are the
#: parent project's and resolve from ``gates.GOVERNANCE_ROOT``.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

#: The orchestration steps, distinct from the fifteen canonical STAGES. A step is a thing the
#: operator asks for; a stage is a thing a take passes through. `cli.py` has used these four
#: since before this package existed, and push is deliberately not among them.
STEPS = ("generate", "validate", "readme", "register")

#: Where registration goes under the data root when no catalog is named: the local experimental
#: catalog (``<SOMA_DATA_ROOT>/experimental/catalog``), never ``<bundle parent>/catalog``, which is
#: inside the lineage container and is no catalog at all.
CATALOG_RELPATH = "experimental/catalog"

#: Per-file model overrides ``scripts/poc/anthro_smpl.py`` honours (AMASS and PRISM load their
#: models through it). They bypass the folder the run hashes, so a run records them when set.
MODEL_FILE_OVERRIDES = ("SOMA_SMPL_MODEL_MALE", "SOMA_SMPL_MODEL_FEMALE")

#: The flags that tell an entrypoint where to read its source (every one the registry v2 scripts
#: declare) and PRISM's ``--data-root``, which moves its output and its source together. The
#: runner hands the generators its data root as ``SOMA_DATA_ROOT`` and records the source root
#: they see (``SOMA_SOURCE_ROOT``, else ``<data root>/extracted``); a passthrough argument that
#: set one of these would read or write somewhere the run record does not name, so it is refused.
SOURCE_LOCATION_FLAGS = ("--source-root", "--root", "--hknu-root", "--amass-root", "--extracted")
DATA_ROOT_FLAGS = ("--data-root",)

BUNDLE_INDEX = "INDEX.json"

#: Written into the bundle directory when a fresh build starts (before the bundle entrypoint) and
#: removed only when the whole generate step has completed (generator, post-steps,
#: ``KNOWN_LIMITATIONS.json``, source manifest). A post-step that fails, or a run that is killed,
#: leaves the generator's INDEX.json beside takes nothing finished; this file says so to validate,
#: readme, register, push and ``soma-synth pipeline``, which refuse the bundle. The corpus stage
#: writes it into the corpus directory before the corpus entrypoint and removes it once the corpus
#: has its fingerprint and passed its class check: no run reuses a corpus holding it. Nothing
#: resumes from it.
GENERATING_MARKER = ".generating"
GENERATING_MARKER_SCHEMA = "soma_synth_generating_v2"

#: The generate step's lock, beside the bundle directory and the corpus directory it builds or
#: reads: ``<dir>.lock``, created with ``O_CREAT | O_EXCL`` and removed when the step ends. A run
#: that finds one is refused; a lock a killed run left is removed by hand, never by a run.
LOCK_SUFFIX = ".lock"
LOCK_SCHEMA = "soma_synth_generate_lock_v1"

#: A directory the runner moved aside to build its replacement fresh, and the output of a
#: replacement that failed, both beside the directory on the same volume:
#: ``<dir>.replaced-<run>`` and ``<dir>.failed-<run>``. ``<run>`` is the first
#: :data:`RUN_TOKEN_LENGTH` characters of the run identity, with the ``-N`` of a repeated run's
#: record (``run_<identity>-2``).
REPLACED_TAG = "replaced"
FAILED_TAG = "failed"
RUN_TOKEN_LENGTH = 12
_SIDE_DIRECTORY = re.compile(r"^(?P<base>.+)\.(?P<tag>replaced|failed)-(?P<run>[^\\/]+)$")

#: What is copied out of a directory the run moved aside into the run record's ``replaced/``: the
#: small files that say what that generation was (retention rule 1.3(b)), never its takes. A
#: corpus's own fingerprint file is added to these by name.
REPLACED_EVIDENCE_FILES = (
    BUNDLE_INDEX, "SUMMARY.json", "_run.json", "VALIDATION_REPORT.json", "validation_ledger.json",
    "KNOWN_LIMITATIONS.json", "README.md", "DATA_DESCRIPTION_EN.md", "DATASET_DESCRIPTION_EN.md",
    "reduced_model_fit.json", "PROVENANCE_RETROFIT.json", "PUSH_RECEIPT.txt", GENERATING_MARKER,
)
REPLACED_EVIDENCE_PATTERNS = ("INDEX_shard_*.json",)

#: The run record's ``logs/``: what the run printed (``runner.log``) and each child step's stdout
#: and stderr, teed there as the console shows them (``pipeline.generate``): the corpus entrypoint
#: (``corpus.log``), the bundle entrypoint and its merge (``generate.log``, a shard's
#: ``generate.shard<i>of<N>.log``) and each post-step (``post_step.<script>.log``). A refusal and
#: an unexpected failure add ``refusal.txt`` / ``error.txt``.
RUNNER_LOG = "runner.log"
CORPUS_LOG = "corpus.log"

#: Where, inside the run directory, a copy of what the run replaced is kept.
REPLACED_DIR = "replaced"
#: Where, inside the run directory, a copy of the owner record handed in with ``--authorized-by``
#: is kept.
AUTHORIZATION_DIR = "authorization"

#: The evidence folder beside the lineage directories, and the record each generation in it has.
SUPERSEDED_DIR = "_superseded"
SUPERSEDED_RECORD = "SUPERSEDED.md"

_REPLACE_NOTE = (
    "Every generate step builds into an empty directory and no run resumes one: a run that "
    "stopped, or was killed, is built again from the start. A replacement moves the whole "
    "directory aside to <dir>.replaced-<run>, keeps copies of its small evidence files in the run "
    "record and builds into an empty directory; on success the aside directory is deleted -- "
    "except a production directory's (any directory directly under "
    "runs/experimental_generation_poc_demo), which is left for the data owner's approved deletion "
    "(retention rule 1.5) -- and on failure it is renamed back. A production directory holding "
    "a finished generation is replaced in the order retention rule 1.2 sets: its evidence files "
    "copied to _superseded/<name>_<generated_utc date>/ with a SUPERSEDED.md first, so the "
    "replacement is refused unless a _superseded/<name>_* folder beside it holds a copy of that "
    "very INDEX.json (or corpus fingerprint) and its SUPERSEDED.md; a scratch directory is "
    "replaced without that check."
)


class RunnerRefusal(RuntimeError):
    """The runner declined to act, and says on whose authority."""


@dataclass
class StepOutcome:
    step: str
    status: str                       # "ok" | "skipped" | "refused" | "failed"
    detail: str = ""

    def as_json(self) -> dict[str, str]:
        return {"step": self.step, "status": self.status, "detail": self.detail}


@dataclass
class RunResult:
    run_identity: str
    run_directory: Path
    source_name: str
    dataset_dir: Path
    outcomes: list[StepOutcome] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return any(o.status == "refused" for o in self.outcomes)

    @property
    def ok(self) -> bool:
        """Nothing failed and nothing was refused: a refused run did not do what it was asked,
        and a script reading the exit code must not take it for a success."""
        return not any(o.status in ("failed", "refused") for o in self.outcomes)

    def summary(self) -> str:
        lines = [
            f"run {self.run_identity}  source={self.source_name}",
            f"  dataset : {self.dataset_dir}",
            f"  record  : {self.run_directory}",
        ]
        for outcome in self.outcomes:
            mark = {"ok": "  ok", "skipped": " skip", "refused": " REF", "failed": "FAIL"}[outcome.status]
            lines.append(f"  [{mark}] {outcome.step}" + (f" -- {outcome.detail}" if outcome.detail else ""))
        return "\n".join(lines)


def _plan(start_at: str, stop_after: str) -> tuple[str, ...]:
    return STEPS[STEPS.index(start_at): STEPS.index(stop_after) + 1]


def _logging_into(path: Path, log: Callable[[str], None]) -> Callable[[str], None]:
    """``log``, and every message also appended to ``path`` (the run record's ``runner.log``). A
    message that cannot be appended is still shown: the record is not worth failing a run for."""
    def both(message: str) -> None:
        log(message)
        try:
            with open(path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(f"{message}\n")
        except OSError:
            pass
    return both


def _same_folder(a: Path, b: Path) -> bool:
    return a == b or os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))


def _shaped_like_lineage(target: Path, lineage: str) -> bool:
    """Whether ``target`` is ``<any root>/runs/experimental_generation_poc_demo/<lineage>``.

    The runner's own data root is not the only one a production lineage can sit under: a run
    whose data root is a scratch folder can still be pointed at another root's lineage directory,
    and the generators' guard (``paths.is_default_lineage``) looks only under the data root the
    runner hands them.
    """
    wanted = [os.path.normcase(part) for part in (*Path(paths_mod.POC_DEMO).parts, lineage)]
    for candidate in (Path(target), Path(target).resolve()):
        parts = [os.path.normcase(part) for part in candidate.parts]
        if len(parts) > len(wanted) and parts[-len(wanted):] == wanted:
            return True
    return False


def _production_locations(path: Path | str) -> list[tuple[int, str]]:
    """(depth below a ``runs/experimental_generation_poc_demo`` container, the container's child
    that holds ``path``) for ``path`` as written and as resolved; empty when it is below none."""
    container = [os.path.normcase(part) for part in Path(paths_mod.POC_DEMO).parts]
    found = []
    for candidate in (Path(path), Path(path).resolve()):
        parts = list(candidate.parts)
        folded = [os.path.normcase(part) for part in parts]
        for start in range(len(folded) - len(container) + 1):
            if folded[start:start + len(container)] == container:
                depth = len(folded) - start - len(container)
                found.append((depth, parts[start + len(container)] if depth else ""))
                break
    return found


def production_depth(path: Path | str) -> int | None:
    """How far below a ``runs/experimental_generation_poc_demo`` container ``path`` lies, as
    written or resolved: 1 for a lineage directory itself, 2 or more for anything inside one,
    None when it is not below a container (0 is the container itself)."""
    found = _production_locations(path)
    return max(depth for depth, _ in found) if found else None


def inside_a_lineage(path: Path | str) -> bool:
    """Whether ``path`` lies inside (not at) a lineage directory under the container. The
    evidence folders (``_superseded``, ``_runs``, ``_manifest_backfill``) are not lineages; the
    output guard refuses everything inside them with its own, more specific reason."""
    evidence = {os.path.normcase(name) for name in paths_mod.EVIDENCE_FOLDERS}
    return any(depth >= 2 and os.path.normcase(child) not in evidence
               for depth, child in _production_locations(path))


def is_production_directory(path: Path | str) -> bool:
    """Whether ``path`` is a production directory or lies inside one: any directory at or below a
    direct child of a ``runs/experimental_generation_poc_demo``, under any root, as written or
    resolved -- the lineages the registry declares and every other one. Deleting its payload
    needs the data owner's approval (retention rule 1.5), so a replacement never deletes what it moved
    aside from one."""
    depth = production_depth(path)
    return depth is not None and depth >= 1


def lock_path(target: Path | str) -> Path:
    """``<target>.lock``: the generate step's lock beside a bundle or corpus directory, taken on
    the resolved directory so two spellings of one directory (a junction, a symlink) share it."""
    resolved = Path(target).resolve()
    return resolved.with_name(resolved.name + LOCK_SUFFIX)


def lock_refusal(lock: Path | str, target: Path | str, what: str) -> str:
    """Why a generate step may not take ``target``: another step holds its lock. Names the
    lock's owner and age, and how to tell a stale lock and remove it (no run deletes one)."""
    lock = Path(lock)
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    started = payload.get("started_utc")
    try:
        since = datetime.strptime(str(started), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()
    except ValueError:
        try:
            since = lock.stat().st_mtime
        except OSError:
            since = time.time()
    minutes = max(0, int(time.time() - since)) // 60
    pid, host = payload.get("pid", "?"), payload.get("host", "?")
    return (
        f"refusing to generate: the {what} directory {target} is locked by another generate "
        f"step -- {lock} names pid {pid} on {host}, run {payload.get('run_identity', '?')}, "
        f"started {started or '?'} ({minutes // 60}h{minutes % 60:02d}m ago). One generate step "
        "at a time builds into a directory. If that run is still going, let it finish. If it is "
        f"not -- on {host}, `tasklist /FI \"PID eq {pid}\"` (Windows) or `ps -p {pid}` shows no "
        "such process, or another program -- the lock is stale: its run was killed before it "
        f"could remove it. Delete {lock.name} by hand and run again; the {what} that run was "
        "building is unfinished, and building it again needs --replace-existing. No run deletes "
        "a lock it did not take"
    )


def selected_args(args: Sequence[str], selection: Sequence[str]) -> list[str]:
    """The arguments in ``args`` that are one of the ``selection`` flags.

    ``--subjects=S04`` counts as ``--subjects``, and so does an abbreviation argparse would
    accept (``--subj``): a prefix of a selection flag is treated as that flag.
    """
    chosen = []
    for arg in args:
        text = str(arg)
        if not text.startswith("--") or len(text) <= 2:
            continue
        name = text.split("=", 1)[0]
        if any(flag == name or flag.startswith(name) for flag in selection):
            chosen.append(text)
    return chosen


def _argparse_flag(name: str, declared: Sequence[str]) -> str | None:
    """The declared flag argparse takes ``name`` for: the flag itself, or the one declared flag
    ``name`` is a prefix of (``allow_abbrev``). None when it names none of them, or several."""
    if name in declared:
        return name
    matches = [flag for flag in declared if flag.startswith(name)]
    return matches[0] if len(matches) == 1 else None


def controlled_args(args: Sequence[str], controlled: Sequence[str],
                    declared: Sequence[str]) -> list[str]:
    """The arguments in ``args`` that would set one of the ``controlled`` flags.

    :func:`selected_args` finds every spelling (``--out``, ``--out=X``, a prefix such as ``--ou``);
    a spelling argparse would resolve to a *declared* flag that is not controlled is then let
    through (``--s`` is the hknu bundle's ``--subjects``, not ``--source-root``). A spelling that
    resolves to nothing is kept: it is a prefix of a controlled flag the entrypoint may not even
    declare, and refusing it costs nothing a real run needs.
    """
    hits = []
    for text in selected_args(args, controlled):
        resolved = _argparse_flag(text.split("=", 1)[0], declared)
        if resolved is None or resolved in controlled:
            hits.append(text)
    return hits


def _unused_run_root(root: Path) -> Path:
    """``root``, or a numbered sibling when a run with the same identity already wrote there.

    A run directory is immutable once written (``run_directory.py``). The identity is the same
    for an identical repeat, and reopening its directory would overwrite the first run's record,
    so the repeat is recorded beside it (``run_<identity>-2``) instead.
    """
    if not root.exists():
        return root
    number = 2
    while (candidate := root.with_name(f"{root.name}-{number}")).exists():
        number += 1
    return candidate


def _governance_file_sha256(relative: str) -> object:
    """The sha256 of a parent-project file the policy names (its decision record), read from the
    governance root (``gates.GOVERNANCE_ROOT``), or why it cannot be given: a standalone
    soma-synth checkout has no governance root and does not carry the parent's records."""
    root = gates_mod.GOVERNANCE_ROOT
    if root is None:
        return provenance_mod.Unavailable(f"{relative} is not in this checkout").as_json()
    target = root / relative
    if not target.is_file():
        return provenance_mod.Unavailable(
            f"{relative} is not under the governance root {root}").as_json()
    return provenance_mod.file_sha256(target)


# ------------------------------------------------------------------ side directories
def side_directory(path: Path | str) -> tuple[str, str, str] | None:
    """(the directory it stands beside, ``replaced`` or ``failed``, the run token) when ``path``
    is named like a directory a run moved aside or a failed replacement's output; else None."""
    match = _SIDE_DIRECTORY.match(Path(path).name)
    return (match["base"], match["tag"], match["run"]) if match else None


def side_directories(target: Path | str) -> list[Path]:
    """The ``<target>.replaced-*`` and ``<target>.failed-*`` directories beside ``target``."""
    target = Path(target)
    if not target.parent.is_dir():
        return []
    prefixes = tuple(os.path.normcase(f"{target.name}.{tag}-") for tag in (REPLACED_TAG, FAILED_TAG))
    return sorted(path for path in target.parent.iterdir()
                  if os.path.normcase(path.name).startswith(prefixes))


def _read_marker(directory: Path) -> dict[str, object]:
    """A ``.generating`` marker's JSON, or {} when it is missing or does not parse (read
    leniently: a refusal must not fail on a marker it cannot read)."""
    try:
        payload = json.loads((Path(directory) / GENERATING_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _marker_origin(directory: Path) -> str:
    """Which run left the ``.generating`` marker in ``directory``, for a refusal."""
    payload = _read_marker(directory)
    where = payload.get("run_directory_relative") or payload.get("run_directory")
    if not where:
        return "a run whose marker does not parse"
    return f"run record {where}, started {payload.get('started_utc', '?')}"


def _holds_nothing_but_the_marker(directory: Path) -> bool:
    return Path(directory).is_dir() and all(child.name == GENERATING_MARKER
                                            for child in Path(directory).iterdir())


def unfinished_bundle_refusal(dataset_dir: Path | str, *, step: str,
                              source: str | None = None) -> str | None:
    """Why ``step`` (validate, readme, register, push) must not take ``dataset_dir``, with the
    command that builds it; None when nothing stands in the way.

    Refused: a directory a run moved aside (``*.replaced-*``) or a failed replacement's output
    (``*.failed-*``), neither of which is the bundle, and a bundle whose generate step did not
    finish (``.generating``). ``source`` is the source to name in the command, else the one the
    marker names.
    """
    target = Path(dataset_dir)
    side = side_directory(target)
    if side is not None:
        base, tag, token = side
        lineage = target.with_name(base)
        if tag == REPLACED_TAG:
            return (
                f"refusing to {step} {target}: it is not a bundle but the generation of {lineage} "
                f"that run {token} moved aside to build its replacement fresh. Work on {lineage}. "
                "The aside directory stays only because it is a production directory's payload, "
                "whose deletion needs the data owner's approval (retention rule 1.5), or because that run "
                "was killed before it could settle it; delete it once the generation in "
                f"{lineage.name} is accepted, or remove {lineage.name} and rename it back to "
                "restore it"
            )
        return (
            f"refusing to {step} {target}: it is not a bundle but the output of run {token}, "
            f"whose replacement of {lineage} failed; {lineage} was restored to the generation it "
            "held before. Inspect this directory, then delete it; generate again with `soma-synth "
            f"run --source {source or '<source>'} {lineage} --replace-existing`"
        )
    marker = target / GENERATING_MARKER
    if marker.is_file():
        named = source or _read_marker(target).get("source") or "<source>"
        return (
            f"refusing to {step} {target}: its generation did not finish -- {marker.name} is "
            f"there ({_marker_origin(target)}). A fresh build writes it before the bundle "
            f"entrypoint runs and removes it only after the post-steps, "
            f"{limitations_mod.FILE_NAME} and source_manifest.json; an INDEX.json beside it may "
            "be the generator's, over takes a post-step never reached. No run resumes a bundle: "
            f"build it again with `soma-synth run --source {named} {target} --replace-existing`, "
            "which moves this directory aside and builds the bundle into an empty one"
        )
    return None


def declared_lineages(registry: stages_mod.PipelineRegistry) -> dict[str, str]:
    """{lineage directory name: what it is} for every lineage the registry declares: each
    source's ``bundle_lineage`` and ``corpus.lineage`` (empty for a v1 registry)."""
    declared: dict[str, str] = {}
    for name in sorted(registry.sources):
        source = registry.for_source(name)
        if source.bundle_lineage:
            declared[source.bundle_lineage] = f"the {name} bundle lineage"
        if source.corpus is not None:
            declared[source.corpus.lineage] = f"the {name} corpus lineage"
    return declared


def other_lineage_refusal(target: Path | str, own: str, what: str,
                          registry: stages_mod.PipelineRegistry) -> str | None:
    """Why ``target`` must not be the ``what`` directory of a source whose own lineage for it is
    ``own``: it is another lineage the registry declares, under any
    ``runs/experimental_generation_poc_demo``. None when it is not."""
    for lineage, meaning in declared_lineages(registry).items():
        if lineage != own and _shaped_like_lineage(Path(target), lineage):
            return (
                f"{target} is {meaning} ({paths_mod.POC_DEMO}/{lineage}), not "
                + (f"this source's {what} lineage {own}" if own else f"a {what} of this source")
                + f". Give the {what} its own lineage directory or a scratch directory"
            )
    return None


def production_directory_refusal(target: Path | str, own: str, what: str,
                                 registry: stages_mod.PipelineRegistry) -> str | None:
    """Why ``target`` must not be the ``what`` directory of a source whose own lineage for it is
    ``own``: another lineage the registry declares (:func:`other_lineage_refusal`), or any other
    production directory (:func:`is_production_directory`) -- one nothing declares is no less
    someone's payload. None for the source's own lineage and for a directory outside the
    lineage containers."""
    problem = other_lineage_refusal(target, own, what, registry)
    lineage_itself = any(depth == 1 for depth, _ in _production_locations(target))
    if problem or not lineage_itself or (own and _shaped_like_lineage(Path(target), own)):
        return problem
    return (
        f"{target} is a production directory ({paths_mod.POC_DEMO}/{Path(target).name}) that is "
        + (f"not this source's {what} lineage {own}" if own
           else f"not a {what} lineage of this source (the registry declares none)")
        + f". Give the {what} its own lineage directory or a scratch directory"
    )


@dataclass
class _DirectoryPlan:
    """What the generate step found in the bundle or the corpus directory, and will do."""

    what: str                          # "bundle" | "corpus"
    target: Path
    #: what the directory holds, as a refusal names it; None when it is missing or empty
    state: str | None = None
    #: it holds a finished generation: its INDEX.json or fingerprint, and no ``.generating``
    finished: bool = False
    #: moved aside to ``<dir>.replaced-<run>`` before the fresh build (``--replace-existing``)
    replace: bool = False
    #: the corpus: "reused", "built" or "rebuilt"
    action: str = ""


@dataclass
class _Build:
    """A directory this run started building fresh, and what it moved aside for it: what a
    failure puts back, and what a success deletes (``aside``, unless ``production``)."""

    what: str                          # "bundle" | "corpus"
    target: Path
    #: the directory did not exist before (or was moved aside): a failure may remove it
    created: bool
    aside: Path | None = None
    production: bool = False


@dataclass
class PipelineRunner:
    """One run. Open the record, walk the steps, close the record."""

    source_name: str
    #: The bundle directory. Default: the source's ``bundle_lineage`` under the data root.
    dataset_dir: Path | None = None
    #: The data root. Default: ``SOMA_DATA_ROOT`` (``paths.data_root``), which must then be set.
    data_root: Path | None = None
    runs_root: Path | None = None
    registry: stages_mod.PipelineRegistry | None = None
    #: Where owner authorization records are looked for. Tests point it at a scratch directory;
    #: a real run never needs to change it.
    decisions_dir: Path | None = None
    #: The retarget corpus of a source that has a corpus stage. Default: the corpus lineage
    #: directory under the data root.
    corpus_dir: Path | None = None

    def __post_init__(self) -> None:
        self.registry = self.registry or stages_mod.default_registry()
        # an unknown source is refused before anything else is resolved or created
        self.source = self.registry.for_source(self.source_name)
        self.data_root = Path(self.data_root) if self.data_root is not None else paths_mod.data_root()
        if self.dataset_dir is None:
            if not self.source.bundle_lineage:
                raise stages_mod.PipelineError(
                    f"{self.source_name}: {self.registry.path.name} names no bundle_lineage, so "
                    "there is no default bundle directory; pass one"
                )
            self.dataset_dir = paths_mod.lineage_dir(self.source.bundle_lineage, root=self.data_root)
        self.dataset_dir = Path(self.dataset_dir)
        corpus = self.source.corpus
        if corpus is not None:
            self.corpus_dir = (Path(self.corpus_dir) if self.corpus_dir is not None
                               else paths_mod.lineage_dir(corpus.lineage, root=self.data_root))
        elif self.corpus_dir is not None:
            raise stages_mod.PipelineError(
                f"{self.source_name} has no corpus stage in {self.registry.path.name}; a corpus "
                "directory means nothing for it"
            )
        if self.runs_root is None:
            self.runs_root = self.dataset_dir.parent / paths_mod.RUN_RECORDS
        # per run: the directories the generate step builds (and what it moved aside), the
        # locks it holds, and the run's short name for the side directories
        self._builds: list[_Build] = []
        self._locks: list[tuple[Path, bytes]] = []
        self._run_token = ""

    # ------------------------------------------------------------- locations
    def _relative(self, path: Path | str | None) -> str | None:
        return provenance_mod.relative_to_root(path, self.data_root)

    def default_catalog_dir(self) -> Path:
        return Path(self.data_root) / CATALOG_RELPATH

    def child_environment(self) -> dict[str, str]:
        """The generators' environment: this process's, with ``SOMA_DATA_ROOT`` set to the run's
        data root, so their defaults and body models resolve where the run record says."""
        env = dict(os.environ)
        env[paths_mod.DATA_ROOT_ENV] = str(self.data_root)
        return env

    def source_root_directory(self) -> tuple[Path, str]:
        """(the source root the generators will read, where it came from).

        The children inherit ``SOMA_SOURCE_ROOT`` from this process (:meth:`child_environment`)
        and are handed this run's data root, so they read ``SOMA_SOURCE_ROOT`` when it is set and
        ``<data root>/extracted`` otherwise (``paths.source_root``). A relative value raises
        ``PathConfigError`` here, before the run record is opened, as it would in every child.
        """
        override = paths_mod.source_root_override()
        if override is not None:
            return override, paths_mod.SOURCE_ROOT_ENV
        return Path(self.data_root) / paths_mod.EXTRACTED, "data_root"

    def _refuse_output_locations(self, *, catalog: Path | None, reads_corpus: bool,
                                 writes_corpus: bool) -> None:
        """Refuse, before a run record is opened, writing where no run may write.

        The generators guard their own outputs (``paths.check_output_dir``), but only once they
        run, and only under the data root the runner hands them. Several writes are the
        runner's own and come earlier or without any generator: the locks beside the bundle and
        corpus directories, moving them aside for a replacement or rebuild (and back after a
        failure), the ``.generating`` markers, the run record (by default
        ``<bundle parent>/_runs``), the bundle's validation report, ledger, README and
        ``KNOWN_LIMITATIONS.json``, and the catalog. So:

        * the bundle directory, and the corpus directory when this run may build or rebuild it,
          get the generators' guard against this run's data root and ``SOMA_DATA_ROOT``
          (``paths.check_output_dir(root=...)``): nothing inside ``extracted``,
          ``raw_archives``, the ``SOMA_SOURCE_ROOT`` source folders, the body-model
          directories, ``_superseded``, ``_manifest_backfill`` or ``_runs``, and not a
          container itself (a cloud-synchronised folder or a team shared drive draws only a
          warning, ``paths.warn_if_synced``). The bundle is also kept off the corpus it reads,
          as its generator would;
        * the run records folder and the catalog (when registering) get the narrower
          ``paths.check_record_dir``: ``_runs`` is where records go; everything else above is
          refused, and so are the containers themselves and anything inside a lineage
          directory;
        * the bundle directory, and the corpus directory whenever the run reads it, may not be a
          production directory other than this source's own lineage
          (:func:`production_directory_refusal`: amass's bundle lineage handed to prism, a corpus
          lineage given as a bundle, a bundle lineage given as ``--corpus``, an undeclared
          directory under the lineage container), nor a directory a run moved aside or a failed
          replacement's output (``*.replaced-*``, ``*.failed-*``).

        The refusal is raised rather than recorded, because the run record would be written into
        the very folder refused, or beside it; ``soma-synth run`` exits 3. A directory the run
        writes that lies inside a production directory (a take folder of a lineage, say) is
        refused first, since that is the most specific thing wrong with it.
        """
        root = self.data_root
        written = [("the bundle directory", "bundle", Path(self.dataset_dir))]
        if writes_corpus:
            written.append(("the corpus directory", "corpus", Path(self.corpus_dir)))
        for label, what, target in written:
            if inside_a_lineage(target):
                raise RunnerRefusal(
                    f"refusing to run {self.source_name}: {label}: {target} lies inside a "
                    f"production directory under {paths_mod.POC_DEMO}; a {what} is built into a "
                    "lineage directory itself or into a scratch directory, never into a folder "
                    "inside another generation")
        checks: list[tuple[str, Callable[[], object]]] = [(
            "the bundle directory",
            lambda: paths_mod.check_output_dir(
                self.dataset_dir, inputs=[self.corpus_dir] if reads_corpus else [], root=root),
        )]
        if writes_corpus:
            checks.append(("the corpus directory",
                           lambda: paths_mod.check_output_dir(self.corpus_dir, root=root)))
        checks.append(("the run records folder",
                       lambda: paths_mod.check_record_dir(self.runs_root, root=root)))
        if catalog is not None:
            checks.append(("the catalog", lambda: paths_mod.check_record_dir(catalog, root=root)))
        for label, check in checks:
            try:
                check()
            except paths_mod.OutputLocationRefused as error:
                raise RunnerRefusal(f"refusing to run {self.source_name}: {label}: {error}") \
                    from error
        for label, what, target in written:
            # the guards above know the evidence folders of this run's data root and of
            # SOMA_DATA_ROOT; one of another data root is no place for a bundle either
            found = paths_mod.evidence_folder(target)
            if found is not None:
                raise RunnerRefusal(
                    f"refusing to run {self.source_name}: {label}: {target} lies inside the "
                    f"evidence folder {paths_mod.POC_DEMO}/{found} of another data root, which "
                    "keeps records only (retention rules 1.3(b), 1.4)")
        corpus = self.source.corpus
        directories = [("the bundle directory", "bundle", Path(self.dataset_dir),
                        self.source.bundle_lineage)]
        if reads_corpus and corpus is not None:
            directories.append(("the corpus directory", "corpus", Path(self.corpus_dir),
                                corpus.lineage))
        for label, what, target, own in directories:
            problem = production_directory_refusal(target, own, what, self.registry)
            side = side_directory(target)
            if side is not None:
                how = ("moved aside" if side[1] == REPLACED_TAG
                       else "left when its replacement failed")
                problem = (f"{target} is a directory run {side[2]} {how}, not a {what} "
                           f"directory; name {target.with_name(side[0])}")
            if problem:
                raise RunnerRefusal(f"refusing to run {self.source_name}: {label}: {problem}")

    def _corpus_present(self) -> bool:
        """The corpus directory holds a finished corpus: its fingerprint file, and no
        ``.generating`` beside it (a corpus stage that did not complete leaves one)."""
        corpus = self.source.corpus
        if not (corpus and self.corpus_dir):
            return False
        target = Path(self.corpus_dir)
        return ((target / corpus.fingerprint).is_file()
                and not (target / GENERATING_MARKER).is_file())

    def _corpus_writable(self) -> bool:
        """Whether a run may write the corpus directory at all (the output guard lets it
        through). One no run may write -- a corpus read from an evidence or read-only folder --
        cannot be rebuilt under a run reading it, so it is read without a lock, and nothing is
        written beside it."""
        try:
            paths_mod.check_output_dir(self.corpus_dir, root=self.data_root)
        except paths_mod.OutputLocationRefused:
            return False
        return True

    def _refuse_unfinished_bundle(self, step: str) -> None:
        """Refuse validate, readme or register over a bundle whose generate step did not
        complete, or a directory that is no bundle at all (:func:`unfinished_bundle_refusal`);
        the message names the command that builds it."""
        problem = unfinished_bundle_refusal(self.dataset_dir, step=step, source=self.source_name)
        if problem:
            raise RunnerRefusal(problem)

    def _write_marker(self, run: RunDirectory, directory: Path, what: str) -> None:
        """Write this run's ``.generating`` marker into ``directory``: which run is building it,
        for a refusal to name. Nothing reads it to resume."""
        payload = {
            "schema": GENERATING_MARKER_SCHEMA,
            "note": (
                f"soma-synth run is building this {what}, or a run was stopped before its "
                "generate step completed. "
                + ("validate, readme, register and push refuse the bundle while this file is "
                   "here. " if what == "bundle" else "No run reuses the corpus while this file "
                   "is here. ")
                + "No run resumes it: it is built again with --replace-existing"
                + ("." if what == "bundle" else " --rebuild-corpus.")
            ),
            "what": what,
            "source": self.source_name,
            **self._located("run_directory", run.root),
            "run_identity": getattr(self, "_run_identity", None),
            "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        (Path(directory) / GENERATING_MARKER).write_text(json.dumps(payload, indent=2) + "\n",
                                                         encoding="utf-8", newline="\n")

    def _finish_generating(self, run: RunDirectory) -> None:
        (Path(self.dataset_dir) / GENERATING_MARKER).unlink(missing_ok=True)
        run.record_bundle({"generating_marker_removed": True})
        run.record_stage("bundle_generation_complete")

    # ------------------------------------------------------------------ locks
    def _lock(self, run: RunDirectory, target: Path, what: str) -> None:
        """Take ``<target>.lock`` for this generate step, or refuse (:func:`lock_refusal`).

        Created with ``O_CREAT | O_EXCL``, so of two runs that race for it one gets it; it
        records the process, the host, the start time and the run identity for the refusal
        the other gets. :meth:`_unlock` removes it when the step ends.
        """
        path = lock_path(target)
        data = (json.dumps({
            "schema": LOCK_SCHEMA,
            "note": (f"soma-synth run holds this while its generate step builds or reads the "
                     f"{what} directory {Path(target).name}; it is removed when the step ends. "
                     "Left by a run that was killed, it is stale: check that the process below "
                     "is gone, then delete this file by hand."),
            "what": what,
            "source": self.source_name,
            **self._located("directory", target),
            "pid": os.getpid(),
            "host": platform.node(),
            "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_identity": getattr(self, "_run_identity", None),
            **self._located("run_directory", run.root),
        }, indent=2) + "\n").encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                                 | getattr(os, "O_BINARY", 0))
        except FileExistsError:
            raise RunnerRefusal(lock_refusal(path, target, what)) from None
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        self._locks.append((path, data))
        run.record_stage(f"{what}_locked")

    def _unlock(self, log: Callable[[str], None]) -> None:
        """Remove the locks this step took -- only while they still hold what it wrote: a lock
        an operator removed and another run took since is that run's."""
        for path, data in reversed(self._locks):
            try:
                if path.read_bytes() == data:
                    path.unlink()
                else:
                    log(f"    {path} no longer holds this run's lock; left as it is")
            except FileNotFoundError:
                log(f"    {path} was removed while this run held it")
            except OSError as error:
                log(f"    could not remove the lock {path} ({error}); delete it by hand")
        self._locks = []

    def _located(self, key: str, path: Path | str | None) -> dict[str, object]:
        """``{key}_relative`` for a path, plus ``{key}`` (absolute) only when the path is outside
        the data root. Teammates' run records are collected for licence tracing (ADR-0041), so a
        record carries a PC's absolute paths only where there is no data-root-relative form."""
        relative = self._relative(path)
        located: dict[str, object] = {f"{key}_relative": relative}
        if relative is None and path is not None:
            located[key] = str(path)
        return located

    def _is_lineage_directory(self, target: Path | str | None, lineage: str) -> bool:
        """Whether ``target`` is a production lineage directory: the one under this run's data
        root, or one shaped like it under any other root."""
        if target is None or not lineage:
            return False
        target = Path(target)
        return (_same_folder(target, paths_mod.lineage_dir(lineage, root=self.data_root))
                or _shaped_like_lineage(target, lineage))

    # ------------------------------------------------------------- identity
    def body_model_directory(self) -> Path | None:
        """Where this source's body-model set sits for this run: the folder the generators load
        from (``paths.body_model_dir``, honouring ``SOMA_BODY_MODEL_DIR``)."""
        declared = self.registry.body_model_set_for(self.source_name)
        if declared is None:
            return None
        return body_models_mod.set_directory(declared.relative_path, self.data_root)

    def identity_inputs(self, *, rebuild_corpus: bool = False,
                        resolved_config: Mapping[str, object] | None = None,
                        ) -> run_identity_mod.RunIdentityInputs:
        """The identity's inputs, the same fields as ever.

        The input corpus is the one the bundle will read. Its fingerprint is taken only when the
        run will reuse it: a corpus that is missing, or that the run is asked to rebuild, is made
        in the run, and hashing it (or the copy about to be replaced) before it exists would name
        something the bundle was not built from. The run manifest's ``corpus`` block records what
        was built and its fingerprint afterwards.

        ``resolved_config`` (the ``config`` block of ``resolved_config.yaml``) fills
        ``resolved_config_sha256``, which GENERATION_PIPELINE_STANDARD §3.1 names, so two runs
        that differ only in their output directory, their selection or their steps have different
        identities and do not share a run directory. The data root and an absolute path that has
        a data-root-relative form are left out of the hash, so the same run on another PC keeps
        its identity.
        """
        revision = provenance_mod.code_revision()
        corpus = self.source.corpus
        if corpus is not None:
            relative = self._relative(self.corpus_dir)
            input_corpus = relative if relative is not None else Path(self.corpus_dir).as_posix()
            corpus_root = (Path(self.corpus_dir)
                           if self._corpus_present() and not rebuild_corpus else None)
        else:
            input_corpus = next(iter(self.source.inputs.values()), None)
            corpus_root = Path(self.data_root) / input_corpus if input_corpus else None
        # The set, not one file: every generator picks per subject, so the identity has to move
        # when any model in the set does (body_models.set_digest).
        model_digest = body_models_mod.set_digest(
            body_models_mod.set_hashes(self.body_model_directory())
        )
        config_digest = None
        if resolved_config is not None:
            portable = {key: value for key, value in resolved_config.items()
                        if key != "data_root"
                        and not (key == "dataset_dir"
                                 and resolved_config.get("dataset_dir_relative") is not None)}
            config_digest = hashlib.sha256(
                run_identity_mod.canonical_json(portable).encode("utf-8")).hexdigest()
        return run_identity_mod.RunIdentityInputs(
            spec_id=self.registry.spec_id,
            spec_version=self.registry.spec_version,
            source_name=self.source_name,
            entrypoint=self.source.entrypoint.script,
            resolved_config_sha256=config_digest,
            code_revision=revision.get("commit") if isinstance(revision, dict) else None,
            code_dirty=revision.get("dirty") if isinstance(revision, dict) else None,
            input_corpus=input_corpus,
            input_corpus_sha256=(
                run_identity_mod.corpus_fingerprint(corpus_root) if corpus_root else None
            ),
            body_model_sha256=model_digest,
        )

    # --------------------------------------------------------------- decision
    def _declared_artifact_class(self) -> str | None:
        """The class this source's bundles are made as, per the registry; None for a v1 registry
        (which declares none; an owner record then carries the class, checked by the gates)."""
        policy = self.registry.generation_policy
        declared = self.source.artifact_class or (policy.artifact_class if policy else "")
        return declared or None

    def _standing_decision(self) -> dict[str, object]:
        policy = self.registry.generation_policy
        assert policy is not None
        return {
            "allowed": True,
            "source_name": self.source_name,
            **policy.as_json(),
            "decision_record_sha256": _governance_file_sha256(policy.decision_record),
            "registry": self.registry.display_path(),
            "authorization": None,
            "note": (
                "Generated under the standing decision of " + policy.adr + " ("
                + policy.decision_record + " " + policy.decision_section + "): "
                "experimental_non_candidate / internal_only generation by an internal user on "
                "their own machine, without an owner record per run. It releases no hold; "
                "canonical and internal-candidate generation stay under the holds."
            ),
        }

    def _decide(self, run: RunDirectory, authorized_by: Path | None) -> None:
        """Take the generation decision, record it, or raise RunnerRefusal."""
        declared = self._declared_artifact_class()
        if declared is not None and declared != stages_mod.EXPERIMENTAL_ARTIFACT_CLASS:
            reason = (
                f"{self.registry.path.name} declares artifact_class {declared!r} for source "
                f"{self.source_name!r}; this runner makes "
                f"{stages_mod.EXPERIMENTAL_ARTIFACT_CLASS!r} bundles only (ADR-0041). Canonical "
                "and internal-candidate generation stay under the parent project's generation holds"
            )
            run.record_generation_decision({
                "allowed": False, "source_name": self.source_name, "basis": "artifact_class",
                "artifact_class": declared, "reasons": [reason], "authorization": None,
            })
            raise RunnerRefusal(f"generation is not authorised for source {self.source_name!r}: "
                                + reason)

        policy = self.registry.generation_policy
        if authorized_by is None and policy is not None:
            # the standing decision applies the same conditions on every machine
            run.record_stage("generation_standing_decision")
            run.record_generation_decision(self._standing_decision())
            return

        try:
            decision = gates_mod.generation_decision(
                self.source_name, authorized_by=authorized_by,
                **({"decisions_dir": self.decisions_dir} if self.decisions_dir else {}),
            )
        except gates_mod.GateError as error:
            # A record that cannot authorise this run is a refusal with a reason, not a crash.
            raise RunnerRefusal(
                f"generation is not authorised for source {self.source_name!r}: {error}"
            ) from error
        run.record_stage("generation_gate_checked")
        record: dict[str, object] = {
            **decision.as_json(),
            "basis": "owner_record" if decision.authorization is not None else "gates",
        }
        if decision.authorization is not None and authorized_by is not None:
            # The record's words are what authorised the run: its hash says which version of the
            # file that was, and a copy keeps them with the run (see gates.py).
            copy = run.root / AUTHORIZATION_DIR / Path(authorized_by).name
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(authorized_by, copy)
            record["authorization_record_sha256"] = provenance_mod.file_sha256(authorized_by)
            record["authorization_record_copy"] = copy.relative_to(run.root).as_posix()
        run.record_generation_decision(record)
        if not decision.allowed:
            raise RunnerRefusal(decision.refusal_message())
        if decision.authorization is not None:
            run.record_stage("generation_authorized_by_owner_record")
            (run.root / "logs" / "authorization.txt").write_text(
                decision.authorization_message(), encoding="utf-8", newline="\n"
            )

    # ---------------------------------------------------------------- steps
    def _refuse_partial_into_lineage(self, target: Path, lineage: str, selection: Sequence[str],
                                     args: Sequence[str], what: str) -> None:
        """Refuse selection arguments aimed at a production lineage directory.

        The arguments exist for sample runs (``--subjects austra``); a partial run into the
        lineage directory rewrites its INDEX.json, fit record and description over the sample
        alone (the rule ``paths.refuse_partial_run_into_lineage`` enforces inside the
        generators, applied here before anything runs). Only the registry's ``selection_args``
        are refused; ``--force``, ``--compress`` or ``--resume`` reach the production lineage.
        """
        chosen = selected_args(args, selection)
        if not chosen or not self._is_lineage_directory(target, lineage):
            return
        scratch = Path(self.data_root) / "tmp" / lineage
        raise RunnerRefusal(
            f"refusing to pass {' '.join(chosen)} to the {what} entrypoint: its output {target} is "
            f"a production lineage directory ({paths_mod.POC_DEMO}/{lineage}), and a run over "
            f"part of the source would rewrite it over the selection alone. Give the {what} a "
            f"scratch directory instead (e.g. {scratch})"
        )

    def _refuse_sample_corpus_into_production(self) -> None:
        """Refuse building the production bundle from any corpus but its own corpus lineage.

        The partial-run check above looks at each entrypoint's own arguments, which a sample
        corpus in a scratch folder, handed to the default bundle directory with no bundle
        arguments, does not trip. A production bundle is therefore built only from its own
        corpus lineage, so a sample corpus cannot rewrite its INDEX.json.
        """
        corpus = self.source.corpus
        if corpus is None or not self._is_lineage_directory(self.dataset_dir,
                                                             self.source.bundle_lineage):
            return
        own = Path(self.dataset_dir).parent / corpus.lineage
        if _same_folder(Path(self.corpus_dir), own):
            return
        scratch = Path(self.data_root) / "tmp" / self.source.bundle_lineage
        raise RunnerRefusal(
            f"refusing to build {self.dataset_dir} from the corpus {self.corpus_dir}: the bundle "
            f"directory is a production lineage directory ({paths_mod.POC_DEMO}/"
            f"{self.source.bundle_lineage}), which is built from its own corpus lineage {own} "
            "only. A bundle built from another corpus (a sample) would rewrite its INDEX.json over "
            f"that corpus alone. Give the bundle a scratch directory too (e.g. {scratch})"
        )

    def _refuse_sample_corpus_build_into_production(self, *, rebuild_corpus: bool,
                                                    corpus_args: Sequence[str],
                                                    bundle_args: Sequence[str]) -> None:
        """Refuse a sample run building (or rebuilding) a corpus in a production corpus lineage.

        The corpus stage builds the corpus when its fingerprint is missing or a rebuild is asked
        for, and its directory defaults to the corpus lineage under the data root. A sample run
        -- a bundle outside the production bundle lineage beside that corpus, or any selection
        argument -- that took the default would build the production corpus as a side effect of
        the sample: the whole source converted into the lineage directory when the corpus is
        missing, or a partial corpus when a selection reaches the corpus entrypoint. A sample
        run that *reuses* a finished production corpus only reads it and is not refused.
        """
        corpus = self.source.corpus
        if corpus is None:
            return
        builds = rebuild_corpus or not self._corpus_present()
        if not builds or not self._is_lineage_directory(self.corpus_dir, corpus.lineage):
            return
        production_bundle = Path(self.corpus_dir).parent / self.source.bundle_lineage
        why = []
        if not _same_folder(Path(self.dataset_dir), production_bundle):
            why.append(f"the bundle directory {self.dataset_dir} is not the production bundle "
                       f"lineage {production_bundle}")
        chosen = (selected_args(bundle_args, self.source.entrypoint.selection_args)
                  + selected_args(corpus_args, corpus.selection_args))
        if chosen:
            why.append(f"it selects part of the source ({' '.join(chosen)})")
        if not why:
            return
        scratch = Path(self.data_root) / "tmp" / corpus.lineage
        raise RunnerRefusal(
            f"refusing to {'rebuild' if self._corpus_present() else 'build'} the corpus "
            f"{self.corpus_dir} in a sample run: it is a production corpus lineage directory "
            f"({paths_mod.POC_DEMO}/{corpus.lineage}), and this run is a sample because "
            + " and ".join(why) + ". A sample builds its own corpus: pass --corpus "
            f"<scratch dir> (e.g. --corpus {scratch})"
        )

    def _refuse_controlled_args(self, corpus_args: Sequence[str],
                                bundle_args: Sequence[str]) -> None:
        """Refuse passthrough arguments that would override a flag the runner sets itself.

        The runner puts the output flag (and the bundle's corpus flag) first on each command line
        and the passthrough arguments after it, where argparse lets a repeated flag win. An
        ``--corpus-arg=--out`` or ``--bundle-arg=--paired`` would then write or read somewhere
        other than what the run record names, and a source-location flag would read a source the
        record does not name (the record carries ``SOMA_SOURCE_ROOT``, else
        ``<data root>/extracted``). ``--x=v`` spellings and argparse abbreviations count
        (:func:`controlled_args`).
        """
        locations = {
            **{flag: "the data root is --data-root of soma-synth run (or SOMA_DATA_ROOT)"
               for flag in DATA_ROOT_FLAGS},
            **{flag: "the source root is SOMA_SOURCE_ROOT (default <data root>/extracted)"
               for flag in SOURCE_LOCATION_FLAGS},
        }
        entry = self.source.entrypoint
        corpus = self.source.corpus
        bundle_controlled = {
            entry.output_arg: "the bundle directory is the dataset_dir argument of soma-synth run",
            **locations,
        }
        # (what, passthrough arguments, {controlled flag: where to set it instead}, declared)
        checks = [("bundle", bundle_args, bundle_controlled,
                   (*entry.required_args, *entry.optional_args, entry.output_arg))]
        if corpus is not None:
            bundle_controlled[corpus.bundle_arg] = "the corpus is --corpus <dir> of soma-synth run"
            checks.append(("corpus", corpus_args,
                           {corpus.output_arg: "the corpus directory is --corpus <dir> of "
                                               "soma-synth run", **locations},
                           (*corpus.required_args, *corpus.optional_args, corpus.output_arg)))
        problems = []
        for what, args, controlled, declared in checks:
            for text in controlled_args(args, tuple(controlled), declared):
                name = text.split("=", 1)[0]
                flag = _argparse_flag(name, declared)
                if flag not in controlled:
                    # a prefix of a controlled flag the entrypoint does not declare
                    flag = next(f for f in controlled if f.startswith(name))
                problems.append(f"--{what}-arg {text} sets {flag}, which the runner controls "
                                f"({controlled[flag]})")
        if problems:
            raise RunnerRefusal("refusing to run: " + "; ".join(problems))

    def _refuse_unused_args(self, *, generate_cmd, corpus_args: Sequence[str],
                            bundle_args: Sequence[str]) -> None:
        """Refuse passthrough arguments that would reach no entrypoint.

        They are refused because a dropped selection turns a sample run into a full run: corpus
        arguments for a source without a corpus stage (PRISM, AMASS) would never reach the
        selection checks, and the run would cover every subject. An explicit generate command
        replaces the registry's plan, so no passthrough argument reaches anything either. (A
        corpus the run reuses takes no corpus arguments; that is logged and recorded rather than
        refused, so that a sample run whose bundle failed can be run again with the same
        command.)
        """
        if generate_cmd and (corpus_args or bundle_args):
            given = [*(f"--corpus-arg {arg}" for arg in corpus_args),
                     *(f"--bundle-arg {arg}" for arg in bundle_args)]
            raise RunnerRefusal(
                f"refusing to run: an explicit generate command replaces the registry's plan, so "
                f"{' '.join(given)} would reach no entrypoint. Put them in the command"
            )
        if corpus_args and self.source.corpus is None:
            raise RunnerRefusal(
                f"refusing to run: {self.source_name} has no corpus stage in "
                f"{self.registry.path.name}, so --corpus-arg {' '.join(corpus_args)} would reach "
                "no entrypoint and the run would cover the whole source. An argument for the "
                "bundle entrypoint goes in --bundle-arg, where a selection is checked against "
                "the production lineage"
            )

    def _superseded_copy(self, directory: Path, file: Path) -> Path | None:
        """The ``SUPERSEDED.md`` of a ``_superseded/<name>_*`` folder beside ``directory`` that
        holds a copy of ``file`` with the same SHA-256, or None.

        That is step (1) of retention rule 1.2 done for this very generation: its evidence
        copied out (1.3(b)) and the record written (1.3(c)). Matched by content rather than by
        the folder's date (``<name>_<INDEX generated_utc date>``), because a corpus fingerprint
        carries no ``generated_utc``.
        """
        container = Path(directory).parent / SUPERSEDED_DIR
        if not container.is_dir():
            return None
        digest = provenance_mod.file_sha256(file)
        wanted = os.path.normcase(Path(directory).name)
        for folder in sorted(container.iterdir()):
            name = os.path.normcase(folder.name)
            if not folder.is_dir() or not (name == wanted or name.startswith(wanted + "_")):
                continue
            copy, record = folder / file.name, folder / SUPERSEDED_RECORD
            if (copy.is_file() and record.is_file()
                    and provenance_mod.file_sha256(copy) == digest):
                return record
        return None

    # ------------------------------------------------------------ fresh directories
    def _found(self, target: Path, what: str, evidence: str) -> tuple[str | None, bool]:
        """(what ``target`` holds, in the words a refusal uses -- None when it is missing or
        empty; whether that is a finished generation: ``evidence`` there, no ``.generating``)."""
        if not target.exists():
            return None, False
        if not target.is_dir():
            raise RunnerRefusal(f"refusing to generate: the {what} directory {target} is a file")
        entries = sorted(child.name for child in target.iterdir())
        if not entries:
            return None, False
        if (target / GENERATING_MARKER).is_file():
            return (f"an unfinished generation ({GENERATING_MARKER} of "
                    f"{_marker_origin(target)})"), False
        if (target / evidence).is_file():
            return f"a finished {what} ({evidence})", True
        shown = ", ".join(entries[:3]) + (", ..." if len(entries) > 3 else "")
        return (f"{len(entries)} entries ({shown}) but no {evidence} and no "
                f"{GENERATING_MARKER}"), False

    def _refuse_side_directories(self, plan: _DirectoryPlan) -> None:
        """Refuse replacing a directory while an earlier replacement's aside directory or failed
        output stands beside it, saying how to restore or remove each."""
        standing = side_directories(plan.target)
        if not standing:
            return
        name = plan.target.name
        described = []
        for path in standing:
            _, tag, token = side_directory(path) or ("", "", "?")
            if tag == REPLACED_TAG:
                described.append(
                    f"{path.name} holds the generation run {token} moved aside: to keep what "
                    f"{name} holds now, delete {path.name} (a production directory's only with "
                    f"the data owner's approval, retention rule 1.5); to restore it instead, remove {name} "
                    f"and rename {path.name} back to {name}")
            else:
                described.append(
                    f"{path.name} holds the output of run {token}, whose replacement failed and "
                    "put the generation before it back: inspect it, then delete it")
        if plan.state is not None:
            opening = (f"refusing to replace the {plan.what} {plan.target}: a directory is replaced "
                       "one generation at a time")
        else:
            opening = (f"refusing to build the {plan.what} {plan.target}: a directory holds one "
                       "generation at a time")
        raise RunnerRefusal(opening + ", and beside it " + "; ".join(described) + ". Then run again")

    def _refuse_replacing_production(self, plan: _DirectoryPlan, file: Path) -> None:
        """A production directory holding a finished generation is replaced only once step (1)
        of retention rule 1.2 was done for that very generation (:meth:`_superseded_copy`)."""
        if not (plan.finished and is_production_directory(plan.target)):
            return                       # a scratch directory, or no finished generation
        if self._superseded_copy(plan.target, file) is not None:
            return
        name = plan.target.name
        raise RunnerRefusal(
            f"refusing to replace the {plan.what} {plan.target}: it is a production directory "
            f"({paths_mod.POC_DEMO}/{name}) holding a finished generation, and no "
            f"{SUPERSEDED_DIR}/{name}_* folder beside it holds a copy of its {file.name} "
            f"(sha256 {provenance_mod.file_sha256(file)}) with a {SUPERSEDED_RECORD}. "
            + _REPLACE_NOTE
        )

    def _plan_corpus(self, *, rebuild: bool, replace_existing: bool) -> _DirectoryPlan:
        """What the corpus stage will do with the corpus directory, refusing what it must not.

        A finished corpus (its fingerprint, no ``.generating``) that no rebuild is asked of is
        reused. Otherwise the corpus is built into an empty directory: directly where the
        directory is missing or empty; where it holds anything -- a finished corpus, an
        unfinished build, other files -- only with ``--rebuild-corpus --replace-existing``, which
        move it aside first (and, on a production directory holding a finished corpus, only
        once its evidence is in ``_superseded``).
        """
        corpus = self.source.corpus
        assert corpus is not None and self.corpus_dir is not None
        plan = _DirectoryPlan("corpus", Path(self.corpus_dir))
        plan.state, plan.finished = self._found(plan.target, "corpus", corpus.fingerprint)
        if plan.finished and not rebuild:
            plan.action = "reused"
            return plan
        plan.action = "rebuilt" if plan.finished else "built"
        if plan.state is None:
            self._refuse_side_directories(plan)
            return plan
        if not (rebuild and replace_existing):
            raise RunnerRefusal(
                f"refusing to {'rebuild' if plan.finished else 'build'} the corpus "
                f"{plan.target}: it already holds {plan.state}. Pass --rebuild-corpus "
                "--replace-existing to move it aside and build the corpus into an empty "
                "directory, or name a missing or empty directory with --corpus. " + _REPLACE_NOTE
            )
        self._refuse_side_directories(plan)
        self._refuse_replacing_production(plan, plan.target / corpus.fingerprint)
        plan.replace = True
        return plan

    def _plan_bundle(self, *, replace_existing: bool) -> _DirectoryPlan:
        """What the generate step will do with the bundle directory, refusing what it must not:
        a directory that is missing or empty is built into; one that holds anything -- a
        finished bundle, an unfinished one, other files -- only with ``--replace-existing``,
        which moves it aside first (and, on a production directory holding a finished bundle,
        only once its evidence is in ``_superseded``)."""
        plan = _DirectoryPlan("bundle", Path(self.dataset_dir))
        plan.state, plan.finished = self._found(plan.target, "bundle", BUNDLE_INDEX)
        if plan.state is None:
            self._refuse_side_directories(plan)
            return plan
        if not replace_existing:
            raise RunnerRefusal(
                f"refusing to generate into {plan.target}: it already holds {plan.state}. Pass "
                "--replace-existing to move it aside and build the bundle into an empty "
                "directory, or name a missing or empty directory. " + _REPLACE_NOTE
            )
        self._refuse_side_directories(plan)
        self._refuse_replacing_production(plan, plan.target / BUNDLE_INDEX)
        plan.replace = True
        return plan

    def _side_path(self, target: Path, tag: str) -> Path:
        """``<target>.<tag>-<run>``, numbered further when that name is taken."""
        name = f"{target.name}.{tag}-{self._run_token}"
        candidate, number = target.with_name(name), 2
        while candidate.exists():
            candidate, number = target.with_name(f"{name}-{number}"), number + 1
        return candidate

    def _evidence_files(self, directory: Path, what: str) -> list[Path]:
        names = list(REPLACED_EVIDENCE_FILES)
        corpus = self.source.corpus
        if what == "corpus" and corpus is not None:
            names.append(corpus.fingerprint)
        found = [directory / name for name in dict.fromkeys(names) if (directory / name).is_file()]
        for pattern in REPLACED_EVIDENCE_PATTERNS:
            found += sorted(path for path in directory.glob(pattern) if path.is_file())
        return found

    def _start_build(self, run: RunDirectory, plan: _DirectoryPlan,
                     log: Callable[[str], None]) -> None:
        """Make ``plan.target`` an empty directory holding this run's ``.generating`` marker,
        moving what it holds aside first when the plan replaces it (:meth:`_move_aside`)."""
        if plan.replace:
            self._move_aside(run, plan, log)
        else:
            self._builds.append(_Build(plan.what, plan.target, created=not plan.target.exists()))
        plan.target.mkdir(parents=True, exist_ok=True)
        self._write_marker(run, plan.target, plan.what)
        run.record_stage(f"{plan.what}_generating_marker_written")

    def _move_aside(self, run: RunDirectory, plan: _DirectoryPlan,
                    log: Callable[[str], None]) -> None:
        """Rename ``plan.target`` aside on the same volume, then copy its small evidence files
        into the run record. The aside directory is registered the moment ``os.replace``
        returns, so any failure from there on -- one copying the evidence included -- renames
        it back (:meth:`_settle_failure`)."""
        target = plan.target
        aside = self._side_path(target, REPLACED_TAG)
        production = is_production_directory(target)
        try:
            os.replace(target, aside)
        except OSError as error:
            raise RuntimeError(f"could not move {target} aside to {aside.name}: {error}; nothing "
                               "was changed") from error
        self._builds.append(_Build(plan.what, target, created=True, aside=aside,
                                   production=production))
        run.record_replaced(plan.what, {
            **self._located("directory", target),
            **self._located("aside", aside),
            "aside_name": aside.name,
            "found": plan.state,
            "production_directory": production,
            "moved_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "outcome": "moved_aside",
        })
        run.record_stage(f"{plan.what}_moved_aside")
        evidence = []
        for source in self._evidence_files(aside, plan.what):
            copy = run.root / REPLACED_DIR / plan.what / source.name
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, copy)
            evidence.append({"file": source.name, "sha256": provenance_mod.file_sha256(source),
                             "copy": copy.relative_to(run.root).as_posix()})
        run.record_replaced(plan.what, {"evidence": evidence})
        log(f"    {target} ({plan.state}) moved aside to {aside.name}; building the {plan.what} "
            "into an empty directory")

    def _settle_failure(self, run: RunDirectory, log: Callable[[str], None]) -> list[str]:
        """After a generate step that failed: every directory this run built goes back to what
        it found, and nothing is deleted but this run's lone marker. A replaced directory's fresh
        output becomes ``<dir>.failed-<run>`` and the aside directory is renamed back; a
        directory built where nothing stood keeps what the run wrote, marker included (an
        unfinished generation). Either is removed when it holds only the marker."""
        notes = []
        for build in reversed(self._builds):
            target, entry, failed = build.target, {}, None
            try:
                if _holds_nothing_but_the_marker(target):
                    (target / GENERATING_MARKER).unlink(missing_ok=True)
                    if build.created:
                        target.rmdir()
                elif build.aside is not None and target.exists():
                    failed = self._side_path(target, FAILED_TAG)
                    os.replace(target, failed)
                    entry.update({**self._located("failed_output", failed),
                                  "failed_output_name": failed.name})
                if build.aside is not None:
                    os.replace(build.aside, target)
            except OSError as error:
                entry.update({"outcome": "restore_failed",
                              "restore_error": f"{type(error).__name__}: {error}"})
                note = (f"COULD NOT restore {target} ({error}): "
                        + (f"the previous {build.what} is still at {build.aside}; rename it back "
                           "by hand" if build.aside is not None else "it is left as it is"))
            else:
                if build.aside is None:
                    continue
                entry["outcome"] = "restored"
                note = (f"the previous {build.what} was restored to {target}"
                        + (f" (this run's output is kept at {failed.name})" if failed else ""))
            if build.aside is not None:
                run.record_replaced(build.what, entry)
                run.record_stage(f"{build.what}_{entry['outcome']}")
            log(f"    {note}")
            notes.append(note)
        self._builds = []
        return notes

    def _settle_success(self, run: RunDirectory, log: Callable[[str], None]) -> list[str]:
        """After a generate step that succeeded: every aside directory is deleted, except a
        production directory's, whose payload is deleted only with the data owner's approval
        (retention rule 1.5). Returns what the step's outcome says about those it kept."""
        notes = []
        for build in self._builds:
            if build.aside is None:
                continue
            entry: dict[str, object] = {}
            note = None
            if not build.aside.exists():
                entry["outcome"] = "gone"
            elif build.production:
                entry["outcome"] = "kept_production_directory"
                note = (f"the previous {build.what} is kept at {build.aside}: {build.target} is a "
                        "production directory, and deleting its payload needs the owner's "
                        f"approval (retention rule 1.5); {build.target.name} is not replaced again "
                        "until it is deleted")
            else:
                try:
                    shutil.rmtree(build.aside)
                except OSError as error:
                    entry.update({"outcome": "delete_failed",
                                  "delete_error": f"{type(error).__name__}: {error}"})
                    note = (f"the previous {build.what} at {build.aside} could not be deleted "
                            f"({error}); delete it by hand")
                else:
                    entry["outcome"] = "deleted"
                    log(f"    deleted {build.aside.name} (the previous {build.what})")
            run.record_replaced(build.what, entry)
            if note:
                log(f"    {note}")
                notes.append(note)
        self._builds = []
        return notes

    def _corpus_class_check(self, fingerprint_path: Path) -> tuple[dict[str, object], list[str]]:
        """What the corpus's fingerprint file says about its class, and what is wrong with it.

        The hknu and gaitex fingerprints carry ``artifact_class`` and ``distribution_scope``;
        the addbiomechanics SUMMARY.json carries neither (its artifacts do), so a fingerprint
        that declares nothing is recorded as undeclared rather than failed.
        """
        policy = self.registry.generation_policy
        expected = {
            "artifact_class": policy.artifact_class if policy else stages_mod.EXPERIMENTAL_ARTIFACT_CLASS,
            "distribution_scope": policy.distribution_scope if policy else stages_mod.INTERNAL_ONLY,
        }
        try:
            payload = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            return ({"declared": False, "error": f"{type(error).__name__}: {error}"},
                    [f"{fingerprint_path} does not parse, so the corpus class cannot be checked"])
        payload = payload if isinstance(payload, dict) else {}
        found = {key: payload.get(key) for key in expected if key in payload}
        problems = [f"corpus {fingerprint_path.name} {key} is {value!r}, not {expected[key]!r}"
                    for key, value in found.items() if value != expected[key]]
        return {"declared": bool(found), **found}, problems

    def _corpus_stage(self, run: RunDirectory, plan: _DirectoryPlan, *,
                      corpus_args: Sequence[str], env: Mapping[str, str],
                      log: Callable[[str], None]) -> StepOutcome | None:
        """Build the corpus into an empty directory when the plan says so, and record it either
        way. Returns a failed outcome, or None when the bundle may go ahead."""
        corpus = self.source.corpus
        assert corpus is not None
        target = plan.target
        fingerprint_path = target / corpus.fingerprint
        reused = plan.action == "reused"
        record: dict[str, object] = {
            "lineage": corpus.lineage,
            **self._located("directory", target),
            "entrypoint": corpus.script,
            "present_at_open": plan.finished,
            # what the directory held at open, in the refusals' words (None: missing or empty)
            "found_at_open": plan.state,
            "fingerprint_file": corpus.fingerprint,
            "bundle_arg": corpus.bundle_arg,
            "corpus_args": list(corpus_args),
            "action": plan.action,
            "built_in_run": not reused,
            # a reused corpus takes no corpus arguments; the log says so, and so does the record
            "corpus_args_used": not reused,
            # what it held was moved aside to <dir>.replaced-<run> (manifest.json replaced.corpus)
            "moved_aside": plan.replace,
            "identity_note": (
                "the run identity fingerprints the corpus only when it existed at open and was "
                "reused; a corpus built in this run is fingerprinted here"
            ),
        }
        if reused:
            log(f"    corpus {target} present ({corpus.fingerprint}); reused"
                + (f"; corpus arguments {' '.join(corpus_args)} not used (--rebuild-corpus "
                   "builds it again)" if corpus_args else ""))
            run.record_stage("corpus_reused")
        else:
            self._start_build(run, plan, log)
            log(f"    building the corpus into {target} ({plan.action})")
            code = generate_mod.run_command(
                corpus.argv(sys.executable, _REPOSITORY_ROOT, target, corpus_args), env=env,
                log_path=run.root / "logs" / CORPUS_LOG)
            if code != 0:
                run.record_corpus({**record, "failed": f"corpus entrypoint exited {code}"})
                return StepOutcome("generate", "failed", f"corpus entrypoint exited {code}")
            run.record_stage(f"corpus_{plan.action}")
        if not fingerprint_path.is_file():
            run.record_corpus({**record, "failed": f"no {corpus.fingerprint}"})
            return StepOutcome(
                "generate", "failed",
                f"the corpus {target} has no {corpus.fingerprint}, so it cannot be fingerprinted",
            )
        run.record_source_asset(f"{corpus.input_key}/{corpus.fingerprint}", fingerprint_path)
        class_record, problems = self._corpus_class_check(fingerprint_path)
        record.update({
            "fingerprint_sha256": provenance_mod.file_sha256(fingerprint_path),
            "class_check": class_record,
        })
        run.record_corpus(record)
        if problems:
            # a corpus built in this run keeps its marker: no later run reuses it
            run.record_stage("artifact_class_mismatch")
            (run.root / "logs" / "artifact_class.txt").write_text(
                "\n".join(problems) + "\n", encoding="utf-8", newline="\n")
            return StepOutcome("generate", "failed", "; ".join(problems))
        if not reused:
            (target / GENERATING_MARKER).unlink(missing_ok=True)
            run.record_stage("corpus_generation_complete")
        return None

    def _step_generate(self, run: RunDirectory, *, skip: bool,
                       generate_cmd: Sequence[str] | str | None,
                       authorized_by: Path | None, jobs: int = 1,
                       rebuild_corpus: bool = False, corpus_args: Sequence[str] = (),
                       bundle_args: Sequence[str] = (), replace_existing: bool = False,
                       log: Callable[[str], None] = print) -> StepOutcome:
        if skip:
            return StepOutcome("generate", "skipped", "--skip-generate")

        # the registry's plan reads the corpus; an explicit command replaces that plan
        corpus = self.source.corpus if not generate_cmd else None
        self._refuse_unused_args(generate_cmd=generate_cmd, corpus_args=corpus_args,
                                 bundle_args=bundle_args)
        self._refuse_controlled_args(corpus_args, bundle_args)
        try:
            # one generate step per directory at a time, taken before the directories are
            # looked at: what the refusals and plans below see stays what the step acts on. The
            # corpus is locked whether this run builds it or only reads it, so no other run
            # rebuilds it under the bundle reading it (unless no run may write it at all).
            if corpus is not None and self._corpus_writable():
                self._lock(run, Path(self.corpus_dir), "corpus")
            self._lock(run, Path(self.dataset_dir), "bundle")
            if not generate_cmd:
                self._refuse_partial_into_lineage(self.dataset_dir, self.source.bundle_lineage,
                                                  self.source.entrypoint.selection_args,
                                                  bundle_args, "bundle")
            if corpus is not None:
                self._refuse_sample_corpus_into_production()
                if rebuild_corpus or not self._corpus_present():
                    self._refuse_partial_into_lineage(self.corpus_dir, corpus.lineage,
                                                      corpus.selection_args, corpus_args,
                                                      "corpus")
                self._refuse_sample_corpus_build_into_production(
                    rebuild_corpus=rebuild_corpus, corpus_args=corpus_args,
                    bundle_args=bundle_args)
            corpus_plan = (self._plan_corpus(rebuild=rebuild_corpus,
                                             replace_existing=replace_existing)
                           if corpus is not None else None)
            bundle_plan = self._plan_bundle(replace_existing=replace_existing)
            self._decide(run, authorized_by)
            run.record_bundle({
                "index_present": (bundle_plan.target / BUNDLE_INDEX).is_file(),
                # what the directory held at open, in the refusals' words (None: missing or empty)
                "found_at_open": bundle_plan.state,
                # what it held was moved aside to <dir>.replaced-<run> (manifest.json replaced)
                "moved_aside": bundle_plan.replace,
                "generating_marker": GENERATING_MARKER,
                "generating_marker_removed": False,
            })
            try:
                outcome = self._generate(run, corpus_plan, bundle_plan, generate_cmd=generate_cmd,
                                         jobs=jobs, corpus_args=corpus_args,
                                         bundle_args=bundle_args, log=log)
            except BaseException:
                self._settle_failure(run, log)
                raise
            if outcome.status == "ok":
                notes = self._settle_success(run, log)
            else:
                notes = self._settle_failure(run, log)
                if (Path(self.dataset_dir) / GENERATING_MARKER).is_file():
                    log(f"    {GENERATING_MARKER} is left in {self.dataset_dir}: validate, readme, "
                        "register and push refuse the bundle; build it again with "
                        "--replace-existing")
            return StepOutcome(outcome.step, outcome.status, "; ".join([outcome.detail, *notes]))
        finally:
            self._unlock(log)

    def _generate(self, run: RunDirectory, corpus_plan: _DirectoryPlan | None,
                  bundle_plan: _DirectoryPlan, *, generate_cmd, jobs: int,
                  corpus_args: Sequence[str], bundle_args: Sequence[str],
                  log: Callable[[str], None]) -> StepOutcome:
        """The generate step after its refusals and its decision: corpus stage, bundle
        entrypoint, post-steps, class check, limitations, source manifest."""
        corpus = self.source.corpus
        env = self.child_environment()
        extra: list[str] = []
        if corpus_plan is not None:
            failed = self._corpus_stage(run, corpus_plan, corpus_args=corpus_args, env=env, log=log)
            if failed is not None:
                return failed
            extra = [corpus.bundle_arg, str(self.corpus_dir)]
        # after the corpus stage and its class check: a corpus that failed leaves the bundle
        # directory as it was
        self._start_build(run, bundle_plan, log)
        # pipeline.generate, not cli: the runner does not import the CLI, which imports the
        # runner. An explicit command replaces the registry's plan
        # entirely -- no corpus stage, no post-steps, nothing added to it (run_generate ignores
        # extra_args then); the class check below still applies to what it wrote.
        code = generate_mod.run_generate(
            self.source_name, self.dataset_dir, self.data_root, generate_cmd, jobs=jobs,
            extra_args=[*extra, *bundle_args], env=env, registry=self.registry,
            log_dir=run.root / "logs")
        if code != 0:
            return StepOutcome("generate", "failed", f"generator exited {code}")
        run.record_stage("bundle_entrypoint_completed")
        for step in () if generate_cmd else self.source.post_steps:
            argv = step.argv(sys.executable, _REPOSITORY_ROOT, self.dataset_dir, self.data_root)
            code = generate_mod.run_command(
                argv, env=env, log_path=run.root / "logs" / f"post_step.{Path(step.script).stem}.log")
            if code != 0:
                return StepOutcome("generate", "failed",
                                   f"post-step {Path(step.script).name} exited {code}")
            run.record_stage(f"post_step:{Path(step.script).name}")

        policy = self.registry.generation_policy
        problems = generate_mod.bundle_class_problems(
            self.dataset_dir,
            artifact_class=policy.artifact_class if policy else stages_mod.EXPERIMENTAL_ARTIFACT_CLASS,
            distribution_scope=policy.distribution_scope if policy else stages_mod.INTERNAL_ONLY,
        )
        if problems:
            run.record_stage("artifact_class_mismatch")
            (run.root / "logs" / "artifact_class.txt").write_text(
                "\n".join(problems) + "\n", encoding="utf-8", newline="\n")
            return StepOutcome("generate", "failed", "; ".join(problems))

        detail = self._render_limitations(run)
        manifest = provenance_mod.source_manifest(self.dataset_dir, self.data_root)
        manifest["source_files"] = self._trace_source_files(
            corpus_plan, generate_cmd=generate_cmd, corpus_args=corpus_args,
            bundle_args=bundle_args, log=log)
        run.record_source_manifest(manifest)
        run.record_stage("source_manifest_written")
        # the whole step is done: only now is the bundle finished
        self._finish_generating(run)
        traced = manifest["source_files"]
        hashed = (f"; {traced['count']} source files hashed" if isinstance(traced, dict)
                  and isinstance(traced.get("count"), int) else "")
        return StepOutcome(
            "generate", "ok",
            f"{detail}; source_manifest.json {manifest['count']} references over "
            f"{manifest['manifests_scanned']} take manifests{hashed}",
        )

    def _trace_source_files(self, corpus_plan: _DirectoryPlan | None, *, generate_cmd,
                            corpus_args: Sequence[str], bundle_args: Sequence[str],
                            log: Callable[[str], None]) -> dict[str, object]:
        """``source_manifest.json``'s ``source_files``: every source file this run's corpus and
        bundle stages could read, with the SHA-256 of the whole file (``source_trace.trace``).

        The corpora name their sources by name or by a first-MiB digest, and no generator output
        may change, so the full hashes live in the run record. A failure to list or hash is
        recorded, not raised: the bundle is finished by then, and the record says what is
        missing from it."""
        if generate_cmd:
            return {"schema": source_trace_mod.SCHEMA, "status": "not_recorded",
                    "reason": "an explicit generate command replaces the registry's plan, so which "
                              "source files it read is not known"}
        source_root, _ = self.source_root_directory()
        entry, corpus = self.source.entrypoint, self.source.corpus
        try:
            block = source_trace_mod.trace(
                self.source_name, source_root,
                corpus_built=corpus_plan is not None and corpus_plan.action != "reused",
                corpus_args=corpus_args, bundle_args=bundle_args,
                corpus_declared=((*corpus.required_args, *corpus.optional_args, corpus.output_arg)
                                 if corpus is not None else ()),
                bundle_declared=(*entry.required_args, *entry.optional_args, entry.output_arg),
                log=log)
        except (OSError, paths_mod.PathConfigError) as error:
            log(f"    source files could not be traced: {type(error).__name__}: {error}")
            block = {"schema": source_trace_mod.SCHEMA, "status": "unavailable",
                     "reason": f"{type(error).__name__}: {error}"}
        return {**block, **self._located("source_root", source_root)}

    def _render_limitations(self, run: RunDirectory) -> str:
        """Put the bundle's KNOWN_LIMITATIONS.json beside what the generator wrote.

        The validator requires the file of every bundle. Rendering it here, from the one config,
        means a generator never has to know the file exists; a bundle the config does not name
        gets no file, and validate then says so, which is the right place for that finding.
        """
        config = limitations_mod.load_config()
        name = self.dataset_dir.name
        if name not in config.bundles:
            run.record_stage("limitations_block_missing")
            return (f"no {limitations_mod.FILE_NAME}: {name!r} has no block in "
                    f"{limitations_mod.CONFIG_RELPATH}")
        written = limitations_mod.write_rendered(self.dataset_dir, config)
        run.record_stage("limitations_rendered")
        run.record_output_artifact(limitations_mod.FILE_NAME, written)
        return f"{limitations_mod.FILE_NAME} rendered ({len(config.bundles[name])} entries)"

    def _step_validate(self, run: RunDirectory, *, level: str, full: bool,
                       readme_follows: bool = False) -> StepOutcome:
        from soma_synth.datasets import catalog
        from soma_synth.validation import checks
        from soma_synth.validation.runner import validate

        self._refuse_unfinished_bundle("validate")
        ledger_path = self.dataset_dir / "validation_ledger.json"
        ledger = catalog.ValidationLedger(ledger_path)
        # what this run writes into the bundle after the validation whatever its outcome: this
        # step's report and ledger. A fresh bundle has neither yet, and the report must not warn
        # that the bundle lacks what the run is about to write (validate on its own, which writes
        # neither, still does).
        pending = ["VALIDATION_REPORT.json", ledger_path.name]
        report = validate(
            str(self.dataset_dir), level=level, ledger=ledger, full=full, conformance=True,
            pending_files=pending,
        )
        # README.md is written by the readme step, which runs only after a validation that
        # passed: a failed one stops the run (fail-fast) and the bundle is left without it, so
        # the warning stays. Withdrawn only when the report passed and the plan has that step.
        readme_warning = checks.missing_distribution_file("README.md")
        if readme_follows and report.ok and readme_warning in report.findings:
            report.findings.remove(readme_warning)
        ledger.save()
        (self.dataset_dir / "VALIDATION_REPORT.json").write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        run.record_stage("validate")
        run.record_output_artifact("VALIDATION_REPORT.json",
                                   self.dataset_dir / "VALIDATION_REPORT.json")
        # `counts` lives in to_dict(), not on the object; failures/warnings are properties.
        counts = {"FAIL": len(report.failures), "WARN": len(report.warnings)}
        self._quality = {
            "ok": report.ok,
            "counts": counts,
            "takes_checked": report.takes_checked,
            "takes_trusted": report.takes_trusted,
        }
        for finding in report.failures:
            if finding.take_id:
                run.record_exclusion(finding.take_id, finding.message)
        checked, trusted = report.takes_checked, report.takes_trusted
        scope = f"{checked} checked"
        if trusted:
            scope += f", {trusted} trusted from the ledger"
        detail = f"{'PASS' if report.ok else 'FAIL'} {counts} over {scope}"
        return StepOutcome("validate", "ok" if report.ok else "failed", detail)

    def _step_readme(self, run: RunDirectory) -> StepOutcome:
        from soma_synth.docs_gen import readme as readme_mod

        self._refuse_unfinished_bundle("readme")
        written = readme_mod.write_dataset_readme(self.dataset_dir)
        run.record_stage("readme")
        run.record_output_artifact("README.md", written)
        return StepOutcome("readme", "ok", str(Path(written).name))

    def _step_register(self, run: RunDirectory, *, catalog_dir: Path) -> StepOutcome:
        from soma_synth.contracts import qmd_unified8_smpl18_spec as spec
        from soma_synth.datasets import catalog

        self._refuse_unfinished_bundle("register")
        assets = catalog.register_dataset(
            catalog_dir, self.dataset_dir, source=self.source_name,
            spec_version=spec.SPEC_VERSION, data_root=self.data_root,
            # the catalog's journal names the run whose record explains this registration
            registered_by={"tool": "soma-synth run", "run_id": run.root.name},
        )
        run.record_stage("register")
        # what the registration did: a dataset registered again supersedes its earlier entries
        # (catalog.merge_registration)
        done = f" ({assets.summary()})" if isinstance(assets, catalog.Registration) else ""
        return StepOutcome("register", "ok", f"{len(assets)} assets{done}")

    # ------------------------------------------------------------------ run
    def run(
        self,
        *,
        start_at: str = "generate",
        stop_after: str = "register",
        skip_generate: bool = False,
        generate_cmd: Sequence[str] | str | None = None,
        authorized_by: Path | None = None,
        jobs: int = 1,
        level: str = "L4",
        full: bool = False,
        catalog_dir: Path | None = None,
        rebuild_corpus: bool = False,
        corpus_args: Sequence[str] = (),
        bundle_args: Sequence[str] = (),
        replace_existing: bool = False,
        log: Callable[[str], None] = print,
    ) -> RunResult:
        corpus_args = [str(arg) for arg in corpus_args or ()]
        bundle_args = [str(arg) for arg in bundle_args or ()]
        # a relative owner record names a file under the governance root, whatever the working
        # directory; the gates, the recorded intent and the run's copy all read that one file
        authorized_by = gates_mod.resolve_record_path(authorized_by) if authorized_by else None
        catalog = Path(catalog_dir) if catalog_dir else self.default_catalog_dir()
        corpus = self.source.corpus
        policy = self.registry.generation_policy
        steps = _plan(start_at, stop_after)
        generates = "generate" in steps and not skip_generate
        # the registry's plan reads the corpus; it builds it when missing or asked to rebuild
        reads_corpus = generates and not generate_cmd and corpus is not None
        self._refuse_output_locations(
            catalog=catalog if "register" in steps else None,
            reads_corpus=reads_corpus,
            writes_corpus=reads_corpus and (rebuild_corpus or not self._corpus_present()),
        )
        body_model_dir = self.body_model_directory()
        source_root, source_root_from = self.source_root_directory()
        # Paths inside the data root are recorded by their relative form only (``_located``);
        # ``dataset_dir`` and ``data_root`` keep their absolute form.
        config = {
            "dataset_dir": str(self.dataset_dir),
            "data_root": str(self.data_root) if self.data_root else None,
            "level": level,
            "full": full,
            "steps": list(steps),
            "resume_predicate": self.source.resume,
            "parallel": self.source.parallel,
            "emitter": self.source.emitter,
            # Intent, recorded before the gates are consulted: which record the operator
            # offered, whether or not it turns out to authorise anything, and which version of
            # the file it was.
            "authorized_by": str(authorized_by) if authorized_by else None,
            "authorized_by_sha256": (provenance_mod.file_sha256(authorized_by)
                                     if authorized_by else None),
            "jobs": int(jobs),
            # -- ADR-0041: where things are, relative to the data root, so a record read on
            #    another machine still says where under its data root they were
            "dataset_dir_relative": self._relative(self.dataset_dir),
            **self._located("catalog_dir", catalog),
            **self._located("runs_root", self.runs_root),
            "registry": self.registry.display_path(),
            "registry_schema": self.registry.schema,
            # What the run will ask for, written before it asks: None when it takes no decision
            # (no generate step, or --skip-generate). What it got -- including a refusal -- is
            # manifest.json generation_decision.basis; tracing carries both.
            "generation_basis_intended": (
                None if not generates
                else "owner_record" if authorized_by
                else "standing_decision" if policy is not None else "gates"
            ),
            "artifact_class": self._declared_artifact_class(),
            # the source root the generators read: they inherit SOMA_SOURCE_ROOT, else they
            # default to <the data root handed to them>/extracted
            **self._located("source_root", source_root),
            "source_root_from": source_root_from,
            **self._located("body_model_dir", body_model_dir),
            "body_model_dir_from": body_models_mod.set_directory_source(),
            "body_model_file_overrides": {
                name: {**self._located("path", os.environ[name]),
                       "sha256": provenance_mod.file_sha256(os.environ[name])}
                for name in MODEL_FILE_OVERRIDES if os.environ.get(name)
            },
            "generate_cmd": (generate_cmd if isinstance(generate_cmd, str)
                             else list(generate_cmd) if generate_cmd else None),
            # whether the operator allowed replacing what a bundle (with rebuild, a corpus)
            # directory holds: it is moved aside and the new generation built into an empty one
            "replace_existing": bool(replace_existing),
            "corpus": None if corpus is None else {
                "lineage": corpus.lineage,
                **self._located("directory", self.corpus_dir),
                "entrypoint": corpus.script,
                "fingerprint_file": corpus.fingerprint,
                "bundle_arg": corpus.bundle_arg,
                "present_at_open": self._corpus_present(),
                "rebuild": bool(rebuild_corpus),
            },
            "corpus_args": corpus_args,
            "bundle_args": bundle_args,
            "post_steps": [
                {"script": step.script, "args": list(step.args)}
                for step in self.source.post_steps
            ],
        }
        inputs = self.identity_inputs(rebuild_corpus=rebuild_corpus, resolved_config=config)
        identity = run_identity_mod.compute(inputs)
        self._run_identity = identity          # named in the .generating markers and the locks
        root = _unused_run_root(Path(self.runs_root) / run_identity_mod.run_directory_name(inputs))
        # <dir>.replaced-<token> / <dir>.failed-<token>: the identity's first characters, and the
        # -N of a repeated run's record, so two runs never name a side directory alike
        self._run_token = (identity[:RUN_TOKEN_LENGTH]
                           + root.name[len(run_identity_mod.run_directory_name(inputs)):])
        self._builds, self._locks = [], []

        declared_models = self.registry.body_model_set_for(self.source_name)
        record = provenance_mod.ProvenanceRecord(
            source_name=self.source_name,
            spec_id=self.registry.spec_id,
            spec_version=self.registry.spec_version,
            # §10.2 item 6. The whole set, because the generators choose a file per subject;
            # smpl18 owns which file each gender selects (pipeline/body_models.py).
            body_model_set=declared_models,
            body_model_dir=body_model_dir,
            body_model_selection=self.source.body_model_selection,
            data_root=Path(self.data_root),
        )
        run = RunDirectory(
            root=root,
            identity_inputs=inputs,
            provenance=record,
            resolved_config=config,
        ).open()
        log = _logging_into(root / "logs" / RUNNER_LOG, log)

        # Same candidate list the fingerprint uses, imported rather than repeated, so the run
        # identity never hashes a corpus the source_assets record says nothing about.
        # A source with a corpus stage records its corpus in that stage instead, once it exists.
        if corpus is None:
            for key, relative in self.source.inputs.items():
                candidate = Path(self.data_root) / relative
                for name in run_identity_mod.CORPUS_SUMMARY_NAMES:
                    if (candidate / name).is_file():
                        run.record_source_asset(f"{key}/{name}", candidate / name)
                        break
                else:
                    run.record_stage("source_asset_unavailable")
        elif self._corpus_present() and not rebuild_corpus:
            run.record_source_asset(f"{corpus.input_key}/{corpus.fingerprint}",
                                    Path(self.corpus_dir) / corpus.fingerprint)

        result = RunResult(
            run_identity=identity, run_directory=root,
            source_name=self.source_name, dataset_dir=self.dataset_dir,
        )
        self._quality: Mapping[str, object] = {}

        current = "generate"
        try:
            for step in _plan(start_at, stop_after):
                current = step
                log(f"[{step}] ...")
                if step == "generate":
                    outcome = self._step_generate(
                        run, skip=skip_generate, generate_cmd=generate_cmd,
                        authorized_by=Path(authorized_by) if authorized_by else None,
                        jobs=jobs, rebuild_corpus=rebuild_corpus, corpus_args=corpus_args,
                        bundle_args=bundle_args, replace_existing=replace_existing, log=log,
                    )
                elif step == "validate":
                    outcome = self._step_validate(run, level=level, full=full,
                                                  readme_follows="readme" in steps)
                elif step == "readme":
                    outcome = self._step_readme(run)
                else:
                    outcome = self._step_register(run, catalog_dir=catalog)
                result.outcomes.append(outcome)
                log(f"    {outcome.status}" + (f": {outcome.detail}" if outcome.detail else ""))
                if outcome.status == "failed":
                    log("    stopping (fail-fast)")
                    break
        except RunnerRefusal as refusal:
            result.outcomes.append(StepOutcome(current, "refused", str(refusal).splitlines()[0]))
            (root / "logs" / "refusal.txt").write_text(str(refusal), encoding="utf-8", newline="\n")
            log(str(refusal))
        except Exception as error:  # noqa: BLE001 - the record must survive the failure
            # A run that dies mid-way still has to leave the record §10.1 requires; a traceback
            # on the terminal and nothing on disk is the situation this package exists to end.
            import traceback

            (root / "logs" / "error.txt").write_text(
                traceback.format_exc(), encoding="utf-8", newline="\n"
            )
            result.outcomes.append(
                StepOutcome("run", "failed", f"{type(error).__name__}: {error}")
            )
            log(f"    unexpected failure recorded to {root / 'logs' / 'error.txt'}")
        finally:
            run.close(
                takes_written=int(self._quality.get("takes_checked", 0) or 0),
                takes_trusted=int(self._quality.get("takes_trusted", 0) or 0),
                quality=dict(self._quality),
            )

        return result
