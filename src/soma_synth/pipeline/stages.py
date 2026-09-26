"""The canonical generation stages, and the registry that says how each source runs them.

The first seven names are ``PIPELINE_GOVERNANCE.md`` §5.1 verbatim -- that document requires
adapters to separate them, which the generators do not: they load each other by file path and
have no module boundaries to hang a stage on. The remaining eight are the synthesis and emission
this lineage adds after them.

Naming the stages as values is what makes "did this source pass validate_native" a question the
code can answer.

Backed by ``configs/datasets/source_pipelines_v2.yaml`` (the default registry, ADR-0041), which
is ``source_pipelines_v1.yaml`` plus the standing generation policy, each source's bundle lineage
and artifact class, each entrypoint's selection flags, the retarget-corpus stage of the three
sources that have one and PRISM's post-step. v1 still loads: a used config is never edited in
place, and a run record names the registry it ran under. See
``docs/guides/GENERATION_PIPELINE_STANDARD.md``.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from soma_synth.pipeline import paths as paths_mod

SCHEMA_V1 = "source_pipelines_v1"
SCHEMA_V2 = "source_pipelines_v2"
#: The current schema; :func:`load` reads both.
SCHEMA = SCHEMA_V2
SCHEMAS = (SCHEMA_V1, SCHEMA_V2)
REGISTRY_V1_RELPATH = "configs/datasets/source_pipelines_v1.yaml"
REGISTRY_V2_RELPATH = "configs/datasets/source_pipelines_v2.yaml"
REGISTRY_RELPATH = REGISTRY_V2_RELPATH

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = _REPOSITORY_ROOT / REGISTRY_RELPATH
REGISTRY_V1_PATH = _REPOSITORY_ROOT / REGISTRY_V1_RELPATH

#: The only artifact class the standing decision (ADR-0041) lets a run make, and its scope.
EXPERIMENTAL_ARTIFACT_CLASS = "experimental_non_candidate"
INTERNAL_ONLY = "internal_only"

#: What a post-step's arguments may name; anything else in braces is refused at load.
_POST_STEP_PLACEHOLDERS = ("dataset_dir", "data_root")
_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")


class PipelineError(ValueError):
    """The pipeline registry is malformed, or was asked something it cannot answer."""


class Stage(str, Enum):
    """Ordered. ``list(Stage)`` is the canonical sequence."""

    # -- PIPELINE_GOVERNANCE 5.1, verbatim
    REGISTER_ASSET = "register_asset"
    INSPECT_NATIVE = "inspect_native"
    VALIDATE_LICENSE = "validate_license"
    LOAD_NATIVE = "load_native"
    VALIDATE_NATIVE = "validate_native"
    CONVERT_INTERMEDIATE = "convert_intermediate"
    EMIT_SOURCE_QC_AND_PROVENANCE = "emit_source_qc_and_provenance"
    # -- what this lineage adds after them
    FIT_SHAPE = "fit_shape"
    BUILD_LARGE = "build_large"
    BUILD_ANTHRO = "build_anthro"
    SYNTHESISE_SMALL = "synthesise_small"
    RESAMPLE_100HZ = "resample_100hz"
    EMIT_TAKE = "emit_take"
    EMIT_BUNDLE = "emit_bundle"
    CLOSE_RUN = "close_run"


#: The seven the sealed document names. Kept separate so a change to this lineage's own stages
#: cannot silently edit what §5.1 requires.
GOVERNANCE_STAGES: tuple[Stage, ...] = tuple(list(Stage)[:7])

#: The one stage whose path genuinely differs by source; every other stage does the same work
#: everywhere. Which path is taken follows from ``small_mode``, never from the source name.
SOURCE_VARIANT_STAGE = Stage.SYNTHESISE_SMALL


@dataclass(frozen=True)
class Entrypoint:
    script: str
    required_args: tuple[str, ...]
    optional_args: tuple[str, ...]
    #: The generators disagree about what to be told: the bundle directory, its parent, or the
    #: data root. Declared per source so the caller does not branch on the source name.
    output_arg: str
    output_arg_kind: str
    #: How a request for N workers reaches this generator, if it can take one: `jobs` names the
    #: flag of a generator with its own pool; `shard` and `merge` name the flags of one that is
    #: run as N processes over disjoint slices and then joined. Empty means it runs alone.
    parallel_args: Mapping[str, str] = None  # type: ignore[assignment]
    #: v2: the flags that select part of the source (subjects, trials, a limit). A run given one
    #: of them writes an INDEX over the selection alone, so the runner refuses them when the
    #: output is a production lineage directory; any other flag (``--force``) is not refused.
    selection_args: tuple[str, ...] = ()

    def argv(self, python: str, repository_root: Path, dataset_dir: Path, data_root: Path | None,
             extra_args: Sequence[str] = ()):
        """The command line that generates this source, with the output argument filled in.

        ``extra_args`` follow the output argument: the corpus the runner built (``--paired
        <corpus>``) and any arguments the operator passed for a sample run.
        """
        if self.output_arg_kind == "dataset_dir":
            value = dataset_dir
        elif self.output_arg_kind == "dataset_parent":
            value = dataset_dir.parent
        elif self.output_arg_kind == "data_root":
            value = data_root if data_root is not None else dataset_dir.parent
        else:
            raise PipelineError(f"unknown output_arg_kind: {self.output_arg_kind!r}")
        return [python, str(repository_root / self.script), self.output_arg, str(value),
                *(str(arg) for arg in extra_args)]

    def plan(self, python: str, repository_root: Path, dataset_dir: Path, data_root: Path | None,
             jobs: int = 1, extra_args: Sequence[str] = ()) -> tuple[list[list[str]], list[str] | None]:
        """(concurrent command lines, follow-up command line) for `jobs` workers.

        A generator without parallel_args, or asked for one worker, runs once. A `jobs` generator
        runs once with its flag. A `shard` generator runs `jobs` times concurrently with
        `--shard i/N` and then once more with `--merge`, which is the follow-up. ``extra_args``
        are on every one of them.
        """
        base = self.argv(python, repository_root, dataset_dir, data_root, extra_args)
        args = dict(self.parallel_args or {})
        if jobs <= 1 or not args:
            return [base], None
        if "shard" in args and "merge" in args:
            shards = [base + [args["shard"], f"{i}/{jobs}"] for i in range(jobs)]
            return shards, base + [args["merge"]]
        if "jobs" in args:
            return [base + [args["jobs"], str(jobs)]], None
        raise PipelineError(
            f"parallel_args {sorted(args)} name neither a jobs flag nor a shard/merge pair"
        )


@dataclass(frozen=True)
class Corpus:
    """The retarget corpus a bundle is built from, and the entrypoint that builds it (v2).

    hknu, gaitex and addbiomechanics convert their source to an SMPL-24 corpus first and build
    the bundle from that corpus. The runner builds the corpus from the extracted sources when it
    is missing (ADR-0041), into its own lineage directory, and hands it to the bundle entrypoint
    through ``bundle_arg``.
    """

    script: str
    #: The lineage directory the corpus lives in under the lineage container.
    lineage: str
    #: The flag the corpus entrypoint writes through; its value is the corpus directory.
    output_arg: str
    #: The file whose sha256 stands for the whole corpus (``run_identity.CORPUS_SUMMARY_NAMES``).
    fingerprint: str
    #: The bundle entrypoint's flag that names the corpus to read.
    bundle_arg: str
    #: The ``inputs`` key of the source that names this corpus (checked at load).
    input_key: str
    required_args: tuple[str, ...] = ()
    optional_args: tuple[str, ...] = ()
    path_loads: tuple[str, ...] = ()
    library_imports: tuple[str, ...] = ()
    #: The corpus entrypoint's flags that select part of the source; see ``Entrypoint``.
    selection_args: tuple[str, ...] = ()

    def argv(self, python: str, repository_root: Path, corpus_dir: Path,
             extra_args: Sequence[str] = ()) -> list[str]:
        """The command line that builds the corpus into ``corpus_dir``."""
        return [python, str(repository_root / self.script), self.output_arg, str(corpus_dir),
                *(str(arg) for arg in extra_args)]


@dataclass(frozen=True)
class PostStep:
    """A script the runner runs over the finished bundle, before the limitations are rendered.

    PRISM's ``synthesize_insole_heading.py`` is the one there is: it writes
    ``heading_synth_reference.npz`` beside every take.
    """

    script: str
    #: Arguments after the script; ``{dataset_dir}`` and ``{data_root}`` are filled in.
    args: tuple[str, ...]
    description: str = ""

    def argv(self, python: str, repository_root: Path, dataset_dir: Path,
             data_root: Path | None) -> list[str]:
        values = {"dataset_dir": str(dataset_dir),
                  "data_root": str(data_root) if data_root is not None else ""}
        return [python, str(repository_root / self.script),
                *(arg.format(**values) for arg in self.args)]


@dataclass(frozen=True)
class GenerationPolicy:
    """The standing decision a run generates under when no owner record is handed in (v2).

    ADR-0041 records the decision (research/decisions/2026-09-25_source_distribution_and_
    pipeline_split_decision.md section 4) that internal users generate
    ``experimental_non_candidate`` / ``internal_only`` bundles on their own machines without a
    record per run. The registry names it so the runner can copy it into every run record; the
    class, scope and ``releases_holds: false`` are checked at load, because a policy that said
    anything else would not be this decision.
    """

    basis: str
    adr: str
    adr_path: str
    decision_record: str
    decision_section: str
    artifact_class: str
    distribution_scope: str
    releases_holds: bool
    description: str = ""

    def as_json(self) -> dict[str, object]:
        return {
            "basis": self.basis,
            "adr": self.adr,
            "adr_path": self.adr_path,
            "decision_record": self.decision_record,
            "decision_section": self.decision_section,
            "artifact_class": self.artifact_class,
            "distribution_scope": self.distribution_scope,
            "releases_holds": self.releases_holds,
        }


@dataclass(frozen=True)
class SourcePipeline:
    """How one source is produced, as it is today rather than as it should be."""

    source_name: str
    small_mode: str
    entrypoint: Entrypoint
    emitter: str
    resume: str
    parallel: str
    #: The model is chosen per subject, so this names the SET and how one is picked out of it --
    #: never a single file, which would misdescribe every generator here.
    body_model_set: str
    body_model_selection: str
    #: The smpl18 profile describing this source's conversion to SMPL-24: a shipped profile name
    #: or a repository-relative path. Empty until a source has one.
    smpl18_profile: str
    stochastic: bool
    path_loads: tuple[str, ...]
    library_imports: tuple[str, ...]
    inputs: Mapping[str, str]
    configs: Mapping[str, str]
    reads_env: tuple[str, ...]
    # -- v2 (ADR-0041); empty for a v1 registry
    #: The class this source's bundles are made as. The runner makes experimental_non_candidate
    #: only; a source declaring anything else is refused.
    artifact_class: str = ""
    #: The lineage directory the bundle lives in (``paths.lineage_dir``); the runner's default
    #: output when no bundle directory is given.
    bundle_lineage: str = ""
    corpus: Corpus | None = None
    post_steps: tuple[PostStep, ...] = field(default=())

    def uses_shared_emitter(self) -> bool:
        return self.emitter != "inline"


@dataclass(frozen=True)
class BodyModelSet:
    """Where a named body-model set lives, and who resolves a gender to a file inside it."""

    name: str
    relative_path: str
    resolver: str

    def directory(self, data_root: Path | str | None) -> Path | None:
        """The set's directory under a data root, or None when the run has no data root."""
        return None if data_root is None else Path(data_root) / self.relative_path


@dataclass(frozen=True)
class PipelineRegistry:
    spec_id: str
    spec_version: str
    canonical_stages: tuple[Stage, ...]
    resume_predicates: Mapping[str, str]
    sources: Mapping[str, SourcePipeline]
    body_model_sets: Mapping[str, BodyModelSet]
    path: Path
    schema: str = SCHEMA_V1
    #: v2 only: the standing decision a run without an owner record generates under.
    generation_policy: GenerationPolicy | None = None

    def display_path(self) -> str:
        """The registry file relative to the repository when it is inside it, for a run record."""
        try:
            return self.path.resolve().relative_to(_REPOSITORY_ROOT).as_posix()
        except ValueError:
            return self.path.as_posix()

    def body_model_set_for(self, source_name: str) -> BodyModelSet | None:
        """The set a source generates from, or None when it names none."""
        name = self.for_source(source_name).body_model_set
        if not name:
            return None
        try:
            return self.body_model_sets[name]
        except KeyError:
            known = ", ".join(sorted(self.body_model_sets)) or "none"
            raise PipelineError(
                f"{source_name}: body_model_set {name!r} is not declared in {self.path.name}; "
                f"declared: {known}"
            ) from None

    def for_source(self, source_name: str) -> SourcePipeline:
        try:
            return self.sources[source_name]
        except KeyError:
            known = ", ".join(sorted(self.sources))
            raise PipelineError(
                f"source {source_name!r} is not declared in {self.path.name}; known: {known}"
            ) from None


def _tuple(block: Mapping, key: str) -> tuple[str, ...]:
    value = block.get(key) or []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PipelineError(f"{key!r} must be a list")
    return tuple(str(v) for v in value)


def _flag(name: str, where: str, value: object) -> str:
    text = str(value or "")
    if not text.startswith("-"):
        raise PipelineError(f"{name}: {where} must be a command-line flag, got {value!r}")
    return text


def _selection_args(name: str, where: str, block: Mapping) -> tuple[str, ...]:
    """A v2 ``selection_args`` list: required (it may be empty), flags only, and each one an
    argument the same block declares. A missing list would silently turn the runner's
    partial-run refusal off for that entrypoint, so it is an error rather than a default."""
    if "selection_args" not in block:
        raise PipelineError(
            f"{name}: {where}.selection_args is required in {SCHEMA_V2} (an empty list when the "
            "entrypoint has no flag that selects part of the source)"
        )
    flags = _tuple(block, "selection_args")
    declared = (*_tuple(block, "required_args"), *_tuple(block, "optional_args"))
    for flag in flags:
        _flag(name, f"{where}.selection_args", flag)
        if flag not in declared:
            raise PipelineError(
                f"{name}: {where}.selection_args names {flag!r}, which is not one of the "
                f"entrypoint's arguments ({where}.required_args / optional_args)"
            )
    return flags


def _lineage(name: str, where: str, value: object) -> str:
    text = str(value or "")
    try:
        paths_mod.lineage_dir(text, root=Path("."))
    except paths_mod.PathConfigError:
        raise PipelineError(f"{name}: {where} {value!r} is not a lineage directory name") from None
    return text


def _parse_corpus(name: str, raw: Mapping, inputs: Mapping[str, str],
                  entry_args: Sequence[str]) -> Corpus:
    script = str(raw.get("script") or "")
    if not script:
        raise PipelineError(f"{name}: corpus.script is required")
    lineage = _lineage(name, "corpus.lineage", raw.get("lineage"))
    fingerprint = str(raw.get("fingerprint") or "")
    if not fingerprint or "/" in fingerprint or "\\" in fingerprint:
        raise PipelineError(
            f"{name}: corpus.fingerprint must name the corpus's summary file (SUMMARY.json, "
            f"_run.json), got {fingerprint!r}"
        )
    bundle_arg = _flag(name, "corpus.bundle_arg", raw.get("bundle_arg"))
    if bundle_arg not in entry_args:
        raise PipelineError(
            f"{name}: corpus.bundle_arg {bundle_arg!r} is not an argument of the bundle "
            "entrypoint (entrypoint.required_args / optional_args)"
        )
    # The corpus must be the input the source already names, so the run identity (which
    # fingerprints `inputs`) and the corpus stage cannot describe two different directories.
    wanted = f"{paths_mod.POC_DEMO}/{lineage}"
    keys = [key for key, value in inputs.items() if str(value) == wanted]
    if len(keys) != 1:
        raise PipelineError(
            f"{name}: corpus.lineage {lineage!r} must be named by exactly one entry of inputs "
            f"({wanted!r}); inputs: {dict(inputs)}"
        )
    return Corpus(
        script=script,
        lineage=lineage,
        output_arg=_flag(name, "corpus.output_arg", raw.get("output_arg")),
        fingerprint=fingerprint,
        bundle_arg=bundle_arg,
        input_key=keys[0],
        required_args=_tuple(raw, "required_args"),
        optional_args=_tuple(raw, "optional_args"),
        path_loads=_tuple(raw, "path_loads"),
        library_imports=_tuple(raw, "library_imports"),
        selection_args=_selection_args(name, "corpus", raw),
    )


def _parse_post_steps(name: str, raw: object) -> tuple[PostStep, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise PipelineError(f"{name}: post_steps must be a list")
    steps = []
    for position, block in enumerate(raw):
        block = block or {}
        script = str(block.get("script") or "")
        if not script:
            raise PipelineError(f"{name}: post_steps[{position}].script is required")
        args = _tuple(block, "args")
        for arg in args:
            unknown = [p for p in _PLACEHOLDER.findall(arg) if p not in _POST_STEP_PLACEHOLDERS]
            if unknown:
                allowed = ", ".join("{" + p + "}" for p in _POST_STEP_PLACEHOLDERS)
                raise PipelineError(
                    f"{name}: post_steps[{position}] names {unknown}; a post-step argument may "
                    f"name only {allowed}"
                )
        steps.append(PostStep(script=script, args=args,
                              description=str(block.get("description") or "")))
    return tuple(steps)


def _parse_policy(raw: object) -> GenerationPolicy:
    if not isinstance(raw, Mapping):
        raise PipelineError(
            f"{SCHEMA_V2} requires a generation_policy block: the decision a run without an "
            "owner record generates under (ADR-0041)"
        )
    missing = [key for key in ("basis", "adr", "adr_path", "decision_record", "decision_section",
                               "artifact_class", "distribution_scope", "releases_holds")
               if key not in raw or raw.get(key) in (None, "")]
    if missing:
        raise PipelineError(f"generation_policy is missing {', '.join(missing)}")
    if raw["releases_holds"] is not False:
        raise PipelineError(
            "generation_policy.releases_holds must be false: the standing decision releases no "
            "hold (ADR-0041; the parent project's hold transitions)"
        )
    if raw["artifact_class"] != EXPERIMENTAL_ARTIFACT_CLASS:
        raise PipelineError(
            f"generation_policy.artifact_class is {raw['artifact_class']!r}; the standing "
            f"decision covers {EXPERIMENTAL_ARTIFACT_CLASS!r} only -- canonical and "
            "internal-candidate generation stay under the holds"
        )
    if raw["distribution_scope"] != INTERNAL_ONLY:
        raise PipelineError(
            f"generation_policy.distribution_scope is {raw['distribution_scope']!r}; the "
            f"standing decision covers {INTERNAL_ONLY!r} only"
        )
    return GenerationPolicy(
        basis=str(raw["basis"]), adr=str(raw["adr"]), adr_path=str(raw["adr_path"]),
        decision_record=str(raw["decision_record"]),
        decision_section=str(raw["decision_section"]),
        artifact_class=str(raw["artifact_class"]),
        distribution_scope=str(raw["distribution_scope"]),
        releases_holds=False,
        description=str(raw.get("description") or ""),
    )


def _parse_source(name: str, raw: Mapping, resume_predicates: Mapping[str, str],
                  schema: str = SCHEMA_V1) -> SourcePipeline:
    entry = raw.get("entrypoint") or {}
    script = str(entry.get("script") or "")
    if not script:
        raise PipelineError(f"{name}: entrypoint.script is required")
    output_arg = str(entry.get("output_arg") or "")
    output_arg_kind = str(entry.get("output_arg_kind") or "")
    if not output_arg or not output_arg_kind:
        raise PipelineError(
            f"{name}: entrypoint.output_arg and output_arg_kind are required; without them the "
            "caller has to know which flag this generator writes through, which is the hardcoding "
            "this registry exists to remove"
        )

    resume = str(raw.get("resume") or "")
    if resume not in resume_predicates:
        raise PipelineError(
            f"{name}: resume {resume!r} is not one of the declared predicates "
            f"({', '.join(sorted(resume_predicates))})"
        )

    parallel_args = dict(entry.get("parallel_args") or {})
    if any(not isinstance(v, str) or not v.startswith("-") for v in parallel_args.values()):
        raise PipelineError(f"{name}: parallel_args values must be command-line flags")

    inputs = dict(raw.get("inputs") or {})
    v2: dict[str, object] = {}
    selection: tuple[str, ...] = ()
    if schema == SCHEMA_V2:
        selection = _selection_args(name, "entrypoint", entry)
        artifact_class = str(raw.get("artifact_class") or "")
        if not artifact_class:
            raise PipelineError(f"{name}: artifact_class is required in {SCHEMA_V2}")
        entry_args = (*_tuple(entry, "required_args"), *_tuple(entry, "optional_args"))
        corpus_block = raw.get("corpus")
        v2 = {
            "artifact_class": artifact_class,
            "bundle_lineage": _lineage(name, "bundle_lineage", raw.get("bundle_lineage")),
            "corpus": (_parse_corpus(name, corpus_block, inputs, entry_args)
                       if corpus_block else None),
            "post_steps": _parse_post_steps(name, raw.get("post_steps")),
        }
    else:
        unexpected = [key for key in ("artifact_class", "bundle_lineage", "corpus", "post_steps")
                      if key in raw]
        unexpected += [f"entrypoint.{key}" for key in ("selection_args",) if key in entry]
        if unexpected:
            raise PipelineError(f"{name}: {', '.join(unexpected)} need schema {SCHEMA_V2}")

    return SourcePipeline(
        source_name=name,
        small_mode=str(raw.get("small_mode") or ""),
        entrypoint=Entrypoint(
            script=script,
            required_args=_tuple(entry, "required_args"),
            optional_args=_tuple(entry, "optional_args"),
            output_arg=output_arg,
            output_arg_kind=output_arg_kind,
            parallel_args=parallel_args,
            selection_args=selection,
        ),
        emitter=str(raw.get("emitter") or "inline"),
        resume=resume,
        parallel=str(raw.get("parallel") or "none"),
        body_model_set=str(raw.get("body_model_set") or ""),
        body_model_selection=str(raw.get("body_model_selection") or ""),
        smpl18_profile=str(raw.get("smpl18_profile") or ""),
        stochastic=bool(raw.get("stochastic", False)),
        path_loads=_tuple(raw, "path_loads"),
        library_imports=_tuple(raw, "library_imports"),
        inputs=inputs,
        configs=dict(raw.get("configs") or {}),
        reads_env=_tuple(raw, "reads_env"),
        **v2,  # type: ignore[arg-type]
    )


def load(path: Path | str | None = None) -> PipelineRegistry:
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if not registry_path.exists():
        raise PipelineError(f"source pipeline registry not found: {registry_path}")

    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    schema = raw.get("schema")
    if schema not in SCHEMAS:
        raise PipelineError(f"expected schema {' or '.join(map(repr, SCHEMAS))}, got {schema!r}")

    declared = _tuple(raw, "canonical_stages")
    expected = tuple(stage.value for stage in Stage)
    if declared != expected:
        raise PipelineError(
            "canonical_stages in the registry disagree with the Stage enum.\n"
            f"  registry: {declared}\n  code:     {expected}"
        )

    predicates = dict(raw.get("resume_predicates") or {})
    if not predicates:
        raise PipelineError("resume_predicates is empty")

    sources = {
        name: _parse_source(name, block or {}, predicates, schema)
        for name, block in (raw.get("sources") or {}).items()
    }
    if not sources:
        raise PipelineError("registry declares no sources")

    policy = None
    if schema == SCHEMA_V2:
        policy = _parse_policy(raw.get("generation_policy"))
    elif "generation_policy" in raw:
        raise PipelineError(f"generation_policy needs schema {SCHEMA_V2}")

    body_model_sets = {}
    for name, block in (raw.get("body_model_sets") or {}).items():
        block = block or {}
        relative_path = str(block.get("relative_path") or "")
        resolver = str(block.get("resolver") or "")
        if not relative_path or not resolver:
            raise PipelineError(
                f"body_model_sets.{name}: relative_path and resolver are required; without them "
                "the runner cannot say which files a run generated from, which §10.2 item 6 asks"
            )
        if Path(relative_path).is_absolute():
            raise PipelineError(
                f"body_model_sets.{name}: relative_path must be relative to the data root"
            )
        body_model_sets[name] = BodyModelSet(name=name, relative_path=relative_path, resolver=resolver)

    return PipelineRegistry(
        spec_id=str(raw.get("spec_id") or ""),
        spec_version=str(raw.get("spec_version") or ""),
        canonical_stages=tuple(Stage(value) for value in declared),
        resume_predicates=predicates,
        sources=sources,
        body_model_sets=body_model_sets,
        path=registry_path,
        schema=str(schema),
        generation_policy=policy,
    )


@functools.lru_cache(maxsize=1)
def default_registry() -> PipelineRegistry:
    return load()
