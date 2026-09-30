"""soma-synth CLI: the generation pipeline's commands.

Subcommands: ``run`` (the runner: generate -> validate -> readme -> register with the run record
PIPELINE_GOVERNANCE 10.1 and ADR-0041 require), ``validate`` (L0-L4 + ledger + report),
``readme`` (field-by-field dataset docs), ``register`` (append a dataset's assets to the experimental
catalog), ``status`` (summarize a catalog/ledger), ``pipeline`` (validate -> readme -> register an
existing bundle in order, fail-fast, guiding the next step after each; its generate step is
refused, because a generation outside ``run`` leaves no record of its decision, sources or code,
and it refuses the output locations ``run`` refuses and a bundle whose generation did not finish),
``pipeline-doc`` (render the per-source table of the standard from the registry),
``provenance`` (report or retrofit PIPELINE_GOVERNANCE 10.1/10.2 coverage of a bundle),
``hash-sources`` (write a SHA256SUMS list of the source folders; reads only) and
``stage-sources`` (copy the listed source files that are missing onto the local source root,
verified against the list; never deletes or overwrites; ``--verify-only`` checks without copying).

The WearGait estimated pipeline and the governed source-parsing operations are not here; they stay in
the parent project, which consumes this package. Bundles are not pushed from here either.

``validate``, ``readme``, ``register`` and ``pipeline`` refuse (exit 3, naming the command that
finishes it) a directory a run moved aside or a failed replacement's output (``*.replaced-*``,
``*.failed-*``) and a bundle whose generation did not finish (``.generating``); ``register`` and
``pipeline`` also refuse a directory that is another lineage the registry declares. They refuse
the locations ``run`` refuses too (exit 3, ``paths.check_bundle_dir``): a bundle folder inside an
evidence folder (``_superseded``, ``_runs``, ``_manifest_backfill``) of any data root, or inside
``extracted`` / ``raw_archives`` -- ``validate`` when it writes its report or ledger into the
bundle, ``readme`` always, ``register`` because kept evidence is not live data -- and a report,
ledger or catalog written where no record goes.

``register`` and ``pipeline`` write asset paths relative to the data root: ``--data-root``, else
``SOMA_DATA_ROOT`` (exit 2 when neither is set), as for ``run``; a bundle that is not inside it is
refused (exit 2), since its paths would match none of the dataset's earlier entries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from soma_synth.contracts import qmd_unified8_smpl18_spec as spec
from soma_synth.datasets import catalog

# The generator launch has one copy, in pipeline.generate (ADR-0041); these names are
# re-exported because callers and tests reach them as cli.<name>.
from soma_synth.pipeline.generate import (  # noqa: F401
    _default_generate_cmd,
    _refuse_paired_weargait,
    bundle_class_problems,
    plan_generate_commands,
    run_generate,
)
from soma_synth.validation.runner import validate

# The four lifecycle steps, in order. ``pipeline`` walks a contiguous slice of this list.
STEPS = ("generate", "validate", "readme", "register")

#: The exit code of a run the runner refused (``run``), or of a ``pipeline`` asked to generate.
#: 1 is a step that failed and 2 a location or registry that could not be resolved.
EXIT_REFUSED = 3


def _bundle_state_refusal(dataset_dirs, step: str, source: str | None = None) -> str | None:
    """The runner's refusal of a bundle that is not one to ``step`` yet (or at all): an aside
    directory or a failed replacement's output, or a bundle whose generation did not finish
    (``runner.unfinished_bundle_refusal``, which names the command that builds it). None when
    every directory may be taken."""
    from soma_synth.pipeline.runner import unfinished_bundle_refusal

    for dataset_dir in dataset_dirs:
        problem = unfinished_bundle_refusal(dataset_dir, step=step, source=source)
        if problem:
            return problem
    return None


def _other_lineage_refusal(dataset_dir, source: str) -> str | None:
    """Why ``dataset_dir`` is not a bundle of ``source``: another lineage the registry declares
    (another source's bundle lineage, or a corpus lineage), under any
    ``runs/experimental_generation_poc_demo``. None when it is not, or the source is unknown."""
    from soma_synth.pipeline import stages as stages_mod
    from soma_synth.pipeline.runner import other_lineage_refusal

    registry = stages_mod.default_registry()
    try:
        own = registry.for_source(source).bundle_lineage
    except stages_mod.PipelineError:
        return None
    return other_lineage_refusal(dataset_dir, own, "bundle", registry)


def _written_into(dataset_dir, paths) -> list[str]:
    """The names of the files among ``paths`` that go straight into the bundle folder: what the
    command writes there after validating, which the validation is told not to report missing."""
    folder = Path(dataset_dir).resolve()
    return [Path(path).name for path in paths if path and Path(path).resolve().parent == folder]


def _outside_data_root(dataset_dir, data_root) -> str | None:
    """Why ``dataset_dir`` cannot be registered with paths relative to ``data_root``, or None.

    The catalog keeps asset paths relative to the data root, and a registration finds the
    dataset's earlier entries by them. A bundle outside the root would be stored as ``../...``
    (or not at all on another drive) and match none of them, so its assets would be appended a
    second time beside the live ones."""
    bundle, root = Path(dataset_dir).resolve(), Path(data_root).resolve()
    if root in bundle.parents:
        return None
    return (f"{dataset_dir} is not inside the data root {data_root}: the catalog keeps asset paths "
            "relative to the data root, so a registration needs the root that holds the bundle "
            "(--data-root, default $SOMA_DATA_ROOT)")


def _refused_location(command: str, plan, dataset_dir, data_root=None, **where) -> int | None:
    """Print why ``command`` may not write where it would (:func:`_location_refusal`) and return
    its exit code; None when nothing is refused."""
    refusal = _location_refusal(plan, Path(dataset_dir),
                                Path(data_root) if data_root else None, **where)
    if refusal is None:
        return None
    code, message = refusal
    print(f"{command}: {message}", file=sys.stderr)
    return code


def _cmd_validate(args) -> int:
    refusal = _bundle_state_refusal([args.dataset_dir], "validate")
    if refusal:
        print(refusal, file=sys.stderr)
        return EXIT_REFUSED
    # the report and the ledger are the only files validate writes; without them it only reads
    code = _refused_location("validate", ("validate",), args.dataset_dir,
                             validation_files=[p for p in (args.report, args.ledger) if p])
    if code is not None:
        return code
    ledger = catalog.ValidationLedger(args.ledger) if args.ledger else None
    rep = validate(
        args.dataset_dir,
        level=args.level,
        sample=args.sample,
        ledger=ledger,
        full=args.full,
        conformance=args.conformance,
        pending_files=_written_into(args.dataset_dir, (args.report, args.ledger)),
    )
    if ledger is not None:
        ledger.save()
    print(rep.summary())
    if args.report:
        Path(args.report).write_text(
            json.dumps(rep.to_dict(), indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    return 0 if rep.ok else 1


def _cmd_run(args) -> int:
    """Run the pipeline through the runner, leaving the section 10.1 record behind."""
    if _refuse_paired_weargait(args.source):
        return 2
    from soma_synth.pipeline.paths import PathConfigError
    from soma_synth.pipeline.runner import PipelineRunner, RunnerRefusal
    from soma_synth.pipeline.stages import PipelineError

    try:
        runner = PipelineRunner(
            source_name=args.source,
            dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
            data_root=Path(args.data_root) if args.data_root else None,
            runs_root=Path(args.runs_root) if args.runs_root else None,
            corpus_dir=Path(args.corpus) if args.corpus else None,
        )
        result = runner.run(
            start_at=args.start_at,
            stop_after=args.stop_after,
            skip_generate=args.skip_generate,
            authorized_by=Path(args.authorized_by) if args.authorized_by else None,
            jobs=args.jobs,
            level=args.level,
            full=args.full,
            catalog_dir=Path(args.catalog) if args.catalog else None,
            rebuild_corpus=args.rebuild_corpus,
            corpus_args=args.corpus_arg,
            bundle_args=args.bundle_arg,
            replace_existing=args.replace_existing,
        )
    except RunnerRefusal as refusal:
        # refused before its record could be opened (an output in a read-only or evidence
        # folder; a synchronised folder draws only a warning)
        print(refusal, file=sys.stderr)
        return EXIT_REFUSED
    except (PathConfigError, PipelineError) as error:
        # a location or registry the run cannot resolve, before its record is opened
        print(error, file=sys.stderr)
        return 2
    print()
    print(result.summary())
    # a refusal is not a success: a script reading the exit code must be able to tell
    if getattr(result, "refused", False):
        return EXIT_REFUSED
    return 0 if result.ok else 1


def _cmd_provenance(args) -> int:
    """Report §10.2 coverage for a bundle, and optionally write the retrofit sidecar."""
    from soma_synth.pipeline import audit

    report = audit.audit_bundle(args.dataset_dir)
    present, total = report.provenance_coverage
    print(f"provenance[{Path(args.dataset_dir).name}] source={report.source_name or '?'}")
    print(f"  PIPELINE_GOVERNANCE 10.2 items: {present}/{total}")
    for item in report.missing_items:
        note = audit.NOT_RECOVERABLE.get(item)
        print(f"    missing: {item}" + (f"  ({note})" if note else ""))
    missing_run = report.missing_run_artifacts
    print(f"  PIPELINE_GOVERNANCE 10.1 run artifacts missing: "
          f"{len(missing_run)}/{len(missing_run) + (7 - len(missing_run))}")
    for name in missing_run:
        print(f"    missing: {name}")

    if args.retrofit:
        audit.retrofit_provenance(args.dataset_dir, body_model_path=args.body_model)
        print(f"  wrote {Path(args.dataset_dir) / audit.SIDECAR_NAME}")
    return 0


def _cmd_pipeline_doc(args) -> int:
    """Regenerate the standard's per-source table from the registry."""
    from soma_synth.pipeline import render

    target = Path(args.document) if args.document else (
        Path(__file__).resolve().parents[2] / "docs/guides/GENERATION_PIPELINE_STANDARD.md"
    )
    if args.render:
        print(f"wrote {render.write_into(target)}")
    else:
        print(render.render_table())
    return 0


def _cmd_status(args) -> int:
    if args.catalog:
        acp = Path(args.catalog) / "asset_catalog.json"
        if acp.exists():
            ac = json.loads(acp.read_text(encoding="utf-8"))
            assets = ac.get("assets", [])
            by: dict[tuple, int] = {}
            for a in assets:
                by[(a.get("source"), a.get("spec_version"))] = by.get(
                    (a.get("source"), a.get("spec_version")), 0
                ) + 1
            print(f"asset_catalog: {len(assets)} assets")
            for (src, ver), n in sorted(by.items(), key=lambda kv: str(kv[0])):
                print(f"  {src} {ver}: {n} assets")
        else:
            print(f"no asset_catalog.json under {args.catalog}")
    if args.ledger:
        led = catalog.ValidationLedger(args.ledger)
        n_pass = sum(1 for e in led.takes.values() if e.get("status") == "PASS")
        print(f"ledger: {len(led.takes)} takes ({n_pass} PASS)")
    if not args.catalog and not args.ledger:
        print("status: pass --catalog DIR and/or --ledger FILE")
    return 0


def _cmd_readme(args) -> int:
    from soma_synth.docs_gen import readme
    from soma_synth.pipeline import paths as paths_mod

    refusal = _bundle_state_refusal(args.datasets, "readme")
    if refusal:
        print(refusal, file=sys.stderr)
        return EXIT_REFUSED
    for dataset in args.datasets:
        # README.md goes into the bundle: never into an evidence or a read-only folder
        code = _refused_location("readme", ("readme",), dataset)
        if code is not None:
            return code
    if args.top_level_root:
        # <folder>/README.md: never the data root's own (retention rule 1.4,
        # docs/guides/RETENTION_RULES.md), never inside the read-only or evidence folders
        try:
            paths_mod.check_readme_root(args.top_level_root)
        except paths_mod.OutputLocationRefused as error:
            print(f"readme: {error}", file=sys.stderr)
            return EXIT_REFUSED
        except paths_mod.PathConfigError as error:
            print(f"readme: {error}", file=sys.stderr)
            return 2
    for d in args.datasets:
        print(f"wrote {readme.write_dataset_readme(d)}")
    if args.top_level_root:
        print(f"wrote {readme.write_top_level_readme(args.top_level_root, args.datasets)}")
    return 0


def _cmd_register(args) -> int:
    if _refuse_paired_weargait(args.source):
        return 2
    refusal = _bundle_state_refusal([args.dataset_dir], "register", args.source)
    other = _other_lineage_refusal(args.dataset_dir, args.source)
    if refusal or other:
        print(refusal or f"register: refusing: {other}", file=sys.stderr)
        return EXIT_REFUSED
    # kept evidence is not registered as live data, and the catalog goes where run puts one
    code = _refused_location("register", ("register",), args.dataset_dir, args.data_root,
                             catalog_dir=Path(args.catalog))
    if code is not None:
        return code
    # The asset paths are relative to the data root: --data-root, else SOMA_DATA_ROOT, as for
    # `run` and `pipeline`. The bundle's parent is the lineage container, not a data root: paths
    # relative to it ('gaitex_unified8/...') would match none of the dataset's entries
    # ('runs/experimental_generation_poc_demo/gaitex_unified8/...').
    from soma_synth.pipeline import paths as paths_mod

    try:
        data_root = Path(args.data_root) if args.data_root else paths_mod.data_root()
    except paths_mod.PathConfigError as error:
        print(f"register: {error} (or pass --data-root)", file=sys.stderr)
        return 2
    outside = _outside_data_root(args.dataset_dir, data_root)
    if outside is not None:
        print(f"register: {outside}", file=sys.stderr)
        return 2
    sv = args.spec_version or spec.SPEC_VERSION
    assets = catalog.register_dataset(
        Path(args.catalog), args.dataset_dir, source=args.source, spec_version=sv, data_root=data_root,
        registered_by={"tool": "soma-synth register"},
    )
    print(f"registered {len(assets)} assets ({args.source} {sv}) under {args.catalog}: "
          f"{assets.summary()}")
    return 0


# ------------------------------------------------------------------ sources
def _cmd_hash_sources(args) -> int:
    """Write the SHA256SUMS list of the source folders under the source root. Reads only."""
    from soma_synth.pipeline import paths as paths_mod
    from soma_synth.pipeline import source_files

    try:
        folders = source_files.parse_sources_option(args.sources)
        root = Path(args.source_root) if args.source_root else paths_mod.source_root()
        if not root.is_dir():
            raise source_files.SourceFilesError(f"--source-root {root} is not a folder")
        print(f"hash-sources: {', '.join(folders)} under {root} -> {args.out}")
        result = source_files.hash_sources(root, folders, args.out, force=args.force)
    except paths_mod.OutputLocationRefused as error:
        print(f"hash-sources: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except paths_mod.PathConfigError as error:
        print(f"hash-sources: {error}", file=sys.stderr)
        return 2
    print(f"wrote {result.out}: {result.files} files, {result.bytes / 1e9:.2f} GB, hashed in "
          f"{result.seconds:.1f} s")
    return 0


def _cmd_stage_sources(args) -> int:
    """Copy the listed source files that are missing onto the source root, verified."""
    from soma_synth.pipeline import paths as paths_mod
    from soma_synth.pipeline import source_files

    try:
        folders = source_files.parse_sources_option(args.sources) if args.sources else None
        if args.to:
            to_root = Path(args.to)
        else:
            override = paths_mod.source_root_override()
            to_root = override if override is not None else paths_mod.data_root() / paths_mod.EXTRACTED
        verb = "verifying" if args.verify_only else f"staging from {args.from_root} into"
        print(f"stage-sources: {verb} {to_root} against {args.sums}")
        result = source_files.stage_sources(
            to_root, args.sums, from_root=args.from_root, folders=folders,
            verify_only=args.verify_only)
    except paths_mod.OutputLocationRefused as error:
        print(f"stage-sources: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except paths_mod.PathConfigError as error:
        print(f"stage-sources: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        # a copy that failed part-way (a full disk, a drive that went away): its temporary file
        # is gone and nothing listed was overwritten; run again to stage the rest
        print(f"stage-sources: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    counts = (f"{result.listed} listed: {len(result.present)} present, {len(result.copied)} "
              f"copied ({result.bytes_copied / 1e9:.2f} GB), {len(result.missing)} missing, "
              f"{len(result.conflicts)} different, {len(result.unavailable)} not in --from, "
              f"{len(result.corrupt)} corrupt copies; {result.seconds:.1f} s")
    print(f"stage-sources: {counts}")
    if result.leftovers:
        print(f"stage-sources: {len(result.leftovers)} leftover staging file(s) (*.staging, listed "
              "above) from an interrupted run: not source files, never deleted by this command; "
              "remove them by hand")
    code = result.exit_code
    if code == EXIT_REFUSED:
        print("stage-sources: refused: a destination file differs from the list; nothing was "
              "overwritten", file=sys.stderr)
    return code


# ------------------------------------------------------------------ pipeline

def _next_step_guide(step: str, ds: Path, parent: Path, ledger: Path, report: Path,
                     catalog_dir: Path, source: str) -> str:
    """One line telling the user the command for the step that comes AFTER ``step``."""
    guides = {
        "generate": (f"NEXT -> validate: soma-synth validate {ds} --full "
                     f"--ledger {ledger} --report {report}"),
        "validate": f"NEXT -> readme: soma-synth readme {ds} --top-level-root {parent}",
        "readme": (f"NEXT -> register: soma-synth register {ds} --catalog {catalog_dir} "
                   f"--source {source} --data-root <SOMA_DATA_ROOT>   [experimental/catalog]"),
        "register": ("NEXT -> done: the bundle is registered in the local catalog and stays "
                     "on this machine (INTERNAL-ONLY). soma-synth does not push bundles anywhere."),
    }
    return guides[step]


def _pipeline_generate_refusal(source: str) -> str:
    return (
        f"pipeline: refusing to generate {source}. This command does not generate: generation "
        f"goes through `soma-synth run --source {source}`, which takes the generation decision "
        "(ADR-0041's standing decision, or an owner record with --authorized-by), runs the "
        "corpus stage and the post-steps, and leaves the run record with the source hashes, the "
        "code revisions and that decision. A generation outside `soma-synth run` (this "
        "command's generate step, a generator script run directly) records none of that and is "
        "not covered by the standing decision. Pass --skip-generate (or --start-at validate) to "
        "validate, document and register a bundle that exists."
    )


def _inside(path: Path, folder: Path) -> bool:
    resolved, base = Path(path).resolve(), Path(folder).resolve()
    return resolved == base or base in resolved.parents


def _location_refusal(plan, ds: Path, data_root: Path | None, *, catalog_dir: Path | None = None,
                      validation_files=(), top_level_root: Path | None = None
                      ) -> tuple[int, str] | None:
    """The runner's output-location refusals, for the steps in ``plan`` that write: those of
    ``pipeline``, and of ``validate``, ``readme`` and ``register`` on their own.

    validate writes ``validation_ledger.json`` and ``VALIDATION_REPORT.json`` (into the bundle
    unless --ledger / --report say otherwise; standalone, only the ones given) and readme writes
    ``README.md`` into the bundle; so a bundle directory written into gets
    ``paths.check_bundle_dir(root=<data root>)``, as the runner's does: nothing inside an evidence
    folder of any data root, nor inside ``extracted``, ``raw_archives``, the source or body-model
    folders, not a container (a synchronised folder draws only a warning). A ledger or report
    written outside the bundle gets ``paths.check_record_file``: its folder the run records' check (``_runs``
    allowed; not a container, not inside a lineage directory, not a read-only or evidence folder),
    and never one of the data plane's own files; the bundle itself is then only read. register
    reads the bundle, which may not be evidence or a read-only folder either
    (``check_bundle_dir(writes=False)``: kept evidence is not filed as live data), and writes the
    catalog, which gets ``paths.check_record_dir``. readme's ``--top-level-root`` gets
    ``paths.check_readme_root`` (never the data root or ``SOMA_SOURCE_ROOT`` itself, never inside
    a read-only or evidence folder). None when nothing is refused, else (exit code, the error): 3
    for a refused location, 2 for a location that cannot be resolved.
    """
    from soma_synth.pipeline import paths as paths_mod

    writes_bundle = "readme" in plan or (
        "validate" in plan and any(_inside(Path(p), ds) for p in validation_files))
    try:
        if writes_bundle:
            paths_mod.check_bundle_dir(ds, root=data_root)
        elif "register" in plan:
            paths_mod.check_bundle_dir(ds, root=data_root, writes=False, action="register")
        if "validate" in plan:
            for path in validation_files:
                if not _inside(Path(path), ds):
                    paths_mod.check_record_file(path, root=data_root)
        if "register" in plan and catalog_dir is not None:
            paths_mod.check_record_dir(catalog_dir, root=data_root)
        if "readme" in plan and top_level_root is not None:
            paths_mod.check_readme_root(top_level_root, root=data_root)
    except paths_mod.OutputLocationRefused as error:
        return EXIT_REFUSED, str(error)
    except paths_mod.PathConfigError as error:
        return 2, str(error)
    return None


def _cmd_pipeline(args) -> int:
    if _refuse_paired_weargait(args.source):
        return 2
    from soma_synth.docs_gen import readme as readme_mod
    from soma_synth.pipeline import paths as paths_mod
    from soma_synth.pipeline.runner import CATALOG_RELPATH

    plan = STEPS[STEPS.index(args.start_at): STEPS.index(args.stop_after) + 1]
    if "generate" in plan and not args.skip_generate:
        # ADR-0041: the untraced generation path is closed; `run` is the one that records
        print(_pipeline_generate_refusal(args.source), file=sys.stderr)
        return EXIT_REFUSED

    ds = Path(args.dataset_dir)
    parent = ds.parent
    ledger_path = Path(args.ledger) if args.ledger else ds / "validation_ledger.json"
    report_path = Path(args.report) if args.report else ds / "VALIDATION_REPORT.json"
    # The data root is SOMA_DATA_ROOT unless --data-root names one, as for `run`: registration
    # writes paths relative to it, and the catalog defaults to <data root>/experimental/catalog.
    # The bundle's parent is the lineage container, which is no data root and holds no catalog.
    data_root = Path(args.data_root) if args.data_root else None
    if data_root is None and "register" in plan:
        try:
            data_root = paths_mod.data_root()
        except paths_mod.PathConfigError as error:
            print(f"pipeline: {error} (or pass --data-root)", file=sys.stderr)
            return 2
    catalog_dir = (Path(args.catalog) if args.catalog
                   else data_root / CATALOG_RELPATH if data_root is not None
                   else Path("<SOMA_DATA_ROOT>") / CATALOG_RELPATH)

    refusal = _location_refusal(
        plan, ds, data_root, catalog_dir=catalog_dir, validation_files=(ledger_path, report_path),
        top_level_root=Path(args.top_level_root) if args.top_level_root else None)
    if refusal is not None:
        code, message = refusal
        print(f"pipeline: refusing: {message}" if code == EXIT_REFUSED else f"pipeline: {message}",
              file=sys.stderr)
        return code
    outside = _outside_data_root(ds, data_root) if "register" in plan else None
    if outside is not None:
        print(f"pipeline: {outside}", file=sys.stderr)
        return 2
    if set(plan) & {"validate", "readme", "register"}:
        # an aside or a failed replacement's output, or a generation that did not finish; and
        # another source's lineage
        first = next(step for step in plan if step in ("validate", "readme", "register"))
        state = _bundle_state_refusal([ds], first, args.source)
        other = _other_lineage_refusal(ds, args.source)
        if state or other:
            print(f"pipeline: {state}" if state else f"pipeline: refusing: {other}",
                  file=sys.stderr)
            return EXIT_REFUSED

    n = len(plan)

    for k, step in enumerate(plan, 1):
        print(f"[{k}/{n}] {step} ...")
        rc = 0
        if step == "generate":
            print("   skipped (--skip-generate; dataset assumed already built)")
        elif step == "validate":
            ledger = catalog.ValidationLedger(ledger_path)
            rep = validate(str(ds), level=args.level, ledger=ledger, full=args.full)
            ledger.save()
            report_path.write_text(
                json.dumps(rep.to_dict(), indent=2) + "\n", encoding="utf-8", newline="\n"
            )
            print("   " + rep.summary().splitlines()[0])
            rc = 0 if rep.ok else 1
        elif step == "readme":
            print(f"   wrote {readme_mod.write_dataset_readme(ds)}")
            if args.top_level_root:
                print(f"   wrote {readme_mod.write_top_level_readme(args.top_level_root, [ds])}")
        elif step == "register":
            assets = catalog.register_dataset(
                catalog_dir, ds, source=args.source, spec_version=spec.SPEC_VERSION, data_root=data_root,
                registered_by={"tool": "soma-synth pipeline"},
            )
            print(f"   registered {len(assets)} assets under {catalog_dir}: {assets.summary()}")

        if rc != 0:
            print(f"   FAILED at step '{step}' (rc={rc}) -- stopping (fail-fast).")
            return rc
        print("   " + _next_step_guide(step, ds, parent, ledger_path, report_path, catalog_dir, args.source))

    print("pipeline complete.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="soma-synth",
        description="Generate/validate/document/register experimental synthetic-IMU datasets "
                    "(spec_id=qmd_unified8_smpl18).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate a produced dataset dir against the spec")
    v.add_argument("dataset_dir")
    v.add_argument("--level", default="L4", choices=["L0", "L1", "L2", "L3", "L4"])
    v.add_argument("--sample", type=int, default=None)
    v.add_argument("--full", action="store_true", help="ignore the ledger; re-check every take")
    v.add_argument("--report", help="write VALIDATION_REPORT.json here")
    v.add_argument("--ledger", help="incremental validation ledger path (read + update)")
    v.add_argument(
        "--no-conformance",
        dest="conformance",
        action="store_false",
        help="skip the dataset-profile conformance judgement (ADR-0040); on by default",
    )
    v.set_defaults(func=_cmd_validate, conformance=True)

    rn = sub.add_parser(
        "run",
        help="run the pipeline through the runner: writes a run directory with the seven "
             "artifacts PIPELINE_GOVERNANCE 10.1 requires. Generates experimental_non_candidate "
             "bundles only, under the standing decision of ADR-0041 (no record needed) or an "
             "owner record, which works only when soma-synth is mounted in the parent project "
             "as packages/soma-synth",
    )
    rn.add_argument("dataset_dir", nargs="?", default=None,
                    help="the bundle directory (default: the source's bundle lineage under "
                         "the data root)")
    rn.add_argument("--source", required=True)
    rn.add_argument("--data-root", default=None, help="default $SOMA_DATA_ROOT")
    rn.add_argument("--runs-root", default=None, help="where the run record goes")
    rn.add_argument("--catalog", default=None,
                    help="default <data root>/experimental/catalog")
    rn.add_argument("--skip-generate", action="store_true")
    rn.add_argument(
        "--authorized-by", default=None,
        help="an owner authorization record under the parent project's research/decisions/ "
             "that names this source and the experimental_non_candidate class, checked "
             "against the parent's gates and copied into the run record. It works only when "
             "soma-synth is mounted in the parent project (packages/soma-synth) and is refused "
             "otherwise; a relative path resolves against the parent project's root. Without "
             "it the run records the registry's standing decision (ADR-0041), which needs no "
             "record",
    )
    rn.add_argument("--corpus", default=None,
                    help="the retarget corpus of hknu/gaitex/addbiomechanics (default: its "
                         "lineage directory under the data root); a finished one is reused, a "
                         "missing or empty one is built")
    rn.add_argument("--rebuild-corpus", action="store_true",
                    help="build the retarget corpus even when a finished one is there; a corpus "
                         "directory that holds anything (finished or not) is rebuilt only with "
                         "--replace-existing as well")
    rn.add_argument("--replace-existing", action="store_true",
                    help="every run builds into an empty directory and none resumes: a bundle "
                         "directory that holds anything (a finished bundle, an unfinished one, "
                         "other files) is refused without this flag, and so is a corpus "
                         "directory that has to be built (with --rebuild-corpus). With it the "
                         "run moves the whole directory aside to <dir>.replaced-<run>, keeps "
                         "copies of its small evidence files in the run record and builds into "
                         "an empty directory; on success the aside directory is deleted (a "
                         "production directory's -- any directory directly under "
                         "runs/experimental_generation_poc_demo -- is kept for the data owner's "
                         "approved deletion, retention rule 1.5), on failure it is renamed back. A "
                         "production directory holding a finished generation is replaced in the "
                         "order retention rule 1.2 sets, so on one this is refused unless a "
                         "_superseded/<name>_* folder beside it holds a copy of its INDEX.json "
                         "(or fingerprint) and its SUPERSEDED.md")
    rn.add_argument("--corpus-arg", action="append", default=[], metavar="ARG",
                    help="one more argument for the corpus entrypoint, repeatable; use the = "
                         "form for flags: --corpus-arg=--subjects --corpus-arg=austra. A flag "
                         "the runner sets (the corpus output, a source location) is refused: "
                         "use --corpus and SOMA_SOURCE_ROOT")
    rn.add_argument("--bundle-arg", action="append", default=[], metavar="ARG",
                    help="one more argument for the bundle entrypoint, repeatable (= form for "
                         "flags); a selection flag (the registry's selection_args) is refused "
                         "when the bundle directory is a production lineage directory, and a "
                         "flag the runner sets (the bundle output, the corpus flag, a source "
                         "location) always")
    rn.add_argument(
        "--jobs", type=int, default=1,
        help="workers for the generate step, reached through the registry's parallel_args "
             "(a process-pool flag, or N shards then a merge); 1 runs the generator alone",
    )
    rn.add_argument("--start-at", default="generate", choices=list(STEPS))
    rn.add_argument("--stop-after", default="register", choices=list(STEPS))
    rn.add_argument("--level", default="L4", choices=["L0", "L1", "L2", "L3", "L4"])
    rn.add_argument("--full", action="store_true")
    rn.set_defaults(func=_cmd_run)

    pv = sub.add_parser(
        "provenance",
        help="report PIPELINE_GOVERNANCE 10.1/10.2 coverage for a bundle; --retrofit writes the "
             "sidecar of what is still measurable",
    )
    pv.add_argument("dataset_dir")
    pv.add_argument("--retrofit", action="store_true",
                    help="write PROVENANCE_RETROFIT.json; never touches take manifests")
    pv.add_argument("--body-model", default=None,
                    help="path to the SMPL model used, so its sha256 can be recorded")
    pv.set_defaults(func=_cmd_provenance)

    pd = sub.add_parser(
        "pipeline-doc",
        help="render the per-source table of GENERATION_PIPELINE_STANDARD.md from the registry",
    )
    pd.add_argument("--render", action="store_true", help="write into the document, not stdout")
    pd.add_argument("--document", default=None)
    pd.set_defaults(func=_cmd_pipeline_doc)

    s = sub.add_parser("status", help="summarize a catalog and/or ledger")
    s.add_argument("--catalog")
    s.add_argument("--ledger")
    s.set_defaults(func=_cmd_status)

    r = sub.add_parser("readme", help="generate a field-by-field README.md per dataset (+ optional top-level)")
    r.add_argument("datasets", nargs="+")
    r.add_argument("--top-level-root", help="also write a top-level README.md here across the given datasets")
    r.set_defaults(func=_cmd_readme)

    rg = sub.add_parser("register", help="register a validated dataset's assets into experimental/catalog")
    rg.add_argument("dataset_dir")
    rg.add_argument("--catalog", required=True)
    rg.add_argument("--source", required=True)
    rg.add_argument("--spec-version", default=None)
    rg.add_argument("--data-root", default=None,
                    help="the data root the asset paths are relative to (default "
                         "$SOMA_DATA_ROOT; refused when neither is set); the bundle must be "
                         "inside it")
    rg.set_defaults(func=_cmd_register)

    pl = sub.add_parser(
        "pipeline",
        help="validate -> readme -> register an existing bundle in order (fail-fast), guiding the "
             "next step after each; it does not generate (use `run`)",
    )
    pl.add_argument("dataset_dir")
    pl.add_argument("--source", required=True)
    pl.add_argument("--catalog", default=None,
                    help="catalog dir (default <data root>/experimental/catalog)")
    pl.add_argument("--data-root", default=None, help="the data root (default $SOMA_DATA_ROOT)")
    pl.add_argument("--top-level-root", default=None, help="also write a top-level README.md here")
    pl.add_argument("--generate-cmd", default=None,
                    help="retired: the generate step is refused; generate with `soma-synth run`")
    pl.add_argument("--skip-generate", action="store_true",
                    help="dataset already built; skip step 1 (required unless --start-at is "
                         "past generate: this command refuses to generate)")
    pl.add_argument("--start-at", default="generate", choices=list(STEPS))
    pl.add_argument("--stop-after", default="register", choices=list(STEPS))
    pl.add_argument("--level", default="L4", choices=["L0", "L1", "L2", "L3", "L4"])
    pl.add_argument("--full", action="store_true", help="validate: re-check every take (ignore ledger)")
    pl.add_argument("--ledger", default=None, help="ledger path (default <dataset>/validation_ledger.json)")
    pl.add_argument("--report", default=None, help="report path (default <dataset>/VALIDATION_REPORT.json)")
    pl.set_defaults(func=_cmd_pipeline)

    hs = sub.add_parser(
        "hash-sources",
        help="write a SHA256SUMS list (GNU coreutils format, '<sha256>  <folder>/<path>', sorted) "
             "of the source folders; reads the sources only",
    )
    hs.add_argument("--source-root", default=None,
                    help="the folder holding the source folders (default $SOMA_SOURCE_ROOT, else "
                         "$SOMA_DATA_ROOT/extracted)")
    hs.add_argument("--sources", default=None,
                    help="comma-separated source folders (amass, prism, gaitex, hknu_fullbody, "
                         "addbiomechanics; 'hknu' means hknu_fullbody); default all five")
    hs.add_argument("--out", required=True,
                    help="the list file to write; refused inside the source root, inside "
                         "extracted/ or raw_archives/ of $SOMA_DATA_ROOT, inside an evidence "
                         "folder (_superseded, _runs, _manifest_backfill), the lineage container "
                         "runs/experimental_generation_poc_demo or a bundle, or over an "
                         "existing file that is not such a list (see --force); a synchronised "
                         "or shared folder draws a warning")
    hs.add_argument("--force", action="store_true",
                    help="replace an existing --out file that is not a SHA256SUMS list this "
                         "command wrote (the location refusals still apply)")
    hs.set_defaults(func=_cmd_hash_sources)

    ss = sub.add_parser(
        "stage-sources",
        help="copy every listed source file missing under the source root from a folder holding "
             "the source folders, each verified against the list before it is renamed into "
             "place; never deletes or overwrites. Exit 3 (nothing "
             "copied) when a destination file differs from the list, 1 when something could not "
             "be staged",
    )
    ss.add_argument("--from", dest="from_root", default=None,
                    help="the folder holding the source folders (only read; may be a "
                         "synchronised or shared folder). Required unless --verify-only")
    ss.add_argument("--to", default=None,
                    help="the source root to fill (default $SOMA_SOURCE_ROOT, else "
                         "$SOMA_DATA_ROOT/extracted); not inside an evidence folder, not the "
                         "data root itself; a synchronised or shared folder draws a warning")
    ss.add_argument("--sources", default=None,
                    help="comma-separated source folders to take from the list (default every "
                         "folder it names)")
    ss.add_argument("--sums", required=True, help="the SHA256SUMS list (from hash-sources)")
    ss.add_argument("--verify-only", action="store_true",
                    help="copy nothing; report the listed files that are missing or differ under "
                         "the source root (exit 1 when any). Either way, the unverified copies "
                         "an interrupted run left (*.staging) are reported, never deleted")
    ss.set_defaults(func=_cmd_stage_sources)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
