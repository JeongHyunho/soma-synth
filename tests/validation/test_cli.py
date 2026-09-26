"""soma-synth CLI (validate / status / register / pipeline / run, and which commands it offers)."""

import json
import types
from pathlib import Path

import pytest

from soma_synth import cli
from soma_synth.datasets import catalog

from test_validation_checks import make_dataset


def test_cli_validate_clean_returns_0(tmp_path: Path, capsys) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    rc = cli.main(["validate", str(ds), "--level", "L4"])
    assert rc == 0 and "PASS" in capsys.readouterr().out


def test_cli_validate_broken_returns_1(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    (ds / "amass-take000" / "large_reference.npz").unlink()  # missing deliverable -> L0 FAIL
    assert cli.main(["validate", str(ds), "--level", "L4"]) == 1


def test_cli_validate_writes_report_and_updates_ledger(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=2)
    report, ledger = tmp_path / "rep.json", tmp_path / "ledger.json"
    rc = cli.main(
        ["validate", str(ds), "--level", "L2", "--report", str(report), "--ledger", str(ledger)]
    )
    assert rc == 0
    d = json.loads(report.read_text(encoding="utf-8"))
    assert d["spec_version"] == "faithful-v2" and d["ok"] is True
    assert json.loads(ledger.read_text(encoding="utf-8"))["schema"] == catalog.SCHEMA_LEDGER
    # second run trusts everything from the ledger
    assert cli.main(["validate", str(ds), "--level", "L2", "--ledger", str(ledger)]) == 0


def test_cli_status_reports_catalog(tmp_path: Path, capsys) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    catdir = tmp_path / "catalog"
    catdir.mkdir()
    catalog.register_dataset(catdir, ds, source="amass", spec_version="faithful-v2", data_root=tmp_path)
    rc = cli.main(["status", "--catalog", str(catdir)])
    out = capsys.readouterr().out
    assert rc == 0 and "amass" in out.lower() and "faithful-v2" in out


def test_cli_offers_the_pipeline_commands_only() -> None:
    """The WearGait estimated pipeline, the governed source-parsing operations and the informational
    generate/push commands stayed in the parent project; soma-synth has the pipeline's commands
    and the two that list and stage the source files."""
    parser = cli.build_parser()
    commands = next(action for action in parser._subparsers._group_actions).choices
    assert set(commands) == {"run", "validate", "readme", "register", "status", "pipeline",
                             "pipeline-doc", "provenance", "hash-sources", "stage-sources"}
    for gone in (["estimated", "--source", "weargait_pd", "--stage", "verify"],
                 ["source-operation", "a", "b", "c", "d"], ["generate", "amass"],
                 ["push", "somedir"]):
        with pytest.raises(SystemExit):
            cli.main(gone)


def test_cli_register_writes_catalog(tmp_path: Path, capsys) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    catdir = tmp_path / "catalog"
    rc = cli.main(
        ["register", str(ds), "--catalog", str(catdir), "--source", "amass", "--data-root", str(tmp_path)]
    )
    assert rc == 0
    ac = json.loads((catdir / "asset_catalog.json").read_text(encoding="utf-8"))
    assert len(ac["assets"]) > 0 and any(a["source"] == "amass" for a in ac["assets"])


def _pipeline_argv(ds: Path, tmp_path: Path) -> list[str]:
    # the top-level README goes into a folder of its own: the data root's README.md is the data
    # plane's own and is never overwritten (retention rule 1.4)
    top = tmp_path / "top"
    top.mkdir(exist_ok=True)
    return [
        "pipeline", str(ds), "--source", "amass", "--skip-generate",
        "--catalog", str(tmp_path / "catalog"), "--data-root", str(tmp_path),
        "--top-level-root", str(top),
    ]


def test_cli_pipeline_skipgen_runs_validate_readme_register(tmp_path: Path, capsys) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    rc = cli.main(_pipeline_argv(ds, tmp_path))
    out = capsys.readouterr().out
    assert rc == 0
    # each of the four steps announced, and a NEXT-step guide printed after each
    for step in ("generate", "validate", "readme", "register"):
        assert step in out
    assert out.count("NEXT") >= 4
    # the register step's guide ends the pipeline: soma-synth pushes nothing
    assert "does not push" in out
    # side effects: README + catalog + ledger + report all produced
    assert (ds / "README.md").exists()
    assert (tmp_path / "catalog" / "asset_catalog.json").exists()
    assert (ds / "validation_ledger.json").exists()
    assert (ds / "VALIDATION_REPORT.json").exists()


def test_cli_pipeline_refuses_to_generate_and_points_to_run(tmp_path: Path, monkeypatch,
                                                            capsys) -> None:
    """ADR-0041: this command's generate step launched a generator with no generation
    decision, no run record and no source hashes. It is refused before anything runs, with a
    pointer to `soma-synth run`, which records all three."""
    ds = make_dataset(tmp_path / "amass_x", n=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("the pipeline command launched a generator")

    monkeypatch.setattr(cli, "run_generate", forbidden)
    for extra in ([], ["--generate-cmd", "not-executed"]):
        argv = ["pipeline", str(ds), "--source", "amass", "--catalog", str(tmp_path / "catalog"),
                *extra]
        assert cli.main(argv) == cli.EXIT_REFUSED
        assert "soma-synth run --source amass" in capsys.readouterr().err
    # fail-fast: nothing downstream ran
    assert not (ds / "README.md").exists()
    assert not (ds / "VALIDATION_REPORT.json").exists()
    assert not (tmp_path / "catalog" / "asset_catalog.json").exists()


def test_cli_pipeline_registers_into_the_experimental_catalog_of_the_data_root(
        tmp_path: Path, monkeypatch, capsys) -> None:
    """The default catalog was <bundle parent>/catalog, inside the lineage container; it is now
    <data root>/experimental/catalog, as for `run`, with the data root from SOMA_DATA_ROOT."""
    ds = make_dataset(tmp_path / "runs" / "amass_x", n=1)
    monkeypatch.setenv("SOMA_DATA_ROOT", str(tmp_path))
    rc = cli.main(["pipeline", str(ds), "--source", "amass", "--skip-generate"])
    capsys.readouterr()
    assert rc == 0
    assert (tmp_path / "experimental" / "catalog" / "asset_catalog.json").exists()
    assert not (ds.parent / "catalog").exists()
    assets = json.loads((tmp_path / "experimental" / "catalog" / "asset_catalog.json").read_text(
        encoding="utf-8"))["assets"]
    # relative to the data root, not to the bundle's parent (the old default data root)
    assert assets and all(a["relative_path"].startswith("runs/amass_x/") for a in assets)

    monkeypatch.delenv("SOMA_DATA_ROOT")
    assert cli.main(["pipeline", str(ds), "--source", "amass", "--start-at", "register"]) == 2
    assert "SOMA_DATA_ROOT" in capsys.readouterr().err


POC = "runs/experimental_generation_poc_demo"


def _tree(folder: Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*"))


def test_cli_pipeline_refuses_the_output_locations_the_runner_refuses(tmp_path: Path,
                                                                     capsys) -> None:
    """validate and readme write VALIDATION_REPORT.json, validation_ledger.json and README.md
    into the bundle directory, and register writes the catalog. `pipeline --skip-generate` did so
    into any directory it was given; it now refuses what `run` refuses, before writing."""
    evidence = make_dataset(tmp_path / POC / "_superseded" / "amass_x", n=1)
    extracted = make_dataset(tmp_path / "extracted" / "amass_x", n=1)
    for ds, label in ((evidence, "_superseded"), (extracted, "extracted")):
        before = _tree(ds)
        rc = cli.main(["pipeline", str(ds), "--source", "amass", "--skip-generate",
                       "--data-root", str(tmp_path), "--catalog", str(tmp_path / "catalog")])
        assert rc == cli.EXIT_REFUSED
        assert label in capsys.readouterr().err
        assert _tree(ds) == before
    assert not (tmp_path / "catalog").exists()

    # the catalog inside a lineage directory, or a container itself
    work = make_dataset(tmp_path / "work" / "amass_x", n=1)
    before = _tree(work)
    for catalog_dir in (tmp_path / POC / "amass_faithful_full" / "catalog", tmp_path / POC):
        rc = cli.main(["pipeline", str(work), "--source", "amass", "--skip-generate",
                       "--data-root", str(tmp_path), "--catalog", str(catalog_dir)])
        assert rc == cli.EXIT_REFUSED
        capsys.readouterr()
    assert _tree(work) == before
    assert not (tmp_path / POC / "amass_faithful_full").exists()

    # validating into files elsewhere writes nothing into the bundle, so reading it is not refused
    out = tmp_path / "out"
    out.mkdir()
    before = _tree(evidence)
    rc = cli.main(["pipeline", str(evidence), "--source", "amass", "--start-at", "validate",
                   "--stop-after", "validate", "--data-root", str(tmp_path),
                   "--ledger", str(out / "ledger.json"), "--report", str(out / "report.json")])
    capsys.readouterr()
    assert rc == 0
    assert (out / "report.json").exists()
    assert _tree(evidence) == before


def _files(folder: Path) -> dict[str, bytes]:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*")
            if p.is_file()}


@pytest.mark.parametrize("where", ["_superseded", "_runs", "_manifest_backfill", "extracted",
                                   "raw_archives"])
@pytest.mark.parametrize("command", ["validate", "readme", "register"])
def test_cli_refuses_a_bundle_folder_in_an_evidence_or_source_folder(
        tmp_path: Path, monkeypatch, capsys, command, where) -> None:
    """`readme` on a folder under _superseded overwrote the README.md kept there and exited 0;
    validate wrote its report and ledger beside the kept ones, and register filed the evidence as
    live assets. They refuse it now as `run` does (exit 3), before writing anything. An evidence
    folder is one under any data root, so SOMA_DATA_ROOT need not name it; extracted/ and
    raw_archives/ are those of the data root."""
    root = tmp_path / "data"
    if where.startswith("_"):
        ds = make_dataset(root / POC / where / "gaitex_unified8_2026-09-16", n=1)
        monkeypatch.delenv("SOMA_DATA_ROOT", raising=False)
    else:
        ds = make_dataset(root / where / "gaitex" / "gaitex_unified8", n=1)
        monkeypatch.setenv("SOMA_DATA_ROOT", str(root))
    monkeypatch.delenv("SOMA_SOURCE_ROOT", raising=False)
    (ds / "README.md").write_text("# kept with the evidence\n", encoding="utf-8")
    before = _files(ds)
    argv = {
        "validate": ["validate", str(ds), "--level", "L2",
                     "--report", str(ds / "VALIDATION_REPORT.json"),
                     "--ledger", str(ds / "validation_ledger.json")],
        "readme": ["readme", str(ds)],
        "register": ["register", str(ds), "--catalog", str(tmp_path / "catalog"), "--source",
                     "amass"],
    }[command]
    assert cli.main(argv) == cli.EXIT_REFUSED
    err = capsys.readouterr().err
    verb = "register" if command == "register" else "write"
    assert err.startswith(f"{command}: refusing to {verb} {ds}"), err
    assert where in err
    assert _files(ds) == before                              # the kept README.md included
    assert not (tmp_path / "catalog").exists()


def test_cli_validate_only_reads_an_evidence_folder_it_writes_nothing_into(
        tmp_path: Path, monkeypatch, capsys) -> None:
    ds = make_dataset(tmp_path / POC / "_superseded" / "amass_x_2026-09-01", n=1)
    monkeypatch.setenv("SOMA_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("SOMA_SOURCE_ROOT", raising=False)
    before = _files(ds)
    out = tmp_path / "out"
    out.mkdir()
    assert cli.main(["validate", str(ds), "--level", "L2"]) == 0
    assert cli.main(["validate", str(ds), "--level", "L2", "--report", str(out / "report.json"),
                     "--ledger", str(out / "ledger.json")]) == 0
    capsys.readouterr()
    assert _files(ds) == before and (out / "report.json").exists()
    # a report written elsewhere gets the record-file guard, as the pipeline's does: the run
    # records folder takes one, another evidence folder does not
    target = tmp_path / POC / "_runs" / "check" / "report.json"
    target.parent.mkdir(parents=True)
    assert cli.main(["validate", str(ds), "--level", "L2", "--report", str(target),
                     "--ledger", str(out / "ledger.json")]) == 0
    capsys.readouterr()
    refused = tmp_path / POC / "_manifest_backfill" / "report.json"
    assert cli.main(["validate", str(ds), "--level", "L2", "--report", str(refused)]) == \
        cli.EXIT_REFUSED
    assert "_manifest_backfill" in capsys.readouterr().err and not refused.exists()


def test_cli_pipeline_registers_no_evidence_folder_either(tmp_path: Path, capsys) -> None:
    """`pipeline --start-at register` wrote nothing into the bundle, so its location was never
    checked: the evidence was registered."""
    ds = make_dataset(tmp_path / POC / "_superseded" / "amass_x_2026-09-01", n=1)
    rc = cli.main(["pipeline", str(ds), "--source", "amass", "--start-at", "register",
                   "--data-root", str(tmp_path), "--catalog", str(tmp_path / "catalog")])
    assert rc == cli.EXIT_REFUSED
    err = capsys.readouterr().err
    assert err.startswith("pipeline: refusing: refusing to register") and "_superseded" in err
    assert not (tmp_path / "catalog").exists()


def test_cli_register_puts_the_catalog_where_run_would(tmp_path: Path, capsys) -> None:
    ds = make_dataset(tmp_path / "work" / "amass_x", n=1)
    for catalog_dir in (tmp_path / POC / "amass_faithful_full" / "catalog", tmp_path / POC,
                        tmp_path / "extracted" / "catalog"):
        rc = cli.main(["register", str(ds), "--catalog", str(catalog_dir), "--source", "amass",
                       "--data-root", str(tmp_path)])
        assert rc == cli.EXIT_REFUSED, catalog_dir
        assert capsys.readouterr().err.startswith("register: refusing to write")
        assert not (catalog_dir / "asset_catalog.json").exists()


def test_cli_pipeline_refuses_a_bundle_whose_generation_did_not_finish(tmp_path: Path,
                                                                       capsys) -> None:
    from soma_synth.pipeline.runner import GENERATING_MARKER

    ds = make_dataset(tmp_path / "amass_x", n=1)
    (ds / GENERATING_MARKER).write_text("{}", encoding="utf-8")
    rc = cli.main(_pipeline_argv(ds, tmp_path))
    assert rc == cli.EXIT_REFUSED
    assert "soma-synth run --source amass" in capsys.readouterr().err
    assert not (ds / "README.md").exists()
    assert not (ds / "VALIDATION_REPORT.json").exists()
    assert not (tmp_path / "catalog" / "asset_catalog.json").exists()


def _unfinished(tmp_path: Path, state: str) -> Path:
    """A dataset the runner has not finished (or that is not a bundle): what validate, readme,
    register, pipeline and push must refuse."""
    from soma_synth.pipeline import runner

    if state == "marker":
        ds = make_dataset(tmp_path / "amass_x", n=1)
        (ds / runner.GENERATING_MARKER).write_text('{"source": "amass"}', encoding="utf-8")
    else:                                    # an aside, or a failed replacement's output
        ds = make_dataset(tmp_path / f"amass_x.{state}-0123456789ab", n=1)
    return ds


_STATES = [("marker", "--replace-existing"), ("replaced", "moved aside"), ("failed", "failed")]


@pytest.mark.parametrize("state, hint", _STATES)
@pytest.mark.parametrize("command", ["validate", "readme", "register", "pipeline"])
def test_cli_refuses_a_bundle_the_runner_has_not_finished(tmp_path: Path, capsys, command, state,
                                                          hint) -> None:
    """A .generating marker, or a directory that is an aside or a failed replacement's output:
    not a bundle to validate, document or register. Exit 3, naming what builds it, and nothing
    written."""
    ds = _unfinished(tmp_path, state)
    before = _tree(ds)
    argv = {
        "validate": ["validate", str(ds), "--level", "L2"],
        "readme": ["readme", str(ds)],
        "register": ["register", str(ds), "--catalog", str(tmp_path / "catalog"), "--source",
                     "amass", "--data-root", str(tmp_path)],
        "pipeline": _pipeline_argv(ds, tmp_path),
    }[command]
    assert cli.main(argv) == cli.EXIT_REFUSED
    assert hint in capsys.readouterr().err
    assert _tree(ds) == before
    assert not (tmp_path / "catalog").exists()


@pytest.mark.parametrize("command", ["register", "pipeline"])
def test_cli_refuses_another_sources_lineage(tmp_path: Path, capsys, command) -> None:
    """Registering the amass bundle lineage as prism (or a corpus lineage as a bundle) files one
    source's data under another's name."""
    ds = make_dataset(tmp_path / POC / "amass_faithful_full", n=1)
    before = _tree(ds)
    argv = (["register", str(ds), "--catalog", str(tmp_path / "catalog"), "--source", "prism",
             "--data-root", str(tmp_path)] if command == "register"
            else ["pipeline", str(ds), "--source", "prism", "--skip-generate", "--data-root",
                  str(tmp_path), "--catalog", str(tmp_path / "catalog")])
    assert cli.main(argv) == cli.EXIT_REFUSED
    assert "the amass bundle lineage" in capsys.readouterr().err
    assert _tree(ds) == before and not (tmp_path / "catalog").exists()


def test_cli_pipeline_guards_a_ledger_or_report_written_outside_the_bundle(tmp_path: Path,
                                                                          capsys) -> None:
    """--ledger and --report outside the bundle were written wherever they pointed: into the
    extracted sources, another lineage's directory, or over the data root's own README.md."""
    ds = make_dataset(tmp_path / "work" / "amass_x", n=1)
    (tmp_path / "README.md").write_text("# the data plane\n", encoding="utf-8")
    (tmp_path / "extracted" / "amass").mkdir(parents=True)
    (tmp_path / POC / "prism_faithful_full").mkdir(parents=True)
    before = _tree(tmp_path)
    for flag, target in (("--report", tmp_path / "extracted" / "amass" / "report.json"),
                         ("--ledger", tmp_path / "raw_archives" / "ledger.json"),
                         ("--report", tmp_path / POC / "prism_faithful_full" / "report.json"),
                         ("--report", tmp_path / "README.md"),
                         ("--ledger", tmp_path / POC / "_superseded" / "x" / "ledger.json")):
        rc = cli.main(["pipeline", str(ds), "--source", "amass", "--start-at", "validate",
                       "--stop-after", "validate", "--data-root", str(tmp_path), flag,
                       str(target)])
        assert rc == cli.EXIT_REFUSED, target
        capsys.readouterr()
    assert _tree(tmp_path) == before
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# the data plane\n"
    # a folder of its own, or the run records folder, is where they may go
    out = tmp_path / POC / "_runs" / "check"
    out.mkdir(parents=True)
    rc = cli.main(["pipeline", str(ds), "--source", "amass", "--start-at", "validate",
                   "--stop-after", "validate", "--data-root", str(tmp_path),
                   "--report", str(out / "report.json"), "--ledger", str(out / "ledger.json")])
    capsys.readouterr()
    assert rc == 0 and (out / "report.json").exists()


def test_the_top_level_readme_never_overwrites_the_data_roots_own(tmp_path: Path, monkeypatch,
                                                                  capsys) -> None:
    """`--top-level-root <data root>` wrote <data root>/README.md, which retention rule 1.4 says is
    never overwritten; so did it for the source root, and inside the evidence folders. The
    lineage container stays allowed."""
    ds = make_dataset(tmp_path / POC / "amass_x", n=1)
    sources = tmp_path / "sources"
    sources.mkdir()
    monkeypatch.setenv("SOMA_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SOMA_SOURCE_ROOT", str(sources))
    (tmp_path / "README.md").write_text("# the data plane\n", encoding="utf-8")
    (tmp_path / POC / "_superseded" / "amass_x_v1").mkdir(parents=True)
    (tmp_path / "extracted").mkdir()
    for top in (tmp_path, sources, tmp_path / POC / "_superseded" / "amass_x_v1",
                tmp_path / "extracted"):
        for argv in (["readme", str(ds), "--top-level-root", str(top)],
                     ["pipeline", str(ds), "--source", "amass", "--start-at", "readme",
                      "--stop-after", "readme", "--top-level-root", str(top)]):
            assert cli.main(argv) == cli.EXIT_REFUSED, (argv[0], top)
            assert "README.md" in capsys.readouterr().err
        assert not (top / "README.md").exists() or top == tmp_path
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# the data plane\n"
    assert not (ds / "README.md").exists()          # refused before anything was written

    assert cli.main(["readme", str(ds), "--top-level-root", str(tmp_path / POC)]) == 0
    capsys.readouterr()
    assert (tmp_path / POC / "README.md").exists() and (ds / "README.md").exists()


def test_cli_pipeline_stops_when_validate_fails(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    (ds / "amass-take000" / "large_reference.npz").unlink()  # L0 FAIL
    rc = cli.main(_pipeline_argv(ds, tmp_path))
    assert rc != 0
    assert not (ds / "README.md").exists()  # readme/register never reached
    assert not (tmp_path / "catalog" / "asset_catalog.json").exists()


# ------------------------------------------------------------------ how `run` reaches the runner
# How `run` hands its arguments to PipelineRunner.
def _fake_runner(monkeypatch, calls):
    from soma_synth.pipeline import runner

    class FakeRunner:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(ok=True, summary=lambda: "fixture")

    monkeypatch.setattr(runner, "PipelineRunner", FakeRunner)


def test_other_source_run_keeps_authorization_and_jobs(monkeypatch, capsys):
    calls = []
    _fake_runner(monkeypatch, calls)
    assert cli.main([
        "run", "not-created", "--source", "prism", "--jobs", "4",
        "--authorized-by", "not-read.md", "--stop-after", "generate",
    ]) == 0
    assert calls[0]["source_name"] == "prism"
    assert calls[1]["jobs"] == 4
    assert calls[1]["authorized_by"] == Path("not-read.md")
    assert calls[1]["stop_after"] == "generate"
    assert capsys.readouterr().out.strip() == "fixture"


def test_run_takes_the_corpus_and_generator_arguments(monkeypatch, capsys):
    """The bundle directory is optional (the lineage under the data root), and a sample run
    passes the corpus and the generator arguments through."""
    calls = []
    _fake_runner(monkeypatch, calls)
    assert cli.main([
        "run", "--source", "gaitex", "--stop-after", "generate", "--corpus", "scratch-corpus",
        "--rebuild-corpus", "--corpus-arg=--subjects", "--corpus-arg=austra",
        "--bundle-arg=--subjects", "--bundle-arg", "austra",
    ]) == 0
    assert calls[0]["dataset_dir"] is None
    assert calls[0]["corpus_dir"] == Path("scratch-corpus")
    assert calls[1]["rebuild_corpus"] is True
    assert calls[1]["corpus_args"] == ["--subjects", "austra"]
    assert calls[1]["bundle_args"] == ["--subjects", "austra"]
    capsys.readouterr()


def test_run_reports_an_unconfigured_data_root_instead_of_a_traceback(monkeypatch, capsys):
    monkeypatch.delenv("SOMA_DATA_ROOT", raising=False)
    assert cli.main(["run", "--source", "prism", "--skip-generate"]) == 2
    assert "SOMA_DATA_ROOT is not set" in capsys.readouterr().err
