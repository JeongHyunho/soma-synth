"""`pipeline.generate`: the generator launch the runner uses, without the CLI.

The launch logic lives in ``pipeline.generate`` and ``cli.py`` imports it, so the runner never
imports the CLI. These tests pin that the runner does not reach the CLI, that the CLI's names are
the pipeline's objects, and the launch behaviour itself.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from soma_synth import cli
from soma_synth.pipeline import generate
from soma_synth.pipeline import stages as stages_mod

REPO = Path(__file__).resolve().parents[2]
PIPELINE = REPO / "src" / "soma_synth" / "pipeline"
CLI = REPO / "src" / "soma_synth" / "cli.py"


def _registered_sources() -> list[str]:
    return sorted(stages_mod.default_registry().sources)


def _imports(path: Path) -> set[str]:
    """Every module an ``import`` or ``from ... import`` in the file names, at any depth."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


@pytest.mark.parametrize("module", ["runner.py", "generate.py"])
def test_no_import_of_the_cli_anywhere_in_the_file(module):
    """Function-level imports included: the old one sat inside ``_step_generate``."""
    found = sorted(n for n in _imports(PIPELINE / module)
                   if n == "soma_synth.cli" or n.startswith("soma_synth.cli."))
    assert not found, f"pipeline/{module} imports the CLI: {found}"


def test_importing_the_runner_does_not_load_the_cli():
    """A clean interpreter, so a CLI another test imported cannot hide or fake the result."""
    code = (
        "import json, sys\n"
        "import soma_synth.pipeline.runner\n"
        "import soma_synth.pipeline.generate\n"
        "print(json.dumps(sorted(m for m in sys.modules if m == 'soma_synth.cli'\n"
        "                        or m.startswith('soma_synth.cli.'))))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO / "packages" / "smpl18" / "src")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


# ----------------------------------------------------------------- one copy
@pytest.mark.parametrize("name", ["run_generate", "plan_generate_commands",
                                  "_default_generate_cmd", "_refuse_paired_weargait",
                                  "bundle_class_problems"])
def test_the_cli_uses_the_pipeline_objects_themselves(name):
    """Not a second copy held equal by a test: the very function objects of pipeline.generate."""
    assert getattr(cli, name) is getattr(generate, name)


@pytest.mark.parametrize("name", ["run_generate", "plan_generate_commands",
                                  "_default_generate_cmd", "_refuse_paired_weargait"])
def test_the_cli_defines_no_launch_function_of_its_own(name):
    defined = {node.name for node in ast.parse(CLI.read_text(encoding="utf-8")).body
               if isinstance(node, ast.FunctionDef)}
    assert name not in defined


def test_the_runner_launches_through_pipeline_generate(monkeypatch, tmp_path):
    """The seam a test replaces is ``pipeline.generate.run_generate``, not the CLI's."""
    from soma_synth.pipeline import runner as runner_mod

    calls = []

    def fake_run_generate(source, dataset_dir, data_root, generate_cmd, jobs=1, **kwargs):
        calls.append((source, Path(dataset_dir), data_root, generate_cmd, jobs, kwargs))
        return 5

    def forbidden(*args, **kwargs):
        pytest.fail("the runner reached cli.run_generate")

    monkeypatch.setattr(generate, "run_generate", fake_run_generate)
    monkeypatch.setattr(cli, "run_generate", forbidden)

    runner = runner_mod.PipelineRunner(source_name="prism", data_root=tmp_path,
                                       dataset_dir=tmp_path / "prism_faithful_full")

    class Recorder:
        """Stands in for the RunDirectory; a failed generator never reaches the limitations."""

        root = tmp_path

        def record_stage(self, *_):
            pass

        def record_generation_decision(self, *_):
            pass

        def record_bundle(self, *_):
            pass

    outcome = runner._step_generate(Recorder(), skip=False, generate_cmd=["fixture"],
                                    authorized_by=None, jobs=3)
    assert outcome.status == "failed" and outcome.detail == "generator exited 5"
    assert len(calls) == 1
    source, dataset, data_root, command, jobs, kwargs = calls[0]
    assert (source, dataset, data_root, command, jobs) == (
        "prism", tmp_path / "prism_faithful_full", tmp_path, ["fixture"], 3)
    # the runner hands its data root to the generator as SOMA_DATA_ROOT
    assert kwargs["env"]["SOMA_DATA_ROOT"] == str(tmp_path)


# ------------------------------------------------------------------ planning
@pytest.mark.parametrize("jobs", [1, 2, 8])
@pytest.mark.parametrize("data_root", [None, "data-root"])
def test_every_registered_source_plans_its_registry_command(tmp_path, jobs, data_root):
    root = (tmp_path / data_root) if data_root else None
    sources = _registered_sources()
    assert sources, "the registry declares no source"
    for source in sources:
        entry = stages_mod.default_registry().for_source(source).entrypoint
        dataset = tmp_path / f"{source}_bundle"
        concurrent, follow_up = generate.plan_generate_commands(source, dataset, root, jobs=jobs)
        base = generate._default_generate_cmd(source, dataset, root)
        assert base[1] == str(REPO / entry.script)
        assert base[2:4] == [entry.output_arg, str(dataset)]
        assert all(argv[:len(base)] == base for argv in concurrent), source
        if follow_up is not None:
            assert follow_up[:len(base)] == base


def test_extra_arguments_follow_the_output_argument_on_every_command(tmp_path):
    extra = ["--raw", str(tmp_path / "corpus"), "--subjects", "2"]
    concurrent, follow_up = generate.plan_generate_commands(
        "amass", tmp_path / "amass_faithful_full", None, jobs=2, extra_args=extra)
    assert [argv[4:8] for argv in concurrent] == [extra, extra]
    assert follow_up[4:8] == extra and follow_up[-1] == "--merge"
    single = generate._default_generate_cmd("addbiomechanics", tmp_path / "b", None,
                                            extra_args=extra)
    assert single[-4:] == extra


def test_prism_writes_through_the_bundle_directory_in_registry_v2(tmp_path):
    """v1 wrote through --data-root, which reads <root>/extracted/prism whatever SOMA_SOURCE_ROOT
    says; v2 writes through --out and leaves the source to SOMA_SOURCE_ROOT."""
    command = generate._default_generate_cmd("prism", tmp_path / "prism_faithful_full", tmp_path)
    assert command[2:4] == ["--out", str(tmp_path / "prism_faithful_full")]


def test_a_given_registry_replaces_the_default(tmp_path):
    v1 = stages_mod.load(stages_mod.REGISTRY_V1_PATH)
    command = generate._default_generate_cmd("prism", tmp_path / "b", tmp_path, registry=v1)
    assert command[2:4] == ["--data-root", str(tmp_path)]


def test_an_unknown_source_is_refused_the_same_way(tmp_path):
    assert generate.plan_generate_commands("no_such_source", tmp_path, None) is None
    assert generate._default_generate_cmd("no_such_source", tmp_path, None) is None
    assert cli._default_generate_cmd("no_such_source", tmp_path, None) is None


# ------------------------------------------------------------------ running
@pytest.mark.parametrize("source", ["weargait_pd", "weargait-pd", "WearGait_PD"])
def test_paired_weargait_is_refused_before_any_process(monkeypatch, capsys, source):
    def forbidden(*args, **kwargs):
        pytest.fail("paired WearGait generation attempted a subprocess")

    monkeypatch.setattr(generate.subprocess, "run", forbidden)
    monkeypatch.setattr(generate.subprocess, "Popen", forbidden)
    assert generate.run_generate(source, "not-created", None, ["not-executed"], jobs=4) == 2
    assert "estimated" in capsys.readouterr().err


@pytest.mark.parametrize("override", [["fixture"], "fixture --argument"])
def test_an_explicit_command_runs_as_given(monkeypatch, capsys, override):
    calls = []

    def fake_run(command, *, shell=False, env=None, check=False):
        calls.append((command, shell, env, check))
        return type("Done", (), {"returncode": 7})()

    monkeypatch.setattr(generate.subprocess, "run", fake_run)
    assert generate.run_generate("prism", "not-created", None, override) == 7
    assert calls == [(override, isinstance(override, str), None, False)]
    capsys.readouterr()


def test_the_child_environment_reaches_every_process(monkeypatch):
    seen = []

    def fake_run(command, *, env=None, check=False):
        seen.append(env)
        return type("Done", (), {"returncode": 0})()

    def spawn(command, *, env=None):
        seen.append(env)
        return type("Proc", (), {"wait": staticmethod(lambda: 0)})()

    monkeypatch.setattr(generate.subprocess, "run", fake_run)
    monkeypatch.setattr(generate.subprocess, "Popen", spawn)
    env = {"SOMA_DATA_ROOT": "fixture"}
    assert generate.run_generate("amass", "not-created", None, None, jobs=2, env=env) == 0
    assert seen == [env, env, env]                 # two shards and the merge
    assert generate.run_command(["fixture"], env=env) == 0
    assert seen[-1] == env


@pytest.mark.parametrize("codes", [(0, 0), (0, 7), (5, 0)])
def test_shards_merge_only_after_every_shard_succeeds(monkeypatch, codes):
    shards = [["fixture", "--shard", "0/2"], ["fixture", "--shard", "1/2"]]
    follow_up = ["fixture", "--merge"]
    started, merged = [], []

    def spawn(command, *, env=None):
        code = codes[len(started)]
        started.append(command)
        return type("Proc", (), {"wait": staticmethod(lambda: code)})()

    def run(command, *, env=None, check=False):
        merged.append(command)
        return type("Done", (), {"returncode": 11})()

    monkeypatch.setattr(generate, "plan_generate_commands",
                        lambda source, dataset_dir, data_root, **kwargs: (shards, follow_up))
    monkeypatch.setattr(generate.subprocess, "Popen", spawn)
    monkeypatch.setattr(generate.subprocess, "run", run)
    expected = next((code for code in codes if code), 11)
    assert generate.run_generate("amass", "not-created", None, None, jobs=2) == expected
    assert started == shards
    assert merged == ([] if any(codes) else [follow_up])


# -------------------------------------------------------------- class check
def _index(tmp_path: Path, **fields) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "INDEX.json").write_text(json.dumps(fields), encoding="utf-8")
    return bundle


def test_a_bundle_of_the_one_class_passes(tmp_path):
    bundle = _index(tmp_path, artifact_class="experimental_non_candidate",
                    distribution_scope="internal_only")
    assert generate.bundle_class_problems(bundle) == []


@pytest.mark.parametrize("fields, fragment", [
    ({"artifact_class": "canonical", "distribution_scope": "internal_only"}, "artifact_class"),
    ({"artifact_class": "experimental_non_candidate", "distribution_scope": "public"},
     "distribution_scope"),
    ({}, "artifact_class"),
])
def test_any_other_class_or_scope_is_a_problem(tmp_path, fields, fragment):
    problems = generate.bundle_class_problems(_index(tmp_path, **fields))
    assert problems and any(fragment in p for p in problems)


def test_a_bundle_without_an_index_cannot_be_confirmed(tmp_path):
    problems = generate.bundle_class_problems(tmp_path / "absent")
    assert problems and "does not exist" in problems[0]


# -------------------------------------------------------------- logs
#: A child that prints to both streams, a line that is not ASCII among them, and exits with a code.
_CHATTY = ("import sys; print('progress: take 1 of 2'); print('\\uac00\\ub098 \\u00e9', flush=True); "
           "print('warning: slow', file=sys.stderr); sys.exit(int(sys.argv[1]))")


def _child(code: int = 0, *extra: str) -> list[str]:
    return [sys.executable, "-c", _CHATTY, str(code), *extra]


@pytest.mark.parametrize("code", [0, 3])
def test_a_logged_command_is_teed_to_the_console_and_its_log(tmp_path, capsys, code):
    """The run record's logs/ stayed empty: the children wrote to the console only. With a log,
    both streams reach the console as before and the log holds the command, both streams and the
    exit code."""
    log = tmp_path / "logs" / "corpus.log"
    assert generate.run_command(_child(code), log_path=log) == code
    shown = capsys.readouterr()
    assert "progress: take 1 of 2" in shown.out and "가나 é" in shown.out
    assert "warning: slow" in shown.err
    text = log.read_text(encoding="utf-8")
    assert text.startswith(f"$ {sys.executable} -c ")
    for line in ("progress: take 1 of 2", "가나 é", "warning: slow"):
        assert line in text
    assert text.rstrip().endswith(f"[exit {code}]")


def test_without_a_log_the_child_inherits_the_console(monkeypatch):
    seen = []

    def fake_run(command, *, env=None, check=False):
        seen.append(command)
        return type("Done", (), {"returncode": 0})()

    def forbidden(*args, **kwargs):
        raise AssertionError("a pipe was opened without a log")

    monkeypatch.setattr(generate.subprocess, "run", fake_run)
    monkeypatch.setattr(generate.subprocess, "Popen", forbidden)
    assert generate.run_command(["fixture"]) == 0
    assert generate.run_generate("prism", "not-created", None, ["fixture"]) == 0
    assert seen == [["fixture"], ["fixture"]]


def test_the_bundle_entrypoint_its_shards_and_its_merge_are_logged(tmp_path, monkeypatch,
                                                                   capsys):
    shards = [_child(0, "--shard", "0/2"), _child(0, "--shard", "1/2")]
    monkeypatch.setattr(generate, "plan_generate_commands",
                        lambda *args, **kwargs: (shards, _child(0, "--merge")))
    logs = tmp_path / "logs"
    assert generate.run_generate("amass", "not-created", None, None, jobs=2, log_dir=logs) == 0
    capsys.readouterr()
    assert sorted(p.name for p in logs.iterdir()) == [
        "generate.log", "generate.shard1of2.log", "generate.shard2of2.log"]
    for index in (1, 2):
        text = (logs / f"generate.shard{index}of2.log").read_text(encoding="utf-8")
        assert f"--shard {index - 1}/2" in text and "[exit 0]" in text
    merge = (logs / "generate.log").read_text(encoding="utf-8")
    assert "--merge" in merge and "progress: take 1 of 2" in merge

    # one entrypoint, and an explicit command given as a string, go to generate.log as well
    single = tmp_path / "single"
    monkeypatch.setattr(generate, "plan_generate_commands", lambda *a, **k: ([_child(5)], None))
    assert generate.run_generate("prism", "not-created", None, None, log_dir=single) == 5
    command = (subprocess.list2cmdline(_child(0)) if os.name == "nt"
               else " ".join(f"'{part}'" for part in _child(0)))
    assert generate.run_generate("prism", "not-created", None, command, log_dir=single) == 0
    capsys.readouterr()
    text = (single / "generate.log").read_text(encoding="utf-8")
    assert text.count("$ ") == 2 and "[exit 5]" in text and "[exit 0]" in text


def test_an_interrupted_logged_shard_run_stops_every_shard(tmp_path, monkeypatch):
    started = []

    class Interrupted(generate._Teed):
        def wait(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt
            return super().wait(timeout)

    def teed(argv, **kwargs):
        started.append(Interrupted(argv, **kwargs))
        return started[-1]

    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    monkeypatch.setattr(generate, "plan_generate_commands",
                        lambda *a, **k: ([sleeper, sleeper], None))
    monkeypatch.setattr(generate, "_Teed", teed)
    with pytest.raises(KeyboardInterrupt):
        generate.run_generate("amass", "not-created", None, None, jobs=2, log_dir=tmp_path)
    assert len(started) == 2 and all(p.poll() is not None for p in started)
