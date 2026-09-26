"""The runner's obligations: generate only what a recorded decision covers, and always leave a
record.

Under ADR-0041 a run without an owner record generates under the registry's standing
decision and records that decision. The gates (``gates.py``) and the owner-record path read the
parent project's M0 evaluator and decision records from the governance root, which exists only
when soma-synth is mounted in the parent project as ``packages/soma-synth``. Every test here
starts from a standalone checkout (no governance root, whichever layout the suite runs from);
the ones that need a governance root build a scratch one. The tests of the live gates and of an
owner record read against them stay in the parent project. No test here runs a real generator:
the generate step is always a fixture command.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from soma_synth.pipeline import gates as gates_mod
from soma_synth.pipeline import provenance as provenance_mod
from soma_synth.pipeline import run_identity as rid
from soma_synth.pipeline import runner as runner_mod
from soma_synth.pipeline import stages as stages_mod
from soma_synth.pipeline.run_directory import RunDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]
#: What gates.py resolved when it was imported, before any test patched it.
IMPORTED_GOVERNANCE_ROOT = gates_mod.GOVERNANCE_ROOT


@pytest.fixture(autouse=True)
def _standalone_checkout(monkeypatch):
    """No governance root unless a test builds one: the suite says the same thing whether it runs
    from a standalone checkout or from the parent project's packages/soma-synth mount."""
    monkeypatch.setattr(gates_mod, "GOVERNANCE_ROOT", None)

#: A generate command that writes the one field set the runner checks after generation, and
#: nothing else: the stand-in for a generator in every runner test of this file.
WRITE_INDEX = (
    "import json, pathlib, sys; d = pathlib.Path(sys.argv[1]); d.mkdir(parents=True, "
    "exist_ok=True); (d / 'INDEX.json').write_text(json.dumps({'artifact_class': "
    "'experimental_non_candidate', 'distribution_scope': 'internal_only', 'takes': []}))"
)


def _fixture_generator(dataset: Path) -> list[str]:
    return [sys.executable, "-c", WRITE_INDEX, str(dataset)]


# ------------------------------------------------------------------- gates
def _parent_layout(parent: Path, *, evaluators: bool = True, decisions: bool = True,
                   mount: Path = gates_mod.MOUNT_RELPATH) -> Path:
    """A scratch parent project with soma-synth mounted at ``mount``; returns the mount."""
    checkout = parent / mount
    checkout.mkdir(parents=True)
    if evaluators:
        (parent / "configs" / "evaluators").mkdir(parents=True)
    if decisions:
        (parent / "research" / "decisions").mkdir(parents=True)
    return checkout


#: A gate config that still says no, as the parent's does while the holds stand.
HELD_GATES = {
    "authority_boundary": {"effect": "READ_ONLY_AUDIT_AUTHORIZED_GENERATION_STILL_STAGED"},
    "state_machines": {"runtime": {"current": "RUNTIME_APPROVAL_HOLD"},
                       "hknu": {"current": "HKNU_SOURCE_HOLD"}},
    "derived_states": {"ALL_SOURCE_PASS": {"current": False}},
}


def _governance_root(tmp_path: Path, monkeypatch) -> Path:
    """A scratch governance root with an evaluator, the gate config it pins and a decisions
    folder, made this module's GOVERNANCE_ROOT."""
    parent = tmp_path / "parent"
    _parent_layout(parent)
    (parent / "configs" / "gates").mkdir(parents=True)
    (parent / "configs" / "gates" / "gates_test.yaml").write_text(
        json.dumps(HELD_GATES), encoding="utf-8")
    (parent / "configs" / "evaluators" / "m0_v1_3.yaml").write_text(
        json.dumps({"gate_config": {"path": "configs/gates/gates_test.yaml"}}), encoding="utf-8")
    monkeypatch.setattr(gates_mod, "GOVERNANCE_ROOT", parent)
    return parent


def test_the_module_resolved_the_governance_root_from_this_checkouts_layout():
    """Nothing but the layout decides it: a standalone checkout has none, a mounted one has the
    parent project's root."""
    assert IMPORTED_GOVERNANCE_ROOT == gates_mod.resolve_governance_root(REPO_ROOT)
    if REPO_ROOT.parent.name != "packages":
        assert IMPORTED_GOVERNANCE_ROOT is None


def test_a_checkout_mounted_as_packages_soma_synth_resolves_to_the_parent(tmp_path: Path):
    checkout = _parent_layout(tmp_path)
    assert checkout == tmp_path / "packages" / "soma-synth"
    assert gates_mod.resolve_governance_root(checkout) == tmp_path


@pytest.mark.parametrize("layout", [
    {"mount": Path("soma-synth")},                           # a standalone checkout beside them
    {"mount": Path("packages") / "soma_synth"},              # not the mount name
    {"mount": Path("vendor") / "soma-synth"},                # not under packages/
    {"evaluators": False},                                   # a parent without the evaluator
    {"decisions": False},                                    # a parent without decision records
])
def test_any_other_layout_has_no_governance_root(tmp_path: Path, layout):
    """Fail-closed: the parent's two folders must both be there, and the checkout must sit at
    <parent>/packages/soma-synth. There is no variable that says otherwise."""
    checkout = _parent_layout(tmp_path, **layout)
    assert gates_mod.resolve_governance_root(checkout) is None


def test_this_checkout_has_no_governance_root():
    """The evaluator and the decision records are the parent project's; soma-synth has neither."""
    assert gates_mod.GOVERNANCE_ROOT is None
    assert gates_mod.governance_evaluator_dir() is None
    assert gates_mod.governance_decisions_dir() is None
    assert "governance root is absent" in gates_mod.governance_root_absent()
    assert "packages/soma-synth" in gates_mod.governance_root_absent()


def test_a_governance_root_missing_a_folder_is_absent(tmp_path: Path, monkeypatch):
    """A patched root is judged by the same two folders the layout check asks for."""
    _parent_layout(tmp_path, decisions=False)
    monkeypatch.setattr(gates_mod, "GOVERNANCE_ROOT", tmp_path)
    absent = gates_mod.governance_root_absent()
    assert "governance root is absent" in absent and "research/decisions" in absent
    with pytest.raises(gates_mod.GateError, match="governance root is absent"):
        gates_mod.generation_decision("hknu")


def test_the_gates_are_read_from_the_governance_root(tmp_path: Path, monkeypatch):
    """The evaluator, the gate config it pins and the decision recorded, all from the root."""
    root = _governance_root(tmp_path, monkeypatch)
    assert gates_mod.governance_root_absent() is None
    assert gates_mod.governance_evaluator_dir() == root / "configs" / "evaluators"
    assert gates_mod.governance_decisions_dir() == root / "research" / "decisions"
    decision = gates_mod.generation_decision("hknu")
    assert decision.allowed is False
    assert decision.evaluator == "m0_v1_3.yaml"
    assert decision.gate_config == "configs/gates/gates_test.yaml"
    assert "state_machines.hknu.current = 'HKNU_SOURCE_HOLD'" in decision.reasons
    assert "The parent project's generation hold forbids generating" in decision.refusal_message()


def test_a_relative_record_path_resolves_against_the_governance_root(tmp_path: Path, monkeypatch):
    """research/decisions/<record>.md names the parent's file from any working directory."""
    root = _governance_root(tmp_path, monkeypatch)
    _record(root)
    relative = Path("research") / "decisions" / "2026-09-14_test_authorization.md"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert gates_mod.resolve_record_path(relative) == root / relative
    assert gates_mod.resolve_record_path(root / relative) == root / relative

    authorization = gates_mod.read_authorization(relative)       # default: the root's decisions
    assert authorization.record == relative.as_posix()
    decision = gates_mod.generation_decision("hknu", authorized_by=relative)
    assert decision.allowed is True
    assert decision.authorization is not None
    assert decision.authorization.record == relative.as_posix()


def test_without_a_governance_root_a_relative_record_path_is_kept_and_refused(tmp_path: Path):
    relative = Path("research") / "decisions" / "x.md"
    assert gates_mod.resolve_record_path(relative) == relative
    with pytest.raises(gates_mod.GateError, match="governance root is absent"):
        gates_mod.read_authorization(relative)


@pytest.mark.parametrize("source", ["prism", "amass", "gaitex", "hknu", "addbiomechanics"])
def test_without_the_governance_root_the_gate_check_refuses_and_says_why(source):
    """Not a hardcoded no and not a crash: an error naming what is missing, for every source."""
    with pytest.raises(gates_mod.GateError, match="governance root is absent"):
        gates_mod.generation_decision(source)


def test_without_the_governance_root_an_owner_record_is_not_read(tmp_path: Path):
    """A well-formed record in a decisions folder still cannot authorise anything here: the gates
    it overrides cannot be read, so the answer is the absent root, not an allowed run."""
    decisions, record = _record(tmp_path)
    with pytest.raises(gates_mod.GateError, match="governance root is absent"):
        gates_mod.generation_decision("hknu", authorized_by=record, decisions_dir=decisions)


# ------------------------------------------------------------------ runner
def _runner(tmp_path: Path, source: str = "gaitex", registry=None) -> runner_mod.PipelineRunner:
    dataset = tmp_path / "bundle_v1"
    dataset.mkdir(parents=True, exist_ok=True)
    return runner_mod.PipelineRunner(
        source_name=source, dataset_dir=dataset, runs_root=tmp_path / "_runs",
        data_root=tmp_path / "data", registry=registry,
    )


def _v1_registry():
    return stages_mod.load(stages_mod.REGISTRY_V1_PATH)


def test_without_a_record_the_runner_takes_the_standing_decision_and_records_it(tmp_path: Path):
    """ADR-0041: no owner record per run. The run generates, and its manifest names the decision
    it generated under -- the basis, the ADR, the decision record, the class, and that it
    releases no hold."""
    runner = _runner(tmp_path)
    result = runner.run(stop_after="generate", generate_cmd=_fixture_generator(runner.dataset_dir),
                        log=lambda _msg: None)

    assert [o.status for o in result.outcomes] == ["ok"]
    assert RunDirectory.missing_artifacts(result.run_directory) == []
    manifest = json.loads((result.run_directory / "manifest.json").read_text(encoding="utf-8"))
    decision = manifest["generation_decision"]
    assert decision["allowed"] is True
    assert decision["basis"] == "standing_decision"
    assert decision["adr"] == "ADR-0041"
    assert decision["decision_record"] == (
        "research/decisions/2026-09-25_source_distribution_and_pipeline_split_decision.md")
    assert decision["decision_section"] == "§4"
    assert decision["artifact_class"] == "experimental_non_candidate"
    assert decision["distribution_scope"] == "internal_only"
    assert decision["releases_holds"] is False
    assert decision["authorization"] is None
    # the decision record is the parent project's: this checkout records why it has no hash
    assert decision["decision_record_sha256"] == {
        "status": provenance_mod.UNAVAILABLE,
        "reason": f"{decision['decision_record']} is not in this checkout"}
    assert manifest["tracing"]["decision_record"] == decision["decision_record"]
    assert manifest["tracing"]["generation_basis"] == "standing_decision"
    metrics = json.loads((result.run_directory / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stage_counts"]["generation_standing_decision"] == 1
    assert not (result.run_directory / "logs" / "authorization.txt").exists()


@pytest.mark.parametrize("present", [True, False])
def test_the_decision_record_is_hashed_from_the_governance_root(tmp_path: Path, monkeypatch,
                                                                 present):
    """Mounted in the parent project, the standing decision's record is the parent's file and its
    hash is recorded; a governance root without it says so instead of inventing one."""
    root = _governance_root(tmp_path, monkeypatch)
    policy = stages_mod.default_registry().generation_policy
    if present:
        (root / policy.decision_record).write_text("# the decision\n", encoding="utf-8")
    runner = _runner(tmp_path / "work")
    result = runner.run(stop_after="generate", generate_cmd=_fixture_generator(runner.dataset_dir),
                        log=lambda _msg: None)
    assert [o.status for o in result.outcomes] == ["ok"]
    manifest = json.loads((result.run_directory / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest["generation_decision"]["decision_record_sha256"]
    if present:
        assert recorded == provenance_mod.file_sha256(root / policy.decision_record)
    else:
        assert recorded == {
            "status": provenance_mod.UNAVAILABLE,
            "reason": f"{policy.decision_record} is not under the governance root {root}"}


def test_mounted_in_the_parent_a_relative_owner_record_authorises_the_run(tmp_path: Path,
                                                                           monkeypatch):
    """--authorized-by research/decisions/<record>.md, from any working directory: the gates,
    the recorded intent and the run's copy all read the governance root's file."""
    root = _governance_root(tmp_path, monkeypatch)
    _, record = _record(root)
    relative = record.relative_to(root)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    dataset = work / "bundle"
    runner = runner_mod.PipelineRunner(
        source_name="hknu", dataset_dir=dataset, runs_root=work / "_runs",
        data_root=work / "data", corpus_dir=work / "corpus",
    )
    result = runner.run(stop_after="generate", authorized_by=relative,
                        generate_cmd=_fixture_generator(dataset), log=lambda _msg: None)
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    manifest = json.loads((result.run_directory / "manifest.json").read_text(encoding="utf-8"))
    decision = manifest["generation_decision"]
    assert decision["basis"] == "owner_record"
    assert decision["authorization"]["record"] == relative.as_posix()
    assert decision["consulted_gate_config"] == "configs/gates/gates_test.yaml"
    assert decision["authorization_record_sha256"] == provenance_mod.file_sha256(record)
    copy = result.run_directory / decision["authorization_record_copy"]
    assert copy.read_bytes() == record.read_bytes()
    assert (result.run_directory / "logs" / "authorization.txt").exists()


def test_a_v1_registry_has_no_standing_decision_and_is_refused_here(tmp_path: Path):
    """The refusal is the interesting path: a run that produced nothing must still be a run
    somebody can find afterwards. A registry without a generation_policy (v1) falls back to the
    gates, which this checkout cannot read: the run is refused before any decision is taken."""
    runner = _runner(tmp_path, registry=_v1_registry())
    result = runner.run(stop_after="generate", generate_cmd=_fixture_generator(runner.dataset_dir),
                        log=lambda _msg: None)

    assert [o.status for o in result.outcomes] == ["refused"]
    assert RunDirectory.missing_artifacts(result.run_directory) == []
    assert (result.run_directory / "logs" / "refusal.txt").exists()
    assert not (runner.dataset_dir / "INDEX.json").exists()        # nothing ran
    manifest = json.loads((result.run_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_decision"] is None          # never got as far as a decision


def test_the_refusal_is_written_where_it_can_be_read_later(tmp_path: Path):
    result = _runner(tmp_path, registry=_v1_registry()).run(
        stop_after="generate", log=lambda _msg: None)
    text = (result.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert "not authorised" in text
    assert "governance root is absent" in text


def _registry_declaring(tmp_path: Path, source: str, artifact_class: str):
    import yaml

    raw = yaml.safe_load(stages_mod.DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
    raw["sources"][source]["artifact_class"] = artifact_class
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return stages_mod.load(path)


@pytest.mark.parametrize("artifact_class", ["canonical", "internal_candidate", "candidate"])
def test_a_source_declaring_another_class_is_refused_before_anything_runs(tmp_path: Path,
                                                                           artifact_class):
    """The standing decision covers experimental_non_candidate only; a source whose registry entry
    says anything else is refused, and the owner-record path does not get round it either."""
    registry = _registry_declaring(tmp_path, "hknu", artifact_class)
    runner = _runner(tmp_path, source="hknu", registry=registry)
    result = runner.run(stop_after="generate", generate_cmd=_fixture_generator(runner.dataset_dir),
                        log=lambda _msg: None)
    assert [o.status for o in result.outcomes] == ["refused"]
    assert not (runner.dataset_dir / "INDEX.json").exists()
    text = (result.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert artifact_class in text and "experimental_non_candidate" in text
    manifest = json.loads((result.run_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_decision"]["allowed"] is False
    assert manifest["generation_decision"]["basis"] == "artifact_class"

    decisions, record = _record(tmp_path)
    runner = runner_mod.PipelineRunner(
        source_name="hknu", dataset_dir=tmp_path / "other", runs_root=tmp_path / "_runs2",
        data_root=tmp_path / "data", registry=registry, decisions_dir=decisions,
    )
    result = runner.run(stop_after="generate", authorized_by=record,
                        generate_cmd=_fixture_generator(runner.dataset_dir), log=lambda _msg: None)
    assert [o.status for o in result.outcomes] == ["refused"]


@pytest.mark.parametrize("index, fragment", [
    ({"artifact_class": "canonical", "distribution_scope": "internal_only"}, "artifact_class"),
    ({"artifact_class": "experimental_non_candidate", "distribution_scope": "public"},
     "distribution_scope"),
    (None, "does not exist"),
])
def test_a_bundle_that_says_another_class_fails_the_generate_step(tmp_path: Path, index, fragment):
    runner = _runner(tmp_path)
    if index is None:
        command = [sys.executable, "-c", "pass"]
    else:
        write = ("import pathlib, sys; "
                 "pathlib.Path(sys.argv[1], 'INDEX.json').write_text(sys.argv[2])")
        command = [sys.executable, "-c", write, str(runner.dataset_dir), json.dumps(index)]
    result = runner.run(stop_after="generate", generate_cmd=command, log=lambda _msg: None)
    assert [o.status for o in result.outcomes] == ["failed"]
    assert fragment in result.outcomes[0].detail
    metrics = json.loads((result.run_directory / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stage_counts"]["artifact_class_mismatch"] == 1


def test_skipping_generation_is_recorded_as_a_skip_not_a_success(tmp_path: Path):
    result = _runner(tmp_path).run(
        stop_after="generate", skip_generate=True, log=lambda _msg: None
    )
    assert result.outcomes[0].status == "skipped"
    assert result.outcomes[0].detail == "--skip-generate"


def test_an_unexpected_failure_still_closes_the_record(tmp_path: Path):
    """A traceback on the terminal and nothing on disk is the situation this package ends."""
    run = _runner(tmp_path)
    run._step_validate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    result = run.run(start_at="validate", stop_after="validate", log=lambda _msg: None)

    assert any(o.status == "failed" for o in result.outcomes)
    assert RunDirectory.missing_artifacts(result.run_directory) == []
    assert (result.run_directory / "logs" / "error.txt").exists()


def test_the_run_identity_reflects_the_input_corpus_not_just_the_source(tmp_path: Path):
    """The gaitex corpus keeps its summary in _run.json rather than SUMMARY.json. Missing that
    once left the identity blind to which corpus it read and the record claiming no source
    asset at all."""
    assert "_run.json" in rid.CORPUS_SUMMARY_NAMES

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "_run.json").write_text('{"a": 1}', encoding="utf-8")
    assert rid.corpus_fingerprint(corpus) is not None
    assert rid.corpus_fingerprint(tmp_path / "empty") is None


def test_the_steps_are_the_four_the_cli_has_always_had_and_push_is_not_among_them():
    assert runner_mod.STEPS == ("generate", "validate", "readme", "register")
    assert "push" not in runner_mod.STEPS


def test_an_unknown_source_is_refused_before_any_directory_is_created(tmp_path: Path):
    with pytest.raises(stages_mod.PipelineError):
        runner_mod.PipelineRunner(
            source_name="no_such_source", dataset_dir=tmp_path / "b", runs_root=tmp_path / "_runs",
            data_root=tmp_path,
        )
    assert not (tmp_path / "_runs").exists()


def test_a_ledger_trusted_take_is_counted_separately_from_written_and_skipped(tmp_path: Path):
    """A run that accepted 8,261 takes on a prior run's evidence reported zero of everything,
    because trusted was folded into neither counter. It is a third outcome and gets its own."""
    root = tmp_path / "run_t"
    record = provenance_mod.ProvenanceRecord(
        source_name="amass", spec_id="qmd_unified8_smpl18", spec_version="faithful-v2"
    )
    inputs = rid.RunIdentityInputs(
        spec_id="qmd_unified8_smpl18", spec_version="faithful-v2",
        source_name="amass", entrypoint="scripts/poc/generate_amass_faithful_all.py",
    )
    run = RunDirectory(root=root, identity_inputs=inputs, provenance=record).open()
    run.close(takes_written=0, takes_trusted=8261)

    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["takes_trusted"] == 8261
    assert metrics["takes_written"] == 0
    assert metrics["takes_skipped"] == 0


def test_the_record_carries_provenance_for_the_source_it_ran(tmp_path: Path):
    result = _runner(tmp_path, source="hknu").run(
        stop_after="generate", skip_generate=True, log=lambda _msg: None
    )
    manifest = json.loads((result.run_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_name"] == "hknu"
    assert manifest["provenance"]["seeds"]["present"] is False
    assert manifest["distribution_scope"] == "internal_only"


# ------------------------------------------------------------- parallelism
def test_jobs_reach_a_generator_the_way_its_registry_entry_says(tmp_path: Path):
    """amass fans out into shards and a merge; addbio takes a pool flag; prism runs alone."""
    # the one launch implementation; cli.plan_generate_commands is this same object
    from soma_synth.pipeline import generate as cli

    shards, merge = cli.plan_generate_commands("amass", tmp_path / "amass_faithful_full", None, jobs=8)
    assert len(shards) == 8
    assert shards[0][-2:] == ["--shard", "0/8"] and shards[7][-2:] == ["--shard", "7/8"]
    assert merge[-1] == "--merge"
    assert all(argv[:-2] == merge[:-1] for argv in shards)          # one base command line

    pooled, follow = cli.plan_generate_commands("addbiomechanics", tmp_path / "addbio_unified8", None, jobs=6)
    assert len(pooled) == 1 and pooled[0][-2:] == ["--jobs", "6"] and follow is None

    alone, follow = cli.plan_generate_commands("prism", tmp_path / "prism_faithful_full", None, jobs=6)
    assert len(alone) == 1 and "--jobs" not in alone[0] and follow is None

    one, follow = cli.plan_generate_commands("amass", tmp_path / "amass_faithful_full", None, jobs=1)
    assert len(one) == 1 and "--shard" not in one[0] and follow is None


@pytest.mark.parametrize("name", ["source_pipelines_v1.yaml", "source_pipelines_v2.yaml"])
def test_an_unknown_parallel_flag_shape_is_refused_at_load(tmp_path: Path, name):
    text = (REPO_ROOT / "configs" / "datasets" / name).read_text(encoding="utf-8")
    broken = text.replace("        jobs: \"--jobs\"", "        jobs: jobs", 1)
    assert broken != text
    path = tmp_path / "registry.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(stages_mod.PipelineError, match="parallel_args"):
        stages_mod.load(path)


# ------------------------------------------------------- owner authorization
AUTHORIZATION = """# 2026-09-14 test record

```yaml authorization
schema: generation_authorization_v1
source_name: {source}
artifact_class: {artifact_class}
authorized_on: 2026-09-14
authorized_by: project owner
scope: one experimental generation, for the test
releases_holds: {releases}
```
"""


def _record(tmp_path: Path, *, source: str = "hknu",
            artifact_class: str = "experimental_non_candidate",
            releases: str = "false") -> tuple[Path, Path]:
    decisions = tmp_path / "research" / "decisions"
    decisions.mkdir(parents=True, exist_ok=True)
    record = decisions / "2026-09-14_test_authorization.md"
    record.write_text(
        AUTHORIZATION.format(source=source, artifact_class=artifact_class, releases=releases),
        encoding="utf-8",
    )
    return decisions, record


# What a record says is read without the gates; checking it against them needs the governance
# root, and those tests stay in the parent project.
def test_a_record_parses_and_names_its_source_and_class(tmp_path: Path):
    decisions, record = _record(tmp_path)
    authorization = gates_mod.read_authorization(record, decisions_dir=decisions)
    assert authorization.source_name == "hknu"
    assert authorization.artifact_class == "experimental_non_candidate"
    assert authorization.authorized_on == "2026-09-14"
    assert authorization.as_json()["releases_holds"] is False


def test_a_record_that_claims_to_release_a_hold_is_refused(tmp_path: Path):
    decisions, record = _record(tmp_path, releases="true")
    with pytest.raises(gates_mod.GateError, match="releases_holds"):
        gates_mod.read_authorization(record, decisions_dir=decisions)


def test_a_file_outside_research_decisions_is_not_a_record(tmp_path: Path):
    decisions, record = _record(tmp_path)
    elsewhere = tmp_path / "elsewhere.md"
    elsewhere.write_text(record.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(gates_mod.GateError, match="must live under"):
        gates_mod.read_authorization(elsewhere, decisions_dir=decisions)


def test_a_record_without_the_machine_readable_block_is_refused(tmp_path: Path):
    decisions = tmp_path / "research" / "decisions"
    decisions.mkdir(parents=True)
    record = decisions / "2026-09-14_prose_only.md"
    record.write_text("# I authorise it\n\nsource hknu, experimental\n", encoding="utf-8")
    with pytest.raises(gates_mod.GateError, match="no ```yaml authorization block"):
        gates_mod.read_authorization(record, decisions_dir=decisions)


@pytest.mark.parametrize("source", ["hknu", "amass"])
def test_the_runner_turns_an_unreadable_owner_record_into_a_refusal_not_a_crash(tmp_path: Path,
                                                                                 source):
    """With --authorized-by the run asks for the owner-record path, which needs the governance
    root: whatever the record says (this source or another), the run is refused, says why, and
    generates nothing. The standing decision is not taken in its place."""
    decisions, record = _record(tmp_path, source=source)
    # a scratch corpus too: a scratch bundle may not build the production corpus lineage
    dataset = tmp_path / "bundle"
    runner = runner_mod.PipelineRunner(
        source_name="hknu", dataset_dir=dataset, runs_root=tmp_path / "_runs",
        decisions_dir=decisions, data_root=tmp_path / "data", corpus_dir=tmp_path / "corpus",
    )
    result = runner.run(stop_after="generate", authorized_by=record,
                        generate_cmd=_fixture_generator(dataset), log=lambda _msg: None)
    assert [o.status for o in result.outcomes] == ["refused"]
    text = (result.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert "governance root is absent" in text
    assert not (dataset / "INDEX.json").exists()
    manifest = json.loads((result.run_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_decision"] is None          # never got as far as a decision


def test_the_runner_renders_the_limitations_file_for_a_bundle_the_config_knows(tmp_path: Path):
    """After a generator returns, the file the validator requires is written from the config."""
    from soma_synth.contracts import dataset_limitations as dl

    dataset = tmp_path / "hknu_unified8"
    dataset.mkdir()
    runner = runner_mod.PipelineRunner(
        source_name="hknu", dataset_dir=dataset, runs_root=tmp_path / "_runs",
        data_root=tmp_path / "data", corpus_dir=tmp_path / "corpus",
    )
    result = runner.run(
        stop_after="generate", generate_cmd=_fixture_generator(dataset), log=lambda _msg: None,
    )
    assert result.outcomes[0].status == "ok"
    manifest = json.loads((result.run_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_decision"]["basis"] == "standing_decision"
    assert dl.problems_in_file(dataset / dl.FILE_NAME, "hknu_unified8", dl.load_config()) == []
