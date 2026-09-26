"""Registry v2 and the runner (ADR-0041): defaults, the corpus stage, post-steps, tracing.

Every generator here is a fixture script written into ``tmp_path`` and registered in a copy of
``source_pipelines_v2.yaml`` whose entrypoints point at it; the data root is a scratch directory.
No test generates anything from real data or reads the data plane.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml
from smpl18.model import select as smpl18_select

from soma_synth.pipeline import body_models, paths
from soma_synth.pipeline import provenance as provenance_mod
from soma_synth.pipeline import runner as runner_mod
from soma_synth.pipeline import stages as stages_mod

REPO = Path(__file__).resolve().parents[2]
V1 = REPO / "configs" / "datasets" / "source_pipelines_v1.yaml"
V2 = REPO / "configs" / "datasets" / "source_pipelines_v2.yaml"
POC = "runs/experimental_generation_poc_demo"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser_options(script: Path) -> dict[str, bool]:
    """{option string: required} of every ``add_argument`` call in ``script``, read from its AST:
    importing a generator would import numpy and scipy and resolve the data plane."""
    options: dict[str, bool] = {}
    for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        names = [arg.value for arg in node.args
                 if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
        assert len(names) == len(node.args), f"{script.name}: {ast.unparse(node)}"
        required = any(keyword.arg == "required" and isinstance(keyword.value, ast.Constant)
                       and keyword.value.value is True for keyword in node.keywords)
        options.update({name: required for name in names})
    assert options, f"{script.name} declares no argparse option"
    return options


def _quiet(_message: str) -> None:
    pass


# ======================================================================= registry v2
class TestRegistryV2:
    def test_v2_is_the_default_and_v1_still_loads(self):
        default = stages_mod.default_registry()
        assert default.path == V2 and default.schema == "source_pipelines_v2"
        v1 = stages_mod.load(V1)
        assert v1.schema == "source_pipelines_v1" and v1.generation_policy is None
        assert set(v1.sources) == set(default.sources)

    def test_the_policy_is_the_standing_decision_of_adr_0041(self):
        policy = stages_mod.default_registry().generation_policy
        assert policy.basis == "standing_decision"
        assert policy.adr == "ADR-0041"
        assert policy.adr_path == "docs/adr/ADR-0041-soma-synth-split-and-teammate-generation.md"
        assert policy.decision_record == (
            "research/decisions/2026-09-25_source_distribution_and_pipeline_split_decision.md")
        # the record is the parent project's; a run here records that its hash is unavailable
        assert not (REPO / policy.decision_record).exists()
        assert policy.decision_section == "§4"
        assert policy.artifact_class == "experimental_non_candidate"
        assert policy.distribution_scope == "internal_only"
        assert policy.releases_holds is False

    def test_every_source_declares_its_class_and_bundle_lineage(self):
        registry = stages_mod.default_registry()
        lineages = {name: registry.for_source(name).bundle_lineage for name in registry.sources}
        assert lineages == {
            "amass": "amass_faithful_full", "prism": "prism_faithful_full",
            "hknu": "hknu_unified8", "gaitex": "gaitex_unified8",
            "addbiomechanics": "addbio_unified8",
        }
        for name in registry.sources:
            assert registry.for_source(name).artifact_class == "experimental_non_candidate"

    @pytest.mark.parametrize("source, script, lineage, fingerprint, flag", [
        ("hknu", "scripts/poc/generate_hknu_faithful.py", "hknu_smpl24_paired", "SUMMARY.json",
         "--paired"),
        ("gaitex", "scripts/poc/generate_gaitex_smpl24.py", "gaitex_smpl24", "_run.json",
         "--retarget"),
        ("addbiomechanics", "scripts/poc/generate_addbio_smpl24.py", "addbio_smpl24_raw",
         "SUMMARY.json", "--raw"),
    ])
    def test_the_three_corpora(self, source, script, lineage, fingerprint, flag):
        corpus = stages_mod.default_registry().for_source(source).corpus
        assert (corpus.script, corpus.lineage, corpus.fingerprint, corpus.bundle_arg) == (
            script, lineage, fingerprint, flag)
        assert corpus.output_arg == "--out"
        assert corpus.input_key == "retarget_corpus"
        assert (REPO / corpus.script).is_file()

    def test_amass_and_prism_have_no_corpus(self):
        registry = stages_mod.default_registry()
        assert registry.for_source("amass").corpus is None
        assert registry.for_source("prism").corpus is None

    def test_prism_writes_through_out_and_runs_the_heading_post_step(self):
        prism = stages_mod.default_registry().for_source("prism")
        assert (prism.entrypoint.output_arg, prism.entrypoint.output_arg_kind) == (
            "--out", "dataset_dir")
        assert [(s.script, s.args) for s in prism.post_steps] == [
            ("scripts/poc/synthesize_insole_heading.py", ("{dataset_dir}",))]
        assert (REPO / prism.post_steps[0].script).is_file()
        assert prism.post_steps[0].argv("py", REPO, Path("B"), None)[2:] == ["B"]

    def test_addbio_no_longer_names_the_cross_check_corpus(self):
        inputs = stages_mod.default_registry().for_source("addbiomechanics").inputs
        assert inputs == {"retarget_corpus": f"{POC}/addbio_smpl24_raw"}

    def test_gaitex_names_the_reader_copy_not_source_parsing(self):
        imports = stages_mod.default_registry().for_source("gaitex").library_imports
        # The registry is kept byte-identical to the parent project's copy, so it still names the
        # modules by the package's name before the split (soma_synthetic_imu).
        assert "soma_synthetic_imu.gaitex_retarget.native" in imports
        assert not any("source_parsing" in name for name in imports)

    def test_every_library_import_the_registry_names_is_a_module_of_this_package(self):
        """Read with the parent package's name mapped to this one, every declared library
        import resolves here: the declaration and the code have not drifted apart."""
        import importlib.util

        registry = stages_mod.default_registry()
        names = sorted({name for source in registry.sources
                        for name in registry.for_source(source).library_imports})
        assert names
        for name in names:
            assert name.startswith("soma_synthetic_imu."), name
            mapped = "soma_synth." + name.removeprefix("soma_synthetic_imu.")
            assert importlib.util.find_spec(mapped) is not None, mapped

    def test_every_source_reads_the_three_path_variables(self):
        registry = stages_mod.default_registry()
        for name in registry.sources:
            assert {"SOMA_DATA_ROOT", "SOMA_SOURCE_ROOT", "SOMA_BODY_MODEL_DIR"} <= set(
                registry.for_source(name).reads_env), name
        assert "ADDBIO_RETARGET_CORPUS" in registry.for_source("addbiomechanics").reads_env

    def test_v2_keeps_what_v1_declared_apart_from_the_declared_changes(self):
        v1, v2 = stages_mod.load(V1), stages_mod.load(V2)
        assert (v1.spec_id, v1.spec_version, v1.canonical_stages, v1.resume_predicates) == (
            v2.spec_id, v2.spec_version, v2.canonical_stages, v2.resume_predicates)
        assert v1.body_model_sets == v2.body_model_sets
        for name in v1.sources:
            old, new = v1.for_source(name), v2.for_source(name)
            for attribute in ("small_mode", "emitter", "resume", "parallel", "body_model_set",
                              "body_model_selection", "smpl18_profile", "stochastic",
                              "path_loads", "configs"):
                assert getattr(old, attribute) == getattr(new, attribute), (name, attribute)
            assert old.entrypoint.script == new.entrypoint.script, name
            assert old.entrypoint.parallel_args == new.entrypoint.parallel_args, name
            if name != "prism":
                assert (old.entrypoint.output_arg, old.entrypoint.output_arg_kind) == (
                    new.entrypoint.output_arg, new.entrypoint.output_arg_kind), name
            expected_inputs = {k: v for k, v in old.inputs.items() if k != "cross_check_corpus"}
            assert new.inputs == expected_inputs, name

    def test_every_registry_flag_is_an_argparse_option_of_its_script(self):
        """The registry describes the scripts; the runner builds command lines, refuses
        selection and controlled flags and resolves argparse abbreviations from it. A flag it
        names that the parser lacks, or a parser option it does not name, makes all of that
        wrong without a run ever failing."""
        registry = stages_mod.default_registry()
        scripts = 0
        for name in registry.sources:
            source = registry.for_source(name)
            entry, corpus = source.entrypoint, source.corpus
            options = _parser_options(REPO / entry.script)
            flags = {flag for flag in options if flag.startswith("-")}
            named = {*entry.required_args, *entry.optional_args, *entry.selection_args,
                     entry.output_arg, *(entry.parallel_args or {}).values()}
            if corpus is not None:
                named.add(corpus.bundle_arg)
            assert named <= flags, (name, sorted(named - flags))
            assert set(entry.required_args) == {f for f in flags if options[f]}, name
            assert flags == {*entry.required_args, *entry.optional_args}, (name, "undeclared")
            scripts += 1
            if corpus is not None:
                options = _parser_options(REPO / corpus.script)
                flags = {flag for flag in options if flag.startswith("-")}
                named = {*corpus.required_args, *corpus.optional_args, *corpus.selection_args,
                         corpus.output_arg}
                assert named <= flags, (name, "corpus", sorted(named - flags))
                assert set(corpus.required_args) == {f for f in flags if options[f]}, name
                assert flags == {*corpus.required_args, *corpus.optional_args}, (name, "corpus")
                scripts += 1
            for step in source.post_steps:
                options = _parser_options(REPO / step.script)
                given = {arg.split("=", 1)[0] for arg in step.args if arg.startswith("-")}
                assert given <= set(options), (name, step.script, sorted(given - set(options)))
                assert {f for f in options if f.startswith("-") and options[f]} <= given
                scripts += 1
        assert scripts == 9            # five bundle entrypoints, three corpora, one post-step

    def test_the_registry_names_no_force_flag_and_no_resume_behaviour(self):
        """Every build is fresh, so nothing tells an entrypoint to rebuild what it would skip,
        and nothing says how a source resumes: the runner resumes none. The operator's own
        --force still passes through."""
        raw = _raw_v2()
        for name, block in raw["sources"].items():
            for key in ("force_arg", "lossy_resume"):
                assert key not in block["entrypoint"], (name, key)
                assert key not in (block.get("corpus") or {}), (name, key)
        for attribute in ("force_arg", "lossy_resume"):
            assert not hasattr(stages_mod.Entrypoint, attribute)
        assert "lossy_resume" not in V2.read_text(encoding="utf-8")
        assert "--resume" in stages_mod.default_registry().for_source(
            "addbiomechanics").corpus.optional_args

    def test_a_v1_file_is_byte_for_byte_what_it_was(self):
        """Used configs are never edited in place."""
        committed = subprocess.run(["git", "-C", str(REPO), "show", "HEAD:" + V1.relative_to(REPO).as_posix()],
                                   capture_output=True, check=False)
        if committed.returncode != 0:
            pytest.skip("git is not available")
        assert committed.stdout.replace(b"\r\n", b"\n") == V1.read_bytes().replace(b"\r\n", b"\n")


def _write_registry(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _raw_v2() -> dict:
    return yaml.safe_load(V2.read_text(encoding="utf-8"))


@pytest.mark.parametrize("key, value, match", [
    ("releases_holds", True, "releases_holds"),
    ("releases_holds", "false", "releases_holds"),
    ("artifact_class", "canonical", "experimental_non_candidate"),
    ("distribution_scope", "public", "internal_only"),
    ("decision_record", "", "missing"),
])
def test_a_policy_that_is_not_the_standing_decision_is_refused_at_load(tmp_path, key, value,
                                                                       match):
    raw = _raw_v2()
    raw["generation_policy"][key] = value
    with pytest.raises(stages_mod.PipelineError, match=match):
        stages_mod.load(_write_registry(tmp_path, raw))


def test_v2_without_a_policy_is_refused_at_load(tmp_path):
    raw = _raw_v2()
    del raw["generation_policy"]
    with pytest.raises(stages_mod.PipelineError, match="generation_policy"):
        stages_mod.load(_write_registry(tmp_path, raw))


def test_a_corpus_the_source_inputs_do_not_name_is_refused_at_load(tmp_path):
    raw = _raw_v2()
    raw["sources"]["hknu"]["corpus"]["lineage"] = "some_other_corpus"
    with pytest.raises(stages_mod.PipelineError, match="inputs"):
        stages_mod.load(_write_registry(tmp_path, raw))


def test_a_corpus_flag_the_bundle_entrypoint_lacks_is_refused_at_load(tmp_path):
    raw = _raw_v2()
    raw["sources"]["gaitex"]["corpus"]["bundle_arg"] = "--no-such-flag"
    with pytest.raises(stages_mod.PipelineError, match="bundle_arg"):
        stages_mod.load(_write_registry(tmp_path, raw))


def test_a_post_step_may_name_only_the_declared_placeholders(tmp_path):
    raw = _raw_v2()
    raw["sources"]["prism"]["post_steps"][0]["args"] = ["{home}"]
    with pytest.raises(stages_mod.PipelineError, match="post_steps"):
        stages_mod.load(_write_registry(tmp_path, raw))


def test_v2_fields_in_a_v1_file_are_refused(tmp_path):
    raw = yaml.safe_load(V1.read_text(encoding="utf-8"))
    raw["sources"]["hknu"]["bundle_lineage"] = "hknu_unified8"
    with pytest.raises(stages_mod.PipelineError, match="source_pipelines_v2"):
        stages_mod.load(_write_registry(tmp_path, raw))
    raw = yaml.safe_load(V1.read_text(encoding="utf-8"))
    raw["sources"]["amass"]["entrypoint"]["selection_args"] = []
    with pytest.raises(stages_mod.PipelineError, match="entrypoint.selection_args"):
        stages_mod.load(_write_registry(tmp_path, raw))


# ======================================================================= fixtures
CORPUS_SCRIPT = textwrap.dedent("""\
    import json, os, pathlib, sys
    args = sys.argv[1:]
    with open(os.environ["FAKE_CALLS"], "a", encoding="utf-8") as log:
        log.write(json.dumps({"script": "corpus", "argv": args,
                              "data_root": os.environ.get("SOMA_DATA_ROOT")}) + "\\n")
    code = int(os.environ.get("FAKE_CORPUS_EXIT", "0"))
    if code:
        sys.exit(code)
    out = pathlib.Path(args[args.index("--out") + 1])
    (out / "S01").mkdir(parents=True, exist_ok=True)
    (out / "S01" / "t1.npz").write_bytes(b"corpus take")
    (out / "SUMMARY.json").write_text(json.dumps({"fixture": True, "argv": args}), encoding="utf-8")
""")

BUNDLE_SCRIPT = textwrap.dedent("""\
    import json, os, pathlib, sys
    args = sys.argv[1:]
    with open(os.environ["FAKE_CALLS"], "a", encoding="utf-8") as log:
        log.write(json.dumps({"script": "bundle", "argv": args,
                              "data_root": os.environ.get("SOMA_DATA_ROOT"),
                              "source_root": os.environ.get("SOMA_SOURCE_ROOT")}) + "\\n")
    early = int(os.environ.get("FAKE_BUNDLE_EARLY_EXIT", "0"))
    if early:
        sys.exit(early)               # refused at start-up, before writing anything
    out = pathlib.Path(args[args.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    # like the real generators: takes first, INDEX.json last, so a failure leaves no INDEX
    (out / "take_a").mkdir(exist_ok=True)
    code = int(os.environ.get("FAKE_BUNDLE_EXIT", "0"))
    if code:
        sys.exit(code)
    corpus = "runs/experimental_generation_poc_demo/hknu_smpl24_paired"
    takes = []
    for name, npz, digest in (("take_a", "S01/t1.npz", "a" * 64), ("take_b", "S01/t2.npz", "b" * 64)):
        (out / name).mkdir(exist_ok=True)
        manifest = {
            "source": {"source_name": "hknu", "relative_path": f"{corpus}/{npz}",
                       "source_asset_sha256": digest},
            "small_synthesis": {"inputs": {"marker_csv": {
                "relative_path": "extracted/fixture/markers.csv", "sha256": "c" * 64}}},
            "paired_measured_counterpart": {"relative_path": corpus},
            "anthro_reconstruction": {"settings_files": [{"given": "x.yaml", "sha256": "d" * 64}]},
        }
        (out / name / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        takes.append({"take_id": name, "rel": name, "status": "ok"})
    index = {"artifact_class": os.environ.get("FAKE_CLASS", "experimental_non_candidate"),
             "distribution_scope": "internal_only", "takes": takes}
    (out / "INDEX.json").write_text(json.dumps(index), encoding="utf-8")
""")

POST_SCRIPT = textwrap.dedent("""\
    import json, os, pathlib, sys
    args = sys.argv[1:]
    with open(os.environ["FAKE_CALLS"], "a", encoding="utf-8") as log:
        log.write(json.dumps({"script": "post", "argv": args}) + "\\n")
    code = int(os.environ.get("FAKE_POST_EXIT", "0"))
    if code:
        sys.exit(code)
    (pathlib.Path(args[0]) / "POST_STEP_RAN").write_text("yes", encoding="utf-8")
""")

#: A bundle entrypoint that skips a take whose manifest.json exists unless it is given --force,
#: as amass, prism and addbiomechanics do (prism records such a take as "skipped"). Each take
#: records the generation that wrote it (FAKE_GENERATION), so a test can tell a take built by
#: this run from one an earlier run left: in a fresh directory there is none to skip.
SKIPPING_BUNDLE_SCRIPT = textwrap.dedent("""\
    import json, os, pathlib, sys
    args = sys.argv[1:]
    with open(os.environ["FAKE_CALLS"], "a", encoding="utf-8") as log:
        log.write(json.dumps({"script": "skipping_bundle", "argv": args}) + "\\n")
    flag = "--out" if "--out" in args else "--out-root"
    out = pathlib.Path(args[args.index(flag) + 1])
    out.mkdir(parents=True, exist_ok=True)
    generation = os.environ.get("FAKE_GENERATION", "1")
    takes = []
    for name in ("take_a", "take_b"):
        manifest = out / name / "manifest.json"
        if manifest.is_file() and "--force" not in args:
            takes.append({"take_id": name, "rel": name, "status": "skipped"})
            continue
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_text(json.dumps({"generation": generation}), encoding="utf-8")
        takes.append({"take_id": name, "rel": name, "status": "ok"})
        if os.environ.get("FAKE_BUNDLE_EXIT"):
            sys.exit(int(os.environ["FAKE_BUNDLE_EXIT"]))    # after one take, before INDEX.json
    index = {"artifact_class": "experimental_non_candidate", "distribution_scope": "internal_only",
             "generation": generation, "takes": takes}
    (out / "INDEX.json").write_text(json.dumps(index), encoding="utf-8")
""")


@pytest.fixture
def fixture_registry(tmp_path, monkeypatch):
    """A copy of registry v2 whose hknu and prism entrypoints are the fixture scripts."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name, text in (("fake_corpus.py", CORPUS_SCRIPT), ("fake_bundle.py", BUNDLE_SCRIPT),
                       ("fake_post.py", POST_SCRIPT),
                       ("fake_skipping_bundle.py", SKIPPING_BUNDLE_SCRIPT)):
        (scripts / name).write_text(text, encoding="utf-8")
    raw = _raw_v2()
    raw["sources"]["hknu"]["entrypoint"]["script"] = str(scripts / "fake_bundle.py")
    raw["sources"]["hknu"]["corpus"]["script"] = str(scripts / "fake_corpus.py")
    raw["sources"]["prism"]["entrypoint"]["script"] = str(scripts / "fake_bundle.py")
    raw["sources"]["prism"]["post_steps"][0]["script"] = str(scripts / "fake_post.py")
    # the three whose real generators skip finished takes: a fake that skips them the same way
    for name in ("amass", "addbiomechanics"):
        raw["sources"][name]["entrypoint"]["script"] = str(scripts / "fake_skipping_bundle.py")
    raw["sources"]["addbiomechanics"]["corpus"]["script"] = str(scripts / "fake_corpus.py")
    calls = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CALLS", str(calls))
    for name in ("FAKE_CORPUS_EXIT", "FAKE_BUNDLE_EXIT", "FAKE_BUNDLE_EARLY_EXIT",
                 "FAKE_POST_EXIT", "FAKE_GENERATION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("FAKE_CLASS", raising=False)
    monkeypatch.delenv(paths.BODY_MODEL_DIR_ENV, raising=False)
    monkeypatch.delenv(paths.SOURCE_ROOT_ENV, raising=False)
    monkeypatch.setenv(paths.SHARED_DRIVE_NAMES_ENV, "Team_Share")
    registry = stages_mod.load(_write_registry(tmp_path, raw))
    return registry, calls


@pytest.fixture
def skipping_registry(fixture_registry, tmp_path):
    """The fixture registry with prism's bundle entrypoint also the take-skipping fake, so all
    three sources whose generators skip finished takes (amass, prism, addbiomechanics) do."""
    _, calls = fixture_registry
    raw = yaml.safe_load((tmp_path / "registry.yaml").read_text(encoding="utf-8"))
    raw["sources"]["prism"]["entrypoint"]["script"] = str(
        tmp_path / "scripts" / "fake_skipping_bundle.py")
    folder = tmp_path / "skipping"
    folder.mkdir()
    return stages_mod.load(_write_registry(folder, raw)), calls


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    return root


def _calls(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _manifest(result) -> dict:
    return json.loads((result.run_directory / "manifest.json").read_text(encoding="utf-8"))


def _resolved(result) -> dict:
    return yaml.safe_load((result.run_directory / "resolved_config.yaml").read_text(encoding="utf-8"))


def _runner(registry, data_root, source="hknu", **kwargs):
    return runner_mod.PipelineRunner(source_name=source, data_root=data_root, registry=registry,
                                     **kwargs)


# ======================================================================= defaults
def test_the_bundle_and_corpus_default_to_their_lineages_under_the_data_root(fixture_registry,
                                                                            data_root):
    registry, _ = fixture_registry
    runner = _runner(registry, data_root)
    assert runner.dataset_dir == data_root / POC / "hknu_unified8"
    assert runner.corpus_dir == data_root / POC / "hknu_smpl24_paired"
    assert runner.runs_root == data_root / POC / "_runs"
    amass = _runner(registry, data_root, source="amass")
    assert amass.dataset_dir == data_root / POC / "amass_faithful_full"
    assert amass.corpus_dir is None


def test_the_data_root_defaults_to_soma_data_root(fixture_registry, data_root, monkeypatch):
    registry, _ = fixture_registry
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(data_root))
    runner = runner_mod.PipelineRunner(source_name="prism", registry=registry)
    assert runner.data_root == data_root
    assert runner.dataset_dir == data_root / POC / "prism_faithful_full"
    monkeypatch.delenv(paths.DATA_ROOT_ENV)
    with pytest.raises(paths.PathConfigError, match="SOMA_DATA_ROOT is not set"):
        runner_mod.PipelineRunner(source_name="prism", registry=registry)


def test_a_corpus_directory_for_a_source_without_a_corpus_is_refused(fixture_registry, data_root):
    registry, _ = fixture_registry
    with pytest.raises(stages_mod.PipelineError, match="no corpus stage"):
        _runner(registry, data_root, source="amass", corpus_dir=data_root / "x")


def test_the_record_names_the_experimental_catalog_and_relative_paths(fixture_registry,
                                                                     data_root):
    registry, _ = fixture_registry
    result = _runner(registry, data_root, source="prism").run(
        stop_after="generate", skip_generate=True, log=_quiet)
    config = _resolved(result)["config"]
    assert config["catalog_dir_relative"] == "experimental/catalog"
    assert config["dataset_dir"] == str(data_root / POC / "prism_faithful_full")   # kept
    assert config["dataset_dir_relative"] == f"{POC}/prism_faithful_full"
    assert config["runs_root_relative"] == f"{POC}/_runs"
    assert config["data_root"] == str(data_root)
    assert config["registry_schema"] == "source_pipelines_v2"
    # --skip-generate takes no decision, so none is intended (and the manifest records none)
    assert config["generation_basis_intended"] is None and "generation_basis" not in config
    assert _manifest(result)["generation_decision"] is None
    assert config["source_root_relative"] == "extracted" and "source_root" not in config
    assert config["source_root_from"] == "data_root"
    assert config["replace_existing"] is False
    # a path inside the data root is recorded by its relative form only: teammates' records are
    # collected for licence tracing, and their absolute paths say nothing another PC can use
    assert "catalog_dir" not in config and "runs_root" not in config
    assert "child_environment" not in config                    # it repeated data_root

    def absolute(node, where=""):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from absolute(value, f"{where}.{key}")
        elif isinstance(node, list):
            for value in node:
                yield from absolute(value, where)
        elif isinstance(node, str) and str(data_root) in node:
            yield where

    assert sorted(absolute(config)) == [".data_root", ".dataset_dir"]


def test_a_path_outside_the_data_root_keeps_its_absolute_form(fixture_registry, data_root,
                                                              tmp_path):
    registry, _ = fixture_registry
    elsewhere = tmp_path / "elsewhere"
    result = _runner(registry, data_root, source="prism", runs_root=elsewhere / "_runs").run(
        stop_after="generate", skip_generate=True, catalog_dir=elsewhere / "catalog", log=_quiet)
    config = _resolved(result)["config"]
    assert config["catalog_dir"] == str(elsewhere / "catalog")
    assert config["catalog_dir_relative"] is None
    assert config["runs_root"] == str(elsewhere / "_runs")
    assert config["runs_root_relative"] is None


def test_registration_goes_to_the_experimental_catalog_by_default(fixture_registry, data_root,
                                                                  monkeypatch):
    from soma_synth.datasets import catalog

    registry, _ = fixture_registry
    seen = []
    monkeypatch.setattr(catalog, "register_dataset",
                        lambda catalog_dir, dataset_dir, **kwargs: seen.append(
                            (Path(catalog_dir), kwargs["data_root"], kwargs["registered_by"])) or [])
    result = _runner(registry, data_root, source="prism").run(
        start_at="register", stop_after="register", log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"]
    # the catalog's journal names the run whose record explains the registration
    assert seen == [(data_root / "experimental" / "catalog", data_root,
                     {"tool": "soma-synth run", "run_id": result.run_directory.name})]
    assert result.run_directory.name.startswith("run_")


# ======================================================================= corpus stage
def test_a_missing_corpus_is_built_and_handed_to_the_bundle(fixture_registry, data_root):
    registry, calls_path = fixture_registry
    runner = _runner(registry, data_root)
    result = runner.run(stop_after="generate", log=_quiet)

    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    calls = _calls(calls_path)
    assert [c["script"] for c in calls] == ["corpus", "bundle"]
    corpus = data_root / POC / "hknu_smpl24_paired"
    assert calls[0]["argv"] == ["--out", str(corpus)]
    assert calls[1]["argv"] == ["--out", str(runner.dataset_dir), "--paired", str(corpus)]
    # the generators see the run's data root, so their defaults and models resolve where the
    # record says
    assert {c["data_root"] for c in calls} == {str(data_root)}

    manifest = _manifest(result)
    record = manifest["corpus"]
    assert record["action"] == "built" and record["built_in_run"] is True
    assert record["present_at_open"] is False
    assert record["fingerprint_file"] == "SUMMARY.json"
    assert record["fingerprint_sha256"] == _sha(corpus / "SUMMARY.json")
    assert record["directory_relative"] == f"{POC}/hknu_smpl24_paired"
    assert manifest["tracing"]["corpus_fingerprint"] == record["fingerprint_sha256"]
    inputs = manifest["run_identity"]["inputs"]
    assert inputs["input_corpus"] == f"{POC}/hknu_smpl24_paired"
    assert inputs["input_corpus_sha256"] is None            # it did not exist at identity time
    assets = json.loads((result.run_directory / "source_assets.json").read_text(encoding="utf-8"))
    asset = assets["retarget_corpus/SUMMARY.json"]
    assert asset["sha256"] == record["fingerprint_sha256"]
    assert asset["relative_path"] == f"{POC}/hknu_smpl24_paired/SUMMARY.json"
    metrics = json.loads((result.run_directory / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stage_counts"]["corpus_built"] == 1


def test_the_run_record_keeps_each_child_steps_output(fixture_registry, data_root, monkeypatch,
                                                     capsys):
    """logs/ stayed empty on a successful run although the README said the logs were there: the
    children wrote to the console only. Each child step is teed into its own log now, and what
    the runner printed goes to runner.log."""
    registry, _ = fixture_registry
    result = _runner(registry, data_root).run(stop_after="generate", log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    logs = result.run_directory / "logs"
    assert sorted(p.name for p in logs.iterdir()) == ["corpus.log", "generate.log", "runner.log"]
    for name, script in (("corpus.log", "fake_corpus.py"), ("generate.log", "fake_bundle.py")):
        text = (logs / name).read_text(encoding="utf-8")
        assert text.startswith("$ ") and script in text.splitlines()[0]
        assert text.rstrip().endswith("[exit 0]")
    runner_log = (logs / "runner.log").read_text(encoding="utf-8")
    assert runner_log.startswith("[generate] ...") and "    ok" in runner_log

    monkeypatch.setenv("FAKE_POST_EXIT", "4")
    failed = _runner(registry, data_root, source="prism",
                     dataset_dir=data_root / "tmp" / "prism_faithful_full").run(
        stop_after="generate", log=_quiet)
    assert [o.status for o in failed.outcomes] == ["failed"]
    logs = failed.run_directory / "logs"
    assert sorted(p.name for p in logs.iterdir()) == [
        "generate.log", "post_step.fake_post.log", "runner.log"]
    assert (logs / "post_step.fake_post.log").read_text(encoding="utf-8").rstrip().endswith(
        "[exit 4]")
    assert "stopping (fail-fast)" in (logs / "runner.log").read_text(encoding="utf-8")
    capsys.readouterr()


def _prebuilt_corpus(data_root: Path) -> Path:
    corpus = data_root / POC / "hknu_smpl24_paired"
    corpus.mkdir(parents=True)
    (corpus / "SUMMARY.json").write_text('{"prebuilt": true}', encoding="utf-8")
    return corpus


def _supersede(lineage_dir: Path, name: str, date: str = "2026-09-25") -> Path:
    """Retention rule 1.2 step (1) for the generation in ``lineage_dir``: its evidence file copied to
    ``_superseded/<lineage>_<date>/`` beside it, with a SUPERSEDED.md."""
    folder = lineage_dir.parent / "_superseded" / f"{lineage_dir.name}_{date}"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lineage_dir / name, folder / name)
    (folder / "SUPERSEDED.md").write_text("# superseded\n", encoding="utf-8")
    return folder


def test_a_present_corpus_is_reused_and_fingerprinted_in_the_identity(fixture_registry,
                                                                     data_root):
    registry, calls_path = fixture_registry
    corpus = _prebuilt_corpus(data_root)
    result = _runner(registry, data_root).run(stop_after="generate", log=_quiet)

    assert [o.status for o in result.outcomes] == ["ok"]
    assert [c["script"] for c in _calls(calls_path)] == ["bundle"]
    manifest = _manifest(result)
    assert manifest["corpus"]["action"] == "reused"
    assert manifest["corpus"]["built_in_run"] is False
    assert manifest["corpus"]["corpus_args_used"] is False
    assert manifest["corpus"]["fingerprint_sha256"] == _sha(corpus / "SUMMARY.json")
    assert manifest["run_identity"]["inputs"]["input_corpus_sha256"] == _sha(corpus / "SUMMARY.json")


def _sides(directory: Path) -> list[str]:
    """The names of the side directories (``.replaced-*``, ``.failed-*``) beside ``directory``."""
    return [path.name for path in runner_mod.side_directories(directory)]


def _evidence(result, what: str) -> dict[str, str]:
    """{file: sha256} of the evidence the run copied out of the directory it moved aside, each
    checked against its copy in the run record."""
    found = {}
    for entry in _manifest(result)["replaced"][what]["evidence"]:
        assert _sha(result.run_directory / entry["copy"]) == entry["sha256"], entry
        found[entry["file"]] = entry["sha256"]
    return found


def test_a_rebuild_builds_the_corpus_into_a_fresh_directory(fixture_registry, data_root):
    """A corpus rewritten in place would keep a trial the new build no longer makes, and the
    bundle would read it. So the old corpus is moved aside whole and the new one is built into an
    empty directory; on a production lineage the aside directory is kept for the data owner's
    approved deletion (retention rule 1.5)."""
    registry, calls_path = fixture_registry
    corpus = _prebuilt_corpus(data_root)
    (corpus / "S99").mkdir()
    (corpus / "S99" / "stale.npz").write_bytes(b"a trial of the old generation")
    old = _sha(corpus / "SUMMARY.json")
    before = _tree(corpus)
    _supersede(corpus, "SUMMARY.json")          # a production corpus lineage: 1.2 step (1) first
    result = _runner(registry, data_root).run(stop_after="generate", rebuild_corpus=True,
                                              replace_existing=True, log=_quiet)

    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    assert [c["script"] for c in _calls(calls_path)] == ["corpus", "bundle"]
    assert not (corpus / "S99").exists()                   # no stale trial in the new corpus
    assert sorted(_tree(corpus)) == ["S01", "S01/t1.npz", "SUMMARY.json"]
    manifest = _manifest(result)
    assert manifest["corpus"]["action"] == "rebuilt"
    assert manifest["corpus"]["present_at_open"] is True and manifest["corpus"]["moved_aside"]
    assert manifest["corpus"]["found_at_open"] == "a finished corpus (SUMMARY.json)"
    assert manifest["corpus"]["fingerprint_sha256"] == _sha(corpus / "SUMMARY.json") != old
    # the identity does not fingerprint the copy the run replaced; the record keeps that copy
    assert manifest["run_identity"]["inputs"]["input_corpus_sha256"] is None
    replaced = manifest["replaced"]["corpus"]
    assert _evidence(result, "corpus") == {"SUMMARY.json": old}
    assert replaced["directory_relative"] == f"{POC}/hknu_smpl24_paired"
    assert replaced["production_directory"] is True
    assert replaced["outcome"] == "kept_production_directory"
    aside = corpus.with_name(replaced["aside_name"])
    assert aside.name.startswith("hknu_smpl24_paired.replaced-") and _sides(corpus) == [aside.name]
    assert _tree(aside) == before                          # moved, not copied or rewritten
    assert str(aside) in result.outcomes[0].detail and "retention rule 1.5" in result.outcomes[0].detail
    assert not (corpus / runner_mod.GENERATING_MARKER).exists()


@pytest.mark.parametrize("flags", [{"rebuild_corpus": True}, {"replace_existing": True}])
def test_a_rebuild_over_a_finished_corpus_needs_both_flags(fixture_registry, data_root, flags):
    """A finished corpus is reused unless --rebuild-corpus asks for a new one, and replacing it
    also needs --replace-existing; --replace-existing alone reuses it (it replaces the bundle)."""
    registry, calls_path = fixture_registry
    corpus = _prebuilt_corpus(data_root)
    result = _runner(registry, data_root).run(stop_after="generate", log=_quiet, **flags)
    if "replace_existing" in flags:
        assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
        assert _manifest(result)["corpus"]["action"] == "reused"
        assert [c["script"] for c in _calls(calls_path)] == ["bundle"]
    else:
        assert [o.status for o in result.outcomes] == ["refused"]
        text = (result.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
        assert "a finished corpus (SUMMARY.json)" in text
        assert "--rebuild-corpus --replace-existing" in text and "retention rule 1.2" in text
        assert _calls(calls_path) == []
        assert not result.ok and result.refused
    assert (corpus / "SUMMARY.json").read_text(encoding="utf-8") == '{"prebuilt": true}'
    assert _sides(corpus) == []


def test_a_failed_rebuild_restores_the_corpus_as_it_was(fixture_registry, data_root, tmp_path,
                                                        monkeypatch):
    """A rebuild that fails deletes nothing: the fresh directory becomes <dir>.failed-<run> and
    the old corpus is renamed back, byte for byte; the next rebuild is refused until the failed
    output is dealt with."""
    registry, calls_path = fixture_registry
    corpus = _prebuilt_corpus(data_root)
    (corpus / "S01").mkdir()
    (corpus / "S01" / "t1.npz").write_bytes(b"the old generation")
    before = _tree(corpus)
    _supersede(corpus, "SUMMARY.json")
    # the corpus entrypoint writes a trial and then fails, before its fingerprint
    failing = tmp_path / "scripts" / "fake_corpus_fails.py"
    failing.write_text(textwrap.dedent("""\
        import json, os, pathlib, sys
        args = sys.argv[1:]
        with open(os.environ["FAKE_CALLS"], "a", encoding="utf-8") as log:
            log.write(json.dumps({"script": "corpus", "argv": args}) + "\\n")
        out = pathlib.Path(args[args.index("--out") + 1])
        (out / "S02").mkdir(parents=True, exist_ok=True)
        (out / "S02" / "t9.npz").write_bytes(b"half a new generation")
        sys.exit(3)
    """), encoding="utf-8")
    raw = yaml.safe_load((tmp_path / "registry.yaml").read_text(encoding="utf-8"))
    raw["sources"]["hknu"]["corpus"]["script"] = str(failing)
    folder = tmp_path / "failing"
    folder.mkdir()
    failing_registry = stages_mod.load(_write_registry(folder, raw))

    failed = _runner(failing_registry, data_root).run(stop_after="generate", rebuild_corpus=True,
                                                      replace_existing=True, log=_quiet)
    assert [o.status for o in failed.outcomes] == ["failed"]
    assert failed.outcomes[0].detail.startswith("corpus entrypoint exited 3")
    assert "restored" in failed.outcomes[0].detail
    assert _tree(corpus) == before                          # the previous state, exactly
    replaced = _manifest(failed)["replaced"]["corpus"]
    assert replaced["outcome"] == "restored"
    failed_output = corpus.with_name(replaced["failed_output_name"])
    assert failed_output.name.startswith("hknu_smpl24_paired.failed-")
    assert _sides(corpus) == [failed_output.name]
    assert (failed_output / "S02" / "t9.npz").is_file()     # kept, not deleted
    assert _evidence(failed, "corpus") == {"SUMMARY.json": _sha(corpus / "SUMMARY.json")}
    assert [c["script"] for c in _calls(calls_path)] == ["corpus"]      # the bundle never ran

    # while the failed output stands beside it, the corpus is not replaced again
    refused = _runner(registry, data_root).run(stop_after="generate", rebuild_corpus=True,
                                               replace_existing=True, log=_quiet)
    assert [o.status for o in refused.outcomes] == ["refused"]
    text = (refused.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert failed_output.name in text and "replaced one generation at a time" in text
    assert _tree(corpus) == before
    shutil.rmtree(failed_output)
    again = _runner(registry, data_root).run(stop_after="generate", rebuild_corpus=True,
                                             replace_existing=True, log=_quiet)
    assert [o.status for o in again.outcomes] == ["ok"], again.outcomes
    assert _manifest(again)["corpus"]["action"] == "rebuilt"


def _interrupted_corpus(corpus: Path) -> dict[str, str]:
    """What a killed corpus build leaves: part of a corpus, even a fingerprint, and the marker."""
    (corpus / "S07").mkdir(parents=True)
    (corpus / "S07" / "partial.npz").write_bytes(b"part of a build that stopped")
    (corpus / "SUMMARY.json").write_text('{"half": true}', encoding="utf-8")
    (corpus / runner_mod.GENERATING_MARKER).write_text(
        json.dumps({"run_directory_relative": "tmp/_runs/run_killed", "started_utc": "then"}),
        encoding="utf-8")
    return _tree(corpus)


@pytest.mark.parametrize("state", ["interrupted", "other files"])
def test_a_corpus_directory_that_is_not_empty_is_rebuilt_only_with_both_flags(
        fixture_registry, data_root, monkeypatch, state):
    """No corpus build is resumed or reused unless it finished: an interrupted build (its
    marker), or files that are no corpus, are refused -- naming what the directory holds --
    until --rebuild-corpus --replace-existing move them aside and build into an empty
    directory. A scratch aside is deleted once the step succeeds."""
    registry, calls_path = fixture_registry
    scratch = data_root / "tmp" / "sample"
    corpus = scratch / "hknu_smpl24_paired"

    def run(**flags):
        return _runner(registry, data_root, dataset_dir=scratch / "hknu_unified8",
                       corpus_dir=corpus).run(stop_after="generate", log=_quiet, **flags)

    # a build whose entrypoint wrote nothing leaves no directory behind
    monkeypatch.setenv("FAKE_CORPUS_EXIT", "3")
    assert [o.status for o in run().outcomes] == ["failed"]
    assert not corpus.exists()
    monkeypatch.delenv("FAKE_CORPUS_EXIT")
    calls_path.unlink()

    if state == "interrupted":
        before = _interrupted_corpus(corpus)
        named = "an unfinished generation (.generating of run record tmp/_runs/run_killed"
    else:
        corpus.mkdir(parents=True)
        (corpus / "notes.txt").write_text("mine", encoding="utf-8")
        before = _tree(corpus)
        named = "1 entries (notes.txt) but no SUMMARY.json and no .generating"
    for flags in ({}, {"replace_existing": True}, {"rebuild_corpus": True}):
        refused = run(**flags)
        assert [o.status for o in refused.outcomes] == ["refused"], flags
        text = (refused.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
        assert named in text and "--rebuild-corpus --replace-existing" in text
        assert _tree(corpus) == before and _sides(corpus) == [] and _calls(calls_path) == []

    result = run(rebuild_corpus=True, replace_existing=True)
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    record = _manifest(result)["corpus"]
    assert record["action"] == "built" and record["moved_aside"] is True
    assert record["found_at_open"].startswith(named.split(" (")[0])
    assert sorted(_tree(corpus)) == ["S01", "S01/t1.npz", "SUMMARY.json"]
    replaced = _manifest(result)["replaced"]["corpus"]
    assert replaced["production_directory"] is False and replaced["outcome"] == "deleted"
    assert _sides(corpus) == []                            # a scratch aside is deleted
    assert [c["script"] for c in _calls(calls_path)] == ["corpus", "bundle"]


def test_the_identity_names_the_corpus_only_when_it_is_reused(fixture_registry, data_root):
    """A corpus the run reuses is fingerprinted into the identity; one it builds is not (it does
    not exist yet). Two runs that reuse the same corpus have the same identity, and the second is
    recorded beside the first rather than over it."""
    registry, _ = fixture_registry
    scratch = data_root / "tmp" / "sample"
    runner = _runner(registry, data_root, dataset_dir=scratch / "hknu_unified8",
                     corpus_dir=scratch / "hknu_smpl24_paired")
    built = runner.run(stop_after="generate", log=_quiet)
    assert _manifest(built)["corpus"]["action"] == "built"
    assert _manifest(built)["run_identity"]["inputs"]["input_corpus_sha256"] is None

    reused = [_runner(registry, data_root, dataset_dir=scratch / "hknu_unified8",
                      corpus_dir=scratch / "hknu_smpl24_paired").run(
        stop_after="generate", replace_existing=True, log=_quiet) for _ in range(2)]
    for result in reused:
        assert _manifest(result)["corpus"]["action"] == "reused"
        assert _manifest(result)["run_identity"]["inputs"]["input_corpus_sha256"] == _sha(
            scratch / "hknu_smpl24_paired" / "SUMMARY.json")
    assert built.run_identity != reused[0].run_identity == reused[1].run_identity
    assert reused[1].run_directory.name == reused[0].run_directory.name + "-2"
    assert (reused[0].run_directory / "manifest.json").exists()


def test_the_resolved_config_is_part_of_the_identity(fixture_registry, data_root):
    """Standard section 3.1 names the resolved config as an identity input. Unfilled, two runs
    that differed only in their output directory or selection had one identity."""
    registry, _ = fixture_registry
    scratch = data_root / "tmp" / "sample"
    first = _runner(registry, data_root, dataset_dir=scratch / "a",
                    corpus_dir=scratch / "c").run(stop_after="generate", skip_generate=True,
                                                  log=_quiet)
    second = _runner(registry, data_root, dataset_dir=scratch / "b",
                     corpus_dir=scratch / "c").run(stop_after="generate", skip_generate=True,
                                                   log=_quiet)
    third = _runner(registry, data_root, dataset_dir=scratch / "a",
                    corpus_dir=scratch / "c").run(stop_after="generate", skip_generate=True,
                                                  bundle_args=["--subjects", "S04"], log=_quiet)
    identities = {first.run_identity, second.run_identity, third.run_identity}
    assert len(identities) == 3
    inputs = _manifest(first)["run_identity"]["inputs"]
    assert len(inputs["resolved_config_sha256"]) == 64


def test_the_identity_does_not_depend_on_where_the_data_root_is(fixture_registry, tmp_path):
    """The data root and absolute paths with a relative form stay out of the identity, so the
    same run on another PC keeps it."""
    registry, _ = fixture_registry
    identities = []
    for name in ("pc_one", "pc_two"):
        root = tmp_path / name
        root.mkdir()
        runner = _runner(registry, root, source="prism")
        config = {"dataset_dir": str(runner.dataset_dir), "data_root": str(root),
                  "dataset_dir_relative": f"{POC}/prism_faithful_full", "steps": ["generate"]}
        identities.append(runner.identity_inputs(resolved_config=config).resolved_config_sha256)
    assert identities[0] == identities[1] is not None


def test_an_explicit_corpus_directory_is_built_and_passed(fixture_registry, data_root):
    registry, calls_path = fixture_registry
    scratch = data_root / "tmp" / "sample"
    runner = _runner(registry, data_root, dataset_dir=scratch / "hknu_unified8",
                     corpus_dir=scratch / "hknu_smpl24_paired")
    result = runner.run(stop_after="generate", log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"]
    calls = _calls(calls_path)
    assert calls[0]["argv"] == ["--out", str(scratch / "hknu_smpl24_paired")]
    assert calls[1]["argv"][-2:] == ["--paired", str(scratch / "hknu_smpl24_paired")]
    assert _manifest(result)["run_identity"]["inputs"]["input_corpus"] == \
        "tmp/sample/hknu_smpl24_paired"


def test_sample_arguments_reach_the_corpus_and_the_bundle(fixture_registry, data_root):
    registry, calls_path = fixture_registry
    scratch = data_root / "tmp" / "sample"
    runner = _runner(registry, data_root, dataset_dir=scratch / "hknu_unified8",
                     corpus_dir=scratch / "hknu_smpl24_paired")
    result = runner.run(stop_after="generate", corpus_args=["--subjects", "S04"],
                        bundle_args=["--subjects", "S04", "--trials", "2"], log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"]
    calls = _calls(calls_path)
    assert calls[0]["argv"] == ["--out", str(scratch / "hknu_smpl24_paired"), "--subjects", "S04"]
    assert calls[1]["argv"] == ["--out", str(scratch / "hknu_unified8"),
                                "--paired", str(scratch / "hknu_smpl24_paired"),
                                "--subjects", "S04", "--trials", "2"]
    config = _resolved(result)["config"]
    assert config["corpus_args"] == ["--subjects", "S04"]
    assert config["bundle_args"] == ["--subjects", "S04", "--trials", "2"]


def test_sample_arguments_aimed_at_a_production_lineage_are_refused(fixture_registry, data_root):
    """A partial run into the lineage directory would rewrite its INDEX over the sample alone."""
    registry, calls_path = fixture_registry
    result = _runner(registry, data_root).run(
        stop_after="generate", bundle_args=["--subjects", "S04"], log=_quiet)
    assert [o.status for o in result.outcomes] == ["refused"]
    assert "production lineage" in result.outcomes[0].detail
    assert _calls(calls_path) == []

    scratch = data_root / "tmp" / "sample" / "hknu_unified8"
    result = _runner(registry, data_root, dataset_dir=scratch).run(
        stop_after="generate", corpus_args=["--subjects", "S04"], log=_quiet)
    assert [o.status for o in result.outcomes] == ["refused"]
    assert "corpus" in result.outcomes[0].detail
    assert _calls(calls_path) == []


def test_a_sample_corpus_is_not_fed_into_the_production_bundle(fixture_registry, data_root):
    """The partial-run check looks at each entrypoint's own arguments. A one-subject corpus in a
    scratch folder handed to the default bundle directory, with no bundle arguments, passed it
    and rebuilt the production INDEX.json from the sample."""
    registry, calls_path = fixture_registry
    sample = data_root / "tmp" / "sample" / "hknu_smpl24_paired"
    production = data_root / POC / "hknu_unified8"
    result = _runner(registry, data_root, corpus_dir=sample).run(
        stop_after="generate", corpus_args=["--subjects", "S04"], log=_quiet)
    assert [o.status for o in result.outcomes] == ["refused"]
    assert "own corpus lineage" in (result.run_directory / "logs" / "refusal.txt").read_text(
        encoding="utf-8")
    assert _calls(calls_path) == []
    assert not (production / "INDEX.json").exists()

    # the same when the sample corpus already exists and no argument is passed at all
    sample.mkdir(parents=True)
    (sample / "SUMMARY.json").write_text('{"sample": true}', encoding="utf-8")
    result = _runner(registry, data_root, corpus_dir=sample).run(stop_after="generate",
                                                                 log=_quiet)
    assert [o.status for o in result.outcomes] == ["refused"]
    assert _calls(calls_path) == []

    # a scratch bundle may be built from it
    result = _runner(registry, data_root, corpus_dir=sample,
                     dataset_dir=sample.parent / "hknu_unified8").run(stop_after="generate",
                                                                      log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"]


def test_a_finished_bundle_is_not_overwritten_without_replace_existing(fixture_registry,
                                                                       data_root):
    """`soma-synth run --source prism` with no other argument used to regenerate the lineage
    in place. Retention rule 1.2 replaces a lineage by evidence, approved deletion, regeneration."""
    registry, calls_path = fixture_registry
    bundle = data_root / POC / "prism_faithful_full"
    bundle.mkdir(parents=True)
    (bundle / "INDEX.json").write_text('{"old": true}', encoding="utf-8")
    old = _sha(bundle / "INDEX.json")

    result = _runner(registry, data_root, source="prism").run(stop_after="generate", log=_quiet)
    assert [o.status for o in result.outcomes] == ["refused"]
    text = (result.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert "already holds a finished bundle (INDEX.json)" in text and "retention rule 1.2" in text
    assert "--replace-existing" in text
    assert _calls(calls_path) == []
    assert _sha(bundle / "INDEX.json") == old
    assert _manifest(result)["generation_decision"] is None      # refused before the decision

    _supersede(bundle, "INDEX.json")              # a production lineage: 1.2 step (1) first
    result = _runner(registry, data_root, source="prism").run(
        stop_after="generate", replace_existing=True, log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    assert _resolved(result)["config"]["replace_existing"] is True
    assert _evidence(result, "bundle") == {"INDEX.json": old}
    kept = _manifest(result)["replaced"]["bundle"]
    assert kept["directory_relative"] == f"{POC}/prism_faithful_full"
    assert kept["production_directory"] is True
    assert kept["outcome"] == "kept_production_directory"
    aside = bundle.with_name(kept["aside_name"])
    assert kept["aside_relative"] == f"{POC}/{aside.name}"
    assert _sha(aside / "INDEX.json") == old and _sha(bundle / "INDEX.json") != old
    # the outcome names the kept aside directory: deleting it needs the data owner's approval
    # (retention rule 1.5)
    assert str(aside) in result.outcomes[0].detail and "retention rule 1.5" in result.outcomes[0].detail
    record = _manifest(result)["bundle"]
    assert record["index_present"] is True and record["moved_aside"] is True
    assert record["found_at_open"] == "a finished bundle (INDEX.json)"
    assert record["generating_marker_removed"] is True


def _fill(bundle: Path, state: str) -> str:
    """A bundle directory holding ``state``; returns how the refusal names it."""
    if state == "finished":
        bundle.mkdir(parents=True)
        (bundle / "INDEX.json").write_text("{}", encoding="utf-8")
        return "already holds a finished bundle (INDEX.json)"
    if state == "unfinished":
        (bundle / "take_a").mkdir(parents=True)
        (bundle / "INDEX.json").write_text("{}", encoding="utf-8")
        (bundle / runner_mod.GENERATING_MARKER).write_text(json.dumps(
            {"run_directory_relative": "tmp/_runs/run_x", "started_utc": "then"}),
            encoding="utf-8")
        return ("already holds an unfinished generation (.generating of run record "
                "tmp/_runs/run_x, started then)")
    if state == "takes without an index":
        (bundle / "take_a").mkdir(parents=True)
        return "already holds 1 entries (take_a) but no INDEX.json and no .generating"
    bundle.mkdir(parents=True)                                  # anything at all
    for name in ("a.txt", "b.txt", "c.txt", "d.txt"):
        (bundle / name).write_text(name, encoding="utf-8")
    return "already holds 4 entries (a.txt, b.txt, c.txt, ...) but no INDEX.json"


@pytest.mark.parametrize("state", ["finished", "unfinished", "takes without an index",
                                   "other files"])
@pytest.mark.parametrize("where", ["scratch", "production"])
def test_a_bundle_directory_that_is_not_empty_is_refused_without_replace_existing(
        fixture_registry, data_root, state, where):
    """No run resumes a bundle, and none writes into a directory holding anything: a finished
    bundle, one whose generation stopped (its marker), takes without an index, other files.
    Each is refused before anything runs, naming what the directory holds."""
    registry, calls_path = fixture_registry
    bundle = (data_root / POC / "prism_faithful_full" if where == "production"
              else data_root / "tmp" / "sample" / "prism_faithful_full")
    named = _fill(bundle, state)
    before = _tree(bundle)
    result = _runner(registry, data_root, source="prism", dataset_dir=bundle).run(
        stop_after="generate", log=_quiet)
    assert [o.status for o in result.outcomes] == ["refused"]
    text = (result.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert f"refusing to generate into {bundle}: it {named}" in text
    assert "--replace-existing" in text and "no run resumes one" in text
    assert _tree(bundle) == before and _sides(bundle) == [] and _calls(calls_path) == []
    assert _manifest(result)["generation_decision"] is None
    assert not runner_mod.lock_path(bundle).exists()


@pytest.mark.parametrize("state", ["unfinished", "takes without an index", "other files"])
@pytest.mark.parametrize("where", ["scratch", "production"])
def test_replace_existing_builds_fresh_and_settles_the_aside(skipping_registry, data_root,
                                                             monkeypatch, state, where):
    """--replace-existing over anything that is no finished generation needs no _superseded
    evidence (there is no finished INDEX.json to supersede): the directory is moved aside and
    the bundle built into an empty one, so the skipping entrypoint skips nothing. A scratch
    aside is deleted; a production directory's is kept and named."""
    registry, calls_path = skipping_registry
    bundle = (data_root / POC / "prism_faithful_full" if where == "production"
              else data_root / "tmp" / "sample" / "prism_faithful_full")
    _fill(bundle, state)
    (bundle / "take_a").mkdir(exist_ok=True)
    (bundle / "take_a" / "manifest.json").write_text('{"generation": "old"}', encoding="utf-8")
    before = _tree(bundle)
    monkeypatch.setenv("FAKE_GENERATION", "2")
    result = _runner(registry, data_root, source="prism", dataset_dir=bundle).run(
        stop_after="generate", replace_existing=True, log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    assert "--force" not in _bundle_argv(calls_path)
    assert _statuses(bundle) == {"take_a": "ok", "take_b": "ok"}      # none "skipped"
    assert _generation(bundle, "take_a") == _generation(bundle, "take_b") == "2"
    assert not (bundle / runner_mod.GENERATING_MARKER).exists()
    assert not (data_root / POC / "_superseded").exists()
    replaced = _manifest(result)["replaced"]["bundle"]
    if where == "scratch":
        assert replaced["outcome"] == "deleted" and _sides(bundle) == []
    else:
        assert replaced["outcome"] == "kept_production_directory"
        (aside,) = runner_mod.side_directories(bundle)
        assert _tree(aside) == before and str(aside) in result.outcomes[0].detail
    assert not runner_mod.lock_path(bundle).exists()


def _old_bundle(bundle: Path) -> dict[str, str]:
    """A finished bundle of an older generation, with what used to survive an in-place
    replacement: a take the new generation does not make, a shard index, a validation report."""
    (bundle / "take_a").mkdir(parents=True)
    (bundle / "take_a" / "manifest.json").write_text('{"generation": "old"}', encoding="utf-8")
    (bundle / "take_stale").mkdir()
    (bundle / "take_stale" / "manifest.json").write_text('{"generation": "old"}', encoding="utf-8")
    (bundle / "INDEX_shard_0.json").write_text('{"shard": "old"}', encoding="utf-8")
    (bundle / "VALIDATION_REPORT.json").write_text('{"ok": true, "old": true}', encoding="utf-8")
    (bundle / "INDEX.json").write_text(json.dumps({
        "artifact_class": "experimental_non_candidate", "distribution_scope": "internal_only",
        "takes": [{"take_id": "take_stale", "rel": "take_stale", "status": "ok"}]}),
        encoding="utf-8")
    return _tree(bundle)


def test_a_failed_replace_restores_even_when_copying_the_evidence_fails(fixture_registry,
                                                                        data_root, monkeypatch):
    """The aside directory is registered the moment os.replace returns: an exception copying
    its evidence files into the run record, before anything was built, renames it back."""
    registry, calls_path = fixture_registry
    bundle = data_root / POC / "prism_faithful_full"
    before = _old_bundle(bundle)
    _supersede(bundle, "INDEX.json")
    real = runner_mod.shutil.copy2

    def copy_fails_from_the_aside(source, destination, *args, **kwargs):
        if ".replaced-" in Path(source).parent.name:
            raise OSError("the disk went away")
        return real(source, destination, *args, **kwargs)

    monkeypatch.setattr(runner_mod.shutil, "copy2", copy_fails_from_the_aside)
    failed = _runner(registry, data_root, source="prism").run(
        stop_after="generate", replace_existing=True, log=_quiet)
    assert [(o.step, o.status) for o in failed.outcomes] == [("run", "failed")]
    assert "the disk went away" in failed.outcomes[0].detail
    assert _tree(bundle) == before and _sides(bundle) == []
    replaced = _manifest(failed)["replaced"]["bundle"]
    assert replaced["outcome"] == "restored" and "failed_output_name" not in replaced
    assert "evidence" not in replaced
    assert _calls(calls_path) == []                         # nothing was built
    assert not runner_mod.lock_path(bundle).exists()


@pytest.mark.parametrize("where", ["exit", "exception"])
def test_a_failed_replace_restores_the_old_bundle_byte_for_byte(fixture_registry, data_root,
                                                                monkeypatch, where):
    """A replacement that fails deletes nothing: the fresh output becomes <dir>.failed-<run>
    and the old bundle is renamed back exactly as it was -- also when the step dies on an
    exception rather than an exit code. A new replace is refused while the failed output
    stands beside it."""
    registry, calls_path = fixture_registry
    bundle = data_root / POC / "prism_faithful_full"
    before = _old_bundle(bundle)
    _supersede(bundle, "INDEX.json")

    if where == "exit":
        monkeypatch.setenv("FAKE_BUNDLE_EXIT", "3")      # after a take, before INDEX.json
    else:
        real = runner_mod.generate_mod.run_generate

        def dies(*args, **kwargs):
            real(*args, **kwargs)
            raise OSError("the disk went away")

        monkeypatch.setattr(runner_mod.generate_mod, "run_generate", dies)
    failed = _runner(registry, data_root, source="prism").run(
        stop_after="generate", replace_existing=True, log=_quiet)
    assert [o.status for o in failed.outcomes] == ["failed"]
    assert failed.outcomes[0].step == ("generate" if where == "exit" else "run")
    assert _tree(bundle) == before                          # the previous state, exactly
    assert not runner_mod.lock_path(bundle).exists()
    replaced = _manifest(failed)["replaced"]["bundle"]
    assert replaced["outcome"] == "restored"
    failed_output = bundle.with_name(replaced["failed_output_name"])
    assert failed_output.name.startswith("prism_faithful_full.failed-")
    assert (failed_output / "take_a").is_dir()              # this run's output, kept
    assert not (failed_output / "take_stale").exists()      # it was built fresh
    assert _sides(bundle) == [failed_output.name]
    assert set(_evidence(failed, "bundle")) == {"INDEX.json", "INDEX_shard_0.json",
                                                "VALIDATION_REPORT.json"}
    assert "post" not in [c["script"] for c in _calls(calls_path)]

    monkeypatch.delenv("FAKE_BUNDLE_EXIT", raising=False)
    refused = _runner(registry, data_root, source="prism").run(
        stop_after="generate", replace_existing=True, log=_quiet)
    assert [o.status for o in refused.outcomes] == ["refused"]
    text = (refused.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert failed_output.name in text and "whose replacement failed" in text
    assert _tree(bundle) == before


def test_replacing_a_production_bundle_needs_its_evidence_in_superseded(fixture_registry,
                                                                         data_root):
    """Retention rule 1.2 replaces a lineage by evidence, approved deletion, regeneration.
    --replace-existing kept only the INDEX.json in the run record, and the generator and the
    runner rewrote the rest of the evidence 1.3(b) lists (descriptions, limitations, report,
    ledger, README) in place: one flag overwrote a production lineage."""
    registry, calls_path = fixture_registry
    bundle = data_root / POC / "prism_faithful_full"
    bundle.mkdir(parents=True)
    (bundle / "INDEX.json").write_text('{"generation": 2}', encoding="utf-8")

    def replace():
        return _runner(registry, data_root, source="prism").run(
            stop_after="generate", replace_existing=True, log=_quiet)

    result = replace()
    assert [o.status for o in result.outcomes] == ["refused"]
    text = (result.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert "production directory" in text and "_superseded/prism_faithful_full_*" in text
    assert "SUPERSEDED.md" in text and "retention rule 1.2" in text
    assert _calls(calls_path) == []
    assert (bundle / "INDEX.json").read_text(encoding="utf-8") == '{"generation": 2}'
    assert _sides(bundle) == []
    assert _manifest(result)["replaced"] is None
    assert _manifest(result)["generation_decision"] is None

    # the evidence of an older generation is not the evidence of this one
    older = data_root / POC / "_superseded" / "prism_faithful_full_2026-09-15"
    older.mkdir(parents=True)
    (older / "INDEX.json").write_text('{"generation": 1}', encoding="utf-8")
    (older / "SUPERSEDED.md").write_text("# superseded\n", encoding="utf-8")
    assert [o.status for o in replace().outcomes] == ["refused"]
    # nor is a copy without its SUPERSEDED.md
    folder = _supersede(bundle, "INDEX.json")
    (folder / "SUPERSEDED.md").unlink()
    assert [o.status for o in replace().outcomes] == ["refused"]
    assert _calls(calls_path) == []

    (folder / "SUPERSEDED.md").write_text("# superseded\n", encoding="utf-8")
    result = replace()
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    assert _evidence(result, "bundle")["INDEX.json"] == _sha(folder / "INDEX.json")

    # the aside directory is kept for the data owner's approved deletion (retention rule 1.5);
    # until it is gone, the lineage is not replaced again
    (aside,) = runner_mod.side_directories(bundle)
    _supersede(bundle, "INDEX.json", date="2026-09-26")
    again = replace()
    assert [o.status for o in again.outcomes] == ["refused"]
    text = (again.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert aside.name in text and "retention rule 1.5" in text
    assert runner_mod.side_directories(bundle) == [aside]


def test_rebuilding_a_production_corpus_needs_its_evidence_in_superseded(fixture_registry,
                                                                         data_root):
    """A corpus fingerprint carries no generated_utc, so the evidence is found by content."""
    registry, calls_path = fixture_registry
    corpus = _prebuilt_corpus(data_root)
    result = _runner(registry, data_root).run(stop_after="generate", rebuild_corpus=True,
                                              replace_existing=True, log=_quiet)
    assert [o.status for o in result.outcomes] == ["refused"]
    text = (result.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert "refusing to replace the corpus" in text
    assert "_superseded/hknu_smpl24_paired_*" in text
    assert _calls(calls_path) == []
    assert (corpus / "SUMMARY.json").read_text(encoding="utf-8") == '{"prebuilt": true}'
    assert _sides(corpus) == []

    _supersede(corpus, "SUMMARY.json", date="2026-09-07")
    result = _runner(registry, data_root).run(stop_after="generate", rebuild_corpus=True,
                                              replace_existing=True, log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    assert _manifest(result)["corpus"]["action"] == "rebuilt"


def test_a_scratch_bundle_is_replaced_without_superseded_evidence(fixture_registry, data_root):
    """A scratch directory needs no _superseded evidence, and its aside directory is deleted
    once the step succeeds; the run record keeps the evidence files."""
    registry, _ = fixture_registry
    scratch = data_root / "tmp" / "sample" / "prism_faithful_full"
    scratch.mkdir(parents=True)
    (scratch / "INDEX.json").write_text('{"old": true}', encoding="utf-8")
    old = _sha(scratch / "INDEX.json")
    result = _runner(registry, data_root, source="prism", dataset_dir=scratch).run(
        stop_after="generate", replace_existing=True, log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    assert _evidence(result, "bundle") == {"INDEX.json": old}
    replaced = _manifest(result)["replaced"]["bundle"]
    assert replaced["production_directory"] is False and replaced["outcome"] == "deleted"
    assert replaced["found"] == "a finished bundle (INDEX.json)"
    assert _sides(scratch) == [] and not (data_root / POC).exists()
    assert "kept" not in result.outcomes[0].detail
    assert not runner_mod.lock_path(scratch).exists()


def test_a_production_directory_under_any_root_keeps_its_aside_and_needs_its_evidence(
        fixture_registry, data_root, tmp_path):
    """Production means any directory directly under a runs/experimental_generation_poc_demo,
    under any root -- not only the one this run's data root holds. Replacing a finished
    generation there needs its _superseded evidence beside it, and the aside is never deleted."""
    registry, calls_path = fixture_registry
    other = tmp_path / "other_root" / POC / "prism_faithful_full"
    other.mkdir(parents=True)
    (other / "INDEX.json").write_text('{"elsewhere": true}', encoding="utf-8")
    before = _tree(other)
    assert runner_mod.is_production_directory(other)
    assert not runner_mod.is_production_directory(data_root / "tmp" / "prism_faithful_full")
    assert not runner_mod.is_production_directory(other.parent)          # the container

    def replace():
        return _runner(registry, data_root, source="prism", dataset_dir=other).run(
            stop_after="generate", replace_existing=True, log=_quiet)

    refused = replace()
    assert [o.status for o in refused.outcomes] == ["refused"]
    assert "_superseded/prism_faithful_full_*" in (
        refused.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert _tree(other) == before and _calls(calls_path) == []

    _supersede(other, "INDEX.json")
    result = replace()
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    replaced = _manifest(result)["replaced"]["bundle"]
    assert replaced["production_directory"] is True
    assert replaced["outcome"] == "kept_production_directory"
    (aside,) = runner_mod.side_directories(other)
    assert _tree(aside) == before and str(aside) in result.outcomes[0].detail


@pytest.mark.parametrize("where", ["dataset_dir", "corpus_dir"])
def test_an_undeclared_production_directory_is_refused_as_a_target(fixture_registry, data_root,
                                                                   tmp_path, where):
    """A directory under the lineage container that is not this source's own bundle or corpus
    lineage -- declared by the registry or not -- is someone's payload: refused before the run
    record opens."""
    registry, calls_path = fixture_registry
    target = tmp_path / "other_root" / POC / "_dryrun_gaitex_marker_small"
    target.mkdir(parents=True)
    (target / "INDEX.json").write_text('{"theirs": true}', encoding="utf-8")
    kwargs = {"dataset_dir": data_root / "tmp" / "sample" / "bundle",
              "corpus_dir": data_root / "tmp" / "sample" / "corpus", where: target}
    before = _tree(tmp_path)
    label = "the bundle directory" if where == "dataset_dir" else "the corpus directory"
    with pytest.raises(runner_mod.RunnerRefusal, match=label) as refused:
        _runner(registry, data_root, **kwargs).run(stop_after="generate",
                                                   replace_existing=True, log=_quiet)
    assert "production directory" in str(refused.value)
    assert "not this source's" in str(refused.value)
    assert _tree(tmp_path) == before and _calls(calls_path) == []


def test_a_failing_corpus_stage_leaves_the_finished_bundle_as_it_was(fixture_registry,
                                                                    data_root, monkeypatch):
    """--replace-existing once moved the bundle's INDEX.json aside before the corpus stage ran; a
    corpus that failed, or was of another class, left a finished bundle reading as unfinished
    although nothing in it had been touched. The bundle directory is moved aside only just
    before the bundle entrypoint."""
    registry, calls_path = fixture_registry
    scratch = data_root / "tmp" / "sample"
    bundle = scratch / "hknu_unified8"
    bundle.mkdir(parents=True)
    (bundle / "INDEX.json").write_text('{"finished": true}', encoding="utf-8")
    corpus = scratch / "hknu_smpl24_paired"

    def run():
        return _runner(registry, data_root, dataset_dir=bundle, corpus_dir=corpus).run(
            stop_after="generate", replace_existing=True, log=_quiet)

    monkeypatch.setenv("FAKE_CORPUS_EXIT", "4")
    result = run()
    assert [o.detail for o in result.outcomes] == ["corpus entrypoint exited 4"]
    assert [c["script"] for c in _calls(calls_path)] == ["corpus"]

    monkeypatch.delenv("FAKE_CORPUS_EXIT")
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "SUMMARY.json").write_text(
        json.dumps({"artifact_class": "canonical", "distribution_scope": "internal_only"}),
        encoding="utf-8")
    classed = run()
    assert classed.outcomes[0].status == "failed" and "artifact_class" in classed.outcomes[0].detail

    for outcome in (result, classed):
        assert (bundle / "INDEX.json").read_text(encoding="utf-8") == '{"finished": true}'
        assert _sides(bundle) == []
        manifest = _manifest(outcome)
        assert manifest["replaced"] is None
        assert manifest["bundle"]["index_present"] is True
        assert manifest["bundle"]["moved_aside"] is True              # planned, never reached
    assert [c["script"] for c in _calls(calls_path)] == ["corpus"]       # no bundle ever ran


def test_corpus_arguments_for_a_source_without_a_corpus_stage_are_refused(fixture_registry,
                                                                         data_root):
    """PRISM has no corpus stage. `--corpus-arg=--only --corpus-arg=prism_subj005` would reach
    no entrypoint and no selection check, and the run would cover the whole source in the
    production lineage."""
    registry, calls_path = fixture_registry
    result = _runner(registry, data_root, source="prism").run(
        stop_after="generate", corpus_args=["--only", "prism_subj005"], log=_quiet)
    assert [o.status for o in result.outcomes] == ["refused"]
    text = (result.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert "has no corpus stage" in text and "--bundle-arg" in text
    assert _calls(calls_path) == []
    assert not (data_root / POC / "prism_faithful_full" / "INDEX.json").exists()


@pytest.mark.parametrize("args", [{"bundle_args": ["--only", "prism_subj005"]},
                                  {"corpus_args": ["--compress"]}])
def test_passthrough_arguments_with_an_explicit_command_are_refused(fixture_registry, data_root,
                                                                   args):
    import sys

    registry, calls_path = fixture_registry
    runner = _runner(registry, data_root, source="prism",
                     dataset_dir=data_root / "tmp" / "sample" / "prism")
    command = [sys.executable, str(Path(registry.for_source("prism").entrypoint.script)),
               "--out", str(runner.dataset_dir)]
    result = runner.run(stop_after="generate", generate_cmd=command, log=_quiet, **args)
    assert [o.status for o in result.outcomes] == ["refused"]
    text = (result.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert "explicit generate command" in text
    assert _calls(calls_path) == []


# ======================================================================= protected folders
def _tree(folder: Path) -> dict[str, str]:
    """Every file and folder under ``folder`` with its content hash (``dir`` for a folder)."""
    return {path.relative_to(folder).as_posix(): ("dir" if path.is_dir() else _sha(path))
            for path in sorted(folder.rglob("*"))}


def _evidence_bundle(data_root: Path) -> Path:
    evidence = data_root / POC / "_superseded" / "prism_faithful_full_v1"
    evidence.mkdir(parents=True)
    (evidence / "INDEX.json").write_text('{"evidence": true}', encoding="utf-8")
    (evidence / "VALIDATION_REPORT.json").write_text('{"kept": true}', encoding="utf-8")
    return evidence


@pytest.mark.parametrize("run_kwargs", [
    {"stop_after": "generate", "replace_existing": True},   # moved INDEX.json aside, then refused
    {"start_at": "validate", "stop_after": "readme"},       # rewrote the preserved report
    {"stop_after": "register", "skip_generate": True},
])
def test_an_evidence_folder_is_refused_as_the_bundle_before_the_record_opens(
        fixture_registry, data_root, run_kwargs):
    """`run --source prism --replace-existing <root>/.../_superseded/prism_faithful_full_v1`
    renamed the evidence INDEX.json to INDEX.json.replacing before the generator's own guard
    refused, and opened its run record at _superseded/_runs; `--start-at validate` overwrote the
    preserved VALIDATION_REPORT.json and wrote a ledger beside it."""
    registry, calls_path = fixture_registry
    evidence = _evidence_bundle(data_root)
    before = _tree(data_root)
    runner = _runner(registry, data_root, source="prism", dataset_dir=evidence)
    with pytest.raises(runner_mod.RunnerRefusal, match="the bundle directory") as refused:
        runner.run(log=_quiet, **run_kwargs)
    assert "_superseded" in str(refused.value)
    assert _tree(data_root) == before              # nothing renamed or written, no record
    assert _calls(calls_path) == []


@pytest.mark.parametrize("folder", ["extracted/prism", "raw_archives/prism", "body_models/smpl",
                                    f"{POC}/_manifest_backfill/prism", f"{POC}/_runs/prism"])
def test_a_read_only_or_evidence_folder_is_refused_as_the_bundle(fixture_registry, data_root,
                                                                 folder):
    """`--skip-generate` on extracted/<source> wrote the ledger, the report and the README into
    extracted/, and an extracted/_runs folder beside it."""
    registry, calls_path = fixture_registry
    target = data_root / folder
    target.mkdir(parents=True)
    (target / "native.bin").write_bytes(b"source")
    before = _tree(data_root)
    with pytest.raises(runner_mod.RunnerRefusal, match="the bundle directory.*read-only"):
        _runner(registry, data_root, source="prism", dataset_dir=target).run(
            stop_after="readme", skip_generate=True, log=_quiet)
    assert _tree(data_root) == before
    assert _calls(calls_path) == []


def test_an_evidence_folder_of_another_data_root_is_refused_as_the_bundle(fixture_registry,
                                                                         data_root, tmp_path):
    """The output guard knows the evidence folders of the run's data root and SOMA_DATA_ROOT;
    `soma-synth readme` on another root's _superseded overwrote the README.md kept there, and a
    run pointed at one wrote into it the same way."""
    registry, calls_path = fixture_registry
    other = tmp_path / "other_root"
    evidence = _evidence_bundle(other)
    before = _tree(other)
    with pytest.raises(runner_mod.RunnerRefusal,
                       match="the bundle directory.*evidence folder .*_superseded"):
        _runner(registry, data_root, source="prism", dataset_dir=evidence).run(
            start_at="validate", stop_after="readme", log=_quiet)
    assert _tree(other) == before
    assert _calls(calls_path) == []


@pytest.mark.parametrize("container", ["", "runs", POC])
def test_a_container_is_refused_as_the_bundle(fixture_registry, data_root, container):
    registry, _ = fixture_registry
    with pytest.raises(runner_mod.RunnerRefusal, match="itself"):
        _runner(registry, data_root, source="prism",
                dataset_dir=data_root / container if container else data_root).run(
            stop_after="readme", skip_generate=True, log=_quiet)


def test_a_corpus_in_an_evidence_folder_is_refused_when_the_run_would_write_it(fixture_registry,
                                                                              data_root):
    """`--corpus <_superseded/..._v1> --rebuild-corpus --replace-existing` renamed its
    SUMMARY.json to SUMMARY.json.rebuilding before the corpus generator's guard refused."""
    registry, calls_path = fixture_registry
    evidence = data_root / POC / "_superseded" / "hknu_smpl24_paired_v1"
    evidence.mkdir(parents=True)
    (evidence / "SUMMARY.json").write_text('{"evidence": true}', encoding="utf-8")
    before = _tree(data_root)
    runner = _runner(registry, data_root, dataset_dir=data_root / "tmp" / "sample" / "bundle",
                     corpus_dir=evidence)
    with pytest.raises(runner_mod.RunnerRefusal, match="the corpus directory"):
        runner.run(stop_after="generate", rebuild_corpus=True, replace_existing=True,
                   log=_quiet)
    assert _tree(data_root) == before
    assert _calls(calls_path) == []
    # a corpus the run only reads is not written, and not refused; no run may write it, so no
    # lock is written beside it either
    result = runner.run(stop_after="generate", log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    assert _manifest(result)["corpus"]["action"] == "reused"
    assert (evidence / "SUMMARY.json").read_text(encoding="utf-8") == '{"evidence": true}'
    assert sorted(path.name for path in evidence.parent.iterdir()) == [evidence.name]


def test_the_bundle_is_kept_off_the_corpus_it_reads(fixture_registry, data_root):
    registry, _ = fixture_registry
    corpus = data_root / "tmp" / "sample" / "corpus"
    with pytest.raises(runner_mod.RunnerRefusal, match="the bundle directory"):
        _runner(registry, data_root, dataset_dir=corpus / "bundle", corpus_dir=corpus).run(
            stop_after="generate", log=_quiet)
    assert not corpus.exists()


@pytest.mark.parametrize("where", ["runs_root", "catalog"])
@pytest.mark.parametrize("folder", ["extracted/prism", "raw_archives", "body_models/smpl",
                                    f"{POC}/_superseded/prism_faithful_full_v1",
                                    f"{POC}/_manifest_backfill"])
def test_run_records_and_the_catalog_stay_out_of_read_only_and_evidence_folders(
        fixture_registry, data_root, where, folder):
    registry, calls_path = fixture_registry
    target = data_root / folder / where
    kwargs = {"dataset_dir": data_root / "tmp" / "bundle"}
    catalog = data_root / "tmp" / "catalog"
    if where == "runs_root":
        kwargs["runs_root"] = target
    else:
        catalog = target
    label = "the run records folder" if where == "runs_root" else "the catalog"
    with pytest.raises(runner_mod.RunnerRefusal, match=label):
        _runner(registry, data_root, source="prism", **kwargs).run(
            stop_after="register", skip_generate=True, catalog_dir=catalog, log=_quiet)
    assert not (data_root / folder).exists()
    assert not (data_root / "tmp").exists()
    assert _calls(calls_path) == []


@pytest.mark.parametrize("where", ["runs_root", "catalog"])
@pytest.mark.parametrize("folder", ["", "runs", POC, f"{POC}/prism_faithful_full/_runs",
                                    f"{POC}/hknu_smpl24_paired/catalog"])
def test_run_records_and_the_catalog_stay_out_of_containers_and_lineage_directories(
        fixture_registry, data_root, where, folder):
    """A container's top level would fill with run_<identity> folders or catalog files, and a
    lineage directory holds a bundle or a corpus; records go in <container>/_runs and the catalog
    in <data root>/experimental/catalog."""
    registry, calls_path = fixture_registry
    target = data_root / folder if folder else data_root
    kwargs = {"dataset_dir": data_root / "tmp" / "bundle"}
    catalog = data_root / "tmp" / "catalog"
    if where == "runs_root":
        kwargs["runs_root"] = target
    else:
        catalog = target
    label = "the run records folder" if where == "runs_root" else "the catalog"
    with pytest.raises(runner_mod.RunnerRefusal, match=label):
        _runner(registry, data_root, source="prism", **kwargs).run(
            stop_after="register", skip_generate=True, catalog_dir=catalog, log=_quiet)
    assert sorted(p.name for p in data_root.iterdir()) == []
    assert _calls(calls_path) == []


def test_soma_data_root_and_the_source_root_are_guarded_beside_the_runs_own_root(
        fixture_registry, data_root, tmp_path, monkeypatch):
    """A run given --data-root hands that root to its generators, whose guard then looks under
    it alone; the configured SOMA_DATA_ROOT and SOMA_SOURCE_ROOT are guarded by the runner. A
    source root that does not exist is guarded, not an error: the runner records it as given."""
    registry, calls_path = fixture_registry
    plane = tmp_path / "plane"
    (plane / "extracted" / "prism").mkdir(parents=True)
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(plane))
    monkeypatch.setenv(paths.SOURCE_ROOT_ENV, str(tmp_path / "sources"))
    for target, label in ((plane / "extracted" / "prism", "SOMA_DATA_ROOT/extracted"),
                          (plane / POC / "_superseded" / "x", "_superseded"),
                          (tmp_path / "sources" / "prism" / "bundle",
                           "source root's prism folder"),
                          (tmp_path / "sources", "source root itself")):
        with pytest.raises(runner_mod.RunnerRefusal, match="the bundle directory") as refused:
            _runner(registry, data_root, source="prism", dataset_dir=target).run(
                stop_after="readme", skip_generate=True, log=_quiet)
        assert label in str(refused.value)
    assert list((plane / "extracted" / "prism").iterdir()) == []
    assert not (tmp_path / "sources").exists()
    # the run's own root is still where its bundles go
    result = _runner(registry, data_root, source="prism").run(stop_after="generate", log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    assert [c["script"] for c in _calls(calls_path)] == ["bundle", "post"]   # only this run


@pytest.mark.parametrize("argv", [
    ["--replace-existing", "--stop-after", "generate"],
    ["--start-at", "validate", "--stop-after", "readme"],
])
def test_the_cli_exits_3_for_a_bundle_in_an_evidence_folder(fixture_registry, data_root,
                                                           monkeypatch, capsys, argv):
    from soma_synth import cli

    registry, calls_path = fixture_registry
    monkeypatch.setattr(stages_mod, "default_registry", lambda: registry)
    evidence = _evidence_bundle(data_root)
    before = _tree(data_root)
    code = cli.main(["run", str(evidence), "--source", "prism", "--data-root", str(data_root),
                     *argv])
    assert code == cli.EXIT_REFUSED
    assert "_superseded" in capsys.readouterr().err
    assert _tree(data_root) == before
    assert _calls(calls_path) == []


@pytest.mark.parametrize("args", [["--subjects", "S04"], ["--subjects=S04"], ["--subj", "S04"],
                                  ["--trials", "2"]])
def test_every_spelling_of_a_selection_flag_is_refused_for_the_production_lineage(
        fixture_registry, data_root, args):
    registry, calls_path = fixture_registry
    result = _runner(registry, data_root).run(stop_after="generate", corpus_args=args,
                                              log=_quiet)
    assert [o.status for o in result.outcomes] == ["refused"]
    assert _calls(calls_path) == []


def test_a_flag_that_selects_nothing_reaches_the_production_lineage(fixture_registry, data_root):
    """--compress, --resume or --force are not samples. The gaitex registry comment tells the
    operator to pass --compress through --corpus-arg."""
    registry, calls_path = fixture_registry
    result = _runner(registry, data_root).run(stop_after="generate", corpus_args=["--compress"],
                                              log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    assert _calls(calls_path)[0]["argv"][-1] == "--compress"


def test_a_lineage_directory_under_another_root_is_refused_too(fixture_registry, data_root,
                                                               tmp_path):
    """The data root can be a scratch folder while the bundle directory names the lineage of
    another root; the generators' own guard looks under the data root only."""
    registry, calls_path = fixture_registry
    other = tmp_path / "other_root" / POC / "hknu_unified8"
    result = _runner(registry, data_root, dataset_dir=other,
                     corpus_dir=data_root / "tmp" / "c").run(
        stop_after="generate", bundle_args=["--subjects", "S04"], log=_quiet)
    assert [o.status for o in result.outcomes] == ["refused"]
    assert _calls(calls_path) == []


def test_a_sample_run_does_not_build_the_production_corpus(fixture_registry, data_root):
    """A sample bundle in a scratch folder took the default corpus directory, the production
    corpus lineage, and built the whole corpus there when it was missing."""
    registry, calls_path = fixture_registry
    production = data_root / POC / "hknu_smpl24_paired"
    scratch = data_root / "tmp" / "sample" / "hknu_unified8"

    result = _runner(registry, data_root, dataset_dir=scratch).run(stop_after="generate",
                                                                   log=_quiet)
    assert [o.status for o in result.outcomes] == ["refused"]
    text = (result.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert "production corpus lineage" in text and "--corpus" in text
    assert "is not the production bundle lineage" in text
    assert _calls(calls_path) == [] and not production.exists()
    assert _manifest(result)["generation_decision"] is None        # refused before anything

    # the production bundle given a selection: refused by the bundle's own check, and the
    # corpus is not built either
    result = _runner(registry, data_root).run(stop_after="generate",
                                              bundle_args=["--trials=2"], log=_quiet)
    assert [o.status for o in result.outcomes] == ["refused"]
    assert _calls(calls_path) == [] and not production.exists()

    # a finished production corpus is only read by a sample: reused, not refused
    _prebuilt_corpus(data_root)
    result = _runner(registry, data_root, dataset_dir=scratch).run(
        stop_after="generate", bundle_args=["--subjects", "S04"], log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    assert _manifest(result)["corpus"]["action"] == "reused"
    assert [c["script"] for c in _calls(calls_path)] == ["bundle"]

    # ... but not rebuilt by one, even with --replace-existing
    result = _runner(registry, data_root, dataset_dir=scratch).run(
        stop_after="generate", rebuild_corpus=True, replace_existing=True, log=_quiet)
    assert [o.status for o in result.outcomes] == ["refused"]
    assert "refusing to rebuild the corpus" in (
        result.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert (production / "SUMMARY.json").read_text(encoding="utf-8") == '{"prebuilt": true}'

    # the production run itself still builds its own corpus (test above: missing corpus built)


@pytest.mark.parametrize("source, corpus_args, bundle_args, flag", [
    ("hknu", ["--out", "X"], [], "--out"),                         # the corpus output
    ("hknu", [], ["--out=X"], "--out"),                            # the bundle output
    ("hknu", [], ["--ou", "X"], "--out"),                          # an abbreviation of it
    ("hknu", [], ["--paired", "X"], "--paired"),                   # the corpus handed over
    ("hknu", [], ["--pa=X"], "--paired"),
    ("hknu", ["--root", "X"], [], "--root"),                       # a source location
    ("hknu", ["--ro=X"], [], "--root"),
    ("hknu", [], ["--hknu-root", "X"], "--hknu-root"),
    ("hknu", [], ["--source-root", "X"], "--source-root"),         # not even an hknu flag
    ("prism", [], ["--data-root", "X"], "--data-root"),
    ("prism", [], ["--source-root=X"], "--source-root"),
    ("prism", [], ["--out", "X"], "--out"),
    ("prism", [], ["--d", "X"], "--data-root"),
    ("amass", [], ["--out", "X"], "--out-root"),                   # argparse: --out-root
    ("amass", [], ["--amass-root", "X"], "--amass-root"),
])
def test_passthrough_arguments_cannot_set_a_flag_the_runner_controls(
        fixture_registry, data_root, source, corpus_args, bundle_args, flag):
    """argparse lets a repeated flag win, and the passthrough arguments come after the runner's
    own. A bundle, corpus or source somewhere the run record does not name is refused."""
    registry, calls_path = fixture_registry
    scratch = data_root / "tmp" / "sample"
    kwargs = {"dataset_dir": scratch / "bundle"}
    if source == "hknu":
        kwargs["corpus_dir"] = scratch / "corpus"
    result = _runner(registry, data_root, source=source, **kwargs).run(
        stop_after="generate", corpus_args=corpus_args, bundle_args=bundle_args, log=_quiet)
    assert [o.status for o in result.outcomes] == ["refused"]
    text = (result.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert f"sets {flag}, which the runner controls" in text
    assert _calls(calls_path) == []
    assert _manifest(result)["generation_decision"] is None


@pytest.mark.parametrize("bundle_args", [["--s", "S04"], ["--subj=S04"], ["--trials", "2"]])
def test_an_abbreviation_argparse_takes_for_another_flag_is_let_through(fixture_registry,
                                                                        data_root, bundle_args):
    """``--s`` is a prefix of ``--source-root`` but the hknu bundle parser reads it as
    ``--subjects``, its only flag starting so."""
    registry, calls_path = fixture_registry
    scratch = data_root / "tmp" / "sample"
    result = _runner(registry, data_root, dataset_dir=scratch / "bundle",
                     corpus_dir=scratch / "corpus").run(
        stop_after="generate", bundle_args=bundle_args, log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    assert _calls(calls_path)[-1]["argv"][-len(bundle_args):] == bundle_args


def test_every_location_flag_the_registry_declares_is_one_the_runner_controls():
    """A new entrypoint flag that names a root has to be added to the runner's list, or a
    passthrough argument could move the source unrecorded."""
    registry = stages_mod.default_registry()
    declared, outputs = set(), set()
    for name in registry.sources:
        source = registry.for_source(name)
        declared |= {*source.entrypoint.required_args, *source.entrypoint.optional_args}
        outputs.add(source.entrypoint.output_arg)
        if source.corpus is not None:
            declared |= {*source.corpus.required_args, *source.corpus.optional_args}
            outputs.add(source.corpus.output_arg)
    locations = {flag for flag in declared - outputs
                 if flag.endswith("-root") or flag in ("--extracted", "--root")}
    assert locations == set(runner_mod.SOURCE_LOCATION_FLAGS) | set(runner_mod.DATA_ROOT_FLAGS)


def test_controlled_args_resolves_spellings_the_way_argparse_does():
    declared = ("--out", "--paired", "--hknu-root", "--subjects", "--trials")
    controlled = ("--out", "--paired", "--source-root", "--root", "--hknu-root")
    args = ["--out=a", "--o", "b", "--s", "c", "--so=d", "--sub", "e", "--h", "f", "--r", "g",
            "--x", "S04", "value", "--"]
    assert runner_mod.controlled_args(args, controlled, declared) == [
        "--out=a", "--o", "--so=d", "--h", "--r"]


def test_the_resolved_config_names_the_source_root_the_generators_see(fixture_registry,
                                                                     data_root, tmp_path,
                                                                     monkeypatch):
    registry, calls_path = fixture_registry
    outside = tmp_path / "sources"
    monkeypatch.setenv(paths.SOURCE_ROOT_ENV, str(outside))
    result = _runner(registry, data_root, source="prism").run(stop_after="generate", log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"]
    config = _resolved(result)["config"]
    assert config["source_root"] == str(outside) and config["source_root_relative"] is None
    assert config["source_root_from"] == "SOMA_SOURCE_ROOT"
    assert _calls(calls_path)[0]["source_root"] == str(outside)   # what the child saw

    inside = data_root / "sources"
    monkeypatch.setenv(paths.SOURCE_ROOT_ENV, str(inside))
    result = _runner(registry, data_root, source="prism").run(
        stop_after="generate", skip_generate=True, log=_quiet)
    config = _resolved(result)["config"]
    assert config["source_root_relative"] == "sources" and "source_root" not in config

    monkeypatch.setenv(paths.SOURCE_ROOT_ENV, "relative/sources")
    with pytest.raises(paths.PathConfigError, match="SOMA_SOURCE_ROOT"):
        _runner(registry, data_root, source="prism").run(stop_after="generate", log=_quiet)


@pytest.mark.parametrize("where", ["dataset_dir", "runs_root", "catalog", "corpus_dir"])
@pytest.mark.parametrize("folder", ["Dropbox", "OneDrive - Org", "Team_Share"])
def test_outputs_in_a_synchronised_folder_are_refused_before_the_record_opens(
        fixture_registry, data_root, tmp_path, where, folder):
    registry, calls_path = fixture_registry
    synced = tmp_path / folder / "x"
    source = "hknu" if where == "corpus_dir" else "prism"
    kwargs = {"dataset_dir": data_root / "tmp" / "bundle", "runs_root": data_root / "tmp" / "_runs"}
    if source == "hknu":
        kwargs["corpus_dir"] = data_root / "tmp" / "corpus"
    catalog = data_root / "tmp" / "catalog"
    if where == "catalog":
        catalog = synced / "catalog"
    else:
        kwargs[where] = synced / where
    runner = _runner(registry, data_root, source=source, **kwargs)
    with pytest.raises(runner_mod.RunnerRefusal, match=f"refusing to run {source}"):
        runner.run(stop_after="register", catalog_dir=catalog, log=_quiet)
    assert not (tmp_path / folder).exists()
    assert not (data_root / "tmp").exists()                  # no run record anywhere
    assert _calls(calls_path) == []
    if where == "catalog":
        # a run that registers nothing does not write the catalog, and is not refused for it
        result = runner.run(stop_after="generate", skip_generate=True, catalog_dir=catalog,
                            log=_quiet)
        assert [o.status for o in result.outcomes] == ["skipped"]
    if where == "corpus_dir":
        # the runner moves a corpus aside for a rebuild and writes its marker and lock itself;
        # a corpus it only reads is not refused, and gets no lock beside it
        synced_corpus = kwargs["corpus_dir"]
        synced_corpus.mkdir(parents=True)
        (synced_corpus / "SUMMARY.json").write_text('{"synced": true}', encoding="utf-8")
        with pytest.raises(runner_mod.RunnerRefusal, match="the corpus directory"):
            runner.run(stop_after="generate", rebuild_corpus=True, replace_existing=True,
                       log=_quiet)
        assert sorted(path.name for path in synced_corpus.parent.iterdir()) == [
            synced_corpus.name]
        assert sorted(path.name for path in synced_corpus.iterdir()) == ["SUMMARY.json"]
        result = runner.run(stop_after="generate", log=_quiet)
        assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
        assert _manifest(result)["corpus"]["action"] == "reused"
        assert sorted(path.name for path in synced_corpus.parent.iterdir()) == [
            synced_corpus.name]


def test_the_cli_exits_3_for_a_synchronised_output(fixture_registry, data_root, tmp_path,
                                                   monkeypatch, capsys):
    from soma_synth import cli

    registry, calls_path = fixture_registry
    monkeypatch.setattr(stages_mod, "default_registry", lambda: registry)
    code = cli.main(["run", str(tmp_path / "Dropbox" / "bundle"), "--source", "prism",
                     "--data-root", str(data_root), "--stop-after", "generate"])
    assert code == cli.EXIT_REFUSED
    assert "cloud-synchronised" in capsys.readouterr().err
    assert _calls(calls_path) == []


def test_the_intended_basis_and_the_one_the_run_got_are_both_recorded(fixture_registry,
                                                                    data_root):
    registry, _ = fixture_registry
    result = _runner(registry, data_root, source="prism").run(stop_after="generate", log=_quiet)
    tracing = _manifest(result)["tracing"]
    assert _resolved(result)["config"]["generation_basis_intended"] == "standing_decision"
    assert tracing["generation_basis"] == tracing["generation_basis_intended"] == \
        _manifest(result)["generation_decision"]["basis"] == "standing_decision"
    assert tracing["decision_record_sha256"] == \
        _manifest(result)["generation_decision"]["decision_record_sha256"]

    # refused before the decision: it intended one and got none
    (data_root / POC / "prism_faithful_full" / "INDEX.json").write_text("{}", encoding="utf-8")
    refused = _runner(registry, data_root, source="prism").run(stop_after="generate", log=_quiet)
    tracing = _manifest(refused)["tracing"]
    assert tracing["generation_basis_intended"] == "standing_decision"
    assert tracing["generation_basis"] is None
    assert _manifest(refused)["generation_decision"] is None


def test_every_selection_flag_is_an_argument_of_its_entrypoint():
    registry = stages_mod.default_registry()
    for name in registry.sources:
        source = registry.for_source(name)
        declared = (*source.entrypoint.required_args, *source.entrypoint.optional_args)
        assert set(source.entrypoint.selection_args) <= set(declared), name
        assert source.entrypoint.selection_args, name
        if source.corpus is not None:
            declared = (*source.corpus.required_args, *source.corpus.optional_args)
            assert set(source.corpus.selection_args) <= set(declared), name
            assert source.corpus.selection_args, name
    assert "--compress" not in registry.for_source("gaitex").corpus.selection_args


def test_a_v2_entrypoint_without_selection_args_is_refused_at_load(tmp_path):
    raw = _raw_v2()
    del raw["sources"]["hknu"]["entrypoint"]["selection_args"]
    with pytest.raises(stages_mod.PipelineError, match="selection_args"):
        stages_mod.load(_write_registry(tmp_path, raw))
    raw = _raw_v2()
    raw["sources"]["hknu"]["corpus"]["selection_args"] = ["--not-an-argument"]
    with pytest.raises(stages_mod.PipelineError, match="selection_args"):
        stages_mod.load(_write_registry(tmp_path, raw))


def test_a_corpus_of_another_class_fails_the_step(fixture_registry, data_root):
    registry, calls_path = fixture_registry
    corpus = data_root / POC / "hknu_smpl24_paired"
    corpus.mkdir(parents=True)
    (corpus / "SUMMARY.json").write_text(
        json.dumps({"artifact_class": "canonical", "distribution_scope": "internal_only"}),
        encoding="utf-8")
    result = _runner(registry, data_root).run(stop_after="generate", log=_quiet)
    assert [o.status for o in result.outcomes] == ["failed"]
    assert "artifact_class" in result.outcomes[0].detail
    assert _manifest(result)["corpus"]["class_check"] == {
        "declared": True, "artifact_class": "canonical", "distribution_scope": "internal_only"}
    assert [c["script"] for c in _calls(calls_path)] == []       # the bundle never ran


def test_a_corpus_fingerprint_that_declares_no_class_is_recorded_as_undeclared(fixture_registry,
                                                                               data_root):
    """The addbiomechanics SUMMARY.json names no class (its artifacts do)."""
    registry, _ = fixture_registry
    _prebuilt_corpus(data_root)
    result = _runner(registry, data_root).run(stop_after="generate", log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"]
    assert _manifest(result)["corpus"]["class_check"] == {"declared": False}


# ======================================================================= any data root
@pytest.mark.parametrize("marker", ["MASTER.md", "state/local_archive_inventory.json"])
def test_a_data_root_holding_the_protected_data_plane_files_is_not_refused(fixture_registry, tmp_path,
                                                                          marker):
    """A data root that holds the protected data-plane files (retention rule 1.4) is not refused:
    the runner applies the same conditions on every machine and still makes the one class only."""
    registry, calls_path = fixture_registry
    plane = tmp_path / "plane"
    (plane / marker).parent.mkdir(parents=True, exist_ok=True)
    (plane / marker).write_text("marker", encoding="utf-8")
    result = _runner(registry, plane, source="prism").run(stop_after="generate", log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    decision = _manifest(result)["generation_decision"]
    assert decision["allowed"] is True and decision["basis"] == "standing_decision"
    assert decision["artifact_class"] == "experimental_non_candidate"
    assert decision["releases_holds"] is False
    assert "by an internal user on their own machine" in decision["note"]
    assert [c["script"] for c in _calls(calls_path)] == ["bundle", "post"]
    assert (plane / marker).read_text(encoding="utf-8") == "marker"      # untouched


def test_the_cli_exits_non_zero_on_a_refusal(fixture_registry, data_root, monkeypatch, capsys):
    from soma_synth import cli

    registry, calls_path = fixture_registry
    bundle = data_root / POC / "prism_faithful_full"
    bundle.mkdir(parents=True)
    (bundle / "INDEX.json").write_text('{"old": true}', encoding="utf-8")
    monkeypatch.setattr(stages_mod, "default_registry", lambda: registry)
    code = cli.main(["run", "--source", "prism", "--data-root", str(data_root),
                     "--stop-after", "generate"])
    assert code == cli.EXIT_REFUSED != 0
    assert "REF" in capsys.readouterr().out
    assert _calls(calls_path) == []


def test_a_failing_corpus_stops_the_step_before_the_bundle(fixture_registry, data_root,
                                                          monkeypatch):
    registry, calls_path = fixture_registry
    monkeypatch.setenv("FAKE_CORPUS_EXIT", "3")
    result = _runner(registry, data_root).run(stop_after="generate", log=_quiet)
    assert [o.status for o in result.outcomes] == ["failed"]
    assert result.outcomes[0].detail == "corpus entrypoint exited 3"
    assert [c["script"] for c in _calls(calls_path)] == ["corpus"]


def test_an_explicit_command_replaces_the_corpus_stage(fixture_registry, data_root):
    import sys

    registry, calls_path = fixture_registry
    runner = _runner(registry, data_root)
    command = [sys.executable, str(Path(registry.for_source("hknu").entrypoint.script)),
               "--out", str(runner.dataset_dir)]
    result = runner.run(stop_after="generate", generate_cmd=command, log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"]
    assert [c["script"] for c in _calls(calls_path)] == ["bundle"]
    assert _manifest(result)["corpus"] is None


# ======================================================================= post-steps
def test_the_post_step_runs_over_the_bundle_after_the_generator(fixture_registry, data_root):
    registry, calls_path = fixture_registry
    runner = _runner(registry, data_root, source="prism")
    result = runner.run(stop_after="generate", log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    calls = _calls(calls_path)
    assert [c["script"] for c in calls] == ["bundle", "post"]
    assert calls[0]["argv"] == ["--out", str(runner.dataset_dir)]
    assert calls[1]["argv"] == [str(runner.dataset_dir)]
    assert (runner.dataset_dir / "POST_STEP_RAN").exists()
    metrics = json.loads((result.run_directory / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stage_counts"]["post_step:fake_post.py"] == 1
    # prism_faithful_full has a limitations block, so the file is rendered after the post-step
    assert (runner.dataset_dir / "KNOWN_LIMITATIONS.json").exists()
    # the step completed, so the bundle is finished: no marker left
    assert not (runner.dataset_dir / runner_mod.GENERATING_MARKER).exists()
    assert _manifest(result)["bundle"]["generating_marker_removed"] is True
    assert metrics["stage_counts"]["bundle_generation_complete"] == 1


def test_a_failed_post_step_leaves_an_unfinished_bundle_that_nothing_takes(
        fixture_registry, data_root, monkeypatch, capsys):
    """The generator writes INDEX.json before the post-steps run, so a post-step that failed
    leaves a bundle that looks finished. Its .generating marker says it is not: validate,
    readme, register and pipeline refuse it, and no run resumes it -- the next generate
    step is refused until --replace-existing builds it again from the start. On a production
    directory what was moved aside is kept and named, and no _superseded evidence is asked for
    an INDEX.json no finished run wrote."""
    from soma_synth import cli

    registry, calls_path = fixture_registry
    bundle = data_root / POC / "prism_faithful_full"          # the production lineage
    monkeypatch.setenv("FAKE_POST_EXIT", "5")
    failed = _runner(registry, data_root, source="prism").run(stop_after="generate", log=_quiet)
    assert [o.status for o in failed.outcomes] == ["failed"]
    assert "post-step fake_post.py exited 5" in failed.outcomes[0].detail
    assert (bundle / "INDEX.json").is_file()        # the generator's, over unfinished takes
    marker = json.loads((bundle / runner_mod.GENERATING_MARKER).read_text(encoding="utf-8"))
    assert marker["source"] == "prism" and marker["run_identity"] == failed.run_identity
    assert marker["run_directory_relative"] == failed.run_directory.relative_to(
        data_root).as_posix()
    # it says who is building the bundle, and nothing to resume from
    assert set(marker) == {"schema", "note", "what", "source", "run_directory_relative",
                           "run_identity", "started_utc"}
    assert not (bundle / "KNOWN_LIMITATIONS.json").exists()
    assert not (failed.run_directory / "source_manifest.json").exists()
    assert _manifest(failed)["bundle"]["generating_marker_removed"] is False
    assert _sides(bundle) == [] and not runner_mod.lock_path(bundle).exists()

    # validate, readme and register refuse it, naming the command that builds it again
    for step in ("validate", "readme", "register"):
        refused = _runner(registry, data_root, source="prism").run(
            start_at=step, stop_after=step, log=_quiet)
        assert [(o.step, o.status) for o in refused.outcomes] == [(step, "refused")]
        text = (refused.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
        assert runner_mod.GENERATING_MARKER in text and failed.run_directory.name in text
        assert f"soma-synth run --source prism {bundle} --replace-existing" in text
    assert not (bundle / "VALIDATION_REPORT.json").exists()
    assert not (bundle / "README.md").exists()
    assert not (data_root / "experimental").exists()
    # and so does the pipeline command (the push script's refusal is tested in the parent
    # project, which keeps that script)
    assert cli.main(["pipeline", str(bundle), "--source", "prism", "--skip-generate",
                     "--data-root", str(data_root)]) == cli.EXIT_REFUSED
    assert "--replace-existing" in capsys.readouterr().err
    assert not (bundle / "README.md").exists()

    # no run resumes it
    monkeypatch.delenv("FAKE_POST_EXIT")
    blocked = _runner(registry, data_root, source="prism").run(stop_after="generate", log=_quiet)
    assert [o.status for o in blocked.outcomes] == ["refused"]
    text = (blocked.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert "an unfinished generation" in text and failed.run_directory.name in text
    assert [c["script"] for c in _calls(calls_path)] == ["bundle", "post"]

    # --replace-existing builds it again from the start; what it held is kept aside and named
    unfinished = _tree(bundle)
    again = _runner(registry, data_root, source="prism").run(
        stop_after="generate", replace_existing=True, log=_quiet)
    assert [o.status for o in again.outcomes] == ["ok"], again.outcomes
    assert [c["script"] for c in _calls(calls_path)] == ["bundle", "post", "bundle", "post"]
    assert not (bundle / runner_mod.GENERATING_MARKER).exists()
    assert (bundle / "KNOWN_LIMITATIONS.json").exists() and (bundle / "POST_STEP_RAN").exists()
    record = _manifest(again)["bundle"]
    assert record["found_at_open"].startswith("an unfinished generation")
    assert record["moved_aside"] is True and record["generating_marker_removed"] is True
    (aside,) = runner_mod.side_directories(bundle)
    assert _tree(aside) == unfinished and str(aside) in again.outcomes[0].detail
    assert not (data_root / POC / "_superseded").exists()

    # finished now: validate takes it
    validated = _runner(registry, data_root, source="prism").run(
        start_at="validate", stop_after="validate", log=_quiet)
    assert [o.status for o in validated.outcomes] != ["refused"]


def test_a_generator_that_fails_before_writing_leaves_the_directory_as_it_found_it(
        fixture_registry, data_root, monkeypatch):
    """The marker is written before the bundle entrypoint runs, which creates the directory; a
    generator refusing at start-up would otherwise leave a lineage directory holding only it.
    A missing directory stays missing, an empty one stays empty."""
    registry, _ = fixture_registry
    bundle = data_root / POC / "prism_faithful_full"
    monkeypatch.setenv("FAKE_BUNDLE_EARLY_EXIT", "2")
    result = _runner(registry, data_root, source="prism").run(stop_after="generate", log=_quiet)
    assert [o.detail for o in result.outcomes] == ["generator exited 2"]
    assert not bundle.exists()
    bundle.mkdir(parents=True)
    result = _runner(registry, data_root, source="prism").run(stop_after="generate", log=_quiet)
    assert [o.detail for o in result.outcomes] == ["generator exited 2"]
    assert bundle.is_dir() and list(bundle.iterdir()) == []


# ======================================================================= fresh directories
def _generation(bundle: Path, take: str) -> str:
    return json.loads((bundle / take / "manifest.json").read_text(encoding="utf-8"))["generation"]


def _bundle_argv(calls_path: Path) -> list[str]:
    return [c for c in _calls(calls_path) if c["script"] in ("bundle", "skipping_bundle")][-1]["argv"]


def _scratch_corpus(data_root: Path, name: str) -> Path:
    corpus = data_root / "tmp" / "sample" / name
    corpus.mkdir(parents=True)
    (corpus / "SUMMARY.json").write_text('{"prebuilt": true}', encoding="utf-8")
    return corpus


def _statuses(bundle: Path) -> dict[str, str]:
    index = json.loads((bundle / "INDEX.json").read_text(encoding="utf-8"))
    return {take["take_id"]: take["status"] for take in index["takes"]}


@pytest.mark.parametrize("source, lineage, corpus", [
    ("amass", "amass_faithful_full", None),
    ("prism", "prism_faithful_full", None),
    ("addbiomechanics", "addbio_unified8", "addbio_smpl24_raw"),
    ("hknu", "hknu_unified8", "hknu_smpl24_paired"),
])
def test_a_replacement_builds_into_a_fresh_directory(skipping_registry, data_root, monkeypatch,
                                                     source, lineage, corpus):
    """amass, prism and addbiomechanics skip a take whose files are there, so a replacement
    over the old directory reused every take and only the index was new; and no entrypoint
    removes a take, a shard index or a validation report it does not write. The old directory
    is moved aside whole, and the entrypoint -- given no flag -- builds into an empty one."""
    registry, calls_path = skipping_registry
    bundle = data_root / "tmp" / "sample" / lineage
    kwargs = {"dataset_dir": bundle}
    if corpus:
        # a corpus the run reuses: the replacement is the only reason to rebuild
        kwargs["corpus_dir"] = _scratch_corpus(data_root, corpus)

    monkeypatch.setenv("FAKE_GENERATION", "1")
    first = _runner(registry, data_root, source=source, **kwargs).run(stop_after="generate",
                                                                      log=_quiet)
    assert [o.status for o in first.outcomes] == ["ok"], first.outcomes
    assert _manifest(first)["bundle"]["moved_aside"] is False
    assert _manifest(first)["bundle"]["found_at_open"] is None
    old_index = _sha(bundle / "INDEX.json")
    (bundle / "take_stale").mkdir()
    (bundle / "take_stale" / "manifest.json").write_text('{"generation": "old"}', encoding="utf-8")
    (bundle / "INDEX_shard_0.json").write_text('{"shard": "old"}', encoding="utf-8")
    (bundle / "VALIDATION_REPORT.json").write_text('{"ok": true}', encoding="utf-8")

    monkeypatch.setenv("FAKE_GENERATION", "2")
    again = _runner(registry, data_root, source=source, **kwargs).run(
        stop_after="generate", replace_existing=True, log=_quiet)
    assert [o.status for o in again.outcomes] == ["ok"], again.outcomes
    assert "--force" not in _bundle_argv(calls_path)
    for stale in ("take_stale", "INDEX_shard_0.json", "VALIDATION_REPORT.json"):
        assert not (bundle / stale).exists(), stale
    if source != "hknu":                     # the hknu fake writes no generation into its takes
        assert _generation(bundle, "take_a") == _generation(bundle, "take_b") == "2"
    assert set(_statuses(bundle).values()) == {"ok"}
    evidence = _evidence(again, "bundle")
    assert evidence["INDEX.json"] == old_index
    assert {"INDEX_shard_0.json", "VALIDATION_REPORT.json"} <= set(evidence)
    assert _manifest(again)["bundle"]["moved_aside"] is True
    assert _manifest(again)["replaced"]["bundle"]["outcome"] == "deleted"
    assert _sides(bundle) == []


def test_a_killed_replacement_is_refused_until_the_operator_settles_it(skipping_registry,
                                                                       data_root, monkeypatch):
    """A replacement that was killed (no failure path ran) leaves the old generation aside and
    part of the new one, with its marker, in the directory. No run adopts or resumes that:
    without --replace-existing the directory is refused as unfinished, with it the aside
    directory beside it is refused, and the refusal says how to restore it or remove it. Once
    it is restored by hand, the replacement goes ahead."""
    registry, calls_path = skipping_registry
    bundle = data_root / "tmp" / "sample" / "amass_faithful_full"

    def run(**flags):
        return _runner(registry, data_root, source="amass", dataset_dir=bundle).run(
            stop_after="generate", log=_quiet, **flags)

    monkeypatch.setenv("FAKE_GENERATION", "1")
    assert [o.status for o in run().outcomes] == ["ok"]
    old = _tree(bundle)
    aside = bundle.with_name("amass_faithful_full.replaced-killed000001")
    os.replace(bundle, aside)
    (bundle / "take_a").mkdir(parents=True)
    (bundle / "take_a" / "manifest.json").write_text('{"generation": "2"}', encoding="utf-8")
    (bundle / runner_mod.GENERATING_MARKER).write_text("{}", encoding="utf-8")
    partial = _tree(bundle)

    refused = run()
    assert [o.status for o in refused.outcomes] == ["refused"]
    assert "an unfinished generation" in (refused.run_directory / "logs" / "refusal.txt").read_text(
        encoding="utf-8")
    refused = run(replace_existing=True)
    assert [o.status for o in refused.outcomes] == ["refused"]
    text = (refused.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert aside.name in text and "one generation at a time" in text
    assert f"rename {aside.name} back to {bundle.name}" in text and "retention rule 1.5" in text
    assert _tree(bundle) == partial and _tree(aside) == old
    assert len(_calls(calls_path)) == 1

    # restored as the refusal says, the replacement goes ahead (a scratch aside is deleted)
    shutil.rmtree(bundle)
    os.replace(aside, bundle)
    monkeypatch.setenv("FAKE_GENERATION", "3")
    result = run(replace_existing=True)
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    assert _generation(bundle, "take_a") == _generation(bundle, "take_b") == "3"
    assert _sides(bundle) == []


def test_an_operators_force_flag_still_reaches_the_entrypoint_once(skipping_registry, data_root):
    """The runner adds no flag; one the operator passes is a passthrough like any other."""
    registry, calls_path = skipping_registry
    bundle = data_root / "tmp" / "sample" / "prism_faithful_full"
    for replace in (False, True):
        result = _runner(registry, data_root, source="prism", dataset_dir=bundle).run(
            stop_after="generate", bundle_args=["--force"], replace_existing=replace,
            log=_quiet)
        assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
        assert _bundle_argv(calls_path).count("--force") == 1


# ======================================================================= locks
def test_a_locked_directory_is_refused_with_its_owner_and_age(fixture_registry, data_root):
    """Two generate steps never build into one directory at once: the lock beside it is taken
    with O_CREAT | O_EXCL, and a run that finds one is refused -- naming the process, the host,
    the run and the age -- without the lock or the directory being touched. No run removes a
    stale lock: the refusal says how to check it and remove it by hand."""
    registry, calls_path = fixture_registry
    scratch = data_root / "tmp" / "sample"
    bundle, corpus = scratch / "hknu_unified8", scratch / "hknu_smpl24_paired"

    def run():
        return _runner(registry, data_root, dataset_dir=bundle, corpus_dir=corpus).run(
            stop_after="generate", log=_quiet)

    for what, target, other in (("bundle", bundle, corpus), ("corpus", corpus, bundle)):
        lock = runner_mod.lock_path(target)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"pid": 4242, "host": "teammate-pc", "run_identity": "abc123",
                                    "started_utc": "2026-09-25T00:00:00Z"}), encoding="utf-8")
        held = lock.read_bytes()
        result = run()
        assert [o.status for o in result.outcomes] == ["refused"]
        text = (result.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
        assert f"the {what} directory {target} is locked by another generate step" in text
        assert "pid 4242 on teammate-pc" in text and "run abc123" in text
        assert "started 2026-09-25T00:00:00Z (" in text and "m ago)" in text
        assert 'tasklist /FI "PID eq 4242"' in text and "ps -p 4242" in text
        assert f"Delete {lock.name} by hand" in text and "No run deletes a lock" in text
        assert lock.read_bytes() == held                      # never removed by a run
        assert not runner_mod.lock_path(other).exists()       # the one this run took is gone
        assert _calls(calls_path) == [] and not bundle.exists() and not corpus.exists()
        assert _manifest(result)["generation_decision"] is None
        lock.unlink()

    # with no lock standing, the run takes both and removes both when the step ends
    result = run()
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    assert not runner_mod.lock_path(bundle).exists() and not runner_mod.lock_path(corpus).exists()
    metrics = json.loads((result.run_directory / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stage_counts"]["bundle_locked"] == metrics["stage_counts"]["corpus_locked"] == 1


@pytest.mark.parametrize("how", ["ok", "exit", "exception", "refusal", "lock taken over"])
def test_the_lock_is_removed_however_the_step_ends(fixture_registry, data_root, monkeypatch,
                                                   how):
    """The lock names this run while the step runs, and is removed when it ends -- succeeded,
    failed, died on an exception or refused -- unless it no longer holds what this run wrote
    (an operator removed it and another run took it), which is then that run's to remove."""
    import platform

    registry, _ = fixture_registry
    bundle = data_root / POC / "prism_faithful_full"
    lock = runner_mod.lock_path(bundle)
    seen = []
    real = runner_mod.generate_mod.run_generate

    def spy(*args, **kwargs):
        seen.append(json.loads(lock.read_text(encoding="utf-8")))
        if how == "exception":
            raise OSError("the disk went away")
        if how == "lock taken over":
            lock.write_text('{"pid": 1, "run_identity": "another"}', encoding="utf-8")
        return real(*args, **kwargs)

    monkeypatch.setattr(runner_mod.generate_mod, "run_generate", spy)
    if how == "exit":
        monkeypatch.setenv("FAKE_BUNDLE_EXIT", "3")
    if how == "refusal":
        bundle.mkdir(parents=True)
        (bundle / "notes.txt").write_text("mine", encoding="utf-8")
    result = _runner(registry, data_root, source="prism").run(stop_after="generate", log=_quiet)
    expected = {"ok": "ok", "exit": "failed", "exception": "failed", "refusal": "refused",
                "lock taken over": "ok"}[how]
    assert [o.status for o in result.outcomes] == [expected], result.outcomes
    if how == "lock taken over":
        assert json.loads(lock.read_text(encoding="utf-8"))["run_identity"] == "another"
        return
    assert not lock.exists()
    if how == "refusal":
        assert seen == []
        return
    (held,) = seen
    assert held["pid"] == os.getpid() and held["host"] == platform.node()
    assert held["run_identity"] == result.run_identity and held["what"] == "bundle"
    assert held["directory_relative"] == f"{POC}/prism_faithful_full"
    assert held["run_directory_relative"] == result.run_directory.relative_to(data_root).as_posix()


# ======================================================================= declared lineages
@pytest.mark.parametrize("source, where, lineage, meaning", [
    ("prism", "dataset_dir", "amass_faithful_full", "the amass bundle lineage"),
    ("prism", "dataset_dir", "hknu_smpl24_paired", "the hknu corpus lineage"),
    ("hknu", "dataset_dir", "hknu_smpl24_paired", "the hknu corpus lineage"),
    ("hknu", "corpus_dir", "gaitex_smpl24", "the gaitex corpus lineage"),
    ("hknu", "corpus_dir", "addbio_unified8", "the addbiomechanics bundle lineage"),
])
@pytest.mark.parametrize("root", ["own", "other"])
def test_another_declared_lineage_is_refused_before_the_record_opens(
        fixture_registry, data_root, tmp_path, source, where, lineage, meaning, root):
    """A source's bundle or corpus may be its own lineage or a scratch directory, never a
    lineage the registry declares for something else -- under this run's data root or any
    other: prism's generation written over amass's bundle, or hknu's bundle read from gaitex's
    corpus."""
    registry, calls_path = fixture_registry
    base = data_root if root == "own" else tmp_path / "other_root"
    target = base / POC / lineage
    target.mkdir(parents=True)
    (target / "INDEX.json").write_text('{"theirs": true}', encoding="utf-8")
    kwargs = {where: target}
    if source == "hknu":
        kwargs.setdefault("dataset_dir", data_root / "tmp" / "sample" / "bundle")
        kwargs.setdefault("corpus_dir", data_root / "tmp" / "sample" / "corpus")
    before = _tree(tmp_path)
    label = "the bundle directory" if where == "dataset_dir" else "the corpus directory"
    with pytest.raises(runner_mod.RunnerRefusal, match=label) as refused:
        _runner(registry, data_root, source=source, **kwargs).run(
            stop_after="register", replace_existing=True, log=_quiet)
    assert meaning in str(refused.value)
    assert _tree(tmp_path) == before and _calls(calls_path) == []


@pytest.mark.parametrize("where, name", [("dataset_dir", "prism_faithful_full.replaced-abc"),
                                         ("dataset_dir", "prism_faithful_full.failed-abc-2"),
                                         ("corpus_dir", "hknu_smpl24_paired.failed-abc")])
def test_an_aside_or_a_failed_output_is_refused_as_the_bundle_or_the_corpus(
        fixture_registry, data_root, where, name):
    registry, calls_path = fixture_registry
    target = data_root / "tmp" / name
    target.mkdir(parents=True)
    (target / "INDEX.json").write_text("{}", encoding="utf-8")
    source = "prism" if where == "dataset_dir" else "hknu"
    kwargs = {where: target}
    if source == "hknu":
        kwargs["dataset_dir"] = data_root / "tmp" / "bundle"
    before = _tree(data_root)
    with pytest.raises(runner_mod.RunnerRefusal, match=name.split(".")[0]):
        _runner(registry, data_root, source=source, **kwargs).run(stop_after="register",
                                                                  log=_quiet)
    assert _tree(data_root) == before and _calls(calls_path) == []
    if where == "dataset_dir":           # and whatever the steps: the directory is not a bundle
        with pytest.raises(runner_mod.RunnerRefusal, match="not a bundle directory"):
            _runner(registry, data_root, source=source, **kwargs).run(
                start_at="validate", stop_after="register", log=_quiet)
        assert _tree(data_root) == before


def test_a_bundle_of_another_class_fails_after_the_generator(fixture_registry, data_root,
                                                            monkeypatch):
    registry, _ = fixture_registry
    monkeypatch.setenv("FAKE_CLASS", "canonical")
    result = _runner(registry, data_root, source="prism").run(stop_after="generate", log=_quiet)
    assert [o.status for o in result.outcomes] == ["failed"]
    assert "artifact_class" in result.outcomes[0].detail
    assert (result.run_directory / "logs" / "artifact_class.txt").exists()
    assert not (result.run_directory / "source_manifest.json").exists()


# ======================================================================= tracing
def test_the_source_manifest_lists_every_reference_the_take_manifests_name(fixture_registry,
                                                                          data_root):
    registry, _ = fixture_registry
    result = _runner(registry, data_root, source="prism").run(stop_after="generate", log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"]
    payload = json.loads((result.run_directory / "source_manifest.json").read_text(encoding="utf-8"))
    corpus = f"{POC}/hknu_smpl24_paired"
    assert payload["schema"] == "source_manifest_v1"
    assert payload["manifests_scanned"] == 2
    assert payload["dataset_dir_relative"] == f"{POC}/prism_faithful_full"
    by_path = {(r["relative_path"], r["sha256"]): r for r in payload["references"]}
    assert set(by_path) == {
        (f"{corpus}/S01/t1.npz", "a" * 64),
        (f"{corpus}/S01/t2.npz", "b" * 64),
        ("extracted/fixture/markers.csv", "c" * 64),
        (corpus, None),
    }
    assert payload["count"] == 4
    assert by_path[("extracted/fixture/markers.csv", "c" * 64)]["takes"] == ["take_a", "take_b"]
    assert by_path[(f"{corpus}/S01/t1.npz", "a" * 64)]["fields"] == ["source"]
    assert by_path[(corpus, None)]["fields"] == ["paired_measured_counterpart"]
    lines = "\n".join(sorted(f"{p}\t{d or ''}" for p, d in by_path))
    assert payload["aggregate_sha256"] == hashlib.sha256(lines.encode("utf-8")).hexdigest()
    # settings files carry a sha256 but name no source; they are not references
    assert not any("x.yaml" in r["relative_path"] for r in payload["references"])

    tracing = _manifest(result)["tracing"]
    assert tracing["source_manifest"]["count"] == 4
    assert tracing["source_manifest"]["aggregate_sha256"] == payload["aggregate_sha256"]
    assert tracing["source_manifest"]["file_sha256"] == _sha(result.run_directory / "source_manifest.json")
    assert tracing["decision_record"].endswith("_source_distribution_and_pipeline_split_decision.md")


def test_the_source_manifest_holds_the_full_hash_of_every_source_file_the_stages_could_read(
        fixture_registry, data_root):
    """The hknu corpus names its sources by subject and trial only, so the run record carries
    the whole-file SHA-256 of the HKNU folder, narrowed to the run's subjects, beside the
    references the take manifests name."""
    registry, _ = fixture_registry
    hknu = data_root / "extracted" / "hknu_fullbody"
    files = {"MATLAB/DatasetInfo.xlsx": b"workbook",
             "Dataset_Processed/S01/S01_Npose.mat": b"s01",
             "Dataset_Processed/S02/S02_Npose.mat": b"s02"}
    for name, data in files.items():
        (hknu / name).parent.mkdir(parents=True, exist_ok=True)
        (hknu / name).write_bytes(data)
    scratch = data_root / "tmp"
    result = _runner(registry, data_root, dataset_dir=scratch / "bundle",
                     corpus_dir=scratch / "corpus").run(
        stop_after="generate", corpus_args=["--subjects", "S01"],
        bundle_args=["--subjects", "S01"], log=_quiet)
    assert [o.status for o in result.outcomes] == ["ok"]
    assert "2 source files hashed" in result.outcomes[0].detail
    payload = json.loads((result.run_directory / "source_manifest.json").read_text(encoding="utf-8"))
    traced = payload["source_files"]
    assert traced["schema"] == "source_files_v1" and traced["source_root_relative"] == "extracted"
    assert "source_root" not in traced                       # inside the data root: relative only
    assert {e["relative_path"]: (e["sha256"], e["bytes"], e["stages"]) for e in traced["files"]} == {
        "extracted/hknu_fullbody/MATLAB/DatasetInfo.xlsx":
            (hashlib.sha256(b"workbook").hexdigest(), 8, ["bundle", "corpus"]),
        "extracted/hknu_fullbody/Dataset_Processed/S01/S01_Npose.mat":
            (hashlib.sha256(b"s01").hexdigest(), 3, ["bundle", "corpus"]),
    }
    # the references the take manifests name are still there, unchanged
    assert payload["count"] == 4 and payload["schema"] == "source_manifest_v1"
    summary = _manifest(result)["tracing"]["source_manifest"]["source_files"]
    assert summary["count"] == 2 and summary["aggregate_sha256"] == traced["aggregate_sha256"]

    # a second sample run reusing that corpus: only the bundle stage reads the folder
    again = _runner(registry, data_root, dataset_dir=scratch / "bundle2",
                    corpus_dir=scratch / "corpus").run(stop_after="generate", log=_quiet)
    traced = json.loads((again.run_directory / "source_manifest.json").read_text(
        encoding="utf-8"))["source_files"]
    assert {s["stage"]: s["reads_source"] for s in traced["stages"]} == {"corpus": False,
                                                                         "bundle": True}
    assert traced["count"] == 3 and all(e["stages"] == ["bundle"] for e in traced["files"])


def test_a_source_without_a_corpus_points_at_its_take_manifests(fixture_registry, data_root):
    registry, _ = fixture_registry
    result = _runner(registry, data_root, source="prism").run(stop_after="generate", log=_quiet)
    traced = json.loads((result.run_directory / "source_manifest.json").read_text(
        encoding="utf-8"))["source_files"]
    assert traced["status"] == "not_recorded" and "take manifests" in traced["reason"]


def test_a_run_that_generated_nothing_says_so_in_the_tracing(fixture_registry, data_root):
    registry, _ = fixture_registry
    result = _runner(registry, data_root, source="prism").run(
        stop_after="generate", skip_generate=True, log=_quiet)
    assert _manifest(result)["tracing"]["source_manifest"]["status"] == provenance_mod.UNAVAILABLE


def test_take_source_references_reads_the_manifest_keys_of_each_bundle_type():
    manifest = {
        "source": {"relative_path": "extracted/prism/subj001/take002.pkl",
                   "source_asset_sha256_take": "e" * 64},
        "other": {"source_asset_id": "extracted/amass/x.npz"},
        "list": [{"relative_path": "extracted/gaitex/a.csv", "sha256": "f" * 64}],
    }
    found = provenance_mod.take_source_references(manifest)
    assert {(f["field"], f["relative_path"], f["sha256"]) for f in found} == {
        ("source", "extracted/prism/subj001/take002.pkl", "e" * 64),
        ("other", "extracted/amass/x.npz", None),
        ("list[0]", "extracted/gaitex/a.csv", "f" * 64),
    }


def test_the_smpl18_revision_is_recorded_beside_the_code_revision(fixture_registry, data_root):
    registry, _ = fixture_registry
    result = _runner(registry, data_root, source="prism").run(
        stop_after="generate", skip_generate=True, log=_quiet)
    lock = json.loads((result.run_directory / "environment.lock").read_text(encoding="utf-8"))
    manifest = _manifest(result)
    record = lock["smpl18_revision"]
    assert record["package"] == "smpl18" and record["version"]
    assert manifest["provenance"]["smpl18_revision"]["source"] == record["source"]
    assert manifest["tracing"]["smpl18_revision"] == manifest["provenance"]["smpl18_revision"]
    checkout = REPO / "packages" / "smpl18"
    if record["source"] == "git":
        head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        assert record["commit"] == head
        assert record["checkout"] == "packages/smpl18"
    else:
        assert record["files"] and len(record["files_digest"]) == 64


def test_an_installed_smpl18_is_recorded_by_its_files(tmp_path):
    package = tmp_path / "site-packages" / "smpl18"
    (package / "model").mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = 'x'\n", encoding="utf-8")
    (package / "model" / "select.py").write_text("pass\n", encoding="utf-8")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "junk.pyc").write_bytes(b"\0")
    record = provenance_mod.smpl18_revision(package)
    assert record["source"] == "package_files"
    assert set(record["files"]) == {"__init__.py", "model/select.py"}
    assert record["files"]["model/select.py"] == _sha(package / "model" / "select.py")
    assert "commit" not in record


def test_the_dirty_paths_keep_their_first_character(monkeypatch):
    """`git status --porcelain` starts a line with a space; stripping the whole output would
    turn ' M README.md' into 'EADME.md' in a run record's code revision."""
    import types

    outputs = {"rev-parse": "abc123\n", "status": " M README.md\n?? configs/new.yaml\n",
               "diff": "diff --git a/README.md b/README.md\n"}

    def fake_run(argv, **kwargs):
        return types.SimpleNamespace(returncode=0, stdout=outputs[argv[3]])

    monkeypatch.setattr(provenance_mod.subprocess, "run", fake_run)
    record = provenance_mod._git_revision(Path("."))
    assert record["commit"] == "abc123"
    assert record["dirty_paths"] == ["README.md", "configs/new.yaml"]
    assert record["patch_scope"] == provenance_mod.PATCH_SCOPE


def test_openpyxl_is_a_tracked_distribution():
    assert "openpyxl" in provenance_mod.dependency_versions()


# ======================================================================= body models
def _models(folder: Path, salt: bytes) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for gender in ("male", "female", "neutral"):
        name = smpl18_select.model_filename(gender)
        (folder / name).write_bytes(salt + gender.encode())
        hashes[name] = _sha(folder / name)
    return hashes


def test_the_default_model_folder_is_hashed_under_the_data_root(fixture_registry, data_root):
    registry, _ = fixture_registry
    expected = _models(data_root / "body_models" / "smpl", b"default-")
    result = _runner(registry, data_root, source="prism").run(
        stop_after="generate", skip_generate=True, log=_quiet)
    manifest = _manifest(result)
    record = manifest["provenance"]["body_model"]
    assert record["directory"] == str(data_root / "body_models" / "smpl")
    assert record["directory_relative"] == "body_models/smpl"
    assert record["directory_from"] == "data_root"
    assert record["models"] == expected
    assert manifest["run_identity"]["inputs"]["body_model_sha256"] == body_models.set_digest(expected)


def test_soma_body_model_dir_is_the_folder_hashed(fixture_registry, data_root, tmp_path,
                                                 monkeypatch):
    """The runner resolves the folder the generators load from, and records that one."""
    registry, _ = fixture_registry
    _models(data_root / "body_models" / "smpl", b"default-")
    moved = tmp_path / "elsewhere" / "models"
    expected = _models(moved, b"moved-")
    monkeypatch.setenv(paths.BODY_MODEL_DIR_ENV, str(moved))
    runner = _runner(registry, data_root, source="prism")
    assert runner.body_model_directory() == moved == paths.body_model_dir(root=data_root)
    result = runner.run(stop_after="generate", skip_generate=True, log=_quiet)
    manifest = _manifest(result)
    record = manifest["provenance"]["body_model"]
    assert record["directory"] == str(moved)
    assert record["directory_from"] == "SOMA_BODY_MODEL_DIR"
    assert record["directory_relative"] is None                # outside the data root
    assert record["models"] == expected
    assert manifest["run_identity"]["inputs"]["body_model_sha256"] == body_models.set_digest(expected)
    config = _resolved(result)["config"]
    assert config["body_model_dir"] == str(moved)
    assert config["body_model_dir_from"] == "SOMA_BODY_MODEL_DIR"


def test_a_per_file_model_override_is_recorded(fixture_registry, data_root, tmp_path,
                                               monkeypatch):
    """anthro_smpl honours SOMA_SMPL_MODEL_{MALE,FEMALE}, which bypass the hashed folder."""
    registry, _ = fixture_registry
    model = tmp_path / "custom_male.npz"
    model.write_bytes(b"custom")
    monkeypatch.setenv("SOMA_SMPL_MODEL_MALE", str(model))
    monkeypatch.delenv("SOMA_SMPL_MODEL_FEMALE", raising=False)
    result = _runner(registry, data_root, source="prism").run(
        stop_after="generate", skip_generate=True, log=_quiet)
    # outside the data root: the absolute path is the only form there is
    assert _resolved(result)["config"]["body_model_file_overrides"] == {
        "SOMA_SMPL_MODEL_MALE": {"path_relative": None, "path": str(model), "sha256": _sha(model)}}

    inside = data_root / "models" / "male.npz"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"inside")
    monkeypatch.setenv("SOMA_SMPL_MODEL_MALE", str(inside))
    result = _runner(registry, data_root, source="prism").run(
        stop_after="generate", skip_generate=True, log=_quiet)
    assert _resolved(result)["config"]["body_model_file_overrides"] == {
        "SOMA_SMPL_MODEL_MALE": {"path_relative": "models/male.npz", "sha256": _sha(inside)}}


def test_set_directory_follows_paths_body_model_dir(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.BODY_MODEL_DIR_ENV, raising=False)
    assert body_models.set_directory("body_models/smpl", tmp_path) == \
        paths.body_model_dir(root=tmp_path)
    assert body_models.set_directory("body_models/smpl", None) is None
    assert body_models.set_directory("", tmp_path) is None
    monkeypatch.setenv(paths.BODY_MODEL_DIR_ENV, str(tmp_path / "m"))
    assert body_models.set_directory("body_models/smpl", tmp_path) == tmp_path / "m"
    assert body_models.set_directory("body_models/smpl", None) == tmp_path / "m"


# ======================================================================= production directories and locks
@pytest.mark.parametrize("where", ["dataset_dir", "corpus_dir"])
def test_a_directory_inside_a_production_lineage_is_refused(fixture_registry, data_root, where):
    """A folder inside a lineage directory (a take of the production bundle, say) is someone's
    payload even though it is not directly under the container: refused before the record opens,
    whatever --replace-existing says, so nothing of it is ever moved aside or deleted."""
    registry, calls_path = fixture_registry
    lineage = data_root / POC / ("hknu_unified8" if where == "dataset_dir" else "hknu_smpl24_paired")
    inner = lineage / "take_stale"
    inner.mkdir(parents=True)
    (inner / "manifest.json").write_text('{"old": true}', encoding="utf-8")
    kwargs = {"dataset_dir": data_root / "tmp" / "sample" / "bundle",
              "corpus_dir": data_root / "tmp" / "sample" / "corpus", where: inner}
    before = _tree(data_root)
    with pytest.raises(runner_mod.RunnerRefusal, match="inside a production directory"):
        _runner(registry, data_root, **kwargs).run(stop_after="generate", replace_existing=True,
                                                   rebuild_corpus=True, log=_quiet)
    assert _tree(data_root) == before and _calls(calls_path) == []


def test_production_depth_counts_below_the_container_as_written_or_resolved(tmp_path):
    container = tmp_path / POC
    assert runner_mod.production_depth(container) == 0
    assert runner_mod.production_depth(container / "hknu_unified8") == 1
    assert runner_mod.production_depth(container / "hknu_unified8" / "take") == 2
    assert runner_mod.production_depth(tmp_path / "tmp" / "sample") is None
    assert runner_mod.is_production_directory(container / "x" / "y")
    assert not runner_mod.is_production_directory(container)


def test_a_fresh_build_beside_a_leftover_aside_is_refused(fixture_registry, data_root):
    """A run killed between moving a bundle aside and creating the new one leaves no bundle and an
    aside beside it; building fresh there would silently leave the old generation standing."""
    registry, calls_path = fixture_registry
    bundle = data_root / "tmp" / "sample" / "hknu_unified8"
    aside = bundle.with_name(bundle.name + ".replaced-killed000001")
    aside.mkdir(parents=True)
    (aside / "INDEX.json").write_text("{}", encoding="utf-8")
    corpus = _prebuilt_corpus(data_root)
    result = _runner(registry, data_root, dataset_dir=bundle, corpus_dir=corpus).run(
        stop_after="generate", log=_quiet)
    assert [o.status for o in result.outcomes] == ["refused"]
    text = (result.run_directory / "logs" / "refusal.txt").read_text(encoding="utf-8")
    assert f"refusing to build the bundle {bundle}" in text and aside.name in text
    assert not bundle.exists() and aside.is_dir() and _calls(calls_path) == []


def test_the_lock_is_taken_on_the_resolved_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a").mkdir()
    assert runner_mod.lock_path(Path("a") / ".." / "bundle") == (tmp_path / "bundle.lock").resolve()


def test_an_interrupted_shard_fan_out_stops_every_shard(monkeypatch):
    """Settling a failed run renames the directory the shards write into, so an interrupt while
    the shards run must stop them all before it propagates."""
    from soma_synth.pipeline import generate as generate_mod

    started = []

    class FakeProc:
        def __init__(self, argv, env=None):
            self.terminated = False
            started.append(self)

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

        def wait(self, timeout=None):
            if not self.terminated:
                raise KeyboardInterrupt
            return -15

    monkeypatch.setattr(generate_mod, "plan_generate_commands",
                        lambda *a, **k: ([["x", "1"], ["x", "2"]], None))
    monkeypatch.setattr(generate_mod.subprocess, "Popen", FakeProc)
    with pytest.raises(KeyboardInterrupt):
        generate_mod.run_generate("amass", "not-created", None, None, jobs=2)
    assert len(started) == 2 and all(proc.terminated for proc in started)
