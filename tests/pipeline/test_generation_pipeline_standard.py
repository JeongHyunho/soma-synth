"""The generation pipeline against what PIPELINE_GOVERNANCE.md actually requires of it.

These tests are written against the sealed document's own words, not against the implementation,
so that a future edit which quietly drops a required artifact or provenance item fails here. The
sealed document itself is the parent project's; the test that reads its §5.1 verbatim stays
there, and the stage list it pins is the one ``stages.GOVERNANCE_STAGES`` carries here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from soma_synth.pipeline import provenance as prov
from soma_synth.pipeline import run_identity as rid
from soma_synth.pipeline import stages as stages_mod
from soma_synth.pipeline.run_directory import (
    REQUIRED_ARTIFACTS,
    RunDirectory,
    RunDirectoryError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STANDARD = REPO_ROOT / "docs/guides/GENERATION_PIPELINE_STANDARD.md"


# ------------------------------------------------------------------ stages
def test_the_registry_and_the_enum_cannot_disagree_about_the_stage_list():
    registry = stages_mod.default_registry()
    assert registry.canonical_stages == tuple(stages_mod.Stage)


def test_only_one_stage_is_declared_source_variant():
    """Every other stage does the same work for every source; that is what makes a standard
    possible at all. If a second stage becomes source-specific, this should be a deliberate act."""
    assert stages_mod.SOURCE_VARIANT_STAGE is stages_mod.Stage.SYNTHESISE_SMALL


def test_every_source_declares_a_known_resume_predicate():
    registry = stages_mod.default_registry()
    for name, source in registry.sources.items():
        assert source.resume in registry.resume_predicates, name


def test_all_five_sources_are_declared():
    registry = stages_mod.default_registry()
    assert set(registry.sources) == {"amass", "prism", "addbiomechanics", "gaitex", "hknu"}


def test_every_declared_entrypoint_script_exists():
    registry = stages_mod.default_registry()
    for name, source in registry.sources.items():
        assert (REPO_ROOT / source.entrypoint.script).is_file(), f"{name}: {source.entrypoint.script}"


def test_an_undeclared_source_is_refused_by_name():
    registry = stages_mod.default_registry()
    with pytest.raises(stages_mod.PipelineError, match="not declared"):
        registry.for_source("no_such_source")


def test_a_registry_whose_stage_list_disagrees_with_the_code_is_refused(tmp_path: Path):
    raw = yaml.safe_load(stages_mod.DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
    raw["canonical_stages"] = raw["canonical_stages"][:-1]
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(stages_mod.PipelineError, match="disagree"):
        stages_mod.load(broken)


# -------------------------------------------------------------- provenance
def test_the_generation_path_carries_no_randomness():
    """seeds() records that there is nothing to seed. This is what stops that becoming a lie.

    If a stochastic step ever enters generation, it must arrive with a recorded seed rather than
    behind a provenance field that says the path is deterministic.
    """
    prov.assert_generation_path_is_deterministic()


@pytest.fixture
def rng_scope_repo(tmp_path, monkeypatch):
    experiment = tmp_path / "src/soma_synth/experimental/calibration.py"
    experiment.parent.mkdir(parents=True)
    experiment.write_text(
        "import numpy as np\nnp.random.default_rng(1)\n", encoding="utf-8", newline="\n",
    )
    monkeypatch.setattr(prov, "_REPOSITORY_ROOT", tmp_path)
    return tmp_path


def test_rng_audit_excludes_non_generation_experiments(rng_scope_repo):
    assert prov.find_rng_uses() == []
    prov.assert_generation_path_is_deterministic()


@pytest.mark.parametrize("relative_path", [
    "src/soma_synth/kinematics/core.py",
    "src/soma_synth/experimental_tools/core.py",
    "scripts/poc/generate_core.py",
])
def test_rng_audit_still_rejects_production_randomness(rng_scope_repo, relative_path):
    core = rng_scope_repo / relative_path
    core.parent.mkdir(parents=True)
    core.write_text("import numpy as np\nnp.random.default_rng(1)\n", encoding="utf-8")
    assert any(Path(path).name == core.name for path, _, _ in prov.find_rng_uses())
    with pytest.raises(AssertionError, match="randomness was found"):
        prov.assert_generation_path_is_deterministic()


@pytest.mark.parametrize("relative_path, statement", [
    ("src/soma_synth/kinematics/core.py",
     "from soma_synth.experimental import calibration"),
    ("src/soma_synth/kinematics/core.py",
     "import soma_synth.experimental.calibration as calibration"),
    ("src/soma_synth/kinematics/core.py",
     "from soma_synth import experimental"),
    ("src/soma_synth/kinematics/core.py",
     "from ..experimental import calibration"),
    ("src/soma_synth/kinematics/core.py",
     "from .. import experimental"),
    ("src/soma_synth/__init__.py", "from .experimental import calibration"),
    ("scripts/poc/generate_fixture.py",
     "from soma_synth.experimental import calibration"),
    ("scripts/poc/generate_fixture.py",
     "importlib.import_module('soma_synth.experimental.calibration')"),
    ("scripts/poc/generate_fixture.py",
     "__import__('soma_synth.experimental.calibration')"),
    ("scripts/poc/generate_fixture.py",
     ("from importlib import import_module as load\n"
      "load('soma_synth.experimental.calibration')")),
    pytest.param(
        "scripts/poc/generate_fixture.py",
        "importlib.import_module(name='soma_synth.experimental.calibration')",
        id="keyword_import",
    ),
    pytest.param(
        "scripts/poc/generate_fixture.py",
        ("from importlib import import_module as load\n"
         "load(name='soma_synth.experimental.calibration')"),
        id="keyword_alias_import",
    ),
    pytest.param(
        "scripts/poc/generate_fixture.py",
        "__import__(name='soma_synth.experimental.calibration')",
        id="keyword_builtin_import",
    ),
])
def test_rng_audit_refuses_experimental_imports(rng_scope_repo, relative_path, statement):
    core = rng_scope_repo / relative_path
    core.parent.mkdir(parents=True, exist_ok=True)
    core.write_text(statement + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(AssertionError, match="generation path imports experimental"):
        prov.assert_generation_path_is_deterministic()


def test_rng_boundary_refuses_unparseable_sources(rng_scope_repo):
    core = rng_scope_repo / "src/soma_synth/broken.py"
    core.write_text("def invalid(:\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        prov.assert_generation_path_is_deterministic()


def test_rng_boundary_refuses_unreadable_sources(rng_scope_repo, monkeypatch):
    core = rng_scope_repo / "src/soma_synth/unreadable.py"
    core.write_text("pass\n", encoding="utf-8")
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == core:
            raise OSError("fixture access denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(OSError, match="fixture access denied"):
        prov.assert_generation_path_is_deterministic()


def test_seeds_states_the_absence_rather_than_leaving_it_blank():
    record = prov.seeds()
    assert record["present"] is False
    assert record["reason"], "an absent seed needs a reason; a blank cannot be audited"


def test_host_environment_carries_nothing_identifying():
    """§10.2: no credential, token, cookie or personal identifier."""
    record = prov.host_environment()
    rendered = json.dumps(record).lower()
    for forbidden in ("user", "hostname", "home", "token", "password", "cookie"):
        assert forbidden not in rendered


def test_an_unmeasurable_item_is_unavailable_with_a_reason_not_a_guess():
    record = prov.body_model_hash(None)
    assert record["status"] == prov.UNAVAILABLE
    assert record["reason"]


def test_provenance_covers_every_item_section_10_2_requires():
    record = prov.ProvenanceRecord(
        source_name="gaitex", spec_id="qmd_unified8_smpl18", spec_version="faithful-v2"
    ).as_json()
    for key in (
        "code_revision", "contract_versions", "source_assets", "spec_snapshot",
        "dependency_versions", "body_model", "seeds", "host_environment",
        "output_artifacts", "quality",
    ):
        assert key in record, key


# ------------------------------------------------------------ run identity
def _inputs(**overrides):
    base = {
        "spec_id": "qmd_unified8_smpl18", "spec_version": "faithful-v2",
        "source_name": "gaitex", "entrypoint": "scripts/poc/generate_gaitex_unified8.py",
    }
    base.update(overrides)
    return rid.RunIdentityInputs(**base)


def test_the_same_inputs_give_the_same_identity():
    assert rid.compute(_inputs()) == rid.compute(_inputs())


@pytest.mark.parametrize(
    "field,value",
    [
        ("code_revision", "cafe1234"),
        ("source_name", "hknu"),
        ("body_model_sha256", "a" * 64),
        ("input_corpus_sha256", "b" * 64),
        ("resolved_config_sha256", "c" * 64),
    ],
)
def test_changing_any_identity_input_changes_the_identity(field, value):
    """§10.1 asks the identity to cover config, code and source. Each must actually move it."""
    assert rid.compute(_inputs()) != rid.compute(_inputs(**{field: value}))


def test_the_identity_explains_how_to_recompute_itself():
    explained = rid.explain(_inputs())
    assert explained["run_identity"] == rid.compute(_inputs())
    assert "sha256" in explained["algorithm"]
    assert explained["canonical_json"]


# ---------------------------------------------------------- run directory
def test_a_closed_run_directory_holds_all_seven_required_artifacts(tmp_path: Path):
    """§10.1's list, in full."""
    root = tmp_path / rid.run_directory_name(_inputs())
    record = prov.ProvenanceRecord(
        source_name="gaitex", spec_id="qmd_unified8_smpl18", spec_version="faithful-v2"
    )
    run = RunDirectory(root=root, identity_inputs=_inputs(), provenance=record).open()
    run.close(takes_written=0)
    assert RunDirectory.missing_artifacts(root) == []
    for name in REQUIRED_ARTIFACTS:
        assert (root / name).exists(), name


def test_a_skipped_take_is_recorded_because_a_silent_skip_is_an_omission(tmp_path: Path):
    root = tmp_path / "run_x"
    record = prov.ProvenanceRecord(
        source_name="hknu", spec_id="qmd_unified8_smpl18", spec_version="faithful-v2"
    )
    run = RunDirectory(root=root, identity_inputs=_inputs(source_name="hknu"), provenance=record).open()
    run.record_skip("take-0007")
    run.record_exclusion("take-0009", "marker gap exceeds policy")
    run.close(takes_written=3)

    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["takes_skipped"] == 1
    assert metrics["skipped_take_ids"] == ["take-0007"]
    exclusions = json.loads((root / "exclusions.json").read_text(encoding="utf-8"))
    assert exclusions["excluded"][0]["reason"]


def test_a_closed_run_cannot_be_reopened(tmp_path: Path):
    """§10.1 calls the run directory immutable."""
    root = tmp_path / "run_y"
    record = prov.ProvenanceRecord(
        source_name="amass", spec_id="qmd_unified8_smpl18", spec_version="faithful-v2"
    )
    run = RunDirectory(root=root, identity_inputs=_inputs(source_name="amass"), provenance=record).open()
    run.close(takes_written=1)
    with pytest.raises(RunDirectoryError, match="already closed"):
        run.close(takes_written=1)


def test_the_run_manifest_says_it_is_not_the_take_manifest(tmp_path: Path):
    """Two different documents share the filename; the run one says so, so a reader is not
    left to infer which contract governs it."""
    root = tmp_path / "run_z"
    record = prov.ProvenanceRecord(
        source_name="prism", spec_id="qmd_unified8_smpl18", spec_version="faithful-v2"
    )
    run = RunDirectory(root=root, identity_inputs=_inputs(source_name="prism"), provenance=record).open()
    run.close(takes_written=1)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert "12.2" in manifest["note"]
    assert manifest["distribution_scope"] == "internal_only"


# -------------------------------------------------------------- documents
def test_the_standard_declares_itself_subordinate_to_the_sealed_document():
    text = STANDARD.read_text(encoding="utf-8")
    assert "PIPELINE_GOVERNANCE.md" in text
    assert "봉인" in text


def test_the_standard_records_the_sealed_documents_stale_scope_rather_than_editing_it():
    """§0 of the pinned document still says three sources, 6 IMU and 15 joints. The pinned
    document is not edited from this repository, so the standard records the discrepancy itself."""
    text = STANDARD.read_text(encoding="utf-8")
    assert "8채널" in text and "18관절" in text


def test_the_per_source_table_is_generated_and_matches_the_registry():
    """The generated block cannot fall behind the registry."""
    from soma_synth.pipeline import render

    text = STANDARD.read_text(encoding="utf-8")
    assert render.BEGIN in text and render.END in text
    generated = text.split(render.BEGIN)[1].split(render.END)[0]
    registry = stages_mod.default_registry()
    for name in registry.sources:
        assert f"`{name}`" in generated, name


def test_rendering_twice_changes_nothing(tmp_path: Path):
    from soma_synth.pipeline import render

    document = tmp_path / "doc.md"
    document.write_text("# heading\n\nbody\n", encoding="utf-8")
    render.write_into(document)
    once = document.read_text(encoding="utf-8")
    render.write_into(document)
    assert document.read_text(encoding="utf-8") == once


# -------------------------------------------------------------------- CLI
def test_every_source_resolves_a_generator_command_without_a_hand_written_override():
    """The point of the registry. Before it, three of the five returned None here and could not
    be driven by `pipeline` at all."""
    from soma_synth.cli import _default_generate_cmd

    dataset = Path("E:/nowhere/bundle_v1")
    for name in stages_mod.default_registry().sources:
        command = _default_generate_cmd(name, dataset, Path("E:/nowhere"))
        assert command is not None, name
        assert command[1].endswith(".py"), name


def test_an_unknown_source_still_returns_none_so_the_callers_contract_holds():
    from soma_synth.cli import _default_generate_cmd

    assert _default_generate_cmd("no_such_source", Path("E:/x/y"), None) is None


@pytest.mark.parametrize(
    "source,expected_flag",
    # prism wrote through --data-root in registry v1; v2 (ADR-0041) writes through --out
    [("amass", "--out-root"), ("prism", "--out"), ("gaitex", "--out")],
)
def test_the_output_flag_comes_from_the_registry_not_from_a_branch(source, expected_flag):
    from soma_synth.cli import _default_generate_cmd

    command = _default_generate_cmd(source, Path("E:/x/bundle"), Path("E:/x"))
    assert command[2] == expected_flag


def test_the_standard_names_the_registry_it_is_generated_from():
    """The machine-readable twin in the header and the generated block follow the default
    registry, which is v2."""
    text = STANDARD.read_text(encoding="utf-8")
    name = stages_mod.DEFAULT_REGISTRY_PATH.name
    assert name == "source_pipelines_v2.yaml"
    header = text.split("\n---\n", 1)[0]
    assert f"configs/datasets/{name}" in header
    from soma_synth.pipeline import render

    generated = text.split(render.BEGIN)[1].split(render.END)[0]
    assert render.render_table().split(render.BEGIN)[1].split(render.END)[0] == generated


def test_the_standard_states_the_generation_policy_and_the_tracing():
    """§9 is where ADR-0041's standing decision is written down for a reader of the standard."""
    text = STANDARD.read_text(encoding="utf-8")
    section = text.split("\n## 9.")[1].split("\n## 10.")[0]
    for needle in ("ADR-0041", "experimental_non_candidate", "--authorized-by",
                   "source_manifest.json", "smpl18", "releases_holds: false"):
        assert needle in section, needle
    mismatches = text.split("\n## 10.")[1]
    assert "PIPELINE_GOVERNANCE.md` §2" in mismatches
    assert "LOCAL_DATA_PLANE.md" in mismatches


# ------------------------------------------------------------------ audit
def test_the_audit_names_what_cannot_be_recovered_instead_of_implying_it_was_missed():
    from soma_synth.pipeline import audit

    for item, reason in audit.NOT_RECOVERABLE.items():
        assert item in audit.PROVENANCE_ITEMS
        assert reason


def test_the_retrofit_warns_that_its_own_environment_is_not_the_generating_one(tmp_path: Path):
    """The retrofit records the machine that wrote the sidecar. Left unlabelled, a reader would
    take it for the machine that made the bundle."""
    from soma_synth.pipeline import audit

    bundle = tmp_path / "bundle_v1"
    take = bundle / "take-0001"
    take.mkdir(parents=True)
    (take / "manifest.json").write_text(
        json.dumps({"source": {"source_name": "gaitex"}, "spec_version": "faithful-v2"}),
        encoding="utf-8",
    )
    record = audit.retrofit_provenance(bundle, write=True)
    assert "NOT the machine that" in record["retrofitted_by"]["warning"]
    assert (bundle / audit.SIDECAR_NAME).exists()


def test_the_retrofit_does_not_touch_the_take_manifest(tmp_path: Path):
    from soma_synth.pipeline import audit

    bundle = tmp_path / "bundle_v1"
    take = bundle / "take-0001"
    take.mkdir(parents=True)
    manifest = take / "manifest.json"
    manifest.write_text(json.dumps({"source": {"source_name": "hknu"}}), encoding="utf-8")
    before = manifest.read_bytes()
    audit.retrofit_provenance(bundle, write=True)
    assert manifest.read_bytes() == before
