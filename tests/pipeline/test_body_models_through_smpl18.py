"""The pipeline resolves and records the body-model set through smpl18.

§10.2 item 6 (the model a run generated from) was the heaviest remaining provenance gap: the
runner never knew where the models were, so every run record said "unavailable". The directory
now comes from the registry and the file names from smpl18, which is what the generators read.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from smpl18.model import select as smpl18_select
from soma_synth.pipeline import body_models, provenance as provenance_mod
from soma_synth.pipeline import stages as stages_mod


@pytest.fixture
def model_set(tmp_path):
    root = tmp_path / "body_models" / "smpl"
    root.mkdir(parents=True)
    written = {}
    for gender, payload in (("male", b"m"), ("female", b"ff"), ("neutral", b"nnn")):
        name = smpl18_select.model_filename(gender)
        (root / name).write_bytes(payload)
        written[name] = hashlib.sha256(payload).hexdigest()
    return root, written


class TestTheRegistryDeclaresTheSet:
    def test_every_source_names_a_declared_set(self):
        registry = stages_mod.default_registry()
        for name in registry.sources:
            declared = registry.body_model_set_for(name)
            assert declared is not None, name
            assert declared.relative_path == "body_models/smpl"
            assert declared.resolver.startswith("smpl18.")

    def test_the_set_directory_hangs_off_the_data_root(self, tmp_path):
        declared = stages_mod.default_registry().body_model_set_for("amass")
        assert declared.directory(tmp_path) == tmp_path / "body_models" / "smpl"
        assert declared.directory(None) is None

    def test_an_undeclared_set_is_an_error_not_a_silent_none(self, tmp_path):
        registry = stages_mod.default_registry()
        source = registry.for_source("amass")
        broken = stages_mod.PipelineRegistry(
            spec_id=registry.spec_id, spec_version=registry.spec_version,
            canonical_stages=registry.canonical_stages,
            resume_predicates=registry.resume_predicates,
            sources={"amass": source}, body_model_sets={}, path=registry.path,
        )
        with pytest.raises(stages_mod.PipelineError, match="body_model_set"):
            broken.body_model_set_for("amass")

    def test_every_source_names_its_smpl18_profile(self):
        registry = stages_mod.default_registry()
        for name in registry.sources:
            assert registry.for_source(name).smpl18_profile, name


class TestTheSetIsHashedThroughSmpl24:
    def test_hashes_every_model_and_digests_the_set(self, model_set):
        root, expected = model_set
        assert body_models.set_hashes(root) == expected
        digest = body_models.set_digest(expected)
        lines = "\n".join(f"{n} {expected[n]}" for n in sorted(expected))
        assert digest == hashlib.sha256(lines.encode("utf-8")).hexdigest()

    def test_a_changed_model_changes_the_digest(self, model_set):
        root, expected = model_set
        before = body_models.set_digest(body_models.set_hashes(root))
        (root / smpl18_select.model_filename("neutral")).write_bytes(b"changed")
        assert body_models.set_digest(body_models.set_hashes(root)) != before

    def test_a_missing_set_is_none_rather_than_an_empty_record(self, tmp_path):
        assert body_models.set_hashes(None) is None
        assert body_models.set_hashes(tmp_path / "absent") is None
        assert body_models.set_digest(None) is None

    def test_the_file_for_a_gender_is_the_one_the_generators_read(self, model_set):
        root, _ = model_set
        from soma_synth.addbio_retarget import shape_fit

        for gender in ("male", "female", "neutral"):
            assert body_models.path_for_gender(gender, root) == root / smpl18_select.model_filename(gender)
            assert shape_fit.clean_model_path_for_gender(gender, root=root.parents[1]) == root / smpl18_select.model_filename(gender)


class TestTheRunRecordCarriesTheSet:
    def test_the_record_names_the_set_its_selection_and_every_hash(self, model_set):
        root, expected = model_set
        declared = stages_mod.default_registry().body_model_set_for("addbiomechanics")
        record = provenance_mod.ProvenanceRecord(
            source_name="addbiomechanics", spec_id="s", spec_version="v",
            body_model_set=declared, body_model_dir=root,
            body_model_selection="by_subject_gender",
        ).as_json()["body_model"]
        assert record["set"] == "smpl_clean"
        assert record["selection"] == "by_subject_gender"
        assert record["resolver"] == declared.resolver
        assert record["models"] == expected
        assert record["set_digest"] == body_models.set_digest(expected)
        json.dumps(record)      # a run record is written as JSON

    def test_without_a_directory_the_record_says_why_rather_than_claiming_a_hash(self):
        declared = stages_mod.default_registry().body_model_set_for("amass")
        record = provenance_mod.ProvenanceRecord(
            source_name="amass", spec_id="s", spec_version="v", body_model_set=declared,
            body_model_dir=None, body_model_selection="by_subject_gender",
        ).as_json()["body_model"]
        assert record["status"] == provenance_mod.UNAVAILABLE
        assert "data root" in record["reason"]

    def test_a_run_without_a_set_still_records_a_single_path(self, tmp_path):
        model = tmp_path / "SMPL_MALE_clean.npz"
        model.write_bytes(b"x")
        record = provenance_mod.ProvenanceRecord(
            source_name="amass", spec_id="s", spec_version="v", body_model_path=model,
        ).as_json()["body_model"]
        assert record == {"path": model.name, "sha256": provenance_mod.file_sha256(model)}
