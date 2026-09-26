from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

MODULE = Path(__file__).parents[2] / "scripts" / "poc" / "generate_prism_measured_all.py"


def _load():
    spec = importlib.util.spec_from_file_location("gen_measured_all", MODULE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fake_build(spec, reduction=None):
    if spec.take_id == "bad":
        raise ValueError("boom")
    small = {"frame_count": np.int64(3), "pair_id": "p"}
    return small, {"a": np.zeros(1)}, {"b": np.zeros(1)}, {"c": np.zeros(1)}, {"identity": {"pair_id": "p"}}


def test_dataset_name_and_reuse_of_faithful_engine():
    m = _load()
    # this builder targets the deployed poc-demo dataset directly (measured small); no separate dir
    assert m.DATASET_DIRNAME == "prism_faithful_full"    # no version suffix (retention rule 1.1)
    # spec enumeration + per-subject anthro fit are reused from the faithful batch engine
    assert m.enumerate_specs is m.faithful_all.enumerate_specs
    assert m.compute_subject_constants is m.faithful_all.compute_subject_constants
    assert callable(m.measured.build_measured)


def test_run_isolates_failures_and_labels_measured(tmp_path):
    m = _load()
    specs = [m.TakeSpec("prism_subj001", "good", "x", "M", None),
             m.TakeSpec("prism_subj001", "bad", "x", "M", None)]
    out_root = tmp_path / "ds"
    index = m.run(specs, str(out_root), build=_fake_build)
    assert index["counts"]["ok"] == 1 and index["counts"]["failed"] == 1
    assert index["dataset_dirname"] == "prism_faithful_full"
    assert index["spec_id"] == "qmd_unified8_smpl18" and index["spec_version"] == "faithful-v2"
    assert "physical_imu_measured_simulation" in index["small_source"]
    saved = json.loads((out_root / "INDEX.json").read_text(encoding="utf-8"))
    assert saved["counts"]["ok"] == 1
    # `rel` is the registry's key; `relative_path` stays for the consumers that branched on it
    good = next(e for e in saved["takes"] if e["status"] == "ok")
    assert good["rel"] == good["relative_path"] == good["take_id"]
    # the bundle-level promotion policy is written from the takes; fake takes carry no block
    policy = json.loads((out_root / m.POLICY_FILE).read_text(encoding="utf-8"))
    assert policy["config_id"] == "prism_faithful_reduced_model_policy_v1"
    assert policy["takes_total"] == 0 and policy["flagged_pairs"] == []
    assert policy["fixed_joint_downstream"]["left_collar"] == ["left_shoulder", "left_elbow", "left_wrist"]
    # resume: a second run skips the already-ok take
    index2 = m.run(specs, str(out_root), build=_fake_build)
    assert any(t["status"] == "skipped" for t in index2["takes"])


def test_the_policy_file_counts_what_the_takes_say(tmp_path):
    """One flagged joint in one of two takes: the counts and pairs follow the manifests."""
    m = _load()
    root = tmp_path / "ds"
    blocks = {
        "prism-subj001-take002": {"left_collar": 0.0672, "right_collar": 0.0301,
                                  "spine1": 0.02, "spine2": 0.02},
        "prism-subj002-take003": {"left_collar": 0.01, "right_collar": 0.01,
                                  "spine1": 0.01, "spine2": 0.01},
    }
    takes = []
    for take, p95s in blocks.items():
        (root / take).mkdir(parents=True)
        promotion = {j: {"p95_m": v, "max_m": v * 2, "promotion_candidate": v > 0.05}
                     for j, v in p95s.items()}
        manifest = {"validation": {"reduced_model_fit_residual_m": {
            "promotion_threshold_p95_m": 0.05, "fixed_joint_promotion": promotion}}}
        (root / take / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        takes.append({"take_id": take, "rel": take, "status": "ok"})
    m.write_reduced_model_policy(root, takes)
    policy = json.loads((root / m.POLICY_FILE).read_text(encoding="utf-8"))
    assert policy["takes_total"] == 2
    assert policy["takes_with_any_candidate"] == 1
    assert policy["flagged_take_count_per_joint"] == {"left_collar": 1}
    assert policy["flagged_take_count_per_subject"] == {"prism-subj001": 1}
    assert policy["flagged_pairs"] == [{"take": "prism-subj001-take002", "joint": "left_collar",
                                        "p95_m": 0.0672}]
    assert policy["promotion_threshold_p95_m"] == 0.05


def test_description_states_measured_semantics_standalone(tmp_path):
    m = _load()
    specs = [m.TakeSpec("prism_subj001", "take002", "x", "M", None)]
    out_root = tmp_path / "ds"
    m.run(specs, str(out_root), build=_fake_build)
    desc = (out_root / "DATA_DESCRIPTION_EN.md").read_text(encoding="utf-8")
    # ADR-0040 D3 retired DATASET_DESCRIPTION_EN.md; one English description per bundle.
    assert not (out_root / "DATASET_DESCRIPTION_EN.md").exists()
    assert "INTERNAL-ONLY" in desc
    assert "experimental_non_candidate" in desc
    assert "anthro_reference.npz" in desc
    assert "gravity-included" in desc
    assert "spec_S_v2" in desc
    # standalone description — no cross-dataset comparison language
    assert "differs" not in desc.lower() and "byte-identical" not in desc.lower()
