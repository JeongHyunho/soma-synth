from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

MODULE = Path(__file__).parents[2] / "scripts" / "poc" / "generate_prism_faithful_all.py"


def _load():
    spec = importlib.util.spec_from_file_location("gen_all", MODULE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_enumerate_specs_covers_declared_counts(tmp_path):
    m = _load()
    root = tmp_path / "extracted" / "prism"
    for subj, takes in (("subj001", ("take002", "take003")), ("subj002", ("take001",))):
        d = root / subj
        d.mkdir(parents=True)
        for t in takes:
            (d / (t + ".pkl")).write_bytes(b"x")
    specs = m.enumerate_specs(str(tmp_path),
                              gender_by_subject={"prism_subj001": "M", "prism_subj002": "F"})
    ids = {(s.subject_id, s.take_id) for s in specs}
    assert ids == {("prism_subj001", "take002"), ("prism_subj001", "take003"),
                   ("prism_subj002", "take001")}
    assert all(s.window is None for s in specs)
    # female subject carries its gender through
    assert {s.subject_id: s.gender for s in specs}["prism_subj002"] == "F"


def _fake_build(spec, reduction=None):
    if spec.take_id == "bad":
        raise ValueError("boom")
    small = {"frame_count": np.int64(3), "pair_id": "p"}
    return small, {"a": np.zeros(1)}, {"b": np.zeros(1)}, {"c": np.zeros(1)}, {"identity": {"pair_id": "p"}}


def test_run_isolates_failures_and_writes_index(tmp_path):
    m = _load()
    specs = [m.TakeSpec("prism_subj001", "good", "x", "M", None),
             m.TakeSpec("prism_subj001", "bad", "x", "M", None)]
    out_root = tmp_path / "ds"
    index = m.run(specs, str(out_root), build=_fake_build)
    assert index["counts"]["ok"] == 1
    assert index["counts"]["failed"] == 1
    saved = json.loads((out_root / "INDEX.json").read_text(encoding="utf-8"))
    assert saved["counts"]["ok"] == 1
    # resume: second run skips the already-ok take
    index2 = m.run(specs, str(out_root), build=_fake_build)
    assert index2["counts"]["skipped"] >= 1


def test_writes_shared_dataset_description(tmp_path):
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
