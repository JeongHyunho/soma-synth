from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from data_root_skips import DATA_ROOT, needs_data

SCRIPT = Path(__file__).parents[2] / "scripts" / "poc" / "generate_prism_faithful.py"
NEEDS_DATA = needs_data((DATA_ROOT / "extracted/prism/subj001/take002.pkl").exists(),
                        "the PRISM take extracted/prism/subj001/take002.pkl")


def _gen():
    spec = importlib.util.spec_from_file_location("gen", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _subj001_spec(g, window):
    return g.TakeSpec(
        subject_id="prism_subj001", take_id="take002",
        source_pkl=str(DATA_ROOT / "extracted/prism/subj001/take002.pkl"),
        gender="M", window=window, relative_path="extracted/prism/subj001/take002.pkl",
    )


def _arrays_equal(npz_path: Path, arrays: dict) -> bool:
    b = np.load(npz_path, allow_pickle=True)
    if set(b.files) != set(arrays):
        return False
    for k in b.files:
        x, y = b[k], np.asarray(arrays[k])
        if x.dtype == object or x.dtype.kind in "US":
            if x.tolist() != y.tolist():
                return False
        elif x.shape != y.shape or not np.array_equal(x, y):
            return False
    return True


def test_take_spec_and_build_signature():
    g = _gen()
    spec = _subj001_spec(g, (1000, 2000))
    assert spec.window == (1000, 2000)
    assert callable(g.build)


@NEEDS_DATA
def test_full_window_covers_all_frames():
    g = _gen()
    pkl = str(DATA_ROOT / "extracted/prism/subj001/take002.pkl")
    raw = g.load_prism_safe(pkl)
    n = int(raw["smpl_params"]["poses"].shape[0])
    spec = g.TakeSpec(subject_id="prism_subj001", take_id="take002", source_pkl=pkl,
                      gender="M", window=None)
    small, large, dev, anthro, manifest = g.build(spec)
    assert int(small["frame_count"]) == n
    assert small["imu_orientation"].shape == (n, 8, 4)
    assert large["joint_rotation"].shape == (n, 18, 4)
    assert manifest["window"]["frame_count"] == n
    assert manifest["window"]["frame_start"] == 0


@NEEDS_DATA
def test_policy_alignment_fields():
    g = _gen()
    spec = _subj001_spec(g, (1000, 2000))
    small, large, dev, anthro, manifest = g.build(spec)
    # insole contacts normalized to bool
    assert dev["foot_contact_mask"].dtype == np.bool_
    # anthro units per SOMA spec (m/kg) + source-unit provenance recorded
    assert 1.0 < float(anthro["height"]) < 2.5           # metres
    assert 30.0 < float(anthro["body_mass"]) < 200.0     # kg
    assert "arm_length" not in anthro                    # forbidden field excluded
    assert "undeclared" in str(anthro["height_provenance"]).lower()
    # manifest references the usage policy + trans-only basis
    pol = manifest["prism_usage_policy"]
    assert pol["config_id"] == "prism_usage_policy_v1"
    assert "trans" in json.dumps(manifest).lower()


@NEEDS_DATA
def test_promotion_candidate_flagging():
    g = _gen()
    small, large, dev, anthro, manifest = g.build(_subj001_spec(g, (1000, 2000)))
    r = manifest["validation"]["reduced_model_fit_residual_m"]
    assert r["promotion_threshold_p95_m"] == 0.05
    fp = r["fixed_joint_promotion"]
    assert set(fp) == {"left_collar", "right_collar", "spine1", "spine2"}
    for d in fp.values():
        assert "p95_m" in d and "max_m" in d and isinstance(d["promotion_candidate"], bool)
    assert isinstance(r["promotion_candidates"], list)
