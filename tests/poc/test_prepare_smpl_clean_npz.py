from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[2] / "scripts" / "poc" / "prepare_smpl_clean_npz.py"


def _load():
    spec = importlib.util.spec_from_file_location("prepare_smpl_clean_npz", MODULE_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_stub_unpickler_captures_chumpy_and_rejects_unknown_global():
    mod = _load()
    stub = mod._StubUnpickler.__new__(mod._StubUnpickler)
    assert mod._StubUnpickler.find_class(stub, "chumpy.ch", "Ch") is mod._Captured
    assert mod._StubUnpickler.find_class(stub, "numpy.core.multiarray", "_reconstruct") is not None
    with pytest.raises(pickle.UnpicklingError):
        mod._StubUnpickler.find_class(stub, "os", "system")
    with pytest.raises(pickle.UnpicklingError):
        mod._StubUnpickler.find_class(stub, "builtins", "eval")


def test_captured_to_array_prefers_x_then_r():
    mod = _load()
    cap = mod._Captured()
    cap.__setstate__({"x": np.arange(6).reshape(2, 3).astype(np.float64)})
    np.testing.assert_array_equal(cap.to_array(), np.arange(6).reshape(2, 3))


def test_convert_writes_required_fields(tmp_path):
    mod = _load()
    fake = {
        "v_template": np.zeros((5, 3), np.float64),
        "shapedirs": np.zeros((5, 3, 10), np.float64),
        "posedirs": np.zeros((5, 3, 207), np.float64),
        "J_regressor": np.zeros((24, 5), np.float64),
        "weights": np.zeros((5, 24), np.float64),
        "f": np.zeros((3, 3), np.int32),
        "kintree_table": np.zeros((2, 24), np.int64),
    }
    src = tmp_path / "fake_model.pkl"
    with open(src, "wb") as fh:
        pickle.dump(fake, fh, protocol=2)
    out = tmp_path / "FAKE_clean.npz"
    mod.convert(str(src), str(out))
    d = np.load(out, allow_pickle=False)
    for key in ("v_template", "shapedirs", "J_regressor", "kintree_parents"):
        assert key in d.files
    assert d["kintree_parents"].shape == (24,)


def test_convert_truncates_shapedirs_to_num_betas(tmp_path):
    mod = _load()
    fake = {
        "v_template": np.zeros((5, 3), np.float64),
        "shapedirs": np.arange(5 * 3 * 300, dtype=np.float64).reshape(5, 3, 300),
        "J_regressor": np.zeros((24, 5), np.float64),
        "kintree_table": np.zeros((2, 24), np.int64),
    }
    src = tmp_path / "big.pkl"
    with open(src, "wb") as fh:
        pickle.dump(fake, fh, protocol=2)
    out = tmp_path / "BIG_clean.npz"
    mod.convert(str(src), str(out), num_betas=10)
    d = np.load(out, allow_pickle=False)
    assert d["shapedirs"].shape == (5, 3, 10)
    np.testing.assert_array_equal(d["shapedirs"], fake["shapedirs"][:, :, :10])
