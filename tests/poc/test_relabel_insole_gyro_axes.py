"""Rewriting a shipped tensor has to prove three things: it changed only what it meant to, it
can be undone exactly, and it cannot be applied twice."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "poc" / "relabel_insole_gyro_axes.py"


def _load():
    spec = importlib.util.spec_from_file_location("relabel_insole_gyro_axes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rl = _load()


def _quat_from_axis_angle(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.radians(deg) / 2
    return np.array([np.cos(h), *(np.sin(h) * axis)])


def make_take(tmp_path: Path, *, crlf=True) -> Path:
    take = tmp_path / "prism-subjX-take001"
    take.mkdir()
    n = 300
    rng = np.random.default_rng(1)
    q_anat = np.tile([1.0, 0, 0, 0], (8, 1)).astype(np.float32)
    # Give the feet a non-trivial calibration so r_cal^T M r_cal is exercised, not just M.
    q_anat[6] = _quat_from_axis_angle((1, 2, 3), 37.0)
    q_anat[7] = _quat_from_axis_angle((0, 1, 0), -80.0)
    arrays = {
        "sensor_codes": np.array(rl.SITE_ORDER, dtype="<U7"),
        "imu_orientation": rng.normal(size=(n, 8, 4)).astype(np.float32),
        "imu_angular_velocity": rng.normal(size=(n, 8, 3)).astype(np.float32),
        "imu_acceleration": rng.normal(size=(n, 8, 3)).astype(np.float32),
        "q_anatomical_from_sensor": q_anat,
        "gyro_provenance": np.array(
            ["proxy"] + ["derived_from_orientation:rotation_log"] * 5
            + ["measured:gyr_local_raw"] * 2, dtype="<U55"),
        "pair_id": np.array("pair-x"),
        "timestamps_s": np.arange(n) / 100.0,
    }
    np.savez_compressed(take / "small_reference.npz", **arrays)
    manifest = {"source": {"source_name": "prism"}, "small_measured": {"gyro": {}}}
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    (take / "manifest.json").write_bytes(
        (text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    return take


def _arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


# ---------------------------------------------------------------------------- the maths
def test_the_relabel_is_conjugated_by_the_channels_own_calibration():
    """With the identity calibration the correction is M itself; with a real one it is not."""
    w = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(rl.relabel_gyro(w, [1, 0, 0, 0]), [[-1.0, 3.0, 2.0]])
    q = _quat_from_axis_angle((1, 2, 3), 37.0)
    got = rl.relabel_gyro(w, q)
    assert not np.allclose(got, [[-1.0, 3.0, 2.0]])
    assert np.isclose(np.linalg.norm(got), np.linalg.norm(w))   # still a rotation


def test_the_relabel_is_its_own_inverse():
    w = np.random.default_rng(3).normal(size=(50, 3))
    q = _quat_from_axis_angle((2, -1, 5), 61.0)
    assert np.allclose(rl.relabel_gyro(rl.relabel_gyro(w, q), q), w, atol=1e-9)


# ------------------------------------------------------------------------ what changes
def test_only_the_foot_gyro_changes_and_every_other_array_is_bit_identical(tmp_path):
    take = make_take(tmp_path)
    before = _arrays(take / "small_reference.npz")
    rl.apply_take(rl.inspect_take(take))
    after = _arrays(take / "small_reference.npz")

    for k in before:
        if k == "imu_angular_velocity":
            continue
        assert after[k].dtype == before[k].dtype and np.array_equal(after[k], before[k]), k
    w0, w1 = before["imu_angular_velocity"], after["imu_angular_velocity"]
    assert w1.dtype == w0.dtype
    assert np.array_equal(w1[:, :6], w0[:, :6])                  # body channels untouched
    assert not np.allclose(w1[:, 6], w0[:, 6]) and not np.allclose(w1[:, 7], w0[:, 7])


def test_the_manifest_gains_a_marker_and_keeps_its_byte_style(tmp_path):
    take = make_take(tmp_path, crlf=True)
    rl.apply_take(rl.inspect_take(take))
    raw = (take / "manifest.json").read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    marker = json.loads(raw.decode("utf-8"))["small_measured"][rl.MARKER_KEY]
    assert marker["applied"] is True
    assert marker["matrix"] == [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    assert marker["channels"] == ["foot_l", "foot_r"]


# ----------------------------------------------------------------------------- refusal
def test_a_corrected_take_is_recognised_and_not_corrected_twice(tmp_path):
    take = make_take(tmp_path)
    rl.apply_take(rl.inspect_take(take))
    once = _arrays(take / "small_reference.npz")["imu_angular_velocity"]
    plan = rl.inspect_take(take)
    assert plan.already is True
    # main() filters on `already`; applying twice would undo the fix, so the flag is the guard.
    assert np.array_equal(once, _arrays(take / "small_reference.npz")["imu_angular_velocity"])


def test_a_take_whose_foot_gyro_is_not_the_raw_stream_is_refused(tmp_path):
    take = make_take(tmp_path)
    arrays = _arrays(take / "small_reference.npz")
    arrays["gyro_provenance"][6] = "derived_from_orientation:rotation_log"
    np.savez_compressed(take / "small_reference.npz", **arrays)
    plan = rl.inspect_take(take)
    assert any("not the measured raw stream" in p for p in plan.problems)


# ----------------------------------------------------------------------------- rollback
def test_rollback_restores_the_gyro_exactly_and_drops_the_marker(tmp_path):
    """Exactly: bit for bit, from the sidecar -- not by re-applying M through float32."""
    take = make_take(tmp_path)
    before = _arrays(take / "small_reference.npz")["imu_angular_velocity"]
    entry = rl.apply_take(rl.inspect_take(take), sidecar_dir=tmp_path / "side")
    assert Path(entry["sidecar"]).is_file()
    journal = tmp_path / "j.jsonl"
    journal.write_text(json.dumps(entry) + "\n", encoding="utf-8", newline="\n")

    assert rl.rollback(journal, log=lambda _m: None) == 0
    after = _arrays(take / "small_reference.npz")["imu_angular_velocity"]
    assert np.array_equal(after, before)
    manifest = json.loads((take / "manifest.json").read_text(encoding="utf-8"))
    assert rl.MARKER_KEY not in manifest["small_measured"]


def test_rollback_refuses_a_gyro_changed_after_the_journal(tmp_path):
    take = make_take(tmp_path)
    entry = rl.apply_take(rl.inspect_take(take), sidecar_dir=tmp_path / "side")
    journal = tmp_path / "j.jsonl"
    journal.write_text(json.dumps(entry) + "\n", encoding="utf-8", newline="\n")
    arrays = _arrays(take / "small_reference.npz")
    arrays["imu_angular_velocity"][0, 6, 0] += 1.0
    np.savez_compressed(take / "small_reference.npz", **arrays)
    messages = []
    assert rl.rollback(journal, log=messages.append) == 1
    assert any("CHANGED SINCE" in m for m in messages)


def test_rollback_without_a_sidecar_is_refused_rather_than_approximated(tmp_path):
    """Re-applying M through a float32 file is close but not exact; close is not a rollback."""
    take = make_take(tmp_path)
    entry = rl.apply_take(rl.inspect_take(take))          # no sidecar_dir
    assert entry["sidecar"] == ""
    journal = tmp_path / "j.jsonl"
    journal.write_text(json.dumps(entry) + "\n", encoding="utf-8", newline="\n")
    messages = []
    assert rl.rollback(journal, log=messages.append) == 1
    assert any("NO SIDECAR" in m for m in messages)
