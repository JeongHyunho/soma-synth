"""The heading synthesis must add a channel without taking one away, remove only the drift, and
refuse to guess.

The failure that matters is not a crash. It is a companion file that a reader mistakes for
measurement, a measured array that quietly stops being what the insole reported, or a
'correction' that flattens the ankle's real rotation along with the drift -- which is exactly
what v1 did and why v2 replaced it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "poc" / "synthesize_insole_heading.py"


def _load():
    spec = importlib.util.spec_from_file_location("synthesize_insole_heading", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sh = _load()
DT = 0.01


def _yaw_series(deg: np.ndarray) -> np.ndarray:
    return sh.rot_z(np.radians(np.asarray(deg, float)))


def _quat_series(n, axis, angle_deg):
    ang = np.radians(np.linspace(0, angle_deg, n))
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    q = np.zeros((n, 4))
    q[:, 0] = np.cos(ang / 2)
    q[:, 1:] = np.sin(ang / 2)[:, None] * axis
    return q


def _gait_contact(n: int, stride: int = 100, stance: int = 60) -> np.ndarray:
    """A periodic stance/swing pattern: contact for `stance` of every `stride` frames."""
    return (np.arange(n) % stride) < stance


def make_take(tmp_path: Path, *, foot_l_R=None, foot_r_R=None, n=1200,
              absolute=(True,) * 6 + (False, False), contact=None) -> Path:
    """A take: identity everywhere except what the caller sets. Optional contact mask."""
    take = tmp_path / "prism-subjX-take001"
    take.mkdir(parents=True, exist_ok=True)
    for given in (foot_l_R, foot_r_R):
        if given is not None:
            n = len(given)
    ori = np.zeros((n, 8, 4))
    ori[:, :, 0] = 1.0
    if foot_l_R is not None:
        ori[:, sh.SITE_ORDER.index("foot_l")] = sh.R_to_quat_wxyz(foot_l_R)
    if foot_r_R is not None:
        ori[:, sh.SITE_ORDER.index("foot_r")] = sh.R_to_quat_wxyz(foot_r_R)
    np.savez_compressed(
        take / "small_reference.npz",
        sensor_codes=np.array(sh.SITE_ORDER, dtype="<U16"),
        imu_orientation=ori.astype(np.float32),
        imu_orientation_absolute_heading=np.array(absolute, bool),
        timestamps_s=np.arange(n) * DT,
        pair_id=np.array("pair-abc"),
    )
    if contact is not None:
        mask = np.stack([contact, contact], axis=1)
        np.savez_compressed(take / sh.CONTACT_FILE, **{sh.CONTACT_KEY: mask})
    return take


def _drifting_foot(n, rate_deg_s, ankle_deg=0.0, stride=100):
    """Heading = linear drift + (optionally) a periodic ankle yaw, one cycle per stride."""
    t = np.arange(n) * DT
    yaw = rate_deg_s * t + ankle_deg * np.sin(2 * np.pi * np.arange(n) / stride)
    return _yaw_series(yaw)


def _heading_deg(R):
    return np.degrees(np.unwrap(sh.heading_of(R)))


# ------------------------------------------------------------------- the measured record
def test_the_measured_file_is_not_touched(tmp_path: Path):
    """Measured data is never overwritten, and the measured stream is the only record of what the
    insole actually reported."""
    take = make_take(tmp_path, foot_l_R=_drifting_foot(1200, 45.0))
    before = (take / "small_reference.npz").read_bytes()
    sh.synthesize_take(take)
    assert (take / "small_reference.npz").read_bytes() == before


def test_the_output_is_a_companion_file_not_a_replacement(tmp_path: Path):
    take = make_take(tmp_path, foot_l_R=_drifting_foot(1200, 45.0))
    sh.synthesize_take(take)
    assert (take / sh.OUTPUT_NAME).is_file()
    assert (take / "small_reference.npz").is_file()


def test_dry_run_writes_nothing(tmp_path: Path):
    take = make_take(tmp_path, foot_l_R=_drifting_foot(1200, 45.0))
    sh.synthesize_take(take, dry_run=True)
    assert not (take / sh.OUTPUT_NAME).exists()


# -------------------------------------------------------------------------- provenance
def test_every_channel_says_where_its_heading_came_from(tmp_path: Path):
    take = make_take(tmp_path, foot_l_R=_drifting_foot(1200, 45.0))
    sh.synthesize_take(take)
    with np.load(take / sh.OUTPUT_NAME, allow_pickle=False) as z:
        source = [str(s) for s in z["imu_orientation_heading_source"]]
        ns = str(z["namespace"])
    assert source[sh.SITE_ORDER.index("foot_l")] == sh.SYNTHESIZED
    assert source[sh.SITE_ORDER.index("foot_r")] == sh.SYNTHESIZED
    assert all(source[i] == sh.MEASURED for i in range(6))
    assert sh.SYNTHESIZED.startswith("synthesized:")   # a corrected heading is not a measurement
    assert ns == sh.NAMESPACE and "v2" in ns


def test_the_companion_file_records_the_drift_it_removed(tmp_path: Path):
    """A correction nobody can quantify afterwards is indistinguishable from a silent edit."""
    take = make_take(tmp_path, foot_l_R=_drifting_foot(1200, 45.0))
    sh.synthesize_take(take)
    with np.load(take / sh.OUTPUT_NAME, allow_pickle=False) as z:
        shift, rate = z["heading_shift_deg"], z["heading_drift_rate_deg_s"]
        strides = z["heading_stride_count"]
    assert shift.shape == (1200, 2)
    assert np.ptp(shift[:, 0]) > 500.0            # 45 deg/s over 12 s, undone
    assert np.allclose(shift[:, 1], 0.0)          # the right foot never drifted
    assert abs(rate[0] - 45.0) < 0.5 and abs(rate[1]) < 1e-6
    assert shift[0, 0] == 0.0                     # anchored at the first frame
    assert strides[0] == 0                        # no contact mask: no strides, least-squares line


# ---------------------------------------------------------------------------- the maths
def test_the_drift_is_removed(tmp_path: Path):
    take = make_take(tmp_path, foot_l_R=_drifting_foot(1200, 45.0))
    sh.synthesize_take(take)
    with np.load(take / sh.OUTPUT_NAME, allow_pickle=False) as z:
        R = sh.quats_to_R(np.asarray(z["imu_orientation"], np.float64))
    assert np.ptp(_heading_deg(R[:, sh.SITE_ORDER.index("foot_l")])) < 1.0


def test_the_ankles_real_rotation_survives_the_correction(tmp_path: Path):
    """The property v2 exists for. A foot that drifts AND swings its heading +-20 deg once per
    stride: the drift goes, the swing stays. v1 would have flattened both."""
    n = 1200
    foot = _drifting_foot(n, 30.0, ankle_deg=20.0, stride=100)
    take = make_take(tmp_path, foot_l_R=foot, contact=_gait_contact(n, stride=100, stance=60))
    sh.synthesize_take(take)
    with np.load(take / sh.OUTPUT_NAME, allow_pickle=False) as z:
        R = sh.quats_to_R(np.asarray(z["imu_orientation"], np.float64))
        rate, strides = z["heading_drift_rate_deg_s"], z["heading_stride_count"]
    heading = _heading_deg(R[:, sh.SITE_ORDER.index("foot_l")])
    assert abs(rate[0] - 30.0) < 1.0                       # the drift rate was found
    assert strides[0] >= 10                                # from the contact signal
    assert 36.0 < np.ptp(heading) < 44.0                   # +-20 deg swing kept, no drift left
    # and the residual is periodic, not a leftover ramp: first and last stride agree
    assert abs(heading[:100].mean() - heading[-100:].mean()) < 2.0

    # Contrast: the v1 transfer against a still shank flattens the swing as well.
    R_in = sh.quats_to_R(np.load(take / "small_reference.npz")["imu_orientation"].astype(np.float64))
    v1, _ = sh.transfer_heading(R_in[:, sh.SITE_ORDER.index("foot_l")],
                                R_in[:, sh.SITE_ORDER.index("shank_l")])
    assert np.ptp(_heading_deg(v1)) < 1.0


def test_tilt_survives_the_heading_correction(tmp_path: Path):
    """Only the rotation about the world vertical may change: gravity anchors the tilt, and the
    tilt is the part of the insole's attitude that was never in doubt."""
    n = 600
    pitch = sh.quats_to_R(_quat_series(n, (0, 1, 0), 40.0))
    foot = np.einsum("tij,tjk->tik", _drifting_foot(n, 60.0), pitch)
    take = make_take(tmp_path, foot_l_R=foot)
    sh.synthesize_take(take)
    with np.load(take / sh.OUTPUT_NAME, allow_pickle=False) as z:
        R_out = sh.quats_to_R(np.asarray(z["imu_orientation"], np.float64))[
            :, sh.SITE_ORDER.index("foot_l")]
    z_axis = np.array([0.0, 0.0, 1.0])
    tilt_in = np.einsum("tji,j->ti", foot, z_axis)
    tilt_out = np.einsum("tji,j->ti", R_out, z_axis)
    assert np.allclose(tilt_in, tilt_out, atol=1e-5)


def test_the_nine_dof_channels_are_copied_through_unchanged(tmp_path: Path):
    n = 300
    shank = sh.quats_to_R(_quat_series(n, (1, 0, 0), 25.0))
    take = make_take(tmp_path, foot_l_R=_drifting_foot(n, 10.0))
    # give the shank some motion so 'unchanged' means something
    with np.load(take / "small_reference.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["imu_orientation"][:, sh.SITE_ORDER.index("shank_l")] = sh.R_to_quat_wxyz(shank)
    np.savez_compressed(take / "small_reference.npz", **data)
    sh.synthesize_take(take)
    with np.load(take / sh.OUTPUT_NAME, allow_pickle=False) as z:
        out = np.asarray(z["imu_orientation"])
    assert np.array_equal(out[:, :6], data["imu_orientation"][:, :6])


def test_the_trend_is_robust_to_one_wild_stride(tmp_path: Path):
    """Theil-Sen, not least squares, through the stride medians: one stride where the foot pivots
    hard must not bend the fitted drift."""
    n = 2000
    yaw = 5.0 * np.arange(n) * DT
    yaw[900:1000] += 90.0                                  # one stride pivots a quarter turn
    take = make_take(tmp_path, foot_l_R=_yaw_series(yaw), contact=_gait_contact(n))
    sh.synthesize_take(take)
    with np.load(take / sh.OUTPUT_NAME, allow_pickle=False) as z:
        rate = float(z["heading_drift_rate_deg_s"][0])
    assert abs(rate - 5.0) < 0.5


def test_quaternion_round_trip_survives_a_half_turn():
    """The naive w-first conversion loses precision near 180 degrees, which is exactly where the
    drifting takes live."""
    R = sh.quats_to_R(_quat_series(50, (0, 0, 1), 179.5))
    back = sh.quats_to_R(sh.R_to_quat_wxyz(R))
    assert np.allclose(R, back, atol=1e-6)


# ------------------------------------------------------------------------------ refusal
def test_a_reference_without_absolute_heading_is_refused(tmp_path: Path):
    """Measuring the trend against an unreferenced heading would move the problem, not fix it."""
    take = make_take(tmp_path, foot_l_R=_drifting_foot(300, 10.0),
                     absolute=(True, True, True, False, False, True, False, False))
    with pytest.raises(sh.SynthesisError, match="reference"):
        sh.synthesize_take(take)


def test_a_take_whose_channels_already_have_heading_is_refused(tmp_path: Path):
    take = make_take(tmp_path, absolute=(True,) * 8)
    with pytest.raises(sh.SynthesisError, match="nothing to do"):
        sh.synthesize_take(take)


def test_an_unexpected_sensor_order_is_refused(tmp_path: Path):
    """The canonical order is fixed; a take that disagrees is not this script's."""
    take = make_take(tmp_path)
    with np.load(take / "small_reference.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["sensor_codes"] = np.array(list(reversed(sh.SITE_ORDER)), dtype="<U16")
    np.savez_compressed(take / "small_reference.npz", **data)
    with pytest.raises(sh.SynthesisError, match="unexpected sensor order"):
        sh.synthesize_take(take)


def test_a_take_without_timestamps_is_refused(tmp_path: Path):
    """The drift is a rate; without a timebase there is nothing to fit it against."""
    take = make_take(tmp_path, foot_l_R=_drifting_foot(300, 10.0))
    with np.load(take / "small_reference.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files if k != "timestamps_s"}
    np.savez_compressed(take / "small_reference.npz", **data)
    with pytest.raises(sh.SynthesisError, match="timestamps"):
        sh.synthesize_take(take)


def test_a_malformed_contact_mask_is_refused_rather_than_ignored(tmp_path: Path):
    take = make_take(tmp_path, foot_l_R=_drifting_foot(300, 10.0))
    np.savez_compressed(take / sh.CONTACT_FILE, **{sh.CONTACT_KEY: np.ones((120, 2), bool)})
    with pytest.raises(sh.SynthesisError, match=sh.CONTACT_KEY):
        sh.synthesize_take(take)
