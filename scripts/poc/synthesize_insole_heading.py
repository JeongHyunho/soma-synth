"""Give PRISM's foot channels a usable heading, without taking away what the insole measured.

The two foot channels are a Moticon OpenGo insole, six-axis. Six axes means an accelerometer and
a gyroscope and no magnetometer, so gravity anchors the tilt and nothing anchors the heading:
it is the gyro integral and it wanders. The bundle has always said so -- every take's
``imu_orientation_absolute_heading`` reads ``[T,T,T,T,T,T,F,F]`` -- and in twelve takes the
wander reaches between one and eight full turns.

**The method (v2, 2026-09-14): remove the drift, keep the ankle.** The foot's heading relative
to the same leg's 9-DOF shank, taken as the world-Z twist of ``R_foot R_shank^T``, is

    d(t) = drift(t) + ankle_yaw(t) + const

where ankle yaw is bounded and roughly periodic over strides, and the drift -- the integral of the
insole gyro's bias about the vertical -- is close to linear in time. So the per-stride medians of
``d`` sample the drift line plus bounded noise, a robust line through them is the drift, and
subtracting that line from the insole's own heading removes the wander while leaving every real
turn of the ankle in place. The line is removed relative to the first frame: a gyro-integrated
heading is exact at the instant it was initialised, so the correction must vanish there.

Nothing in it is a free constant. Stride boundaries are the rising edges of the insole's own
contact signal; the trend is a Theil-Sen slope through per-stride medians; with fewer than two
strides it falls back to a least-squares line over all frames. That was the point of scoring it:
the parent project's PRISM insole-heading method comparison (2026-09-09) runs this function and
the v1 transfer (``transfer_heading``, kept below as the reference arm) on all 300 take/foot
pairs against the SMPL segment, which neither method reads.

**v1 (2026-09-09) replaced the heading changes wholesale with the shank's.** That removed the
drift but also every real ankle rotation about the vertical, which is why it added 1-3 degrees on
already-clean takes. v2 beats it on 288 of 300 pairs, makes no pair worse than measured by more
than a degree on 294 of 300, and leaves the twelve drifting takes at 13-25 degrees against v1's
12-35. Evidence and the comparison that scored the alternatives: the PRISM insole-heading
investigation (parent project record, 2026-09-09) section 5.7.

**Nothing measured is overwritten.** Measured data is never replaced with an estimate, and
beyond the rule, the measured stream is the only record of what the insole actually reported.
The corrected attitude goes in a companion file next to the take, the way PRISM already ships
``development_reference.npz``, and names its provenance per channel.

``imu_acceleration`` and ``imu_angular_velocity`` are not touched, and section 6 of the
investigation says why: the gyro is a body-frame quantity that a heading belief cannot affect,
and the accelerometer's gravity component is exactly invariant under a rotation about the world
vertical; its dynamic component is not, but whether the shipped value carries the drift could not
be determined, and a rewrite that guesses wrong would corrupt a signal that is currently sound.

    python scripts/poc/synthesize_insole_heading.py <bundle> [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The companion file. Named for what it holds, not for the take it belongs to.
OUTPUT_NAME = "heading_synth_reference.npz"
NAMESPACE = "heading_synth/prism_insole_ankle_v2"
#: Where the contact signal lives; PRISM ships it as a development reference.
CONTACT_FILE = "development_reference.npz"
CONTACT_KEY = "foot_contact_mask"

#: Canonical unified8 site order (small/00_sensors.qmd). Repeated here rather than imported so
#: the script can run against a bundle without the package on the path; asserted against the
#: take's own codes below.
SITE_ORDER = (
    "back_T4", "wrist_l", "wrist_r", "shank_l", "shank_r", "occiput", "foot_l", "foot_r",
)
#: Which 9-DOF channel the drift trend is measured against, for which insole. Same leg.
HEADING_DONOR = {"foot_l": "shank_l", "foot_r": "shank_r"}
#: Column of the contact mask for each insole.
CONTACT_COLUMN = {"foot_l": 0, "foot_r": 1}

MEASURED = "measured"
#: The heading is still the insole's own, with its drift removed; it is not a measurement any
#: more, so it is labelled synthesised and says how.
SYNTHESIZED = "synthesized:insole_dedrifted_shank_trend"


class SynthesisError(RuntimeError):
    """The take is not what this script expects, and it declines rather than guessing."""


def quats_to_R(q: np.ndarray) -> np.ndarray:
    """Unit quaternions wxyz (...,4) -> rotation matrices (...,3,3)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rows = [
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ]
    return np.stack(rows, axis=-1).reshape(q.shape[:-1] + (3, 3))


def R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrices (...,3,3) -> unit quaternions wxyz, with a fixed sign convention."""
    m = R.reshape(-1, 3, 3)
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.empty((len(m), 4), np.float64)
    # Branch on the largest diagonal term; the naive w-first formula loses precision near 180 deg.
    big_w = t > 0
    s = np.sqrt(np.maximum(t[big_w] + 1.0, 1e-12)) * 2
    q[big_w, 0] = 0.25 * s
    q[big_w, 1] = (m[big_w, 2, 1] - m[big_w, 1, 2]) / s
    q[big_w, 2] = (m[big_w, 0, 2] - m[big_w, 2, 0]) / s
    q[big_w, 3] = (m[big_w, 1, 0] - m[big_w, 0, 1]) / s
    rest = np.flatnonzero(~big_w)
    for k in rest:
        d = np.array([m[k, 0, 0], m[k, 1, 1], m[k, 2, 2]])
        i = int(np.argmax(d))
        j, l = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(m[k, i, i] - m[k, j, j] - m[k, l, l] + 1.0, 1e-12)) * 2
        q[k, 0] = (m[k, l, j] - m[k, j, l]) / s
        q[k, 1 + i] = 0.25 * s
        q[k, 1 + j] = (m[k, j, i] + m[k, i, j]) / s
        q[k, 1 + l] = (m[k, l, i] + m[k, i, l]) / s
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1.0            # one hemisphere, so two equal rotations compare equal
    return q.reshape(R.shape[:-2] + (4,))


def heading_of(R: np.ndarray) -> np.ndarray:
    """Rotation about the world vertical, in radians (azimuth of the sensor x axis)."""
    return np.arctan2(R[:, 1, 0], R[:, 0, 0])


def rot_z(rad: np.ndarray) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    R = np.zeros((len(rad), 3, 3))
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


# ----------------------------------------------------------------------- v1, kept as reference
def transfer_heading(R_foot: np.ndarray, R_donor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v1: keep the foot's tilt, follow the donor's heading changes. Returns (R_out, delta_rad).

    Superseded by ``dedrift_heading`` for the shipped companion, and kept because the method
    comparison scores exactly this function as its reference arm. The heading difference is
    centred on its own median, so the constant shank-to-foot mount offset is left alone and only
    the time-varying part moves.
    """
    delta = np.unwrap(heading_of(R_donor)) - np.unwrap(heading_of(R_foot))
    delta = delta - np.median(delta)
    return np.einsum("tij,tjk->tik", rot_z(delta), R_foot), delta


# ------------------------------------------------------------------------------ v2, shipped
def ankle_twist(R_foot: np.ndarray, R_shank: np.ndarray) -> np.ndarray:
    """World-Z twist of the foot relative to the shank, unwrapped, radians.

    ``twist_z(Q) = atan2(Q10 - Q01, Q00 + Q11)`` is the rotation about the world vertical that
    best aligns the two, and ``twist_z(Rz(a) Q) = twist_z(Q) + a``, so an insole heading error
    enters it additively. The difference of the two sensors' azimuths would not do: the insole's
    x axis passes through vertical when the foot points down, where its azimuth is undefined and
    an unwrap picks up a spurious turn.
    """
    Q = np.einsum("tij,tkj->tik", R_foot, R_shank)          # R_foot R_shank^T
    return np.unwrap(np.arctan2(Q[:, 1, 0] - Q[:, 0, 1], Q[:, 0, 0] + Q[:, 1, 1]))


def stride_segments(contact: np.ndarray) -> list[tuple[int, int]]:
    """[a, b) frame ranges between consecutive stance onsets (rising edges of contact), keeping
    the partial segments before the first onset and after the last. Every non-empty one counts."""
    c = np.asarray(contact).astype(np.int8)
    onsets = np.flatnonzero(np.diff(c) == 1) + 1
    edges = np.concatenate([[0], onsets, [len(c)]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Robust slope: the median of all pairwise slopes."""
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = dx != 0
    return float(np.median((y[j] - y[i])[ok] / dx[ok]))


def drift_slope(d: np.ndarray, t: np.ndarray, contact: np.ndarray | None) -> tuple[float, int]:
    """Drift rate of ``d(t)`` in rad/s and the number of stride segments it was fitted through.

    Theil-Sen through per-stride medians when there are at least two strides; otherwise an
    ordinary least-squares line over all frames, which is what a single segment reduces to.
    """
    segments = stride_segments(contact) if contact is not None else []
    if len(segments) >= 2:
        centres = np.array([t[a:b].mean() for a, b in segments])
        medians = np.array([np.median(d[a:b]) for a, b in segments])
        return theil_sen_slope(centres, medians), len(segments)
    tt = t - t.mean()
    return float((tt * (d - d.mean())).sum() / (tt * tt).sum()), len(segments)


def dedrift_heading(
    R_foot: np.ndarray, R_shank: np.ndarray, contact: np.ndarray | None, dt: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """v2: remove the linear heading drift, keep the ankle's real rotation.

    Returns (R_out, delta_rad, slope_rad_s, stride_count). The correction is ``-slope * t``,
    zero at the first frame, applied about the world vertical so the tilt is untouched.
    """
    t = np.arange(len(R_foot)) * float(dt)
    d = ankle_twist(R_foot, R_shank)
    slope, strides = drift_slope(d, t, contact)
    delta = -slope * t
    return np.einsum("tij,tjk->tik", rot_z(delta), R_foot), delta, slope, strides


@dataclass
class TakeResult:
    take_id: str
    frames: int
    drift_rate_deg_s: dict[str, float]
    shift_span_deg: dict[str, float]
    strides: dict[str, int]


def _load_contact(take_dir: Path, frames: int) -> np.ndarray | None:
    """The insole contact mask, or None when the take ships none (then a single segment)."""
    path = take_dir / CONTACT_FILE
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        if CONTACT_KEY not in z.files:
            return None
        mask = np.asarray(z[CONTACT_KEY]).astype(bool)
    if mask.ndim != 2 or mask.shape[0] != frames or mask.shape[1] < 2:
        raise SynthesisError(f"{take_dir.name}: {CONTACT_KEY} has shape {mask.shape}")
    return mask


def synthesize_take(take_dir: Path, *, dry_run: bool = False) -> TakeResult:
    """Write one take's companion file."""
    small = take_dir / "small_reference.npz"
    if not small.is_file():
        raise SynthesisError(f"{take_dir.name}: no small_reference.npz")

    with np.load(small, allow_pickle=False) as z:
        codes = [str(c) for c in z["sensor_codes"]]
        ori = np.asarray(z["imu_orientation"], np.float64)
        absolute = np.asarray(z["imu_orientation_absolute_heading"]).astype(bool)
        pair_id = str(z["pair_id"]) if "pair_id" in z.files else ""
        ts = np.asarray(z["timestamps_s"], np.float64) if "timestamps_s" in z.files else None

    if tuple(codes) != SITE_ORDER:
        raise SynthesisError(f"{take_dir.name}: unexpected sensor order {codes}")
    if ori.ndim != 3 or ori.shape[1] != 8 or ori.shape[2] != 4:
        raise SynthesisError(f"{take_dir.name}: imu_orientation has shape {ori.shape}")
    if ts is None or len(ts) < 2:
        raise SynthesisError(f"{take_dir.name}: no usable timestamps_s; the drift is a rate")
    dt = float(np.median(np.diff(ts)))

    targets = [c for c in HEADING_DONOR if not absolute[SITE_ORDER.index(c)]]
    if not targets:
        raise SynthesisError(
            f"{take_dir.name}: every channel already claims absolute heading; nothing to do"
        )
    for code in targets:
        donor = HEADING_DONOR[code]
        if not absolute[SITE_ORDER.index(donor)]:
            # Measuring the trend against a heading that is itself unreferenced would move the
            # problem, not fix it.
            raise SynthesisError(
                f"{take_dir.name}: reference {donor} has no absolute heading either"
            )

    contact_all = _load_contact(take_dir, ori.shape[0])
    R = quats_to_R(ori)
    # The untouched channels are copied as the quaternions the bundle ships, not re-derived
    # through a rotation matrix: bit-identical, so 'unchanged' is checkable with array_equal.
    out_q = ori.astype(np.float32).copy()
    source = np.array([MEASURED] * 8, dtype="<U48")
    shift = np.zeros((ori.shape[0], 2), np.float64)
    rates = np.zeros(2, np.float64)
    strides = np.zeros(2, np.int64)
    result = TakeResult(take_dir.name, ori.shape[0], {}, {}, {})

    for k, code in enumerate(("foot_l", "foot_r")):
        i = SITE_ORDER.index(code)
        if code not in targets:
            continue
        j = SITE_ORDER.index(HEADING_DONOR[code])
        contact = contact_all[:, CONTACT_COLUMN[code]] if contact_all is not None else None
        R_out, delta, slope, n = dedrift_heading(R[:, i], R[:, j], contact, dt)
        out_q[:, i] = R_to_quat_wxyz(R_out).astype(np.float32)
        source[i] = SYNTHESIZED
        shift[:, k] = np.degrees(delta)
        rates[k] = np.degrees(slope)
        strides[k] = n
        result.drift_rate_deg_s[code] = float(rates[k])
        result.shift_span_deg[code] = float(shift[:, k].max() - shift[:, k].min())
        result.strides[code] = int(n)

    if not dry_run:
        np.savez_compressed(
            take_dir / OUTPUT_NAME,
            namespace=np.array(NAMESPACE),
            pair_id=np.array(pair_id),
            sensor_codes=np.array(SITE_ORDER, dtype="<U16"),
            imu_orientation=out_q,
            imu_orientation_heading_source=source,
            heading_shift_deg=shift.astype(np.float32),
            heading_drift_rate_deg_s=rates.astype(np.float32),
            heading_stride_count=strides,
            heading_donor=np.array(
                [HEADING_DONOR.get(c, "") for c in SITE_ORDER], dtype="<U16"
            ),
            stride_source=np.array(
                f"{CONTACT_FILE}:{CONTACT_KEY} rising edges" if contact_all is not None
                else "none: no contact mask shipped, no strides, least-squares line over all frames"
            ),
            method=np.array(
                "the insole's own heading with its linear drift removed. drift = Theil-Sen "
                "slope through per-stride medians of the world-Z twist of R_foot R_shank^T "
                "(strides from the insole contact signal); correction = -slope * t, zero at "
                "the first frame, applied about the world vertical; tilt untouched"
            ),
            usage=np.array(
                "companion reference. small_reference.npz is unchanged and remains the measured "
                "record; imu_orientation_absolute_heading stays False for the insoles because "
                "that is a fact about the hardware, not about this file."
            ),
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="compute but write nothing")
    args = parser.parse_args(argv)

    if not args.bundle.is_dir():
        print(f"not a directory: {args.bundle}", file=sys.stderr)
        return 1

    takes = [p for p in sorted(args.bundle.iterdir())
             if p.is_dir() and (p / "small_reference.npz").is_file()]
    if args.limit:
        takes = takes[: args.limit]

    done, refused = 0, []
    rates: list[float] = []
    fallbacks = 0
    for take in takes:
        try:
            result = synthesize_take(take, dry_run=args.dry_run)
        except SynthesisError as error:
            refused.append(str(error))
            continue
        done += 1
        rates.extend(abs(v) for v in result.drift_rate_deg_s.values())
        fallbacks += sum(1 for n in result.strides.values() if n < 2)

    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {done} companion files in {args.bundle.name}")
    if rates:
        q = np.percentile(rates, [50, 95, 100])
        print(f"  |drift rate| per foot: median {q[0]:.2f}  p95 {q[1]:.2f}  max {q[2]:.2f} deg/s")
    if fallbacks:
        print(f"  feet fitted without strides (least-squares fallback): {fallbacks}")
    if refused:
        print(f"  refused {len(refused)}:")
        for line in refused[:10]:
            print(f"    {line}")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
