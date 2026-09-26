"""Find takes whose retargeted pose does something no body can do, and mark them in the INDEX.

The defect class is a source rotation that no body performs: a segment that flips for a few frames,
or a trunk that is inverted for a whole take. Neither the validator nor the sensor channels can be
relied on to surface it, so it is measured on the pose itself. HKNU examples:

  1. `S12 Side_Squat_2` -- the source head segment flips ~178 deg for about five frames. The worn
     head sensor peaks at 171 deg/s across the same take, so the head did not move; the source
     rotation did. L2 catches it only because the flip persists.
  2. `S03 Walking_4KPH_10` -- the source trunk segment is inverted for 100% of frames (pelvis->trunk
     170 deg, against 4.9-9.0 in the subject's eight other walking takes). Every check passes.
  3. A take can carry |joint_velocity| above 8000 deg/s in `large_reference.npz` and still pass:
     L2 reads `small_reference.npz`, and nothing bands the large artifact.
  4. The gyro is a centered difference, so a flip lasting a single frame cancels out of it exactly.

Two rules, both measured:

  step      Any joint whose global rotation implies an angular rate the spec's own L2 band already
            rejects. The limit is NOT a number chosen here: it is `gyro_abs_max_deg_s` divided by
            the take's frame rate, so at 100 Hz a step over 80 deg is the same 8000 deg/s the
            validator refuses in the small artifact. Applying it to the pose closes the two holes
            that let this class through: L2 reads only `small_reference.npz`, and the gyro is a
            centered difference in which a one-frame flip cancels exactly.

            A fixed angle read off one corpus's distribution would not transfer: a 30 deg gap
            separates HKNU's clean and flagged takes, but AddBiomechanics has no such gap (its two
            populations touch at 29.99 and 30.03). Deriving the limit from the existing band
            accepts two HKNU takes whose steps imply 5706 and 7270 deg/s: under the band, so the
            validator accepts them, and so does this.

  inverted  The trunk turned more than --trunk-limit degrees from the pelvis, at the median. A back
            bends; it does not invert. Measured at the median rather than the max so that a brief
            flip is reported by the step rule and a whole-take inversion by this one.

            150 deg, not 90. HKNU's inverted take sits at 170.02 with nothing else above 27.15,
            but AddBiomechanics runs continuously up to 121.52 with a p99 of 44, and deep
            hip-hinge postures live in that range. Whether a 121 deg trunk is a real
            posture or a defect is not something this scan can decide, so it does not: it reports
            the distribution and excludes only what is unambiguously inverted.

`--apply` rewrites the INDEX entries for offending takes to `status: excluded` with the measurement
as the reason. Excluded takes stay on disk and stay in the INDEX: the validator checks only `ok`
takes, and a take that was dropped silently is a take nobody can audit.

Usage:
    python scripts/diagnostics/pose_discontinuity_scan.py <dataset_dir> [--apply] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

#: The spec's own blow-up guard, in deg/s. Imported rather than restated so that raising the band
#: raises this rule with it; a copied constant is a rule that silently stops matching.
try:
    from soma_synth.validation.checks import DEFAULT_BANDS

    GYRO_ABS_MAX_DEG_S = float(DEFAULT_BANDS.gyro_abs_max_deg_s)
except ImportError:                     # running without src on the path
    GYRO_ABS_MAX_DEG_S = 8000.0

TRUNK_LIMIT_DEG = 150.0
#: Written into the INDEX entries this script excludes, so a re-run can reinstate its own verdicts
#: and leave anyone else's alone.
MARKER = "pose_discontinuity_scan"
#: SMPL indices whose global rotation the bundle carries.
PELVIS, SPINE3 = 0, 9


def _matrices(quat_wxyz: np.ndarray) -> np.ndarray:
    """[T,J,4] w-first -> [T,J,3,3]."""
    frames, joints = quat_wxyz.shape[:2]
    flat = quat_wxyz.reshape(-1, 4)
    xyzw = np.column_stack([flat[:, 1], flat[:, 2], flat[:, 3], flat[:, 0]])
    return Rotation.from_quat(xyzw).as_matrix().reshape(frames, joints, 3, 3)


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle of `a^T b`, in degrees, without forming the product.

    ``trace(A^T B)`` is just the elementwise sum of A and B, so the rotation angle falls out of a
    sum over the last two axes. Building the matrices and handing them to scipy costs about forty
    times more, which is the difference between minutes and hours over a corpus of tens of
    thousands of takes; both give the same angles.
    """
    trace = np.einsum("...ij,...ij->...", a, b)
    return np.degrees(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))


def scan_take(take_dir: pathlib.Path, gyro_limit: float, trunk_limit: float) -> dict:
    with np.load(take_dir / "large_reference.npz", allow_pickle=False) as z:
        globals_ = _matrices(np.asarray(z["smpl_global_orientation_world"], np.float64))
        timestamps = np.asarray(z["timestamps_s"], np.float64)

    frames = globals_.shape[0]
    problems: list[str] = []
    # From the take's own clock, so a corpus at another rate is judged at its own rate rather than
    # at the one this script was written against.
    rate = 1.0 / float(np.median(np.diff(timestamps))) if timestamps.size > 1 else 100.0
    step_limit = gyro_limit / rate

    steps = _angle_between(globals_[:-1], globals_[1:])          # [T-1, 24]
    worst = float(steps.max())
    if worst > step_limit:
        joint = int(steps.max(axis=0).argmax())
        count = int((steps > step_limit).sum())
        problems.append(
            f"pose discontinuity: joint {joint} steps {worst:.1f} deg between adjacent frames "
            f"= {worst * rate:.0f} deg/s at {rate:.0f} Hz, past the spec's {gyro_limit:.0f} deg/s "
            f"band ({count} sample(s) over {step_limit:.1f} deg)"
        )

    trunk = _angle_between(globals_[:, PELVIS], globals_[:, SPINE3])
    trunk_median = float(np.median(trunk))
    if trunk_median > trunk_limit:
        problems.append(
            f"inverted trunk: pelvis->spine3 median {trunk_median:.1f} deg over the whole take; "
            f"a back bends, it does not invert"
        )

    return {"take": take_dir.name, "frames": frames, "max_step_deg": round(worst, 3),
            "trunk_median_deg": round(trunk_median, 3), "problems": problems}


def retell_counts(root: pathlib.Path, counts: dict) -> None:
    """Say the new counts in the root description too, which the generator wrote before this ran.

    The generator emits `DATA_DESCRIPTION_EN.md` at publish time, when nothing has been excluded
    yet, so its take line reads `287 ok, 0 excluded` while the INDEX this scan just wrote reads
    `280 ok, 7 excluded`. Publishing both hands the team a bundle that contradicts itself about what
    is in it. Only the one generated line is touched, and only where it still has the shape the
    generator gives it -- a description someone has since rewritten by hand is left alone and said
    so, rather than being edited from under them.
    """
    wanted = (f"- takes: {counts['ok']} ok, {counts['failed']} failed, "
              f"{counts['excluded']} excluded of {counts['total']}")
    for name in ("DATA_DESCRIPTION_EN.md", "DATASET_DESCRIPTION_EN.md"):
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        new_text, hits = re.subn(r"^- takes: \d+ ok, \d+ failed, \d+ excluded of \d+$",
                                 wanted, text, count=1, flags=re.M)
        if not hits:
            print(f"{name}: no generated take line to update, left untouched -- check by hand "
                  f"that it says {counts['ok']} ok / {counts['excluded']} excluded")
        elif new_text != text:
            path.write_text(new_text, encoding="utf-8", newline="\n")
            print(f"{name}: take line updated to match the INDEX")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--gyro-limit", type=float, default=GYRO_ABS_MAX_DEG_S,
                        help="deg/s; the spec's L2 blow-up guard, converted to a per-frame step")
    parser.add_argument("--trunk-limit", type=float, default=TRUNK_LIMIT_DEG)
    parser.add_argument("--apply", action="store_true",
                        help="rewrite the INDEX, excluding offending takes with the measurement")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    root = pathlib.Path(args.dataset)
    index_path = root / "INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    by_rel = {t.get("rel") or t.get("relative_path"): t for t in index["takes"]}

    # Re-running with a different limit must replace the previous verdict, not add to it. Only this
    # script's own exclusions are cleared, so a status set for any other reason survives.
    reinstated = 0
    for entry in index["takes"]:
        if entry.get("excluded_by") == MARKER:
            entry["status"] = "ok"
            entry.pop("reason", None)
            entry.pop("excluded_by", None)
            reinstated += 1
    if reinstated:
        print(f"reinstated {reinstated} take(s) excluded by a previous run of this scan")

    candidates = [e for e in index["takes"] if e.get("status") == "ok"]
    results = []
    started = time.time()
    for done, entry in enumerate(candidates, 1):
        rel = entry.get("rel") or entry.get("relative_path")
        take_dir = root / rel
        if not (take_dir / "large_reference.npz").exists():
            continue
        results.append(scan_take(take_dir, args.gyro_limit, args.trunk_limit))
        # A 40,000-take corpus is an hour of silence otherwise, which is indistinguishable from a
        # hang. Printed to stderr so `--json` consumers and pipes are unaffected.
        if done % 2000 == 0:
            rate = done / max(time.time() - started, 1e-9)
            print(f"  scanned {done}/{len(candidates)} ({rate:.0f}/s, "
                  f"~{(len(candidates) - done) / rate / 60:.0f} min left)",
                  file=sys.stderr, flush=True)

    bad = [r for r in results if r["problems"]]
    steps = np.array([r["max_step_deg"] for r in results])
    print(f"scanned {len(results)} ok takes in {root.name}")
    if steps.size:
        print(f"max frame-to-frame step: median {np.median(steps):.3f}  "
              f"p99 {np.percentile(steps, 99):.3f}  max {steps.max():.1f} deg")
    print(f"takes with a problem: {len(bad)}\n")
    for r in bad:
        print(f"  {r['take']}  ({r['frames']} frames)")
        for problem in r["problems"]:
            print(f"      {problem}")


    if args.apply:
        for r in bad:
            entry = by_rel.get(r["take"])
            if entry is None:
                continue
            entry["status"] = "excluded"
            entry["reason"] = "; ".join(r["problems"])
            entry["excluded_by"] = MARKER
        ok = sum(1 for t in index["takes"] if t.get("status") == "ok")
        excluded = sum(1 for t in index["takes"] if t.get("status") == "excluded")
        failed = sum(1 for t in index["takes"] if t.get("status") == "failed")
        index["counts"] = {"total": len(index["takes"]), "ok": ok,
                           "failed": failed, "excluded": excluded}
        index["complete"] = failed == 0
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2),
                              encoding="utf-8", newline="\n")
        print(f"\nINDEX updated: ok={ok} excluded={excluded} failed={failed}")
        retell_counts(root, index["counts"])

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8",
                                           newline="\n")
    return 1 if bad and not args.apply else 0


if __name__ == "__main__":
    raise SystemExit(main())
