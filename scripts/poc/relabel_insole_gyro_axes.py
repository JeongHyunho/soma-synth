"""Rewrite the foot gyro of a shipped PRISM bundle onto the axes it should have had.

``generate_prism_measured.py`` mapped the insole's ``gyr_local_raw`` into the spec sensor frame
with ``r_cal^T`` alone, which assumes the raw gyro shares the body frame the insole's own
orientation maps out of. It does not: the two are a fixed relabel apart,
``GYR_LOCAL_AXIS_RELABEL`` = ``(a, b, c) -> (-a, c, b)``, solved for on all 150 takes x 2 feet
and confirmed by integration (a healthy foot's gyro reaches its own orientation within 2.9
degrees with the relabel, 134.6 without). The generator is fixed; this script brings the bundle
that was already generated into line with it.

This is a **transcription correction, not an estimate.** The measured quantity is the raw gyro;
what shipped was that measurement expressed on the wrong axes. Re-expressing it on the right
axes restores fidelity to the measurement rather than replacing it; replacing measured data with
an estimate is the opposite move and stays forbidden. It is nonetheless a rewrite of shipped
tensors, made by decision on 2026-09-14.

What changes: ``imu_angular_velocity[:, foot_l]`` and ``[:, foot_r]`` in ``small_reference.npz``,
by ``w' = r_cal^T M r_cal w`` where ``r_cal`` is the transpose of that channel's shipped
``q_anatomical_from_sensor``. Every other array is carried through and checked bit-for-bit.
The manifest gains ``small_measured.gyro_axis_relabel`` so a reader -- and this script -- can
tell a corrected take from an uncorrected one; running twice is refused, not doubled.

``M`` is its own inverse, but a float32 round trip through the file is not, so rollback does not
recompute: the journal keeps each take's original foot gyro as a ``.npy`` sidecar (about 25 MB
for the bundle), and ``--rollback`` puts it back, removes the marker, and requires the array to
hash to the recorded pre-image before it is kept.

    python scripts/poc/relabel_insole_gyro_axes.py <bundle> [--dry-run]
    python scripts/poc/relabel_insole_gyro_axes.py --rollback <journal.jsonl>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]


def _load_script(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_measured = _load_script("generate_prism_measured", "scripts/poc/generate_prism_measured.py")
_backfill = _load_script("backfill_manifest_fields", "scripts/backfill_manifest_fields.py")

M = _measured.GYR_LOCAL_AXIS_RELABEL
SITE_ORDER = (
    "back_T4", "wrist_l", "wrist_r", "shank_l", "shank_r", "occiput", "foot_l", "foot_r",
)
FOOT_CODES = ("foot_l", "foot_r")
MARKER_KEY = "gyro_axis_relabel"
EVIDENCE = "PRISM insole-heading investigation (parent project record, 2026-09-09) sec 5"


class RelabelError(RuntimeError):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (q / np.linalg.norm(q))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def relabel_gyro(w: np.ndarray, q_anat: np.ndarray) -> np.ndarray:
    """``r_cal^T M r_cal w`` for one channel. ``q_anat`` is ``r_cal^T`` as a quaternion."""
    r_cal = quat_wxyz_to_R(np.asarray(q_anat, np.float64)).T
    T = r_cal.T @ M @ r_cal
    return np.einsum("ij,nj->ni", T, np.asarray(w, np.float64))


@dataclass
class TakePlan:
    take_id: str
    small: Path
    manifest: Path
    already: bool = False
    problems: list[str] = field(default_factory=list)


def inspect_take(take_dir: Path) -> TakePlan:
    plan = TakePlan(take_dir.name, take_dir / "small_reference.npz", take_dir / "manifest.json")
    if not plan.small.is_file() or not plan.manifest.is_file():
        plan.problems.append("missing small_reference.npz or manifest.json")
        return plan
    with np.load(plan.small, allow_pickle=False) as z:
        codes = tuple(str(c) for c in z["sensor_codes"])
        if codes != SITE_ORDER:
            plan.problems.append(f"unexpected sensor order {codes}")
        prov = [str(p) for p in z["gyro_provenance"]]
        for code in FOOT_CODES:
            if not prov[SITE_ORDER.index(code)].startswith("measured:gyr_local_raw"):
                plan.problems.append(f"{code} gyro is not the measured raw stream")
    raw = plan.manifest.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if _backfill.detect_style(raw, manifest) is None:
        plan.problems.append("manifest does not round-trip byte-exactly")
    plan.already = bool((manifest.get("small_measured") or {}).get(MARKER_KEY, {}).get("applied"))
    return plan


def apply_take(plan: TakePlan, *, sidecar_dir: Path | None = None,
               restore: np.ndarray | None = None) -> dict:
    """Rewrite one take and return the journal entry.

    Forward: relabel the two foot channels and, when ``sidecar_dir`` is given, save their
    original values there so rollback can be exact. Rollback: pass ``restore`` (the sidecar's
    array) and the marker is dropped instead of written.
    """
    with np.load(plan.small, allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    before_file = sha256_bytes(plan.small.read_bytes())
    before_arrays = {k: sha256_array(v) for k, v in arrays.items()}

    w = arrays["imu_angular_velocity"]
    feet = [SITE_ORDER.index(c) for c in FOOT_CODES]
    undo = restore is not None
    new = w.astype(np.float64)
    if undo:
        if restore.shape != (w.shape[0], len(feet), 3):
            raise RelabelError(f"{plan.take_id}: sidecar shape {restore.shape} does not fit")
        new[:, feet] = restore
    else:
        q_anat = arrays["q_anatomical_from_sensor"]
        for i in feet:
            new[:, i] = relabel_gyro(w[:, i], q_anat[i])
        if sidecar_dir is not None:
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            np.save(sidecar_dir / f"{plan.take_id}.npy", w[:, feet])
    arrays["imu_angular_velocity"] = new.astype(w.dtype)

    # numpy appends ".npz" to any other suffix, so the temporary name has to end in it already.
    tmp = plan.small.with_name(plan.small.stem + ".rewriting.npz")
    np.savez_compressed(tmp, **arrays)
    with np.load(tmp, allow_pickle=False) as z:
        after_arrays = {k: sha256_array(np.array(z[k])) for k in z.files}
    untouched = [k for k in arrays if k != "imu_angular_velocity"]
    drifted = [k for k in untouched if after_arrays[k] != before_arrays[k]]
    if drifted:
        tmp.unlink()
        raise RelabelError(f"{plan.take_id}: arrays changed that must not: {drifted}")
    os.replace(tmp, plan.small)

    raw = plan.manifest.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    style = _backfill.detect_style(raw, manifest)
    if style is None:
        raise RelabelError(f"{plan.take_id}: manifest no longer round-trips; npz already rewritten")
    block = manifest.setdefault("small_measured", {})
    if undo:
        block.pop(MARKER_KEY, None)
    else:
        block[MARKER_KEY] = {
            "applied": True,
            "channels": list(FOOT_CODES),
            "matrix": M.astype(int).tolist(),
            "formula": "w_corrected = r_cal^T @ M @ r_cal @ w_shipped; r_cal = transpose(q_anatomical_from_sensor)",
            "applied_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "evidence": EVIDENCE,
        }
    written = style.dump(manifest)
    plan.manifest.write_bytes(written)

    return {
        "take_id": plan.take_id,
        "small": str(plan.small),
        "manifest": str(plan.manifest),
        "undo": undo,
        "sidecar": str(sidecar_dir / f"{plan.take_id}.npy") if (sidecar_dir and not undo) else "",
        "sha256_small_before": before_file,
        "sha256_small_after": sha256_bytes(plan.small.read_bytes()),
        "sha256_gyro_before": before_arrays["imu_angular_velocity"],
        "sha256_gyro_after": after_arrays["imu_angular_velocity"],
        "sha256_manifest_before": sha256_bytes(raw),
        "sha256_manifest_after": sha256_bytes(written),
        "untouched_arrays": len(untouched),
    }


def rollback(journal: Path, log=print) -> int:
    restored = refused = 0
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        small = Path(entry["small"])
        if not small.is_file():
            log(f"  MISSING {entry['take_id']}")
            refused += 1
            continue
        if sha256_bytes(small.read_bytes()) == entry["sha256_small_before"]:
            continue
        with np.load(small, allow_pickle=False) as z:
            gyro_now = sha256_array(np.array(z["imu_angular_velocity"]))
        if gyro_now != entry["sha256_gyro_after"]:
            log(f"  CHANGED SINCE {entry['take_id']}: not the gyro this journal wrote; skipped")
            refused += 1
            continue
        sidecar = Path(entry["sidecar"]) if entry.get("sidecar") else None
        if sidecar is None or not sidecar.is_file():
            log(f"  NO SIDECAR {entry['take_id']}: cannot restore exactly; skipped")
            refused += 1
            continue
        plan = inspect_take(small.parent)
        result = apply_take(plan, restore=np.load(sidecar, allow_pickle=False))
        if result["sha256_gyro_after"] != entry["sha256_gyro_before"]:
            log(f"  NOT EXACT {entry['take_id']}: gyro did not return to its pre-image")
            refused += 1
            continue
        restored += 1
    log(f"rollback: {restored} restored, {refused} refused")
    return 1 if refused else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", nargs="?", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--journal-dir", type=Path, default=None)
    parser.add_argument("--rollback", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.rollback:
        return rollback(args.rollback)
    if not args.bundle or not args.bundle.is_dir():
        parser.error("give a bundle directory, or --rollback a journal")

    plans = [inspect_take(p) for p in sorted(args.bundle.iterdir()) if p.is_dir()]
    plans = [p for p in plans if p.small.parent.is_dir()]
    todo = [p for p in plans if not p.problems and not p.already]
    skipped = [p for p in plans if p.already]
    bad = [p for p in plans if p.problems]
    print(f"{args.bundle.name}: {len(todo)} to rewrite, {len(skipped)} already relabelled, "
          f"{len(bad)} refused")
    for p in bad[:10]:
        print(f"  refused {p.take_id}: {'; '.join(p.problems)}")
    if args.dry_run or not todo:
        return 1 if bad else 0

    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    journal_dir = args.journal_dir or (args.bundle.parent / "_manifest_backfill")
    journal_dir.mkdir(parents=True, exist_ok=True)
    journal = journal_dir / f"{stamp}_{args.bundle.name}_gyro_relabel.jsonl"
    sidecars = journal_dir / f"{stamp}_{args.bundle.name}_gyro_relabel"
    with journal.open("w", encoding="utf-8", newline="\n") as fh:
        for plan in todo:
            fh.write(json.dumps(apply_take(plan, sidecar_dir=sidecars), ensure_ascii=False) + "\n")
    print(f"rewrote {len(todo)} takes; journal {journal}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
