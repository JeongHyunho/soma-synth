"""Recover the faithful SMPL root translation for published PRISM runs.

A published run carries `betas`/`gender` on `anthro_reference.npz` and the per-joint global rotations
on `large_reference.npz`, but **no root translation** — so the viewer cannot place the body in the
world and skips the run. The value is not derivable from what is published:
`large["pelvis_position_world_aux"]` is PRISM's pelvis marker (`imu_gt["Pelvis"]["pos_world"]`), a
different quantity from the SMPL root, and the two differ by a time-varying ~0.11 m.

This one-time pass reads each run's own source pickle -- named by `anthro_reference.source_asset_id`
-- and writes the translation to a **sidecar** `smpl_root_translation.npz` next to the run. A sidecar
rather than an edit because the reference npz are the dataset pipeline's artifacts: nothing published
is mutated, and the sidecar can be regenerated or deleted at will. The proper long-term home is
`smpl_trans` inside `large_reference.npz` (per-frame kinematics); once the dataset publishes it there
the viewer prefers it and these sidecars become redundant.

The pickle is read through the numpy-whitelist restricted unpickler, never plain pickle.load.

    python enrich_root_translation.py --runs <dir> [--data-root <dir>] [--dry-run] [--force] [--limit N]

The source recordings are read under the data root (--data-root, else $SOMA_DATA_ROOT; there is no
default) or, when $SOMA_SOURCE_ROOT is set, under that source root.

INTERNAL-ONLY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import generate_prism_faithful as g   # noqa: E402  (load_prism_safe: restricted unpickler)
import viewer_paths                   # noqa: E402

SIDECAR = "smpl_root_translation.npz"
PROVENANCE = ("source_derived:prism_pkl smpl_params.trans (full take, no windowing); the true SMPL "
              "root translation, NOT large.pelvis_position_world_aux (PRISM pelvis marker)")


def find_runs(root):
    """Run folders under `root` holding the small+large reference npz, sorted."""
    out = []
    for cur, _dirs, files in os.walk(root):
        if "small_reference.npz" in files and "large_reference.npz" in files:
            out.append(cur)
    return sorted(out)


def run_frame_count(run_dir):
    z = np.load(os.path.join(run_dir, "large_reference.npz"), allow_pickle=True)
    if "frame_count" in z.files:
        return int(np.asarray(z["frame_count"]))
    return int(np.asarray(z["timestamps_s"]).shape[0])


def resolve_source(asset, data_root=None):
    """Locate a `source_asset_id` under the data root (or $SOMA_SOURCE_ROOT when that is set).

    Datasets differ on whether the id already includes the `extracted/` prefix — PRISM writes
    `prism/subj001/take002.pkl`, AMASS writes `extracted/amass/.../x.npz` — so both spellings are
    tried rather than one being assumed.
    """
    data_root = viewer_paths.data_root(data_root, hint="or pass --data-root <dir>")
    parts = [p for p in asset.split("/") if p]
    inner = parts[1:] if parts and parts[0] == "extracted" else parts
    cands = []
    source_root = os.environ.get("SOMA_SOURCE_ROOT", "").strip()
    if source_root and inner:
        cands.append(os.path.join(source_root, *inner))
    cands.append(os.path.join(data_root, *parts))
    if parts and parts[0] != "extracted":
        cands.append(os.path.join(data_root, "extracted", *parts))
    for c in cands:
        if os.path.isfile(c):
            return c
    return cands[-1]                                       # keeps the error message concrete


def source_recording_for(run_dir, data_root=None):
    """(source_path, source_asset_id, pair_id) from the run's own anthro reference."""
    p = os.path.join(run_dir, "anthro_reference.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{run_dir}: no anthro_reference.npz to name the source")
    z = np.load(p, allow_pickle=True)
    if "source_asset_id" not in z.files:
        raise KeyError(f"{run_dir}: anthro_reference.npz has no source_asset_id")
    asset = str(np.asarray(z["source_asset_id"]))          # e.g. prism/subj001/take002.pkl
    pair_id = str(np.asarray(z["pair_id"])) if "pair_id" in z.files else ""
    return resolve_source(asset, data_root), asset, pair_id


def _source_trans(src_path):
    """(trans[N,3], source_fps) straight from the source recording, whatever format it is.

    PRISM ships pickles (read through the numpy-whitelist restricted unpickler, never plain pickle);
    AMASS ships npz. Any dataset following the contract only has to name its source in
    `anthro_reference.source_asset_id` and carry a root translation in it.
    """
    ext = os.path.splitext(src_path)[1].lower()
    if ext == ".pkl":
        raw = g.load_prism_safe(src_path)
        return np.asarray(raw["smpl_params"]["trans"], np.float64), None
    if ext == ".npz":
        z = np.load(src_path, allow_pickle=True)
        if "trans" not in z.files:
            raise ValueError(f"{src_path}: npz has no 'trans' (keys {list(z.files)})")
        fps = float(np.asarray(z["mocap_framerate"])) if "mocap_framerate" in z.files else None
        return np.asarray(z["trans"], np.float64), fps
    raise ValueError(f"{src_path}: unsupported source format {ext!r}")


def _resample_linear(trans, fps_in, frames_out, fps_out):
    """Put the source translation on the run's clock, the way the manifests document it."""
    n_in = trans.shape[0]
    t_in = np.arange(n_in) / float(fps_in)
    t_out = np.arange(frames_out) / float(fps_out)
    return np.stack([np.interp(t_out, t_in, trans[:, k]) for k in range(3)], axis=1)


def extract_trans(src_path, frames, resample=None):
    """The take's SMPL root translation [frames,3], on the run's own clock."""
    trans, fps_in = _source_trans(src_path)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"{src_path}: trans shape {trans.shape} is not [N,3]")
    if trans.shape[0] != frames:
        rs = resample or {}
        fps_in = rs.get("source_fps", fps_in)
        fps_out = rs.get("target_fps")
        if not fps_in or not fps_out:
            raise ValueError(f"{src_path}: {trans.shape[0]} source frames vs {frames} in the run, and "
                             f"the manifest does not document the rates needed to resample")
        if rs.get("frames_in") not in (None, trans.shape[0]):
            raise ValueError(f"{src_path}: manifest says {rs['frames_in']} source frames, file has "
                             f"{trans.shape[0]} — not the recording this run came from")
        trans = _resample_linear(trans, fps_in, frames, fps_out)
    if not np.all(np.isfinite(trans)):
        raise ValueError(f"{src_path}: root translation has non-finite values")
    return trans


def sidecar_ok(run_dir, frames):
    """True when a valid sidecar for this frame count already exists."""
    p = os.path.join(run_dir, SIDECAR)
    if not os.path.isfile(p):
        return False
    try:
        z = np.load(p, allow_pickle=True)
        t = np.asarray(z["smpl_trans"])
        return t.shape == (frames, 3) and bool(np.all(np.isfinite(t)))
    except Exception:
        return False


def enrich_run(run_dir, dry_run=False, force=False, data_root=None):
    """Write the sidecar for one run. Returns a short status string."""
    frames = run_frame_count(run_dir)
    if not force and sidecar_ok(run_dir, frames):
        return "skip (already enriched)"
    src, asset, pair_id = source_recording_for(run_dir, data_root)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"{run_dir}: source recording not found: {src}")
    resample = {}
    mpath = os.path.join(run_dir, "manifest.json")
    if os.path.isfile(mpath):
        with open(mpath, encoding="utf-8") as fh:
            resample = json.load(fh).get("resample") or {}
    trans = extract_trans(src, frames, resample)

    # Diagnostic, not a gate: `pelvis_position_world_aux` equals the root translation in AMASS
    # bundles but is the pelvis MARKER in PRISM ones, so the two conventions must stay
    # distinguishable rather than be assumed.
    note = ""
    lpath = os.path.join(run_dir, "large_reference.npz")
    if os.path.isfile(lpath):
        lz = np.load(lpath, allow_pickle=True)
        if "pelvis_position_world_aux" in lz.files:
            aux = np.asarray(lz["pelvis_position_world_aux"], np.float64)
            if aux.shape == trans.shape:
                note = f" (pelvis_aux delta {np.max(np.abs(aux - trans)):.2e} m)"

    if dry_run:
        return f"would write {frames} frames from {asset}{note}"
    tmp = os.path.join(run_dir, SIDECAR + ".tmp")
    with open(tmp, "wb") as fh:                            # a file object: savez must not rename it
        np.savez(fh,
                 smpl_trans=trans.astype(np.float32),
                 frame_count=np.int64(frames),
                 pair_id=np.array(pair_id),
                 source_asset_id=np.array(asset),
                 smpl_trans_provenance=np.array(PROVENANCE))
    os.replace(tmp, os.path.join(run_dir, SIDECAR))        # atomic: no half-written sidecar
    return f"wrote {frames} frames from {asset}{note}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write faithful SMPL root-translation sidecars for runs.")
    ap.add_argument("--runs", required=True, help="folder containing the run directories")
    ap.add_argument("--data-root", default=None,
                    help="root holding extracted/prism/... (default $SOMA_DATA_ROOT, which must then be set)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="rewrite sidecars that already look valid")
    ap.add_argument("--limit", type=int, default=0, help="process at most N runs (0 = all)")
    a = ap.parse_args(argv)
    a.data_root = viewer_paths.data_root(a.data_root, hint="or pass --data-root <dir>")

    runs = find_runs(a.runs)
    if a.limit:
        runs = runs[:a.limit]
    print(f"[enrich] {len(runs)} run(s) under {a.runs}")

    done = skipped = 0
    failures = []
    for i, run in enumerate(runs, 1):
        name = os.path.basename(run)
        try:
            status = enrich_run(run, dry_run=a.dry_run, force=a.force, data_root=a.data_root)
        except Exception as exc:                            # keep going; report every failure at the end
            failures.append((name, str(exc)))
            print(f"  [{i}/{len(runs)}] {name}: FAILED {exc}")
            continue
        if status.startswith("skip"):
            skipped += 1
        else:
            done += 1
        print(f"  [{i}/{len(runs)}] {name}: {status}")

    print(f"\n[enrich] written={done} skipped={skipped} failed={len(failures)}")
    for name, why in failures:
        print(f"  FAILED {name}: {why}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
