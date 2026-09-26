"""Batch: build the PRISM poc-demo dataset `prism_faithful_full` with the PHYSICAL-measured small.

This is THE builder for the deployed PRISM poc-demo dataset. Each take gets the physical-measured small
(from PRISM's physical `imu.*`: gravity-included specific force, spec sensor frame S v2 axes) plus the
kinematic large/anthro/development reference (an IMU cannot measure absolute pose/trajectory). One pass
produces the whole bundle: `build_measured` calls the faithful build internally and swaps only `small`.

It reuses the faithful batch engine (enumerate_specs, per-subject anthro constants) as a helper library.
NOTE: `generate_prism_faithful_all.py.__main__` writes the SAME directory with the *ideal* (imu_gt)
small; do not run it for deployment — this measured builder is canonical.

Data-only, resumable, per-take error isolation, INDEX.json coverage. INTERNAL-ONLY,
experimental_non_candidate. Local generation; sharing a bundle is a separate, separately
approved step.

    python scripts/poc/generate_prism_measured_all.py \
        [--source-root <prism folder>] [--out <bundle dir>] [--only prism_subj001] [--limit N] [--force]
    python scripts/poc/generate_prism_measured_all.py --data-root <root> [--only ...]

Input and output are separate: `--source-root` is the PRISM folder itself, the one holding
`<subject>/<take>.pkl` -- not `SOMA_SOURCE_ROOT`, which holds it (default
`$SOMA_SOURCE_ROOT/prism`, by default `$SOMA_DATA_ROOT/extracted/prism`) -- and `--out` the bundle
directory (default `$SOMA_DATA_ROOT/runs/experimental_generation_poc_demo/prism_faithful_full`).
A source folder with no subject folder of `*.pkl` takes, or a selection of no take, stops the run
before anything is written, and so does `--only` / `--limit` when the bundle directory is the
production lineage directory under `$SOMA_DATA_ROOT` (pass a scratch `--out`).
`--data-root <root>` keeps its old meaning for the source and the bundle -- read
`<root>/extracted/prism`, write `<root>/runs/experimental_generation_poc_demo/prism_faithful_full`
-- and cannot be combined with either.

The body models are not covered by `--data-root`: they come from
`$SOMA_DATA_ROOT/body_models/smpl` whichever form is used, as they did before. So `SOMA_DATA_ROOT`
must be set in both forms; without it the run stops before writing anything. With `--data-root`,
`<root>/extracted` is guarded as read-only in addition to whatever the environment configures.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

SPEC_ID = "qmd_unified8_smpl18"
SPEC_VERSION = "faithful-v2"


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fn))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


faithful_all = _load("generate_prism_faithful_all", "generate_prism_faithful_all.py")
measured = _load("generate_prism_measured", "generate_prism_measured.py")
_gen = measured.faithful                                   # faithful generator (write_outputs, safe load)
TakeSpec = _gen.TakeSpec
paths = _gen.paths

# The deployed PRISM poc-demo dataset (measured small). No version suffix: one directory per
# lineage, replaced in place when regenerated (retention rule 1.1, docs/guides/RETENTION_RULES.md).
DATASET_DIRNAME = "prism_faithful_full"
CORE_FILES = faithful_all.CORE_FILES

# Reused faithful batch helpers (spec enumeration + per-subject anthro fit are identical).
enumerate_specs = faithful_all.enumerate_specs
compute_subject_constants = faithful_all.compute_subject_constants


def _write_dataset_description(out_root: Path, counts: dict, total: int) -> None:
    md = f"""# PRISM Small/Large full dataset ({DATASET_DIRNAME})

> **INTERNAL-ONLY.** Do not upload to public Git / cloud / model or dataset hubs.
> **experimental_non_candidate** — measured-sensor simulation reference, NOT contract-compliant,
> NOT quality-gated. Not a training/evaluation label. PRISM_SOURCE_PASS does not authorize this.

One subdirectory per take: `prism-<subj>-<take>/` with `small_reference.npz`, `large_reference.npz`,
`anthro_reference.npz`, `development_reference.npz`, `smpl_root_translation.npz`, `manifest.json`. Per-take
field/shape/convention truth is the take's `manifest.json`; the full field layout is in
`DATA_DESCRIPTION_EN.md`. Lineage: `spec_id=qmd_unified8_smpl18`, `spec_version=faithful-v2` (large uses the
canonical key `smpl_global_orientation_world`).

Small simulates the REAL physical wearable IMU, built from PRISM's physical `imu.*` stream:

- `imu_acceleration` = specific force `R^T(a_world - g)`, **gravity-included (~9.8 m/s^2)** as a real
  accelerometer reads.
- `imu_orientation` = physical `imu.ori` (SO(3)-projected) calibrated into the spec **sensor frame S**
  (`small/00_sensors.qmd`): +Y=proximal, +Z=outward lateral, +X=Y x Z; left limbs mounted 180 deg about Y
  (left +X=posterior). `axis_convention=spec_S_v2`.
- gyro: feet from the measured `gyr_local_raw` (deg/s) rotated to S; body sites derived from orientation.
- `back_T4` has no PRISM trunk sensor -> non-physical spine3 proxy retained (flagged), relabeled to the
  spec back frame. `imu_synth_mask` exposes gap-filled physical frames (R_Wrist ~56%).

`large_reference.npz` (motion), `anthro_reference.npz` (skeleton), and `development_reference.npz`
(insole) are the kinematic references (an IMU cannot measure absolute joint pose or world trajectory
without drift).

Coverage: {counts['ok']} ok / {counts['skipped']} skipped / {counts['failed']} failed of {total}
(see `INDEX.json`). Full take length at native 100 Hz.

Splitting for ML is downstream and carries the mandatory disclosure: the official PRISM split is
chunk-level and 126/150 takes contribute to both sides, so test scores are not an across-take or
across-subject generalisation bound (n=6 subjects).
"""
    # ADR-0040 D3: one English description per bundle, under the name the distribution guide and
    # the L0/L4 checks both require.
    (out_root / "DATA_DESCRIPTION_EN.md").write_text(md, encoding="utf-8")


def _write_root_translation(out_dir, spec, manifest, anthro) -> None:
    """Write smpl_root_translation.npz (SMPL trans, a faithful-v2 CORE deliverable) for the take.

    A real ok take always has a loadable source (build_measured already loaded it); a stub/fake build
    without a real pkl is skipped rather than failing the take.
    """
    try:
        raw = _gen.load_prism_safe(spec.source_pkl)
    except Exception:
        return
    trans = np.asarray(raw["smpl_params"]["trans"], np.float32)
    ws, we = spec.window if spec.window is not None else (0, trans.shape[0])
    np.savez(
        os.path.join(str(out_dir), "smpl_root_translation.npz"),
        smpl_trans=trans[ws:we],
        frame_count=np.int64(we - ws),
        pair_id=np.asarray(str(manifest["identity"]["pair_id"])),
        source_asset_id=np.asarray(str(anthro["source_asset_id"])),
        smpl_trans_provenance=np.asarray(
            "source_derived:prism_pkl smpl_params.trans (full take, window-sliced)"),
    )


POLICY_FILE = "prism_faithful_reduced_model_policy_v1.json"
POLICY_TEXT = ("A fixed joint flagged in a take is a Large-motion promotion CANDIDATE (not "
               "auto-promoted); the schema stays 18 Large + 4 fixed. Promotion (moving a fixed joint "
               "into Large motion) is a separate schema decision applied cohort-wide if a joint is "
               "flagged in a material fraction of takes.")


def write_reduced_model_policy(out_root: Path, takes: list) -> Path:
    """The bundle-level summary of the per-take fixed-joint promotion flags.

    The registry requires this file at the bundle root (dataset_profiles_v1.yaml, prism
    additive.bundle_files), and this function writes it. Every number here is a count over
    the takes' own manifest.validation.reduced_model_fit_residual_m blocks, so the file cannot
    disagree with the takes beside it.
    """
    threshold = _gen.PROMOTE_TAU_P95_M
    joint_names = _gen.anthro_smpl.JOINT24_NAMES
    downstream = {fixed: [joint_names[k] for k in indices]
                  for fixed, indices in _gen.FIXED_JOINT_DOWNSTREAM.items()}
    per_joint: dict = {}
    per_subject: dict = {}
    pairs = []
    with_candidate = 0
    counted = 0
    for entry in takes:
        if entry.get("status") != "ok":
            continue
        path = out_root / entry["rel"] / "manifest.json"
        try:
            block = json.loads(path.read_text(encoding="utf-8"))["validation"]["reduced_model_fit_residual_m"]
        except (OSError, KeyError, ValueError):
            continue                                    # a take without the block says nothing here
        counted += 1
        threshold = block.get("promotion_threshold_p95_m", threshold)
        promotion = block.get("fixed_joint_promotion") or {}
        flagged_here = False
        subject = entry["rel"].rsplit("-take", 1)[0]
        for joint, verdict in promotion.items():
            if verdict.get("promotion_candidate"):
                flagged_here = True
                per_joint[joint] = per_joint.get(joint, 0) + 1
                # Counted per flagged (take, joint) pair, as the hand-written 2026-08 file did
                # despite its key's name; kept so the regenerated file reads the same.
                per_subject[subject] = per_subject.get(subject, 0) + 1
                pairs.append({"take": entry["rel"], "joint": joint,
                              "p95_m": round(float(verdict["p95_m"]), 4)})
        if flagged_here:
            with_candidate += 1
    policy = {
        "config_id": "prism_faithful_reduced_model_policy_v1",
        "version": "1.0.0",
        "distribution_scope": "internal_only",
        "method": ("reduced_model_fit (spec sec-anthro-fit): weld 4 fixed joints to per-subject "
                   "constant + refit spine3/shoulders; distal orientation exact; position residual QC."),
        "promotion_threshold_p95_m": threshold,
        "fixed_joint_downstream": downstream,
        "takes_total": counted,
        "takes_with_any_candidate": with_candidate,
        "flagged_take_count_per_joint": dict(sorted(per_joint.items())),
        "flagged_take_count_per_subject": dict(sorted(per_subject.items())),
        "flagged_pairs": pairs,
        "policy": POLICY_TEXT,
    }
    target = out_root / POLICY_FILE
    target.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")
    return target


def run(specs, out_root: str, build=None, force: bool = False, fits=None) -> dict:
    """`fits` is the subjects' `CorpusFits`; a take it does not name is fitted on its own."""
    build = build or measured.build_measured
    by_take = fits.by_take if fits is not None else {}
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    for spec in specs:
        out_dir = out_root / faithful_all._take_dirname(spec)
        if not force and faithful_all._already_done(out_dir):
            counts["skipped"] += 1
            results.append({"take": faithful_all._take_dirname(spec), "status": "skipped"})
            continue
        try:
            small, large, dev, anthro, manifest = build(
                spec, reduction=by_take.get(faithful_all._take_dirname(spec)))
            _gen.write_outputs(str(out_dir), small, large, dev, anthro, manifest)
            _write_root_translation(out_dir, spec, manifest, anthro)
            counts["ok"] += 1
            results.append({"take": faithful_all._take_dirname(spec), "status": "ok",
                            "pair_id": str(manifest["identity"]["pair_id"]),
                            "frames": int(small["frame_count"])})
        except Exception as exc:  # per-take isolation
            counts["failed"] += 1
            results.append({"take": faithful_all._take_dirname(spec), "status": "failed",
                            "reason": "%s: %s" % (type(exc).__name__, exc)})
    _write_dataset_description(out_root, counts, len(specs))
    # `rel` is the registry's name for the take directory; `relative_path` stays beside it for the
    # consumers that branched on it (the waiver that excused its absence closed on 2026-09-15).
    takes = [
        {"take_id": r["take"], "pair_id": r.get("pair_id"), "status": r["status"],
         "frames": r.get("frames"), "rel": r["take"], "relative_path": r["take"]}
        for r in results
    ]
    write_reduced_model_policy(out_root, takes)
    if fits is not None:
        fits.write(out_root)
    index = {
        "spec_id": SPEC_ID, "spec_version": SPEC_VERSION,
        "dataset_dirname": DATASET_DIRNAME, "source": "prism",
        "artifact_class": "experimental_non_candidate", "distribution_scope": "internal_only",
        "small_source": "physical_imu_measured_simulation (spec_S_v2 axes, gravity-included)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {"ok": counts["ok"], "failed": counts["failed"],
                   "excluded": counts["skipped"], "total": len(specs)},
        "complete": counts["failed"] == 0,
        "takes": takes,
    }
    (out_root / "INDEX.json").write_text(json.dumps(index, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return index


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None,
                    help=f"read <root>/extracted/prism and write <root>/{paths.POC_DEMO}/"
                         f"{DATASET_DIRNAME} (the old single flag); not combinable with "
                         "--source-root or --out")
    ap.add_argument("--source-root", default=None,
                    help="the PRISM folder itself, holding <subject>/<take>.pkl (subj001/"
                         "take002.pkl, ...) -- not SOMA_SOURCE_ROOT, which holds prism/; "
                         "default $SOMA_SOURCE_ROOT/prism")
    ap.add_argument("--out", default=None,
                    help=f"the bundle directory; default $SOMA_DATA_ROOT/{paths.POC_DEMO}/"
                         f"{DATASET_DIRNAME}")
    ap.add_argument("--only", default=None, help="restrict to one subject id, e.g. prism_subj001")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    return ap


def resolve_locations(args, ap: argparse.ArgumentParser | None = None) -> tuple[Path, Path]:
    """(PRISM source folder, bundle directory) for parsed arguments.

    `--data-root` alone gives the source and bundle locations it always gave; otherwise each side
    comes from its own flag or from `pipeline.paths` (SOMA_SOURCE_ROOT / SOMA_DATA_ROOT). Mixing
    the two forms is a usage error rather than a guess at which one was meant. The body models
    are not a location this resolves: see the module docstring."""
    if args.data_root is not None:
        if args.source_root is not None or args.out is not None:
            message = "--data-root cannot be combined with --source-root or --out"
            if ap is not None:
                ap.error(message)
            raise ValueError(message)
        root = Path(args.data_root)
        return root / "extracted" / "prism", paths.lineage_dir(DATASET_DIRNAME, root=root)
    source = Path(args.source_root) if args.source_root is not None else paths.source_dir("prism")
    out = Path(args.out) if args.out is not None else paths.lineage_dir(DATASET_DIRNAME)
    return source, out


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        source, out = resolve_locations(args, ap)
        # with --data-root the whole <root>/extracted is a source tree, as it is by default
        reads = Path(args.data_root) / "extracted" if args.data_root is not None else source
        paths.body_model_dir()     # the models each subject's gender selects live here
        paths.check_output_dir(out, inputs=[reads])
        paths.refuse_partial_run_into_lineage(
            out, DATASET_DIRNAME, faithful_all.selection_flags(args),
            "--data-root" if args.data_root is not None else "--out")
        given = args.data_root is not None or args.source_root is not None
        paths.existing_source_dir("prism", source if given else None)
        faithful_all.require_subject_folders(
            source, flag="--source-root" if args.source_root is not None else None)
    except paths.PathConfigError as error:
        print(error, flush=True)
        return 2
    if args.data_root is not None:
        specs = enumerate_specs(args.data_root)
    else:
        specs = enumerate_specs(source_dir=source)
    if args.only:
        try:
            specs = faithful_all.only_subject(specs, args.only, source)
        except ValueError as error:
            print(error, flush=True)
            return 2
    if args.limit:
        specs = specs[: args.limit]
    if not specs:
        # run() rewrites INDEX.json, the policy file, the fit record and the description over
        # whatever is selected
        print(faithful_all.no_take_selected(source, args), flush=True)
        return 2
    print("fitting per-subject frozen-joint constants (smpl18.reduce) ...", flush=True)
    fits = compute_subject_constants(specs)
    out_root = str(out)
    index = run(specs, out_root, force=args.force, fits=fits)
    print("INDEX:", json.dumps(index["counts"]), "->", out_root, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
