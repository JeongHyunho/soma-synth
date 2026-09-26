"""Batch runner: synthesize the faithful AMASS small/large/anthro reference for every sequence.

Enumerates all *_poses.npz across the audited AMASS sub-datasets, calls the per-sequence
`generate_amass_faithful.build`, and writes one data-only bundle per sequence under a single
dataset root. Resumable (a sequence whose 4 files already validate is skipped), per-sequence error
isolation (a bad file becomes INDEX status 'failed'+reason; the run continues), and shardable
(`--shard i/N`) so several detached processes can cover the ~8354 sequences in parallel with disjoint
output. `--groups <dataset>/<subject>` (repeatable) restricts the run to whole subjects, and so to
whole fit groups. LOCAL-ONLY: reads `$SOMA_SOURCE_ROOT/amass` and writes under `$SOMA_DATA_ROOT`
unless `--amass-root` / `--out-root` say otherwise; never pushes to any shared drive.

A partial selection (`--datasets`, `--groups`, `--limit`) needs an `--out-root` other than the
production lineage directory, and `--merge` stops before writing when it finds no shard index or
no take in them.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util as _ilu
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

# Sibling per-sequence generator (loaded via importlib, mirroring its own anthro import).
_GEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_amass_faithful.py")
_gen_spec = _ilu.spec_from_file_location("generate_amass_faithful", _GEN_PATH)
gen = _ilu.module_from_spec(_gen_spec)
_gen_spec.loader.exec_module(gen)
reduced_model = gen.reduced_model
paths = gen.paths

DATASET_DIRNAME = "amass_faithful_full"    # no version suffix (retention rule 1.1)
BANNER_TOKEN = "INTERNAL-ONLY"
#: The five files the spec's expected_deliverables asks of every take. The resume predicate
#: checks all five, so a take missing smpl_root_translation.npz is rebuilt rather than resumed.
BUNDLE_FILES = ("small_reference.npz", "large_reference.npz", "anthro_reference.npz",
                "smpl_root_translation.npz", "manifest.json")
_NPZ_FILES = tuple(f for f in BUNDLE_FILES if f.endswith(".npz"))


# --------------------------------------------------------------------------
# enumeration / identity
# --------------------------------------------------------------------------
def parse_group(value):
    """`--groups` value `<dataset>/<subject>`, the two folder levels above a sequence file."""
    parts = value.replace("\\", "/").split("/")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(f"{value!r} is not <dataset>/<subject>")
    return parts[0], parts[1]


def enumerate_specs(amass_root, datasets=None, groups=None):
    """Discover every *.npz under the AMASS sub-datasets -> sorted list of AmassSeqSpec.

    Includes both motion sequences (`*_poses.npz`) and non-motion shape files (`shape.npz`,
    gender+betas only). Non-motion files are enumerated so they are explicitly accounted for in the
    run index (status `excluded_non_motion`) rather than silently dropped; see `is_motion_spec`.

    `groups` is a collection of `(dataset, subject)`: only the files whose folder is that subject
    of that dataset are kept, every one of them, so each fit group (one subject, one body model,
    one set of betas) is enumerated whole. A group that matches no file is an error.
    """
    datasets = list(datasets) if datasets else list(gen.AMASS_DATASETS)
    wanted = set(groups) if groups else None
    if wanted is not None:
        datasets = [ds for ds in datasets if any(ds == name for name, _ in wanted)]
    specs, found = [], set()
    for ds in datasets:
        ds_root = os.path.join(amass_root, ds)
        for p in sorted(glob.glob(os.path.join(ds_root, "**", "*.npz"), recursive=True)):
            sub = os.path.basename(os.path.dirname(p))
            if wanted is not None and (ds, sub) not in wanted:
                continue
            found.add((ds, sub))
            seq = os.path.splitext(os.path.basename(p))[0]
            rel = os.path.relpath(p, amass_root).replace("\\", "/")
            specs.append(gen.AmassSeqSpec(
                dataset=ds, subject_id=f"amass_{ds}_{sub}", sequence_id=seq, source_npz=p,
                relative_path="extracted/amass/" + rel))
    if wanted is not None:
        missing = sorted(wanted - found)
        if missing:
            names = ", ".join(f"{a}/{b}" for a, b in missing)
            raise ValueError(f"no AMASS file under {amass_root} for: {names}")
    return specs


def is_motion_spec(spec):
    """AMASS motion sequences are named `<seq>_poses.npz`; `shape.npz` carries no motion."""
    return spec.sequence_id.endswith("_poses")


def bundle_dirname(spec):
    return gen.sanitize_dirname("%s__%s" % (spec.subject_id, spec.sequence_id))


def bundle_is_valid(out_dir):
    """A bundle is resume-skippable when all five files exist and every npz opens cleanly."""
    if not all(os.path.exists(os.path.join(out_dir, f)) for f in BUNDLE_FILES):
        return False
    try:
        with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as fh:
            json.load(fh)
        for f in _NPZ_FILES:
            # allow_pickle=False: only the zip namelist (.files) is read, never unpickled.
            with np.load(os.path.join(out_dir, f), allow_pickle=False) as z:
                _ = z.files
    except Exception:
        return False
    return True


def root_translation_sidecar(large, anthro, manifest):
    """The smpl_root_translation.npz arrays, from what the build already computed.

    In this lineage large.pelvis_position_world_aux IS the SMPL root translation on the 100 Hz
    grid (the pelvis is anchored at the origin of the rest skeleton), so the sidecar the spec
    requires is a copy of it under the spec's own keys and provenance -- not a second read of the
    source npz, which is what the old enrich pass did after the fact.
    """
    trans = np.asarray(large["pelvis_position_world_aux"], np.float32)
    return {
        "smpl_trans": trans,
        "frame_count": np.int64(trans.shape[0]),
        "pair_id": np.asarray(str(manifest["identity"]["pair_id"])),
        "source_asset_id": np.asarray(str(anthro["source_asset_id"])),
        "smpl_trans_provenance": np.asarray(gen.SMPL_TRANS_PROVENANCE),
    }


def write_outputs_generic(out_dir, small, large, anthro, manifest):
    """Write the five-file bundle (small/large/anthro/root-translation npz + manifest.json)."""
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "small_reference.npz"), **small)
    np.savez(os.path.join(out_dir, "large_reference.npz"), **large)
    np.savez(os.path.join(out_dir, "anthro_reference.npz"), **anthro)
    np.savez(os.path.join(out_dir, "smpl_root_translation.npz"),
             **root_translation_sidecar(large, anthro, manifest))
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def _index_header(out_root):
    """What every INDEX (shard or merged) says about the bundle before its take list."""
    return {
        "spec_id": gen.SPEC_ID, "spec_version": gen.SPEC_VERSION,
        "dataset_dirname": os.path.basename(out_root.rstrip("/\\")),
        "source": "amass",
        "artifact_class": "experimental_non_candidate",
        "distribution_scope": "internal_only",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------
# run / index
# --------------------------------------------------------------------------
def _counts(entries):
    c = {"ok": 0, "failed": 0, "skipped": 0, "excluded_non_motion": 0, "resumed": 0}
    for e in entries:
        c[e["status"]] = c.get(e["status"], 0) + 1
        if e.get("resumed"):
            c["resumed"] += 1
    c["total"] = len(entries)
    return c


def _resumed_entry_fields(out_dir):
    """What the INDEX says about a take that was already on disk and valid.

    A resumed take is an ok take -- it validates and it is registered like any other -- so it is
    recorded as `ok` with the frames and pair_id its own manifest carries, plus `resumed: True` so
    the run record can count what it did not rebuild. Recorded as `skipped`, a resumed take
    would drop out of the validator's and catalog's view.
    """
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    frames = (manifest.get("window") or {}).get("frame_count")
    if frames is None:
        with np.load(os.path.join(out_dir, "small_reference.npz"), allow_pickle=False) as z:
            frames = int(np.asarray(z["frame_count"])) if "frame_count" in z.files else None
    return {"status": "ok", "resumed": True, "frames": frames,
            "pair_id": (manifest.get("identity") or {}).get("pair_id")}


# --------------------------------------------------------------------------
# frozen-joint constants: one fit per group, shared by every shard
# --------------------------------------------------------------------------
#: Held while one process fits the groups; the others wait for FIT_FILE instead of fitting too.
FIT_LOCK = reduced_model.FIT_FILE + ".lock"
#: How often a waiting shard looks for the fit file, in seconds. Scheduling only; no data depends on it.
FIT_WAIT_POLL_S = 10


def fit_all_groups(specs, log=None):
    """Fit every group's constants: (settings, {group: record}, {take rel: group}, {take rel: reason}).

    Groups are found from two small fields per sequence, then each group's poses are loaded, pooled
    and fitted and dropped before the next, so the whole corpus is never in memory at once."""
    log = log or (lambda *a: None)
    settings = reduced_model.settings_for_source("amass")
    members, meta, unfitted = {}, {}, {}
    for spec in specs:
        if not is_motion_spec(spec):
            continue
        try:
            group, letter, betas = gen.fit_group_of(spec)
        except Exception as error:  # the take fails in build too; say why it pooled nothing
            unfitted[bundle_dirname(spec)] = "%s: %s" % (type(error).__name__, error)
            continue
        members.setdefault(group, []).append(spec)
        meta[group] = (letter, betas)
    models, groups, takes = {}, {}, {}
    for count, group in enumerate(sorted(members), 1):
        letter, betas = meta[group]
        poses, used = [], []
        for spec in members[group]:
            try:
                data = gen.load_amass_safe(spec.source_npz, dataset=spec.dataset)
            except Exception as error:
                unfitted[bundle_dirname(spec)] = "%s: %s" % (type(error).__name__, error)
                continue
            poses.append(gen.build_pose24(data["poses"]))
            used.append(bundle_dirname(spec))
        if not poses:
            continue
        if letter not in models:
            models[letter] = gen.anthro_smpl.load_smpl_model_for_gender(letter)
        rest = gen.anthro_smpl.rest_joints(models[letter], betas)
        fit = reduced_model.fit_subject(
            poses, rest, settings, group=group,
            basis="every loadable motion sequence of the group, native rate")
        groups[group] = fit.record
        takes.update({rel: group for rel in used})
        if count % 50 == 0:
            log("fitted", count, "of", len(members), "groups")
    return settings, groups, takes, unfitted


def _fits_by_take(document):
    fits = {name: reduced_model.fit_from_record(record) for name, record in document["groups"].items()}
    return {rel: fits[group] for rel, group in document["takes"].items()}


def _fit_file_covers(document, specs, settings):
    wanted = {bundle_dirname(s) for s in specs if is_motion_spec(s)}
    known = set(document["takes"]) | set(document.get("unfitted", {}))
    return document.get("settings") == dict(settings.provenance) and wanted <= known


def ensure_fits(specs, out_root, log=None):
    """{take rel: SubjectFit}, from FIT_FILE when it already covers `specs`, else fitted now.

    Sharded runs start together; the first to create FIT_LOCK fits, the rest wait for its file.
    A lock left by a crashed run has to be removed by hand -- the waiting shards say which file."""
    log = log or (lambda *a: None)
    os.makedirs(out_root, exist_ok=True)
    settings = reduced_model.settings_for_source("amass")
    existing = reduced_model.read_fit_file(out_root)
    if existing is not None and _fit_file_covers(existing, specs, settings):
        return _fits_by_take(existing)
    lock = os.path.join(out_root, FIT_LOCK)
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        log("waiting for the fit file another process is writing (lock %s)" % lock)
        while os.path.exists(lock):
            time.sleep(FIT_WAIT_POLL_S)
        existing = reduced_model.read_fit_file(out_root)
        if existing is not None and _fit_file_covers(existing, specs, settings):
            return _fits_by_take(existing)
        raise RuntimeError("the fit lock was released without a fit file covering this run: %s" % lock)
    try:
        os.close(handle)
        log("fitting frozen-joint constants per subject group")
        settings, groups, takes, unfitted = fit_all_groups(specs, log)
        reduced_model.write_fit_file(out_root, settings, groups, takes, unfitted)
        log("fit file written:", len(groups), "groups,", len(takes), "takes,", len(unfitted), "unfitted")
    finally:
        os.remove(lock)
    return _fits_by_take(reduced_model.read_fit_file(out_root))


def _index_path(out_root, shard):
    if shard is None:
        return os.path.join(out_root, "INDEX.json")
    return os.path.join(out_root, "INDEX_shard_%d_of_%d.json" % (shard[0], shard[1]))


def run(specs, out_root, build_fn=None, write_fn=None, force=False, shard=None, log=None, fits=None):
    """Generate bundles for `specs` (optionally only this shard). Returns the index dict and writes it.

    `fits` maps a take rel to its group's `SubjectFit` (see `ensure_fits`); a take with none is
    built with a fit of its own, which its records call `take` scope."""
    build_fn = build_fn or gen.build
    write_fn = write_fn or write_outputs_generic
    log = log or (lambda *a: None)
    os.makedirs(out_root, exist_ok=True)
    entries = []
    for i, spec in enumerate(specs):
        if shard is not None and (i % shard[1]) != shard[0]:
            continue
        out_dir = os.path.join(out_root, bundle_dirname(spec))
        rel = os.path.relpath(out_dir, out_root).replace("\\", "/")
        # `take_id`, `rel` and `relative_path` all name the take DIRECTORY, which is what the
        # registry's index_take_entry_keys and the validator's read_index expect; the source
        # npz path this entry used to carry under `relative_path` lives on as
        # `source_relative_path`, so nothing downstream mistakes a source file for a take.
        entry = {"take_id": rel, "rel": rel, "relative_path": rel,
                 "dataset": spec.dataset, "subject_id": spec.subject_id, "sequence_id": spec.sequence_id,
                 "source_relative_path": spec.relative_path}
        if not is_motion_spec(spec):
            entry["status"] = "excluded_non_motion"
            entry["reason"] = "shape-only npz (SMPL subject shape: gender+betas, no motion) — nothing to synthesize"
            entries.append(entry)
            continue
        if not force and bundle_is_valid(out_dir):
            entry.update(_resumed_entry_fields(out_dir))
            entries.append(entry)
            continue
        try:
            if fits is None:
                small, large, anthro, manifest = build_fn(spec)
            else:
                small, large, anthro, manifest = build_fn(spec, reduction=fits.get(rel))
            write_fn(out_dir, small, large, anthro, manifest)
            entry["status"] = "ok"
            entry["frames"] = int(small["frame_count"])
            entry["pair_id"] = manifest.get("identity", {}).get("pair_id")
        except Exception as e:  # per-sequence isolation: log + record, keep going
            entry["status"] = "failed"
            entry["reason"] = "%s: %s" % (type(e).__name__, e)
            log("FAILED", spec, entry["reason"])
        entries.append(entry)
        if (len(entries) % 100) == 0:
            log("progress", len(entries), _counts(entries))
    index = dict(_index_header(out_root))
    index.update({"shard": list(shard) if shard else None, "counts": _counts(entries), "takes": entries})
    with open(_index_path(out_root, shard), "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)
    return index


def merge_shards(out_root, total_specs=None):
    """Combine INDEX_shard_*.json into a single INDEX.json (dedup by take, last write wins).

    Raises ValueError, before INDEX.json is written, when `out_root` holds no shard index or its
    shard indexes list no take: the merge would otherwise replace the bundle's INDEX.json with one
    over zero takes, and main() its description too."""
    shard_paths = sorted(glob.glob(os.path.join(out_root, "INDEX_shard_*_of_*.json")))
    if not shard_paths:
        raise ValueError(f"no shard index (INDEX_shard_<i>_of_<N>.json) in {out_root}; nothing "
                         "merged and nothing written. Run the shards (--shard i/N) into this "
                         "--out-root before --merge")
    merged = {}
    for p in shard_paths:
        with open(p, encoding="utf-8") as fh:
            shard = json.load(fh)
        for e in shard["takes"]:
            merged[e["rel"]] = e
    if not merged:
        names = ", ".join(os.path.basename(p) for p in shard_paths)
        raise ValueError(f"the shard indexes in {out_root} ({names}) list no take; nothing merged "
                         "and nothing written")
    entries = sorted(merged.values(), key=lambda e: e["rel"])
    index = dict(_index_header(out_root))
    index.update({"shard": None, "counts": _counts(entries), "takes": entries})
    if total_specs is not None:
        index["expected_total"] = int(total_specs)
        index["complete"] = bool(len(entries) == int(total_specs))
    with open(os.path.join(out_root, "INDEX.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)
    return index


def write_dataset_description(out_root, index):
    counts = index.get("counts", {})
    md = f"""# AMASS faithful synthetic reference (small / large / anthro) — {DATASET_DIRNAME}

**{BANNER_TOKEN}. experimental_non_candidate.** This is an independent replica produced by the PRISM
faithful recipe applied to AMASS; it is NOT the canonical governed artifact, NOT contract-compliant,
and NOT a quality-gate PASS. **INTERNAL-ONLY** — do not upload to any public Git / cloud / model or
dataset hub or public link; not a training/evaluation label. The AMASS field registry is
`activation_state: non_authorizing` — the source audit is not a generation authorization.

## What this is
Per AMASS sequence, a paired **small** (synthetic IMU) / **large** (motion) / **anthro** (skeleton
constants) bundle, plus a per-sequence `manifest.json`. AMASS ships only SMPL-H pose parameters, so
**every IMU channel is synthesized from the SMPL body via forward kinematics** — there is no measured
IMU, and there is **no development_reference** (AMASS has no measured GRF / insole / kinetics).

## Bundle files
- `small_reference.npz` — 8-channel synthetic IMU (`back_T4`, `wrist_l/r`, `shank_l/r`, `occiput`,
  `foot_l/r`): `imu_orientation[M,8,4]` in the **spec sensor frame S** (`small/00_sensors.qmd`: +Y=proximal,
  +Z=outward lateral, +X=Y x Z; left limbs Rot(Y,180)), `imu_acceleration[M,8,3]` specific force
  `f = R^T (a_world - g)` (gravity-included), `imu_angular_velocity[M,8,3]` deg/s, validity/confidence, and
  `imu_orientation_absolute_heading` (`[True x6, False x2]`; insole channels are 6-axis).
  `small_mode='synthetic_from_smpl'`, `axis_convention='spec_S_v2'`.
- `large_reference.npz` — `joint_names(18)` + `root_velocity[M,3]` + `joint_rotation[M,18,4]`
  (`[:,0]`=pelvis global, `[:,1:]`=17 reduced-model local rotations) + `joint_velocity[M,18,3]` +
  `pelvis_position_world_aux[M,3]` + `smpl_global_orientation_world[M,24,4]` (lossless).
  Trajectory: `pos[t] = pelvis_position_world_aux[0] + dt*cumsum(root_velocity)[t]`.
- `anthro_reference.npz` — subject skeleton constants: `joint_position[2,22,3]`, `segment_length[13]`,
  `fixed_joint_rotation[4,4]` (spine1/spine2/collars weld), `betas[16]`, `gender`. `height` is an SMPL
  rest-stature estimate; `body_mass` is `NaN` (AMASS has no measured subject anthropometry).
- `manifest.json` — provenance, resample block, validation, and the unavailable list.

## Conventions
Frame = AMASS world (empirically **Z-up** for every sub-dataset) = spec frame G; quaternions `(w,x,y,z)`;
gravity `g=[0,0,-9.80665]`; angular velocity rotation-log deg/s. Source framerate is **resampled to 100 Hz**
(per-joint Slerp on local rotations + linear interp on trans; a no-op for KIT @100 Hz).

## Known limits
- **Synthetic IMU**: orientation/acceleration/gyro are derived from the SMPL model, not measured. The
  orientation is placed in the spec sensor frame S (a constant anatomical relabel from SMPL rest
  geometry); the physical segment→sensor lever arm is still not modelled.
- **10-beta truncation**: AMASS provides 16 shape betas; the clean SMPL model carries 10, so rest
  geometry uses the first 10 (`smpl10_beta_truncation`).
- **No kinetics**: no GRF / CoP / joint torques / inverse dynamics; no development_reference.
- **No measured height/mass**: height is an SMPL stature estimate; mass is unavailable.
- Reduced model: `spine1`, `spine2` and both collars are frozen to constants that `smpl18.reduce`
  fits **once per subject group** (a subject's sequences that share MoSh betas and body model) over
  their pooled frames; distal global orientation is exact and a small trunk/arm position residual
  remains. `reduced_model_fit.json` at the bundle root records every group's constants, residuals and
  settings; each sequence reports its own residual in `manifest.validation.reduced_model_fit_residual_m`.
- **Loading**: every name/code array is a numpy unicode array, so `np.load(path, allow_pickle=False)`
  reads all of them.
- Very short sequences (<=16 resampled frames) skip the 7 Hz low-pass before differentiation (too short
  to pad-filter); their acceleration is the raw double difference.

## Coverage
Counts (this run): ok={counts.get('ok', 0)}, failed={counts.get('failed', 0)}, skipped={counts.get('skipped', 0)},
excluded_non_motion={counts.get('excluded_non_motion', 0)}, total={counts.get('total', 0)}.
`excluded_non_motion` are AMASS `shape.npz` subject-shape files (gender+betas, no motion) — accounted for,
not silently dropped. Per-sequence status and any failure reasons are in `INDEX.json`.
"""
    # One English description per bundle, under the name the distribution guide and the L0/L4
    # checks both require (ADR-0040 D3). This used to be DATASET_DESCRIPTION_EN.md, a second file
    # alongside DATA_DESCRIPTION_EN.md whose separate purpose stopped being legible: the companion
    # grew longer than the document it was meant to summarise.
    with open(os.path.join(out_root, "DATA_DESCRIPTION_EN.md"), "w", encoding="utf-8") as fh:
        fh.write(md)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_shard(s):
    if not s:
        return None
    i, n = s.split("/")
    return (int(i), int(n))


def partial_selection(args):
    """The selection flags that restrict a run to part of the corpus, as given on the command line.

    `--shard` is not one: the shards of a run together cover the selection, and `--merge` joins them."""
    chosen = []
    if args.datasets:
        chosen.append("--datasets " + " ".join(args.datasets))
    chosen.extend(f"--groups {dataset}/{subject}" for dataset, subject in args.groups or ())
    if args.limit is not None:
        chosen.append(f"--limit {args.limit}")
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser(description="Batch-synthesize the faithful AMASS reference (local-only).")
    ap.add_argument("--amass-root", default=None,
                    help="the AMASS source folder; default $SOMA_SOURCE_ROOT/amass")
    ap.add_argument("--out-root", default=None,
                    help=f"the bundle directory; default $SOMA_DATA_ROOT/{paths.POC_DEMO}/"
                         f"{DATASET_DIRNAME}")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--groups", action="append", type=parse_group, default=None,
                    metavar="DATASET/SUBJECT",
                    help="only this subject's files, i.e. its whole fit groups (repeatable)")
    ap.add_argument("--shard", default=None, help="i/N — process specs where index%%N==i")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap to the first K specs (sampling); not with --groups, whose fit groups "
                         "it would cut")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--merge", action="store_true", help="merge shard indexes into INDEX.json, then describe")
    args = ap.parse_args(argv)
    if args.groups and args.limit is not None:
        # The frozen-joint constants are fitted over a group's pooled frames, so a group cut
        # partway would get other constants than the full run gives it, and say nothing.
        print("--limit cannot be combined with --groups: it would cut a fit group partway and fit "
              "its frozen joints on part of its frames; name fewer groups instead")
        return 2

    try:
        given = args.amass_root is not None
        if not given:
            args.amass_root = str(paths.source_dir("amass"))
        if args.out_root is None:
            args.out_root = str(paths.lineage_dir(DATASET_DIRNAME))
        paths.check_output_dir(args.out_root, inputs=[args.amass_root])
        # A partial selection needs an explicit --out-root outside the production lineage.
        paths.refuse_partial_run_into_lineage(args.out_root, DATASET_DIRNAME,
                                              partial_selection(args), "--out-root")
        if not args.merge:
            paths.body_model_dir()     # ensure_fits loads the models before any take is written
        paths.existing_source_dir("amass", args.amass_root if given else None)
        specs = enumerate_specs(args.amass_root, args.datasets, args.groups)
    except (paths.PathConfigError, ValueError) as error:
        print(error)
        return 2
    if args.limit is not None:
        specs = specs[: args.limit]
    if not specs:
        # Both branches below rewrite INDEX.json and the description (and the fits their fit
        # record) in the output, the production lineage by default, over whatever they selected.
        selection = "datasets: " + ", ".join(args.datasets or gen.AMASS_DATASETS)
        if args.limit is not None:
            selection += f"; --limit {args.limit}"
        print(f"no AMASS file selected under {args.amass_root} ({selection}); nothing written. "
              "--amass-root is the AMASS folder itself (<dataset>/<subject>/*.npz), not the "
              "source root that holds it")
        return 2
    total = len(specs)

    if args.merge:
        try:
            idx = merge_shards(args.out_root, total_specs=total)
        except ValueError as error:     # no shard index, or none lists a take: nothing written
            print(error)
            return 2
        write_dataset_description(args.out_root, idx)
        print("MERGED", idx["counts"], "complete=", idx.get("complete"))
        return 0

    shard = _parse_shard(args.shard)
    fits = ensure_fits(specs, args.out_root, log=lambda *a: print(*a, flush=True))
    idx = run(specs, args.out_root, force=args.force, shard=shard, log=print, fits=fits)
    if shard is None:
        # An un-sharded run is the whole bundle, so it writes the description a sharded run
        # leaves to --merge; without this the bundle lacked the file L0 and L4 require.
        write_dataset_description(args.out_root, idx)
    print("DONE shard=%s" % (shard,), idx["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
