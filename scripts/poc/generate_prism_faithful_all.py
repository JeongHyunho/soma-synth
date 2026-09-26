"""Batch: generate faithful small/large/anthro/development for every PRISM take.

Data-only (no mp4/figures/zip). Resumable, per-take error isolation, INDEX.json coverage.
INTERNAL-ONLY, experimental_non_candidate. Local generation; no push.

    python scripts/poc/generate_prism_faithful_all.py \
        [--data-root <root, default $SOMA_DATA_ROOT>] \
        [--only prism_subj001] [--limit N] [--force]

`--only` / `--limit` select part of the takes, so they are refused when the bundle would be the
production lineage directory under `$SOMA_DATA_ROOT`: give them a scratch `--data-root`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

_GEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_prism_faithful.py")
_spec = importlib.util.spec_from_file_location("generate_prism_faithful", _GEN_PATH)
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)
TakeSpec = _gen.TakeSpec
paths = _gen.paths

DATASET_DIRNAME = "prism_faithful_full"    # no version suffix (retention rule 1.1)
CORE_FILES = ("small_reference.npz", "large_reference.npz", "anthro_reference.npz",
              "development_reference.npz", "manifest.json")


def _prism_dir(data_root, source_dir) -> Path:
    """The PRISM source folder: `source_dir` itself, else `<data_root>/extracted/prism`."""
    if source_dir is not None:
        return Path(source_dir)
    if data_root is None:
        raise ValueError("pass data_root or source_dir")
    return Path(data_root) / "extracted" / "prism"


def _gender_by_subject(data_root: str | None = None, *, source_dir=None) -> dict:
    """Read info.subj_info.gender from each subject's first take (restricted unpickler)."""
    out = {}
    root = _prism_dir(data_root, source_dir)
    for subj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        takes = sorted(subj_dir.glob("*.pkl"))
        if not takes:
            continue
        raw = _gen.load_prism_safe(str(takes[0]))
        out["prism_" + subj_dir.name] = str(raw["info"]["subj_info"]["gender"])
    return out


def enumerate_specs(data_root: str | None = None, gender_by_subject: dict | None = None, *,
                    source_dir=None):
    """Every take under the PRISM source folder: `source_dir` when given, else
    `<data_root>/extracted/prism`. `relative_path` is the logical id `extracted/prism/...` either way."""
    root = _prism_dir(data_root, source_dir)
    genders = (gender_by_subject if gender_by_subject is not None
               else _gender_by_subject(data_root, source_dir=source_dir))
    specs = []
    for subj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        subject_id = "prism_" + subj_dir.name
        for pkl in sorted(subj_dir.glob("*.pkl")):
            take_id = pkl.stem
            specs.append(TakeSpec(
                subject_id=subject_id, take_id=take_id, source_pkl=str(pkl),
                gender=genders.get(subject_id, "M"), window=None,
                relative_path="extracted/prism/%s/%s.pkl" % (subj_dir.name, take_id),
            ))
    return specs


def _take_dirname(spec) -> str:
    return "%s-%s" % (spec.subject_id.replace("prism_", "prism-"), spec.take_id)


def _already_done(out_dir: Path) -> bool:
    return all((out_dir / f).exists() for f in CORE_FILES)


def _write_dataset_description(out_root: Path, counts: dict, total: int) -> None:
    md = f"""# PRISM faithful full dataset ({DATASET_DIRNAME})

> **INTERNAL-ONLY.** Do not upload to public Git / cloud / model or dataset hubs.
> **experimental_non_candidate** — physically-faithful reference replica, NOT contract-compliant,
> NOT quality-gated. Not a training/evaluation label. PRISM_SOURCE_PASS does not authorize this;
> it remains a non-candidate reference.

One subdirectory per take: `prism-<subj>-<take>/` with `small_reference.npz`,
`large_reference.npz`, `anthro_reference.npz`, `development_reference.npz`, `manifest.json`.
Per-take field/shape/convention truth is the take's `manifest.json`, which documents the field
layout in full (Small 8-ch IMU, Large motion, Anthro skeleton, insole development reference).

Coverage: {counts['ok']} ok / {counts['skipped']} skipped / {counts['failed']} failed of {total}
(see `INDEX.json`). Full take length at native 100 Hz.

Provenance highlights (per `prism_usage_policy_v1`): Small IMU from `imu_gt` (model-derived,
permitted for all sites); pelvis trajectory from `imu_gt.Pelvis.pos_world` (trans-only world);
anthro `height`/`body_mass` in SOMA spec units m/kg (PRISM source units undeclared upstream);
`arm_length` excluded; insole `contacts` normalized to bool.

Splitting for ML is downstream and carries the mandatory disclosure: the official PRISM split is
chunk-level and 126/150 takes contribute to both sides, so test scores are not an across-take or
across-subject generalisation bound (n=6 subjects).
"""
    # ADR-0040 D3: one English description per bundle, under the name the distribution guide and
    # the L0/L4 checks both require.
    (out_root / "DATA_DESCRIPTION_EN.md").write_text(md, encoding="utf-8")


def compute_subject_constants(specs):
    """Each subject's frozen-joint constants, fitted once by smpl18.reduce over the pooled frames of
    all of its takes and reused for every one of them (reduced_model.CorpusFits, keyed by take
    directory). Takes posed on different skeletons are fitted apart."""
    reduced_model = _gen.reduced_model
    fits = reduced_model.CorpusFits(reduced_model.settings_for_source("prism"))
    by_subj: dict = {}
    for s in specs:
        by_subj.setdefault(s.subject_id, []).append(s)
    for subject_id, subj_specs in by_subj.items():
        members = [(_take_dirname(s), *_gen.subject_fit_inputs(s)) for s in subj_specs]
        fits.fit(subject_id, members, basis="every take of the subject, whole take")
    return fits


def run(specs, out_root: str, build=None, force: bool = False, fits=None) -> dict:
    """`fits` is the subjects' `CorpusFits`; a take it does not name is fitted on its own."""
    build = build or _gen.build
    by_take = fits.by_take if fits is not None else {}
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    for spec in specs:
        out_dir = out_root / _take_dirname(spec)
        if not force and _already_done(out_dir):
            counts["skipped"] += 1
            results.append({"take": _take_dirname(spec), "status": "skipped"})
            continue
        try:
            small, large, dev, anthro, manifest = build(
                spec, reduction=by_take.get(_take_dirname(spec)))
            _gen.write_outputs(str(out_dir), small, large, dev, anthro, manifest)
            counts["ok"] += 1
            results.append({"take": _take_dirname(spec), "status": "ok",
                            "pair_id": str(manifest["identity"]["pair_id"]),
                            "frames": int(small["frame_count"])})
        except Exception as exc:  # per-take isolation
            counts["failed"] += 1
            results.append({"take": _take_dirname(spec), "status": "failed",
                            "reason": "%s: %s" % (type(exc).__name__, exc)})
    _write_dataset_description(out_root, counts, len(specs))
    if fits is not None:
        fits.write(out_root)
    index = {"dataset": DATASET_DIRNAME, "distribution_scope": "internal_only",
             "artifact_class": "experimental_non_candidate",
             "total_specs": len(specs), "counts": counts, "takes": results}
    (out_root / "INDEX.json").write_text(json.dumps(index, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return index


def require_subject_folders(root, flag: str | None = None) -> None:
    """Refuse a PRISM source folder in which no subject folder holds a *.pkl take.

    `enumerate_specs` would select nothing there, and `run` still rewrites the bundle's INDEX.json,
    fit record and description over zero takes. The likely cause is a source root (the folder
    holding prism/ beside the other sources) given where the PRISM folder itself is meant, so when
    the folder came from `flag` the refusal names the folder that was probably meant."""
    root = Path(root)
    if any(p.is_dir() and any(p.glob("*.pkl")) for p in root.iterdir()):
        return
    message = (f"the PRISM source folder {root} holds no subject folder with *.pkl takes "
               "(<subject>/<take>.pkl); nothing written")
    if flag is not None:
        message += (f". {flag} is the PRISM folder itself, not SOMA_SOURCE_ROOT: did you mean "
                    f"{flag} {root / 'prism'}?")
    raise paths.PathConfigError(message)


def selection_flags(args) -> list:
    """`--only` / `--limit` as given: the flags that restrict a run to part of the takes."""
    return [f"--{name} {value}" for name, value in (("only", args.only), ("limit", args.limit))
            if value is not None]


def no_take_selected(source, args) -> str:
    """What a main prints, and why it stops, when its selection is empty."""
    chosen = selection_flags(args)
    return (f"no PRISM take selected under {source}"
            + (f" by {' '.join(chosen)}" if chosen else "") + "; nothing written")


def only_subject(specs, subject_id: str, where) -> list:
    """The specs of one subject. A subject the source does not hold is an error: an empty
    selection would still rewrite the bundle's INDEX.json, policy file and description with zero
    takes."""
    chosen = [s for s in specs if s.subject_id == subject_id]
    if not chosen:
        known = ", ".join(sorted({s.subject_id for s in specs})) or "none"
        raise ValueError(f"--only {subject_id}: no PRISM take of that subject under {where} "
                         f"(subjects there: {known})")
    return chosen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None,
                    help=f"reads <root>/extracted/prism, writes <root>/{paths.POC_DEMO}/"
                         f"{DATASET_DIRNAME}; default: the PRISM folder under $SOMA_SOURCE_ROOT "
                         "and the bundle under $SOMA_DATA_ROOT")
    ap.add_argument("--only", default=None, help="restrict to one subject id, e.g. prism_subj001")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.data_root is not None:
            source = Path(args.data_root) / "extracted" / "prism"
            reads = Path(args.data_root) / "extracted"
            out_root = str(paths.lineage_dir(DATASET_DIRNAME, root=args.data_root))
        else:
            source = reads = paths.source_dir("prism")
            out_root = str(paths.lineage_dir(DATASET_DIRNAME))
        paths.body_model_dir()     # the models each subject's gender selects live here
        paths.check_output_dir(out_root, inputs=[reads])
        paths.refuse_partial_run_into_lineage(
            out_root, DATASET_DIRNAME, selection_flags(args), "--data-root",
            note="holding extracted/prism (the root is read as well as written; "
                 "generate_prism_measured_all.py --out moves the bundle alone)")
        paths.existing_source_dir("prism", source if args.data_root is not None else None)
        require_subject_folders(source)
    except paths.PathConfigError as error:
        print(error)
        return 2
    if args.data_root is not None:
        specs = enumerate_specs(args.data_root)
    else:
        specs = enumerate_specs(source_dir=source)
    if args.only:
        try:
            specs = only_subject(specs, args.only, source)
        except ValueError as error:
            print(error)
            return 2
    if args.limit:
        specs = specs[: args.limit]
    if not specs:
        # run() rewrites INDEX.json, the fit record and the description over whatever is selected
        print(no_take_selected(source, args))
        return 2
    print("fitting per-subject frozen-joint constants (smpl18.reduce) ...")
    fits = compute_subject_constants(specs)
    index = run(specs, out_root, force=args.force, fits=fits)
    print("INDEX:", json.dumps(index["counts"]), "->", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
