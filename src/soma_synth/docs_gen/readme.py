"""Render a README.md for a dataset (field-by-field) and a top-level README across datasets.

Structure comes from the spec + live npz introspection (header-only, no pickle); one-line semantics come
from field_docs. The output is deterministic, LF, ASCII-safe, and carries the INTERNAL-ONLY banner.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from soma_synth.contracts import field_docs_v1 as fd
from soma_synth.contracts import qmd_unified8_smpl18_spec as spec
from soma_synth.validation import checks, npz_io

_BANNER = "> **INTERNAL-ONLY.** Do not upload to public git / cloud / model or dataset hubs. experimental_non_candidate."
_CONVENTIONS = (
    "- World frame: right-handed, **Z-up**, gravity along -Z.\n"
    "- Quaternions are **wxyz**, unit norm.\n"
    "- `imu_acceleration` is **specific force including gravity** (~9.8 m/s^2 at rest), in the sensor frame.\n"
    "- Angular velocity is **deg/s**; positions/lengths **m**; time **s** on a **100 Hz** grid (T = per-take frames).\n"
    "- Sensor frame: **spec_S_v2** (+Y=proximal, +Z=outward-lateral, +X=Y x Z; left limbs 180 deg about Y).\n"
    "- Orientation is a rigid relabel of the SMPL bone frame at each site's joint (see indexing below) for "
    "`synthetic_from_smpl` datasets; for `measured_physical` it is the worn sensor's, and for "
    "`synthetic_from_markers` it is a rigid relabel of the body fitted to that site's marker cluster, so it "
    "departs from the SMPL joint by the soft-tissue and fit difference the manifest records.\n"
    "- `imu_valid_mask` governs the three IMU arrays cell by cell: where it is False they are NaN and "
    "`imu_confidence` is 0.0; where True they are finite (ADR-0039). Apply it before any filter, derivative "
    "or loss. Datasets emitted before 2026-09-06 carry an all-True mask.\n"
)


def _shape_str(axes: tuple[str, ...]) -> str:
    if not axes:
        return "()"
    return "(" + ", ".join(axes) + ("," if len(axes) == 1 else "") + ")"


def _count(axes: tuple[str, ...]) -> str:
    fixed = [a for a in axes if a not in ("T", "N")]
    if not fixed:
        return "N" if "N" in axes else "1"
    return "x".join(fixed)


def _class_of(artifact: str, field: str) -> str:
    art = spec.ARTIFACTS[artifact]
    if field in {k.name for k in art.core}:
        return "CORE"
    for cap, keys in art.conditional.items():
        if field in {k.name for k in keys}:
            return cap
    return "?"


def _ordered_fields(artifact: str, caps: frozenset[str]):
    art = spec.ARTIFACTS[artifact]
    keys = list(art.core)
    for cap, cap_keys in art.conditional.items():
        if cap in caps:
            keys.extend(cap_keys)
    return keys


def _load_names(npz_path: Path, key: str) -> list[str]:
    try:
        # allow_pickle stays False: since ADR-0040 D2 an object array is a validation FAIL, so a
        # bundle that needs pickle to be read is a bundle this README must not describe as sound.
        z = np.load(npz_path, allow_pickle=False)
        if key in z.files:
            return [str(x) for x in z[key]]
    except Exception:
        pass
    return []


def _profile(dataset_dir) -> dict:
    dataset_dir = Path(dataset_dir)
    idx = checks.read_index(dataset_dir)
    raw = idx["raw"]
    ok = [t for t in idx["takes"] if t["status"] == "ok"]
    take_dir = dataset_dir / ok[0]["rel"] if ok else None
    small_mode, source, caps = None, "", frozenset()
    if take_dir is not None:
        small_mode, source, caps, _f = checks.take_capabilities(take_dir)
    present = [
        a for a, asp in spec.ARTIFACTS.items()
        if (asp.capability is None or asp.capability in caps)
        and take_dir is not None and (take_dir / asp.filename).exists()
    ]
    return {
        "dir": dataset_dir,
        "name": dataset_dir.name,
        "spec_id": raw.get("spec_id", spec.SPEC_ID),
        "spec_version": raw.get("spec_version", spec.SPEC_VERSION),
        "source": source or raw.get("source", ""),
        "small_mode": small_mode,
        "caps": caps,
        "take_count": len(ok),
        "take_dir": take_dir,
        "present": present,
    }


def render_dataset_readme(dataset_dir) -> str:
    p = _profile(dataset_dir)
    td = p["take_dir"]
    lines: list[str] = []
    lines.append(f"# {p['name']} — data reference")
    lines.append("")
    lines.append(_BANNER)
    lines.append("")
    lines.append(
        f"- `spec_id`: **{p['spec_id']}** · `spec_version`: **{p['spec_version']}** · source: **{p['source']}**"
    )
    lines.append(f"- takes (ok): **{p['take_count']}** · `small_mode`: **{p['small_mode']}**")
    lines.append("")
    lines.append("## Loading")
    lines.append("")
    lines.append("```python")
    lines.append("import numpy as np")
    lines.append("z = np.load('<take>/small_reference.npz', allow_pickle=False)")
    lines.append("# every string array is numpy 'U' (fixed-width unicode); nothing in this bundle needs pickle")
    lines.append("```")
    lines.append("")
    lines.append("One subdirectory per take; each take holds: "
                 + ", ".join(f"`{spec.ARTIFACTS[a].filename}`" for a in p["present"]) + ", `manifest.json`.")
    lines.append("")
    lines.append("The bundle root holds `INDEX.json` (select takes by `status == \"ok\"`), "
                 "`KNOWN_LIMITATIONS.json`, and `reduced_model_fit.json`: the four frozen-joint "
                 "constants (spine1, spine2, both collars) that `smpl18.reduce` fitted once per "
                 "subject, the residual they leave, and which fit each take carries.")
    lines.append("")
    lines.append("## Conventions")
    lines.append("")
    lines.append(_CONVENTIONS)

    machine: dict[str, dict] = {}
    for art in p["present"]:
        asp = spec.ARTIFACTS[art]
        schema = npz_io.read_npz_schema(td / asp.filename)
        lines.append(f"## `{asp.filename}` — {fd.ARTIFACT_DOCS[art]}")
        lines.append("")
        lines.append("| field | dtype | shape | count | class | meaning |")
        lines.append("|---|---|---|---|---|---|")
        machine[art] = {}
        for k in _ordered_fields(art, p["caps"]):
            live_dtype = str(schema.get(k.name, (None, "?"))[1])
            if live_dtype == "object":
                live_dtype = "object!"  # ! = a D2 violation; the validator fails this bundle
            cls = _class_of(art, k.name)
            summ = fd.summary(art, k.name)
            lines.append(f"| `{k.name}` | {live_dtype} | {_shape_str(k.axes)} | {_count(k.axes)} | {cls} | {summ} |")
            machine[art][k.name] = {"dtype": live_dtype, "shape": list(k.axes),
                                    "count": _count(k.axes), "class": cls, "summary": summ}
        lines.append("")

    # indexing keys
    lines.append("## Indexing")
    lines.append("")
    lines.append("8 IMU sites, in `sensor_codes` order, mapped to their SMPL joint index:")
    lines.append("")
    lines.append("| idx | site | SMPL joint |")
    lines.append("|---|---|---|")
    for i, site in enumerate(spec.SITE_ORDER):
        lines.append(f"| {i} | `{site}` | {spec.SITE_TO_JOINT[site]} |")
    lines.append("")
    jn = _load_names(td / "large_reference.npz", "joint_names")
    an = _load_names(td / "anthro_reference.npz", "joint_names")
    sn = _load_names(td / "anthro_reference.npz", "segment_names")
    if jn:
        lines.append(f"- `large.joint_names` (18): {', '.join(jn)}")
    if an:
        lines.append(f"- `anthro.joint_names` (22): {', '.join(an)}")
    if sn:
        lines.append(f"- `anthro.segment_names` (13): {', '.join(sn)}")
    lines.append("")

    # enums
    lines.append("## Enums")
    lines.append("")
    for enum, vals in fd.ENUM_DOCS.items():
        lines.append(f"- **{enum}**: " + "; ".join(f"`{v}` = {t}" for v, t in vals.items()))
    lines.append("")

    # machine-parseable schema
    lines.append("## Machine-readable schema")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({"spec_id": p["spec_id"], "spec_version": p["spec_version"],
                             "source": p["source"], "artifacts": machine}, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_dataset_readme(dataset_dir) -> Path:
    out = Path(dataset_dir) / "README.md"
    out.write_text(render_dataset_readme(dataset_dir), encoding="utf-8", newline="\n")
    return out


def render_top_level_readme(dataset_dirs) -> str:
    profiles = [_profile(d) for d in dataset_dirs]
    lines: list[str] = []
    lines.append("# SOMA synthetic-IMU datasets (poc-demo) — data reference")
    lines.append("")
    lines.append(_BANNER)
    lines.append("")
    lines.append(f"All datasets share `spec_id` **{spec.SPEC_ID}** (spec_version **{spec.SPEC_VERSION}**): the "
                 "same 8-site spec_S_v2 IMU + 18-joint kinematics + anthro contract. Each dataset's own "
                 "`README.md` has the full field tables; this file is the shared spec + how they differ.")
    lines.append("")
    lines.append("## Conventions (shared)")
    lines.append("")
    lines.append(_CONVENTIONS)
    lines.append("## Common CORE fields (present in every dataset)")
    lines.append("")
    for art, asp in spec.ARTIFACTS.items():
        if asp.capability is not None:
            continue
        core = ", ".join(f"`{k.name}`" for k in asp.core)
        lines.append(f"- **{asp.filename}**: {core}")
    lines.append("")
    lines.append("## Datasets & differences")
    lines.append("")
    lines.append("| dataset | source | small_mode | takes | capability-conditional present |")
    lines.append("|---|---|---|---|---|")
    for p in profiles:
        conds = []
        if "measured_physical" in p["caps"]:
            conds.append("measured 4-key provenance")
        if "development_reference" in p["caps"] and "development_reference" in p["present"]:
            conds.append("development_reference.npz")
        lines.append(f"| `{p['name']}` | {p['source']} | {p['small_mode']} | {p['take_count']} | "
                     f"{', '.join(conds) if conds else '(none — synthetic)'} |")
    lines.append("")
    # field-presence matrix (conditional artifacts/keys)
    lines.append("## Capability-conditional field presence")
    lines.append("")
    lines.append("| field / artifact | " + " | ".join(f"`{p['name']}`" for p in profiles) + " |")
    lines.append("|---|" + "|".join(["---"] * len(profiles)) + "|")
    cond_rows = [("development_reference.npz", lambda p: "development_reference" in p["present"])]
    for art, asp in spec.ARTIFACTS.items():
        for cap, keys in asp.conditional.items():
            for k in keys:
                cond_rows.append((f"{art}.{k.name}", lambda p, c=cap: c in p["caps"]))
    for label, fn in cond_rows:
        cells = " | ".join("Y" if fn(p) else "-" for p in profiles)
        lines.append(f"| `{label}` | {cells} |")
    lines.append("")
    return "\n".join(lines)


def write_top_level_readme(root_dir, dataset_dirs) -> Path:
    out = Path(root_dir) / "README.md"
    out.write_text(render_top_level_readme(dataset_dirs), encoding="utf-8", newline="\n")
    return out
