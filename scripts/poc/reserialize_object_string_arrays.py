"""Rewrite a bundle's `object`-dtype string arrays as numpy `U`, changing nothing else.

This is a **format migration, not a generation**. No value is recomputed: every numeric array is
copied through unchanged and every string array keeps its elements, gaining only a fixed-width
dtype. That distinction matters here, because `DETERMINISTIC_EXECUTION_CONFIG_HOLD` forbids
generating while it is active, and that hold is not lifted by this work (ADR-0040 D5).

Why bother: an npz holding `dtype=object` cannot be read back without `allow_pickle=True`, so the
defect hands arbitrary-code-execution risk to whoever consumes the bundle. AMASS carries it on
seven arrays per take -- `sensor_codes` and `mount_id` in small, `reference_poses`, `joint_names`,
`segment_names` and `fixed_joint_names` in anthro, `joint_names` in large -- 57,827 occurrences
across 8,261 takes, while every other bundle stores the same arrays correctly.

The output is a new lineage generation (retention rule 1); the source bundle is left untouched.

    python scripts/poc/reserialize_object_string_arrays.py <src_bundle> <dst_bundle> [--limit N]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

#: Files whose arrays are rewritten. Anything else in a take directory is copied verbatim.
NPZ_NAMES = (
    "small_reference.npz",
    "large_reference.npz",
    "anthro_reference.npz",
    "smpl_root_translation.npz",
    "development_reference.npz",
)


class ConversionError(RuntimeError):
    pass


def _convert_array(value: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (array, changed). Only object arrays of str become U; everything else is untouched."""
    if value.dtype != object:
        return value, False
    flat = value.ravel()
    if flat.size and not all(isinstance(v, str) for v in flat):
        # An object array holding something other than strings is not this script's business.
        return value, False
    converted = value.astype(np.str_)
    if converted.shape != value.shape:
        raise ConversionError(f"shape changed {value.shape} -> {converted.shape}")
    if flat.size and not all(a == b for a, b in zip(flat, converted.ravel())):
        raise ConversionError("string contents changed during conversion")
    return converted, True


def convert_npz(src: Path, dst: Path) -> int:
    """Rewrite one npz. Returns how many arrays changed dtype. Verifies before it returns."""
    with np.load(src, allow_pickle=True) as z:
        names = list(z.files)
        original = {name: z[name] for name in names}

    converted: dict[str, np.ndarray] = {}
    changed = 0
    for name, value in original.items():
        new_value, did = _convert_array(value)
        converted[name] = new_value
        changed += int(did)

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **converted)

    # Read back and prove equality. A migration that cannot demonstrate it preserved the data is
    # indistinguishable from one that did not.
    with np.load(dst, allow_pickle=False) as z:
        if set(z.files) != set(names):
            raise ConversionError(f"{src.name}: key set changed")
        for name in names:
            before, after = original[name], z[name]
            if before.shape != after.shape:
                raise ConversionError(f"{src.name}:{name} shape {before.shape} -> {after.shape}")
            if before.dtype == object or before.dtype.kind in ("U", "S"):
                if not np.array_equal(before.astype(np.str_), after.astype(np.str_)):
                    raise ConversionError(f"{src.name}:{name} string contents differ")
            elif not np.array_equal(before, after, equal_nan=True):
                raise ConversionError(f"{src.name}:{name} numeric contents differ")
    return changed


def convert_take(src_take: Path, dst_take: Path) -> int:
    dst_take.mkdir(parents=True, exist_ok=True)
    changed = 0
    for child in sorted(src_take.iterdir()):
        if child.is_dir():
            continue
        if child.name in NPZ_NAMES:
            changed += convert_npz(child, dst_take / child.name)
        else:
            shutil.copy2(child, dst_take / child.name)
    return changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N takes")
    ap.add_argument("--progress-every", type=int, default=250)
    args = ap.parse_args(argv)

    src_root, dst_root = Path(args.src), Path(args.dst)
    if not src_root.is_dir():
        print(f"source bundle not found: {src_root}")
        return 2
    if dst_root.exists() and any(dst_root.iterdir()):
        print(f"destination exists and is not empty: {dst_root}")
        return 2

    takes = [p for p in sorted(src_root.iterdir()) if p.is_dir() and not p.name.startswith("_")]
    if args.limit is not None:
        takes = takes[: args.limit]

    dst_root.mkdir(parents=True, exist_ok=True)
    for child in sorted(src_root.iterdir()):
        if child.is_file():
            shutil.copy2(child, dst_root / child.name)

    total_changed = 0
    for i, take in enumerate(takes, 1):
        total_changed += convert_take(take, dst_root / take.name)
        if i % args.progress_every == 0 or i == len(takes):
            print(f"  {i}/{len(takes)} takes, {total_changed} arrays re-typed", flush=True)

    receipt = {
        "operation": "reserialize_object_string_arrays",
        "note": "format migration; no value recomputed (ADR-0040 D5)",
        "source_bundle": src_root.name,
        "takes_converted": len(takes),
        "arrays_retyped": total_changed,
    }
    (dst_root / "RESERIALIZATION.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"done: {len(takes)} takes, {total_changed} arrays re-typed -> {dst_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
