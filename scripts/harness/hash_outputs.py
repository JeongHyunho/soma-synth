"""Hash a harness run (or the production bundles' sample subjects) in the harness hash format.

Format 2, per file:

* npz: file bytes and sha256, and per member the dtype, shape and sha256 of the DECOMPRESSED .npy
  payload (header excluded), so the hash does not depend on zip compression. Members are read as
  bytes and never unpickled.
* json: raw sha256, the sha256 of a canonical dump (sorted keys) with the wall-clock fields
  generated_utc / applied_utc / generated_at / elapsed_s / elapsed_seconds masked, and the CRLF count.
* markdown: bytes, sha256, and a masked sha256 with ISO-8601 timestamps replaced (the unified8
  descriptions quote the INDEX's generated_utc).
* anything else: bytes and sha256.

On top of that, so a difference can be located and classified rather than only detected:

* manifest-like json: the masked hash of every subtree down to depth 3 (`subtrees`) and, from
  format 2 on, of every subtree at depth 4 (`deep_subtrees`), so that a depth-3 composite such as
  `small_synthesis.inputs.retarget_npz` (relative_path + sha256 + frames) is compared field by
  field. The depth-4 layer is kept apart so a file of either format compares with the other at
  depth 3;
* short unicode npz members: their value (`value`);
* json string fields named in VALUE_KEYS (the retarget `content_hashes` a manifest copies from the
  corpus): their value by dotted path (`values`), so compare_hashes.py can tell which of its
  `adapter=;config=;code=;model=` tokens differs instead of declaring the field by its name;
* INDEX.json (format 2): everything but the take list, verbatim and masked (`index_head`), and the
  take order (`index_order`); with `index_sample` (the take entries themselves) that is the whole
  file, so no INDEX difference is left unlocalized;
* reduced_model_fit.json: its settings block verbatim, a per-group digest, the take -> group map,
  the unfitted list and (format 2) the remaining top-level fields verbatim (`fit`);
* the retarget corpora (hknu_smpl24_paired, gaitex_smpl24, addbio_smpl24_raw): every file, npz
  members as in the corpus ARRAY_SHA256.json evidence (data_sha256, npy_sha256, dtype, shape, fortran).

A run under <data_root>/tmp/harness/<run> names its corpora by a data-root-relative path that
carries the prefix tmp/harness/<run>/. Every hash that such a prefix changes is recorded a second
time with the prefix removed (`norm_*`), and the prefix is named under `normalization`, so
compare_hashes.py --normalize-prefix tmp/harness/<run>/= can compare it with a production bundle.

    python scripts/harness/hash_outputs.py --scratch <run dir> --out <json>
    python scripts/harness/hash_outputs.py --bundles-root <SOMA_DATA_ROOT>/runs/experimental_generation_poc_demo \\
        --sample-only --out <json>      # production, read-only, same format (no corpora)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import os
import pathlib
import re
import sys
import zipfile

import numpy as np
from numpy.lib import format as npf

MASK = {"generated_utc", "applied_utc", "generated_at", "elapsed_s", "elapsed_seconds"}
#: Wall-clock stamps inside markdown (ISO-8601 with a UTC offset), masked like the json fields.
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)")
BUNDLES = ("amass_faithful_full", "prism_faithful_full", "hknu_unified8", "gaitex_unified8",
           "addbio_unified8")
CORPORA = ("hknu_smpl24_paired", "gaitex_smpl24", "addbio_smpl24_raw")
POC = pathlib.Path("runs") / "experimental_generation_poc_demo"
#: The sample filter (subject_id + "|" + take_id contains one of these).
SAMPLES = {
    "amass_faithful_full": ["amass_TotalCapture_s4", "amass_CMU_45", "amass_KIT_63"],
    "prism_faithful_full": ["prism-subj005-"],
    "hknu_unified8": ["hknu_S04"],
    "gaitex_unified8": ["gaitex_austra"],
    "addbio_unified8": ["Hamner2013_Formatted_No_Arm_subject01", "Hammer2013_Formatted_With_Arm_subject01",
                        "Tiziana2019_Formatted_With_Arm_Subject43"],
}
#: Format 2 adds deep_subtrees, index_head / index_order and fit.head. compare_hashes.py reads both.
FORMAT = 2
SUBTREE_DEPTH = 3
DEEP_LEVEL = SUBTREE_DEPTH + 1
SUBTREE_MAX_BYTES = 2 << 20
#: The top-level fields of reduced_model_fit.json the `fit` summary carries in its own form.
FIT_BODY = ("settings", "groups", "takes", "unfitted")
VALUE_MAX_CHARS = 1000
#: json string fields recorded verbatim wherever they sit (compare_hashes.py reads their tokens).
VALUE_KEYS = ("content_hashes", "retarget_content_hashes")
DIGEST = 16


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def masked(obj):
    if isinstance(obj, dict):
        return {k: ("<masked>" if k in MASK else masked(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [masked(v) for v in obj]
    return obj


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")


class Normalizer:
    """Removes the scratch prefix (both slash styles) from strings; a no-op without a prefix."""

    def __init__(self, prefix: str | None):
        self.prefix = prefix
        self.variants = [prefix, prefix.replace("/", "\\")] if prefix else []

    def text(self, value: str) -> str:
        for variant in self.variants:
            value = value.replace(variant, "")
        return value

    def touches(self, value: str) -> bool:
        return any(v in value for v in self.variants)

    def doc(self, obj):
        if isinstance(obj, dict):
            return {self.text(k): self.doc(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.doc(v) for v in obj]
        if isinstance(obj, str):
            return self.text(obj)
        return obj


def subtrees(doc, depth: int = SUBTREE_DEPTH, start: int = 1) -> dict[str, str]:
    """The masked-hash digest of every dict subtree at levels `start`..`depth` (level 1 = the
    top-level keys), by dotted path."""
    out: dict[str, str] = {}

    def walk(obj, path: str, level: int) -> None:
        if path and level >= start:
            out[path] = sha(canonical(obj))[:DIGEST]
        if level >= depth or not isinstance(obj, dict):
            return
        for key, value in obj.items():
            walk(value, f"{path}.{key}" if path else key, level + 1)

    walk(doc, "", 0)
    return out


def recorded_values(doc) -> dict[str, str]:
    """Every short string field named in VALUE_KEYS, by the dotted path `subtrees` uses."""
    out: dict[str, str] = {}

    def walk(obj, path: str) -> None:
        if not isinstance(obj, dict):
            return
        for key, value in obj.items():
            where = f"{path}.{key}" if path else key
            if key in VALUE_KEYS and isinstance(value, str) and len(value) <= VALUE_MAX_CHARS:
                out[where] = value
            walk(value, where)

    walk(doc, "")
    return out


def npz_members(path: pathlib.Path, norm: Normalizer, corpus: bool) -> dict:
    members = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            raw = archive.read(info)
            handle = io.BytesIO(raw)
            version = npf.read_magic(handle)
            reader = npf.read_array_header_1_0 if version == (1, 0) else npf.read_array_header_2_0
            shape, fortran, dtype = reader(handle)
            payload = raw[handle.tell():]
            name = info.filename.removesuffix(".npy")
            member = {"dtype": dtype.str, "shape": list(shape), "data_sha256": sha(payload)}
            if corpus:
                member["fortran"] = bool(fortran)
                member["npy_sha256"] = sha(raw)
            if dtype.kind == "U" and not fortran:
                array = np.frombuffer(payload, dtype=dtype).reshape(shape)
                values = [str(v) for v in array.reshape(-1)]
                if sum(len(v) for v in values) <= VALUE_MAX_CHARS:
                    member["value"] = values[0] if array.ndim == 0 else values
                if any(norm.touches(v) for v in values):
                    fixed = [norm.text(v) for v in values]
                    natural = max(len(v) for v in values) == dtype.itemsize // 4
                    width = max(1, max(len(v) for v in fixed)) if natural else dtype.itemsize // 4
                    fixed_array = np.array(fixed, dtype=f"<U{width}").reshape(shape)
                    member["norm_dtype"] = fixed_array.dtype.str
                    member["norm_data_sha256"] = sha(fixed_array.tobytes())
                    if "value" in member:
                        member["norm_value"] = fixed[0] if array.ndim == 0 else fixed
            members[name] = member
    return members


def hash_file(path: pathlib.Path, norm: Normalizer, *, corpus: bool = False,
              detail: bool = True) -> dict:
    raw = path.read_bytes()
    record = {"bytes": len(raw), "sha256": sha(raw)}
    if path.suffix == ".npz":
        record["members"] = npz_members(path, norm, corpus)
    elif path.suffix == ".json":
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            record["parse_error"] = str(error)
            return record
        view = masked(doc)
        record["masked_sha256"] = sha(canonical(view))
        record["crlf"] = raw.count(b"\r\n")
        normalized = norm.doc(view) if norm.prefix else view
        if normalized != view:
            record["norm_masked_sha256"] = sha(canonical(normalized))
        values = recorded_values(view)
        if values:
            record["values"] = values
        if detail and len(raw) <= SUBTREE_MAX_BYTES:
            trees = subtrees(view)
            deep = subtrees(view, DEEP_LEVEL, start=DEEP_LEVEL)
            record["subtrees"] = trees
            record["deep_subtrees"] = deep
            if normalized != view:
                fixed = subtrees(normalized)
                record["norm_subtrees"] = {k: v for k, v in fixed.items() if trees.get(k) != v}
                fixed = subtrees(normalized, DEEP_LEVEL, start=DEEP_LEVEL)
                record["norm_deep_subtrees"] = {k: v for k, v in fixed.items() if deep.get(k) != v}
    else:
        record["crlf"] = raw.count(b"\r\n")
        if path.suffix == ".md":
            # the unified8 descriptions quote the INDEX's generated_utc ("- generated: ...")
            text = TIMESTAMP.sub("<masked>", raw.decode("utf-8", errors="replace"))
            record["kind"] = "text"
            record["masked_sha256"] = sha(text.encode("utf-8"))
            if norm.touches(text):
                record["norm_masked_sha256"] = sha(norm.text(text).encode("utf-8"))
    return record


def fit_summary(path: pathlib.Path, groups_wanted=None) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    groups = {}
    for name, rec in document.get("groups", {}).items():
        if groups_wanted is not None and name not in groups_wanted:
            continue
        rest = {k: v for k, v in rec.items() if k != "settings"}
        groups[name] = {
            "record_sha": sha(canonical(rec))[:DIGEST],
            "record_without_settings_sha": sha(canonical(rest))[:DIGEST],
            "settings_sha": sha(canonical(rec.get("settings")))[:DIGEST],
            "constants_sha": sha(canonical(rec.get("constants_wxyz")))[:DIGEST],
            "takes": rec.get("takes"), "frames_pooled": rec.get("frames_pooled"),
            "frames_measured": rec.get("frames_measured"), "converged": rec.get("converged"),
            "residual_fitted_m": rec.get("residual_fitted_m"),
        }
    takes = {t: g for t, g in document.get("takes", {}).items()
             if groups_wanted is None or g in groups_wanted}
    return {"settings": document.get("settings"), "settings_sha": sha(canonical(document.get("settings")))[:DIGEST],
            "groups": groups, "takes": takes,
            "unfitted": {k: v for k, v in document.get("unfitted", {}).items()
                         if groups_wanted is None or k in takes},
            # the rest of the file (schema, method, source, counts), so that with the fields above
            # the summary covers every top-level key; counts differ with the scope, like INDEX
            "head": masked({k: v for k, v in document.items() if k not in FIT_BODY}),
            "complete": groups_wanted is None}


def in_sample(bundle: str, entry: dict) -> bool:
    text = entry.get("subject_id", "") + "|" + entry.get("take_id", "")
    return any(p in text for p in SAMPLES[bundle])


def hash_bundle(bundle: str, root: pathlib.Path, norm: Normalizer, sample_only: bool) -> dict:
    index = json.loads((root / "INDEX.json").read_text(encoding="utf-8"))
    entries = [e for e in index["takes"] if in_sample(bundle, e)] if sample_only else list(index["takes"])
    outside = [e["rel"] for e in entries if not in_sample(bundle, e)]
    result = {"generated_utc": index.get("generated_utc"), "counts": index.get("counts"),
              "index_sample": sorted(entries, key=lambda e: e["rel"]), "root_files": {}, "takes": {},
              # INDEX.json = index_head + the entries in index_order; the entries are index_sample
              "index_head": masked({k: v for k, v in index.items() if k != "takes"})}
    if not sample_only:
        result["index_order"] = [e.get("rel") for e in index["takes"]]
    if outside:
        result["takes_outside_sample"] = outside
    for path in sorted(root.iterdir()):
        if path.is_file() and not path.name.startswith("INDEX_shard"):
            big = path.name in ("INDEX.json", "reduced_model_fit.json", "validation_ledger.json",
                                "VALIDATION_REPORT.json")
            result["root_files"][path.name] = hash_file(path, norm, detail=not big)
    fit_path = root / "reduced_model_fit.json"
    if fit_path.is_file():
        wanted = None
        if sample_only:
            document = json.loads(fit_path.read_text(encoding="utf-8"))
            wanted = {document["takes"][e["rel"]] for e in entries if e["rel"] in document["takes"]}
        result["fit"] = fit_summary(fit_path, wanted)
    for entry in entries:
        take_dir = root / entry["rel"]
        if not take_dir.is_dir():
            result["takes"][entry["rel"]] = {"status": entry.get("status"), "missing_dir": True}
            continue
        result["takes"][entry["rel"]] = {
            "status": entry.get("status"),
            "files": {p.name: hash_file(p, norm) for p in sorted(take_dir.iterdir()) if p.is_file()},
        }
    if not sample_only:
        listed = {e["rel"] for e in entries}
        extra = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in listed)
        if extra:
            result["unindexed_dirs"] = extra
    return result


def hash_corpus(root: pathlib.Path, norm: Normalizer) -> dict:
    files = {}
    for current, dirs, names in os.walk(root):
        dirs.sort()
        for name in sorted(names):
            path = pathlib.Path(current) / name
            files[path.relative_to(root).as_posix()] = hash_file(path, norm, corpus=True)
    return {"files": files}


def locate(scratch: pathlib.Path, name: str) -> pathlib.Path | None:
    """<scratch>/runs/... first, then one level down (the PRISM input root <scratch>/prismroot)."""
    candidates = [scratch / POC / name] + sorted(p / POC / name for p in scratch.iterdir() if p.is_dir())
    return next((c for c in candidates if c.is_dir()), None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scratch", default=None, help="a harness run directory")
    parser.add_argument("--bundles-root", default=None,
                        help="hash the bundles under this poc_demo directory instead (read-only)")
    parser.add_argument("--sample-only", action="store_true",
                        help="restrict takes to the harness samples (always used with --bundles-root)")
    parser.add_argument("--data-root", default=os.environ.get("SOMA_DATA_ROOT"),
                        help="the data root the scratch prefix is taken relative to")
    parser.add_argument("--out", required=True)
    parser.add_argument("--note", default=None)
    args = parser.parse_args(argv)
    if bool(args.scratch) == bool(args.bundles_root):
        parser.error("give exactly one of --scratch and --bundles-root")

    doc = {"what": args.note or "harness hashes (format 2)",
           "captured": _dt.datetime.now(_dt.UTC).date().isoformat(), "numpy": np.__version__,
           "python": sys.version.split()[0], "mask": sorted(MASK), "format": FORMAT,
           "subtree_depth": SUBTREE_DEPTH, "deep_level": DEEP_LEVEL, "bundles": {}}
    if args.scratch:
        scratch = pathlib.Path(args.scratch).resolve()
        prefix = None
        if args.data_root:
            try:
                prefix = scratch.relative_to(pathlib.Path(args.data_root).resolve()).as_posix() + "/"
            except ValueError:
                prefix = None
        norm = Normalizer(prefix)
        doc["scratch"] = str(scratch)
        doc["normalization"] = {"prefix": prefix, "replacement": ""} if prefix else None
        run_record = scratch / "_harness" / "HARNESS_RUN.json"
        if run_record.is_file():
            doc["harness_run"] = json.loads(run_record.read_text(encoding="utf-8")).get("code_root")
        for name in BUNDLES:
            root = locate(scratch, name)
            if root is None:
                continue
            doc["bundles"][name] = hash_bundle(name, root, norm, args.sample_only)
            doc["bundles"][name]["location"] = root.relative_to(scratch).as_posix()
            print(name, len(doc["bundles"][name]["takes"]), "takes", flush=True)
        doc["corpora"] = {}
        for name in CORPORA:
            root = locate(scratch, name)
            if root is None:
                continue
            doc["corpora"][name] = hash_corpus(root, norm)
            doc["corpora"][name]["location"] = root.relative_to(scratch).as_posix()
            print(name, len(doc["corpora"][name]["files"]), "files", flush=True)
    else:
        base = pathlib.Path(args.bundles_root)
        norm = Normalizer(None)
        doc["bundles_root"] = str(base)
        doc["normalization"] = None
        for name in BUNDLES:
            if (base / name / "INDEX.json").is_file():
                doc["bundles"][name] = hash_bundle(name, base / name, norm, True)
                print(name, len(doc["bundles"][name]["takes"]), "takes", flush=True)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(doc, handle, indent=1, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print("wrote", out, out.stat().st_size, "bytes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
