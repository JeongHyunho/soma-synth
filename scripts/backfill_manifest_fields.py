"""Add the three fields ADR-0040 D1 promoted to bundles that were built before they existed.

This is a **text edit, not a generation**. No tensor is read and no npz is opened, so it does not
touch what ``DETERMINISTIC_EXECUTION_CONFIG_HOLD`` forbids while it stands (ADR-0040 D5 draws the
same line for the re-serialisation migration).

Three fields, three different provenances, and the difference is the whole design:

``up_axis`` and ``resample`` for PRISM are **derived from the manifest itself**. The bundle
already states its frame twice -- ``global_frame.convention`` says "Z-up, gravity along -Z" and
``canonicalization_config.gravity_prism_world_m_s2`` is ``[0, 0, -9.80665]`` -- so writing
``"up_axis": "z"`` moves a recorded fact to where a machine reads it. The deriver checks **both**
witnesses and skips any take where they disagree, rather than trusting whichever it read first.

``resample`` is derived the same way, from four numbers that must agree: the source rate, the
target rate, the source frame count and the emitted frame count. PRISM's generator asserts
``fps == TARGET_RATE_HZ`` before it writes anything
(``scripts/poc/generate_prism_faithful.py:315``) and contains no interpolation at all, so no take
can hold a resample -- but this tool verifies that per take instead of assuming it. Writing
"no resampling" onto a take that was resampled is the worst thing this script could do.

``source_attribution`` is **not derivable**. It comes from
``configs/datasets/source_attribution_v1.yaml``, and a source whose interpretation still awaits the
data owner's confirmation is refused there rather than guessed at here.

Two safety properties hold every write:

**Byte-exact round-trip.** Before a manifest is touched, the tool re-serialises the *unmodified*
object in the style it detected from the file and requires the result to equal the original bytes.
Bundles differ here -- PRISM writes CRLF with no trailing newline, AMASS LF with one -- so the
style is read from each file, never assumed. A take that does not round-trip exactly is skipped:
without that proof there is no way to promise the write changed only what it meant to.

**Additive only.** An existing key is never overwritten. Combined with the round-trip proof, the
edit is exactly "the same bytes, plus these keys", which is what makes ``--rollback`` able to
verify itself: it removes the recorded keys, re-serialises, and checks the SHA-256 against the
pre-image in the journal. If it does not match, it refuses rather than writing.

    # look, change nothing (the default)
    python scripts/backfill_manifest_fields.py <bundle> [<bundle> ...]

    # write, journalling every change
    python scripts/backfill_manifest_fields.py <bundle> --apply

    # undo one journal, verifying each file returns to its pre-image
    python scripts/backfill_manifest_fields.py --rollback <journal.jsonl>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Mapping

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION_RELPATH = "configs/datasets/source_attribution_v1.yaml"
ATTRIBUTION_SCHEMA = "source_attribution_v1"

#: The fields this tool knows how to add. Anything else is out of its scope by construction.
FIELDS = ("up_axis", "resample", "source_attribution")


class BackfillError(RuntimeError):
    """The tool cannot proceed, and says why rather than doing something approximate."""


# --------------------------------------------------------------------------- serialisation
@dataclass(frozen=True)
class Style:
    """How one bundle happens to write JSON. Read from the file, never assumed."""

    newline: str
    trailing_newline: bool

    def dump(self, obj: Mapping) -> bytes:
        text = json.dumps(obj, ensure_ascii=False, indent=2)
        if self.newline != "\n":
            text = text.replace("\n", self.newline)
        if self.trailing_newline:
            text += self.newline
        return text.encode("utf-8")


def detect_style(raw: bytes, obj: Mapping) -> Style | None:
    """The style under which ``obj`` re-serialises to exactly ``raw``, or None if none does."""
    for newline in ("\r\n", "\n"):
        for trailing in (False, True):
            style = Style(newline=newline, trailing_newline=trailing)
            if style.dump(obj) == raw:
                return style
    return None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------------- derivers
@dataclass
class Derivation:
    """What a deriver concluded: a value, or a reason there isn't one."""

    value: object = None
    skip_reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.skip_reason


def _axis_from_gravity(vector) -> str | None:
    """The up axis a gravity vector implies: one negative component, the rest zero."""
    if not isinstance(vector, (list, tuple)) or len(vector) != 3:
        return None
    try:
        components = [float(v) for v in vector]
    except (TypeError, ValueError):
        return None
    nonzero = [i for i, v in enumerate(components) if v != 0.0]
    if len(nonzero) != 1:
        return None
    index = nonzero[0]
    if components[index] >= 0.0:
        # Gravity points down, so the up axis is the one it opposes. A positive component
        # would mean the frame's convention is the other way round, which is a real difference
        # and not this tool's to paper over.
        return None
    return "xyz"[index]


def _axis_from_convention(text: object) -> str | None:
    """The up axis a convention string states, as in `... right-handed, Z-up, gravity along -Z`."""
    if not isinstance(text, str):
        return None
    lowered = text.lower()
    found = [axis for axis in "xyz" if f"{axis}-up" in lowered]
    return found[0] if len(found) == 1 else None


def derive_up_axis(manifest: Mapping) -> Derivation:
    """Two independent witnesses, and they must agree.

    The prose convention and the gravity vector were written by different parts of the generator.
    Reading only one would make this tool's output depend on which of them is right.
    """
    convention = (manifest.get("global_frame") or {}).get("convention")
    gravity = None
    for key, value in (manifest.get("canonicalization_config") or {}).items():
        if key.startswith("gravity"):
            gravity = value
            break

    from_convention = _axis_from_convention(convention)
    from_gravity = _axis_from_gravity(gravity)

    if from_convention is None and from_gravity is None:
        return Derivation(skip_reason="no up-axis witness in the manifest")
    if from_convention is None:
        return Derivation(skip_reason=f"only one witness (gravity says {from_gravity!r})")
    if from_gravity is None:
        return Derivation(skip_reason=f"only one witness (convention says {from_convention!r})")
    if from_convention != from_gravity:
        return Derivation(
            skip_reason=(
                f"witnesses disagree: convention says {from_convention!r}, "
                f"gravity says {from_gravity!r}"
            )
        )
    return Derivation(value=from_convention)


def derive_resample(manifest: Mapping) -> Derivation:
    """Four numbers that must agree before this take may be called a no-op.

    ``frames_in`` and ``frames_out`` are only unambiguous when the source count, the emitted
    count and the window agree; if they do not, the take was cropped or resampled and its
    resample record is not something to infer.
    """
    source = manifest.get("source") or {}
    config = manifest.get("canonicalization_config") or {}
    window = manifest.get("window") or {}

    native_rate = source.get("native_rate_hz")
    target_rate = config.get("target_rate_hz")
    if native_rate is None or target_rate is None:
        return Derivation(skip_reason="no native_rate_hz or target_rate_hz")
    if float(native_rate) != float(target_rate):
        return Derivation(
            skip_reason=(
                f"rates differ ({native_rate} -> {target_rate}); this take was resampled and "
                "its record cannot be inferred"
            )
        )

    counts = {
        "source.native_frames_total": source.get("native_frames_total"),
        "canonicalization_config.frame_count": config.get("frame_count"),
        "window.frame_count": window.get("frame_count"),
    }
    start, end = window.get("frame_start"), window.get("frame_end_exclusive")
    if start is not None and end is not None:
        counts["window.frame_end_exclusive - frame_start"] = end - start

    missing = [name for name, value in counts.items() if value is None]
    if missing:
        return Derivation(skip_reason=f"no frame count at {', '.join(missing)}")
    distinct = sorted({int(v) for v in counts.values()})
    if len(distinct) != 1:
        detail = ", ".join(f"{name}={value}" for name, value in counts.items())
        return Derivation(
            skip_reason=f"frame counts disagree ({detail}); the take was cropped or resampled"
        )

    frames = distinct[0]
    return Derivation(
        value={
            # float for source_fps, matching every other bundle: gaitex, hknu and addbio all
            # read theirs from source headers that give floats. The value is unchanged.
            "source_fps": float(native_rate),
            "target_fps": target_rate,
            "frames_in": frames,
            "frames_out": frames,
            "method": "none: the source grid is already the target grid, and this "
                      "lineage has no resampler",
            "noop": True,
            "direction": "none",
        }
    )


#: Which fields this tool will derive for which source. A source/field pair that is not here is
#: not derivable, and the tool says so rather than reaching for a default.
DERIVERS: Mapping[str, Mapping[str, Callable[[Mapping], Derivation]]] = {
    "prism": {"up_axis": derive_up_axis, "resample": derive_resample},
}


# ------------------------------------------------------------------------------ attribution
@dataclass(frozen=True)
class AttributionRegistry:
    required_keys: tuple[str, ...]
    resolved: Mapping[str, Mapping[str, str]]
    pending: Mapping[str, str]
    path: Path

    def value_for(self, source: str) -> Derivation:
        if source in self.resolved:
            return Derivation(value=dict(self.resolved[source]))
        if source in self.pending:
            return Derivation(
                skip_reason=(
                    f"{source} attribution awaits the data owner's confirmation "
                    f"({self.path.name}); {self.pending[source]}"
                )
            )
        return Derivation(skip_reason=f"{source} has no entry in {self.path.name}")


def load_attribution(path: Path | None = None) -> AttributionRegistry:
    """The registry, through the contracts reader the generators use since 2026-09-14.

    One reader for one file: a generator and this tool must agree on what a resolved
    attribution is, or a bundle could be generated with a block the backfill would refuse.
    """
    from soma_synth.contracts import source_attribution as attribution_mod

    try:
        config = attribution_mod.load(path or (_REPOSITORY_ROOT / ATTRIBUTION_RELPATH))
    except attribution_mod.AttributionError as error:
        raise BackfillError(str(error)) from error
    return AttributionRegistry(
        required_keys=config.required_keys, resolved=config.resolved,
        pending=config.pending, path=config.path,
    )


# ------------------------------------------------------------------------------------ plan
@dataclass
class TakePlan:
    take_id: str
    manifest_path: Path
    additions: dict[str, object] = field(default_factory=dict)
    skips: dict[str, str] = field(default_factory=dict)
    blocked: str = ""


@dataclass
class BundlePlan:
    bundle: Path
    source: str = ""
    takes: list[TakePlan] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        totals: dict[str, int] = {f: 0 for f in FIELDS}
        for take in self.takes:
            for name in take.additions:
                totals[name] += 1
        return totals


def _source_name(manifest: Mapping) -> str:
    return str((manifest.get("source") or {}).get("source_name") or "")


def iter_takes(bundle: Path) -> Iterator[Path]:
    for child in sorted(bundle.iterdir()):
        if child.is_dir() and (child / "manifest.json").is_file():
            yield child


def plan_bundle(bundle: Path, attribution: AttributionRegistry) -> BundlePlan:
    plan = BundlePlan(bundle=bundle)
    for take_dir in iter_takes(bundle):
        manifest_path = take_dir / "manifest.json"
        raw = manifest_path.read_bytes()
        try:
            manifest = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            plan.unreadable.append(f"{take_dir.name}: {exc}")
            continue

        source = _source_name(manifest)
        plan.source = plan.source or source
        take = TakePlan(take_id=take_dir.name, manifest_path=manifest_path)

        if detect_style(raw, manifest) is None:
            # Without a byte-exact round-trip there is no way to promise a write adds only what
            # it means to, so the take is left alone and reported.
            take.blocked = "manifest does not round-trip byte-exactly; not safe to rewrite"
            plan.takes.append(take)
            continue

        for name in FIELDS:
            if name in manifest:
                continue
            if name == "source_attribution":
                result = attribution.value_for(source)
            else:
                deriver = (DERIVERS.get(source) or {}).get(name)
                result = (
                    deriver(manifest) if deriver
                    else Derivation(skip_reason=f"no deriver for {source}.{name}")
                )
            if result.ok:
                take.additions[name] = result.value
            else:
                take.skips[name] = result.skip_reason
        plan.takes.append(take)
    return plan


# ----------------------------------------------------------------------------------- apply
def apply_take(take: TakePlan) -> dict[str, object]:
    """Write one manifest, and return the journal entry proving what changed."""
    raw = take.manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    style = detect_style(raw, manifest)
    if style is None:
        raise BackfillError(f"{take.take_id}: manifest no longer round-trips; refusing to write")

    already = [k for k in take.additions if k in manifest]
    if already:
        raise BackfillError(f"{take.take_id}: would overwrite {', '.join(already)}")

    before = sha256(raw)
    for name, value in take.additions.items():
        manifest[name] = value
    written = style.dump(manifest)
    take.manifest_path.write_bytes(written)

    return {
        "take_id": take.take_id,
        "path": str(take.manifest_path),
        "added_keys": sorted(take.additions),
        "sha256_before": before,
        "sha256_after": sha256(written),
        "newline": "crlf" if style.newline == "\r\n" else "lf",
        "trailing_newline": style.trailing_newline,
    }


def rollback(journal_path: Path, log=print) -> int:
    """Remove the recorded keys and require the file to return to its recorded pre-image.

    Journals undo in reverse order. A bundle backfilled twice has two journals, and the older
    one no longer describes what is on disk; every entry is refused rather than overwriting the
    newer pass's work with a pre-image that predates it.
    """
    restored = failed = superseded = 0
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        entry = json.loads(line)
        path = Path(entry["path"])
        if not path.is_file():
            log(f"  MISSING {entry['take_id']}: {path}")
            failed += 1
            continue
        raw = path.read_bytes()
        if sha256(raw) == entry["sha256_before"]:
            continue  # already back
        if sha256(raw) != entry["sha256_after"]:
            log(f"  CHANGED SINCE {entry['take_id']}: not the file this journal wrote; skipped")
            failed += 1
            superseded += 1
            continue
        manifest = json.loads(raw.decode("utf-8"))
        for key in entry["added_keys"]:
            manifest.pop(key, None)
        style = Style(
            newline="\r\n" if entry["newline"] == "crlf" else "\n",
            trailing_newline=bool(entry["trailing_newline"]),
        )
        candidate = style.dump(manifest)
        if sha256(candidate) != entry["sha256_before"]:
            log(f"  NOT EXACT {entry['take_id']}: would not restore the pre-image; skipped")
            failed += 1
            continue
        path.write_bytes(candidate)
        restored += 1
    log(f"rollback: {restored} restored, {failed} refused")
    if restored == 0 and superseded == failed and failed:
        log(
            "  every entry was refused because the files have moved on: this journal is "
            "superseded by a later pass over the same bundle. Roll journals back newest first."
        )
    return 1 if failed else 0


# ---------------------------------------------------------------------------------- report
def report(plan: BundlePlan, log=print) -> None:
    log(f"\n=== {plan.bundle.name}  (source={plan.source or '?'}, takes={len(plan.takes)}) ===")
    counts = plan.counts()
    for name in FIELDS:
        log(f"  would add {name:<20} {counts[name]:>6}")

    blocked = [t for t in plan.takes if t.blocked]
    if blocked:
        log(f"  BLOCKED takes: {len(blocked)}")
        for take in blocked[:5]:
            log(f"    {take.take_id}: {take.blocked}")

    reasons: dict[str, list[str]] = {}
    for take in plan.takes:
        for name, why in take.skips.items():
            reasons.setdefault(f"{name}: {why}", []).append(take.take_id)
    for why, takes in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        log(f"  skip x{len(takes):<6} {why}")
        if len(takes) <= 5:
            log(f"      {', '.join(takes)}")
    if plan.unreadable:
        log(f"  UNREADABLE: {len(plan.unreadable)}")
        for line in plan.unreadable[:5]:
            log(f"    {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundles", nargs="*", type=Path, help="bundle directories")
    parser.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    parser.add_argument("--journal-dir", type=Path, default=None,
                        help="where to write the rollback journal (default: alongside the bundle)")
    parser.add_argument("--rollback", type=Path, default=None, help="undo one journal")
    parser.add_argument("--attribution", type=Path, default=None, help="override the registry")
    args = parser.parse_args(argv)

    if args.rollback:
        return rollback(args.rollback)
    if not args.bundles:
        parser.error("give at least one bundle directory, or --rollback a journal")

    attribution = load_attribution(args.attribution)
    print(f"attribution registry: {attribution.path}")
    print(f"  resolved: {sorted(attribution.resolved) or '-'}")
    print(f"  pending : {sorted(attribution.pending) or '-'}")

    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    exit_code = 0
    for bundle in args.bundles:
        if not bundle.is_dir():
            print(f"not a directory: {bundle}", file=sys.stderr)
            exit_code = 1
            continue
        plan = plan_bundle(bundle, attribution)
        report(plan)

        if not args.apply:
            continue
        pending = [t for t in plan.takes if t.additions]
        if not pending:
            print("  nothing to write")
            continue
        journal_dir = args.journal_dir or (bundle.parent / "_manifest_backfill")
        journal_dir.mkdir(parents=True, exist_ok=True)
        journal = journal_dir / f"{stamp}_{bundle.name}.jsonl"
        with journal.open("w", encoding="utf-8", newline="\n") as fh:
            for take in pending:
                fh.write(json.dumps(apply_take(take), ensure_ascii=False) + "\n")
        print(f"  wrote {len(pending)} manifests; journal {journal}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
