"""One shape for "what this bundle cannot promise", the same for every dataset.

Every bundle has caveats -- a proxy channel, a six-axis insole with no heading, a source that
ships only pose parameters, a retarget that had to repair wrapped coordinates. Written as prose
in each bundle's own description, they could be compared only by reading every description, and
not by a program at all.

This module fixes the shape. The values live in ``configs/datasets/known_limitations_v1.yaml``,
one block per bundle, and are rendered into each bundle root as ``KNOWN_LIMITATIONS.json`` by
``scripts/write_known_limitations.py``. The validator requires the file, checks it against this
schema, and checks that it matches the repository's copy, so the two cannot drift apart without
a finding.

The schema is a presence/enumeration judgement only. It declares no threshold and no tolerance
(DETERMINISTIC_EXECUTION_CONFIG_HOLD): a limitation is stated, scoped, and
attributed to evidence, never scored.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

SCHEMA = "dataset_limitations_v1"
FILE_NAME = "KNOWN_LIMITATIONS.json"
CONFIG_RELPATH = "configs/datasets/known_limitations_v1.yaml"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPOSITORY_ROOT / CONFIG_RELPATH

#: What kind of thing a limitation is. The kind says who could remove it.
KINDS = frozenset({
    "hardware_limit",    # the sensor cannot measure it (a 6-axis unit has no heading reference)
    "source_defect",     # the source data has a flaw we did not introduce
    "source_scope",      # the source never recorded it (no GRF, no measured IMU, no arms)
    "pipeline_defect",   # our processing introduced it; fixable on our side
    "proxy",             # a channel stands in for a sensor that does not exist
    "approximation",     # a value is derived under a stated simplifying assumption
    "coverage",          # some takes, sites or subjects are missing or excluded
    "rights",            # what the data may and may not be used for
})

#: How a consumer should treat the affected data.
SEVERITIES = frozenset({
    "info",              # good to know; no change in use
    "caution",           # usable, with the stated consequence in mind
    "do_not_use",        # the affected data must not be used for the stated purpose
})

#: Whether the limitation still stands in the bundle as shipped.
STATUSES = frozenset({"open", "mitigated", "fixed"})

#: Where the limitation applies.
SCOPE_LEVELS = frozenset({"bundle", "site", "array", "take", "subject"})

_REQUIRED = ("id", "kind", "severity", "scope", "statement", "consequence", "evidence",
             "status", "since")
_OPTIONAL = ("mitigation", "related", "fixed_in")
_SCOPE_REQUIRED = ("level",)
_SCOPE_OPTIONAL = ("sites", "arrays", "takes", "subjects", "files")


class LimitationsError(ValueError):
    """The config or a bundle's file does not have the agreed shape."""


@dataclass(frozen=True)
class Limitation:
    id: str
    kind: str
    severity: str
    scope: Mapping[str, Any]
    statement: str
    consequence: str
    evidence: str
    status: str
    since: str
    mitigation: str = ""
    related: tuple[str, ...] = ()
    fixed_in: str = ""

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id, "kind": self.kind, "severity": self.severity,
            "scope": dict(self.scope), "statement": self.statement,
            "consequence": self.consequence, "evidence": self.evidence,
            "status": self.status, "since": self.since,
        }
        if self.mitigation:
            out["mitigation"] = self.mitigation
        if self.related:
            out["related"] = list(self.related)
        if self.fixed_in:
            out["fixed_in"] = self.fixed_in
        return out


def problems_in_entry(raw: Mapping[str, Any], where: str) -> list[str]:
    """Every way one entry fails the schema, as sentences a reader can act on."""
    out: list[str] = []
    missing = [k for k in _REQUIRED if not str(raw.get(k) or "").strip() and k != "scope"]
    if missing:
        out.append(f"{where}: missing {', '.join(missing)}")
    unknown = sorted(set(raw) - set(_REQUIRED) - set(_OPTIONAL))
    if unknown:
        out.append(f"{where}: unknown field(s) {unknown}")
    if raw.get("kind") not in KINDS:
        out.append(f"{where}: kind {raw.get('kind')!r} is not one of {sorted(KINDS)}")
    if raw.get("severity") not in SEVERITIES:
        out.append(f"{where}: severity {raw.get('severity')!r} is not one of {sorted(SEVERITIES)}")
    if raw.get("status") not in STATUSES:
        out.append(f"{where}: status {raw.get('status')!r} is not one of {sorted(STATUSES)}")
    if raw.get("status") == "fixed" and not str(raw.get("fixed_in") or "").strip():
        out.append(f"{where}: status is 'fixed' but fixed_in is empty")
    if raw.get("status") == "mitigated" and not str(raw.get("mitigation") or "").strip():
        out.append(f"{where}: status is 'mitigated' but mitigation is empty")
    since = raw.get("since")
    try:
        _dt.date.fromisoformat(str(since))
    except (TypeError, ValueError):
        out.append(f"{where}: since {since!r} is not an ISO date")
    scope = raw.get("scope")
    if not isinstance(scope, Mapping):
        out.append(f"{where}: scope must be an object")
    else:
        if scope.get("level") not in SCOPE_LEVELS:
            out.append(f"{where}: scope.level {scope.get('level')!r} is not one of {sorted(SCOPE_LEVELS)}")
        for k in set(scope) - set(_SCOPE_REQUIRED) - set(_SCOPE_OPTIONAL):
            out.append(f"{where}: unknown scope field {k!r}")
        for k in _SCOPE_OPTIONAL:
            v = scope.get(k)
            if v is not None and (not isinstance(v, Sequence) or isinstance(v, str)):
                out.append(f"{where}: scope.{k} must be a list")
        level = scope.get("level")
        if level == "site" and not scope.get("sites"):
            out.append(f"{where}: scope.level is 'site' but scope.sites is empty")
        if level == "array" and not scope.get("arrays"):
            out.append(f"{where}: scope.level is 'array' but scope.arrays is empty")
        if level == "take" and not scope.get("takes"):
            out.append(f"{where}: scope.level is 'take' but scope.takes is empty")
    related = raw.get("related")
    if related is not None and (not isinstance(related, Sequence) or isinstance(related, str)):
        out.append(f"{where}: related must be a list")
    return out


def parse_entries(raw_entries: Sequence[Mapping[str, Any]], where: str) -> tuple[Limitation, ...]:
    problems: list[str] = []
    seen: set[str] = set()
    for n, raw in enumerate(raw_entries):
        label = f"{where} entry {n} ({raw.get('id', '<no id>')})"
        problems.extend(problems_in_entry(raw, label))
        if raw.get("id") in seen:
            problems.append(f"{label}: duplicate id")
        seen.add(raw.get("id"))
    if problems:
        raise LimitationsError("; ".join(problems))
    return tuple(
        Limitation(
            id=str(r["id"]), kind=str(r["kind"]), severity=str(r["severity"]),
            scope=dict(r["scope"]), statement=str(r["statement"]).strip(),
            consequence=str(r["consequence"]).strip(), evidence=str(r["evidence"]).strip(),
            status=str(r["status"]), since=str(r["since"]),
            mitigation=str(r.get("mitigation") or "").strip(),
            related=tuple(str(x) for x in (r.get("related") or ())),
            fixed_in=str(r.get("fixed_in") or "").strip(),
        )
        for r in raw_entries
    )


@dataclass(frozen=True)
class Config:
    bundles: Mapping[str, tuple[Limitation, ...]]
    path: Path


def load_config(path: Path | str | None = None) -> Config:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise LimitationsError(f"known-limitations config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if raw.get("schema") != SCHEMA:
        raise LimitationsError(f"expected schema {SCHEMA!r}, got {raw.get('schema')!r}")
    bundles = {}
    for name, entries in (raw.get("bundles") or {}).items():
        bundles[str(name)] = parse_entries(entries or [], f"bundle {name}")
    if not bundles:
        raise LimitationsError("config declares no bundles")
    return Config(bundles=bundles, path=config_path)


def render(bundle: str, entries: Sequence[Limitation], *, generated_utc: str) -> dict[str, Any]:
    """The JSON a bundle root carries. Sorted by id so two renders of one config are identical."""
    return {
        "schema": SCHEMA,
        "bundle": bundle,
        "generated_utc": generated_utc,
        "source_of_truth": CONFIG_RELPATH,
        "entries": [e.as_json() for e in sorted(entries, key=lambda e: e.id)],
    }


def write_rendered(bundle_dir: Path | str, config: Config | None = None, *,
                   generated_utc: str | None = None) -> Path:
    """Render a bundle's file from the config into its root, and return the path.

    The bundle is named by its directory. When the file already exists and nothing but the
    timestamp would change, the old timestamp is kept, so a no-op re-render is byte-identical
    and a diff of the bundle shows only real changes.
    """
    bundle = Path(bundle_dir)
    config = config or load_config()
    name = bundle.name
    if name not in config.bundles:
        raise LimitationsError(f"{name!r} has no block in {CONFIG_RELPATH}")
    stamp = generated_utc or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rendered = render(name, config.bundles[name], generated_utc=stamp)
    target = bundle / FILE_NAME
    if target.is_file():
        try:
            old = json.loads(target.read_text(encoding="utf-8"))
            if {k: v for k, v in old.items() if k != "generated_utc"} == \
                    {k: v for k, v in rendered.items() if k != "generated_utc"}:
                rendered["generated_utc"] = old["generated_utc"]
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    target.write_text(json.dumps(rendered, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")
    return target


def problems_in_file(path: Path, expected_bundle: str | None = None,
                     config: Config | None = None) -> list[str]:
    """Why a bundle's KNOWN_LIMITATIONS.json is not acceptable. Empty means it is."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{FILE_NAME} unreadable: {exc}"]
    problems: list[str] = []
    if raw.get("schema") != SCHEMA:
        problems.append(f"{FILE_NAME} schema is {raw.get('schema')!r}, expected {SCHEMA!r}")
    bundle = str(raw.get("bundle") or "")
    if expected_bundle and bundle != expected_bundle:
        problems.append(f"{FILE_NAME} names bundle {bundle!r}, directory is {expected_bundle!r}")
    entries = raw.get("entries")
    if not isinstance(entries, list):
        return problems + [f"{FILE_NAME} entries must be a list"]
    try:
        parsed = parse_entries(entries, FILE_NAME)
    except LimitationsError as exc:
        return problems + [str(exc)]
    if config is not None:
        expected = config.bundles.get(bundle)
        if expected is None:
            # Reported by the caller as a WARN, not a FAIL: the file is present and well-formed,
            # which is the hard rule; only the drift guard against the repository copy cannot
            # run for a bundle the repository does not know (a fixture, a scratch generation).
            problems.append(
                f"UNVERIFIED: {FILE_NAME} bundle {bundle!r} has no block in {CONFIG_RELPATH}; "
                "its contents could not be checked against the repository copy"
            )
        else:
            got = [e.as_json() for e in sorted(parsed, key=lambda e: e.id)]
            want = [e.as_json() for e in sorted(expected, key=lambda e: e.id)]
            if got != want:
                problems.append(
                    f"{FILE_NAME} differs from {CONFIG_RELPATH}; re-render it "
                    "(scripts/write_known_limitations.py)"
                )
    return problems
