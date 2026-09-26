"""Read the generation gates, so the runner refuses on evidence rather than on a constant.

The parent project's hold rules forbid generating while ``DETERMINISTIC_EXECUTION_CONFIG_HOLD``
stands, and keep that hold and five others until an independent approved transition. A runner
that enforced this with a hardcoded ``ALLOW_GENERATION = False`` would be worthless: the next
person edits the constant. So the answer comes from the gate config the M0 evaluator pins, which
says it in its own words --

    authority_boundary.effect = "READ_ONLY_AUDIT_AUTHORIZED_GENERATION_STILL_STAGED"
    state_machines.runtime.current = "RUNTIME_APPROVAL_HOLD"
    derived_states.ALL_SOURCE_PASS.current = false

The active evaluator is identified by convention -- the highest ``m0_v1_N`` under
``configs/evaluators`` -- because nothing in the governance root points at it.
The decision records which evaluator and which gate config it consulted, so a wrong convention
shows up in the record instead of hiding in it.

One exception to the hold rule: the owner may authorise an
``experimental_non_candidate`` generation of one source in a dated record under
``research/decisions/``. The runner is handed that record's path and reads it; the record must
name the source, the artifact class and the date, and must say in so many words that it releases
no hold. Nothing else unlocks generation -- not a flag, not an environment variable -- so the
refusal stays the default and the exception leaves a file a reader can open, which the run
directory carries.

**The governance root.** The evaluator, its gate config and the owner's decision records belong
to the parent project, not to soma-synth. ``GOVERNANCE_ROOT`` is the checkout every one of them
is read from, and it is fixed once, fail-closed, by where this repository sits: when it is mounted
in the parent project as ``<parent>/packages/soma-synth`` and ``<parent>`` has both
``configs/evaluators`` and ``research/decisions``, the governance root is ``<parent>``; in any
other layout (a standalone soma-synth checkout, a parent without either folder) it is ``None``.
No flag and no environment variable changes it. Without it the gate check and an owner record
(``--authorized-by``) raise :class:`GateError` naming the absent governance root, which the runner
turns into a refusal. A run without an owner record never gets here: a registry v2 carries the
standing decision (ADR-0041) and the runner records that instead.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
#: Where the parent project mounts this repository, relative to the parent's root.
MOUNT_RELPATH = Path("packages") / "soma-synth"
#: The parent's folders that make it a governance root, relative to it.
EVALUATOR_RELPATH = Path("configs") / "evaluators"
#: Where an owner's authorization record must live. A file anywhere else is not one.
DECISIONS_RELPATH = Path("research") / "decisions"


def resolve_governance_root(repository_root: Path) -> Path | None:
    """The parent project's root when ``repository_root`` is its ``packages/soma-synth`` and it
    has both ``configs/evaluators`` and ``research/decisions``; otherwise None (fail-closed)."""
    root = Path(repository_root)
    if (root.name, root.parent.name) != (MOUNT_RELPATH.name, MOUNT_RELPATH.parent.name):
        return None
    parent = root.parent.parent
    if (parent / EVALUATOR_RELPATH).is_dir() and (parent / DECISIONS_RELPATH).is_dir():
        return parent
    return None


#: The checkout holding the M0 evaluator and the decision records, or None when this repository
#: is not mounted in the parent project (see the module docstring). Tests monkeypatch it.
GOVERNANCE_ROOT: Path | None = resolve_governance_root(_REPOSITORY_ROOT)


def governance_evaluator_dir() -> Path | None:
    """``configs/evaluators`` under the governance root, or None without one."""
    return GOVERNANCE_ROOT / EVALUATOR_RELPATH if GOVERNANCE_ROOT is not None else None


def governance_decisions_dir() -> Path | None:
    """``research/decisions`` under the governance root, or None without one."""
    return GOVERNANCE_ROOT / DECISIONS_RELPATH if GOVERNANCE_ROOT is not None else None


#: The gate config says this when generation has not been authorised.
_STAGED_EFFECT = "GENERATION_STILL_STAGED"

AUTHORIZATION_SCHEMA = "generation_authorization_v1"
#: The only artifact class a record may authorise. Canonical output stays behind the holds.
AUTHORIZABLE_ARTIFACT_CLASS = "experimental_non_candidate"
_AUTHORIZATION_BLOCK = re.compile(r"```yaml authorization[ \t]*\r?\n(.*?)\r?\n```", re.S)
_AUTHORIZATION_KEYS = (
    "source_name", "artifact_class", "authorized_on", "authorized_by", "scope", "releases_holds",
)


class GateError(RuntimeError):
    pass


@dataclass(frozen=True)
class Authorization:
    """An owner's recorded decision that one source may be generated as experimental output."""

    record: str
    source_name: str
    artifact_class: str
    authorized_on: str
    authorized_by: str
    scope: str

    def as_json(self) -> dict[str, object]:
        return {
            "record": self.record,
            "source_name": self.source_name,
            "artifact_class": self.artifact_class,
            "authorized_on": self.authorized_on,
            "authorized_by": self.authorized_by,
            "scope": self.scope,
            "releases_holds": False,
        }


@dataclass(frozen=True)
class GenerationDecision:
    """Whether generation may run, and the evidence either way."""

    allowed: bool
    source_name: str
    evaluator: str
    gate_config: str
    reasons: tuple[str, ...]
    #: Present only when the gates refused and an owner record overrode them for this run.
    authorization: Authorization | None = None

    def as_json(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "source_name": self.source_name,
            "consulted_evaluator": self.evaluator,
            "consulted_gate_config": self.gate_config,
            "reasons": list(self.reasons),
            "authorization": self.authorization.as_json() if self.authorization else None,
        }

    def refusal_message(self) -> str:
        joined = "\n".join(f"  - {reason}" for reason in self.reasons)
        return (
            f"generation is not authorised for source {self.source_name!r}.\n"
            f"consulted {self.gate_config} (via {self.evaluator}):\n{joined}\n"
            "The parent project's generation hold forbids generating while "
            "DETERMINISTIC_EXECUTION_CONFIG_HOLD stands, and its hold-transition rule keeps that "
            "hold until an independent approved transition. Run with --skip-generate "
            "to exercise the rest of the pipeline, or hand the runner an owner authorization "
            "record from the parent project's research/decisions/ with --authorized-by."
        )

    def authorization_message(self) -> str:
        """What overrode the gates, and what the gates said. Written beside the run's logs."""
        if self.authorization is None:
            raise GateError("no authorization to describe")
        joined = "\n".join(f"  - {reason}" for reason in self.reasons)
        auth = self.authorization
        return (
            f"generation of source {self.source_name!r} ran as {auth.artifact_class} on the "
            f"owner's record {auth.record} ({auth.authorized_on}, {auth.authorized_by}).\n"
            f"scope: {auth.scope}\n"
            f"the gates still say no, and this record releases none of them:\n"
            f"consulted {self.gate_config} (via {self.evaluator}):\n{joined}\n"
        )


def governance_root_absent() -> str | None:
    """Why the gates cannot be read from this checkout, or None when the governance root is
    there with both of its folders (see the module docstring)."""
    root = GOVERNANCE_ROOT
    if root is None:
        where = (
            f"soma-synth at {_REPOSITORY_ROOT} is not mounted in the parent project as "
            f"{MOUNT_RELPATH.as_posix()} beside the parent's configs/evaluators and "
            "research/decisions"
        )
    else:
        missing = [relative.as_posix() for relative in (EVALUATOR_RELPATH, DECISIONS_RELPATH)
                   if not (root / relative).is_dir()]
        if not missing:
            return None
        where = f"{root} has no {' and no '.join(missing)}"
    return (
        f"the governance root is absent: {where}. The M0 evaluator, its gate config and the "
        "owner's decision records (research/decisions) belong to the parent project, not to "
        "soma-synth, so the per-source gate check and an owner record (--authorized-by) cannot "
        "be read here. A run without --authorized-by generates under the registry's standing "
        "decision (ADR-0041) instead"
    )


def _require_governance_root() -> Path:
    absent = governance_root_absent()
    if absent:
        raise GateError(absent)
    assert GOVERNANCE_ROOT is not None
    return GOVERNANCE_ROOT


def resolve_record_path(path: Path | str) -> Path:
    """An owner record's path as the runner and the gates read it: a relative path resolves
    against the governance root, so ``research/decisions/<record>.md`` names the same file from
    any working directory; an absolute path, or any path without a governance root, is kept."""
    record = Path(path)
    if GOVERNANCE_ROOT is not None and not record.is_absolute():
        return GOVERNANCE_ROOT / record
    return record


def _governance_relative(path: Path) -> str:
    """A path under the governance root as it names it (``configs/...``), else absolute."""
    resolved = path.resolve()
    if GOVERNANCE_ROOT is not None:
        try:
            return resolved.relative_to(GOVERNANCE_ROOT.resolve()).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


def active_evaluator_path() -> Path:
    """The highest-numbered ``m0_v1_N`` evaluator. Convention, and recorded as such."""
    evaluators = _require_governance_root() / EVALUATOR_RELPATH
    candidates: list[tuple[int, Path]] = []
    for path in evaluators.glob("m0_v1_*.yaml"):
        match = re.fullmatch(r"m0_v1_(\d+)\.yaml", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise GateError(f"no m0_v1_N evaluator under {evaluators}")
    return max(candidates)[1]


def _load(path: Path) -> dict:
    # These files are JSON with a .yaml extension, and some carry a BOM.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def active_gate_config_path() -> tuple[Path, Path]:
    """(evaluator, gate config). The evaluator names the gate config it pins."""
    evaluator_path = active_evaluator_path()
    evaluator = _load(evaluator_path)
    relative = (evaluator.get("gate_config") or {}).get("path")
    if not relative:
        raise GateError(f"{evaluator_path.name} names no gate_config")
    return evaluator_path, _require_governance_root() / relative


def read_authorization(path: Path | str, *,
                       decisions_dir: Path | str | None = None) -> Authorization:
    """Parse an owner's authorization record; refuse anything that is not exactly one.

    The record is prose for a reader with one fenced ``yaml authorization`` block for this
    function. Every requirement here is a presence or equality judgement: the block names the
    schema, the source, the artifact class, the date, the person and the scope, and says
    ``releases_holds: false`` in its own words -- a record that cannot say that is not one.

    ``decisions_dir`` defaults to the governance root's ``research/decisions``; without a
    governance root there is none, and the record is refused. A relative ``path`` resolves
    against the governance root (``resolve_record_path``).
    """
    if decisions_dir is None:
        decisions_dir = _require_governance_root() / DECISIONS_RELPATH
    record = resolve_record_path(path)
    if not record.is_file():
        raise GateError(f"authorization record not found: {record}")
    root = Path(decisions_dir).resolve()
    try:
        record.resolve().relative_to(root)
    except ValueError:
        raise GateError(
            f"an authorization record must live under {root}; {record} does not"
        ) from None

    match = _AUTHORIZATION_BLOCK.search(record.read_text(encoding="utf-8"))
    if not match:
        raise GateError(f"{record.name} carries no ```yaml authorization block")
    block = yaml.safe_load(match.group(1)) or {}
    if not isinstance(block, dict):
        raise GateError(f"{record.name}: the authorization block is not a mapping")
    if block.get("schema") != AUTHORIZATION_SCHEMA:
        raise GateError(
            f"{record.name}: authorization schema is {block.get('schema')!r}, "
            f"expected {AUTHORIZATION_SCHEMA!r}"
        )
    missing = [key for key in _AUTHORIZATION_KEYS if key not in block]
    if missing:
        raise GateError(f"{record.name}: authorization block is missing {', '.join(missing)}")
    if block["releases_holds"] is not False:
        raise GateError(
            f"{record.name}: releases_holds must be false; an authorization record cannot move "
            "a hold (the parent project's hold transitions)"
        )
    for key in ("source_name", "artifact_class", "authorized_by", "scope"):
        if not str(block.get(key) or "").strip():
            raise GateError(f"{record.name}: {key} is empty")
    try:
        _dt.date.fromisoformat(str(block["authorized_on"]))
    except ValueError:
        raise GateError(
            f"{record.name}: authorized_on {block['authorized_on']!r} is not an ISO date"
        ) from None

    return Authorization(
        record=_governance_relative(record),
        source_name=str(block["source_name"]).strip(),
        artifact_class=str(block["artifact_class"]).strip(),
        authorized_on=str(block["authorized_on"]),
        authorized_by=str(block["authorized_by"]).strip(),
        scope=str(block["scope"]).strip(),
    )


def generation_decision(
    source_name: str,
    authorized_by: Path | str | None = None,
    *,
    decisions_dir: Path | str | None = None,
) -> GenerationDecision:
    """May this source be generated right now? Every refusal cites the field that caused it.

    ``authorized_by`` names an owner record (see ``read_authorization``). When the gates refuse
    and the record authorises this very source as experimental output, the decision is allowed
    and carries both: the reasons the gates gave, which the record does not erase, and the
    record that overrode them for this run. A record for another source, for canonical output,
    or from outside ``research/decisions/`` is an error, not a refusal -- the caller asked for
    something that cannot be granted and should hear why. Everything is read from the
    governance root; without one this raises :class:`GateError` before reading anything.
    """
    evaluator_path, gate_path = active_gate_config_path()
    gate = _load(gate_path)

    reasons: list[str] = []

    effect = str((gate.get("authority_boundary") or {}).get("effect", ""))
    if _STAGED_EFFECT in effect:
        reasons.append(f"authority_boundary.effect = {effect!r}")

    runtime = (gate.get("state_machines") or {}).get("runtime") or {}
    runtime_state = str(runtime.get("current", ""))
    if runtime_state.endswith("_HOLD"):
        reasons.append(f"state_machines.runtime.current = {runtime_state!r}")

    source_state = str(((gate.get("state_machines") or {}).get(source_name) or {}).get("current", ""))
    if source_state.endswith("_HOLD"):
        reasons.append(f"state_machines.{source_name}.current = {source_state!r}")
    elif not source_state:
        reasons.append(f"state_machines has no entry for source {source_name!r}")

    all_pass = (gate.get("derived_states") or {}).get("ALL_SOURCE_PASS") or {}
    if all_pass.get("current") is False:
        reasons.append("derived_states.ALL_SOURCE_PASS.current = false")

    authorization: Authorization | None = None
    if reasons and authorized_by is not None:
        authorization = read_authorization(authorized_by, decisions_dir=decisions_dir)
        if authorization.source_name != source_name:
            raise GateError(
                f"{Path(authorized_by).name} authorises source {authorization.source_name!r}, "
                f"not {source_name!r}"
            )
        if authorization.artifact_class != AUTHORIZABLE_ARTIFACT_CLASS:
            raise GateError(
                f"{Path(authorized_by).name} names artifact_class "
                f"{authorization.artifact_class!r}; only {AUTHORIZABLE_ARTIFACT_CLASS!r} can be "
                "authorised by a record -- anything else needs the hold transition the parent "
                "project's governance describes"
            )

    return GenerationDecision(
        allowed=(not reasons) or authorization is not None,
        source_name=source_name,
        evaluator=evaluator_path.name,
        gate_config=_governance_relative(gate_path),
        reasons=tuple(reasons),
        authorization=authorization,
    )
