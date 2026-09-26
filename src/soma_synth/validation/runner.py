"""Orchestrate L0-L4 over a dataset dir into a ValidationReport.

Checks every ``ok`` take (optionally a sample) at the requested level. Given a validation ledger
(``datasets/catalog.py``), a take whose content, spec version, generator and validator digests
all match a prior PASS is trusted instead of checked again (``takes_trusted``); ``full`` checks
every take.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from soma_synth.contracts import qmd_unified8_smpl18_spec as spec
from soma_synth.validation import checks
from soma_synth.validation.report import ValidationReport

_LEVELS = ("L0", "L1", "L2", "L3", "L4")


def _check_take(td: Path, take_id: str, depth: int, bands: checks.PhysicalBands) -> list:
    findings = list(checks.check_take_l1(td, take_id))
    if depth >= _LEVELS.index("L2"):
        findings += checks.check_take_l2(td, take_id, bands=bands)
    if depth >= _LEVELS.index("L3"):
        findings += checks.check_take_l3_axes(td, take_id)
    if depth >= _LEVELS.index("L4"):
        findings += checks.check_take_l4_governance(td, take_id)
    return findings


def validate(
    dataset_dir,
    level: str = "L2",
    sample: int | None = None,
    bands: checks.PhysicalBands = checks.DEFAULT_BANDS,
    ledger=None,
    generator_digest: str = "",
    validator_digest: str | None = None,
    full: bool = False,
    conformance: bool = False,
    pending_files: Iterable[str] = (),
) -> ValidationReport:
    """Validate a dataset dir at ``level``. With ``ledger``, skip takes whose freshness key matches a prior
    PASS (unless ``full``); newly checked takes are recorded (caller saves the ledger).

    ``conformance`` additionally judges the bundle against the dataset profile registry (ADR-0040):
    the root files it owes, its INDEX take-entry keys, manifest keys it declares nowhere, and
    waivers that have outlived their expiry. It defaults off so that existing callers and fixture
    bundles are unaffected; the CLI turns it on. ``pending_files`` names the bundle-root files the
    caller writes after this validation whatever its outcome (its report, its ledger): the
    conformance judgement does not report them missing. A file only a later step writes, and only
    after a validation that passed (a run's README.md), is not pending: see
    ``checks.check_dataset_profile_conformance``."""
    dataset_dir = Path(dataset_dir)
    if level not in _LEVELS:
        raise ValueError(f"unknown level:{level}")
    depth = _LEVELS.index(level)

    rep = ValidationReport(dataset_dir, spec.SPEC_ID, spec.SPEC_VERSION)
    rep.add(checks.check_l0_structure(dataset_dir))
    if conformance:
        rep.add(checks.check_dataset_profile_conformance(dataset_dir,
                                                         pending_files=pending_files))
    if depth < _LEVELS.index("L1"):
        return rep

    if ledger is not None and validator_digest is None:
        from soma_synth.datasets import catalog

        validator_digest = catalog.default_validator_digest()

    idx = checks.read_index(dataset_dir)
    ok_takes = [t for t in idx["takes"] if t["status"] == "ok"]
    if sample is not None:
        ok_takes = ok_takes[:sample]

    for t in ok_takes:
        td = dataset_dir / t["rel"]
        if not td.is_dir():
            continue
        take_id = t["take_id"]
        key = None
        if ledger is not None:
            from soma_synth.datasets import catalog

            _sm, _src, caps, _f = checks.take_capabilities(td)
            key = catalog.FreshnessKey(
                catalog.take_digest(td, spec.expected_deliverables(caps)),
                spec.SPEC_VERSION,
                generator_digest,
                validator_digest,
            )
            if not full and ledger.is_fresh(take_id, key):
                rep.takes_trusted += 1
                continue

        findings = _check_take(td, take_id, depth, bands)
        if conformance:
            findings += checks.check_take_profile_conformance(td, take_id)
        rep.add(findings)
        rep.takes_checked += 1
        if ledger is not None:
            status = "FAIL" if any(f.severity == "FAIL" for f in findings) else "PASS"
            ledger.record(take_id, key, status)

    if depth >= _LEVELS.index("L4"):
        rep.add(checks.check_l4_dataset(dataset_dir))
    return rep
