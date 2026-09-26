"""Validation findings + report (machine JSON + ASCII human summary, cp949-safe)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Finding:
    level: str          # "L0".."L4"
    check: str          # short slug, e.g. "schema", "physical"
    severity: str       # "FAIL" | "WARN"
    message: str
    take_id: str | None = None
    artifact: str | None = None

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "take_id": self.take_id,
            "artifact": self.artifact,
        }


class ValidationReport:
    def __init__(self, dataset_dir, spec_id: str, spec_version: str) -> None:
        self.dataset_dir = str(dataset_dir)
        self.spec_id = spec_id
        self.spec_version = spec_version
        self.findings: list[Finding] = []
        self.takes_checked = 0
        self.takes_trusted = 0

    def add(self, finding: Finding | Iterable[Finding]) -> None:
        if isinstance(finding, Finding):
            self.findings.append(finding)
        else:
            self.findings.extend(finding)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "FAIL"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "WARN"]

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "dataset_dir": self.dataset_dir,
            "spec_id": self.spec_id,
            "spec_version": self.spec_version,
            "ok": self.ok,
            "takes_checked": self.takes_checked,
            "takes_trusted": self.takes_trusted,
            "counts": {"FAIL": len(self.failures), "WARN": len(self.warnings)},
            "findings": [f.as_dict() for f in self.findings],
        }

    def summary(self, limit: int = 20) -> str:
        lines = [
            f"validation {self.spec_id}:{self.spec_version}  ->  {'PASS' if self.ok else 'FAIL'}",
            f"  dataset: {self.dataset_dir}",
            f"  takes checked={self.takes_checked} trusted={self.takes_trusted}",
            f"  FAIL={len(self.failures)}  WARN={len(self.warnings)}",
        ]
        for f in self.failures[:limit]:
            where = f.take_id or "-"
            lines.append(f"  FAIL [{f.level}/{f.check}] {where}: {f.message}")
        extra = len(self.failures) - limit
        if extra > 0:
            lines.append(f"  ... (+{extra} more FAIL)")
        return "\n".join(lines)
