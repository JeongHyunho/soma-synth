"""Write the seven run-directory artifacts ``PIPELINE_GOVERNANCE.md`` §10.1 requires.

The take manifests carry the field contract from §12.2; the run directory records the *run*:
what was configured, what it read, what it skipped, what it excluded and why.

Two of the seven are written when the run opens (``resolved_config.yaml``, ``environment.lock``)
because they describe intent and must not be back-filled from what happened. One accumulates
(``source_assets.json``). Three are written by ``close_run``. ``logs/`` exists from the start; the
runner fills it (``runner.RUNNER_LOG``, and each child step's output teed by ``pipeline.generate``).

Beside the seven, a run that generated a bundle writes ``source_manifest.json``: the source files
its take manifests name, with their hashes (ADR-0041; teammates' bundles stay on their PCs, so a
licence withdrawal is traced through the run records), and in its ``source_files`` block every
source file the run's corpus and bundle stages could read, with the SHA-256 of the whole file
(``source_trace.py``). ``manifest.json`` carries its count and
aggregate digest in a ``tracing`` block, together with the code and ``smpl18`` revisions and the
decision the run generated under.

A run directory is immutable once closed. Reopening one is a new run with a new identity; the
runner gives an identical repeat a numbered sibling (``run_<identity>-2``) rather than reopening.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import yaml

from soma_synth.pipeline import provenance as provenance_mod
from soma_synth.pipeline.run_identity import RunIdentityInputs, explain

#: §10.1, verbatim and in its order.
REQUIRED_ARTIFACTS = (
    "resolved_config.yaml",
    "manifest.json",
    "environment.lock",
    "source_assets.json",
    "metrics.json",
    "exclusions.json",
    "logs",
)

_OPEN_ARTIFACTS = ("resolved_config.yaml", "environment.lock", "logs")
_CLOSE_ARTIFACTS = ("manifest.json", "metrics.json", "exclusions.json")
#: Written once the bundle exists; not one of §10.1's seven.
SOURCE_MANIFEST = "source_manifest.json"


class RunDirectoryError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )


@dataclass
class RunDirectory:
    """One run's immutable record. Open it, record as you go, close it once."""

    root: Path
    identity_inputs: RunIdentityInputs
    provenance: provenance_mod.ProvenanceRecord
    resolved_config: Mapping[str, object] = field(default_factory=dict)
    _closed: bool = field(default=False, init=False)
    _skipped: list[str] = field(default_factory=list, init=False)
    _excluded: list[dict[str, object]] = field(default_factory=list, init=False)
    _stage_counts: dict[str, int] = field(default_factory=dict, init=False)
    _opened_utc: str = field(default="", init=False)
    _generation_decision: dict[str, object] | None = field(default=None, init=False)
    _corpus: dict[str, object] | None = field(default=None, init=False)
    _bundle: dict[str, object] | None = field(default=None, init=False)
    _source_manifest: dict[str, object] | None = field(default=None, init=False)
    _replaced: dict[str, object] = field(default_factory=dict, init=False)

    # ---------------------------------------------------------------- open
    def open(self) -> "RunDirectory":
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(exist_ok=True)
        self._opened_utc = _utc_now()

        (self.root / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "opened_utc": self._opened_utc,
                    "run_identity": explain(self.identity_inputs),
                    "config": dict(self.resolved_config),
                },
                allow_unicode=True, sort_keys=False,
            ),
            encoding="utf-8", newline="\n",
        )
        _write_json(
            self.root / "environment.lock",
            {
                "dependency_versions": provenance_mod.dependency_versions(),
                "host_environment": provenance_mod.host_environment(),
                "code_revision": provenance_mod.code_revision(),
                "smpl18_revision": provenance_mod.smpl18_revision(),
            },
        )
        self._flush_source_assets()
        return self

    # ------------------------------------------------------------- record
    def record_source_asset(self, key: str, path: Path | str) -> None:
        self.provenance.record_source_asset(key, path)
        self._flush_source_assets()

    def record_output_artifact(self, key: str, path: Path | str) -> None:
        self.provenance.record_output_artifact(key, path)

    def record_stage(self, stage: str) -> None:
        self._stage_counts[stage] = self._stage_counts.get(stage, 0) + 1

    def record_skip(self, take_id: str) -> None:
        """A skipped take is recorded. A silent skip is not a resume, it is an omission."""
        self._skipped.append(take_id)

    def record_exclusion(self, take_id: str, reason: str) -> None:
        self._excluded.append({"take_id": take_id, "reason": reason})

    def record_generation_decision(self, decision: Mapping[str, object]) -> None:
        """What the gates said, and -- when an owner record overrode them -- which record.

        Kept in the run manifest rather than only in ``logs/`` so that a bundle made under the
        holds carries the authority it ran on in the same file as its identity.
        """
        self._generation_decision = dict(decision)

    def record_corpus(self, corpus: Mapping[str, object]) -> None:
        """Which retarget corpus the bundle was built from, whether this run built it, and its
        fingerprint. Kept in the run manifest beside the identity, which could only fingerprint a
        corpus that already existed when the run opened."""
        self._corpus = dict(corpus)

    def record_bundle(self, bundle: Mapping[str, object]) -> None:
        """What the generate step found and did in the bundle directory: what it held at open
        (an INDEX.json, an unfinished generation, other files), whether that was moved aside for
        a fresh build, and whether the generation completed. Later calls add to the block."""
        self._bundle = {**(self._bundle or {}), **dict(bundle)}

    def record_replaced(self, what: str, entry: Mapping[str, object]) -> None:
        """A directory the run moved aside to build ``what`` (the bundle or the corpus) fresh:
        where it went, what it held, the evidence files copied into ``replaced/``, and what
        became of it (deleted, kept, restored). Later calls add to the entry."""
        self._replaced[what] = {**self._replaced.get(what, {}), **dict(entry)}

    def record_source_manifest(self, payload: Mapping[str, object]) -> Path:
        """Write ``source_manifest.json`` (see :func:`provenance.source_manifest`) and keep its
        summary for the run manifest's ``tracing`` block."""
        path = self.root / SOURCE_MANIFEST
        _write_json(path, payload)
        self._source_manifest = {
            "path": SOURCE_MANIFEST,
            "count": payload.get("count"),
            "manifests_scanned": payload.get("manifests_scanned"),
            "aggregate_sha256": payload.get("aggregate_sha256"),
            "aggregate_algorithm": payload.get("aggregate_algorithm"),
            "file_sha256": provenance_mod.file_sha256(path),
        }
        traced = payload.get("source_files")
        if isinstance(traced, Mapping):
            # the full-file hashes of the source files the run's stages could read
            # (source_trace.py): their count, size and aggregate, beside the references'
            self._source_manifest["source_files"] = {
                key: traced.get(key)
                for key in ("status", "reason", "folder", "count", "bytes", "seconds",
                            "aggregate_sha256", "aggregate_algorithm")
                if key in traced
            }
        return path

    def _tracing(self, provenance: Mapping[str, object]) -> dict[str, object]:
        """ADR-0041's three items in one place: the source hash list, the code and smpl18
        revisions, and the decision the run generated under."""
        decision = self._generation_decision or {}
        if decision.get("basis") == "standing_decision":
            record = decision.get("decision_record")
            record_sha256 = decision.get("decision_record_sha256")
        else:
            authorization = decision.get("authorization")
            record = authorization.get("record") if isinstance(authorization, dict) else None
            record_sha256 = decision.get("authorization_record_sha256") if record else None
        return {
            "reference": "ADR-0041; PIPELINE_GOVERNANCE.md section 12",
            "source_manifest": self._source_manifest or provenance_mod.Unavailable(
                "no bundle was generated by this run, so no take manifest was read"
            ).as_json(),
            "corpus_fingerprint": (self._corpus or {}).get("fingerprint_sha256"),
            "code_revision": provenance.get("code_revision"),
            "smpl18_revision": provenance.get("smpl18_revision"),
            # what the run got (None: it took no decision, or was refused before it), beside what
            # resolved_config.yaml says it set out to ask for
            "generation_basis": decision.get("basis") if decision else None,
            "generation_basis_intended": self.resolved_config.get("generation_basis_intended"),
            "decision_record": record,
            "decision_record_sha256": record_sha256,
        }

    def _flush_source_assets(self) -> None:
        _write_json(self.root / "source_assets.json", self.provenance.source_assets)

    # --------------------------------------------------------------- close
    def close(
        self,
        *,
        takes_written: int,
        takes_trusted: int = 0,
        quality: Mapping[str, object] | None = None,
    ) -> None:
        if self._closed:
            raise RunDirectoryError("run directory already closed; a reopen is a new run")
        self.provenance.quality = dict(quality or {})

        # `takes_trusted` is separate from `takes_written` and from `takes_skipped`, because a
        # ledger-trusted take is a third thing: it was neither rebuilt nor passed over, it was
        # accepted on a prior run's evidence. Folding it into either would make a run that
        # accepted 8,261 takes report zero of everything.
        _write_json(
            self.root / "metrics.json",
            {
                "opened_utc": self._opened_utc,
                "closed_utc": _utc_now(),
                "takes_written": takes_written,
                "takes_trusted": takes_trusted,
                "takes_skipped": len(self._skipped),
                "skipped_take_ids": sorted(self._skipped),
                "stage_counts": dict(sorted(self._stage_counts.items())),
            },
        )
        _write_json(
            self.root / "exclusions.json",
            {"count": len(self._excluded), "excluded": self._excluded},
        )
        provenance = self.provenance.as_json()
        _write_json(
            self.root / "manifest.json",
            {
                "governance_reference": "PIPELINE_GOVERNANCE.md section 10.1",
                "note": (
                    "This is the RUN manifest. The per-take manifest.json is a different "
                    "document, governed by the paired contract section 12.2."
                ),
                "run_identity": explain(self.identity_inputs),
                "source_name": self.provenance.source_name,
                "spec_id": self.provenance.spec_id,
                "spec_version": self.provenance.spec_version,
                "provenance": provenance,
                "generation_decision": self._generation_decision,
                "corpus": self._corpus,
                "bundle": self._bundle,
                "replaced": self._replaced or None,
                "tracing": self._tracing(provenance),
                "distribution_scope": "internal_only",
            },
        )
        self._closed = True

    # ---------------------------------------------------------------- audit
    @staticmethod
    def missing_artifacts(root: Path | str) -> list[str]:
        """Which of the seven a directory lacks. Empty means §10.1 is satisfied."""
        base = Path(root)
        return [name for name in REQUIRED_ARTIFACTS if not (base / name).exists()]
