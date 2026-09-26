"""Judge an existing bundle against §10.1 and §10.2, and retrofit what can still be recovered.

Two jobs, deliberately in one module because they share the same table of what is required.

``audit_bundle`` answers "how much of the governance provenance does this bundle actually
carry", counted against the items of §10.2 and the seven run-directory artifacts of §10.1.

``retrofit_provenance`` fills in what is still measurable today and says plainly what is not.
Most of the gap is not recoverable after the fact -- nothing can say which commit produced a
bundle that did not record it -- and the retrofit records that rather than guessing. A provenance
file that quietly invents its own history is worse than the absence it replaces.

The retrofit writes a sidecar and never touches take manifests, so a validated bundle is not
changed by it; the record it adds is, by construction, partial.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from soma_synth.pipeline import provenance as provenance_mod
from soma_synth.pipeline.run_directory import REQUIRED_ARTIFACTS

SIDECAR_NAME = "PROVENANCE_RETROFIT.json"

#: §10.2's ten items, and where each would be evidenced in a take manifest. The tokens are what
#: the audit looks for; absence of all of them means the item is not carried.
PROVENANCE_ITEMS: Mapping[str, tuple[str, ...]] = {
    "code_revision": ("code_hash", "code_revision", "git_revision", "dirty"),
    "contract_versions": ("contract_version", "spec_version", "canonicalization_config"),
    "source_asset_and_license": ("source_asset_sha256", "license"),
    "spec_snapshot": ("spec_documents", "spec_hash", "spec_snapshot"),
    "dependency_versions": ("python_version", "dependencies", "environment"),
    "body_model_hash": ("model_hash", "body_model_sha256", "smpl_model_sha256"),
    "seeds": ("seed",),
    "host_environment": ("host_environment", "platform"),
    "artifact_hashes": ("artifact_sha256", "deliverable_sha256", "output_hash"),
    "quality_and_quarantine": ("quality_gate", "validation", "excluded", "quarantine"),
}

#: Items nothing can recover once the run is over. Listed so the retrofit refuses them by name
#: rather than appearing to have tried.
NOT_RECOVERABLE = {
    "code_revision": "the producing commit was never recorded and cannot be inferred from output",
    "dependency_versions": "the interpreter and library versions at generation time are gone",
    "host_environment": "the producing host was never recorded",
}


@dataclass(frozen=True)
class BundleAudit:
    dataset_dir: Path
    source_name: str
    present_items: tuple[str, ...]
    missing_items: tuple[str, ...]
    missing_run_artifacts: tuple[str, ...]

    @property
    def provenance_coverage(self) -> tuple[int, int]:
        return len(self.present_items), len(PROVENANCE_ITEMS)

    def as_json(self) -> dict[str, object]:
        present, total = self.provenance_coverage
        return {
            "dataset_dir": str(self.dataset_dir),
            "source_name": self.source_name,
            "governance_reference": "PIPELINE_GOVERNANCE.md sections 10.1 and 10.2",
            "provenance_items_present": list(self.present_items),
            "provenance_items_missing": list(self.missing_items),
            "provenance_coverage": f"{present}/{total}",
            "run_directory_artifacts_missing": list(self.missing_run_artifacts),
        }


def _first_take(dataset_dir: Path) -> Path | None:
    for child in sorted(dataset_dir.iterdir()):
        if child.is_dir() and not child.name.startswith("_") and (child / "manifest.json").exists():
            return child
    return None


def _source_name(manifest: Mapping[str, object]) -> str:
    source = manifest.get("source")
    if isinstance(source, Mapping):
        return str(source.get("source_name", ""))
    return ""


def audit_bundle(dataset_dir: Path | str) -> BundleAudit:
    """What this bundle carries of what §10.1 and §10.2 require."""
    root = Path(dataset_dir)
    take = _first_take(root)
    if take is None:
        raise FileNotFoundError(f"no take with a manifest under {root}")

    text = (take / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(text)
    lowered = text.lower()

    present = tuple(
        item for item, tokens in PROVENANCE_ITEMS.items()
        if any(token.lower() in lowered for token in tokens)
    )
    missing = tuple(item for item in PROVENANCE_ITEMS if item not in present)

    return BundleAudit(
        dataset_dir=root,
        source_name=_source_name(manifest),
        present_items=present,
        missing_items=missing,
        missing_run_artifacts=tuple(
            name for name in REQUIRED_ARTIFACTS if not (root / name).exists()
        ),
    )


def body_model_set_hashes(model_dir: Path | str | None) -> dict[str, object]:
    """Hash every model in the set, because the generators choose one per subject.

    Naming a single file would misdescribe every generator here: all five call
    ``clean_model_path_for_gender``, so a bundle of mixed-sex subjects was built from more than
    one of these files. Hashing the set records exactly what was available to it.
    """
    if model_dir is None:
        return provenance_mod.Unavailable("no body model directory supplied").as_json()
    root = Path(model_dir)
    if not root.is_dir():
        return provenance_mod.Unavailable(f"not a directory: {root}").as_json()
    models = sorted(root.glob("*.npz"))
    if not models:
        return provenance_mod.Unavailable(f"no .npz under {root.name}").as_json()
    return {
        "directory": root.name,
        "selection": "by_subject_gender",
        "models": {p.name: provenance_mod.file_sha256(p) for p in models},
    }


def retrofit_provenance(
    dataset_dir: Path | str,
    *,
    body_model_path: Path | str | None = None,
    write: bool = True,
) -> dict[str, object]:
    """Record what is still measurable about an already-generated bundle.

    Measurable now: the hashes of the artifacts as they sit on disk, and the body model if the
    caller can point at the one that was used. Not measurable: anything about the machine or the
    code at the time, which is why ``NOT_RECOVERABLE`` names those explicitly instead of leaving
    them to look merely absent.

    The result is a sidecar. The take manifests are not touched.
    """
    root = Path(dataset_dir)
    audit = audit_bundle(root)

    artifacts: dict[str, object] = {}
    take = _first_take(root)
    if take is not None:
        for deliverable in sorted(p for p in take.iterdir() if p.is_file()):
            artifacts[f"{take.name}/{deliverable.name}"] = provenance_mod.file_sha256(deliverable)
    for name in ("INDEX.json", "VALIDATION_REPORT.json", "DATA_DESCRIPTION_EN.md"):
        candidate = root / name
        if candidate.exists():
            artifacts[name] = provenance_mod.file_sha256(candidate)

    record: dict[str, object] = {
        "note": (
            "Retrofitted after generation. Items that could only have been captured while the "
            "run was happening are named as unrecoverable rather than estimated."
        ),
        "governance_reference": "PIPELINE_GOVERNANCE.md section 10.2",
        "dataset_dir": root.name,
        "source_name": audit.source_name,
        "audit": audit.as_json(),
        "recovered": {
            "artifact_hashes": artifacts,
            "body_model_set": body_model_set_hashes(body_model_path),
            "seeds": provenance_mod.seeds(),
        },
        "unrecoverable": {
            item: provenance_mod.Unavailable(reason).as_json()
            for item, reason in NOT_RECOVERABLE.items()
        },
        "retrofitted_by": {
            "dependency_versions": provenance_mod.dependency_versions(),
            "host_environment": provenance_mod.host_environment(),
            "code_revision": provenance_mod.code_revision(),
            "warning": (
                "These describe the machine that wrote this sidecar, NOT the machine that "
                "generated the bundle. They are not a substitute for the unrecoverable items."
            ),
        },
    }

    if write:
        (root / SIDECAR_NAME).write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n",
        )
    return record
