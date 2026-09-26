"""Give a run an identity: a hash of its config, code and source identity.

``PIPELINE_GOVERNANCE.md`` §10.1 requires a ``run_id`` that hashes config, code and source
identity. ``run_id`` stays a per-take descriptive label (``'gaitex-austra-gwo-unified8-marker'``)
because it is a required manifest field; ``run_identity`` is a separate hash of config, code and
source identity, so two runs from different code can be told apart (see
``docs/guides/GENERATION_PIPELINE_STANDARD.md`` §3.1, which records the difference from §10.1's
wording).

The hash is over a canonical JSON encoding: sorted keys, no insignificant whitespace, so the same
inputs give the same identity on any machine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

#: 32 hex characters. Long enough that collision is not a practical concern, short enough to be a
#: directory name a person can read back.
IDENTITY_LENGTH = 32


@dataclass(frozen=True)
class RunIdentityInputs:
    """Everything §10.1 asks to be folded into the identity.

    A field left None is recorded as null and still participates in the hash, so a run made
    without a body model is distinguishable from one made with it -- rather than silently
    hashing to the same value.
    """

    spec_id: str
    spec_version: str
    source_name: str
    entrypoint: str
    resolved_config_sha256: str | None = None
    code_revision: str | None = None
    code_dirty: bool | None = None
    input_corpus: str | None = None
    input_corpus_sha256: str | None = None
    body_model_sha256: str | None = None

    def as_canonical_mapping(self) -> dict[str, object]:
        return {
            "spec_id": self.spec_id,
            "spec_version": self.spec_version,
            "source_name": self.source_name,
            "entrypoint": self.entrypoint,
            "resolved_config_sha256": self.resolved_config_sha256,
            "code_revision": self.code_revision,
            "code_dirty": self.code_dirty,
            "input_corpus": self.input_corpus,
            "input_corpus_sha256": self.input_corpus_sha256,
            "body_model_sha256": self.body_model_sha256,
        }


def canonical_json(mapping: Mapping[str, object]) -> str:
    """Sorted keys, tight separators, no ASCII escaping games. Stable across machines."""
    return json.dumps(mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute(inputs: RunIdentityInputs) -> str:
    payload = canonical_json(inputs.as_canonical_mapping())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:IDENTITY_LENGTH]


def run_directory_name(inputs: RunIdentityInputs) -> str:
    """``run_<identity>`` -- prefixed so a run directory is recognisable among sibling paths."""
    return f"run_{compute(inputs)}"


def explain(inputs: RunIdentityInputs) -> dict[str, object]:
    """The identity together with what produced it, so a reader can recompute rather than trust."""
    mapping = inputs.as_canonical_mapping()
    return {
        "run_identity": compute(inputs),
        "algorithm": f"sha256(canonical_json(inputs))[:{IDENTITY_LENGTH}]",
        "governance_reference": "PIPELINE_GOVERNANCE.md section 10.1",
        "inputs": mapping,
        "canonical_json": canonical_json(mapping),
    }


#: What an input corpus calls its own summary. Three names because the corpora disagree:
#: hknu and addbio keep SUMMARY.json, gaitex keeps _run.json, and the unified8 bundles keep
#: INDEX.json. Without _run.json here a gaitex run record would claim no source asset at all.
CORPUS_SUMMARY_NAMES = ("SUMMARY.json", "INDEX.json", "_run.json")


def corpus_fingerprint(corpus_root: Path | str) -> str | None:
    """A stable stand-in for "which input corpus was this".

    Hashing every take of a 40,000-take corpus on each run would cost more than it is worth, so
    the corpus's own summary is hashed instead. Those already record the corpus's shape and
    counts, so a corpus that changed will change this. Returns None when none of them exists,
    and the caller records that as unavailable rather than substituting something weaker.
    """
    root = Path(corpus_root)
    for name in CORPUS_SUMMARY_NAMES:
        candidate = root / name
        if candidate.is_file():
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()
    return None
