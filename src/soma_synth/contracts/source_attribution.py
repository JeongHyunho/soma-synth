"""What a generator writes into ``manifest.source_attribution``, read from the one registry.

ADR-0040 D1 made ``source_attribution`` a universal manifest field; ``dataset_profiles_v1.yaml``
says which keys it must carry; ``configs/datasets/source_attribution_v1.yaml`` holds the values
themselves, because a rights string is a config value and must not become a code constant.
Generators read the registry through this module at emit time, and
``scripts/backfill_manifest_fields.py`` reads it through this module too.

A source whose interpretation still awaits the data owner's confirmation cannot be written -- the
registry says so in its own words, and a generator that filled the field anyway would put an
unconfirmed rights claim into every manifest. ``resolved_for`` raises for such a source, and for
one the registry does not know.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

SCHEMA = "source_attribution_v1"
CONFIG_RELPATH = "configs/datasets/source_attribution_v1.yaml"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPOSITORY_ROOT / CONFIG_RELPATH


class AttributionError(ValueError):
    """The registry is malformed, or was asked for a source it cannot vouch for."""


@dataclass(frozen=True)
class AttributionConfig:
    required_keys: tuple[str, ...]
    resolved: Mapping[str, Mapping[str, str]]
    pending: Mapping[str, str]
    path: Path

    def value_for(self, source_name: str) -> dict[str, str]:
        if source_name in self.resolved:
            return dict(self.resolved[source_name])
        if source_name in self.pending:
            raise AttributionError(
                f"{source_name} attribution awaits the data owner's confirmation "
                f"({self.path.name}); "
                f"{self.pending[source_name]}"
            )
        raise AttributionError(f"{source_name} has no entry in {self.path.name}")


def load(path: Path | str | None = None) -> AttributionConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise AttributionError(f"attribution registry not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if raw.get("schema") != SCHEMA:
        raise AttributionError(f"expected schema {SCHEMA!r}, got {raw.get('schema')!r}")

    required = tuple(raw.get("required_keys") or ())
    if not required:
        raise AttributionError("required_keys is empty; the shared shape is the point of the file")

    resolved: dict[str, dict[str, str]] = {}
    pending: dict[str, str] = {}
    for source, block in (raw.get("sources") or {}).items():
        block = block or {}
        status = str(block.get("status") or "")
        if status == "resolved":
            value = block.get("value") or {}
            missing = [k for k in required if not str(value.get(k) or "").strip()]
            if missing:
                raise AttributionError(
                    f"{source}: resolved attribution is missing {', '.join(missing)}"
                )
            resolved[source] = {k: str(v).strip() for k, v in value.items()}
        # The status token of a source awaiting the data owner's confirmation. It keeps the
        # spelling of source_attribution_v1, a used schema, so it is not renamed.
        elif status == "pending_user_confirmation":
            pending[source] = str(block.get("blocked_on") or "no reason recorded").strip()
        else:
            raise AttributionError(f"{source}: unknown status {status!r}")

    return AttributionConfig(
        required_keys=required, resolved=resolved, pending=pending, path=config_path
    )


def resolved_for(source_name: str, path: Path | str | None = None) -> dict[str, str]:
    """The block a generator writes for ``source_name``; raises where it must not write one."""
    return load(path).value_for(source_name)
