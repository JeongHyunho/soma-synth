"""The one place a synthetic bundle may declare how it differs from every other bundle.

Backed by ``configs/datasets/dataset_profiles_v1.yaml`` and authorised by ADR-0040. Before this
module the same per-source rule was written out four times -- once in
``qmd_unified8_smpl18_spec.derive_capabilities`` and three more in ``validation.checks`` -- so a
change to one of them silently disagreed with the other three.

The registry is a whitelist. ``universal`` is what every bundle owes; ``additive`` is what a
source legitimately carries **because of** a capability; ``waivers`` are the things that are
wrong and not yet fixed. A waiver carries ``reason``, ``remedy`` and ``expires``, and an expired
waiver is a failure -- without that, a temporary exemption becomes a permanent one by doing
nothing at all, which is how the drift ADR-0040 documents came about.

The registry holds no tolerance, threshold or numeric band; every rule it expresses is a
presence/equality judgement (DETERMINISTIC_EXECUTION_CONFIG_HOLD).
"""

from __future__ import annotations

import datetime as _dt
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

SCHEMA = "dataset_profiles_v1"
#: Repository-relative, so the record a report cites is the record a reader can open.
REGISTRY_RELPATH = "configs/datasets/dataset_profiles_v1.yaml"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = _REPOSITORY_ROOT / REGISTRY_RELPATH

#: Waiver fields that may not be omitted. `expires` matters most: without it a waiver never lapses.
_REQUIRED_WAIVER_FIELDS = ("target", "reason", "remedy", "expires")


class ProfileError(ValueError):
    """The registry is malformed, or something was asked of it that it cannot answer."""


@dataclass(frozen=True)
class Waiver:
    target: str
    reason: str
    remedy: str
    expires: _dt.date

    def is_expired(self, today: _dt.date) -> bool:
        return today > self.expires


@dataclass(frozen=True)
class Universal:
    bundle_files: frozenset[str]
    #: Needed before a bundle may be shared, but produced after it is first validated -- two of
    #: them by validation itself. Missing ones are a WARN, never a FAIL, because requiring
    #: validation's own output in order to validate would be circular.
    distribution_files: frozenset[str]
    take_deliverables: frozenset[str]
    manifest_keys: frozenset[str]
    #: Keys every ``source_attribution`` block owes. ADR-0040 D1 made the field universal but
    #: left its shape open; these are the intersection of the two bundles that had one.
    source_attribution_keys: frozenset[str]
    index_take_entry_keys: frozenset[str]
    string_arrays_dtype: str


@dataclass(frozen=True)
class Profile:
    source_name: str
    small_mode: str
    capabilities: frozenset[str]
    #: Files a take of this source may carry beyond the universal deliverables. Together with
    #: ``Universal.take_deliverables`` this is the allowlist ``allowed_take_files`` enforces.
    additive_take_deliverables: frozenset[str]
    additive_manifest_keys: frozenset[str]
    additive_source_attribution_keys: frozenset[str]
    additive_bundle_files: frozenset[str]
    additive_bundle_file_globs: tuple[str, ...]
    waivers: tuple[Waiver, ...]

    def waiver_for(self, target: str) -> Waiver | None:
        for w in self.waivers:
            if w.target == target:
                return w
        return None


@dataclass(frozen=True)
class Registry:
    spec_id: str
    spec_version: str
    universal: Universal
    profiles: Mapping[str, Profile]
    path: Path

    def profile_for(self, source_name: str) -> Profile | None:
        return self.profiles.get(source_name)

    def declared_capabilities(self, source_name: str) -> frozenset[str]:
        """Capabilities this source is declared to have.

        Unknown sources return the empty set rather than raising: that reproduces the
        behaviour of the hardcoded rule this replaced, where anything that was not PRISM
        simply had no development reference. A source missing from the registry is reported
        by the conformance check, which is where an unknown dataset should surface -- not
        from inside a per-take capability lookup.
        """
        profile = self.profiles.get(source_name)
        return profile.capabilities if profile else frozenset()

    def allowed_manifest_keys(self, source_name: str) -> frozenset[str]:
        profile = self.profiles.get(source_name)
        extra = profile.additive_manifest_keys if profile else frozenset()
        return self.universal.manifest_keys | extra

    def allowed_source_attribution_keys(self, source_name: str) -> frozenset[str]:
        profile = self.profiles.get(source_name)
        extra = profile.additive_source_attribution_keys if profile else frozenset()
        return self.universal.source_attribution_keys | extra

    def required_bundle_files(self, source_name: str) -> frozenset[str]:
        profile = self.profiles.get(source_name)
        extra = profile.additive_bundle_files if profile else frozenset()
        return self.universal.bundle_files | extra

    def allowed_take_files(self, source_name: str) -> frozenset[str]:
        """Every file a take directory of this source may contain. Anything else is undeclared."""
        profile = self.profiles.get(source_name)
        extra = profile.additive_take_deliverables if profile else frozenset()
        return self.universal.take_deliverables | extra

    def optional_take_files(self, source_name: str) -> frozenset[str]:
        """Files a take of this source may carry but does not owe: the profile's additive set."""
        profile = self.profiles.get(source_name)
        return profile.additive_take_deliverables if profile else frozenset()

    def expired_waivers(self, source_name: str, today: _dt.date) -> tuple[Waiver, ...]:
        profile = self.profiles.get(source_name)
        if profile is None:
            return ()
        return tuple(w for w in profile.waivers if w.is_expired(today))


def _as_frozenset(block: Mapping, key: str) -> frozenset[str]:
    value = block.get(key) or []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProfileError(f"{key!r} must be a list, got {type(value).__name__}")
    return frozenset(str(v) for v in value)


def _parse_waiver(raw: Mapping, source_name: str) -> Waiver:
    missing = [f for f in _REQUIRED_WAIVER_FIELDS if raw.get(f) in (None, "")]
    if missing:
        raise ProfileError(
            f"{source_name}: waiver {raw.get('target', '<no target>')!r} is missing "
            f"{', '.join(missing)}; a waiver without an expiry never expires"
        )
    expires = raw["expires"]
    if isinstance(expires, _dt.datetime):
        expires = expires.date()
    if not isinstance(expires, _dt.date):
        # PyYAML gives a date for an unquoted YYYY-MM-DD; a quoted one arrives as str.
        try:
            expires = _dt.date.fromisoformat(str(expires))
        except ValueError as exc:
            raise ProfileError(
                f"{source_name}: waiver {raw['target']!r} has an unparseable expires "
                f"{expires!r}; use YYYY-MM-DD"
            ) from exc
    return Waiver(
        target=str(raw["target"]),
        reason=str(raw["reason"]).strip(),
        remedy=str(raw["remedy"]).strip(),
        expires=expires,
    )


def _parse_profile(name: str, raw: Mapping) -> Profile:
    additive = raw.get("additive") or {}
    globs = additive.get("bundle_file_globs") or []
    waivers = tuple(_parse_waiver(w, name) for w in (raw.get("waivers") or []))

    duplicate_targets = {
        w.target for w in waivers if sum(1 for x in waivers if x.target == w.target) > 1
    }
    if duplicate_targets:
        raise ProfileError(
            f"{name}: duplicate waiver targets {sorted(duplicate_targets)}"
        )

    return Profile(
        source_name=name,
        small_mode=str(raw.get("small_mode") or ""),
        capabilities=_as_frozenset(raw, "capabilities"),
        additive_take_deliverables=_as_frozenset(additive, "take_deliverables"),
        additive_manifest_keys=_as_frozenset(additive, "manifest_keys"),
        additive_source_attribution_keys=_as_frozenset(additive, "source_attribution_keys"),
        additive_bundle_files=_as_frozenset(additive, "bundle_files"),
        additive_bundle_file_globs=tuple(str(g) for g in globs),
        waivers=waivers,
    )


def load(path: Path | str | None = None) -> Registry:
    """Read and validate the registry. Raises ``ProfileError`` rather than degrading."""
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if not registry_path.exists():
        raise ProfileError(f"dataset profile registry not found: {registry_path}")

    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if raw.get("schema") != SCHEMA:
        raise ProfileError(f"expected schema {SCHEMA!r}, got {raw.get('schema')!r}")

    universal_raw = raw.get("universal") or {}
    dtype = str(universal_raw.get("string_arrays_dtype") or "")
    if not dtype:
        raise ProfileError("universal.string_arrays_dtype is required")

    universal = Universal(
        bundle_files=_as_frozenset(universal_raw, "bundle_files"),
        distribution_files=_as_frozenset(universal_raw, "distribution_files"),
        take_deliverables=_as_frozenset(universal_raw, "take_deliverables"),
        manifest_keys=_as_frozenset(universal_raw, "manifest_keys"),
        source_attribution_keys=_as_frozenset(universal_raw, "source_attribution_keys"),
        index_take_entry_keys=_as_frozenset(universal_raw, "index_take_entry_keys"),
        string_arrays_dtype=dtype,
    )
    if not universal.manifest_keys:
        raise ProfileError("universal.manifest_keys is empty")
    if not universal.source_attribution_keys:
        raise ProfileError("universal.source_attribution_keys is empty")

    profiles = {
        name: _parse_profile(name, block or {})
        for name, block in (raw.get("profiles") or {}).items()
    }
    if not profiles:
        raise ProfileError("registry declares no profiles")

    return Registry(
        spec_id=str(raw.get("spec_id") or ""),
        spec_version=str(raw.get("spec_version") or ""),
        universal=universal,
        profiles=profiles,
        path=registry_path,
    )


@functools.lru_cache(maxsize=1)
def default_registry() -> Registry:
    """The registry at its repository path, parsed once."""
    return load()
