"""Sensor sites, declared in configuration rather than enumerated in code.

Where a virtual sensor sits is a contract question, not an implementation detail. This
project's canonical order is six sensors; the external Small/Large specification describes
eight. That divergence is unresolved and, until a contract change
and an ADR settle it, code must not take a side by hard-coding either list.

So a site is data: which markers define its rigid body, how the sensor origin is offset
from them, which anatomical direction resolves the plate normal, and what provenance each
of its channels carries. Both site sets live in the same registry file and neither is
privileged. Adding, removing or re-scoping a site is a configuration change.

Two offset kinds exist because GAITEX supports two situations:

``plate_normal``
    The site coincides with a real GAITEX sensor plate. The offset runs along the plate's
    own inward normal by a depth the hardware fixes, so the rotation and the position are
    both determined by the observed constellation.

``model_local``
    The site is somewhere the study put no sensor -- the occiput, the T4 level of the
    thorax. The rigid body is still observed, but the offset from it comes from an
    anthropometric model. Rotation stays ``source_derived``; position becomes ``estimated``.

``unresolved``
    The site is declared and its rigid body is known, but the offset has not been
    determined yet. This exists so that a gap can be stated rather than papered over: GAITEX
    puts no marker at the occiput or at the T4 level, so the offsets for those sites are
    genuinely open. Declaring them ``unresolved`` keeps the site addressable and
    machine-checkable while making any attempt to synthesise its position fail loudly
    instead of quietly returning a number nobody derived.

Config values are read, never defaulted. The registry file is JSON
carrying a ``.yaml`` extension, which is this repository's convention for machine-read
configuration; ``prism_field_registry_v4.yaml`` and ``source_parsing_v1_5.yaml`` are the
same shape, and it is why PyYAML is not a dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "InwardReference",
    "SensorSite",
    "SiteOffset",
    "SiteRegistry",
    "SiteRegistryError",
    "inward_reference_world",
    "load_site_registry",
]

_PLATE_NORMAL = "plate_normal"
_MODEL_LOCAL = "model_local"
_UNRESOLVED = "unresolved"
_OFFSET_KINDS = frozenset({_PLATE_NORMAL, _MODEL_LOCAL, _UNRESOLVED})
_PROVENANCE = frozenset(
    {"measured", "source_derived", "estimated", "proxy", "unavailable"}
)


class SiteRegistryError(ValueError):
    """Raised when a site registry is malformed or a site is not declared."""


@dataclass(frozen=True)
class InwardReference:
    """Anatomical markers whose midpoint lies towards the body from the plate.

    Only the direction is used, so the markers need to be on the body side of the plate;
    their exact positions do not enter the lever arm.
    """

    markers: tuple[str, ...]


@dataclass(frozen=True)
class SiteOffset:
    """How the sensor origin is displaced from the rigid body's centroid."""

    kind: str
    depth_m: float | None
    in_plane_offset_m: tuple[float, float] | None
    local_offset_m: tuple[float, float, float] | None
    unresolved_reason: str

    @property
    def is_resolved(self) -> bool:
        return self.kind != _UNRESOLVED


@dataclass(frozen=True)
class SensorSite:
    """One declared virtual sensor location."""

    name: str
    markers: tuple[str, ...]
    offset: SiteOffset
    inward_reference: InwardReference | None
    position_provenance: str
    rotation_provenance: str
    note: str
    measured_counterpart: str | None
    """Name of the source's own sensor at this site, if it mounted one.

    Validation-only: it says which measured stream may be compared against the synthesis
    at this site. It never enters the synthesis path and is ``None`` for sites GAITEX did
    not instrument.
    """


@dataclass(frozen=True)
class SiteRegistry:
    """All declared sites plus the named sets that group them."""

    config_id: str
    version: str
    activation_state: str
    sites: Mapping[str, SensorSite]
    site_sets: Mapping[str, tuple[str, ...]]

    def require(self, name: str) -> SensorSite:
        """Return a site, refusing rather than substituting when it is not declared."""
        try:
            return self.sites[name]
        except KeyError as error:
            raise SiteRegistryError(
                f"site {name!r} is not declared; the registry holds "
                f"{sorted(self.sites)}"
            ) from error

    def measured_counterparts(self) -> dict[str, str]:
        """Site name to the source's own sensor name, for every instrumented site."""
        return {
            site.name: site.measured_counterpart
            for site in self.sites.values()
            if site.measured_counterpart is not None
        }

    def require_set(self, name: str) -> tuple[SensorSite, ...]:
        """Return the sites of a named set, in the order the set declares them."""
        try:
            members = self.site_sets[name]
        except KeyError as error:
            raise SiteRegistryError(
                f"site set {name!r} is not declared; the registry holds "
                f"{sorted(self.site_sets)}"
            ) from error
        return tuple(self.require(member) for member in members)


def _require(document: Mapping[str, object], key: str, path: Path) -> object:
    if key not in document:
        raise SiteRegistryError(f"{path}: missing required key {key!r}")
    return document[key]


def _parse_offset(raw: object, site: str, path: Path) -> SiteOffset:
    if not isinstance(raw, dict):
        raise SiteRegistryError(f"{path}: site {site!r} offset must be an object")
    kind = raw.get("kind")
    if kind not in _OFFSET_KINDS:
        raise SiteRegistryError(
            f"{path}: site {site!r} offset kind {kind!r} must be one of {sorted(_OFFSET_KINDS)}"
        )

    if kind == _UNRESOLVED:
        reason = raw.get("unresolved_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise SiteRegistryError(
                f"{path}: site {site!r} is unresolved, so it must say why in "
                f"'unresolved_reason' -- an unexplained gap is indistinguishable from an "
                f"oversight"
            )
        return SiteOffset(
            kind=kind,
            depth_m=None,
            in_plane_offset_m=None,
            local_offset_m=None,
            unresolved_reason=reason,
        )

    if kind == _PLATE_NORMAL:
        if "depth_m" not in raw or "in_plane_offset_m" not in raw:
            raise SiteRegistryError(
                f"{path}: site {site!r} needs depth_m and in_plane_offset_m; the hold "
                f"forbids a code-side default for either"
            )
        in_plane = raw["in_plane_offset_m"]
        if not isinstance(in_plane, list) or len(in_plane) != 2:
            raise SiteRegistryError(
                f"{path}: site {site!r} in_plane_offset_m must be a 2-element list"
            )
        return SiteOffset(
            kind=kind,
            depth_m=float(raw["depth_m"]),
            in_plane_offset_m=(float(in_plane[0]), float(in_plane[1])),
            local_offset_m=None,
            unresolved_reason="",
        )

    if "local_offset_m" not in raw:
        raise SiteRegistryError(f"{path}: site {site!r} needs local_offset_m")
    local = raw["local_offset_m"]
    if not isinstance(local, list) or len(local) != 3:
        raise SiteRegistryError(
            f"{path}: site {site!r} local_offset_m must be a 3-element list"
        )
    return SiteOffset(
        kind=kind,
        depth_m=None,
        in_plane_offset_m=None,
        local_offset_m=(float(local[0]), float(local[1]), float(local[2])),
        unresolved_reason="",
    )


def _parse_site(name: str, raw: object, path: Path) -> SensorSite:
    if not isinstance(raw, dict):
        raise SiteRegistryError(f"{path}: site {name!r} must be an object")

    markers = raw.get("markers")
    if not isinstance(markers, list) or len(markers) < 3:
        raise SiteRegistryError(
            f"{path}: site {name!r} needs at least 3 markers to define a rigid body"
        )
    if len(set(markers)) != len(markers):
        raise SiteRegistryError(f"{path}: site {name!r} repeats a marker")

    for field in ("position_provenance", "rotation_provenance"):
        value = raw.get(field)
        if value not in _PROVENANCE:
            raise SiteRegistryError(
                f"{path}: site {name!r} {field} {value!r} must be one of {sorted(_PROVENANCE)}"
            )

    offset = _parse_offset(raw.get("offset"), name, path)
    reference_raw = raw.get("inward_reference")
    reference: InwardReference | None = None
    if reference_raw is not None:
        if not isinstance(reference_raw, dict) or not isinstance(
            reference_raw.get("markers"), list
        ):
            raise SiteRegistryError(
                f"{path}: site {name!r} inward_reference needs a markers list"
            )
        reference = InwardReference(markers=tuple(reference_raw["markers"]))
    if offset.kind == _PLATE_NORMAL and reference is None:
        raise SiteRegistryError(
            f"{path}: site {name!r} uses a plate normal, so it needs an inward_reference "
            f"to resolve the normal's sign without consulting the measured IMU"
        )

    return SensorSite(
        name=name,
        markers=tuple(markers),
        offset=offset,
        inward_reference=reference,
        position_provenance=str(raw["position_provenance"]),
        rotation_provenance=str(raw["rotation_provenance"]),
        note=str(raw.get("note", "")),
        measured_counterpart=_optional_string(raw, "measured_counterpart", name, path),
    )


def _optional_string(raw: dict, key: str, site: str, path: Path) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SiteRegistryError(f"{path}: site {site!r} {key} must be a non-empty string")
    return value


def load_site_registry(path: Path) -> SiteRegistry:
    """Load and validate a sensor-site registry."""
    path = Path(path)
    if not path.is_file():
        raise SiteRegistryError(f"missing site registry: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SiteRegistryError(f"{path}: not valid JSON") from error
    if not isinstance(document, dict):
        raise SiteRegistryError(f"{path}: expected an object at the top level")

    raw_sites = _require(document, "sites", path)
    if not isinstance(raw_sites, dict) or not raw_sites:
        raise SiteRegistryError(f"{path}: 'sites' must be a non-empty object")
    sites = {name: _parse_site(name, raw, path) for name, raw in raw_sites.items()}

    raw_sets = document.get("site_sets", {})
    if not isinstance(raw_sets, dict):
        raise SiteRegistryError(f"{path}: 'site_sets' must be an object")
    site_sets: dict[str, tuple[str, ...]] = {}
    for set_name, members in raw_sets.items():
        if not isinstance(members, list) or not members:
            raise SiteRegistryError(f"{path}: site set {set_name!r} must be a non-empty list")
        unknown = [member for member in members if member not in sites]
        if unknown:
            raise SiteRegistryError(
                f"{path}: site set {set_name!r} names undeclared sites {unknown}"
            )
        site_sets[set_name] = tuple(members)

    return SiteRegistry(
        config_id=str(_require(document, "config_id", path)),
        version=str(_require(document, "version", path)),
        activation_state=str(document.get("activation_state", "non_authorizing")),
        sites=sites,
        site_sets=site_sets,
    )


def inward_reference_world(
    cluster_centroid_m: np.ndarray, reference_markers_m: np.ndarray
) -> np.ndarray:
    """World direction from the plate towards the body, per frame.

    ``reference_markers_m`` is ``(frames, markers, 3)``. The midpoint of the anatomical
    markers minus the plate centroid gives a vector pointing into the limb, whose sign is
    all the plate normal needs.
    """
    if reference_markers_m.ndim != 3 or reference_markers_m.shape[2] != 3:
        raise SiteRegistryError(
            f"expected (frames, markers, 3) reference markers, got {reference_markers_m.shape}"
        )
    if cluster_centroid_m.shape != reference_markers_m.shape[:1] + (3,):
        raise SiteRegistryError(
            f"centroid {cluster_centroid_m.shape} does not match reference markers "
            f"{reference_markers_m.shape}"
        )
    # Averaging only the visible landmarks, counted explicitly rather than by nanmean, so a
    # frame where every landmark is occluded yields NaN without warning instead of a mean of
    # an empty slice.
    visible = np.isfinite(reference_markers_m).all(axis=2)
    counts = visible.sum(axis=1)
    summed = np.where(visible[:, :, None], reference_markers_m, 0.0).sum(axis=1)
    midpoint = np.where(
        counts[:, None] > 0, summed / np.maximum(counts, 1)[:, None], np.nan
    )
    return midpoint - cluster_centroid_m


def marker_union(sites: Sequence[SensorSite]) -> tuple[str, ...]:
    """Every marker any of the given sites depends on, deduplicated and ordered."""
    seen: dict[str, None] = {}
    for site in sites:
        for marker in site.markers:
            seen.setdefault(marker, None)
        if site.inward_reference is not None:
            for marker in site.inward_reference.markers:
                seen.setdefault(marker, None)
    return tuple(seen)
