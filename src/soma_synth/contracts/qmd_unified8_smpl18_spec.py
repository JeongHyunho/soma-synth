"""Declarative dataset spec for lineage ``spec_id=qmd_unified8_smpl18``, ``spec_version=faithful-v2``.

This is the single source of truth for the on-disk schema of the poc-demo / experimental synthetic-IMU
datasets (PRISM measured + AMASS synthetic small/large/anthro). It validates **structure and declared
invariants, not values** (see ``derive_capabilities`` / allowed enums). Frozen against the first
generated PRISM and AMASS bundles (2026-08-18).

Governance: experimental_non_candidate, INTERNAL-ONLY. Lineage is keyed by spec_id/spec_version, never a
canonical contract version.

``small_mode`` names how the eight-site small was built and therefore which validator rules apply.
``synthetic_from_markers`` (ADR-0039) is a small synthesised from optical marker clusters rather than from
SMPL forward kinematics: its orientation is a rigid relabel of a marker-fitted body, not of the SMPL joint
that ``large`` carries, so the L3 axis check treats it as it treats a measured sensor. In that mode --
and in any mode -- ``imu_valid_mask`` governs the three IMU arrays cell by cell: a cell the mask calls
valid is finite in every component, a cell it calls invalid is NaN in every component and 0.0 in
``imu_confidence``. Nothing is filled with zeros, identities or neighbours (paired contract 2.2 and
10.6). ``timestamps_s`` is finite and uniform on every frame regardless.
"""

from __future__ import annotations

import functools as _functools  # underscored: the module's public names stay what they were
from dataclasses import dataclass, field
from typing import Mapping

from . import dataset_profiles

SPEC_ID = "qmd_unified8_smpl18"
SPEC_VERSION = "faithful-v2"

# Allowed enum values (structural gate; the spec does not assert numeric values).
SMALL_MODES = frozenset({"measured_physical", "synthetic_from_smpl", "synthetic_from_markers"})
#: Modes whose orientation is NOT a constant relabel of the SMPL joint rotation in ``large`` -- a worn
#: sensor sits on soft tissue, a marker cluster is fitted to the skin -- so L3 reports dispersion as a
#: WARN above a wide band instead of failing the take at the synthetic band.
RELABEL_TOLERANT_SMALL_MODES = frozenset({"measured_physical", "synthetic_from_markers"})
AXIS_CONVENTIONS = frozenset({"spec_S_v2"})
SUBJECT_SCOPES = frozenset(
    {"subject_constant_trial_frame_invariant", "sequence_constant_frame_invariant"}
)
CAPABILITIES = frozenset({"measured_physical", "development_reference"})

# spec_S_v2 site geometry: the canonical 8-site order (index -> name) and each site's SMPL joint.
# Orientation is expressed in sensor frame S; by construction it is a rigid relabel of the SMPL global
# (bone) rotation at these joints, so imu_orientation[:, i] == gR[:, SITE_TO_JOINT[SITE_ORDER[i]]] @ const.
SITE_ORDER = (
    "back_T4",
    "wrist_l",
    "wrist_r",
    "shank_l",
    "shank_r",
    "occiput",
    "foot_l",
    "foot_r",
)
SITE_TO_JOINT = {
    "back_T4": 9,
    "wrist_l": 18,
    "wrist_r": 19,
    "shank_l": 4,
    "shank_r": 5,
    "occiput": 15,
    "foot_l": 10,
    "foot_r": 11,
}

# Dtype classes the spec speaks in (numpy dtype -> one of these via ``dtype_class``).
DTYPE_CLASSES = frozenset({"f4", "f8", "bool", "i8", "U", "object"})

# Symbolic array axes: digit strings are FIXED sizes; "T" is the per-take frame axis (free but consistent
# within an artifact); "N" is a free length used once (betas).
_FREE_AXES = frozenset({"T", "N"})


class SpecError(ValueError):
    """Raised for spec-usage errors (unknown artifact / small_mode / unsupported dtype)."""


def dtype_class(np_dtype) -> str:
    """Map a numpy dtype to one of ``DTYPE_CLASSES``."""
    kind = np_dtype.kind
    if kind == "f":
        return f"f{np_dtype.itemsize}"
    if kind == "i":
        return f"i{np_dtype.itemsize}"
    if kind == "b":
        return "bool"
    if kind == "U":
        return "U"
    if kind == "O":
        return "object"
    raise SpecError(f"unsupported-dtype:{np_dtype!s}")


@dataclass(frozen=True)
class KeySpec:
    """One array key: its dtype class and symbolic axes (rank == len(axes))."""

    name: str
    dtype: str
    axes: tuple[str, ...]

    @property
    def rank(self) -> int:
        return len(self.axes)


@dataclass(frozen=True)
class ArtifactSpec:
    """One npz artifact: core keys, capability-conditional keys, and its own capability gate.

    ``capability is None`` -> the artifact is always present. Otherwise it is present only when that
    capability is derived for the take (e.g. development_reference is PRISM-only).
    """

    filename: str
    core: tuple[KeySpec, ...]
    conditional: dict[str, tuple[KeySpec, ...]] = field(default_factory=dict)
    capability: str | None = None


def _ks(name: str, dtype: str, *axes: str) -> KeySpec:
    return KeySpec(name=name, dtype=dtype, axes=tuple(axes))


# --------------------------------------------------------------------------- small
_SMALL = ArtifactSpec(
    filename="small_reference.npz",
    core=(
        _ks("imu_orientation", "f4", "T", "8", "4"),
        _ks("imu_acceleration", "f4", "T", "8", "3"),
        _ks("imu_angular_velocity", "f4", "T", "8", "3"),
        _ks("imu_valid_mask", "bool", "T", "8"),
        _ks("imu_confidence", "f4", "T", "8"),
        _ks("imu_orientation_absolute_heading", "bool", "8"),
        _ks("sensor_codes", "U", "8"),
        _ks("mount_id", "U", "8"),
        _ks("small_mode", "U"),
        _ks("axis_convention", "U"),
        _ks("pair_id", "U"),
        _ks("timestamps_s", "f8", "T"),
        _ks("frame_count", "i8"),
    ),
    conditional={
        "measured_physical": (
            _ks("q_anatomical_from_sensor", "f4", "8", "4"),
            _ks("p_segment_to_sensor_m", "f4", "8", "3"),
            _ks("imu_synth_mask", "bool", "T", "8"),
            _ks("gyro_provenance", "U", "8"),
        )
    },
)

# --------------------------------------------------------------------------- large
_LARGE = ArtifactSpec(
    filename="large_reference.npz",
    core=(
        _ks("joint_names", "U", "18"),
        _ks("joint_rotation", "f4", "T", "18", "4"),
        _ks("joint_velocity", "f4", "T", "18", "3"),
        _ks("root_velocity", "f4", "T", "3"),
        _ks("pelvis_position_world_aux", "f4", "T", "3"),
        _ks("smpl_global_orientation_world", "f4", "T", "24", "4"),
        _ks("pair_id", "U"),
        _ks("timestamps_s", "f8", "T"),
        _ks("frame_count", "i8"),
    ),
)

# --------------------------------------------------------------------------- anthro
_ANTHRO = ArtifactSpec(
    filename="anthro_reference.npz",
    core=(
        _ks("namespace", "U"),
        _ks("subject_scope", "U"),
        _ks("subject_scope_note", "U"),
        _ks("reference_poses", "U", "2"),
        _ks("joint_names", "U", "22"),
        _ks("joint_position", "f4", "2", "22", "3"),
        _ks("joint_position_frame", "U"),
        _ks("segment_names", "U", "13"),
        _ks("segment_length", "f4", "13"),
        _ks("fixed_joint_names", "U", "4"),
        _ks("fixed_joint_rotation", "f4", "4", "4"),
        _ks("sex", "U"),
        _ks("height", "f4"),
        _ks("body_mass", "f4"),
        _ks("betas", "f4", "N"),
        _ks("betas_num", "i8"),
        _ks("gender", "U"),
        _ks("height_provenance", "U"),
        _ks("body_mass_provenance", "U"),
        _ks("sex_provenance", "U"),
        _ks("betas_provenance", "U"),
        _ks("skeleton_provenance", "U"),
        _ks("fixed_joint_rotation_provenance", "U"),
        _ks("reconstruction_note", "U"),
        _ks("pair_id", "U"),
        _ks("subject_id", "U"),
        _ks("source_asset_id", "U"),
    ),
)

# --------------------------------------------------------------- root_translation
_ROOT_TRANSLATION = ArtifactSpec(
    filename="smpl_root_translation.npz",
    core=(
        _ks("smpl_trans", "f4", "T", "3"),
        _ks("frame_count", "i8"),
        _ks("pair_id", "U"),
        _ks("source_asset_id", "U"),
        _ks("smpl_trans_provenance", "U"),
    ),
)

# ---------------------------------------------------------- development_reference
_DEVELOPMENT_REFERENCE = ArtifactSpec(
    filename="development_reference.npz",
    core=(
        _ks("namespace", "U"),
        _ks("grf_feet_source_native", "f4", "T", "2", "3"),
        _ks("grf_combined_source_native", "f4", "T", "3"),
        _ks("cop_feet_world", "f4", "T", "2", "3"),
        _ks("cop_combined_world", "f4", "T", "3"),
        _ks("foot_contact_mask", "bool", "T", "2"),
        _ks("vertical_force_provenance", "U"),
        _ks("cop_provenance", "U"),
        _ks("contacts_provenance", "U"),
        _ks("usage", "U"),
        _ks("pair_id", "U"),
        _ks("timestamps_s", "f8", "T"),
    ),
    capability="development_reference",
)

# Insertion order defines ``artifact_names()`` order.
ARTIFACTS: dict[str, ArtifactSpec] = {
    "small": _SMALL,
    "large": _LARGE,
    "anthro": _ANTHRO,
    "root_translation": _ROOT_TRANSLATION,
    "development_reference": _DEVELOPMENT_REFERENCE,
}

# Manifest field expectations (structural; used by validator L1/L4, not asserted here beyond membership).
#: Manifest keys every take owes, whatever its source. Read from the dataset profile registry
#: (``configs/datasets/dataset_profiles_v1.yaml``) rather than restated here, so that the set the
#: validator enforces and the set the registry documents cannot drift apart -- ADR-0040 D1, which
#: promoted ``up_axis``, ``resample`` and ``source_attribution`` into it. The paired contract
#: §12.2 requires a smaller floor (the three scope keys); this is the execution layer above it.
#:
#: ``MANIFEST_CORE_FIELDS`` is read on first access, not at import: importing the spec for its
#: site table does not parse the registry file. It is the same frozenset, computed once and cached,
#: and it is still reached as a module attribute (module ``__getattr__``, PEP 562).
@_functools.cache
def _manifest_core_fields() -> frozenset[str]:
    return dataset_profiles.default_registry().universal.manifest_keys


def __getattr__(name: str):
    if name == "MANIFEST_CORE_FIELDS":
        return _manifest_core_fields()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), "MANIFEST_CORE_FIELDS"})


# Unified INDEX schema, shared by every generator's INDEX.json.
INDEX_FIELDS = frozenset(
    {
        "spec_id",
        "spec_version",
        "dataset_dirname",
        "source",
        "artifact_class",
        "distribution_scope",
        "generated_utc",
        "counts",
        "complete",
        "takes",
    }
)
INDEX_TAKE_FIELDS = frozenset({"take_id", "pair_id", "status", "frames", "relative_path"})


def artifact_names() -> tuple[str, ...]:
    """The frozen artifact order."""
    return tuple(ARTIFACTS)


def derive_capabilities(small_mode: str, source_name: str) -> frozenset[str]:
    """Derive the take's capabilities from its manifest signals.

    ``measured_physical`` follows from the mode: it gates the 4 measured small keys, and a mode
    is the only thing that can establish it. Everything source-specific -- today that is
    ``development_reference``, which only PRISM has -- comes from the dataset profile registry
    (ADR-0040 D4) rather than from a source name written into this function. A source the
    registry does not know contributes nothing, which is what the previous ``== "prism"`` test
    did for every non-PRISM source; an unknown source is reported by the conformance check, not
    from inside a per-take lookup.
    """
    if small_mode not in SMALL_MODES:
        raise SpecError(f"unknown-small-mode:{small_mode}")
    caps = set()
    if small_mode == "measured_physical":
        caps.add("measured_physical")
    caps |= dataset_profiles.default_registry().declared_capabilities(source_name)
    return frozenset(caps & CAPABILITIES)


def expected_deliverables(capabilities: frozenset[str]) -> tuple[str, ...]:
    """Frozen per-take deliverable filenames for the given capabilities. No README."""
    names = {"manifest.json"}
    for art in ARTIFACTS.values():
        if art.capability is None or art.capability in capabilities:
            names.add(art.filename)
    return tuple(sorted(names))


def _allowed_keyspecs(
    art: ArtifactSpec, capabilities: frozenset[str]
) -> tuple[dict[str, KeySpec], set[str]]:
    """Return (name -> KeySpec allowed for these capabilities, set of forbidden conditional names)."""
    allowed: dict[str, KeySpec] = {k.name: k for k in art.core}
    forbidden: set[str] = set()
    for cap, keys in art.conditional.items():
        if cap in capabilities:
            for k in keys:
                allowed[k.name] = k
        else:
            forbidden.update(k.name for k in keys)
    return allowed, forbidden


def check_arrays(
    artifact: str,
    capabilities: frozenset[str],
    arrays: Mapping[str, tuple[tuple[int, ...], str]],
) -> tuple[tuple[str, str], ...]:
    """Structurally check one artifact's arrays against the spec.

    ``arrays`` maps key -> (shape, dtype_class). Returns a tuple of ``(severity, message)`` where severity
    is ``"FAIL"`` or ``"WARN"``. Empty tuple means clean. Every deviation this function detects is a
    FAIL, an ``object`` array where a unicode one is expected included (see the dtype comment below).
    """
    if artifact not in ARTIFACTS:
        raise SpecError(f"unknown-artifact:{artifact}")
    art = ARTIFACTS[artifact]
    allowed, forbidden = _allowed_keyspecs(art, capabilities)

    issues: list[tuple[str, str]] = []
    present = set(arrays)

    for name in sorted(set(allowed) - present):
        issues.append(("FAIL", f"missing-key:{name}"))

    t_values: set[int] = set()
    for name in sorted(present):
        if name in forbidden:
            issues.append(("FAIL", f"forbidden-key:{name} (capability absent)"))
            continue
        ks = allowed.get(name)
        if ks is None:
            issues.append(("FAIL", f"unexpected-key:{name}"))
            continue
        shape, dclass = arrays[name]

        # dtype class. A string array stored as `object` is not excused (ADR-0040 D2). Such an
        # array cannot be read back without `allow_pickle=True`, so the defect hands
        # arbitrary-code-execution risk to whoever consumes the bundle -- too heavy to carry as
        # a warning, and the paired contract §13 already lists dtype among the required
        # validations.
        if dclass != ks.dtype:
            issues.append(
                ("FAIL", f"dtype-mismatch:{name} expected {ks.dtype} got {dclass}")
            )

        # rank
        if len(shape) != ks.rank:
            issues.append(
                ("FAIL", f"rank-mismatch:{name} expected rank {ks.rank} got {len(shape)}")
            )
            continue

        # per-axis sizes
        for axis, size in zip(ks.axes, shape):
            if axis in _FREE_AXES:
                if axis == "T":
                    t_values.add(int(size))
                continue
            if int(size) != int(axis):
                issues.append(
                    ("FAIL", f"axis-size-mismatch:{name} expected {axis} got {size}")
                )

    if len(t_values) > 1:
        issues.append(
            ("FAIL", f"free-axis-T-inconsistent within {artifact}: {sorted(t_values)}")
        )

    return tuple(issues)
