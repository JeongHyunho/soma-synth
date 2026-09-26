"""Generator-independent validation checks: L0 structure, L1 schema, L2 physical.

Never imports a generator. Reads only produced npz/manifest/INDEX under a dataset dir. Physical bands are
the 2026-08-18 calibration (40 PRISM + 40 AMASS takes); wide enough not to false-fail real data, tight
enough to catch gravity-free accel, non-unit quats, wrong rate, non-finite values.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soma_synth.contracts import dataset_limitations, dataset_profiles
from soma_synth.contracts import qmd_unified8_smpl18_spec as spec
from soma_synth.validation import npz_io
from soma_synth.validation.report import Finding


def _declared_caps(source: str) -> frozenset[str]:
    """Source-derived capabilities, from the one registry that declares them.

    Used where the mode is unavailable or unusable and only the source is known. The rule is read
    from the registry rather than written out at each call site, so the validator and the spec
    module cannot hold diverging copies of it (ADR-0040 D4).
    """
    return dataset_profiles.default_registry().declared_capabilities(source)


@dataclass(frozen=True)
class PhysicalBands:
    # Recalibrated 2026-08-18/19 on the FULL AMASS 8261-take distribution: per-site median |a| ranged
    # [3.94, 63.89]. Upper 80 accepts legitimate high-dynamics (KIT running feet ~64). The lower bound is
    # NOT a plain median floor: an airborne-dominant jump/hop (5 BMLmovi/CMU takes) legitimately has a
    # per-site median below 1g (>50% of frames near-weightless), yet its p95 spikes to 30-42 at landing.
    # A gravity-missing bug instead stays low everywhere. So a site is flagged only when median < min AND
    # p95 < 1g (gravity is never reached) -- this clears the jumps while still catching gravity-free accel.
    accel_site_median_min: float = 5.0        # m/s^2 ; gravity-included specific force
    accel_site_median_max: float = 80.0
    accel_gravity_floor_p95: float = 9.80665  # a valid site must reach >=1g at its p95 (gravity present)
    quat_norm_dev_max: float = 1.0e-3
    gyro_abs_max_deg_s: float = 8000.0     # blow-up guard (angular_velocity in deg/s)
    dt_nominal_s: float = 0.01
    dt_tol_s: float = 1.0e-4
    valid_mask_min_coverage: float = 0.5


DEFAULT_BANDS = PhysicalBands()
_DESC_GLOB = "*DATA_DESCRIPTION*.md"


# ---------------------------------------------------------------- helpers
def _load_manifest(take_dir: Path) -> dict:
    p = take_dir / "manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return {}


def _source_name(manifest: dict) -> str:
    src = manifest.get("source")
    if isinstance(src, dict):
        return str(src.get("source_name", ""))
    return ""


def _check_source_attribution_shape(manifest: dict, take_id: str, profile) -> list[Finding]:
    """What a present ``source_attribution`` must carry, and what it may add.

    An attribution that names a licence but not the dataset, or a dataset but not what was
    changed, does not discharge the obligation it exists to record -- which is why the missing
    keys are a FAIL rather than a note. A key the registry does not know is a FAIL too: the
    shapes drifted apart once already because nothing objected.
    """
    attribution = manifest.get("source_attribution")
    if attribution is None:
        return []  # absence is the manifest_keys rule's business, not this one's.

    source = _source_name(manifest)
    registry = dataset_profiles.default_registry()
    if not isinstance(attribution, dict):
        return [
            Finding(
                "L1", "manifest", "FAIL",
                f"source_attribution is {type(attribution).__name__}, not an object",
                take_id=take_id,
            )
        ]

    findings: list[Finding] = []
    waived = {w.target for w in profile.waivers} if profile is not None else set()
    for key in sorted(registry.universal.source_attribution_keys - set(attribution)):
        if f"manifest.source_attribution.{key}" in waived:
            continue
        findings.append(
            Finding(
                "L1", "manifest", "FAIL",
                f"source_attribution lacks {key!r}", take_id=take_id,
            )
        )
    for key in sorted(set(attribution) - registry.allowed_source_attribution_keys(source)):
        findings.append(
            Finding(
                "L1", "manifest", "FAIL",
                f"undeclared source_attribution key {key!r}; add it to the profile's "
                "additive.source_attribution_keys or remove it",
                take_id=take_id,
            )
        )
    return findings


def _small_mode(take_dir: Path) -> str | None:
    p = take_dir / "small_reference.npz"
    if not p.exists():
        return None
    got = npz_io.load_numeric(p, ["small_mode"])
    if "small_mode" in got:
        return str(got["small_mode"])
    return None


def take_capabilities(take_dir: Path) -> tuple[str | None, str, frozenset[str], list[Finding]]:
    """Return (small_mode, source, capabilities, findings). Bad small_mode -> FAIL + fallback caps."""
    manifest = _load_manifest(take_dir)
    source = _source_name(manifest)
    small_mode = _small_mode(take_dir)
    findings: list[Finding] = []
    if small_mode is None:
        return None, source, _declared_caps(source), findings
    try:
        caps = spec.derive_capabilities(small_mode, source)
    except spec.SpecError:
        findings.append(
            Finding("L1", "schema", "FAIL", f"unknown small_mode:{small_mode}", artifact="small")
        )
        caps = _declared_caps(source)
    return small_mode, source, caps, findings


def read_index(dataset_dir: Path) -> dict:
    """Parse INDEX.json, tolerant of both the legacy and the unified shape."""
    raw = json.loads((dataset_dir / "INDEX.json").read_text(encoding="utf-8"))
    takes = []
    for e in raw.get("takes", []):
        if not isinstance(e, dict):
            continue
        take_id = e.get("take_id") or e.get("take") or e.get("sequence_id")
        rel = e.get("relative_path") or take_id
        takes.append({"take_id": take_id, "rel": rel, "status": e.get("status", "ok")})
    counts = raw.get("counts", {}) if isinstance(raw.get("counts"), dict) else {}
    unified = "spec_version" in raw and {"ok", "failed", "excluded", "total"} <= set(counts)
    return {"raw": raw, "takes": takes, "counts": counts, "unified": unified}


# ---------------------------------------------------------------- L0
def check_l0_structure(dataset_dir) -> list[Finding]:
    dataset_dir = Path(dataset_dir)
    findings: list[Finding] = []

    if not (dataset_dir / "INDEX.json").exists():
        return [Finding("L0", "structure", "FAIL", "missing INDEX.json")]
    if not list(dataset_dir.glob(_DESC_GLOB)):
        findings.append(Finding("L0", "structure", "FAIL", "missing *DATA_DESCRIPTION*.md at root"))

    idx = read_index(dataset_dir)
    ok_takes = [t for t in idx["takes"] if t["status"] == "ok"]
    index_dirs = {t["rel"] for t in idx["takes"]}

    for t in ok_takes:
        tid, rel = t["take_id"], t["rel"]
        td = dataset_dir / rel
        if not td.is_dir():
            findings.append(Finding("L0", "structure", "FAIL", f"missing take dir: {rel}", take_id=tid))
            continue
        source = _source_name(_load_manifest(td))
        for fname in spec.expected_deliverables(_declared_caps(source)):
            if not (td / fname).exists():
                findings.append(
                    Finding("L0", "structure", "FAIL", f"missing deliverable {fname}", take_id=tid)
                )

    # orphan take dirs (subdirs on disk not referenced by INDEX)
    for child in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        if child.name not in index_dirs:
            findings.append(
                Finding("L0", "structure", "FAIL", f"orphan take dir (not in INDEX): {child.name}")
            )

    # counts self-consistency (strict only for unified INDEX)
    c = idx["counts"]
    if idx["unified"]:
        if c["ok"] != len(ok_takes):
            findings.append(
                Finding("L0", "structure", "FAIL", f"INDEX ok={c['ok']} != ok-take entries={len(ok_takes)}")
            )
        if c["ok"] + c["failed"] + c["excluded"] != c["total"]:
            findings.append(Finding("L0", "structure", "FAIL", "INDEX counts do not sum to total"))
    return findings


# ---------------------------------------------------------------- L1
def _arrays_schema(npz_path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    out = {}
    for name, (shape, dtype) in npz_io.read_npz_schema(npz_path).items():
        out[name] = (shape, spec.dtype_class(dtype))
    return out


def check_take_l1(take_dir, take_id: str) -> list[Finding]:
    take_dir = Path(take_dir)
    manifest = _load_manifest(take_dir)
    _sm, _src, caps, cap_findings = take_capabilities(take_dir)
    # Finding is frozen; re-tag capability findings with this take_id.
    findings: list[Finding] = [
        Finding(f.level, f.check, f.severity, f.message, take_id=take_id, artifact=f.artifact)
        for f in cap_findings
    ]

    # manifest core fields. A field the source's profile waives is reported as a WARN carrying the
    # expiry rather than a FAIL: the waiver is the record that this gap is known, dated and owned
    # (ADR-0040 D4). It stays visible, and it stops being a WARN the day the waiver lapses, at
    # which point check_dataset_profile_conformance fails the bundle outright.
    source_name = _source_name(manifest)
    profile = dataset_profiles.default_registry().profile_for(source_name)
    missing = spec.MANIFEST_CORE_FIELDS - set(manifest)
    for field in sorted(missing):
        waiver = profile.waiver_for(f"manifest.{field}") if profile else None
        if waiver is not None:
            findings.append(
                Finding(
                    "L1", "manifest", "WARN",
                    f"missing manifest field:{field} (waived until "
                    f"{waiver.expires.isoformat()}; {waiver.remedy})",
                    take_id=take_id,
                )
            )
        else:
            findings.append(
                Finding(
                    "L1", "manifest", "FAIL", f"missing manifest field:{field}", take_id=take_id
                )
            )

    # `source_attribution`'s shape, checked only where the field is present -- its absence is the
    # rule above's business, and running both would report one gap twice. ADR-0040 D1 required
    # the field without saying what goes in it, and the two bundles that had one disagreed
    # (gaitex eight keys, addbio six); the registry now pins their intersection.
    findings.extend(_check_source_attribution_shape(manifest, take_id, profile))

    # per-artifact schema
    frame_lengths: dict[str, int] = {}
    for art, aspec in spec.ARTIFACTS.items():
        if aspec.capability is not None and aspec.capability not in caps:
            continue
        npz_path = take_dir / aspec.filename
        if not npz_path.exists():
            continue  # presence is L0's job
        arrays = _arrays_schema(npz_path)
        dtype_waiver = (
            profile.waiver_for("npz.string_arrays_dtype") if profile is not None else None
        )
        string_dtype = dataset_profiles.default_registry().universal.string_arrays_dtype
        for sev, msg in spec.check_arrays(art, caps, arrays):
            # A string array stored as `object` fails since ADR-0040 D2. Where the source's
            # profile waives that specific defect, it is reported as a dated WARN instead --
            # same treatment as a waived manifest field, and it reverts to FAIL on expiry.
            if (
                dtype_waiver is not None
                and sev == "FAIL"
                and msg.startswith("dtype-mismatch:")
                and f"expected {string_dtype} got object" in msg
            ):
                sev = "WARN"
                msg = f"{msg} (waived until {dtype_waiver.expires.isoformat()})"
            findings.append(Finding("L1", "schema", sev, msg, take_id=take_id, artifact=art))
        if "timestamps_s" in arrays:
            frame_lengths[art] = arrays["timestamps_s"][0][0]

    # cross-artifact frame-count (T) consistency
    if len(set(frame_lengths.values())) > 1:
        findings.append(
            Finding(
                "L1", "schema", "FAIL",
                f"cross-artifact frame-count (T) mismatch: {frame_lengths}", take_id=take_id,
            )
        )

    # betas / betas_num invariant
    apath = take_dir / spec.ARTIFACTS["anthro"].filename
    if apath.exists():
        sch = npz_io.read_npz_schema(apath)
        got = npz_io.load_numeric(apath, ["betas_num"])
        if "betas" in sch and "betas_num" in got:
            blen = sch["betas"][0][0] if sch["betas"][0] else 0
            bnum = int(got["betas_num"])
            if blen != bnum:
                findings.append(
                    Finding(
                        "L1", "schema", "FAIL",
                        f"betas/betas_num mismatch: len(betas)={blen} != betas_num={bnum}",
                        take_id=take_id, artifact="anthro",
                    )
                )
    return findings


# ---------------------------------------------------------------- L2
def _finite_fail(name, arr, take_id, findings):
    if not np.all(np.isfinite(arr)):
        findings.append(Finding("L2", "physical", "FAIL", f"non-finite values in {name}", take_id=take_id))
        return False
    return True


def _valid_mask(s: dict, cells: tuple[int, int]) -> np.ndarray:
    """``imu_valid_mask`` as a (T,8) bool, or all-True when it is absent or mis-shaped (L1 reports that)."""
    vm = s.get("imu_valid_mask")
    if vm is None or np.asarray(vm).shape != cells:
        return np.ones(cells, dtype=bool)
    return np.asarray(vm, dtype=bool)


def _mask_discipline(name, arr, mask, take_id, findings) -> bool:
    """The mask governs the cell (ADR-0039): valid -> every component finite; invalid -> every component NaN.

    A finite value under a False mask is a fill -- a zero, an identity, a neighbour -- which the paired
    contract (2.2, 10.6) forbids as a stand-in for a missing sample. Both directions FAIL.
    Returns True when the valid cells can feed the physical bands.
    """
    if arr.shape[:2] != mask.shape:
        return _finite_fail(name, arr, take_id, findings)
    finite = np.isfinite(arr)
    all_finite = finite.all(axis=-1)
    any_finite = finite.any(axis=-1)
    bad_valid = int((mask & ~all_finite).sum())
    if bad_valid:
        findings.append(
            Finding("L2", "physical", "FAIL",
                    f"non-finite values in {name} on {bad_valid} cells imu_valid_mask calls valid",
                    take_id=take_id, artifact="small")
        )
    bad_invalid = int((~mask & any_finite).sum())
    if bad_invalid:
        findings.append(
            Finding("L2", "physical", "FAIL",
                    f"finite values in {name} on {bad_invalid} cells imu_valid_mask calls invalid "
                    f"(a masked-out cell is NaN, never a fill)",
                    take_id=take_id, artifact="small")
        )
    return bad_valid == 0


def check_take_l2(take_dir, take_id: str, bands: PhysicalBands = DEFAULT_BANDS) -> list[Finding]:
    take_dir = Path(take_dir)
    findings: list[Finding] = []
    small_p = take_dir / "small_reference.npz"
    if not small_p.exists():
        return findings
    s = npz_io.load_numeric(
        small_p,
        ["imu_orientation", "imu_acceleration", "imu_angular_velocity", "imu_valid_mask", "timestamps_s",
         "imu_confidence"],
    )

    ori = s["imu_orientation"].astype(np.float64)
    acc = s["imu_acceleration"].astype(np.float64)
    gyr = s["imu_angular_velocity"].astype(np.float64)
    ts = s["timestamps_s"].astype(np.float64)
    mask = _valid_mask(s, acc.shape[:2])
    # An array whose frame axis disagrees with the mask is L1's finding (cross-artifact T); here it is
    # read as all-valid so the bands still run instead of an index error hiding the L1 message.
    ori_mask = mask if ori.shape[:2] == mask.shape else np.ones(ori.shape[:2], dtype=bool)
    gyr_mask = mask if gyr.shape[:2] == mask.shape else np.ones(gyr.shape[:2], dtype=bool)

    # Every band below is computed over the cells the mask calls valid and nothing else; the mask's
    # own consistency with the arrays is checked first, in both directions. timestamps_s is finite on
    # every frame regardless of the mask -- the time axis is the pair's, not the sensor's.
    ori_ok = _mask_discipline("imu_orientation", ori, ori_mask, take_id, findings)
    acc_ok = _mask_discipline("imu_acceleration", acc, mask, take_id, findings)
    gyr_ok = _mask_discipline("imu_angular_velocity", gyr, gyr_mask, take_id, findings)
    _finite_fail("timestamps_s", ts, take_id, findings)

    # imu_confidence follows the mask too (ADR-0039 rule 3): 0.0 exactly where the mask is False, a
    # finite value in (0, 1] where it is True. A shape that disagrees with the mask is L1's finding.
    if "imu_confidence" in s:
        conf = np.asarray(s["imu_confidence"], dtype=np.float64)
        if conf.shape == mask.shape:
            non_finite = int((~np.isfinite(conf)).sum())
            if non_finite:
                findings.append(
                    Finding("L2", "physical", "FAIL", f"imu_confidence is not finite on {non_finite} cells",
                            take_id=take_id, artifact="small"))
            else:
                under_false = int(((~mask) & (conf != 0.0)).sum())
                if under_false:
                    findings.append(
                        Finding("L2", "physical", "FAIL",
                                f"imu_confidence != 0.0 on {under_false} cells imu_valid_mask calls invalid",
                                take_id=take_id, artifact="small"))
                out_of_band = int((mask & ((conf <= 0.0) | (conf > 1.0))).sum())
                if out_of_band:
                    findings.append(
                        Finding("L2", "physical", "FAIL",
                                f"imu_confidence outside (0, 1] on {out_of_band} cells imu_valid_mask calls valid",
                                take_id=take_id, artifact="small"))

    n_sites = mask.shape[1]
    valid_frames = mask.sum(axis=0)                 # (8,)
    for i in np.where(valid_frames == 0)[0]:
        findings.append(
            Finding("L2", "physical", "FAIL", f"site {int(i)}: imu_valid_mask has no valid frame",
                    take_id=take_id, artifact="small")
        )

    # quaternion unit norm (SO(3) for quats), over valid cells
    if ori_ok and ori_mask.any():
        dev = float(np.abs(np.linalg.norm(ori[ori_mask], axis=-1) - 1.0).max())
        if dev > bands.quat_norm_dev_max:
            findings.append(
                Finding("L2", "physical", "FAIL", f"quat norm deviation {dev:.2e} > {bands.quat_norm_dev_max:.0e}", take_id=take_id, artifact="small")
            )

    # accelerometer gravity-included specific force (per-site magnitude, valid rows of that site)
    if acc_ok:
        mag = np.linalg.norm(acc, axis=2)          # (T,8)
        med = np.full(n_sites, np.nan)
        p95 = np.full(n_sites, np.nan)
        for i in range(n_sites):
            rows = mask[:, i]
            if rows.any():
                med[i] = np.median(mag[rows, i])
                p95[i] = np.percentile(mag[rows, i], 95.0)
        seen = np.isfinite(med)
        hi = float(med[seen].max()) if seen.any() else float("nan")
        # upper bound: no site should sit implausibly high on the median (runaway / unit error)
        if hi > bands.accel_site_median_max:
            findings.append(
                Finding(
                    "L2", "physical", "FAIL",
                    f"accel per-site median |a| exceeds {bands.accel_site_median_max}: max-site {hi:.2f}",
                    take_id=take_id, artifact="small",
                )
            )
        # gravity-presence: a site whose median is below the low floor is only a defect when it ALSO
        # never reaches ~1g (p95 < gravity floor) -- that means gravity is missing from the signal.
        # An airborne-dominant jump/hop has a low median but a high p95 (landing spikes), so it passes.
        grav_missing = seen & (med < bands.accel_site_median_min) & (p95 < bands.accel_gravity_floor_p95)
        if bool(grav_missing.any()):
            bad = np.where(grav_missing)[0]
            i = int(bad[np.argmin(p95[bad])])
            findings.append(
                Finding(
                    "L2", "physical", "FAIL",
                    f"accel gravity-missing: site {i} median {med[i]:.2f} < {bands.accel_site_median_min} "
                    f"and p95 {p95[i]:.2f} < {bands.accel_gravity_floor_p95:.2f} (1g)",
                    take_id=take_id, artifact="small",
                )
            )

    # gyro blow-up guard, over valid cells
    if gyr_ok and gyr_mask.any():
        gmax = float(np.linalg.norm(gyr[gyr_mask], axis=-1).max())
        if gmax > bands.gyro_abs_max_deg_s:
            findings.append(
                Finding("L2", "physical", "FAIL", f"gyro |w| {gmax:.1f} exceeds {bands.gyro_abs_max_deg_s}", take_id=take_id, artifact="small")
            )

    # timestamp rate + monotonic
    if ts.size > 1 and np.all(np.isfinite(ts)):
        d = np.diff(ts)
        if np.any(d <= 0):
            findings.append(Finding("L2", "physical", "FAIL", "timestamps not strictly increasing", take_id=take_id))
        dt = float(np.median(d))
        if abs(dt - bands.dt_nominal_s) > bands.dt_tol_s:
            findings.append(
                Finding("L2", "physical", "FAIL", f"timestamp rate/dt off: median dt={dt:.6f} (want {bands.dt_nominal_s})", take_id=take_id)
            )

    # valid_mask coverage (overall; the per-site floor is the no-valid-frame rule above -- a site's
    # lower coverage is measured and recorded on the manifest, not thresholded here)
    cov = float(mask.astype(np.float64).mean())
    if cov < bands.valid_mask_min_coverage:
        findings.append(
            Finding("L2", "physical", "FAIL", f"valid_mask coverage {cov:.3f} < {bands.valid_mask_min_coverage}", take_id=take_id, artifact="small")
        )
    return findings


# ---------------------------------------------------------------- L3 (spec_S_v2 axes)
@dataclass(frozen=True)
class AxisBands:
    synth_relabel_max_deg: float = 1.0      # synthetic relabel must be constant (calib: 0.000)
    measured_relabel_warn_deg: float = 75.0  # measured tolerates mount noise (calib: feet ~30, max ~65)


DEFAULT_AXIS_BANDS = AxisBands()
_GLOBAL_KEYS = ("smpl_global_orientation_world", "smpl_global_orientation_prism_world")


def _quats_to_R(q: np.ndarray) -> np.ndarray:
    """Unit quaternions wxyz (...,4) -> rotation matrices (...,3,3). A NaN or zero-norm quaternion
    becomes a NaN matrix, which the caller filters out; it must not become a warning-as-error."""
    with np.errstate(invalid="ignore", divide="ignore"):
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rows = [
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ]
    return np.stack(rows, axis=-1).reshape(q.shape[:-1] + (3, 3))


def _mean_rotation(rs: np.ndarray) -> np.ndarray:
    """Chordal-L2 mean rotation of (n,3,3) via SVD projection to SO(3)."""
    u, _s, vt = np.linalg.svd(rs.mean(axis=0))
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u = u.copy()
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def _geodesic_deg(ref: np.ndarray, rs: np.ndarray) -> np.ndarray:
    rel = np.einsum("ji,tjk->tik", ref, rs)  # ref^T @ rs
    tr = np.trace(rel, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def check_take_l3_axes(take_dir, take_id: str, bands: AxisBands = DEFAULT_AXIS_BANDS) -> list[Finding]:
    """Cross-source spec_S_v2 check: orientation must be a rigid constant relabel of the SMPL global
    rotation at each site's joint. SMPL-synthetic -> FAIL if not constant; a measured sensor or a
    marker-fitted body (``spec.RELABEL_TOLERANT_SMALL_MODES``) -> WARN only, above a wide band.

    Only the frames ``imu_valid_mask`` calls valid for a site enter that site's statistic (ADR-0039);
    a take without the mask is read as all-valid.
    """
    take_dir = Path(take_dir)
    small_p, large_p = take_dir / "small_reference.npz", take_dir / "large_reference.npz"
    if not (small_p.exists() and large_p.exists()):
        return []
    small_mode = _small_mode(take_dir)
    if small_mode not in spec.SMALL_MODES:
        return []  # L1 reports an unusable/absent small_mode
    s = npz_io.load_numeric(small_p, ["imu_orientation", "imu_valid_mask"])
    gl = npz_io.load_numeric(large_p, list(_GLOBAL_KEYS))
    glob = next((gl[k] for k in _GLOBAL_KEYS if k in gl), None)
    if "imu_orientation" not in s or glob is None:
        return []
    ori = np.asarray(s["imu_orientation"], dtype=np.float64)
    glob = np.asarray(glob, dtype=np.float64)
    if ori.ndim != 3 or ori.shape[1] != len(spec.SITE_ORDER) or glob.shape[1] <= max(spec.SITE_TO_JOINT.values()):
        return [Finding("L3", "axes", "FAIL", "orientation/global shape unusable for axis check", take_id=take_id)]
    if glob.shape[0] != ori.shape[0]:
        return [Finding("L3", "axes", "FAIL", "orientation/global frame counts differ", take_id=take_id)]
    mask = _valid_mask(s, ori.shape[:2])

    ori_r = _quats_to_R(ori)     # (T,8,3,3)
    g_r = _quats_to_R(glob)      # (T,24,3,3)
    tolerant = small_mode in spec.RELABEL_TOLERANT_SMALL_MODES
    findings: list[Finding] = []
    for i, site in enumerate(spec.SITE_ORDER):
        rows = mask[:, i]
        if int(rows.sum()) < 2:
            continue  # L2 reports a site with no valid frame; one frame has no dispersion to measure
        j = spec.SITE_TO_JOINT[site]
        relab = np.einsum("tji,tjk->tik", g_r[rows][:, j], ori_r[rows][:, i])  # gR^T @ ori
        # A NaN or zero-norm quaternion under a True mask is L2's finding; here it must not reach the
        # SVD, whose LinAlgError would abort the whole run instead of failing this take.
        finite = np.isfinite(relab).all(axis=(1, 2))
        if int(finite.sum()) < 2:
            findings.append(
                Finding("L3", "axes", "FAIL",
                        f"site {site}: fewer than two valid frames carry a usable orientation",
                        take_id=take_id, artifact="small"))
            continue
        relab = relab[finite]
        try:
            mean = _mean_rotation(relab)
        except np.linalg.LinAlgError as error:
            findings.append(
                Finding("L3", "axes", "FAIL", f"site {site}: relabel mean undefined ({error})",
                        take_id=take_id, artifact="small"))
            continue
        med = float(np.median(_geodesic_deg(mean, relab)))
        if not tolerant and med > bands.synth_relabel_max_deg:
            findings.append(
                Finding("L3", "axes", "FAIL",
                        f"site {site} relabel not constant (median {med:.2f} deg) — broken spec_S_v2 axes or joint map",
                        take_id=take_id, artifact="small")
            )
        elif tolerant and med > bands.measured_relabel_warn_deg:
            findings.append(
                Finding("L3", "axes", "WARN",
                        f"site {site} relabel dispersion {med:.1f} deg (> {bands.measured_relabel_warn_deg}) "
                        f"under {small_mode} — orientation is not a constant relabel of the segment; "
                        "reported, not diagnosed (the bundle's KNOWN_LIMITATIONS.json names the "
                        "cause where it is known)",
                        take_id=take_id, artifact="small")
            )
    return findings


# ---------------------------------------------------------------- L4 (governance / layout §6)
# An absolute filesystem path in a value: a single drive letter followed by a separator (not a URL
# scheme like "https://"), or a POSIX path: "/", a folder name, "/" again, under any root (/Users,
# /home, /data, /opt, /srv, /scratch, /nfs, /gpfs, /workspace, ...: a Linux machine keeps its data
# anywhere). The POSIX form counts only at the start of the value or after whitespace, a quote, "=",
# ":", a bracket or a separator, so a logical id (extracted/..., runs/.../tmp/...), a unit (m/s^2) and
# a URL (https://host/home/..., file:///home/..., where the path follows a "/") are not flagged; a
# lone "/tmp" names no path under it and is not flagged either.
_ABS_PATH_RE = re.compile(
    r"(?<![A-Za-z])[A-Za-z]:[\\/]"
    r"|(?:^|(?<=[\s\"'=:(\[,;<>|]))/[^/\s\"'<>|\\]+/"
)
_CANONICAL_RE = re.compile(r"\bcanonical\b", re.IGNORECASE)
_BARE_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
# The 'canonical' token is banned only in lineage identifiers, not descriptive free-text (layout §6):
# a manifest may legitimately say "canonical SOMA world GRF not generated" in prose.
_LINEAGE_FIELDS = ("spec_id", "spec_version")


def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v)


def check_take_l4_governance(take_dir, take_id: str) -> list[Finding]:
    take_dir = Path(take_dir)
    manifest = _load_manifest(take_dir)
    if not manifest:
        return [Finding("L4", "governance", "FAIL", "missing or unreadable manifest.json", take_id=take_id)]
    findings: list[Finding] = []

    def _fail(msg: str) -> None:
        findings.append(Finding("L4", "governance", "FAIL", msg, take_id=take_id))

    for key, expected in (("distribution_scope", "internal_only"), ("artifact_class", "experimental_non_candidate")):
        if key not in manifest:
            _fail(f"missing {key}")
        elif manifest[key] != expected:
            _fail(f"{key} != {expected!r} (got {manifest[key]!r})")

    if manifest.get("quality_gate", "NOT_EVALUATED") != "NOT_EVALUATED":
        _fail(f"quality_gate != 'NOT_EVALUATED' (got {manifest.get('quality_gate')!r})")

    if manifest.get("spec_id") not in (None, spec.SPEC_ID):
        _fail(f"spec_id != {spec.SPEC_ID!r} (got {manifest.get('spec_id')!r})")
    sv = manifest.get("spec_version")
    if sv is not None and sv != spec.SPEC_VERSION:
        _fail(f"spec_version != {spec.SPEC_VERSION!r} (got {sv!r})")
    if isinstance(sv, str) and _BARE_SEMVER_RE.match(sv):
        _fail(f"spec_version canonical-form value {sv!r} forbidden (use a spec-tag, not a bare semver)")

    ident = manifest.get("identity")
    if not (isinstance(ident, dict) and ident.get("contract_id")):
        _fail("missing PoC contract_id in identity (layout §6 requires it)")

    for lf in _LINEAGE_FIELDS:
        val = manifest.get(lf)
        if isinstance(val, str) and _CANONICAL_RE.search(val):
            _fail(f"forbidden 'canonical' token in lineage field {lf}: {val!r}")
    for sval in _walk_strings(manifest):
        if _ABS_PATH_RE.search(sval):
            _fail(f"absolute path in a value: {sval[:60]!r}")
    return findings


def _dataset_source(dataset_dir: Path) -> str:
    """The bundle's source name, from INDEX.json, falling back to any take's manifest."""
    try:
        raw = json.loads((dataset_dir / "INDEX.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    src = raw.get("source")
    if isinstance(src, str) and src:
        return src
    for t in read_index(dataset_dir)["takes"][:1]:
        return _source_name(_load_manifest(dataset_dir / t["rel"]))
    return ""


def missing_distribution_file(fname: str) -> Finding:
    """The WARN the conformance judgement gives for a distribution file the bundle lacks; one
    finding equals another for the same file, so a caller can find it in a report."""
    return Finding("L4", "conformance", "WARN",
                   f"missing {fname}; the bundle is not distributable until it exists")


def check_dataset_profile_conformance(
    dataset_dir,
    registry=None,
    today: _dt.date | None = None,
    pending_files=(),
) -> list[Finding]:
    """Judge the bundle against the dataset profile registry (ADR-0040 D4).

    Everything here is a presence or equality judgement; the registry declares no tolerance and
    this function introduces none. A difference the registry does not declare is a FAIL, which is
    the property that stops the next dataset from quietly inventing its own envelope -- the drift
    that produced a manifest where 25 of 49 keys were non-universal and nobody noticed.

    ``pending_files`` names bundle-root files the caller itself writes after this validation
    whatever its outcome (a run's validate step writes VALIDATION_REPORT.json and
    validation_ledger.json): a distribution file among them is not reported missing, because it
    exists when the caller is done. A file a later step writes only if this validation passes is
    not one of them: a run stops after a failed validation, before its readme step writes
    README.md, so the runner does not pass README.md here; it withdraws that warning
    (:func:`missing_distribution_file`) from a report that passed, when its readme step follows.
    A validation that writes none of them passes nothing and is warned.
    """
    dataset_dir = Path(dataset_dir)
    registry = registry if registry is not None else dataset_profiles.default_registry()
    today = today or _dt.date.today()
    pending = frozenset(pending_files)
    findings: list[Finding] = []

    source = _dataset_source(dataset_dir)
    profile = registry.profile_for(source)
    if profile is None:
        findings.append(
            Finding(
                "L0", "conformance", "FAIL",
                f"source {source!r} has no profile in {registry.path.name}; "
                "declare it before this bundle can be judged",
            )
        )
        return findings

    waived = {w.target for w in profile.waivers}

    # -- expired waivers. A waiver that outlives its expiry is the failure mode the expiry exists
    #    to prevent, so it fails rather than merely warning.
    for w in registry.expired_waivers(source, today):
        findings.append(
            Finding(
                "L4", "conformance", "FAIL",
                f"waiver {w.target!r} expired {w.expires.isoformat()} (remedy: {w.remedy})",
            )
        )

    # -- bundle root files the generator owes
    for fname in sorted(registry.required_bundle_files(source)):
        if (dataset_dir / fname).exists() or f"bundle_files.{fname}" in waived:
            continue
        findings.append(
            Finding("L0", "conformance", "FAIL", f"missing bundle file {fname}")
        )

    # -- the bundle's own statement of what it cannot promise, in the shape every bundle shares.
    #    Presence is covered by bundle_files above; here it must parse, fit the schema, and match
    #    the repository's copy, so the two cannot drift apart without a finding.
    limitations_path = dataset_dir / dataset_limitations.FILE_NAME
    if limitations_path.is_file():
        try:
            config = dataset_limitations.load_config()
        except dataset_limitations.LimitationsError as exc:
            findings.append(Finding("L4", "conformance", "FAIL", f"known-limitations config: {exc}"))
        else:
            for problem in dataset_limitations.problems_in_file(
                limitations_path, dataset_dir.name, config
            ):
                severity = "WARN" if problem.startswith("UNVERIFIED: ") else "FAIL"
                findings.append(
                    Finding("L4", "conformance", severity, problem.removeprefix("UNVERIFIED: "))
                )

    # -- files a bundle needs before it is shared. Two of them are validation's own output, so
    #    their absence during validation is expected, not a defect: WARN, never FAIL. One the
    #    caller writes after this validation is not missing at all (pending_files).
    for fname in sorted(registry.universal.distribution_files):
        if (dataset_dir / fname).exists() or f"distribution_files.{fname}" in waived \
                or fname in pending:
            continue
        findings.append(missing_distribution_file(fname))

    # -- INDEX take-entry keys
    try:
        raw = json.loads((dataset_dir / "INDEX.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    entries = [e for e in (raw.get("takes") or []) if isinstance(e, dict)]
    if entries:
        required = registry.universal.index_take_entry_keys
        observed = set(entries[0])
        missing = sorted(required - observed)
        for key in missing:
            alt = f"index.take_entry_keys.{'relative_path' if key == 'rel' else key}"
            if alt in waived:
                continue
            findings.append(
                Finding(
                    "L0", "conformance", "FAIL",
                    f"INDEX take entries lack {key!r} (observed: {sorted(observed)})",
                )
            )

    return findings


def check_take_profile_conformance(take_dir, take_id: str, registry=None) -> list[Finding]:
    """Manifest keys the registry does not account for (ADR-0040 D4)."""
    take_dir = Path(take_dir)
    registry = registry if registry is not None else dataset_profiles.default_registry()
    manifest = _load_manifest(take_dir)
    if not manifest:
        return []
    source = _source_name(manifest)
    profile = registry.profile_for(source)
    if profile is None:
        return []

    allowed = registry.allowed_manifest_keys(source)
    waived = {w.target for w in profile.waivers}
    findings: list[Finding] = []
    for key in sorted(set(manifest) - allowed):
        if f"manifest.{key}" in waived:
            continue
        findings.append(
            Finding(
                "L1", "conformance", "FAIL",
                f"undeclared manifest key {key!r}; add it to the profile's additive block "
                "or remove it",
                take_id=take_id,
            )
        )

    # Files in the take directory the registry does not account for. Every file in a take
    # directory must be declared in the registry, so a companion file passes because a rule
    # allows it, not because no rule mentions it. Presence of the universal deliverables is L0's
    # job; this is the allowlist. A declared companion that is absent is a WARN: the bundle offers
    # it, the take does not owe it.
    allowed_files = registry.allowed_take_files(source)
    present = {p.name for p in take_dir.iterdir() if p.is_file()}
    for name in sorted(present - allowed_files):
        if f"take_files.{name}" in waived:
            continue
        findings.append(
            Finding(
                "L0", "conformance", "FAIL",
                f"undeclared file {name!r} in take directory; declare it under the profile's "
                "additive.take_deliverables or remove it",
                take_id=take_id,
            )
        )
    for name in sorted(registry.optional_take_files(source) - present):
        if f"take_files.{name}" in waived:
            continue
        findings.append(
            Finding(
                "L0", "conformance", "WARN",
                f"declared companion {name!r} absent from this take",
                take_id=take_id,
            )
        )
    return findings


def check_l4_dataset(dataset_dir) -> list[Finding]:
    dataset_dir = Path(dataset_dir)
    findings: list[Finding] = []
    descs = list(dataset_dir.glob(_DESC_GLOB))
    if not descs:
        findings.append(Finding("L4", "governance", "FAIL", "no *DATA_DESCRIPTION*.md at dataset root"))
    for p in descs:
        if "INTERNAL-ONLY" not in p.read_text(encoding="utf-8", errors="ignore"):
            findings.append(Finding("L4", "governance", "FAIL", f"{p.name} missing INTERNAL-ONLY banner"))
    return findings
