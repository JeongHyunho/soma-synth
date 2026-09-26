"""Eight virtual IMU sites from GAITEX marker clusters, on the frames the SMPL retarget kept.

The decisions below are recorded in ADR-0039 (2026-09-06). In brief:

* **Placement**. A site whose registry offset is ``unresolved`` is placed by a policy the
  settings file declares per site: ``offset_twin`` borrows another registered site's plate offset
  (``back_T4`` borrows ``chest``, because it is the same THOR plate with the same worn unit beneath
  it), ``declared_zero`` puts the sensor at the rigid body's own origin and says so. Nothing is
  defaulted: a site without a policy and without a resolved offset is refused.
* **Sensor frame**. ``imu_orientation`` is ``q_world_from_sensor`` in the spec frame S
  (+Y proximal, +Z outward lateral or subject-right, +X = Y x Z). S is built from landmarks per
  site, frame by frame in world, carried into the cluster frame and averaged into ONE constant
  relabel per take, which is what ``synthesise_virtual_imu`` takes as ``rotation_cluster_from_sensor``.
  The construction kinds are few and named (``limb``, ``foot``, ``forearm``, ``trunk``, ``head``);
  which landmarks each site uses is data in the settings file, not a constant here.
* **Forearm** (ADR-0037 section 5). The three upper-limb markers are not a rigid body -- the
  epicondyle sits on the humerus -- so the forearm pose is built anatomically and conditioned
  with the styloid-pair distance and an angular-rate ceiling instead of the plate rigidity test.
* **Pair window**. Small keeps every frame of the retarget's span except leading and
  trailing frames on which no site is valid. Inside the window a site's invalid cells are NaN in
  all three arrays and False in ``imu_valid_mask``. One thing IS bridged, before the signal is
  synthesised: a gap in a cluster's pose no longer than the settings' ``max_interpolated_gap_frames``
  is filled by the shared gap policy (Hermite translation, squad rotation) and those frames ship
  as valid; the manifest lists them per site (``interpolated_runs_bundle_rows``). Longer gaps,
  filter edges and withdrawn glitches are not filled and are NaN.

Every numeric constant comes from the settings document (``gaitex_synthesis_v1_1.yaml``); this
module holds none: thresholds live in versioned config, not in code constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from soma_synth.contracts.qmd_unified8_smpl18_spec import SITE_ORDER, SITE_TO_JOINT
from soma_synth.kinematics import gap_policy, markers, rigid_body
from soma_synth.sensors import lever_arm, site_registry, virtual_imu
from soma_synth.validation import gaitex_reference

__all__ = [
    "MarkerSmallError",
    "PairWindow",
    "SiteSynthesis",
    "SynthesisSettings",
    "compare_with_worn_units",
    "load_settings",
    "pair_window",
    "relabel_dispersion_vs_large",
    "stack_small",
    "synthesise_site",
    "synthesise_sites",
]

FRAME_KINDS = ("limb", "foot", "forearm", "trunk", "head")
PLACEMENT_KINDS = ("registry", "offset_twin", "declared_zero")


class MarkerSmallError(ValueError):
    """Raised when a site cannot be synthesised as configured, or the result violates its own rules."""


# --------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class SynthesisSettings:
    """The settings document, parsed once. ``document`` keeps the raw mapping for the manifest."""

    config_id: str
    version: str
    dt_s: float
    cutoff_hz: float
    filter_order: int
    edge_trim_frames: int
    max_gap_frames: int
    max_marker_speed_m_per_s: float
    rigid_tolerance_m: float
    gravity_world_m_per_s2: np.ndarray
    forearm_pair_tolerance_m: float
    forearm_max_rate_deg_per_s: float
    forearm_max_gap_twist_deg: float
    forearm_label_swap_max_withdrawn_fraction: float
    frame_minimum_frames: int
    placement: Mapping[str, Mapping[str, object]]
    frame_rules: Mapping[str, Mapping[str, object]]
    document: Mapping[str, object] = field(repr=False)


def load_settings(path: str | Path) -> SynthesisSettings:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("grid", "filter", "gap_policy", "marker_conditioning", "frames",
                "forearm_conditioning", "placement", "sensor_frame_rules"):
        if key not in document:
            raise MarkerSmallError(f"{path}: settings lack the {key!r} block")
    forearm = document["forearm_conditioning"]
    settings = SynthesisSettings(
        config_id=str(document["config_id"]),
        version=str(document["version"]),
        dt_s=float(document["grid"]["dt_s"]),
        cutoff_hz=float(document["filter"]["cutoff_hz"]),
        filter_order=int(document["filter"]["order"]),
        edge_trim_frames=int(document["filter"]["edge_trim_frames"]),
        max_gap_frames=int(document["gap_policy"]["max_interpolated_gap_frames"]),
        max_marker_speed_m_per_s=float(document["marker_conditioning"]["max_marker_speed_m_per_s"]),
        rigid_tolerance_m=float(document["marker_conditioning"]["rigid_distance_tolerance_m"]),
        gravity_world_m_per_s2=np.asarray(document["frames"]["gravity_world_m_per_s2"], dtype=np.float64),
        forearm_pair_tolerance_m=float(forearm["styloid_pair_distance_tolerance_m"]),
        forearm_max_rate_deg_per_s=float(forearm["max_rate_deg_per_s"]),
        forearm_max_gap_twist_deg=float(forearm["max_gap_twist_deg"]),
        forearm_label_swap_max_withdrawn_fraction=float(forearm["label_swap_max_withdrawn_fraction"]),
        frame_minimum_frames=int(document["sensor_frame_rules"]["minimum_frames"]),
        placement={k: v for k, v in document["placement"].items() if isinstance(v, dict)},
        frame_rules={k: v for k, v in document["sensor_frame_rules"].items() if isinstance(v, dict)},
        document=document,
    )
    for name, rule in settings.frame_rules.items():
        if rule.get("kind") not in FRAME_KINDS:
            raise MarkerSmallError(f"{path}: sensor_frame_rules[{name!r}].kind must be one of {FRAME_KINDS}")
    for name, policy in settings.placement.items():
        if policy.get("kind") not in PLACEMENT_KINDS:
            raise MarkerSmallError(f"{path}: placement[{name!r}].kind must be one of {PLACEMENT_KINDS}")
    return settings


# --------------------------------------------------------------------------- small helpers
def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _centroid(positions: np.ndarray) -> np.ndarray:
    visible = np.isfinite(positions).all(axis=2)
    count = visible.sum(axis=1)
    summed = np.where(visible[:, :, None], positions, 0.0).sum(axis=1)
    return np.where(count[:, None] > 0, summed / np.maximum(count, 1)[:, None], np.nan)


def _midpoint(stream, names: Sequence[str]) -> np.ndarray:
    """Mean of the named markers that this trial carries, per frame; NaN where a carried one is occluded.

    A landmark set is a description, not a requirement: twelve of the 73 trials are short a
    marker or two, and the direction a midpoint gives is the same whether two or one of the
    malleoli supply it. A set of which the trial carries nothing is refused.
    """
    present = _present(stream, names)
    positions, _ = markers.apply_occlusion_sentinel(stream.select(present))
    return positions.mean(axis=1)


def _present(stream, names: Sequence[str]) -> list[str]:
    """The named markers this trial carries, in the order named; refused when it carries none."""
    present = [name for name in names if name in stream.marker_names]
    if not present:
        raise MarkerSmallError(f"none of the landmarks {list(names)} is in this trial")
    return present


def _optional(value: float) -> float | None:
    """A diagnostic that is undefined is written as null, never as NaN (the manifest is JSON)."""
    return float(value) if np.isfinite(value) else None


def _twist_about_y_deg(relative: np.ndarray) -> float:
    """The swing-twist decomposition's twist angle about the local +Y of a relative rotation."""
    xyzw = Rotation.from_matrix(relative).as_quat()
    return float(np.degrees(2.0 * np.arctan2(abs(xyzw[1]), abs(xyzw[3]))))


def _wxyz_from_matrices(rotation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    quaternion = np.full((rotation.shape[0], 4), np.nan)
    if mask.any():
        xyzw = Rotation.from_matrix(rotation[mask]).as_quat()
        quaternion[mask] = np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])
    return quaternion


def _matrices_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(
        np.column_stack([quaternion[:, 1], quaternion[:, 2], quaternion[:, 3], quaternion[:, 0]])
    ).as_matrix()


def _mean_rotation(rotations: np.ndarray) -> np.ndarray:
    """Chordal mean of (n,3,3) rotations, projected onto SO(3)."""
    u, _s, vt = np.linalg.svd(rotations.mean(axis=0))
    mean = u @ vt
    if np.linalg.det(mean) < 0:
        u = u.copy()
        u[:, -1] *= -1
        mean = u @ vt
    return mean


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    return virtual_imu.contiguous_runs(np.asarray(mask, dtype=bool))


# --------------------------------------------------------------------------- poses
@dataclass(frozen=True)
class BodyPose:
    """A rigid body's pose per frame: ``world = rotation @ local + origin``; ``solved`` says where."""

    rotation: np.ndarray
    origin: np.ndarray
    solved: np.ndarray
    frame_source: str
    diagnostics: dict


def rigid_cluster_pose(stream, site: site_registry.SensorSite, settings: SynthesisSettings
                       ) -> tuple[BodyPose, markers.MarkerConditioning, np.ndarray]:
    """Stage A conditioning + rigid fit of a plate or head cluster. Returns pose, conditioning, local shape."""
    cluster = stream.select(list(site.markers))
    conditioned = markers.condition_cluster(
        cluster, dt_s=settings.dt_s, max_speed_m_per_s=settings.max_marker_speed_m_per_s,
        rigid_tolerance_m=settings.rigid_tolerance_m,
    )
    shape = rigid_body.mean_local_shape(conditioned.positions_m)
    pose = rigid_body.solve_pose_series(shape, conditioned.positions_m)
    residual = pose.residual_rms_m[pose.solved]
    body = BodyPose(
        rotation=pose.rotation, origin=pose.translation, solved=pose.solved,
        frame_source="rigid_cluster_fit(%s)" % ",".join(site.markers),
        diagnostics={
            "solved_fraction": float(pose.solved.mean()),
            "fit_residual_median_mm": float(np.median(residual) * 1e3) if residual.size else None,
            # samples are (frame, marker) cells; frames count a frame once however many markers went
            "withdrawn_sentinel_samples": int(conditioned.sentinel_mask.sum()),
            "withdrawn_speed_frames": int(conditioned.speed_mask.any(axis=1).sum()),
            "withdrawn_rigid_samples": int(conditioned.rigid_mask.sum()),
        },
    )
    return body, conditioned, shape


def anatomical_forearm_pose(stream, site: site_registry.SensorSite, settings: SynthesisSettings) -> BodyPose:
    """Forearm frame from (epicondyle, radial styloid, ulnar styloid), conditioned for what a plate test cannot see.

    Long axis: styloid midpoint -> epicondyle. Second axis: the styloid pair. The third is then
    determined. ``origin`` is the styloid midpoint. Axis LABELS here are a working frame, not the
    spec frame S -- ``sensor_frame_relabel`` builds S from the same landmarks.

    Conditioning, in order: occlusion sentinel; marker speed ceiling; styloid-pair distance
    outside its own median by more than the declared tolerance (a ghost marker or a one-sided
    jump changes that distance, elbow motion does not); frame-to-frame rotation rate of the built
    frame above the declared ceiling, which withdraws both frames of the step.

    A radial/ulnar LABEL SWAP leaves the pair distance unchanged and, when it happens inside a
    gap, is invisible to the rate test too. It is a half turn about the long axis, so the twist
    about +Y between the last usable frame before each interior gap and the first after it is
    compared against ``forearm_max_gap_twist_deg``; every crossing above it flips the labelling
    PARITY of the frames that follow. GAITEX has no hand marker that could say which parity is
    the anatomical one, so the parity carrying more usable frames is taken as it, and the frames
    of the other parity are WITHDRAWN (not relabelled) and listed on the manifest. When the
    minority exceeds ``forearm_label_swap_max_withdrawn_fraction`` of the usable frames the
    majority is no longer a reason and the site is refused for the take (``MarkerSmallError``).
    A trial whose labels are swapped throughout has no crossing to see and is not detected here.
    """
    names = list(site.markers)
    if len(names) != 3:
        raise MarkerSmallError(f"{site.name}: a forearm site needs (HLE, RSP, USP), got {names}")
    points, _ = markers.apply_occlusion_sentinel(stream.select(names))
    fast = markers.flag_speed_outliers(points, dt_s=settings.dt_s,
                                       max_speed_m_per_s=settings.max_marker_speed_m_per_s)
    points = points.copy()
    points[fast] = np.nan
    usable = np.isfinite(points).all(axis=(1, 2))
    withdrawn_speed = int(fast.any(axis=1).sum())

    elbow, radial, ulnar = points[:, 0], points[:, 1], points[:, 2]
    pair = np.linalg.norm(radial - ulnar, axis=1)
    reference = float(np.nanmedian(pair[usable])) if usable.any() else float("nan")
    pair_bad = usable & (np.abs(pair - reference) > settings.forearm_pair_tolerance_m)
    usable &= ~pair_bad

    wrist = 0.5 * (radial + ulnar)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = _unit(elbow - wrist)
        lateral = radial - ulnar
        lateral = _unit(lateral - y * np.sum(lateral * y, axis=1, keepdims=True))
        x = _unit(np.cross(lateral, y))
        rotation = np.stack([x, y, np.cross(x, y)], axis=2)

    rate_bad = np.zeros_like(usable)
    both = usable[:-1] & usable[1:]
    if both.any():
        relative = np.einsum("nji,njk->nik", rotation[:-1][both], rotation[1:][both])
        angle = np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(), axis=1)) / settings.dt_s
        step = np.where(both)[0][angle > settings.forearm_max_rate_deg_per_s]
        rate_bad[step] = True
        rate_bad[step + 1] = True
    usable &= ~rate_bad

    # label-swap parity across interior gaps (see the docstring)
    runs = _contiguous_runs(usable)
    gap_twist: list[float] = []
    crossings: list[dict] = []
    parity = 0
    parities: list[int] = []
    for previous, current in zip(runs[:-1], runs[1:]):
        before, after = previous[1] - 1, current[0]
        twist = _twist_about_y_deg(rotation[before].T @ rotation[after])
        gap_twist.append(twist)
        if twist > settings.forearm_max_gap_twist_deg:
            parity ^= 1
            crossings.append({"before": int(before), "after": int(after), "twist_deg": round(twist, 2)})
        parities.append(parity)
    parities = [0] + parities                      # the first run defines parity 0
    frames_of = [sum(b - a for (a, b), p in zip(runs, parities) if p == q) for q in (0, 1)]
    minority = int(np.argmin(frames_of))
    withdrawn_runs = [[int(a), int(b)] for (a, b), p in zip(runs, parities) if p == minority and frames_of[minority] > 0]
    withdrawn_swap = int(sum(b - a for a, b in withdrawn_runs))
    if withdrawn_swap:
        fraction = withdrawn_swap / float(usable.sum())
        if fraction > settings.forearm_label_swap_max_withdrawn_fraction:
            raise MarkerSmallError(
                f"{site.name}: {len(crossings)} label-swap crossings leave {fraction:.0%} of the usable "
                f"frames in the minority parity (limit {settings.forearm_label_swap_max_withdrawn_fraction:.0%}); "
                f"which labelling is anatomical cannot be decided, so the site is refused for this take")
        for a, b in withdrawn_runs:
            usable[a:b] = False

    # the two humeral edges: a swap exchanges them, so their separation says whether the pair-distance
    # test could ever see one (it cannot; the twist test above is what does)
    edge_radial = np.linalg.norm(radial - elbow, axis=1)
    edge_ulnar = np.linalg.norm(ulnar - elbow, axis=1)
    edges = usable & np.isfinite(edge_radial) & np.isfinite(edge_ulnar)
    edge_radial_mm = float(np.median(edge_radial[edges]) * 1e3) if edges.any() else float("nan")
    edge_ulnar_mm = float(np.median(edge_ulnar[edges]) * 1e3) if edges.any() else float("nan")

    rotation = np.where(usable[:, None, None], rotation, np.nan)
    origin = np.where(usable[:, None], wrist, np.nan)
    return BodyPose(
        rotation=rotation, origin=origin, solved=usable,
        frame_source="anatomical_forearm_frame(%s)" % ",".join(names),
        diagnostics={
            "solved_fraction": float(usable.mean()),
            "fit_residual_median_mm": None,
            "withdrawn_speed_frames": withdrawn_speed,
            "withdrawn_styloid_pair": int(pair_bad.sum()),
            "withdrawn_rate": int(rate_bad.sum()),
            "styloid_pair_reference_mm": _optional(reference * 1e3),
            "gaps_checked_for_twist": len(gap_twist),
            "gap_twist_max_deg": _optional(max(gap_twist)) if gap_twist else None,
            # crossings and runs are on the CSV (whole-trial) frame index, not bundle rows: the parity
            # is judged over the whole trial, and a withdrawn island may lie outside the pair window
            "label_swap_crossings_csv_frames": crossings,
            "withdrawn_label_swap_frames": withdrawn_swap,
            "withdrawn_label_swap_runs_csv_frames": withdrawn_runs,
            "humeral_edge_median_mm": {"hle_rsp": _optional(edge_radial_mm), "hle_usp": _optional(edge_ulnar_mm)},
        },
    )


# --------------------------------------------------------------------------- placement
@dataclass(frozen=True)
class Placement:
    kind: str
    source: str
    offset_site: site_registry.SensorSite | None
    provisional: bool

    def as_manifest(self) -> dict:
        return {"kind": self.kind, "source": self.source, "provisional": self.provisional}


def site_placement(site: site_registry.SensorSite, registry: site_registry.SiteRegistry,
                   settings: SynthesisSettings) -> Placement:
    """Where the sensor sits relative to the body: the registry when it says, the policy when it does not."""
    policy = settings.placement.get(site.name)
    if site.offset.is_resolved:
        if policy is not None and policy.get("kind") != "registry":
            raise MarkerSmallError(
                f"{site.name}: the registry resolves its offset, but placement policy says {policy.get('kind')!r}")
        return Placement("registry", "gaitex_sensor_sites_v1:%s" % site.name, site, False)
    if policy is None:
        raise MarkerSmallError(
            f"{site.name}: offset is unresolved ({site.offset.unresolved_reason}) and no placement "
            f"policy is declared; a default here would be a hidden decision")
    kind = str(policy["kind"])
    if kind == "offset_twin":
        twin = registry.require(str(policy["twin"]))
        if not twin.offset.is_resolved:
            raise MarkerSmallError(f"{site.name}: twin {twin.name} has no resolved offset either")
        if tuple(twin.markers) != tuple(site.markers):
            raise MarkerSmallError(
                f"{site.name}: twin {twin.name} is a different cluster ({twin.markers} vs {site.markers})")
        return Placement("offset_twin", "gaitex_sensor_sites_v1:%s" % twin.name, twin, True)
    if kind == "declared_zero":
        return Placement("declared_zero", "gaitex_synthesis_v1_1:placement", None, True)
    raise MarkerSmallError(f"{site.name}: placement kind {kind!r} cannot resolve an unresolved offset")


def plate_geometry(stream, offset_site: site_registry.SensorSite, conditioned: markers.MarkerConditioning,
                   pose: BodyPose, shape: np.ndarray) -> tuple[lever_arm.PlateFrame, np.ndarray, dict]:
    """Plate frame with its normal signed towards the body, the lever arm the registry declares, and
    how unanimously the frames agreed on that sign.

    The inward-reference markers pass through the occlusion sentinel first. GAITEX writes an
    occluded marker as exactly (0, 0, 0), which is a point at the laboratory origin metres away:
    left in, a few such frames outweigh every real one in the mean and can flip the normal, the
    lever arm and every axis rule that reads the plate. ``inward_sign_agreement`` is the fraction
    of usable frames whose own inward projection has the chosen sign; below one half the sign is
    contradicted by most of the take and the site is refused.
    """
    if offset_site.inward_reference is None:
        raise MarkerSmallError(f"{offset_site.name}: a plate offset needs an inward reference to sign its normal")
    references = _present(stream, offset_site.inward_reference.markers)
    reference_positions, _ = markers.apply_occlusion_sentinel(stream.select(references))
    inward_world = site_registry.inward_reference_world(_centroid(conditioned.positions_m), reference_positions)
    inward_local = lever_arm.inward_reference_local(pose.rotation, inward_world)
    plate = lever_arm.estimate_plate_frame(shape, inward_reference_local=inward_local)
    lever = lever_arm.lever_arm_local(
        plate, depth_m=float(offset_site.offset.depth_m),
        in_plane_offset_m=np.asarray(offset_site.offset.in_plane_offset_m, dtype=np.float64))
    usable = pose.solved & np.isfinite(inward_world).all(axis=1)
    projection = np.einsum("fji,fj->fi", pose.rotation[usable], inward_world[usable]) @ plate.normal_local
    agreement = float((projection > 0.0).mean()) if usable.any() else float("nan")
    if not agreement > 0.5:
        raise MarkerSmallError(
            f"{offset_site.name}: the plate normal's sign is contradicted on {1.0 - agreement:.0%} of "
            f"the usable frames; the inward reference does not decide it")
    diagnostics = {
        "inward_reference_markers": references,
        "inward_reference_frames": int(usable.sum()),
        "inward_sign_agreement": agreement,
    }
    return plate, lever, diagnostics


# --------------------------------------------------------------------------- sensor frame S
def _axis_from_rule(spec, stream, pose: BodyPose, plate: lever_arm.PlateFrame | None,
                    world_up: np.ndarray, minimum_frames: int, missing: list[str]) -> tuple[np.ndarray, str]:
    """One S axis per frame in WORLD coordinates (NaN where undefined), and which construction gave it.

    ``spec`` may be a list of alternatives, tried in order; the first that the trial can build on
    at least ``minimum_frames`` solved frames wins -- a landmark that is carried but occluded
    nearly throughout falls through the same as one that is absent -- and its name is returned so
    the manifest can say which one it was. Landmarks named but not carried are appended to
    ``missing``.
    """
    if isinstance(spec, list):
        failures = []
        for candidate in spec:
            attempt: list[str] = []
            try:
                axis, used = _axis_from_rule(candidate, stream, pose, plate, world_up, minimum_frames, attempt)
            except MarkerSmallError as error:
                failures.append(str(error))
                continue
            buildable = int((pose.solved & np.isfinite(axis).all(axis=1)).sum())
            if buildable < minimum_frames:
                failures.append(f"{used}: only {buildable} solved frames carry it (need {minimum_frames})")
                continue
            missing.extend(attempt)
            return axis, used
        raise MarkerSmallError("no alternative of the axis rule applies: " + "; ".join(failures))
    kind = str(spec["kind"])
    sign = float(spec.get("sign", 1.0))
    if sign not in (1.0, -1.0):
        raise MarkerSmallError(f"axis rule sign must be +1 or -1, got {sign}")
    axis, used = _axis_from_kind(kind, spec, stream, pose, plate, world_up, missing)
    if sign < 0:
        used = "-" + used
    return sign * axis, used


def _named(stream, names: Sequence[str], missing: list[str]) -> str:
    """The label of a landmark set as this trial carries it; the absent ones go to ``missing``."""
    present = _present(stream, names)
    missing.extend(name for name in names if name not in present)
    return "+".join(present)


def _axis_from_kind(kind: str, spec: Mapping[str, object], stream, pose: BodyPose,
                    plate: lever_arm.PlateFrame | None, world_up: np.ndarray,
                    missing: list[str]) -> tuple[np.ndarray, str]:
    frames = pose.rotation.shape[0]
    if kind == "landmarks":
        return _midpoint(stream, spec["to"]) - _midpoint(stream, spec["from"]), "landmarks(%s->%s)" % (
            _named(stream, spec["from"], missing), _named(stream, spec["to"], missing))
    if kind == "toward_cluster_origin":
        return pose.origin - _midpoint(stream, spec["from"]), "toward_cluster_origin(%s)" % _named(
            stream, spec["from"], missing)
    if kind == "plate_normal_outward":
        if plate is None:
            raise MarkerSmallError("plate_normal_outward needs a plate frame, and this site has none")
        # normal_local points INTO the body; outward is its negation, carried to world per frame
        return -np.einsum("tij,j->ti", pose.rotation, plate.normal_local), "plate_normal_outward"
    if kind == "world_up":
        return np.broadcast_to(world_up, (frames, 3)).copy(), "world_up"
    if kind == "plane_normal":
        names = list(spec["markers"])
        if len(names) != 3 or any(name not in stream.marker_names for name in names):
            raise MarkerSmallError(f"plane_normal needs exactly three carried markers, got {names}")
        points, _ = markers.apply_occlusion_sentinel(stream.select(names))
        normal = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        sign = np.sign(np.sum(normal * world_up, axis=1))
        sign = np.where(sign == 0, 1.0, sign)
        return normal * sign[:, None], "plane_normal(%s)" % "+".join(names)
    if kind == "styloid_dorsal":
        # (radial - ulnar) x (proximal) is the palmar/dorsal axis of the wrist; the rule's `sign`
        # (+1 right, -1 left, applied by the caller) turns it out of the back of the hand on both arms
        radial_minus_ulnar = _midpoint(stream, [spec["radial"]]) - _midpoint(stream, [spec["ulnar"]])
        proximal = _midpoint(stream, spec["proximal_to"]) - _midpoint(stream, spec["proximal_from"])
        _named(stream, spec["proximal_to"], missing)
        _named(stream, spec["proximal_from"], missing)
        return np.cross(radial_minus_ulnar, proximal), "styloid_dorsal(%s-%s x proximal)" % (spec["radial"], spec["ulnar"])
    raise MarkerSmallError(f"unknown axis rule kind {kind!r}")


def sensor_frame_relabel(rule: Mapping[str, object], stream, pose: BodyPose,
                         plate: lever_arm.PlateFrame | None, world_up: np.ndarray,
                         minimum_frames: int) -> tuple[np.ndarray, dict]:
    """The constant rotation carrying sensor-frame S coordinates into the cluster frame.

    The rule names a primary axis (kept exactly), a secondary axis (orthogonalised against the
    primary) and which of X/Y/Z each is; the third completes a right-handed triad. Axes are built
    per frame in world from landmarks, expressed in the cluster frame, and averaged over solved
    frames, so the relabel is one matrix per take and the orientation stays a rigid relabel of the
    fitted body -- which is the property the L3 check reads. Fewer than ``minimum_frames`` frames
    able to build both axes refuses the site.
    """
    primary_name = str(rule["primary"])
    secondary_name = str(rule["secondary"])
    if {primary_name, secondary_name} - {"x", "y", "z"} or primary_name == secondary_name:
        raise MarkerSmallError(f"frame rule names axes {primary_name!r}/{secondary_name!r}")
    missing: list[str] = []
    primary_world, primary_used = _axis_from_rule(
        rule[primary_name], stream, pose, plate, world_up, minimum_frames, missing)
    secondary_world, secondary_used = _axis_from_rule(
        rule[secondary_name], stream, pose, plate, world_up, minimum_frames, missing)
    usable = pose.solved & np.isfinite(primary_world).all(axis=1) & np.isfinite(secondary_world).all(axis=1)
    if int(usable.sum()) < minimum_frames:
        raise MarkerSmallError(f"fewer than {minimum_frames} frames can build the sensor frame")
    rotation_t = np.transpose(pose.rotation[usable], (0, 2, 1))          # cluster <- world
    primary_local = _unit(np.einsum("tij,tj->ti", rotation_t, primary_world[usable])).mean(axis=0)
    secondary_local = _unit(np.einsum("tij,tj->ti", rotation_t, secondary_world[usable])).mean(axis=0)
    primary_local = _unit(primary_local)
    secondary_local = secondary_local - primary_local * np.dot(secondary_local, primary_local)
    secondary_local = _unit(secondary_local)
    axes = {primary_name: primary_local, secondary_name: secondary_local}
    third_name = ({"x", "y", "z"} - {primary_name, secondary_name}).pop()
    # right-handed completion: x = y x z, y = z x x, z = x x y
    if third_name == "x":
        axes["x"] = np.cross(axes["y"], axes["z"])
    elif third_name == "y":
        axes["y"] = np.cross(axes["z"], axes["x"])
    else:
        axes["z"] = np.cross(axes["x"], axes["y"])
    relabel = np.column_stack([axes["x"], axes["y"], axes["z"]])
    if abs(np.linalg.det(relabel) - 1.0) > 1e-6:
        raise MarkerSmallError("sensor frame relabel is not a proper rotation")
    # how far the per-frame construction scatters about the constant: a large value means the
    # landmarks and the cluster do not move together (the head-plane or a limb landmark on skin)
    per_frame = _unit(np.einsum("tij,tj->ti", rotation_t, primary_world[usable]))
    scatter = np.degrees(np.arccos(np.clip(per_frame @ primary_local, -1.0, 1.0)))
    return relabel, {
        "kind": str(rule["kind"]), "primary": primary_name, "secondary": secondary_name,
        "primary_construction": primary_used, "secondary_construction": secondary_used,
        "landmarks_missing": sorted(set(missing)),
        "frames_used": int(usable.sum()),
        "primary_axis_scatter_deg_median": float(np.median(scatter)),
        "primary_axis_scatter_deg_p95": float(np.percentile(scatter, 95)),
    }


def _plate_against_landmarks(rule: Mapping[str, object], stream, pose: BodyPose, plate: lever_arm.PlateFrame,
                             world_up: np.ndarray, minimum_frames: int) -> float | None:
    """Where an axis rule offers both the plate's outward normal and a landmark construction, the
    angle between the two in the cluster frame. They describe the same direction, so a value near
    180 deg means the plate normal's sign is wrong (an inward reference that failed) or the
    landmarks are mislabelled; either way the site must not ship. None when the trial cannot
    build the landmark alternative."""
    for axis in ("x", "y", "z"):
        candidates = rule.get(axis)
        if not isinstance(candidates, list):
            continue
        kinds = [str(c.get("kind")) for c in candidates]
        if "plate_normal_outward" not in kinds or "landmarks" not in kinds:
            continue
        landmark_rule = candidates[kinds.index("landmarks")]
        try:
            axis_world, _used = _axis_from_rule(landmark_rule, stream, pose, plate, world_up, minimum_frames, [])
        except MarkerSmallError:
            return None
        usable = pose.solved & np.isfinite(axis_world).all(axis=1)
        if int(usable.sum()) < minimum_frames:
            return None
        rotation_t = np.transpose(pose.rotation[usable], (0, 2, 1))
        landmark_local = _unit(_unit(np.einsum("tij,tj->ti", rotation_t, axis_world[usable])).mean(axis=0))
        outward = -plate.normal_local
        return float(np.degrees(np.arccos(np.clip(landmark_local @ outward, -1.0, 1.0))))
    return None


# --------------------------------------------------------------------------- one site
@dataclass(frozen=True)
class SiteSynthesis:
    name: str
    signal: virtual_imu.VirtualImuSignal
    rotation_cluster_from_sensor: np.ndarray
    lever_arm_local: np.ndarray
    placement: Placement
    frame_source: str
    diagnostics: dict
    interpolated: np.ndarray     # (frames,) True where the gap policy bridged the pose (still valid)

    def manifest_block(self) -> dict:
        block = {"frame_source": self.frame_source, "placement": self.placement.as_manifest(),
                 "lever_arm_local_m": [round(float(v), 5) for v in self.lever_arm_local]}
        block.update(self.diagnostics)
        return block


def synthesise_site(stream, site_name: str, registry: site_registry.SiteRegistry,
                    settings: SynthesisSettings) -> SiteSynthesis:
    site = registry.require(site_name)
    rule = settings.frame_rules.get(site_name)
    if rule is None:
        raise MarkerSmallError(f"{site_name}: no sensor_frame_rules entry; the spec frame cannot be guessed")
    placement = site_placement(site, registry, settings)
    world_up = -_unit(settings.gravity_world_m_per_s2)

    if rule["kind"] == "forearm":
        pose = anatomical_forearm_pose(stream, site, settings)
        plate = None
        conditioned = None
        shape = None
    else:
        pose, conditioned, shape = rigid_cluster_pose(stream, site, settings)
        plate = None
    lever = np.zeros(3)
    plate_diagnostics: dict = {}
    if placement.offset_site is not None:
        if conditioned is None:
            raise MarkerSmallError(f"{site_name}: a plate offset needs a rigid cluster, not a forearm frame")
        plate, lever, plate_diagnostics = plate_geometry(stream, placement.offset_site, conditioned, pose, shape)
    elif rule["kind"] in ("foot", "trunk") or any(
            isinstance(v, dict) and v.get("kind") == "plate_normal_outward" for v in rule.values()):
        # the frame rule needs the plate's normal even when no offset is taken from it
        if conditioned is None:
            raise MarkerSmallError(f"{site_name}: plate_normal_outward needs a rigid cluster")
        twin = registry.require(str(settings.placement.get(site_name, {}).get("plate_twin", site_name)))
        plate, _, plate_diagnostics = plate_geometry(stream, twin, conditioned, pose, shape)
    if plate is not None:
        plate_diagnostics.update({
            "plate_planarity_mm": float(plate.planarity_m * 1e3),
            "plate_edge_lengths_mm": [round(float(e) * 1e3, 2) for e in plate.edge_lengths_m]})

    relabel, frame_diagnostics = sensor_frame_relabel(rule, stream, pose, plate, world_up, settings.frame_minimum_frames)
    if plate is not None:
        against = _plate_against_landmarks(rule, stream, pose, plate, world_up, settings.frame_minimum_frames)
        frame_diagnostics["plate_outward_vs_landmark_axis_deg"] = against
        if against is not None and against > 90.0:
            raise MarkerSmallError(
                f"{site_name}: the plate's outward normal is {against:.0f} deg from the landmark axis that "
                f"names the same direction; the plate sign or the landmarks are wrong")

    quaternion = _wxyz_from_matrices(np.nan_to_num(pose.rotation, nan=0.0), pose.solved)
    filled = gap_policy.fill_pose_gaps(pose.origin, quaternion, pose.solved, max_gap_frames=settings.max_gap_frames)
    rotation = pose.rotation.copy()
    interpolated = filled.status == gap_policy.FrameStatus.INTERPOLATED
    if interpolated.any():
        rotation[interpolated] = _matrices_from_wxyz(filled.quaternion[interpolated])
    signal = virtual_imu.synthesise_virtual_imu(
        translation=np.nan_to_num(filled.translation, nan=0.0),
        rotation_world_from_cluster=np.nan_to_num(rotation, nan=0.0),
        usable=filled.usable,
        lever_arm_local=lever,
        rotation_cluster_from_sensor=relabel,
        dt_s=settings.dt_s,
        cutoff_hz=settings.cutoff_hz,
        filter_order=settings.filter_order,
        edge_trim_frames=settings.edge_trim_frames,
        gravity_world_m_per_s2=settings.gravity_world_m_per_s2,
    )
    diagnostics = dict(pose.diagnostics)
    diagnostics.update(plate_diagnostics)
    diagnostics["interpolated_frames"] = int(interpolated.sum())
    diagnostics["valid_fraction_of_trial"] = float(signal.valid.mean())
    diagnostics["sensor_frame"] = frame_diagnostics
    return SiteSynthesis(
        name=site_name, signal=signal, rotation_cluster_from_sensor=relabel, lever_arm_local=lever,
        placement=placement, frame_source=pose.frame_source, diagnostics=diagnostics,
        interpolated=interpolated & signal.valid,
    )


def synthesise_sites(stream, registry: site_registry.SiteRegistry, settings: SynthesisSettings,
                     site_order: Sequence[str] = SITE_ORDER) -> dict[str, SiteSynthesis]:
    return {name: synthesise_site(stream, name, registry, settings) for name in site_order}


# --------------------------------------------------------------------------- pair window
@dataclass(frozen=True)
class PairWindow:
    """The frames small and large share, in source (CSV) index and in retarget-array rows."""

    source_start: int
    source_stop: int
    retarget_start: int
    retarget_stop: int
    trimmed_leading: int
    trimmed_trailing: int
    valid: np.ndarray            # (T, n_sites) inside the window
    site_order: tuple[str, ...]
    interpolated: np.ndarray | None = None   # (T, n_sites) inside the window: valid AND bridged by the gap policy

    @property
    def frames(self) -> int:
        return self.source_stop - self.source_start

    @property
    def rows(self) -> slice:
        """Rows of the retarget arrays (pose, trans, timestamps) that the window covers."""
        return slice(self.source_start - self.retarget_start, self.source_stop - self.retarget_start)

    def usable_runs(self, site: str) -> list[tuple[int, int]]:
        """Contiguous valid runs of a site, in BUNDLE rows (0 = the first frame of the window)."""
        return _contiguous_runs(self.valid[:, self.site_order.index(site)])

    def interpolated_runs(self, site: str) -> list[tuple[int, int]]:
        """Contiguous runs the gap policy bridged, in bundle rows; empty when not recorded."""
        if self.interpolated is None:
            return []
        return _contiguous_runs(self.interpolated[:, self.site_order.index(site)])

    def as_manifest(self, trial_frames: int) -> dict:
        any_invalid = int((~self.valid.all(axis=1)).sum())
        sites = {}
        for k, name in enumerate(self.site_order):
            entry = {
                "valid_fraction": round(float(self.valid[:, k].mean()), 4),
                "usable_runs_bundle_rows": [[int(a), int(b)] for a, b in self.usable_runs(name)],
            }
            if self.interpolated is not None:
                entry["interpolated_frames"] = int(self.interpolated[:, k].sum())
                entry["interpolated_runs_bundle_rows"] = [[int(a), int(b)] for a, b in self.interpolated_runs(name)]
            sites[name] = entry
        return {
            "csv_frame_start": self.source_start, "csv_frame_stop": self.source_stop,
            "retarget_frame_start": self.retarget_start, "retarget_frame_stop": self.retarget_stop,
            "frames": self.frames,
            "frames_trimmed_at_ends": {"leading": self.trimmed_leading, "trailing": self.trimmed_trailing},
            "trial_frames": int(trial_frames),
            "fraction_of_trial": round(self.frames / float(trial_frames), 4),
            "fraction_of_retarget": round(self.frames / float(self.retarget_stop - self.retarget_start), 4),
            "frames_with_any_invalid_site": any_invalid,
            "invalid_cells": int((~self.valid).sum()),
            "runs_basis": "bundle rows: 0 is csv_frame_start, i.e. retarget row "
                          "(csv_frame_start - retarget_frame_start); half-open [start, stop)",
            "sites": sites,
        }


def pair_window(valid_by_site: np.ndarray, source_index: np.ndarray,
                site_order: Sequence[str] = SITE_ORDER,
                interpolated_by_site: np.ndarray | None = None) -> PairWindow:
    """Keep the retarget's frames, minus leading/trailing frames on which no site is valid.

    ``valid_by_site`` is (N_source, n_sites) on the marker file's own index; ``source_index`` is
    the retarget's contiguous run of source frames. Interior gaps stay in the window (NaN
    under a False mask); only the ends, where every site is inside a filter edge, are trimmed,
    because a frame no site can speak for carries nothing for the pair. ``interpolated_by_site``,
    same shape as ``valid_by_site``, marks the valid frames whose pose the gap policy bridged, so
    the manifest can list them.
    """
    valid_by_site = np.asarray(valid_by_site, dtype=bool)
    index = np.asarray(source_index, dtype=int)
    if index.ndim != 1 or index.size == 0 or not np.all(np.diff(index) == 1):
        raise MarkerSmallError("source_index must be one contiguous run of frames")
    if valid_by_site.ndim != 2 or valid_by_site.shape[1] != len(site_order):
        raise MarkerSmallError(f"valid_by_site must be (frames, {len(site_order)}), got {valid_by_site.shape}")
    inside = index[index < valid_by_site.shape[0]]
    if inside.size != index.size:
        raise MarkerSmallError("the retarget covers frames the marker file does not have")
    window = valid_by_site[index]
    any_valid = window.any(axis=1)
    if not any_valid.any():
        raise MarkerSmallError("no frame of the retarget span has a valid site")
    first = int(np.argmax(any_valid))
    last = int(len(any_valid) - np.argmax(any_valid[::-1]))
    interpolated = None
    if interpolated_by_site is not None:
        interpolated_by_site = np.asarray(interpolated_by_site, dtype=bool)
        if interpolated_by_site.shape != valid_by_site.shape:
            raise MarkerSmallError("interpolated_by_site must have the shape of valid_by_site")
        if (interpolated_by_site & ~valid_by_site).any():
            raise MarkerSmallError("an interpolated frame the signal does not call valid")
        interpolated = interpolated_by_site[index][first:last]
    return PairWindow(
        source_start=int(index[0] + first), source_stop=int(index[0] + last),
        retarget_start=int(index[0]), retarget_stop=int(index[-1] + 1),
        trimmed_leading=first, trimmed_trailing=int(len(any_valid) - last),
        valid=window[first:last], site_order=tuple(site_order), interpolated=interpolated,
    )


# --------------------------------------------------------------------------- assembly
def stack_small(syntheses: Mapping[str, SiteSynthesis], window: PairWindow,
                confidence_by_site: Mapping[str, float],
                site_order: Sequence[str] = SITE_ORDER) -> dict[str, np.ndarray]:
    """The four signal arrays, the mask and the confidence column for the window, mask-disciplined.

    A cell the mask calls valid is finite in every component; a cell it calls invalid is NaN in
    every component and 0.0 in confidence (ADR-0039). Both directions are asserted here so that
    a defect in the synthesis cannot reach the validator as a "clean" bundle.
    """
    sl = slice(window.source_start, window.source_stop)
    frames = window.frames
    n = len(site_order)
    orientation = np.full((frames, n, 4), np.nan, dtype=np.float32)
    acceleration = np.full((frames, n, 3), np.nan, dtype=np.float32)
    rate = np.full((frames, n, 3), np.nan, dtype=np.float32)
    valid = np.zeros((frames, n), dtype=bool)
    confidence = np.zeros((frames, n), dtype=np.float32)
    for k, name in enumerate(site_order):
        signal = syntheses[name].signal
        site_valid = signal.valid[sl]
        if not np.array_equal(site_valid, window.valid[:, k]):
            raise MarkerSmallError(f"{name}: window validity disagrees with the signal's own mask")
        orientation[site_valid, k] = signal.orientation_wxyz[sl][site_valid]
        acceleration[site_valid, k] = signal.acceleration_m_per_s2[sl][site_valid]
        rate[site_valid, k] = signal.angular_velocity_deg_per_s[sl][site_valid]
        valid[:, k] = site_valid
        confidence[site_valid, k] = float(confidence_by_site[name])
    for label, array in (("imu_orientation", orientation), ("imu_acceleration", acceleration),
                         ("imu_angular_velocity", rate)):
        finite = np.isfinite(array).all(axis=-1)
        if (valid & ~finite).any():
            raise MarkerSmallError(f"{label}: a valid cell is not finite")
        if (~valid & np.isfinite(array).any(axis=-1)).any():
            raise MarkerSmallError(f"{label}: an invalid cell carries a finite value")
    return {
        "imu_orientation": orientation, "imu_acceleration": acceleration,
        "imu_angular_velocity": rate, "imu_valid_mask": valid, "imu_confidence": confidence,
    }


# --------------------------------------------------------------------------- evidence
def compare_with_worn_units(syntheses: Mapping[str, SiteSynthesis], imu_stream, window: PairWindow,
                            registry: site_registry.SiteRegistry, settings: SynthesisSettings,
                            counterpart_twin: Mapping[str, str], minimum_frames: int) -> dict:
    """T1 orientation comparison against the worn XSens unit, on the window, per instrumented site."""
    counterparts = registry.measured_counterparts()
    validation = settings.document["validation"]
    n_imu = imu_stream.quaternions.shape[0]
    result: dict = {}
    for name, synthesis in syntheses.items():
        sensor = counterparts.get(counterpart_twin.get(name, name))
        if sensor is None:
            continue
        if sensor not in imu_stream.imu_names:
            result[name] = {"status": "measured_counterpart_absent", "sensor": sensor}
            continue
        measured = imu_stream.quaternions[:, imu_stream.index_of(sensor)]
        select = np.arange(window.source_start, min(window.source_stop, n_imu))
        ok = synthesis.signal.valid[select] & np.isfinite(measured[select]).all(axis=1)
        if int(ok.sum()) < minimum_frames:
            result[name] = {"status": "too_few_comparable_frames", "sensor": sensor, "frames": int(ok.sum())}
            continue
        # The frames stay on their own time axis with NaN where either side is missing, rather than
        # being packed together: the comparison differentiates both streams, and packing would put a
        # gap's two edges next to each other as if one sample apart.
        synthesised = np.where(ok[:, None], synthesis.signal.orientation_wxyz[select], np.nan)
        counterpart = np.where(ok[:, None], measured[select], np.nan)
        try:
            comparison = gaitex_reference.compare_orientation(
                synthesised, counterpart, dt_s=settings.dt_s,
                minimum_rate_rad_per_s=float(validation["minimum_rate_rad_per_s"]))
        except gaitex_reference.ValidationError as error:
            result[name] = {"status": "comparison_failed", "sensor": sensor, "note": str(error)}
            continue
        result[name] = {
            "status": "ok", "sensor": sensor, "frames": int(comparison.compared_frames),
            "median_deg": comparison.median_error_deg, "p95_deg": comparison.p95_error_deg,
            "max_deg": comparison.max_error_deg,
            "rate_correlation": comparison.angular_velocity_correlation,
        }
    return result


def _relative_rotations(orientation: np.ndarray, valid: np.ndarray, reference_wxyz: np.ndarray,
                        k: int, joint: int | None) -> np.ndarray | None:
    rows = valid[:, k]
    if int(rows.sum()) < 2:
        return None
    small_r = _matrices_from_wxyz(orientation[rows, k].astype(np.float64))
    reference = reference_wxyz[rows, k] if joint is None else reference_wxyz[rows, joint]
    reference_r = _matrices_from_wxyz(reference.astype(np.float64))
    return np.einsum("tji,tjk->tik", reference_r, small_r)


def relabel_dispersion_vs_large(orientation: np.ndarray, valid: np.ndarray,
                                large_global_wxyz: np.ndarray,
                                site_order: Sequence[str] = SITE_ORDER) -> dict[str, float | None]:
    """What validator L3 measures: median geodesic dispersion of gR^T @ R_small about its mean, per
    site. None where fewer than two valid frames exist (never NaN: the manifest is JSON)."""
    if orientation.shape[0] != large_global_wxyz.shape[0]:
        raise MarkerSmallError("small and large frame counts differ")
    out: dict[str, float | None] = {}
    for k, name in enumerate(site_order):
        relabel = _relative_rotations(orientation, valid, large_global_wxyz, k, SITE_TO_JOINT[name])
        if relabel is None:
            out[name] = None
            continue
        mean = _mean_rotation(relabel)
        relative = np.einsum("ji,tjk->tik", mean, relabel)
        trace = np.trace(relative, axis1=1, axis2=2)
        out[name] = float(np.median(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))))
    return out


def relabel_offset_vs_reference(orientation: np.ndarray, valid: np.ndarray, reference_wxyz: np.ndarray,
                                site_order: Sequence[str] = SITE_ORDER) -> dict[str, float | None]:
    """The MEAN relabel between this small and a reference small on the same sites, as one angle per
    site: the geodesic angle of the chordal mean of R_ref^T @ R_small over valid frames.

    The dispersion above cannot see a constant error. A plate whose normal was signed the wrong
    way, or a wrist whose styloids swapped labels for the whole take, is a perfectly constant
    relabel of the SMPL joint -- rotated by a half turn. Against the emitter's own SMPL-derived
    small, built with the spec's anatomical relabel, that shows as an offset near 180 deg where
    the disagreement of a correct labelling is tens of degrees at most.
    """
    if orientation.shape != reference_wxyz.shape:
        raise MarkerSmallError("small and reference orientation shapes differ")
    out: dict[str, float | None] = {}
    for k, name in enumerate(site_order):
        relabel = _relative_rotations(orientation, valid, reference_wxyz, k, None)
        if relabel is None:
            out[name] = None
            continue
        mean = _mean_rotation(relabel)
        out[name] = float(np.degrees(np.arccos(np.clip((np.trace(mean) - 1.0) / 2.0, -1.0, 1.0))))
    return out
