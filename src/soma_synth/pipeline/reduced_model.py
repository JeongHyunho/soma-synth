"""The 18-joint reduced model, fitted once per subject through ``smpl18``.

A bundle's ``large_reference.npz`` stores 18 joints: ``spine1``, ``spine2`` and both collars are
frozen to per-subject constants, the two hands are dropped, and the kept joint below each frozen
one absorbs what was removed. Everything about that reduction belongs to ``smpl18.reduce`` -- which
joints freeze, how the absorbers take over, how the four constants are fitted and what the fit
costs.

This module does the two things the package leaves to its caller, and nothing else:

* **which frames are pooled.** ``smpl18`` fits per subject ("the constants describe that body's
  posture in the frozen joints and must agree across its trials", ``docs/primer.md`` 5.2). The
  caller hands over every take of one subject, and the frames are sampled evenly over their
  concatenation, so a long take weighs what its length says rather than what a per-take cap would.
* **where the numbers come from.** The ``reduce`` section of the source's own ``smpl18`` profile,
  named in ``configs/datasets/source_pipelines_v1.yaml`` (``smpl18_profile``), never a default here.

It returns the four constants and the record ``docs/primer.md`` 5.3 asks a corpus to carry: the
constants, which joint absorbed each, the residual of the fit and of its starting guess, the
per-joint residual, the frames used and whether the solver converged.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import smpl18
from smpl18 import reduce as smpl18_reduce
from smpl18.profile import Profile
from smpl18.skeleton import definition as smpl18_skeleton
from smpl18.skeleton.rotations import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)

from soma_synth.pipeline import stages

__all__ = [
    "ABSORBERS",
    "AFFECTED_BY_FREEZE",
    "CorpusFits",
    "FIT_FILE",
    "FIT_FILE_SCHEMA",
    "FROZEN_JOINTS",
    "FROZEN_JOINT_NAMES",
    "JOINT18_NAMES",
    "KEEP18",
    "METHOD",
    "RECONSTRUCTION_RECIPE",
    "ReductionError",
    "ReductionSettings",
    "SubjectFit",
    "anthro_provenance",
    "fit_from_record",
    "fit_subject",
    "group_by_skeleton",
    "pooled_rotations",
    "read_fit_file",
    "reconstruction_block",
    "reduce_local",
    "settings_for_source",
    "write_fit_file",
]

#: The package's answers, re-exported so a generator has one import for all of them.
FROZEN_JOINTS: tuple[int, ...] = smpl18_skeleton.FROZEN_JOINTS
FROZEN_JOINT_NAMES: tuple[str, ...] = smpl18_skeleton.FROZEN_JOINT_NAMES
KEEP18: tuple[int, ...] = smpl18_skeleton.KEEP18
JOINT18_NAMES: tuple[str, ...] = smpl18_skeleton.JOINT18_NAMES
AFFECTED_BY_FREEZE: tuple[int, ...] = smpl18_skeleton.AFFECTED_BY_FREEZE
ABSORBERS: Mapping[int, int] = smpl18_skeleton.ABSORBERS

#: What a record names as the fit, so a reader can find the code that made the constants.
METHOD = "smpl18.reduce.fit_constants"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_JOINT_NAMES = smpl18_skeleton.JOINT_NAMES


class ReductionError(ValueError):
    """The reduction cannot be set up as asked: no profile, no settings, or unusable poses."""


@dataclass(frozen=True)
class ReductionSettings:
    """The settings a source's fit reads, and where they came from."""

    source_name: str
    #: The profile's merged settings; ``smpl18.reduce.fit_constants`` reads its ``reduce`` section.
    settings: Mapping[str, Any]
    #: Plain values for a manifest: the profile as the registry names it, its hash, the settings
    #: files it pulled in with their hashes, and the ``reduce`` numbers. No absolute path.
    provenance: Mapping[str, Any]

    @property
    def sample_frames(self) -> int:
        return smpl18_reduce.FitSettings.from_settings(self.settings).sample_frames


def _profile_argument(given: str) -> str:
    """A shipped profile name stays a name; a repository-relative path becomes absolute."""
    if "/" in given or "\\" in given or given.endswith((".yaml", ".yml")):
        return str(_REPOSITORY_ROOT / given)
    return given


def settings_for_source(
    source_name: str, registry: stages.PipelineRegistry | None = None
) -> ReductionSettings:
    """Load the source's ``smpl18`` profile and take its ``reduce`` settings.

    Refuses rather than falls back: a source with no profile, or a profile whose settings lack the
    ``reduce`` section, has no numbers to fit with, and inventing them here is what the settings
    discipline of both repositories forbids.
    """
    registry = registry or stages.default_registry()
    given = registry.for_source(source_name).smpl18_profile
    if not given:
        raise ReductionError(
            f"{source_name}: no smpl18_profile in {registry.path.name}, so there are no reduce "
            "settings to fit the frozen-joint constants with"
        )
    profile = Profile.load(_profile_argument(given))
    # FitSettings names a missing key itself; calling it here makes the failure happen at setup,
    # before any take is read, rather than on the first subject.
    reduce_settings = smpl18_reduce.FitSettings.from_settings(profile.settings)
    provenance = {
        "smpl18_profile": given,
        "profile_id": profile.id,
        "profile_sha256": profile.sha256,
        "settings_files": [
            {"given": entry.given, "sha256": entry.sha256}
            for entry in profile.referenced_files()
            if entry.role.startswith("settings")
        ],
        "reduce": {
            "sample_frames": reduce_settings.sample_frames,
            "optimiser": reduce_settings.optimiser,
            "max_evaluations": reduce_settings.max_evaluations,
        },
        "smpl18_version": smpl18.__version__,
    }
    return ReductionSettings(source_name=source_name, settings=profile.settings,
                             provenance=provenance)


def _as_pose_series(pose: np.ndarray, position: int) -> np.ndarray:
    array = np.asarray(pose, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (smpl18_skeleton.NUM_JOINTS, 3):
        raise ReductionError(
            f"take {position}: expected an ({smpl18_skeleton.NUM_JOINTS}-joint) axis-angle pose "
            f"of shape (T, {smpl18_skeleton.NUM_JOINTS}, 3), got {array.shape}"
        )
    return array


def pooled_rotations(poses: Sequence[np.ndarray], most: int) -> tuple[np.ndarray, int]:
    """Local rotations ``(N, 24, 3, 3)`` sampled evenly over the concatenation of ``poses``.

    ``poses`` are one subject's takes as axis-angle ``(T_k, 24, 3)``. The indices are the ones
    ``smpl18.reduce.sample_indices`` would pick from the concatenated array, found without building
    it, and only the picked frames are converted to matrices. Returns the rotations and the number
    of frames pooled before sampling.
    """
    if not poses:
        raise ReductionError("no takes to pool")
    series = [_as_pose_series(pose, position) for position, pose in enumerate(poses)]
    lengths = np.array([pose.shape[0] for pose in series], dtype=np.int64)
    total = int(lengths.sum())
    picks = smpl18_reduce.sample_indices(total, most)
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    owner = np.searchsorted(starts, picks, side="right") - 1
    rows = np.empty((picks.size, smpl18_skeleton.NUM_JOINTS, 3), dtype=np.float64)
    for position, pose in enumerate(series):
        chosen = owner == position
        rows[chosen] = pose[picks[chosen] - starts[position]]
    return axis_angle_to_matrix(rows), total


def _position_error(error: smpl18_reduce.PositionError) -> dict[str, float]:
    return {"rms": float(error.rms_m), "max": float(error.max_m)}


@dataclass(frozen=True)
class SubjectFit:
    """One subject's four constants and the record that says what they cost."""

    #: ``(4, 3, 3)`` in the order of ``FROZEN_JOINTS``.
    constants: np.ndarray
    record: dict[str, Any] = field(default_factory=dict)


def fit_subject(
    poses: Sequence[np.ndarray],
    rest_joints: np.ndarray,
    settings: ReductionSettings,
    *,
    group: str,
    basis: str,
    scope: str = "subject",
) -> SubjectFit:
    """Fit one subject's frozen-joint constants over all of its takes.

    ``poses`` are the takes as ``(T_k, 24, 3)`` axis-angle, ``rest_joints`` the ``(24, 3)``
    skeleton every one of them was posed on. ``group`` names what was pooled (normally the subject
    id), ``basis`` says which frames of each take were handed over, and ``scope`` is ``subject``
    unless a caller could only give one take -- in which case it must say ``take``.
    """
    rest = np.asarray(rest_joints, dtype=np.float64)
    if rest.shape != (smpl18_skeleton.NUM_JOINTS, 3):
        raise ReductionError(f"{group}: rest joints must be (24, 3), got {rest.shape}")
    local, pooled = pooled_rotations(poses, settings.sample_frames)
    fitted = smpl18_reduce.fit_constants(local, rest, settings=settings.settings)
    per_joint = smpl18_reduce.per_joint_rms(local, rest, fitted.constants)
    record = {
        "method": METHOD,
        "scope": scope,
        "group": group,
        "takes": len(poses),
        "frames_pooled": pooled,
        "frames_measured": int(fitted.frames),
        "frames_basis": basis,
        "frozen_joints": list(FROZEN_JOINT_NAMES),
        "absorbed_by": {_JOINT_NAMES[frozen]: _JOINT_NAMES[absorber]
                        for frozen, absorber in ABSORBERS.items()},
        # Full precision: a sharded run rebuilds the constants from this list (fit_from_record),
        # and every shard has to emit the same numbers.
        "constants_wxyz": matrix_to_quaternion(fitted.constants).tolist(),
        "residual_initial_m": _position_error(fitted.initial),
        "residual_fitted_m": _position_error(fitted.fitted),
        "residual_note": "joint-centre displacement of the joints the freeze can move, reduced "
                         "minus original, over the measured frames; initial is each frozen "
                         "joint's mean rotation, the solver's starting guess",
        "per_joint_rms_m": {name: float(value) for name, value in per_joint.items()},
        "converged": bool(fitted.success),
        "solver_message": fitted.message,
        "settings": dict(settings.provenance),
    }
    return SubjectFit(constants=np.asarray(fitted.constants, dtype=np.float64), record=record)


def group_by_skeleton(
    name: str, members: Sequence[tuple[str, np.ndarray, np.ndarray]]
) -> dict[str, tuple[list[str], list[np.ndarray], np.ndarray]]:
    """Split one subject's takes by the rest skeleton they were posed on.

    ``members`` are ``(take key, pose (T, 24, 3), rest joints (24, 3))``. One set of constants is
    only meaningful against one skeleton, so takes on different skeletons are fitted apart. A
    subject whose takes share one skeleton keeps ``name`` as its group name; otherwise each group
    is ``name|rest:<digest>`` so the split is visible in every record.
    Returns ``{group name: (take keys, poses, rest joints)}``.
    """
    by_digest: dict[str, tuple[list[str], list[np.ndarray], np.ndarray]] = {}
    for key, pose, rest in members:
        rest = np.asarray(rest, dtype=np.float64)
        digest = hashlib.sha256(np.ascontiguousarray(rest, "<f8").tobytes()).hexdigest()[:12]
        keys, poses, _ = by_digest.setdefault(digest, ([], [], rest))
        keys.append(key)
        poses.append(pose)
    if len(by_digest) == 1:
        return {name: next(iter(by_digest.values()))}
    return {f"{name}|rest:{digest}": group for digest, group in sorted(by_digest.items())}


def reduce_local(local_rotations: np.ndarray, constants: np.ndarray) -> np.ndarray:
    """Freeze the four joints at ``constants``; every distal world orientation is kept exactly.

    ``(T, 24, 3, 3)`` in and out: the absorbed rotation is written on the kept joint below, and
    ``KEEP18`` then picks the 18 a bundle stores.
    """
    return smpl18_reduce.apply(local_rotations, constants)


def anthro_provenance(fit: SubjectFit) -> str:
    """The ``fixed_joint_rotation_provenance`` string for constants that came from ``fit``."""
    record = fit.record
    return (
        f"reduced_model_fit: {METHOD} over {record['takes']} take(s) of {record['group']} "
        f"({record['scope']} scope; {record['frames_measured']} of {record['frames_pooled']} pooled "
        "frames measured), minimising the joint-centre displacement the freeze causes; spine3 and "
        "the shoulders absorb the removed rotation, so distal global orientation is exact. The "
        "manifest's anthro_reconstruction.reduced_model_fit carries the residuals."
    )


#: How a reader rebuilds the 22-joint chain from a bundle; one sentence for every source.
RECONSTRUCTION_RECIPE = (
    "world_joint[j] = pelvis_position_world_aux + FK(anthro.joint_position[t_pose], "
    "{large.joint_rotation(18, reduced model)} u {anthro.fixed_joint_rotation(4, frozen)}); "
    "pelvis global = large.joint_rotation[:,0]; distal global orientation matches raw SMPL exactly."
)


def reconstruction_block(fit: SubjectFit, lossless_alternative: str) -> dict[str, Any]:
    """The manifest's ``anthro_reconstruction`` block, with the fit record inside it."""
    return {
        "recipe": RECONSTRUCTION_RECIPE,
        "fixed_joints": list(FROZEN_JOINT_NAMES),
        "method": f"{METHOD} ({fit.record['scope']} scope): spine3<-spine1,spine2 ; "
                  "L/R_shoulder<-L/R_collar",
        "lossless_alternative": lossless_alternative,
        "reduced_model_fit": fit.record,
    }


def fit_from_record(record: Mapping[str, Any]) -> SubjectFit:
    """Rebuild a fit from its record: the constants are ``constants_wxyz`` at full precision."""
    quaternions = np.asarray(record["constants_wxyz"], dtype=np.float64)
    if quaternions.shape != (len(FROZEN_JOINTS), 4):
        raise ReductionError(f"{record.get('group')}: constants_wxyz must be (4, 4)")
    return SubjectFit(constants=quaternion_to_matrix(quaternions), record=dict(record))


#: The bundle-root record of every fit a bundle was reduced with (``docs/primer.md`` 5.3 of the
#: smpl18 package): one entry per fit group, and which group each take used. Every bundle carries
#: it (``configs/datasets/dataset_profiles_v1.yaml`` universal.bundle_files).
FIT_FILE = "reduced_model_fit.json"
FIT_FILE_SCHEMA = "reduced_model_fit"


def write_fit_file(
    bundle_dir: Path | str,
    settings: ReductionSettings,
    groups: Mapping[str, Mapping[str, Any]],
    takes: Mapping[str, str],
    unfitted: Mapping[str, str] | None = None,
) -> Path:
    """Write ``FIT_FILE``: the settings, each group's record, and each take's group.

    ``takes`` maps a take directory name (INDEX ``rel``) to the group whose constants it carries;
    ``unfitted`` maps a take that could not be pooled to the reason, so its absence from a group is
    stated rather than inferred. Written through a temporary file and a rename, because sharded
    runs read it concurrently.
    """
    unknown = sorted(set(takes.values()) - set(groups))
    if unknown:
        raise ReductionError(f"takes name fit groups that were not fitted: {unknown[:5]}")
    document = {
        "schema": FIT_FILE_SCHEMA,
        "method": METHOD,
        "source": settings.source_name,
        "settings": dict(settings.provenance),
        "group_count": len(groups),
        "take_count": len(takes),
        "groups": {name: dict(groups[name]) for name in sorted(groups)},
        "takes": {name: takes[name] for name in sorted(takes)},
        "unfitted": {name: (unfitted or {})[name] for name in sorted(unfitted or {})},
    }
    target = Path(bundle_dir) / FIT_FILE
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, target)
    return target


@dataclass
class CorpusFits:
    """A bundle's fits as a generator builds them: fit a subject, look a take up, write the record.

    Take keys are take directory names (INDEX ``rel``), so ``FIT_FILE`` names the same takes the
    INDEX does.
    """

    settings: ReductionSettings
    by_take: dict[str, SubjectFit] = field(default_factory=dict)
    groups: dict[str, dict[str, Any]] = field(default_factory=dict)
    takes: dict[str, str] = field(default_factory=dict)

    def fit(self, name: str, members: Sequence[tuple[str, np.ndarray, np.ndarray]],
            basis: str) -> None:
        """Fit one subject's takes, split by skeleton, and remember which fit each take uses."""
        for group, (keys, poses, rest) in group_by_skeleton(name, members).items():
            fitted = fit_subject(poses, rest, self.settings, group=group, basis=basis)
            self.groups[group] = fitted.record
            for key in keys:
                self.by_take[key] = fitted
                self.takes[key] = group

    def add_records(self, groups: Mapping[str, Mapping[str, Any]],
                    takes: Mapping[str, str]) -> None:
        """Take in records fitted elsewhere (a worker process), as ``groups`` and ``takes``."""
        for name, record in groups.items():
            self.groups[name] = dict(record)
        for key, group in takes.items():
            self.takes[key] = group
            self.by_take[key] = fit_from_record(self.groups[group])

    def write(self, bundle_dir: Path | str) -> Path:
        return write_fit_file(bundle_dir, self.settings, self.groups, self.takes)


def read_fit_file(bundle_dir: Path | str) -> dict[str, Any] | None:
    """The bundle's ``FIT_FILE``, or None when it has none yet."""
    path = Path(bundle_dir) / FIT_FILE
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != FIT_FILE_SCHEMA:
        raise ReductionError(f"{path}: schema {document.get('schema')!r}, not {FIT_FILE_SCHEMA!r}")
    return document
