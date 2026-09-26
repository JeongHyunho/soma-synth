"""Assemble one converted trial into the file a downstream reader will use.

The artifact has to be legible on its own, because whoever reads it will not read this code.
Three things carry that weight:

  provenance   every joint says whether it was measured, derived from the lumbar, or absent.
               `absent` is a marker; without the label an identity rotation reads as a
               measurement of a body part that happened to be still
  frame        the rotation from the source's Y-up world is written down, so the transform is
               reversible rather than a fact about how the file was made
  attribution  ten fields the licence requires. A file missing them cannot be published even
               internally, so building one without them fails here rather than later

Nothing is stored as a Python object: a reader that has to pass `allow_pickle=True` is a reader
that has to trust the file, and this file travels well beyond this code.
"""

from __future__ import annotations

import pathlib

import numpy as np

from .attribution import ADDBIOMECHANICS, attribution_fields, resolve_source

__all__ = [
    "ARTIFACT_CLASS",
    "LUMBAR_DISTRIBUTION",
    "QUALITY_GATE",
    "SPEC_ID",
    "SPEC_VERSION",
    "build_artifact",
    "write_artifact",
]

#: Deliberately not `contract_version` or `canonical`, which name the governed pipeline's
#: vocabulary; this layer moves no gate and must not borrow words that imply it did.
#: The AddBiomechanics corpus's id, kept here for callers that name that corpus specifically.
#: An artifact takes its `spec_id` from its own archive instead -- see `build_artifact` -- so
#: this must not drift away from the archive it stands for.
SPEC_ID = ADDBIOMECHANICS.spec_id
#: `retarget-v2` writes coordinates the source's low-pass pass had filtered across a 2*pi wrap
#: as angles rather than scalars (see `wrap_repair`). A trial with no wrap is byte-identical to
#: `retarget-v1`; a trial with one is not, so the lineage has to say which it is.
SPEC_VERSION = "retarget-v2"

ARTIFACT_CLASS = "experimental_non_candidate"
QUALITY_GATE = "NOT_EVALUATED"
DISTRIBUTION_SCOPE = "internal_only"

#: The lumbar split rule, named so downstream can tell a rule from a measurement.
LUMBAR_DISTRIBUTION = "equal_thirds_v1"

_JOINTS = 24


def _text(value) -> np.ndarray:
    return np.array(str(value), dtype=np.str_)


def build_artifact(
    *,
    source_folder: str,
    subject: str,
    variant: str,
    pass_used: str,
    trial_name: str,
    betas,
    gender: str,
    pose,
    trans,
    joint_provenance,
    source_coordinates,
    rest_alignment,
    root_offset,
    frame_rotation_source_to_world,
    timestamps_s,
    source_rate_hz: float,
    position_error_m: float,
    adapter_hash: str,
    config_hash: str,
    code_hash: str,
    model_hash: str,
    wrap_repaired_coordinates=(),
) -> dict[str, np.ndarray]:
    """Build the key/array mapping for one trial. Raises rather than emit something partial."""
    pose = np.asarray(pose, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)

    if pose.ndim != 3 or pose.shape[1] != _JOINTS or pose.shape[2] != 3:
        raise ValueError(f"pose must be (frames, {_JOINTS}, 3); got {pose.shape}")
    frames = pose.shape[0]
    if trans.shape != (frames, 3):
        raise ValueError(f"trans has {trans.shape[0]} frames, pose has {frames}")
    if timestamps.shape != (frames,):
        raise ValueError(f"timestamps_s has {timestamps.shape[0]} frames, pose has {frames}")
    if len(joint_provenance) != _JOINTS:
        raise ValueError(f"joint_provenance must cover all {_JOINTS} joints")
    if len(source_coordinates) != _JOINTS:
        raise ValueError(f"source_coordinates must cover all {_JOINTS} joints")

    # raises if the study cannot be attributed, before anything is written
    attribution = attribution_fields(
        source_folder,
        adapter_hash=adapter_hash,
        config_hash=config_hash,
        code_hash=code_hash,
        model_hash=model_hash,
    )

    built: dict[str, np.ndarray] = {
        # From the archive, not from this module: `spec_id` says which corpus a file belongs to,
        # and two archives passing through one retarget do not form one corpus.
        "spec_id": _text(resolve_source(source_folder).archive.spec_id),
        "spec_version": _text(SPEC_VERSION),
        "artifact_class": _text(ARTIFACT_CLASS),
        "quality_gate": _text(QUALITY_GATE),
        "distribution_scope": _text(DISTRIBUTION_SCOPE),

        "subject": _text(subject),
        "trial_name": _text(trial_name),
        "variant": _text(variant),
        "pass_used": _text(pass_used),

        "betas": np.asarray(betas, dtype=np.float32),
        "gender": _text(gender),
        "pose": pose.astype(np.float32),
        "trans": trans.astype(np.float32),

        "joint_provenance": np.asarray(
            [str(p) for p in joint_provenance], dtype=np.str_
        ),
        # ragged by nature -- a joint consumes one, two or no coordinates -- so each is joined
        # into one string rather than padded, which would invent empty coordinate names
        "source_dof_names": np.asarray(
            [",".join(names) for names in source_coordinates], dtype=np.str_
        ),
        # empty when the source's filter never met a wrap in this trial, which is the usual
        # case; a reader must be able to tell "nothing needed repairing" from "not checked",
        # so this is written on every trial rather than only on the repaired ones
        "wrap_repaired_coordinates": _text(",".join(str(c) for c in wrap_repaired_coordinates)),
        "rest_alignment": np.asarray(rest_alignment, dtype=np.float32),
        "root_offset": np.asarray(root_offset, dtype=np.float32),
        "lumbar_distribution": _text(LUMBAR_DISTRIBUTION),
        "frame_rotation_source_to_world": np.asarray(
            frame_rotation_source_to_world, dtype=np.float32
        ),

        # float64: a 0.005 s step accumulated over a long trial loses frames at float32
        "timestamps_s": timestamps,
        "source_rate_hz": np.asarray(source_rate_hz, dtype=np.float64),
        "joint_centre_error_m": np.asarray(position_error_m, dtype=np.float64),
    }
    built.update({key: _text(value) for key, value in attribution.items()})
    return built


def write_artifact(
    path: str | pathlib.Path, built: dict[str, np.ndarray], *, compress: bool = False
) -> pathlib.Path:
    """Write the artifact, readable without `allow_pickle` either way.

    Compression is worth having at cohort scale -- axis-angle rotations deflate to about 39%,
    turning 23 GB into 9 GB -- and costs the reader nothing, since `np.load` handles both.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (np.savez_compressed if compress else np.savez)(path, **built)
    return path
