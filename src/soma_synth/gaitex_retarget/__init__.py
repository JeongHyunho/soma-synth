"""GAITEX's side of the OpenSim-to-SMPL retarget.

The retarget itself is source-agnostic once a source can hand it coordinates, a model and a
world frame; ``addbio_retarget`` already holds that machinery and none of it is copied here.
What GAITEX needs is a reader, because its frames live in OpenSim ``.mot`` and ``.trc`` files
rather than in a ``.b3d`` container, and because one of the coordinates it publishes is not
usable as published.
"""

from .gaitex_frames import (
    GaitexFile,
    GaitexFrames,
    GaitexReaderError,
    GaitexSubject,
    GaitexTrial,
    MissingPelvisStations,
    discover_subjects,
    open_trial,
    read_frames,
)
from .joint_ranges import JointRangeLimits, JointRangeReport, load_limits, measure
from .provenance import restrict_to_driven, undriven_summary
from .shape import PooledTargets, knee_bracketing_pairs, pool_segment_lengths

__all__ = [
    "GaitexFile",
    "GaitexFrames",
    "GaitexReaderError",
    "GaitexSubject",
    "GaitexTrial",
    "JointRangeLimits",
    "JointRangeReport",
    "MissingPelvisStations",
    "PooledTargets",
    "discover_subjects",
    "knee_bracketing_pairs",
    "load_limits",
    "measure",
    "open_trial",
    "pool_segment_lengths",
    "read_frames",
    "restrict_to_driven",
    "undriven_summary",
]
