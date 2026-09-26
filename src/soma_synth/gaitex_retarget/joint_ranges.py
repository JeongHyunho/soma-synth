"""How much of a trial the source's own inverse kinematics failed to solve.

GAITEX publishes trials whose joint angles no body can produce -- one sweeps a knee through a
full revolution. Nothing downstream notices: the retarget is happy to place a limb anywhere, and
the unified8 validator's L2 band bounds gyroscope magnitude, which a joint turning smoothly
through 360 degrees passes comfortably.

The bounds live in ``configs/datasets/gaitex_joint_ranges_v1.yaml`` because they are thresholds,
and threshold values are never fixed as code constants. This module holds no
default: with no config there is no gate, and the caller is told so rather than silently getting
one that was invented here.

Frames outside the bounds are counted, never dropped. Dropping them would change what a take
contains without the manifest saying so, and would put a gap back into a time base that
``longest_solved_span`` exists to keep uniform.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib

import numpy as np

__all__ = [
    "JointRangeLimits",
    "JointRangeReport",
    "load_limits",
    "measure",
]


@dataclass(frozen=True)
class JointRangeLimits:
    """Bounds in degrees, keyed by OpenSim coordinate name, and where they came from."""

    config_id: str
    version: str
    limits_deg: dict[str, tuple[float, float]]
    span_limits_deg: dict[str, float]


@dataclass(frozen=True)
class JointRangeReport:
    """What a trial did against those bounds.

    ``worst`` names the coordinate with the largest share of frames outside, or is ``None`` when
    nothing left range. ``per_coordinate`` carries every coordinate that was checked, including
    the clean ones, so a reader can tell "checked and fine" from "never looked at".
    """

    frames_checked: int
    coordinates_checked: tuple[str, ...]
    coordinates_absent: tuple[str, ...]
    per_coordinate: dict[str, dict[str, float]]
    frames_outside: int
    worst: str | None
    failed_solve: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return self.frames_outside == 0

    @property
    def solve_came_apart(self) -> bool:
        """A span no joint can traverse. This, and not the level, means broken data."""
        return bool(self.failed_solve)

    def as_manifest_block(self) -> dict:
        """The shape written onto an artifact."""
        return {
            "config_id": "gaitex_joint_ranges_v1",
            "frames_checked": int(self.frames_checked),
            "frames_outside_any": int(self.frames_outside),
            "fraction_outside_any": (
                float(self.frames_outside) / float(self.frames_checked)
                if self.frames_checked else 0.0
            ),
            "worst_coordinate": self.worst,
            "solve_came_apart": bool(self.failed_solve),
            "coordinates_whose_span_is_impossible": list(self.failed_solve),
            "coordinates_checked": list(self.coordinates_checked),
            "coordinates_absent_from_model": list(self.coordinates_absent),
            "per_coordinate": self.per_coordinate,
            "policy": "measured and recorded; no frame was dropped and no take was refused",
            "how_to_read_this": (
                "`solve_came_apart` is the one to filter on: it means a joint traversed more "
                "than its own range of motion, which is broken data. `frames_outside_any` also "
                "counts frames whose value merely sits outside the bound while the span stays "
                "ordinary -- a posture, a task, or a difference in where the model puts zero. "
                "Those are not the same thing and should not be filtered alike."
            ),
        }


def load_limits(path: pathlib.Path) -> JointRangeLimits:
    """Read the declared bounds. Raises rather than defaulting if the file is not there."""
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    limits = {
        name: (float(bound["min"]), float(bound["max"]))
        for name, bound in config["limits_deg"].items()
    }
    if not limits:
        raise ValueError(f"{path.name} declares no limits; there is nothing to check against")
    spans = {
        name: float(value) for name, value in config.get("span_limits_deg", {}).items()
    }
    return JointRangeLimits(
        config_id=str(config["config_id"]),
        version=str(config["version"]),
        limits_deg=limits,
        span_limits_deg=spans,
    )


def measure(
    values_rad: np.ndarray,
    coordinate_names: list[str] | tuple[str, ...],
    limits: JointRangeLimits,
    *,
    frame_mask: np.ndarray | None = None,
) -> JointRangeReport:
    """Count frames whose joint angles leave the declared bounds.

    ``values_rad`` is the coordinate array in radians, as the reader returns it. ``frame_mask``
    restricts the measurement to the frames a take will actually carry, so the number on the
    artifact describes the take rather than the trial it was cut from.
    """
    names = list(coordinate_names)
    if values_rad.ndim != 2 or values_rad.shape[1] != len(names):
        raise ValueError(
            f"values_rad {values_rad.shape} does not match {len(names)} coordinate names"
        )
    selected = (
        np.ones(values_rad.shape[0], dtype=bool) if frame_mask is None
        else np.asarray(frame_mask, dtype=bool)
    )
    frames = int(selected.sum())

    checked: list[str] = []
    absent: list[str] = []
    per_coordinate: dict[str, dict[str, float]] = {}
    outside_any = np.zeros(frames, dtype=bool)
    failed: list[str] = []
    worst, worst_share = None, 0.0

    for name, (low, high) in sorted(limits.limits_deg.items()):
        if name not in names:
            absent.append(name)
            continue
        checked.append(name)
        degrees = np.rad2deg(values_rad[selected, names.index(name)])
        outside = (degrees < low) | (degrees > high)
        outside_any |= outside
        share = float(outside.mean()) if frames else 0.0
        span = float(np.ptp(degrees)) if frames else 0.0
        span_limit = limits.span_limits_deg.get(name)
        # The level can sit outside for a whole trial and still be a posture. The span cannot:
        # a joint that traverses more than its own range of motion did not move, it broke.
        came_apart = span_limit is not None and span > span_limit
        if came_apart:
            failed.append(name)
        per_coordinate[name] = {
            "min_deg": float(degrees.min()) if frames else 0.0,
            "max_deg": float(degrees.max()) if frames else 0.0,
            "span_deg": span,
            "limit_min_deg": low,
            "limit_max_deg": high,
            "span_limit_deg": span_limit,
            "frames_outside": int(outside.sum()),
            "fraction_outside": share,
            "span_exceeds_limit": bool(came_apart),
        }
        if share > worst_share:
            worst, worst_share = name, share

    return JointRangeReport(
        frames_checked=frames,
        coordinates_checked=tuple(checked),
        coordinates_absent=tuple(absent),
        per_coordinate=per_coordinate,
        frames_outside=int(outside_any.sum()),
        worst=worst,
        failed_solve=tuple(sorted(failed)),
    )
