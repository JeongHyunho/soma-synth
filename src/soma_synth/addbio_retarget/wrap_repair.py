"""Undo the 2*pi staircase the source filtered through.

AddBiomechanics' `lowPassFilter` pass filters every generalised coordinate as a plain scalar.
A coordinate that wraps carries a one-frame 2*pi step, and a low-pass filter run over that step
does not preserve it: it sweeps the value through a whole turn across several frames and
overshoots past |pi|, which a wrapped angle can never reach. `dynamics` is solved from that
result, so the pass the retarget reads carries the defect -- on the corrupted subject measured
here it reproduced `lowPassFilter` byte for byte in all 45 trials, though that is not general:
on subjects with no wrap the two passes differ by up to 0.29 rad.

Nothing here is keyed on a coordinate name: wraps occur in several coordinates and studies -- the
pelvis heading, and also `arm_rot_r` in seven `vanderZee2022` trials of a sixteen-subject
sample -- so the repair applies to any column whose unwrap counter moves.

A repaired column loses whatever the dynamics pass had done to it. All seven of those
`arm_rot_r` trials carry a real refinement there, up to 0.306 rad. It is not added back: that
refinement was solved from the corrupted trajectory, so restoring it would carry the corruption
into the repair by another door.

The repair filters what the source should have filtered: the *unwrapped* angle. Coordinates with
no staircase are left exactly as the source wrote them, so a trial the defect never touched comes
back bit-identical rather than merely close.

Filter constants are read from each trial's own pass header rather than chosen here. `ADR-0034`
measured `order=2` and `cutoff=30.0 Hz` across all 40,537 `lowPassFilter` passes with no
exception; reading them per trial means a file that departs from that is followed rather than
overruled.

Two forms of the repair were measured against each other before this one was written. Filtering
the unwrapped angle directly is what remains. The alternative exploits the filter's linearity --
`filtered(unwrapped) = stored - 2*pi*filtered(k)` -- and preserves the source's own filtered
signal exactly away from a wrap, which is why it looked better; but its error term carries a 2*pi
factor per step, so wherever our copy of the filter differs slightly from theirs the mistake is
amplified by the staircase. Across the 45 trials of one corrupted subject it left six of them
above 11,900 deg/s, still far outside the physics band, while the direct form held every trial
between 145 and 803 deg/s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt

__all__ = [
    "FilterSpec",
    "RepairedPositions",
    "WrapRepairError",
    "filter_spec_for",
    "repair_positions",
    "wrap_counter",
]

_LOW_PASS_FILTER = "lowPassFilter"
_KINEMATICS = "kinematics"

# scipy pads with 3 * max(len(a), len(b)) samples; below that filtfilt cannot run at all
_MIN_FRAMES = 3 * 3 + 1


class WrapRepairError(RuntimeError):
    """The repair cannot be carried out on this trial, and will not be approximated."""


@dataclass(frozen=True)
class FilterSpec:
    """What one trial's own `lowPassFilter` header declares."""

    order: int
    cutoff_hz: float
    rate_hz: float

    @property
    def normalised_cutoff(self) -> float:
        return self.cutoff_hz / (self.rate_hz / 2.0)

    def design(self) -> tuple[np.ndarray, np.ndarray]:
        wn = self.normalised_cutoff
        if not 0.0 < wn < 1.0:
            raise WrapRepairError(
                f"cutoff {self.cutoff_hz} Hz is not below Nyquist for a {self.rate_hz} Hz "
                f"trial (normalised {wn:.4f}); refusing to invent a filter"
            )
        if self.order < 1:
            raise WrapRepairError(f"filter order {self.order} is not a filter")
        return butter(self.order, wn, btype="low")


@dataclass(frozen=True)
class RepairedPositions:
    """Coordinates with every 2*pi staircase filtered as an angle instead of a scalar."""

    pos: np.ndarray                      # (T, num_coordinates)
    repaired_indices: tuple[int, ...]    # columns this changed, in column order
    wrap_events: tuple[int, ...]         # staircase transitions found, one per repaired column

    @property
    def repaired(self) -> bool:
        return bool(self.repaired_indices)


def filter_spec_for(b3d, trial_index: int) -> FilterSpec:
    """Read the filter this trial says was applied to it."""
    trial = b3d.trials[trial_index]
    header = b3d.header.trial_header[trial_index]
    declared = None
    for entry, name in zip(header.processing_pass_header, trial.pass_names):
        if name == _LOW_PASS_FILTER:
            declared = entry
            break
    if declared is None:
        raise WrapRepairError(
            f"trial {trial_index} has no {_LOW_PASS_FILTER} pass: {trial.pass_names}"
        )
    return FilterSpec(
        order=int(declared.lowpass_filter_order),
        cutoff_hz=float(declared.lowpass_cutoff_frequency),
        rate_hz=float(trial.source_rate_hz),
    )


def wrap_counter(angles: np.ndarray) -> np.ndarray:
    """The integer staircase separating a wrapped angle from its continuous form.

    `angles == np.unwrap(angles) + 2*pi*wrap_counter(angles)` holds exactly, so a column whose
    counter never moves carried no wrap and needs nothing done to it.
    """
    angles = np.asarray(angles, dtype=np.float64)
    return np.rint((angles - np.unwrap(angles)) / (2.0 * np.pi))


def repair_positions(
    kinematics_pos: np.ndarray,
    filtered_pos: np.ndarray,
    spec: FilterSpec,
) -> RepairedPositions:
    """Rebuild the coordinates the source filtered across a wrap; leave the rest untouched.

    `kinematics_pos` is the unfiltered pass, which still holds the one-frame wrap that says
    where the branch cut was crossed. `filtered_pos` is the pass the retarget consumes -- the
    `lowPassFilter` values, which `dynamics` carries unchanged.

    Both must cover the same contiguous frames of the same trial: `np.unwrap` reads a step
    between neighbours and `filtfilt` reads the series as a whole, so a thinned or reordered
    subset would be answering about a signal that was never sampled. Thin afterwards.
    """
    kinematics_pos = np.asarray(kinematics_pos, dtype=np.float64)
    filtered_pos = np.asarray(filtered_pos, dtype=np.float64)
    if kinematics_pos.shape != filtered_pos.shape:
        raise WrapRepairError(
            f"passes disagree on shape: kinematics {kinematics_pos.shape}, "
            f"filtered {filtered_pos.shape}"
        )
    if kinematics_pos.ndim != 2:
        raise WrapRepairError(f"expected (frames, coordinates); got {kinematics_pos.shape}")

    out = filtered_pos.copy()
    repaired: list[int] = []
    events: list[int] = []
    numerator, denominator = None, None

    for column in range(kinematics_pos.shape[1]):
        wrapped = kinematics_pos[:, column]
        counter = wrap_counter(wrapped)
        transitions = int(np.count_nonzero(np.diff(counter)))
        if transitions == 0:
            continue                      # the source filtered a signal with no step in it
        if kinematics_pos.shape[0] < _MIN_FRAMES:
            raise WrapRepairError(
                f"coordinate {column} wraps but the trial holds "
                f"{kinematics_pos.shape[0]} frames, fewer than the {_MIN_FRAMES} the filter "
                "needs; the trial cannot be repaired and must not be passed off as clean"
            )
        if numerator is None:
            numerator, denominator = spec.design()
        out[:, column] = filtfilt(numerator, denominator, np.unwrap(wrapped))
        repaired.append(column)
        events.append(transitions)

    return RepairedPositions(
        pos=out,
        repaired_indices=tuple(repaired),
        wrap_events=tuple(events),
    )


def read_repaired_positions(
    b3d,
    trial_index: int,
    read_pass_frames,
    *,
    pass_index: int | None = None,
) -> RepairedPositions:
    """Read the whole trial and return the coordinates of `pass_index`, repaired.

    `pass_index` defaults to the trial's dynamics pass, which is what the retarget consumes.
    The base matters: `dynamics` is not a copy of `lowPassFilter`. It re-solves the pose and
    across the subjects measured here it differs by up to 0.29 rad -- except on the corrupted
    subject, where it reproduced `lowPassFilter` exactly in all 45 trials. So the untouched
    columns are handed back from the pass the caller asked for, and only a column carrying a
    staircase is rebuilt; a repaired column loses whatever the dynamics pass had done to it,
    which is the price of removing a 2*pi artefact from it.

    `read_pass_frames` is passed in rather than imported so this module stays free of the
    reader it repairs, and so a caller that already holds frames can use `repair_positions`
    directly.
    """
    trial = b3d.trials[trial_index]
    try:
        kinematics_index = trial.pass_names.index(_KINEMATICS)
    except ValueError as exc:
        raise WrapRepairError(
            f"trial {trial_index} has no {_KINEMATICS} pass: {trial.pass_names}"
        ) from exc
    if pass_index is None:
        pass_index = trial.dynamics_pass_index

    frames = range(trial.length)
    kinematics = read_pass_frames(b3d, trial_index, frames, pass_index=kinematics_index)
    consumed = read_pass_frames(b3d, trial_index, frames, pass_index=pass_index)
    return repair_positions(kinematics.pos, consumed.pos, filter_spec_for(b3d, trial_index))
