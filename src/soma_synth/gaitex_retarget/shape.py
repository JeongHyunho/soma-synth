"""Fit one body per subject from four disagreeing skeletons, and say how much they disagreed.

GAITEX scales its OpenSim model per TRIAL, so a subject arrives with one skeleton per condition
and they do not agree. Measured across the cohort, a subject's own right femur varies by ``11.4``
mm at the median and ``50.3`` mm at worst -- about ``2.7`` percent of its length.

The disagreement is not a property of the condition. Ranking each subject's conditions by femur
length and again by tibia length puts them in different orders: ``rd`` gives the shortest femur
and the longest tibia. What it is, is the knee centre moving. The femur and tibia deviations from
a subject's own median are strongly anticorrelated -- ``r = -0.766`` on the right and ``-0.642``
on the left over 72 trial deviations -- so the two segments trade millimetres while the limb
keeps its length. The composite confirms it: hip to ankle varies by ``1.42`` percent on the right
where its two parts vary by ``2.68`` and ``2.74``.

Two consequences, and this module implements both. A constant measured with noise is estimated by
pooling, not by choosing a favourite trial, so the targets are the median across a subject's
trials. And a fit anchored on the stable composite is better conditioned than one that trusts the
wandering knee, so the hip-to-ankle span is offered as a target beside its two parts -- which
``segment_pairs_from`` already supports, since it emits non-adjacent torso paths by the same
mechanism.

What pooling cannot do is make the disagreement go away, so it is measured and carried rather
than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PooledTargets", "knee_bracketing_pairs", "pool_segment_lengths"]

# SMPL-24 indices. The bracket spans the knee, which is the joint whose centre wanders.
_HIP_L, _HIP_R = 1, 2
_ANKLE_L, _ANKLE_R = 7, 8
_KNEE_L, _KNEE_R = 4, 5


@dataclass(frozen=True)
class PooledTargets:
    """Segment lengths for a subject, and the spread the pooling absorbed.

    ``spread_m`` is the full range across that subject's trials, per segment. It is the
    anthropometry's own uncertainty and belongs on the artifact: a consumer that sees a femur of
    ``0.424`` m should be able to see that the source offered values ``11`` mm apart for it.
    """

    lengths_m: dict[tuple[int, int], float]
    spread_m: dict[tuple[int, int], float]
    trials_pooled: int


def knee_bracketing_pairs(
    pairs: dict[tuple[int, int], tuple[str, str]],
) -> dict[tuple[int, int], tuple[str, str]]:
    """Add hip-to-ankle to the fit targets, on each side that has both ends.

    The thigh and shank pairs are kept rather than replaced. Dropping them would leave the
    knee's position along the limb unconstrained; keeping both lets the stable span set the
    limb's length while the noisy pair still places the joint inside it.
    """
    extended = dict(pairs)
    for hip, knee, ankle in ((_HIP_L, _KNEE_L, _ANKLE_L), (_HIP_R, _KNEE_R, _ANKLE_R)):
        upper, lower = pairs.get((hip, knee)), pairs.get((knee, ankle))
        if upper and lower:
            extended[(hip, ankle)] = (upper[0], lower[1])
    return extended


def pool_segment_lengths(per_trial: list[dict[tuple[int, int], float]]) -> PooledTargets:
    """Median length per segment across a subject's trials, with the range they spanned.

    A segment absent from some trials is pooled over the trials that have it, because a
    condition that lost a marker should not remove the segment from the subject.
    """
    if not per_trial:
        raise ValueError("no trials to pool")
    observed: dict[tuple[int, int], list[float]] = {}
    for lengths in per_trial:
        for segment, value in lengths.items():
            if np.isfinite(value):
                observed.setdefault(segment, []).append(float(value))

    return PooledTargets(
        lengths_m={
            segment: float(np.median(values)) for segment, values in observed.items()
        },
        spread_m={
            segment: float(max(values) - min(values)) for segment, values in observed.items()
        },
        trials_pooled=len(per_trial),
    )
