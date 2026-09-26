"""Say a joint is measured only where the source says something measured it.

``correspondence_for`` intersects a written table with the skeleton a model file declares, and
for AddBiomechanics that is the right question: its inverse kinematics is driven by markers over
the whole body, so a joint the model declares is a joint the solve had evidence for.

GAITEX is solved from nine inertial units, and its skeleton is complete regardless. The model
declares ``acromial_r``, so the table calls the right shoulder measured; but nothing above the
sternum wears a sensor, and the shoulder coordinates in the published ``.mot`` are what the
solver relaxed to, not what it observed. ``arm_add_r`` decays from ``-70.9`` degrees at the first
frame to ``-0.004`` at the last, and ``neck_extension`` and every wrist coordinate are identically
zero for the whole trial. Shipping those as ``measured`` would put a falsehood in every artifact.

The source states the answer itself. ``metadata_<subject>_<condition>.json`` carries an
``inverse_kinematics`` block naming each driven body with its markers and its sensor, so the
intersection is taken against that rather than against the skeleton. Nothing is hardcoded and
nothing is special-cased: a source that later instruments the arms degrades back the same way,
and the left toe falls out on its own because GAITEX put a sensor on the right one only.
"""

from __future__ import annotations

from dataclasses import replace

from ..addbio_retarget.smpl_correspondence import ABSENT, SmplJointSource

__all__ = ["restrict_to_driven", "undriven_summary"]


def restrict_to_driven(
    table: tuple[SmplJointSource, ...], driven_bodies
) -> tuple[SmplJointSource, ...]:
    """Downgrade every joint whose driving body the source did not instrument.

    A downgraded entry is made identical in shape to one the skeleton never declared: no centre
    to fit to and no source coordinates. That matters beyond labelling -- the retarget leaves an
    absent joint at its rest rotation, so the relaxation artifact is not merely relabelled, it is
    replaced by a pose that claims nothing.
    """
    driven = frozenset(driven_bodies)
    return tuple(
        entry
        if entry.source_body in driven
        else replace(
            entry,
            provenance=ABSENT,
            centre_joint=None,
            source_coordinates=(),
            source_body=None,
        )
        for entry in table
    )


def undriven_summary(
    table: tuple[SmplJointSource, ...], restricted: tuple[SmplJointSource, ...]
) -> dict[str, str]:
    """Which joints the skeleton offered and the sensors did not, keyed by SMPL joint name.

    Written into the manifest so a reader can see what was withheld and why, rather than
    inferring it from an absence.
    """
    return {
        before.smpl_name: (
            f"the model declares this joint through {before.source_body}, but "
            f"{before.source_body} carries no sensor in this trial, so its rotation is the "
            "solver's relaxation rather than a measurement and is not carried"
        )
        for before, after in zip(table, restricted)
        if before.provenance != ABSENT and after.provenance == ABSENT
    }
