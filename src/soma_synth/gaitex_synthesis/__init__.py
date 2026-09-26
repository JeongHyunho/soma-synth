"""GAITEX's marker-driven small: eight virtual IMU sites synthesised from optical marker clusters.

The synthesis machinery itself lives in ``sensors/`` and ``kinematics/`` and is not copied here.
What this package adds is the part no earlier generator needed: the eight-site assembly, the spec sensor
frame built from landmarks, the placement policy for sites whose offset the registry leaves
unresolved, and the pair window that keeps small on large's frames with NaN under a False mask
(ADR-0039).
"""

from .marker_small import (
    MarkerSmallError,
    PairWindow,
    SiteSynthesis,
    SynthesisSettings,
    compare_with_worn_units,
    load_settings,
    pair_window,
    relabel_dispersion_vs_large,
    stack_small,
    synthesise_site,
    synthesise_sites,
)

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
