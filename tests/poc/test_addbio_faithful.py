"""The two claims the AddBio synthetic corpus makes about itself.

The channels themselves are checked by reading a built corpus back (a diagnostic of the parent
project, whose own tests stay there with it). What is worth
pinning here is the labelling: which of the eight sites the source drives. It was wrong once.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parents[2]
GENERATOR = REPO / "scripts" / "poc" / "generate_addbio_faithful.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load(GENERATOR, "addbio_faithful")


def _provenance(**overrides) -> np.ndarray:
    """A No_Arm-shaped provenance vector, overridable per joint."""
    values = ["measured"] * 24
    for joint in (3, 6, 9):
        values[joint] = "derived_lumbar"
    for joint in (12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23):
        values[joint] = "absent"
    for joint, value in overrides.items():
        values[int(joint)] = value
    return np.array(values, dtype=np.str_)


def test_the_labels_come_from_the_driving_joint(generator):
    sources = generator.site_sources(_provenance())
    orientation = dict(zip(generator.FAITHFUL.SENSOR_CODES, sources["site_orientation_source"]))
    assert orientation["shank_l"] == "measured"          # joint 4
    assert orientation["foot_r"] == "measured"           # joint 11
    assert orientation["back_T4"] == "derived_lumbar"    # joint 9, the lumbar split
    assert orientation["occiput"] == "absent"            # joint 15, no head DOF anywhere
    assert orientation["wrist_l"] == "absent"            # joint 18, No_Arm


def test_the_arm_variants_differ_and_the_labels_follow(generator):
    no_arm = generator.site_sources(_provenance())
    with_arm = generator.site_sources(
        _provenance(**{"16": "measured", "17": "measured", "18": "measured",
                       "19": "measured", "20": "measured", "21": "measured"})
    )
    codes = generator.FAITHFUL.SENSOR_CODES
    backed_no_arm = dict(zip(codes, no_arm["site_source_backed"]))
    backed_with_arm = dict(zip(codes, with_arm["site_source_backed"]))
    assert backed_no_arm["wrist_l"] is np.False_
    assert backed_with_arm["wrist_l"] is np.True_
    # occiput does not improve with arms: no model in the corpus has a head coordinate
    assert backed_no_arm["occiput"] is np.False_
    assert backed_with_arm["occiput"] is np.False_


def test_the_orientation_joint_decides_not_the_position_joint(generator):
    """A site whose driver is measured stays backed even where its position joint is absent.

    The rule is deliberate and worth pinning: the driving joint sets the gyro outright and decides
    how gravity lands in the accelerometer, so it is the sharper of the two tests. Writing it the
    other way round would mark a real channel as fabricated.
    """
    # wrist_l is driven by joint 18 and positioned by joint 20
    sources = generator.site_sources(_provenance(**{"18": "measured", "20": "absent"}))
    at = generator.FAITHFUL.SENSOR_CODES.index("wrist_l")
    assert sources["site_orientation_source"][at] == "measured"
    assert sources["site_position_source"][at] == "absent"
    assert bool(sources["site_source_backed"][at]) is True


def test_back_T4_position_is_named_as_the_proxy_it_is(generator):
    sources = generator.site_sources(_provenance())
    at = generator.FAITHFUL.SENSOR_CODES.index("back_T4")
    assert sources["site_position_source"][at] == generator.CHEST_PROXY
