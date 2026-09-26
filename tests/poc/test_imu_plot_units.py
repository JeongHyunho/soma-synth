"""The IMU plot must name the unit the generators actually wrote.

The gyro axis read `rad/s` while both faithful builders fill the channel from
`angular_velocity_deg_s()`. Nothing caught it, because a wrong unit produces a perfectly normal
looking figure — the numbers are simply 57.3x the physical quantity, and a reader has no way to tell
from the plot alone. These tests tie the label to the writers rather than restating it.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

POC = Path(__file__).parents[2] / "scripts" / "poc"
GENERATORS = ("generate_prism_faithful.py", "generate_amass_faithful.py")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(Path(name).stem, POC / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assignments_to(source: str, target: str) -> set[str]:
    """Every plain function name called anywhere on the right of `<target>[...] = ...`.

    The whole right-hand expression is walked rather than just its outermost call: the PRISM builder
    writes `win(angular_velocity_deg_s(...)).astype(...)`, so the producer that decides the unit sits
    two calls in.
    """
    tree = ast.parse(source)
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for lhs in node.targets:
            base = lhs
            while isinstance(base, ast.Subscript):
                base = base.value
            if isinstance(base, ast.Name) and base.id == target:
                for sub in ast.walk(node.value):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                        out.add(sub.func.id)
    return out


def test_plot_labels_the_gyro_axis_in_the_unit_the_generators_write() -> None:
    imu_plot = _load("imu_plot.py")
    assert imu_plot.GYRO_UNIT == "deg/s"
    for gen in GENERATORS:
        callees = _assignments_to((POC / gen).read_text(encoding="utf-8"), "imu_angular_velocity")
        assert callees, f"{gen} no longer assigns imu_angular_velocity from a call"
        assert "angular_velocity_deg_s" in callees, (
            f"{gen} fills imu_angular_velocity via {sorted(callees)} — no deg/s producer, while the "
            f"plot axis says {imu_plot.GYRO_UNIT}. Change both or neither.")
        wrong = {c for c in callees if "rad" in c.lower()}
        assert not wrong, f"{gen} also writes that channel via {sorted(wrong)}"


def test_the_gyro_axis_string_actually_carries_the_unit() -> None:
    """GYRO_UNIT is only useful if the axis is built from it — the old label was a literal."""
    src = (POC / "imu_plot.py").read_text(encoding="utf-8")
    assert "angular vel. [rad/s]" not in src, "the hard-coded rad/s axis label is back"
    assert "f\"angular vel. [{GYRO_UNIT}]\"" in src, "the gyro axis no longer uses GYRO_UNIT"


def _write_small(tmp_path, frames, sites, **extra) -> None:
    import numpy as np

    np.savez(tmp_path / "small_reference.npz",
             sensor_codes=np.array(sites),
             imu_acceleration=np.zeros((frames, len(sites), 3), np.float32),
             imu_angular_velocity=np.zeros((frames, len(sites), 3), np.float32),
             imu_orientation=np.tile(np.array([1, 0, 0, 0], np.float32), (frames, len(sites), 1)),
             **extra)


def test_a_site_the_manifest_calls_unbacked_is_said_so_on_the_plot(tmp_path) -> None:
    """A manifest may list sites whose channel is FK of a joint the source never observed (the
    2026-09-05 GAITEX bundle did, with `imu_confidence` 0.1 there; the regenerated bundle carries
    `worn_unit_compared_sites` / `declared_placement_sites` instead). Whatever list a bundle ships,
    the plot must carry it onto the figure, or the curves read like any measured sensor's."""
    import json

    import numpy as np

    imu_plot = _load("imu_plot.py")
    frames, sites = 5, imu_plot.SITES
    conf = np.full((frames, len(sites)), 0.6, np.float32)
    conf[:, sites.index("wrist_l")] = 0.1
    _write_small(tmp_path, frames, sites, imu_confidence=conf)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "unbacked_sites": ["wrist_l"],
        "worn_unit_compared_sites": ["shank_l"],      # the regenerated bundle's two other lists
        "declared_placement_sites": ["occiput"],
        "small_synthesis": {"imu_confidence_semantics": {"invalid_cell": 0.0}},   # a declared scale
    }), encoding="utf-8")

    data = imu_plot.load_imu(str(tmp_path))
    assert data["unbacked"] == {"wrist_l"}
    assert data["compared"] == {"shank_l"} and data["declared"] == {"occiput"}
    assert data["confidence_semantics"] is True
    assert "NOT source-backed" in imu_plot.site_note(data, "wrist_l")
    assert "confidence 0.10" in imu_plot.site_note(data, "wrist_l")
    assert imu_plot.site_note(data, "shank_l") == "confidence 0.60   ·   vs worn unit"
    assert imu_plot.site_note(data, "occiput") == "confidence 0.60   ·   declared mount"
    assert imu_plot.site_note(data, "foot_r") == "confidence 0.60"

    fig, axes = imu_plot.make_figure(300, 400)
    imu_plot.draw(fig, axes, data, "wrist_l", 1)
    assert "NOT source-backed" in axes[0].get_title()
    imu_plot.draw(fig, axes, data, "shank_l", 1)
    assert "NOT source-backed" not in axes[0].get_title()


def test_masked_frames_are_shaded_and_counted_and_kept_out_of_the_confidence(tmp_path) -> None:
    """The regenerated GAITEX small is NaN wherever imu_valid_mask is False (marker gaps; one
    wrist-trial keeps 18 % of its frames). The plot must show the gap as a gap, say how much of the
    channel is there, and not average the 0.0 confidence of the missing cells into the number."""
    import numpy as np

    imu_plot = _load("imu_plot.py")
    frames, sites = 10, imu_plot.SITES
    w = sites.index("wrist_r")
    mask = np.ones((frames, len(sites)), bool)
    mask[2:5, w] = False                        # one gap of three frames
    mask[9, w] = False                          # and the LAST frame: a gap the shading must not lose
    conf = np.full((frames, len(sites)), 0.6, np.float32)
    conf[~mask] = 0.0
    _write_small(tmp_path, frames, sites, imu_confidence=conf, imu_valid_mask=mask)
    data = imu_plot.load_imu(str(tmp_path))
    for key in ("accel", "gyro", "orient"):
        data[key][~mask] = np.nan               # what the generator writes under the mask

    assert imu_plot.invalid_spans(data, "wrist_r") == [(2, 5), (9, 10)]
    assert imu_plot.invalid_spans(data, "shank_l") == []
    assert abs(imu_plot.valid_fraction(data, "wrist_r") - 0.6) < 1e-9
    note = imu_plot.site_note(data, "wrist_r")
    assert "valid 60%" in note and "confidence 0.60" in note, note
    # no manifest here, so the number carries no declared meaning and is marked as corpus-local
    assert imu_plot.site_note(data, "shank_l") == "confidence 0.60 (corpus scale)"

    fig, axes = imu_plot.make_figure(300, 400)
    imu_plot.draw(fig, axes, data, "wrist_r", 3)          # frame 3 = index 2, inside the gap
    assert all(len(ax.patches) == 2 for ax in axes), [len(ax.patches) for ax in axes]
    widths = [float(np.ptp(p.get_verts()[:, 0])) for p in axes[0].patches]
    assert all(wd > 0 for wd in widths), f"a shaded gap has zero width: {widths}"
    assert "valid 60%" in axes[0].get_title() and "4 frames masked" in axes[0].get_title()
    assert "CURSOR ON A MASKED FRAME" in axes[0].get_title()
    imu_plot.draw(fig, axes, data, "wrist_r", 1)          # frame 1 is valid: no cursor warning
    assert "CURSOR" not in axes[0].get_title()
    imu_plot.draw(fig, axes, data, "shank_l", 3)
    assert all(len(ax.patches) == 0 for ax in axes)


def test_a_run_without_manifest_or_confidence_plots_as_before(tmp_path) -> None:
    imu_plot = _load("imu_plot.py")
    _write_small(tmp_path, 3, imu_plot.SITES)
    data = imu_plot.load_imu(str(tmp_path))
    assert data["unbacked"] == set() and data["confidence"] is None
    assert imu_plot.site_note(data, "wrist_l") == ""
    fig, axes = imu_plot.make_figure(300, 400)
    imu_plot.draw(fig, axes, data, "wrist_l", 1)
    assert axes[0].get_title() == "IMU: wrist_l   ·   frame 1"
