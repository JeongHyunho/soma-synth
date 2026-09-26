"""No fcurve may be filled one frame at a time.

keyframe_insert sorts each key into a growing fcurve, so calling it once per frame is superlinear
in take length. The invariant enforced here: in the viewer and render scripts, a `for` loop that
iterates over frames must not call keyframe_insert. Bulk writers (one insert to bootstrap the
curve, then foreach_set) loop over components, not frames, and are untouched by this.
"""
from __future__ import annotations

import ast
from pathlib import Path

POC = Path(__file__).parents[2] / "scripts" / "poc"
FILES = ("viewer_scene.py", "render_prism_mesh_imu.py", "smpl_rig.py", "viewer_app.py")


def _mentions_frames(node: ast.AST) -> bool:
    """Does this loop iterate over something frame-shaped?"""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and "frame" in sub.id.lower():
            return True
        if isinstance(sub, ast.Attribute) and "frame" in sub.attr.lower():
            return True
    return False


def _per_frame_keyframe_loops(source: str) -> list[int]:
    """Line numbers of `for <over frames>: ... keyframe_insert(...)` loops."""
    tree = ast.parse(source)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor)) or not _mentions_frames(node.iter):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "keyframe_insert"):
                bad.append(node.lineno)
                break
    return bad


def test_no_fcurve_is_filled_one_frame_at_a_time() -> None:
    offenders = {}
    for name in FILES:
        path = POC / name
        if not path.exists():
            continue
        lines = _per_frame_keyframe_loops(path.read_text(encoding="utf-8"))
        if lines:
            offenders[name] = lines
    assert not offenders, (
        f"keyframe_insert is being called inside a loop over frames at {offenders}. Fill the fcurve "
        f"in bulk instead: one keyframe_insert to create the curve, then keyframe_points.add() + "
        f"foreach_set('co', ...), as viewer_scene.bulk_keyframe does.")


def test_the_check_can_actually_see_the_pattern() -> None:
    """A ban nobody can trip is not a ban — show it fires on a per-frame loop."""
    per_frame = (
        "def setup(pivot, frames, pos):\n"
        "    for i, f in enumerate(frames):\n"
        "        pivot.location = pos[i]\n"
        "        pivot.keyframe_insert('location', frame=f)\n"
    )
    assert _per_frame_keyframe_loops(per_frame) == [2]
    bulk = (
        "def setup(obj, frames, values, indices):\n"
        "    for axis in indices:\n"
        "        obj.keyframe_insert('location', index=axis, frame=frames[0])\n"
    )
    assert _per_frame_keyframe_loops(bulk) == [], "the bulk writer must not be flagged"
