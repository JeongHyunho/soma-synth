"""Alignment probe: print the world position/orientation of the bones of the mesh the SMPL-X
add-on built, at frame 1.

Loads the PRISM animation (fps=100, no resample) → Blender frame 1 = PRISM window frame 0
(= absolute 1000). Prints the raw positions before ground correction, to compare with the PRISM
world (to judge whether they are aligned).

    blender -b --python probe_align.py -- --npz <prism_take002_smplx.npz>

--npz (the --out of prism_to_smplx.py) is required. The file is checked first. render_amass is the
separately distributed presentation package and this tool needs it: it is loaded from where
viewer_paths finds it ($SOMA_ANIM_DIR, this folder; online-only files not downloaded to this PC are
skipped), and a missing one fails by name.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viewer_paths  # noqa: E402

viewer_paths.add_to_sys_path()
import bpy, addon_utils  # noqa: E402
import render_amass as ra  # noqa: E402

BONES = ["pelvis", "spine1", "spine2", "spine3", "neck", "head",
         "left_wrist", "right_wrist", "left_ankle", "left_knee"]


def main(argv=None):
    argv = viewer_paths.script_argv() if argv is None else argv
    npz = viewer_paths.require(viewer_paths.option(argv, "--npz"), "-- --npz",
                               "the SMPL-X animation npz (prism_to_smplx.py --out)")
    if not os.path.isfile(npz):
        raise SystemExit(f"no SMPL-X animation npz at {npz}")

    addon_utils.enable(ra.ADDON, default_set=True, persistent=True)
    ra.clear_scene()
    armature, mesh = ra.load_animation(npz, 100)      # fps=100 → no resample
    scene = bpy.context.scene
    scene.frame_set(1)                                 # = PRISM window frame 0 (abs 1000)

    print("=== BLENDER frame1 (pre-ground-lock) bone world pose ===")
    for bn in BONES:
        pb = armature.pose.bones[bn]
        M = armature.matrix_world @ pb.matrix
        loc = M.to_translation(); q = M.to_quaternion()
        # head (joint centre) = pb.head in world
        hw = armature.matrix_world @ pb.head
        print(f"PROBE {bn} head {hw.x:+.4f} {hw.y:+.4f} {hw.z:+.4f} "
              f"quat {q.w:+.4f} {q.x:+.4f} {q.y:+.4f} {q.z:+.4f}")


if __name__ == "__main__":
    main()
