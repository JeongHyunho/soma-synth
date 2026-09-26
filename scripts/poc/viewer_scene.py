"""Blender-side scene builder for the Small/Large interactive viewer.

Reuses the presentation SMPL-X pipeline (`render_amass`) and the mesh+IMU-axes builder
(`render_prism_mesh_imu`) UNCHANGED, and only swaps the mp4/render TAIL for: add an orbit camera,
save a .blend, and offscreen-render 3 frames (start/mid/end) to PNG for the automatic pixel +
structural checks. Opening the interactive viewport is the launcher's job (viewer_open.py on the
saved .blend), so this script is headless build-and-verify only.

Launched (headless) by viewer_launch.py, e.g.:
  blender --background --python viewer_scene.py -- --anim <smplx.npz> --blend <out.blend> --verify-out <dir>

Requires the SMPL-X addon; the launcher sets BLENDER_USER_RESOURCES so it loads.
INTERNAL-ONLY.
"""
import argparse
import json
import os
import sys

# The presentation package (render_amass, gear_kit, ...) is distributed separately and required
# here; viewer_paths finds it ($SOMA_ANIM_DIR, or beside these scripts in a portable bundle).
POC_DIR = os.path.dirname(os.path.abspath(__file__))
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)
import viewer_paths                          # noqa: E402
ANIM_DIR = viewer_paths.add_to_sys_path()

import bpy                                   # noqa: E402
import addon_utils                           # noqa: E402
import numpy as np                           # noqa: E402
from mathutils import Vector                 # noqa: E402

import render_amass as ra                    # noqa: E402
import render_prism_mesh_imu as rmi          # noqa: E402


def parse():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--anim", required=True, help="anim npz (SMPL-X AMASS, or native-SMPL for --body smpl)")
    p.add_argument("--body", default="smplx", choices=["smplx", "smpl"],
                   help="smpl=native exact-shape rig from subject betas; smplx=addon cross-space approx")
    p.add_argument("--model", default=None, help="clean SMPL model npz (required for --body smpl)")
    p.add_argument("--blend", default=None, help="path to save the .blend scene")
    p.add_argument("--range", default=None, help="a:b:s override; default = full take")
    p.add_argument("--fps", type=int, default=100)
    p.add_argument("--ground-step", type=int, default=2, help="ground-lock sample stride (1=every frame, slower)")
    p.add_argument("--verify-out", default=None, help="dir for verify PNGs")
    p.add_argument("--res", default="1200x900")
    return p.parse_args(a)


def build_scene(args):
    """Construct the subject body mesh + 8 IMU RGB-axes scene (reuse only).

    --body smpl : native SMPL rig from the subject's own betas = EXACT shape (smpl_rig, addon-free).
    --body smplx: Meshcapade SMPL-X addon, betas in SMPL-X's space = approximate cross-space shape.
    """
    if args.body == "smpl":
        if not args.model:
            raise SystemExit("[viewer] --body smpl requires --model <clean SMPL npz>")
        ra.clear_scene()
        import smpl_rig
        armature, mesh = smpl_rig.load_smpl_animation(args.anim, args.model, fps=args.fps)
    else:
        addon_utils.enable(ra.ADDON, default_set=True, persistent=True)
        ra.clear_scene()
        armature, mesh = ra.load_animation(args.anim, args.fps)  # addon sets frame_start/end
    scene = bpy.context.scene
    if args.range:
        ra.set_range(scene, args.range)
    # else: keep the full [1, T] range (what the viewer wants)

    ra.ground_lock(armature, mesh, scene, step=args.ground_step, mode="lock", win=24)
    ra.setup_world(scene)
    ra.setup_floor(0.20)
    ra.body_material(mesh, ra.SKIN)
    rmi.add_imu_axes(armature, mesh, scene)                   # 8 IMU rigid straps + RGB spec axes
    pelvis, _, _ = ra.body_frame(armature, scene, (scene.frame_start + scene.frame_end) // 2)
    ra.setup_lighting(Vector((pelvis.x, pelvis.y, 0.9)))
    ra.configure_render(scene, "EEVEE", args.res, 64)
    return scene, armature, mesh


CAMERA_SAMPLES = 1200          # cap on pelvis evaluations, whatever the take length


def bulk_keyframe(obj, data_path, values, frames, indices=None):
    """Write a whole fcurve at once instead of one keyframe_insert per frame.

    keyframe_insert sorts each new key into a growing fcurve, so filling a curve frame by frame is
    superlinear in the take length, and a long take froze Blender for many times longer than a
    take half its length. One insert bootstraps the curve and foreach_set fills
    the rest — the same shape smpl_rig and add_imu_axes already use, and the stored values are
    identical, so this is a cost change and not a behaviour change.
    """
    values = np.asarray(values, np.float64)
    if values.ndim == 1:
        values = values[:, None]
    n_frames, n_comp = values.shape
    indices = range(n_comp) if indices is None else list(indices)

    action = obj.animation_data.action if obj.animation_data else None
    if action:                                   # a curve left over from an earlier pass would be
        for fc in list(action.fcurves):          # appended to rather than replaced
            if fc.data_path == data_path and fc.array_index in indices:
                action.fcurves.remove(fc)

    current = list(getattr(obj, data_path))
    for c, ai in enumerate(indices):
        current[ai] = float(values[0, c])
    setattr(obj, data_path, tuple(current))
    for ai in indices:
        obj.keyframe_insert(data_path, index=ai, frame=frames[0])

    fcs = {fc.array_index: fc for fc in obj.animation_data.action.fcurves
           if fc.data_path == data_path}
    fnum = np.asarray(frames, np.float64)
    linear = np.ones(n_frames, np.int32)         # 1 == LINEAR, matching the old kp.interpolation
    for c, ai in enumerate(indices):
        fc = fcs[ai]
        fc.keyframe_points.add(n_frames - 1)     # frame[0] is already in from the bootstrap
        co = np.empty(n_frames * 2)
        co[0::2] = fnum
        co[1::2] = values[:, c]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", linear)
        fc.update()


def setup_orbit_camera(scene, armature, dist=5.2, height=0.25, aim_z=0.85):
    """Orbit camera rig: an OrbitPivot that FOLLOWS the walking subject, with the camera
    parented behind it. Rotate OrbitPivot about Z (or use the native viewport) to orbit."""
    frames = ra.frames_of(scene)
    pivot = bpy.data.objects.new("OrbitPivot", None)
    scene.collection.objects.link(pivot)
    pivot.rotation_mode = "XYZ"

    cam_data = bpy.data.cameras.new("ViewerCam")
    cam_data.lens = 50
    cam = bpy.data.objects.new("ViewerCam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.parent = pivot
    cam.location = (0.0, -dist, height)                       # behind + slightly above the pivot
    con = cam.constraints.new("TRACK_TO")
    con.target = pivot
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"

    # Each body_frame() costs a frame_set and a full depsgraph evaluation, so asking for every frame
    # of a 26 k take is 26 k rig evaluations for a path that is then averaged over 21 frames anyway.
    # Sample it instead and interpolate, the way ground_lock already samples its mesh: the camera
    # follows the walk, and the walk is exactly the low-frequency part that survives the smoothing.
    step = max(1, -(-len(frames) // CAMERA_SAMPLES))
    sample = frames[::step]
    if sample[-1] != frames[-1]:
        sample.append(frames[-1])
    pelvis_xy = np.array([[(p := ra.body_frame(armature, scene, f)[0]).x, p.y] for f in sample],
                         dtype=np.float64)
    pos = np.stack([np.interp(frames, sample, pelvis_xy[:, i]) for i in (0, 1)], axis=1)
    pos = ra.smooth2(pos, 21)

    track = np.empty((len(frames), 3), np.float64)
    track[:, :2] = pos
    track[:, 2] = aim_z
    bulk_keyframe(pivot, "location", track, frames)
    print(f"[viewer] orbit camera rig (follow+orbit) over {len(frames)} frames "
          f"({len(sample)} pelvis samples, step {step})")
    return pivot, cam


def render_verify_frames(scene, out_dir):
    """Offscreen-render start/mid/end frames for the automatic pixel check."""
    os.makedirs(out_dir, exist_ok=True)
    fs, fe = scene.frame_start, scene.frame_end
    picks = {"start": fs, "mid": (fs + fe) // 2, "end": fe}
    scene.render.image_settings.file_format = "PNG"
    written = {}
    for name, f in picks.items():
        scene.frame_set(f)
        path = os.path.join(out_dir, f"verify_{name}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        written[name] = {"frame": f, "path": path}
        print(f"[verify] rendered {name} frame {f} -> {path}")
    print("VERIFY_FRAMES " + json.dumps(written))
    return written


def assert_structure(scene):
    """Structurally prove the scene: exactly 1 SMPL-X mesh, 8 IMU pivots, 8*3 axis shafts + heads."""
    objs = list(bpy.data.objects)
    meshes = [o for o in objs if o.type == "MESH" and o.name.startswith("SMPL")]
    pivots = [o for o in objs if o.name.startswith("imupiv_")]
    shafts = [o for o in objs if o.name.startswith("ax_") and o.name.endswith("_shaft")]
    heads = [o for o in objs if o.name.startswith("ax_") and o.name.endswith("_head")]
    ok = (len(meshes) == 1 and len(pivots) == 8 and len(shafts) == 24 and len(heads) == 24)
    print("STRUCT " + json.dumps({"smplx_mesh": len(meshes), "imu_pivots": len(pivots),
                                  "axis_shafts": len(shafts), "axis_heads": len(heads), "ok": ok}))
    return ok


def purge_unused():
    """Drop unused data-blocks (SMPL-X textures are unused after the flat skin material) to shrink the .blend."""
    try:
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception:
        for coll in (bpy.data.images, bpy.data.materials, bpy.data.textures):
            for blk in list(coll):
                if getattr(blk, "users", 1) == 0:
                    try:
                        coll.remove(blk)
                    except Exception:
                        pass


def main():
    args = parse()
    scene, armature, mesh = build_scene(args)
    setup_orbit_camera(scene, armature)

    if args.blend:
        os.makedirs(os.path.dirname(args.blend) or ".", exist_ok=True)
        purge_unused()
        bpy.ops.wm.save_as_mainfile(filepath=args.blend)
        print("[viewer] saved .blend ->", args.blend)

    print(f"[viewer] scene: mesh_verts={len(mesh.data.vertices)} "
          f"frames=[{scene.frame_start},{scene.frame_end}] camera={scene.camera.name}")

    struct_ok = assert_structure(scene)
    out = args.verify_out or os.path.join(os.path.dirname(args.blend or "."), "verify_frames")
    render_verify_frames(scene, out)
    if not struct_ok:
        raise SystemExit("[viewer] structural assertion FAILED")
    print("[viewer] done (build + verify)")


if __name__ == "__main__":
    main()
