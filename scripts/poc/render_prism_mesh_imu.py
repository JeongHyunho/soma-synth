"""Render the PRISM SMPL-X body mesh + CV-standard RGB coordinate axes of the 8 IMU sites
(attached to the skin surface).

Reuses the separately distributed render_amass.py presentation package (add-on, ground
correction, look, camera; required) +
- IMU position: projected onto the «mesh surface» each frame (limbs and head = nearest surface
  point, chest = forward ray cast), then lifted slightly along the surface normal → it sits on the
  skin instead of inside the body.
- Axes: the computer-vision standard coordinate frame (OpenCV/Open3D/ROS style) — pure R/G/B,
  shaft (cylinder) + arrowhead (cone). Oriented by the frame of the IMU site's SMPL-X segment
  (bone).  X=red · Y=green · Z=blue.

  blender -b --python render_prism_mesh_imu.py -- --anim <npz> --out <dir> --range 1:333:1

INTERNAL-ONLY.
"""
import argparse, os, sys
import numpy as np                 # standard location: independent of the sys.path edits

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import viewer_paths                         # noqa: E402
ANIM_DIR = viewer_paths.add_to_sys_path()   # render_amass (viewer_paths.anim_dir)

import bpy
import addon_utils
from mathutils import Vector, Matrix
import render_amass as ra

# IMU site -> (mount, orienting bone, surface projection, axis frame)   (per the 00_sensors.qmd spec)
#   mount = ("bone", bone) / ("along", distal bone, proximal bone, frac)
#   frame (spec axis convention): ("limb", "L"/"R")=+Y proximal, +Z lateral, +X=YxZ / "trunk", "head"=+Y up, +Z right, +X forward / "foot"
IMU_SITES = [
    ("back_T4", ("along", "spine3", "neck", 0.20), "spine3", "backward", "trunk"),  # upper thoracic back (T4)
    ("occiput", ("bone", "head"), "head", "backward", "head"),                       # back of the head
    ("wrist_l", ("along", "left_wrist", "left_elbow", 0.16), "left_elbow", "lateral_L", ("limb", "L")),
    ("wrist_r", ("along", "right_wrist", "right_elbow", 0.16), "right_elbow", "lateral_R", ("limb", "R")),
    ("shank_l", ("along", "left_ankle", "left_knee", 0.72), "left_knee", "forward", ("limb", "L")),   # front of the upper shin
    ("shank_r", ("along", "right_ankle", "right_knee", 0.72), "right_knee", "forward", ("limb", "R")),
    ("foot_l",  ("along", "left_foot", "left_ankle", 0.5), "left_foot", "down", ("foot", "L")),   # FSR insole IMU (middle of the sole)
    ("foot_r",  ("along", "right_foot", "right_ankle", 0.5), "right_foot", "down", ("foot", "R")),
]

# CV standard: X=red, Y=green, Z=blue (pure, emissive)
AXES = [("x", (1, 0, 0), (0.90, 0.03, 0.03)),
        ("y", (0, 1, 0), (0.05, 0.75, 0.05)),
        ("z", (0, 0, 1), (0.05, 0.20, 0.95))]
SHAFT_LEN, SHAFT_R = 0.090, 0.0060
HEAD_LEN, HEAD_R = 0.034, 0.0135
SURFACE_LIFT = 0.020        # distance lifted above the skin (m)
REF_FRAME = 100             # reference (near-neutral) frame for computing the rigid attachment points
Z = Vector((0, 0, 1))


def parse():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--anim", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--range", default="1:333:1")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--res", default="1200x900")
    p.add_argument("--view", default="follow", choices=["follow", "static"])
    p.add_argument("--list-bones", action="store_true")
    return p.parse_args(a)


def emissive(name, rgb):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = 0.4
    for k, v in (("Emission Color", (*rgb, 1.0)), ("Emission Strength", 1.8)):
        if k in b.inputs:
            b.inputs[k].default_value = v
    return m


def make_arrow(pivot, site, ax, edir, rgb):
    """A shaft (cylinder) + head (cone) arrow along the pivot's local +edir."""
    mat = emissive(f"m_{site}_{ax}", rgb)
    q = Z.rotation_difference(Vector(edir))
    # shaft
    bpy.ops.mesh.primitive_cylinder_add(radius=SHAFT_R, depth=SHAFT_LEN)
    sh = bpy.context.object; sh.name = f"ax_{site}_{ax}_shaft"
    sh.rotation_mode = "QUATERNION"; sh.rotation_quaternion = q
    sh.location = Vector(edir) * (SHAFT_LEN / 2.0)
    sh.data.materials.append(mat)
    # head (cone)
    bpy.ops.mesh.primitive_cone_add(radius1=HEAD_R, radius2=0.0, depth=HEAD_LEN, vertices=20)
    hd = bpy.context.object; hd.name = f"ax_{site}_{ax}_head"
    hd.rotation_mode = "QUATERNION"; hd.rotation_quaternion = q
    hd.location = Vector(edir) * (SHAFT_LEN + HEAD_LEN / 2.0)
    hd.data.materials.append(mat)
    for o in (sh, hd):
        o.parent = pivot; o.matrix_parent_inverse = Matrix.Identity(4)
        bpy.ops.object.shade_smooth()


def build_frame(scene, site):
    piv = bpy.data.objects.new(f"imupiv_{site}", None)
    scene.collection.objects.link(piv); piv.rotation_mode = "QUATERNION"
    for ax, edir, rgb in AXES:
        make_arrow(piv, site, ax, edir, rgb)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.011, segments=14, ring_count=10)
    d = bpy.context.object; d.name = f"imudot_{site}"
    d.data.materials.append(emissive(f"m_{site}_dot", (0.02, 0.02, 0.02)))
    d.parent = piv; d.matrix_parent_inverse = Matrix.Identity(4)
    return piv


def site_direction(proj, fwd, med):
    """Attachment direction (world). med = the body's rightward axis (L->R hip), fwd = anterior.
    forward = front, backward = back (back, occiput), lateral: left limb = -med, right limb = +med."""
    if proj == "forward":
        return fwd
    if proj == "backward":
        return -fwd
    if proj == "lateral_L":
        return -med
    if proj == "lateral_R":
        return med
    if proj == "down":
        return Vector((0, 0, -1))   # down from the sole (insole)
    return fwd


def sensor_frame_quat(frame, mount, armature, bones, med):
    """sensor->world quaternion by the 00_sensors.qmd axis convention.
    limb: +Y = proximal (knee/elbow), +Z = lateral (right = +med, left = -med), +X = YxZ (right = forward, left = backward).
    trunk/head/foot: +Y = up, +Z = the subject's right (med), +X = forward."""
    up = Vector((0, 0, 1))
    if isinstance(frame, tuple) and frame[0] == "limb":
        Yax = _head(armature, bones, mount[2]) - _head(armature, bones, mount[1])  # proximal direction
        Zax = med if frame[1] == "R" else -med
    elif isinstance(frame, tuple) and frame[0] == "foot":
        # feet are side-aware too: +Z = lateral (right = +med, left = -med) → left foot +X = backward (00_sensors.qmd)
        Yax = up
        Zax = med if frame[1] == "R" else -med
    else:
        Yax = up
        Zax = med
    Yax = Yax.normalized()
    Xax = Yax.cross(Zax); Xax.normalize()
    Zax = Xax.cross(Yax); Zax.normalize()          # orthonormalise (right-handed)
    return Matrix((Xax, Yax, Zax)).transposed().to_quaternion()   # columns = X,Y,Z


def surface_origin(ev_mesh, Mw, Mw_inv, world_pos, world_dir):
    """Ray cast to the surface along world_dir (outward from the joint centre) -> surface point + normal. The nearest point on failure."""
    o = Mw_inv @ world_pos
    d = (Mw_inv.to_3x3() @ world_dir).normalized()
    ok, loc, nrm, _ = ev_mesh.ray_cast(o, d)
    if not ok:
        ok, loc, nrm, _ = ev_mesh.closest_point_on_mesh(o)
    if not ok:
        return world_pos, Vector((0, 0, 1))
    return Mw @ loc, (Mw.to_3x3() @ nrm).normalized()


def _head(armature, bones, name):
    return armature.matrix_world @ bones[name].head


def mount_position(armature, bones, mount):
    """World position of the strap attachment point (distal part of the segment)."""
    if mount[0] == "bone":
        return _head(armature, bones, mount[1])
    _, distal, proximal, frac = mount           # frac from the distal joint towards the proximal one
    d = _head(armature, bones, distal)
    p = _head(armature, bones, proximal)
    return d + frac * (p - d)


def add_imu_axes(armature, mesh, scene):
    bones = armature.pose.bones
    needed = set()
    for _, mount, ob, _, _ in IMU_SITES:
        needed.add(ob)
        needed.update(mount[1:] if mount[0] == "bone" else mount[1:3])
    for bn in needed:
        if bn not in bones:
            raise SystemExit(f"[imu] bone missing: {bn} | have {sorted(b.name for b in bones)}")
    pivots = {s: build_frame(scene, s) for s, *_ in IMU_SITES}

    # (1) at the reference frame, compute the attachment point (bone-local) + spec axes (bone-local rotation) — a rigid strap
    ref = max(scene.frame_start, min(REF_FRAME, scene.frame_end))
    scene.frame_set(ref)
    _, med, _ = ra.body_frame(armature, scene, ref)        # med = the body's rightward axis (hip width, stable)
    fwd = Vector((0, 0, 1)).cross(med)                     # stable forward from the pelvis
    fwd = fwd.normalized() if fwd.length > 1e-6 else Vector((0, 1, 0))
    ev = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    Mw = mesh.matrix_world; Mw_inv = Mw.inverted()
    rig = {}
    for site, mount, ob, proj, frame in IMU_SITES:
        Pw = mount_position(armature, bones, mount)
        surf, nrm = surface_origin(ev, Mw, Mw_inv, Pw, site_direction(proj, fwd, med))
        origin = surf + nrm * SURFACE_LIFT
        Obw = armature.matrix_world @ bones[ob].matrix
        q_spec = sensor_frame_quat(frame, mount, armature, bones, med)     # spec axes (sensor->world)
        q_local = Obw.to_quaternion().inverted() @ q_spec                  # bone->sensor (rigidly fixed)
        rig[site] = (Obw.inverted() @ origin, q_local, ob)

    # (2) place and rotate each frame «rigidly with the bone» (moving with the segment, no sliding).
    # Calling keyframe_insert every frame repeats a sorted insert into the fcurve, superlinear in
    # the number of frames. As in smpl_rig, the samples are collected first, bound once at frame 1,
    # and the rest filled in bulk with foreach_set.
    frames = list(ra.frames_of(scene))
    T = len(frames)
    sites = [s for s, *_ in IMU_SITES]
    loc = np.empty((T, len(sites), 3), np.float64)
    quat = np.empty((T, len(sites), 4), np.float64)
    # This loop needs only the «bone matrices», but frame_set re-evaluates the whole depsgraph —
    # including the armature deform of the 6890-vertex mesh. The surface computation
    # (surface_origin) is already done at the reference frame above, so hiding the mesh during the
    # loop removes that work (the loop takes about half the time). The bone matrices are
    # bit-identical (matching checksum), so the result does not change.
    was_hidden = mesh.hide_viewport
    mesh.hide_viewport = True
    try:
        for i, f in enumerate(frames):
            scene.frame_set(f)
            for k, site in enumerate(sites):
                loc_off, q_local, ob = rig[site]
                Obw = armature.matrix_world @ bones[ob].matrix
                loc[i, k] = Obw @ loc_off
                q = Obw.to_quaternion() @ q_local
                quat[i, k] = (q.w, q.x, q.y, q.z)
    finally:
        mesh.hide_viewport = was_hidden

    # q and -q are the same rotation, but a sign flip interpolates the long way between keyframes — keep the sign continuous.
    for i in range(1, T):
        d = np.einsum("kj,kj->k", quat[i], quat[i - 1])
        quat[i][d < 0] *= -1.0

    fnum = np.asarray(frames, np.float64)
    lin = np.ones(T, np.int32)
    for k, site in enumerate(sites):
        p = pivots[site]
        p.rotation_mode = "QUATERNION"
        p.location = tuple(loc[0, k])
        p.keyframe_insert("location", frame=frames[0])            # frame 1 creates the fcurve
        p.rotation_quaternion = tuple(quat[0, k])
        p.keyframe_insert("rotation_quaternion", frame=frames[0])
        fcs = {(fc.data_path, fc.array_index): fc for fc in p.animation_data.action.fcurves}
        for path, arr, n in (("location", loc, 3), ("rotation_quaternion", quat, 4)):
            for ai in range(n):
                fc = fcs[(path, ai)]
                fc.keyframe_points.add(T - 1)                     # frame 1 is already there
                co = np.empty(T * 2)
                co[0::2] = fnum
                co[1::2] = arr[:, k, ai]
                fc.keyframe_points.foreach_set("co", co)
                fc.keyframe_points.foreach_set("interpolation", lin)
                fc.update()
    print(f"[imu] rigid spec-frame straps (ref={ref}) at {[s for s, *_ in IMU_SITES]}")


def main():
    args = parse()
    addon_utils.enable(ra.ADDON, default_set=True, persistent=True)
    ra.clear_scene()
    armature, mesh = ra.load_animation(args.anim, args.fps)
    scene = bpy.context.scene
    ra.set_range(scene, args.range)
    if args.list_bones:
        print("BONES:", sorted(b.name for b in armature.pose.bones)); return

    ra.ground_lock(armature, mesh, scene, step=1, mode="lock", win=24)
    ra.setup_world(scene)
    ra.setup_floor(0.20)
    ra.body_material(mesh, ra.SKIN)
    add_imu_axes(armature, mesh, scene)
    pelvis, _, _ = ra.body_frame(armature, scene, (scene.frame_start + scene.frame_end) // 2)
    ra.setup_lighting(Vector((pelvis.x, pelvis.y, 0.9)))
    ra.setup_camera(scene, armature, follow=(args.view == "follow"), dist=4.6)
    ra.configure_render(scene, "EEVEE", args.res, 64)

    scene.render.filepath = args.out.rstrip("/\\") + "/frame_"
    print(f"[render] seq {scene.frame_start}:{scene.frame_end}")
    bpy.ops.render.render(animation=True)
    print("[render] done")


if __name__ == "__main__":
    main()
