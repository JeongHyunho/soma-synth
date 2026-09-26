"""Build a native SMPL armature + skinned mesh in Blender (the direct-SMPL, exact-shape path).

Replaces `render_amass.load_animation` (which drives the Meshcapade SMPL-X addon with betas placed
in SMPL-X's *different* shape space) with a real SMPL body: the subject's own SMPL betas produce the
EXACT rest mesh (`v_shaped`), skinned to a 24-bone SMPL armature and posed by the lossless per-joint
rotations. Bone names are the SMPL-standard joint names, so `render_prism_mesh_imu.add_imu_axes`,
`render_amass.body_frame`, and `render_amass.ground_lock` all keep working UNCHANGED.

Rest bones are built axis-identity (head at J_rest, tail +Y) so a pose bone's local quaternion equals
the SMPL local rotation and armature FK reproduces the SMPL global transforms — verified in-scene.
Pose blend shapes (posedirs) are NOT applied (Blender LBS has no equivalent); SHAPE is exact, only
pose-dependent skin correctives are omitted. bpy required. INTERNAL-ONLY.
"""
import os
import sys

import numpy as np

import bpy
from mathutils import Matrix, Vector

POC_DIR = os.path.dirname(os.path.abspath(__file__))
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)
import smpl_model as sm   # noqa: E402  (numpy only)

# SMPL 24-joint kinematic order — names must match render_prism_mesh_imu.IMU_SITES / body_frame.
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
    "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hand", "right_hand",
]


def _rotmats_to_quats(R):
    """[N,3,3] -> [N,4] (w,x,y,z) via mathutils (C, exact)."""
    return np.array([Matrix(r.tolist()).to_quaternion()[:] for r in R], dtype=np.float64)


def _local_quats_all(Rglob, parents):
    """Per-frame per-joint LOCAL rotations as sign-continuous quaternions [T,24,4] (w,x,y,z)."""
    T, nJ = Rglob.shape[0], len(parents)
    Q = np.empty((T, nJ, 4))
    for t in range(T):
        Q[t] = _rotmats_to_quats(sm.global_to_local_R(Rglob[t], parents))
    for k in range(nJ):                       # enforce sign continuity so LINEAR interp never spins
        for t in range(1, T):
            if np.dot(Q[t, k], Q[t - 1, k]) < 0.0:
                Q[t, k] = -Q[t, k]
    return Q


def _build_mesh(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(map(int, f)) for f in faces])
    me.validate()
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _build_armature(name, J):
    arm_data = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = arm_data.edit_bones
    made = []
    for k, nm in enumerate(SMPL_JOINT_NAMES):
        eb = ebs.new(nm)
        h = Vector((float(J[k, 0]), float(J[k, 1]), float(J[k, 2])))
        eb.head = h
        eb.tail = h + Vector((0.0, 0.1, 0.0))     # +Y tail, roll 0 -> rest bone matrix == identity
        eb.roll = 0.0
        made.append(eb)
    return arm, arm_data, made


def build_smpl_rig(model, betas, quats, trans, fps=100, name="SMPL", verify=True):
    """Create (armature, mesh) for the subject's exact SMPL body, posed & keyframed over the take."""
    parents = np.asarray(model["kintree_parents"], np.int64)
    Rglob = sm.quat_wxyz_to_rotmat(np.asarray(quats, np.float64))          # [T,24,3,3]
    v_shaped, J = sm.smpl_shape(model, betas)
    trans = np.asarray(trans, np.float64)
    T = Rglob.shape[0]

    arm, arm_data, made = _build_armature(f"{name}-arm", J)
    for k, eb in enumerate(made):
        p = int(parents[k])
        if p >= 0:
            eb.parent = made[p]
            eb.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh = _build_mesh(f"{name}-mesh", v_shaped, model["faces"])
    W = np.asarray(model["weights"], np.float64)                            # [6890,24]
    vgs = [mesh.vertex_groups.new(name=nm) for nm in SMPL_JOINT_NAMES]
    for k in range(len(SMPL_JOINT_NAMES)):
        col = W[:, k]
        for v in np.nonzero(col > 1e-6)[0]:
            vgs[k].add([int(v)], float(col[v]), "REPLACE")
    mesh.parent = arm
    amod = mesh.modifiers.new("Armature", "ARMATURE")
    amod.object = arm
    amod.use_vertex_groups = True

    # rest sanity: every pose bone's rest rotation must be identity (axis-identity bones)
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    rest_dev = max(float(np.linalg.norm(np.array(arm.pose.bones[nm].matrix.to_3x3()) - np.eye(3)))
                   for nm in SMPL_JOINT_NAMES)
    if rest_dev > 1e-5:
        raise RuntimeError(f"[smpl_rig] rest bones not axis-identity (dev {rest_dev:.2e})")

    # keyframe pose-bone local rotations + armature root translation.
    # keyframe_insert (not the legacy action.fcurves.new bulk API) so Blender 4.4+ slotted actions bind
    # the fcurves to the armature; then bulk-fill the auto-created fcurves via foreach_set for speed.
    # Root translation goes on the PELVIS pose-bone location (not the armature object), leaving
    # armature.location.z free for render_amass.ground_lock. Rest bones are axis-identity, so a root
    # pose-bone location equals a world translation (world pelvis = J_rest[0] + trans, per the oracle).
    Q = _local_quats_all(Rglob, parents)                                   # [T,24,4]
    pbs = [arm.pose.bones[nm] for nm in SMPL_JOINT_NAMES]
    for k, pb in enumerate(pbs):                                           # frame 1 bootstraps binding
        pb.rotation_quaternion = tuple(float(x) for x in Q[0, k])
        pb.keyframe_insert("rotation_quaternion", frame=1)
    pelvis_pb = arm.pose.bones["pelvis"]
    pelvis_pb.location = tuple(float(x) for x in trans[0])
    pelvis_pb.keyframe_insert("location", frame=1)

    frames = np.arange(1, T + 1, dtype=np.float64)
    lin = np.ones(T, dtype=np.int32)
    act = arm.animation_data.action
    fcurves = {(fc.data_path, fc.array_index): fc for fc in act.fcurves}
    for k, nm in enumerate(SMPL_JOINT_NAMES):
        dp = f'pose.bones["{nm}"].rotation_quaternion'
        for ai in range(4):
            fc = fcurves[(dp, ai)]
            fc.keyframe_points.add(T - 1)                                  # frame 1 already inserted
            co = np.empty(T * 2)
            co[0::2] = frames
            co[1::2] = Q[:, k, ai]
            fc.keyframe_points.foreach_set("co", co)
            fc.keyframe_points.foreach_set("interpolation", lin)
            fc.update()
    ploc = 'pose.bones["pelvis"].location'
    for ai in range(3):
        fc = fcurves[(ploc, ai)]
        fc.keyframe_points.add(T - 1)
        co = np.empty(T * 2)
        co[0::2] = frames
        co[1::2] = trans[:, ai]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", lin)
        fc.update()

    scene.frame_start, scene.frame_end = 1, T
    scene.render.fps = int(fps)
    bpy.context.view_layer.update()

    if verify:
        _verify_pose(arm, Rglob, parents, frame=min(100, T))
    print(f"[smpl_rig] built exact SMPL rig: verts={len(mesh.data.vertices)} bones={len(made)} "
          f"frames=[1,{T}] stature~{np.ptp(v_shaped[:,1]):.3f}m (rest)")
    return arm, mesh


def _verify_pose(arm, Rglob, parents, frame):
    """Prove armature FK reproduces the SMPL global rotations at a posed frame (world = armature-space,
    since the object carries only a translation)."""
    scene = bpy.context.scene
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    Rg = Rglob[frame - 1]
    worst = 0.0
    for k, nm in enumerate(SMPL_JOINT_NAMES):
        Rb = np.array(arm.pose.bones[nm].matrix.to_3x3())
        cos = (np.trace(Rb.T @ Rg[k]) - 1.0) / 2.0
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        worst = max(worst, ang)
    if worst > 0.15:   # Blender evaluates armature FK in float32; ~0.02 deg leaf accumulation is normal
        raise RuntimeError(f"[smpl_rig] FK != SMPL global at frame {frame}: {worst:.2e} deg")
    print(f"[smpl_rig] FK verified vs SMPL global rotations @f{frame}: max {worst:.2e} deg")


def load_smpl_animation(anim_npz, model_npz, fps=100, verify=True):
    """Interface parallel to render_amass.load_animation: read the packaged SMPL anim + model -> rig.

    anim_npz keys: betas[10], quats[T,24,4] (global, prism-world, w,x,y,z), trans[T,3], (fps optional).
    """
    a = np.load(anim_npz, allow_pickle=True)
    model = sm.load_model_npz(model_npz)
    fps = int(a["fps"]) if "fps" in a.files else fps
    return build_smpl_rig(model, a["betas"], a["quats"], a["trans"], fps=fps, verify=verify)
