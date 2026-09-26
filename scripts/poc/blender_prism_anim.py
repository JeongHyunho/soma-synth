"""Blender(bpy) — PRISM faithful 참조 3D 애니메이션 -> mp4 (로컬, INTERNAL-ONLY).

실행:
  blender -b --python blender_prism_anim.py -- --run <take folder> [--smoke]
  --smoke : 단일 프레임 PNG만(스모크 테스트)

--run 은 export_anim_data.py 가 anim_data.npz 를 쓴 take 폴더다. 반드시 준다. anim_data.npz 가
있는지 먼저 확인하고, 그 뒤에야 장면을 만들고 <run>/prism_faithful_animation.mp4 (또는 --smoke 면
<run>/figures/anim_smoke.png) 를 쓴다. --run 이 번들·코퍼스(위 폴더에 INDEX.json 이 있다),
lineage 컨테이너, 증거 폴더, 원천 폴더 안이면 거부한다(viewer_paths.writable_folder):
export_anim_data.py 에 준 scratch 사본을 준다.

내용: imu_gt 8관절 스켈레톤 + 6 virtual-IMU 방향 박스 + 발 vertical-GRF 화살표.
PRISM world = Z-up = Blender Z-up (좌표 그대로). Workbench 엔진(헤드리스 안정).
INTERNAL-ONLY 스탬프를 프레임에 번인.
"""
import os, sys
import numpy as np
import bpy
import mathutils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viewer_paths  # noqa: E402

STRIDE = 2                     # 1000 -> 500 프레임
FPS = 50                       # 10 s 영상
CAM_OFFSET = mathutils.Vector((1.0, -4.0, 1.8))

# Okabe-Ito RGBA
BLUE = (0.0, 0.45, 0.70, 1); VERM = (0.84, 0.37, 0.0, 1); GREEN = (0.0, 0.62, 0.45, 1)
GRAY = (0.18, 0.18, 0.18, 1); BONE = (0.62, 0.62, 0.62, 1); IMUCOL = (0.94, 0.89, 0.26, 1)
RED = (0.86, 0.08, 0.08, 1); FLOOR = (0.86, 0.86, 0.86, 1)
IMU_OFFSET = mathutils.Vector((0.0, 0.13, 0.0))   # 센서 로컬 +Y로 띄워 방향 가시화

JOINT_COLOR = {"pelvis": GRAY, "head": GREEN, "knee_l": BLUE, "foot_l": BLUE,
               "wrist_l": BLUE, "knee_r": VERM, "foot_r": VERM, "wrist_r": VERM}


def mat_color(obj, rgba):
    obj.color = rgba


def new_sphere(name, r, rgba):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=16, ring_count=8)
    o = bpy.context.object; o.name = name; mat_color(o, rgba)
    bpy.ops.object.shade_smooth()
    return o


def new_cyl(name, r, rgba):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=1.0)
    o = bpy.context.object; o.name = name; mat_color(o, rgba)
    o.rotation_mode = "QUATERNION"
    return o


def new_box(name, size, rgba):
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    o = bpy.context.object; o.name = name; o.scale = size; mat_color(o, rgba)
    o.rotation_mode = "QUATERNION"
    return o


AXIS_LEN = 0.20
AXES = [("x", (1, 0, 0), (0.90, 0.10, 0.10, 1)),   # X=red
        ("y", (0, 1, 0), (0.10, 0.80, 0.20, 1)),   # Y=green
        ("z", (0, 0, 1), (0.10, 0.45, 0.95, 1))]   # Z=blue


def new_axis(name, edir, rgba):
    """센서 로컬 축 표시용 얇은 화살대(원기둥). 로컬 +Z를 edir로 회전."""
    o = new_cyl(name, 0.014, rgba)
    o.rotation_quaternion = mathutils.Vector((0, 0, 1)).rotation_difference(mathutils.Vector(edir))
    o.location = mathutils.Vector(edir) * (AXIS_LEN / 2.0)
    o.scale = (1, 1, AXIS_LEN)
    return o


Z = mathutils.Vector((0, 0, 1))


def set_bone(o, p, c, k):
    p = mathutils.Vector(p); c = mathutils.Vector(c); dv = c - p
    L = dv.length
    o.location = (p + c) / 2.0
    o.rotation_quaternion = Z.rotation_difference(dv.normalized()) if L > 1e-9 else (1, 0, 0, 0)
    o.scale = (1, 1, max(L, 1e-6))
    for dp in ("location", "rotation_quaternion", "scale"):
        o.keyframe_insert(dp, frame=k)


def key(o, k, loc=None, quat=None, scale=None):
    if loc is not None:
        o.location = loc; o.keyframe_insert("location", frame=k)
    if quat is not None:
        o.rotation_quaternion = quat; o.keyframe_insert("rotation_quaternion", frame=k)
    if scale is not None:
        o.scale = scale; o.keyframe_insert("scale", frame=k)


def main(argv=None):
    argv = viewer_paths.script_argv() if argv is None else argv
    smoke = "--smoke" in argv
    run = viewer_paths.require(viewer_paths.option(argv, "--run"), "-- --run",
                               "the take folder holding anim_data.npz",
                               writes=viewer_paths.COPY_THE_TAKE)
    viewer_paths.writable_folder(run, "-- --run")       # the mp4 or figures/ go there: never a bundle
    data = os.path.join(viewer_paths.require_files(run, ("anim_data.npz",)), "anim_data.npz")
    out_mp4 = os.path.join(run, "prism_faithful_animation.mp4")
    out_png = os.path.join(run, "figures", "anim_smoke.png")

    d = np.load(data, allow_pickle=True)
    J = d["J"].astype(float)                 # [F,8,3]
    JOINTS = [str(x) for x in d["joints"]]
    BONES = d["bones"]                       # [7,2]
    imu_pos = d["imu_pos"].astype(float)     # [F,6,3]
    imu_quat = d["imu_quat"].astype(float)   # [F,6,4] wxyz
    sensor_codes = [str(x) for x in d["sensor_codes"]]
    foot_pos = d["foot_pos"].astype(float)   # [F,2,3]
    grf_vert = d["grf_vert"].astype(float)   # [F,2]
    contact = d["contact"].astype(bool)      # [F,2]
    grf_ref = float(d["grf_ref"]) or 1.0
    F = J.shape[0]
    frames = list(range(0, F, STRIDE))

    # ---------- scene reset ----------
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # ---------- ground ----------
    bpy.ops.mesh.primitive_plane_add(size=40, location=(J[:, :, 0].mean(), J[:, :, 1].mean(), 0.0))
    mat_color(bpy.context.object, FLOOR)

    # ---------- build objects ----------
    joint_obj = [new_sphere(f"joint_{j}", 0.055 if j != "pelvis" else 0.07, JOINT_COLOR[j]) for j in JOINTS]
    bone_obj = [new_cyl(f"bone_{i}", 0.02, BONE) for i in range(len(BONES))]
    imu_pivot = []
    for c in sensor_codes:
        piv = bpy.data.objects.new(f"imupiv_{c}", None)      # Empty
        scene.collection.objects.link(piv); piv.rotation_mode = "QUATERNION"
        box = new_box(f"imu_{c}", (0.11, 0.075, 0.04), IMUCOL)
        for child in [box] + [new_axis(f"imuax_{c}_{ax}", e, col) for ax, e, col in AXES]:
            child.parent = piv                                # pivot이 원점이라 local=world 유지
            child.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        box.location = (0, 0, 0); box.rotation_quaternion = (1, 0, 0, 0)
        imu_pivot.append(piv)
    grf_obj = [new_cyl(f"grf_{s}", 0.03, RED) for s in ("L", "R")]

    # camera + track pivot
    pivot = bpy.data.objects.new("pivot", None); scene.collection.objects.link(pivot)
    cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam); scene.camera = cam
    tt = cam.constraints.new(type="TRACK_TO"); tt.target = pivot
    tt.track_axis = "TRACK_NEGATIVE_Z"; tt.up_axis = "UP_Y"
    # sun
    sun_data = bpy.data.lights.new("sun", type="SUN"); sun_data.energy = 3.0
    sun = bpy.data.objects.new("sun", sun_data); scene.collection.objects.link(sun)
    sun.rotation_euler = (0.6, 0.2, 0.4)

    # ---------- keyframe ----------
    for k, f in enumerate(frames, start=1):
        for oi, j in enumerate(JOINTS):
            key(joint_obj[oi], k, loc=tuple(J[f, oi]))
        for bi, (a, b) in enumerate(BONES):
            set_bone(bone_obj[bi], J[f, a], J[f, b], k)
        for si in range(6):
            q = mathutils.Quaternion(tuple(imu_quat[f, si]))          # wxyz
            loc = mathutils.Vector(imu_pos[f, si]) + (q @ IMU_OFFSET)  # 센서에서 살짝 띄움
            key(imu_pivot[si], k, loc=loc, quat=tuple(imu_quat[f, si]))  # box+RGB축이 따라옴
        for s in range(2):
            h = float(np.clip(grf_vert[f, s] / grf_ref, 0, 2.2)) * 0.45 if contact[f, s] else 0.0
            fp = foot_pos[f, s]
            key(grf_obj[s], k, loc=(fp[0], fp[1], fp[2] + h / 2.0), quat=(1, 0, 0, 0),
                scale=(1, 1, max(h, 1e-6)))
        pivot.location = tuple(J[f, 0]); pivot.keyframe_insert("location", frame=k)
        cam.location = mathutils.Vector(J[f, 0]) + CAM_OFFSET; cam.keyframe_insert("location", frame=k)

    # linear interpolation (mocap 충실)
    for act in bpy.data.actions:
        for fc in act.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"

    # ---------- render settings ----------
    scene.render.engine = "BLENDER_WORKBENCH"
    disp = scene.display.shading
    disp.light = "STUDIO"; disp.color_type = "OBJECT"; disp.show_shadows = True
    disp.show_cavity = True
    scene.world = bpy.data.worlds.new("w"); scene.world.color = (0.05, 0.06, 0.08)
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, len(frames)
    # stamp burn-in
    scene.render.use_stamp = True
    scene.render.use_stamp_note = True
    scene.render.stamp_note_text = ("INTERNAL-ONLY  ·  experimental_non_candidate  ·  NOT physically validated"
                                    "  ·  PRISM world Z-up")
    scene.render.use_stamp_frame = True
    scene.render.use_stamp_time = False
    scene.render.stamp_font_size = 18
    scene.render.stamp_foreground = (1, 1, 1, 1)
    scene.render.stamp_background = (0, 0, 0, 0.55)

    if smoke:
        scene.frame_set(len(frames) // 2)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = out_png
        bpy.ops.render.render(write_still=True)
        print("SMOKE PNG ->", out_png)
    else:
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "HIGH"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        scene.render.filepath = out_mp4
        bpy.ops.render.render(animation=True)
        print("MP4 ->", out_mp4)


if __name__ == "__main__":
    main()
