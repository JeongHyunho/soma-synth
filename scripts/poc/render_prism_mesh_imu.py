"""PRISM SMPL-X 바디 메시 + 8개 IMU 부위의 CV-표준 RGB 좌표축(피부 표면 부착) 렌더.

따로 배포되는 render_amass.py 발표 패키지(애드온·접지보정·룩·카메라)를 재사용한다(필수) +
- IMU 위치: 프레임마다 «메시 표면»에 투영(사지·머리=최근접 표면점, 가슴=전방 레이캐스트) 후
  표면 법선으로 살짝 띄워 배치 → 몸속에 박히지 않고 피부 위에 놓인다.
- 축: computer-vision 표준 좌표프레임(OpenCV/Open3D/ROS 류) — 순수 R/G/B, shaft(원기둥)+화살촉(원뿔).
  방향은 해당 IMU 부위 SMPL-X 분절(본) 프레임.  X=red · Y=green · Z=blue.

  blender -b --python render_prism_mesh_imu.py -- --anim <npz> --out <dir> --range 1:333:1

INTERNAL-ONLY.
"""
import argparse, os, sys
import numpy as np                 # 표준 위치: sys.path 조작과 무관하다

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import viewer_paths                         # noqa: E402
ANIM_DIR = viewer_paths.add_to_sys_path()   # render_amass (viewer_paths.anim_dir)

import bpy
import addon_utils
from mathutils import Vector, Matrix
import render_amass as ra

# IMU site -> (mount, 방향본, 표면투영, 축프레임)   (00_sensors.qmd 스펙 반영)
#   mount = ("bone", 본) / ("along", 원위본, 근위본, frac)
#   frame(스펙 축규약): ("limb", "L"/"R")=+Y근위·+Z바깥측방·+X=YxZ / "trunk"·"head"=+Y상방·+Z우측·+X전방 / "foot"
IMU_SITES = [
    ("back_T4", ("along", "spine3", "neck", 0.20), "spine3", "backward", "trunk"),  # 등 상부흉추(T4)
    ("occiput", ("bone", "head"), "head", "backward", "head"),                       # 후두부
    ("wrist_l", ("along", "left_wrist", "left_elbow", 0.16), "left_elbow", "lateral_L", ("limb", "L")),
    ("wrist_r", ("along", "right_wrist", "right_elbow", 0.16), "right_elbow", "lateral_R", ("limb", "R")),
    ("shank_l", ("along", "left_ankle", "left_knee", 0.72), "left_knee", "forward", ("limb", "L")),   # 정강이 상부 전면
    ("shank_r", ("along", "right_ankle", "right_knee", 0.72), "right_knee", "forward", ("limb", "R")),
    ("foot_l",  ("along", "left_foot", "left_ankle", 0.5), "left_foot", "down", ("foot", "L")),   # FSR 인솔 IMU(족저 중앙)
    ("foot_r",  ("along", "right_foot", "right_ankle", 0.5), "right_foot", "down", ("foot", "R")),
]

# CV 표준: X=red, Y=green, Z=blue (순수·발광)
AXES = [("x", (1, 0, 0), (0.90, 0.03, 0.03)),
        ("y", (0, 1, 0), (0.05, 0.75, 0.05)),
        ("z", (0, 0, 1), (0.05, 0.20, 0.95))]
SHAFT_LEN, SHAFT_R = 0.090, 0.0060
HEAD_LEN, HEAD_R = 0.034, 0.0135
SURFACE_LIFT = 0.020        # 피부 위로 띄우는 거리(m)
REF_FRAME = 100             # 강체 부착점을 계산할 기준(중립에 가까운) 프레임
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
    """shaft(원기둥)+head(원뿔) 화살표를 pivot 로컬 +edir 방향으로."""
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
    """부착 방향(world). med=몸 오른쪽축(L->R hip), fwd=전방(anterior).
    forward=앞, backward=뒤(등/후두부), 가쪽: 왼쪽 사지=-med, 오른쪽 사지=+med."""
    if proj == "forward":
        return fwd
    if proj == "backward":
        return -fwd
    if proj == "lateral_L":
        return -med
    if proj == "lateral_R":
        return med
    if proj == "down":
        return Vector((0, 0, -1))   # 족저(인솔) 하방
    return fwd


def sensor_frame_quat(frame, mount, armature, bones, med):
    """00_sensors.qmd 축규약 sensor->world 쿼터니언.
    limb: +Y=근위(무릎/팔꿈치), +Z=바깥측방(우=+med,좌=-med), +X=YxZ(우=전방/좌=후방).
    trunk/head/foot: +Y=상방, +Z=대상자 우측(med), +X=전방."""
    up = Vector((0, 0, 1))
    if isinstance(frame, tuple) and frame[0] == "limb":
        Yax = _head(armature, bones, mount[2]) - _head(armature, bones, mount[1])  # 근위방향
        Zax = med if frame[1] == "R" else -med
    elif isinstance(frame, tuple) and frame[0] == "foot":
        # 발도 side-aware: +Z=바깥측방(우=+med, 좌=-med) → 좌발 +X=후방 (00_sensors.qmd)
        Yax = up
        Zax = med if frame[1] == "R" else -med
    else:
        Yax = up
        Zax = med
    Yax = Yax.normalized()
    Xax = Yax.cross(Zax); Xax.normalize()
    Zax = Xax.cross(Yax); Zax.normalize()          # 직교 정규화(우수 좌표계)
    return Matrix((Xax, Yax, Zax)).transposed().to_quaternion()   # 열 = X,Y,Z


def surface_origin(ev_mesh, Mw, Mw_inv, world_pos, world_dir):
    """world_dir 로 표면 레이캐스트(관절중심에서 바깥으로) -> 표면점+법선. 실패 시 최근접점."""
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
    """스트랩 부착 지점(분절 위 원위부) world 좌표."""
    if mount[0] == "bone":
        return _head(armature, bones, mount[1])
    _, distal, proximal, frac = mount           # 원위관절에서 근위쪽으로 frac
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

    # (1) 기준 프레임에서 부착점(본-로컬) + 스펙 축(본-로컬 회전)을 계산 — 강체 스트랩
    ref = max(scene.frame_start, min(REF_FRAME, scene.frame_end))
    scene.frame_set(ref)
    _, med, _ = ra.body_frame(armature, scene, ref)        # med=몸 오른쪽축(엉덩이 너비, 안정적)
    fwd = Vector((0, 0, 1)).cross(med)                     # 골반 기준 안정 전방
    fwd = fwd.normalized() if fwd.length > 1e-6 else Vector((0, 1, 0))
    ev = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    Mw = mesh.matrix_world; Mw_inv = Mw.inverted()
    rig = {}
    for site, mount, ob, proj, frame in IMU_SITES:
        Pw = mount_position(armature, bones, mount)
        surf, nrm = surface_origin(ev, Mw, Mw_inv, Pw, site_direction(proj, fwd, med))
        origin = surf + nrm * SURFACE_LIFT
        Obw = armature.matrix_world @ bones[ob].matrix
        q_spec = sensor_frame_quat(frame, mount, armature, bones, med)     # 스펙 축(sensor->world)
        q_local = Obw.to_quaternion().inverted() @ q_spec                  # 본->센서 (강체 고정)
        rig[site] = (Obw.inverted() @ origin, q_local, ob)

    # (2) 매 프레임 «본 강체»로 배치·회전 (분절과 함께, 미끄러짐 없음).
    # 프레임마다 keyframe_insert 를 부르면 fcurve 에 정렬 삽입이 반복돼 프레임 수에 초선형이 된다.
    # smpl_rig 와 같은 방식으로, 먼저 샘플만 모은 뒤
    # 프레임 1에서 한 번만 바인딩하고 나머지는 foreach_set 으로 벌크 채운다.
    frames = list(ra.frames_of(scene))
    T = len(frames)
    sites = [s for s, *_ in IMU_SITES]
    loc = np.empty((T, len(sites), 3), np.float64)
    quat = np.empty((T, len(sites), 4), np.float64)
    # 이 루프는 «본 행렬»만 필요한데 frame_set 은 depsgraph 전체를 다시 평가한다 — 6890 정점 메시의
    # armature deform 까지. 표면 계산(surface_origin)은 위 기준 프레임에서 이미 끝났으므로, 루프
    # 동안 메시를 숨기면 그만큼이 빠진다(루프 시간이 약 절반으로 준다). 본 행렬은
    # 비트 단위로 동일(checksum 일치)이라 결과는 바뀌지 않는다.
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

    # q 와 -q 는 같은 회전이지만 부호가 튀면 키프레임 사이를 먼 길로 보간한다 — 부호를 이어 붙인다.
    for i in range(1, T):
        d = np.einsum("kj,kj->k", quat[i], quat[i - 1])
        quat[i][d < 0] *= -1.0

    fnum = np.asarray(frames, np.float64)
    lin = np.ones(T, np.int32)
    for k, site in enumerate(sites):
        p = pivots[site]
        p.rotation_mode = "QUATERNION"
        p.location = tuple(loc[0, k])
        p.keyframe_insert("location", frame=frames[0])            # 프레임 1이 fcurve 를 만든다
        p.rotation_quaternion = tuple(quat[0, k])
        p.keyframe_insert("rotation_quaternion", frame=frames[0])
        fcs = {(fc.data_path, fc.array_index): fc for fc in p.animation_data.action.fcurves}
        for path, arr, n in (("location", loc, 3), ("rotation_quaternion", quat, 4)):
            for ai in range(n):
                fc = fcs[(path, ai)]
                fc.keyframe_points.add(T - 1)                     # 프레임 1은 이미 들어가 있다
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
