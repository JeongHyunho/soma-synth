"""정렬 프로브: SMPL-X 애드온이 만든 메시의 본 world 위치/방향을 프레임1에서 출력.

PRISM anim(fps=100, no resample)로 로드 → Blender frame1 = PRISM window frame0 (=abs 1000).
접지보정 전 raw 위치를 찍어 PRISM world 와 비교(정렬 여부 판정).

    blender -b --python probe_align.py -- --npz <prism_take002_smplx.npz>

--npz(prism_to_smplx.py 의 --out)는 반드시 준다. 파일이 있는지 먼저 확인한다. render_amass 는 따로
배포되는 발표 패키지이며 이 도구에 필요하다: viewer_paths 가 찾은 곳($SOMA_ANIM_DIR, 이 폴더; 로컬에
내려받지 않은 온라인 전용 파일은 건너뛴다)에서 불러오고, 없으면 이름으로 실패한다.
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
        # head(관절중심) = pb.head in world
        hw = armature.matrix_world @ pb.head
        print(f"PROBE {bn} head {hw.x:+.4f} {hw.y:+.4f} {hw.z:+.4f} "
              f"quat {q.w:+.4f} {q.x:+.4f} {q.y:+.4f} {q.z:+.4f}")


if __name__ == "__main__":
    main()
