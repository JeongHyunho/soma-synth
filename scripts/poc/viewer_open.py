"""Configure the interactive viewport on an already-loaded .blend (GUI only).

Passed by the launcher as `blender <scene.blend> --python viewer_open.py`. Because the .blend
is loaded first, a real screen exists here (unlike a --background build), so the camera view,
material shading and autoplay actually take effect. INTERNAL-ONLY.
"""
import bpy

sc = bpy.context.scene
sc.frame_set(sc.frame_start)

screen = getattr(bpy.context, "screen", None)
for area in (screen.areas if screen else []):        # empty in --background (no window)
    if area.type == "VIEW_3D":
        sp = area.spaces.active
        try:
            sp.shading.type = "MATERIAL"                 # show emissive RGB axes + skin
            sp.region_3d.view_perspective = "CAMERA"     # look through ViewerCam
            sp.overlay.show_relationship_lines = False
        except Exception as e:
            print("[open] viewport tweak skipped:", repr(e))

try:
    bpy.ops.screen.animation_play()
    print("[open] autoplay started; frames", sc.frame_start, "-", sc.frame_end,
          "camera", sc.camera.name if sc.camera else None)
except Exception as e:
    print("[open] autoplay skipped:", repr(e))
