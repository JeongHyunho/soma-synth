#!/usr/bin/env python
"""One-line launcher for the Small/Large interactive viewer.

Reconstructs SMPL params from the distributed small/large/development_reference npz and opens a
Blender viewport of the (approximate cross-space) subject-shape SMPL-X mesh with the 8 IMU
positions + RGB sensor axes (play / scrub / orbit). Not an mp4. INTERNAL-ONLY.

    python scripts/poc/viewer_launch.py --run <dir>            # enrich-once (if needed) -> build (cache-aware) -> open
    python scripts/poc/viewer_launch.py --run <dir> --verify   # force a fresh build + pixel check, then open
    python scripts/poc/viewer_launch.py --run <dir> --rebuild  # force rebuild
    python scripts/poc/viewer_launch.py --run <dir> --no-open  # build/verify only

`--run` is a folder holding the three npz (or pass --small / --large / --dev). There is no default
run.

DATA-PLANE: heavy generated artifacts (.blend, recon npz, PNGs, manifest) are written under
<data root>/artifacts/viewers/<content_id>/ -- the data root is --data-root, else $SOMA_DATA_ROOT,
never the repo. The content_id hashes the actual bytes of the three input npz +
every output-affecting script (computed AFTER enrichment); the completion marker is written
atomically only after build + pixel verification pass, so a stale or unverified scene is never
reused.

Requires Blender 4.5 + the Meshcapade SMPL-X addon (BLENDER_USER_RESOURCES auto-detected, incl. an
app-container (MSIX) path on Windows). Blender exe: --blender, else $BLENDER_EXE, else `blender` on
PATH, else the platform's usual install location.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import viewer_paths                                              # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))            # repository root
POC = os.path.join(REPO, "scripts", "poc")
ANIM_DIR = viewer_paths.anim_dir()                               # "" when no copy is readable here
RENDER_AMASS = os.path.join(ANIM_DIR, "render_amass.py") if ANIM_DIR else ""   # external reused dependency
ADDON_REL = os.path.join("extensions", "user_default", "smplx_blender_addon")
INPUT_NPZ = ("small_reference.npz", "large_reference.npz", "development_reference.npz")
# license-gated MPI SMPL body model (male matches PRISM subj001); pkl -> clean npz, both in the
# body-model folder ($SOMA_BODY_MODEL_DIR, else <data root>/body_models/smpl)
SMPL_MODEL_PKL_NAME = "basicmodel_m_lbs_10_207_0_v1.1.0.pkl"
SMPL_MODEL_CLEAN_NAME = "SMPL_MALE_clean.npz"
#: Where Blender 4.5 usually is when neither --blender, $BLENDER_EXE nor PATH names it.
_BLENDER_USUAL = {
    "win32": r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    "darwin": "/Applications/Blender.app/Contents/MacOS/Blender",
}
SCRIPTS_FOR_HASH = ("smpl_recon.py", "smpl_model.py", "smpl_rig.py", "viewer_scene.py",
                    "render_prism_mesh_imu.py", "viewer_verify_pixels.py", "viewer_open.py",
                    "viewer_paths.py")                           # decides WHICH render_amass is used
VIEWER_VERSION = "4"                                             # bump when output-affecting behavior changes
SHAPE_NOTE = {
    "smpl": "exact native SMPL shape from the subject's own betas (no cross-space approximation)",
    "smplx": ("approximate unvalidated cross-space shape: SMPL betas placed in SMPL-X shape space "
              "(different PCA basis); not a validated real SMPL-X fit"),
}


def default_blender():
    """$BLENDER_EXE, else `blender` on PATH, else the platform's usual location ("" when none)."""
    return (os.environ.get("BLENDER_EXE") or shutil.which("blender")
            or _BLENDER_USUAL.get(sys.platform, ""))


def find_user_resources():
    cands = []
    if os.environ.get("LOCALAPPDATA"):                   # Windows, incl. the MSIX app container
        cands += glob.glob(os.path.join(os.environ["LOCALAPPDATA"], "Packages", "*", "LocalCache",
                                        "Roaming", "Blender Foundation", "Blender", "4.5"))
    if os.environ.get("APPDATA"):
        cands.append(os.path.join(os.environ["APPDATA"], "Blender Foundation", "Blender", "4.5"))
    home = os.path.expanduser("~")
    cands.append(os.path.join(home, "Library", "Application Support", "Blender", "4.5"))  # macOS
    cands.append(os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config"),
                              "blender", "4.5"))                                          # Linux
    for c in cands:
        if c and os.path.isdir(os.path.join(c, ADDON_REL)):
            return c
    return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_id(run, ground_step, body, model_clean):
    """Content-address by the ACTUAL BYTES of the three input npz + every output-affecting script/config
    (+ the clean SMPL model for --body smpl). Call AFTER enrichment. mtime is never used (unreliable)."""
    h = hashlib.sha256()
    h.update(f"v{VIEWER_VERSION}:body{body}:gs{ground_step}:".encode())
    for name in INPUT_NPZ:
        h.update((name + ":").encode())
        h.update(bytes.fromhex(sha256_file(os.path.join(run, name))))
    for s in SCRIPTS_FOR_HASH:
        h.update(bytes.fromhex(sha256_file(os.path.join(POC, s))))
    if body == "smpl" and os.path.isfile(model_clean):
        h.update(b"smpl_model:")
        h.update(bytes.fromhex(sha256_file(model_clean)))
    # The dependency's bytes when a copy resolves, an explicit sentinel when none does: a build made
    # without it must not share an id with one made with it.
    h.update(b"render_amass:")
    h.update(bytes.fromhex(sha256_file(RENDER_AMASS)) if RENDER_AMASS and os.path.isfile(RENDER_AMASS)
             else b"none")
    return h.hexdigest()[:16]


def is_enriched(dev_path):
    import numpy as np
    z = np.load(dev_path, allow_pickle=True)
    return ("betas" in z.files) and ("smpl_trans" in z.files)


def git_state():
    try:
        commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", REPO, "status", "--porcelain"],
                                             text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "dirty": None}


def logical_run_id(run, data_root):
    """Logical asset id relative to the data root (never an absolute path)."""
    try:
        rel = os.path.relpath(run, data_root)
        if not rel.startswith(".."):
            return rel.replace("\\", "/")
    except Exception:
        pass
    return os.path.basename(run)


def write_manifest(out_dir, run, cid, ground_step, body, data_root):
    gs = git_state()
    manifest = {
        "distribution_scope": "internal_only",
        "artifact_class": "experimental_non_candidate",
        "artifact": f"small_large_{body}_imu_viewer",
        "body_model": body,
        "content_id": cid,
        "viewer_version": VIEWER_VERSION,
        "ground_step": ground_step,
        "code_commit": gs["commit"],
        "code_dirty": gs["dirty"],
        "inputs_run_logical": logical_run_id(run, data_root),
        "input_sha256": {name: sha256_file(os.path.join(run, name)) for name in INPUT_NPZ},
        "generating_scripts_sha256": {s: sha256_file(os.path.join(POC, s)) for s in SCRIPTS_FOR_HASH},
        "shape_note": SHAPE_NOTE[body],
        "smpl_model_note": ("native SMPL body model (MPI, license-gated, non-redistributable); "
                            "baked mesh is a research derivative — INTERNAL-ONLY, non-commercial."
                            if body == "smpl" else "Meshcapade SMPL-X addon body."),
        "axes_note": "RGB sensor axes = 00_sensors.qmd spec frame, rigid-strapped; foot frames side-aware.",
        "trans_note": "root translation = development_reference.smpl_trans (faithful source SMPL trans).",
        "handling": "INTERNAL-ONLY; do not upload. Data payloads stay under the data root.",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def ensure_clean_smpl_model(pkl, clean):
    """Return the clean SMPL model npz, extracting it once (safe unpickler) from the licensed pkl."""
    if os.path.isfile(clean):
        return clean
    if not os.path.isfile(pkl):
        sys.exit(f"[view] --body smpl needs the MPI SMPL body model, which is absent:\n"
                 f"         clean npz : {clean}\n         source pkl: {pkl}\n"
                 f"       Register at https://smpl.is.tue.mpg.de, download 'SMPL for Python v1.1.0', and put the\n"
                 f"       male model {SMPL_MODEL_PKL_NAME} in {os.path.dirname(pkl)} (INTERNAL-ONLY),\n"
                 f"       or pass --body smplx for the approximate (addon) viewer.")
    print("[view] clean-extracting the SMPL model (one-time; safe numpy/chumpy-shim unpickler) ...")
    subprocess.run([sys.executable, os.path.join(POC, "smpl_model.py"), "--extract", pkl, clean], check=True)
    return clean


def main():
    ap = argparse.ArgumentParser(description="Interactive subject body mesh + 8 IMU RGB-axes viewer.")
    ap.add_argument("--run", default=None, help="dir holding the three npz (required unless "
                                                "--small / --large / --dev name them)")
    ap.add_argument("--small")
    ap.add_argument("--large")
    ap.add_argument("--dev", help="explicit npz paths; their common folder is used as --run")
    ap.add_argument("--data-root", default=None,
                    help="where the built scene goes (<data root>/artifacts/viewers); default "
                         "$SOMA_DATA_ROOT, which must then be set")
    ap.add_argument("--body", default="smpl", choices=["smpl", "smplx"],
                    help="smpl=exact native shape from the subject's betas (default); smplx=addon cross-space approx")
    ap.add_argument("--smpl-model", default=None,
                    help="clean SMPL model npz (--body smpl); default SMPL_MALE_clean.npz in "
                         "$SOMA_BODY_MODEL_DIR, else <data root>/body_models/smpl")
    ap.add_argument("--blender", default=None,
                    help="Blender 4.5 executable; default $BLENDER_EXE, else `blender` on PATH, "
                         "else the platform's usual location")
    ap.add_argument("--enrich", action="store_true", help="force the one-time dev-npz enrichment")
    ap.add_argument("--rebuild", action="store_true", help="force rebuild the scene")
    ap.add_argument("--verify", action="store_true", help="force a fresh build + pixel check")
    ap.add_argument("--no-open", action="store_true", help="build/verify only; do not open the GUI")
    ap.add_argument("--ground-step", type=int, default=2)
    args = ap.parse_args()

    run = (args.run or (os.path.dirname(args.large) if args.large else
           os.path.dirname(args.small) if args.small else
           os.path.dirname(args.dev) if args.dev else None))
    if not run:
        sys.exit("[view] name the run: --run <folder holding the three npz> (or --small / --large / "
                 "--dev). The bundles are under $SOMA_DATA_ROOT/runs/experimental_generation_poc_demo")
    data_root = viewer_paths.data_root(args.data_root, hint="or pass --data-root <dir>")
    model_dir = viewer_paths.body_model_dir(data_root)
    smpl_model_pkl = os.path.join(model_dir, SMPL_MODEL_PKL_NAME)
    smpl_model = args.smpl_model or os.path.join(model_dir, SMPL_MODEL_CLEAN_NAME)
    blender = args.blender or default_blender()
    if not blender or not os.path.isfile(blender):
        sys.exit(f"[view] Blender not found: {blender or '(none)'} (pass --blender or set $BLENDER_EXE)")
    for name in INPUT_NPZ:
        if not os.path.isfile(os.path.join(run, name)):
            sys.exit(f"[view] missing input {name} in {run}")
    dev_path = os.path.join(run, "development_reference.npz")

    env = dict(os.environ)
    if args.body == "smplx":                         # only the approximate path needs the SMPL-X addon
        ures = find_user_resources()
        if ures:
            env["BLENDER_USER_RESOURCES"] = ures
            print("[view] BLENDER_USER_RESOURCES =", ures)
        else:
            print(f"[view] WARNING: SMPL-X addon not found; set BLENDER_USER_RESOURCES to the folder whose "
                  f"'{ADDON_REL}' exists.")

    # 0. one-time enrichment (reads pkl; identity-guarded) — BEFORE content_id so the id reflects it
    if args.enrich or not is_enriched(dev_path):
        print("[view] enriching development_reference.npz (one-time; reads pkl) ...")
        subprocess.run([sys.executable, os.path.join(POC, "smpl_recon.py"), "--run", run, "--enrich"],
                       check=True)
    else:
        print("[view] dev npz already enriched (betas + smpl_trans present)")

    model_clean = ensure_clean_smpl_model(smpl_model_pkl, smpl_model) if args.body == "smpl" else ""

    # content-address AFTER enrichment (input npz bytes + scripts + SMPL model + body + ground_step)
    cid = content_id(run, args.ground_step, args.body, model_clean)
    out_dir = os.path.join(data_root, "artifacts", "viewers", f"prism-viewer-{args.body}-{cid}")
    os.makedirs(out_dir, exist_ok=True)
    anim = os.path.join(out_dir, f"recon_{args.body}_anim.npz")
    blend = os.path.join(out_dir, "small_large_viewer.blend")
    vdir = os.path.join(out_dir, "verify_frames")
    marker = os.path.join(out_dir, ".content_id")
    print(f"[view] body={args.body}  output bundle = {out_dir}")

    cached = (os.path.exists(blend) and os.path.exists(marker)
              and open(marker, encoding="utf-8").read().strip() == cid)
    if args.rebuild or args.verify or not cached:
        print(f"[view] reconstructing (npz-only, body={args.body}) + building + verifying scene ...")
        if os.path.exists(marker):
            os.remove(marker)                       # invalidate the cache until this build is verified
        emit_flag = "--emit-smpl" if args.body == "smpl" else "--emit-smplx"
        subprocess.run([sys.executable, os.path.join(POC, "smpl_recon.py"), "--run", run, emit_flag, anim],
                       check=True)
        build_cmd = [blender, "--background", "--python", os.path.join(POC, "viewer_scene.py"), "--",
                     "--body", args.body, "--anim", anim, "--blend", blend,
                     "--ground-step", str(args.ground_step), "--verify-out", vdir]
        if args.body == "smpl":
            build_cmd += ["--model", model_clean]
        subprocess.run(build_cmd, check=True, env=env)
        # ALWAYS pixel-verify before trusting the cache marker
        subprocess.run([sys.executable, os.path.join(POC, "viewer_verify_pixels.py"), "--dir", vdir],
                       check=True)
        write_manifest(out_dir, run, cid, args.ground_step, args.body, data_root)
        tmp = marker + ".tmp"                        # write the marker LAST, atomically
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(cid)
        os.replace(tmp, marker)
        print("[view] build verified; cache marker written")
    else:
        print("[view] reusing content-addressed verified scene:", blend)

    if not args.no_open:
        print("[view] opening interactive viewer ->", blend)
        subprocess.Popen([blender, blend, "--python", os.path.join(POC, "viewer_open.py")], env=env)
        print("[view] launched. Play: Spacebar | Orbit: MMB-drag or rotate 'OrbitPivot' about Z.")


if __name__ == "__main__":
    main()
