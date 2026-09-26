# Small/Large Interactive Viewer  ·  INTERNAL-ONLY

Interactive Blender viewport of the subject's **body mesh** animated with the **8 IMU positions + RGB
sensor axes**, reconstructed **only** from the distributed `small / large / development_reference` npz.
Play / scrub / orbit — **not** an mp4.

Two body models:

| `--body` | Mesh | Shape fidelity | Needs |
|---|---|---|---|
| **`smpl`** (default) | native **SMPL** (6890 verts) rig | **EXACT** — the subject's own SMPL betas | licensed MPI SMPL model in the body-model folder |
| `smplx` | Meshcapade **SMPL-X** (10475 verts) | **approximate** cross-space | SMPL-X Blender add-on |

## Where the data comes from

The viewer and its tools read:

| What | From |
|---|---|
| bundles (runs) | `$SOMA_DATA_ROOT/runs/experimental_generation_poc_demo/<lineage>/` — bundles stay on the PC that made them (ADR-0041); or a folder named with `--folder <dir>` (repeatable) / `SOMA_VIEWER_DATASET`; or a portable bundle's own `data/runs/` |
| SMPL body models | `SOMA_VIEWER_MODEL_<GENDER>`, the portable bundle's `data/`, then `$SOMA_BODY_MODEL_DIR`, else `$SOMA_DATA_ROOT/body_models/smpl` |
| scenes built by `viewer_launch.py` | `<data root>/artifacts/viewers/` (`--data-root`, else `$SOMA_DATA_ROOT`) |
| `render_amass` / `gear_kit` (the presentation package: **required** by the viewer, the scene builder and the PRISM mesh tools; not in this repository, distributed separately with the portable viewer bundle) | `$SOMA_ANIM_DIR`, or beside the scripts (a portable bundle carries its copies there); missing, the import fails by name. No folder above a checkout is searched, and a file that is not stored on this PC (a Dropbox/OneDrive online-only placeholder: Windows attribute OFFLINE, RECALL_ON_OPEN or RECALL_ON_DATA_ACCESS) is skipped without being opened, since opening one waits for the download |

A tool that needs the data root and is given neither `--data-root` (or its explicit paths) nor
`SOMA_DATA_ROOT` stops and says so.

## Run (one line)

```
python scripts/poc/viewer_launch.py --run <run folder>                 # exact-shape SMPL body (default)
python scripts/poc/viewer_launch.py --run <run folder> --body smplx    # approximate SMPL-X (add-on) fallback
```

`--run <dir>` is required (a folder holding the npz trio; or `--small/--large/--dev`); `--verify` = fresh
build + headless pixel check; `--rebuild` forces regeneration; `--no-open` builds only; `--data-root` says
where the scene is written (default `$SOMA_DATA_ROOT`); `--blender` (default `$BLENDER_EXE`, then
`blender` on `PATH`, then the platform's usual install).
In the viewer: **Spacebar** plays; **middle-mouse-drag** orbits (or rotate `OrbitPivot` about Z);
scrub the timeline — frames **1–1000 = 10 s @ 100 Hz**.

## Portable bundle — interactive app (`.exe`, distributed separately)

The bundle ships Blender + the scripts + the SMPL model + the run data + matplotlib/scipy (installed into
Blender's own Python), and `launch.exe` runs `scripts/viewer_app.py`, which **builds the scene live** and
adds two interactive features (see `viewer_app.py` / `imu_plot.py`):

- **Run selector** — an N-panel (`SOMA` tab) dropdown of the enriched runs found in the bundles under
  `$SOMA_DATA_ROOT` (and any `--folder` / `SOMA_VIEWER_DATASET`, and the bundle's own `data/runs/`);
  picking one rebuilds the SMPL body + IMU animation from that run (npz-only, a few seconds).
- **IMU live plot** — an IMU-site dropdown; the sensor's acceleration / angular-velocity / orientation
  time-series render (matplotlib → in-Blender Image Editor) with a vertical cursor that tracks the current
  frame during playback. (3D-click IMU selection is a planned upgrade; the dropdown is the first version.)

```
SOMA_SmallLarge_Viewer/
  launch.exe / run.bat       # double-click (sets SOMA_VIEWER_DATA, runs scripts/viewer_app.py)
  blender/                   # portable Blender 4.5 (GPL) + numpy/scipy/matplotlib in its Python
  scripts/                   # viewer_app, viewer_scene, viewer_paths, smpl_{rig,model,recon}, imu_plot,
                             #   render_amass, gear_kit, render_prism_mesh_imu (an older bundle also
                             #   carries generate_prism_faithful, which the app no longer imports)
  data/
    SMPL_MALE_clean.npz      # licensed SMPL model (clean-extracted; INTERNAL-ONLY)
    runs/<run-id>/           # enriched small/large/development_reference.npz per run
  README.txt
```

This repository does not build the bundle (~1 GB); it is distributed separately, and it is where the
presentation package comes from. On an internal Windows PC, double-click `launch.exe` — no
Python/Blender/add-on install needed. The viewer lists the bundles under that PC's `SOMA_DATA_ROOT`; set
`SOMA_VIEWER_DATASET` (or run `blender\blender.exe --python scripts\viewer_app.py -- --folder <dir>`) to
look at another folder. `deploy_viewer.ps1 -Source <bundle> -Dest <folder>` refreshes the bundle's scripts
from this checkout and copies it (dry-run unless `-Execute`); there is no default source or destination,
so a deployment to the shared drive is a decision someone writes down as `-Dest`.
**INTERNAL-ONLY**: the body derives from the license-gated SMPL model — distribute only to machines under
the same non-commercial research licence (ADR-0005). A single self-contained `.blend` (baked, viewer-only,
no run-switch/plot) can still be produced by `viewer_launch.py` for a lighter static share.

**What runs inside Blender imports only what Blender's Python has.** That Python carries numpy, and the
bundle installs scipy and matplotlib into it; it has no PyYAML, no protobuf and no `soma_synth` or `smpl18`.
So `viewer_app.py`, `viewer_scene.py`, `viewer_open.py`, `render_prism_mesh_imu.py`,
`blender_prism_anim.py` and `probe_align.py` import only the standard library, numpy, `bpy`/`mathutils`/
`addon_utils` and sibling scripts, and every sibling they reach adds at most scipy and matplotlib. Data
locations come from `viewer_paths.py` (standard library only), never from `soma_synth.pipeline.paths`.
`smpl_recon.py` loads `generate_prism_faithful.py` (which imports `soma_synth` and `smpl18`) only inside
ENRICH and the reconstruction check, which `viewer_launch.py` runs with your own Python; the app uses its
npz-only VIEW helpers. `tests/poc/test_blender_side_imports.py` checks all of this with the AST, and
`deploy_viewer.ps1` checks it once more by importing `viewer_app` with the bundle's Blender.

**The PRISM figure and animation tools never write into a bundle.** `export_anim_data.py` writes
`anim_data.npz` into its `--run` and `blender_prism_anim.py` the mp4 (or `figures/anim_smoke.png`);
`prism_to_smplx.py` writes its `--out`. Before writing anything each
refuses a folder inside a bundle or corpus (it, or a folder above it, holds `INDEX.json`), inside
`runs/experimental_generation_poc_demo`, inside an evidence folder (`_superseded`, `_runs`,
`_manifest_backfill`) or among the sources (`viewer_paths.writable_folder`). Copy the take to a scratch
folder and name the copy.

## The SMPL body model (exact shape)

`--body smpl` needs the MPI SMPL body model (the mesh generator; the PRISM dataset ships shape *parameters*
but not the model). It is license-gated:

1. Register at <https://smpl.is.tue.mpg.de>, accept the non-commercial licence.
2. Download **"SMPL for Python v1.1.0"**, take `basicmodel_m_lbs_10_207_0_v1.1.0.pkl` (male = PRISM subj001).
3. Put it in the body-model folder, `$SOMA_BODY_MODEL_DIR` or `${SOMA_DATA_ROOT}/body_models/smpl/`
   (INTERNAL-ONLY — never in the repository tree).

The launcher clean-extracts it once (`smpl_model.py`, a restricted numpy/chumpy-shim unpickler — no code
from the pickle runs) to `SMPL_MALE_clean.npz`.

## Which datasets open, and on which body

Any collection following the paired contract opens: a run folder is `small_reference.npz` +
`large_reference.npz` + a shape reference (`anthro_reference.npz` or `development_reference.npz`)
carrying `betas` and `gender`, with the root translation in the shape reference, the large reference or
a `smpl_root_translation.npz` sidecar. That is every bundle `soma-synth run` writes under
`$SOMA_DATA_ROOT/runs/experimental_generation_poc_demo/` (`prism_faithful_full`, `amass_faithful_full`,
`gaitex_unified8`, `hknu_unified8`, `addbio_unified8`); the evidence folders beside them (`_runs`,
`_superseded`, `_manifest_backfill`) are not walked. A bundle's `INDEX.json`
drives discovery, and its `subject_id` field groups the Collection dropdown by subject (GAITEX names its
takes `<subject>_<condition>`, so 72 takes list as 19 subjects).

The body is drawn on the SMPL model the subject's betas were fitted in, chosen by `gender`:
`SMPL_MALE_clean.npz`, `SMPL_FEMALE_clean.npz` or `SMPL_NEUTRAL_clean.npz` (GAITEX records no sex, so its
generator declared neutral and fitted on that template; the same betas stand 1.94 m there and 1.64 m on
the male one). A bundle lacking the right model falls back to the male one and prints that the body is not
the subject's. `viewer_app.py -- --run <collection>/<take>` opens a chosen take first (a bare take name is
ambiguous once two collections hold it, e.g. a scratch copy of the bundle next to it).

Headless check of a checkout against the bundle's Blender (SELFTEST_OK is the pass signal; without
`--python-exit-code 1` Blender exits 0 even when the script raises):

```
blender -b --python-exit-code 1 --python scripts/poc/viewer_app.py -- --selftest --run gaitex_unified8/gaitex_austra_ng --out <png>
blender -b --python-exit-code 1 --python scripts/poc/viewer_app.py -- --selftest --folder <a folder of runs>
```

What a channel is worth is written on the plot. The title carries the site's mean `imu_confidence` over
its valid frames and what the manifest says the site is: `vs worn unit` (compared with the worn XSens unit
on that take: GAITEX back_T4, shanks, feet), `declared mount` (mount offset declared zero: GAITEX wrists,
occiput), or a red `NOT source-backed` (a bundle whose channel is FK of a joint the source never observed).
Frames where `imu_valid_mask` is False are NaN in the data and shaded grey in every panel, with the valid
fraction in the title and under the IMU dropdown when it is below 100 % (GAITEX marker gaps: 0.34 % of
cells overall, but `gaitex_austra_ng` `wrist_r` keeps 18 %). Older bundles change only in that their title
gains the `confidence` line too (PRISM 0.4 / 0.7, AMASS 0.4–0.6): the number's meaning is per corpus and
the manifests carry no site lists to qualify it. No shipped manifest carries `unbacked_sites` any more
(the 2026-09-05 GAITEX bundle did); the red note stays for any bundle that declares one.

## Where outputs go (data plane)

Heavy generated artifacts (`.blend`, reconstructed npz, verify PNGs, `manifest.json`) are written to
`<data root>/artifacts/viewers/prism-viewer-<body>-<content_id>/` (`--data-root`, else
`$SOMA_DATA_ROOT`) — **not** into the repo. The bundle is **content-addressed** by the input npz +
generating scripts + the SMPL model + body mode, so a stale scene is never reused. Only the launcher + this README are tracked.

## Reconstruction — two phases

**Enrich (once, reads the PRISM pkl):** add to `development_reference.npz`, all `source_derived` —
`betas`[10] (real SMPL shape), `gender`, and **`smpl_trans`[T,3]** = the **faithful** source SMPL root
translation. (`large["pelvis_position_world_aux"]` is the pelvis *joint* world position and differs from
the SMPL `trans` by up to ~0.1 m, so it is **not** used as the `trans`.)

**View (npz-only):** reconstruct from the three npz alone —
- **pose** = per-joint GLOBAL orientations `large["smpl_global_orientation_prism_world"]` (24×quat);
  the SMPL rig applies them as local rotations (`local[j] = G[parent]ᵀ·G[j]`, root = `G[0]`), **lossless**;
- **trans** = `development_reference["smpl_trans"]` (faithful), on the pelvis bone;
- **shape** = `development_reference["betas"]` → the SMPL forward's exact `v_shaped` rest mesh.

All three `pair_id` values are validated; the pkl is never read during viewing.

### Native-SMPL rig (`smpl_rig.py`, exact path)

The subject betas produce the **exact** SMPL rest mesh; it is skinned to a 24-bone SMPL armature (axis-
identity rest bones, so a pose-bone local quaternion equals the SMPL local rotation) and posed by the
lossless rotations. Bone names are SMPL-standard, so `render_amass.body_frame` / `ground_lock` and
`render_prism_mesh_imu.add_imu_axes` are reused **unchanged**. Pose blend shapes (`posedirs`) are the one
omission (Blender LBS has no equivalent) — **shape is exact**, only pose-dependent skin correctives are not
applied.

## Verification

| Check | Result |
|---|---|
| SMPL forward, identity (betas=0, pose=I) → `v_template` | max `4.9e-8` |
| Skinning weights partition of unity | max dev `4.5e-8` |
| Betas deform the body (real male) | neutral 1.79 m → subject **1.63 m** stature |
| Rig FK vs SMPL global rotations (in Blender) | max **0.02°** (float32 armature FK) |
| Bone lengths pose-invariant (rigid FK) | std `3e-16` m |
| Feet ground-locked | mesh world z-range `[0.00, 1.61]` m |
| Scene structure | exactly 1 body mesh + 8 IMU pivots + 24 axis shafts + 24 heads |
| Headless start/mid/end + pixel check | skin-tone mesh (separate from axes) + R/G/B axes → `all_pass` |

Tests on synthetic data: `tests/poc/test_smpl_model.py` (forward math), `tests/poc/test_smpl_recon.py` (inverse-FK),
`tests/poc/test_viewer_paths.py` (where data, models and runs come from; no hardcoded location),
`tests/poc/test_blender_side_imports.py` (what runs inside Blender imports only what its Python has),
`tests/poc/test_prism_tools_require_inputs.py` (the PRISM tools name their inputs and check them first).
Regenerate the scene: `python scripts/poc/viewer_launch.py --run <run folder> --verify --no-open`.

## What it shows

- Subject **SMPL male** mesh (6890 verts) — **exact real shape** from the subject's betas (`--body smpl`).
- 8 IMU sites strapped to the skin — `chest`, `head`, `wrist_l/r`, `shank_l/r`, `insole_l/r` — each with
  **X = red, Y = green, Z = blue** spec axes (`00_sensors.qmd`; foot frames **side-aware**).
- Orbit camera that follows the walking subject; feet ground-locked to the floor.

## Limitations

- **`--body smpl` (default): shape is EXACT**; the only omission is pose blend shapes (`posedirs`) — a
  small pose-dependent skin correction, not a shape error.
- **`--body smplx`: shape is APPROXIMATE** — SMPL `betas` placed into the SMPL-X shape space (different PCA
  basis; the families are not interchangeable). Kept only as an add-on-based fallback.
- RGB axes are the **spec mounting frame** rigidly strapped to the body bone (raycast surface placement),
  not the `small`-npz measured quaternion. Coordinates = PRISM world (Z-up) = spec `G`.
- `experimental_non_candidate`, INTERNAL-ONLY (see the bundle `manifest.json`).

## Files

- **Tracked (repo):** `scripts/poc/{smpl_recon,smpl_model,smpl_rig,viewer_scene,viewer_app,viewer_paths,imu_plot,viewer_verify_pixels,viewer_open,viewer_launch}.py`,
  `scripts/poc/deploy_viewer.ps1`, `tests/poc/{test_smpl_model,test_smpl_recon,test_viewer_paths,test_blender_side_imports,test_prism_tools_require_inputs}.py`,
  `docs/guides/small_large_viewer.md` (this file). `viewer_app.py` = the in-Blender interactive app (run
  selector + IMU live plot); `viewer_paths.py` = where data, models and the presentation package
  are found (no bpy); `imu_plot.py` = the headless/Blender-shared matplotlib IMU plotter.
- **Generated (under the data root):** `artifacts/viewers/prism-viewer-<body>-<cid>/{small_large_viewer.blend, recon_<body>_anim.npz, verify_frames/, manifest.json, .content_id}`;
  `body_models/smpl/{basicmodel_m_*.pkl (licensed), SMPL_MALE_clean.npz}`.
