"""Interactive SOMA Small/Large viewer — runs INSIDE Blender.

Two interactive features on top of the exact-shape SMPL body + 8 IMU RGB-axes scene:
  1. **Run selector** — pick a dataset run from a dropdown; the SMPL body + IMU animation rebuild from
     that run's small/large + shape-reference npz (npz-only; a few seconds).
  2. **IMU live plot** — pick an IMU site (dropdown); its acceleration / angular-velocity / orientation
     time-series show in an in-Blender Image Editor with a vertical cursor that tracks the current frame.
     (3D-click selection of an IMU is a planned upgrade; the dropdown is the reliable first version.)

Headless self-test:  blender -b --python-exit-code 1 --python viewer_app.py -- --selftest [--run <collection>/<take>] [--folder <dir>] --out <png>
                     (SELFTEST_OK is the pass signal; without --python-exit-code Blender exits 0 on a raise)
GUI:                 blender --python viewer_app.py [-- --folder <dir>]

Runs come from the bundles under $SOMA_DATA_ROOT/runs/experimental_generation_poc_demo, from a
folder given with --folder (repeatable) or $SOMA_VIEWER_DATASET, and from a portable bundle's own
data/runs (viewer_paths.run_roots).

numpy + matplotlib run in Blender's own Python. INTERNAL-ONLY.
"""
import json
import os
import sys
import tempfile
import time

import numpy as np

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import viewer_paths                          # noqa: E402
ANIM_DIR = viewer_paths.add_to_sys_path()    # render_amass / gear_kit (viewer_paths.anim_dir)

import render_amass as ra                    # noqa: E402
import render_prism_mesh_imu as rmi          # noqa: E402
import smpl_rig                              # noqa: E402
import smpl_model                            # noqa: E402
import smpl_recon                            # noqa: E402
import viewer_scene                          # noqa: E402  (setup_orbit_camera)
import imu_plot                              # noqa: E402


# --------------------------------------------------------------------------- data layout
# The bundles live on the local machine, under
# $SOMA_DATA_ROOT/runs/experimental_generation_poc_demo/<lineage> (ADR-0041: bundles stay on the
# machine that made them). The viewer lists those, any
# folder named with --folder or $SOMA_VIEWER_DATASET first, and a portable bundle's own data/runs
# last (viewer_paths.run_roots).
BUNDLE_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))    # scripts/ sits inside the bundle
_PAIR_NPZ = ("small_reference.npz", "large_reference.npz")
_MAX_SCAN_DEPTH = 4          # a folder may be a network mount: never walk it unbounded
_EXPLICIT_ROOTS = []         # --folder <dir>, in the order given (main)


def run_roots():
    """Existing folders to scan for runs, highest priority first, de-duplicated."""
    return viewer_paths.run_roots(BUNDLE_ROOT, HERE, _EXPLICIT_ROOTS)


MODEL_FILES = {"male": "SMPL_MALE_clean.npz", "female": "SMPL_FEMALE_clean.npz",
               "neutral": "SMPL_NEUTRAL_clean.npz"}


def model_gender(gender):
    """Which of the three published SMPL models a subject's betas belong to.

    'female' for anything starting f/F, 'neutral' for n/N, else 'male'. Neutral is a template of its
    own, not a midpoint: GAITEX records no sex, so its generator fitted the betas on the neutral
    model, and only that model reproduces the fitted segment lengths — on gaitex_austra_gwo the same
    betas stand 1.94 m on the neutral template and 1.64 m on the male one.
    """
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8", "replace")
    g = str(gender).strip().lower()
    if g in ("female", "f"):
        return "female"
    if g in ("neutral", "n"):
        return "neutral"
    return "male"                       # including "", "None" and anything unrecorded, as before


def _resolve_model(gender="male"):
    """The licensed SMPL model for a subject's gender ("" if this machine carries none).

    The model ships WITH the viewer (it is not dataset content), and it is chosen per subject rather
    than fixed: betas live in the shape space of THEIR OWN gendered model, so posing a woman's betas
    on the male template does not approximate her body, it draws a different one. That used to be
    invisible because PRISM was male-only; the regenerated collections are not — AMASS and PRISM
    both include female subjects, and every GAITEX take is neutral.
    """
    g = model_gender(gender)
    for c in viewer_paths.model_candidates(MODEL_FILES[g], g, BUNDLE_ROOT, HERE):
        if os.path.isfile(c):
            return os.path.abspath(c)
    return ""


MODEL_NPZ = _resolve_model()          # male — the model a bundle is expected to carry at minimum


def fallback_model():
    """The model a subject is drawn on when its own is not here: the first of the three present.

    A bundle need not carry the male model at all — a GAITEX-only share is neutral-only — so the
    fallback is whatever exists, not the male file.
    """
    for g in MODEL_FILES:
        path = _resolve_model(g)
        if path:
            return path
    return ""


def model_for_gender(gender):
    """(model npz to draw this subject with, "" or why it is not the right one).

    A fallback is always reported: a bundle may legitimately ship the male model alone, and the
    woman then drawn on it is not the subject whose betas produced her.
    """
    want = model_gender(gender)
    path = _resolve_model(want)
    if path:
        return path, ""
    other = fallback_model()
    if not other:
        raise SystemExit(f"[viewer_app] no SMPL model npz found (need {MODEL_FILES[want]} in "
                         "$SOMA_BODY_MODEL_DIR, $SOMA_DATA_ROOT/body_models/smpl or the bundle's "
                         "data/)")
    return other, (f"no {want} SMPL model on this machine — drawn on "
                   f"{os.path.basename(other)}, so the BODY SHAPE IS NOT THIS SUBJECT'S")


_VERDICT = {}                       # "run_dir|mtime" -> viewable?, so a rescan re-opens nothing
_VERDICT_DIRTY = [False]
SCAN_CACHE = os.path.join(tempfile.gettempdir(), "soma_viewer_scan_cache.json")
SCAN_CACHE_VERSION = 2              # bump whenever _is_enriched changes what counts as viewable


def _load_scan_cache():
    """Carry verdicts across launches — the first scan of a big collection is the expensive one.

    Keys embed the folder's mtime, so a run that changes is re-examined. The file also carries a
    version: when the viewability RULE changes, every stored verdict is wrong regardless of mtime
    (that is exactly how the AMASS collection would have stayed hidden after it became loadable), so
    a version bump throws the whole cache away.
    """
    try:
        with open(SCAN_CACHE, encoding="utf-8") as fh:
            blob = json.load(fh)
        if isinstance(blob, dict) and blob.get("version") == SCAN_CACHE_VERSION:
            _VERDICT.update(blob.get("verdicts") or {})
    except Exception:
        pass


def _save_scan_cache():
    if not _VERDICT_DIRTY[0]:
        return
    try:
        tmp = SCAN_CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"version": SCAN_CACHE_VERSION, "verdicts": _VERDICT}, fh)
        os.replace(tmp, SCAN_CACHE)
        _VERDICT_DIRTY[0] = False
    except Exception as exc:
        print("[viewer_app] scan cache not saved:", exc)


def _is_enriched(run_dir):
    """A run is viewable only with the SMPL shape AND a faithful root translation.

    betas/gender come from the shape reference; the translation from that file, the large reference,
    a sidecar, or a sibling SMPL-X npz. Opening npz is what makes a scan expensive, so the checks
    that need only a directory listing run first and every verdict is memoised against the folder's
    mtime — the AMASS collection alone is 8 k unviewable runs.
    """
    try:
        key = f"{run_dir}|{os.path.getmtime(run_dir)}"
    except OSError:
        return False
    if key in _VERDICT:                 # a rescan must never re-open the same npz
        return _VERDICT[key]

    verdict = False
    try:
        path = smpl_recon.shape_reference_path(run_dir)
        large = os.path.join(run_dir, "large_reference.npz")
        if path and os.path.isfile(large):
            z = np.load(path, allow_pickle=True)
            if "betas" in z.files and "gender" in z.files:
                # A run needs a body (betas/gender) and per-joint rotations. The root translation is
                # NOT required: without it the take still shows correctly, in place.
                verdict = bool(smpl_recon.global_rotation_key(np.load(large, allow_pickle=True)))
    except Exception:
        verdict = False

    _VERDICT[key] = verdict
    _VERDICT_DIRTY[0] = True
    return verdict


def run_key(coll_dir, run_name):
    """The identifier a run is listed under: its collection folder plus its own name.

    Two collections can hold the same take ids (e.g. a faithful and a measured PRISM collection),
    so a bare name would make one collection shadow the other entirely. The collection qualifies
    it; the dropdown still shows the short name.
    """
    return f"{os.path.basename(coll_dir.rstrip(os.sep))}/{run_name}"


def key_for_dir(run_dir):
    """The listing key for a run folder on disk."""
    run_dir = run_dir.rstrip(os.sep)
    return run_key(os.path.dirname(run_dir), os.path.basename(run_dir))


def run_name_of(key):
    """The short run name shown in the dropdown."""
    return key.rsplit("/", 1)[-1]


_SUBJECT_OF = {}                    # run key -> "<collection>/<subject_id>" when its catalog says so


def run_group_of(key):
    """The collection a run belongs to: its dataset folder plus the subject the take came from.

    A catalog that names each take's `subject_id` settles it (GAITEX: `<collection>/gaitex_austra`
    holds that subject's four conditions). Otherwise the take name is cut at its take marker:
    `<collection>/prism-subj001-take002` -> `<collection>/prism-subj001`, and
    `<collection>/amass_BMLmovi_Subject_1_F_MoSh__Subject_1_F_1_poses` ->
    `<collection>/amass_BMLmovi_Subject_1_F_MoSh`.
    A run matching no pattern becomes its own group, so nothing is ever hidden from the list.
    """
    if key in _SUBJECT_OF:
        return _SUBJECT_OF[key]
    prefix, name = ("", key) if "/" not in key else key.rsplit("/", 1)
    for sep in ("__", "-take", "_take"):
        i = name.rfind(sep)
        if i > 0:
            name = name[:i]
            break
    return f"{prefix}/{name}" if prefix else name


CATALOG = "INDEX.json"

# A catalog names each run's folder in whichever field its generator used, and four spellings have
# shipped. Reading only the two we knew about cost the fast path silently: the regenerated
# faithful-v2 catalogs name folders `relative_path` / `take_id`, not one entry resolved, and
# discovery fell back to the very walk the catalog exists to avoid (minutes on a local disk, far
# longer on a network share) with nothing on screen to say why. Read every spelling instead of
# tracking the latest.
TAKE_NAME_FIELDS = ("out_dir", "relative_path", "take", "take_id")


def _take_folder(entry):
    """The run folder a catalog entry points at, or "" when it names none."""
    for field in TAKE_NAME_FIELDS:
        value = entry.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _take_subject(entry):
    """The subject a catalog entry attributes its take to, or "" when it says nothing."""
    for field in ("subject_id", "subject"):
        value = entry.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _runs_from_catalog(coll_dir):
    """[(run folder name, subject id or "")] listed by a collection's published catalog, or None if it
    has no usable one.

    Walking a published collection means stat-ing and opening npz for thousands of folders; over a
    network drive that can be a freeze of several minutes before the first frame. The dataset ships
    an INDEX.json next to those folders that answers the same question in one read, so it is used
    when present and the walk stays as the fallback.
    """
    path = os.path.join(coll_dir, CATALOG)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            takes = json.load(fh).get("takes")
    except Exception as exc:
        print(f"[viewer_app] catalog unreadable, walking instead ({path}): {exc}")
        return None
    if not isinstance(takes, list):
        return None
    names = []
    for t in takes:
        if not isinstance(t, dict) or t.get("status") not in (None, "ok"):
            continue
        name = _take_folder(t)
        if name:
            names.append((name, _take_subject(t)))
    if not names and takes:
        # The catalog listed runs and none of them named a folder we understand. That is a schema the
        # viewer has not been taught, and it is worth a line: the walk below still finds everything,
        # so the only symptom otherwise is that opening the panel got slow for no visible reason.
        keys = sorted({k for t in takes[:5] if isinstance(t, dict) for k in t})
        print(f"[viewer_app] {path}: no entry names a folder via {TAKE_NAME_FIELDS} "
              f"(entries carry {keys}) — falling back to a full walk of this collection")
    return names or None


def gather_runs():
    """Every ENRICHED run folder across run_roots() → sorted [(id, dir)]; first root wins on id ties.

    A run folder is one holding small+large reference npz plus a shape reference (anthro or
    development). Un-enriched runs (no betas) can't be viewed without the licensed pkl, so they are
    excluded from the dropdown rather than offered and then failing on select.
    """
    found, seen, subjects = [], set(), {}
    for root in run_roots():
        base = root.rstrip("\\/").count(os.sep)
        for cur, dirs, files in os.walk(root):
            # evidence and run records beside the lineage folders hold no run to view
            dirs[:] = [d for d in dirs if d not in viewer_paths.EVIDENCE_FOLDERS]
            if CATALOG in files:
                listed = _runs_from_catalog(cur)
                if listed is not None:
                    dirs[:] = []                  # the catalog replaces the walk for this collection
                    present = set(os.listdir(cur))
                    for name, subject in listed:
                        rel = name.replace("\\", "/").strip("/")
                        key = run_key(cur, rel)
                        if key in seen:
                            continue
                        run_dir = os.path.join(cur, *rel.split("/"))
                        # A plain folder name is answered from the one listing above; only a nested
                        # relative_path costs a stat, and no shipped catalog uses one yet.
                        here = rel in present if "/" not in rel else os.path.isdir(run_dir)
                        if here:
                            seen.add(key)
                            found.append((key, run_dir))
                            if subject:
                                subjects[key] = run_key(cur, subject)
                    continue
            if cur.count(os.sep) - base >= _MAX_SCAN_DEPTH:
                dirs[:] = []
            if all(f in files for f in _PAIR_NPZ) and _is_enriched(cur):
                dirs[:] = []                      # a run folder never nests another run
                key = run_key(os.path.dirname(cur), os.path.basename(cur))
                if key not in seen:
                    seen.add(key)
                    found.append((key, cur))
    # Collections named with a leading underscore (_dryrun_*, _superseded) are scratch next to the
    # real ones; they stay listed but sort last, so startup and the picker head land on a real one.
    found.sort(key=lambda kd: (kd[0].startswith("_"), kd[0]))
    _SUBJECT_OF.clear()                       # a scan replaces the map, so a dropped field or a
    _SUBJECT_OF.update(subjects)              # vanished collection does not keep grouping runs
    return found


_RUNS = []                          # cached [(id, dir)] — kept alive for the EnumProperty callback
_ENUM_ITEMS = [("NONE", "(no runs found)", "")]   # ditto: Blender frees enum strings it does not own
_GROUP_ITEMS = [("NONE", "(no runs found)", "")]
_SCAN = {"t": 0.0}
RESCAN_SEC = 5.0                    # the dropdown callback fires on every redraw — don't re-walk a
                                    # network drive that often; 5 s keeps "open the list" feeling live
GROUND_SAMPLES = 1000               # cap on ground_lock mesh evaluations, whatever the take length
_STATE = {"fig": None, "axes": None, "img": None, "data": None, "run_dir": None, "last": 0.0}


def refresh_runs(force=False):
    """Re-scan the roots for runs (throttled unless forced) and return the current [(id, dir)].

    Called every time the run dropdown is drawn, so a run copied into the dataset folder while the
    viewer is open shows up without restarting Blender.
    """
    global _RUNS
    now = time.time()
    if not force and _RUNS and (now - _SCAN["t"]) < RESCAN_SEC:
        return _RUNS
    _SCAN["t"] = now
    # runs already proven unopenable stay out of every list, count and group from here on
    _RUNS = [(rid, d) for rid, d in gather_runs() if rid not in _BAD_RUNS]
    _save_scan_cache()
    return _RUNS


# --------------------------------------------------------------------------- scene build
def build_scene_for_run(run_dir, ground_step=4):
    """Clear + rebuild the exact-shape SMPL body + 8 IMU RGB-axes scene from one run (npz-only).

    Every read that can fail happens BEFORE the scene is cleared. A run whose npz are missing or not
    yet synced then leaves the previous scene untouched, instead of wiping it and failing half-built.
    """
    d = smpl_recon.smpl_inputs_from_run(run_dir)       # reads small/large/anthro (+ the translation)
    imu = imu_plot.load_imu(run_dir)                   # reads small again, for the plot
    model_npz, note = model_for_gender(d["gender"])    # raises, by name, when no model at all is here
    if note:
        print(f"[viewer_app] {note}")
    model = smpl_model.load_model_npz(model_npz)

    ra.clear_scene()                                   # from here on the run is known to be readable
    arm, mesh = smpl_rig.build_smpl_rig(model, d["betas"], d["quats"], d["trans"], fps=d["fps"], verify=False)
    scene = bpy.context.scene
    # ground_lock evaluates the mesh once per sampled frame and interpolates between samples, so its
    # cost is set by the sample COUNT, not the take length. Full PRISM takes run to ~13 k frames; hold
    # the count at GROUND_SAMPLES so a long take costs the same as a short one. At 100 Hz that is still
    # ~7 Hz of floor-contact detail, well above gait frequency.
    step = max(ground_step, -(-d["quats"].shape[0] // GROUND_SAMPLES))
    ra.ground_lock(arm, mesh, scene, step=step, mode="lock", win=24)
    ra.setup_world(scene)
    ra.setup_floor(0.20)
    ra.body_material(mesh, ra.SKIN)
    rmi.add_imu_axes(arm, mesh, scene)
    pelvis, _, _ = ra.body_frame(arm, scene, (scene.frame_start + scene.frame_end) // 2)
    ra.setup_lighting(Vector((pelvis.x, pelvis.y, 0.9)))
    viewer_scene.setup_orbit_camera(scene, arm)
    _STATE["data"] = imu
    _STATE["run_dir"] = run_dir
    _STATE["run_id"] = key_for_dir(run_dir)
    _STATE["trans_source"] = d["trans_source"]
    _STATE["rotation_key"] = d["rotation_key"]
    _STATE["body_model"] = model_npz
    _STATE["body_model_note"] = note
    print(f"[viewer_app] rotations '{d['rotation_key']}' · root {d['trans_source']}")
    print(f"[viewer_app] subject {d['gender']} · body {os.path.basename(model_npz)}")
    if hasattr(scene, "soma_viewer"):                 # re-apply toggles after a run-switch rebuild
        _apply_visibility()
    print(f"[viewer_app] scene built for run: {os.path.basename(run_dir)}")
    return scene, arm, mesh


# A catalog lists what the dataset PUBLISHED, which is not the same as what this machine can read
# right now — on a streaming share the folder exists long before its npz do. Rather than pay a scan
# to pre-verify thousands of runs, a run proves itself the first time it is opened
# and a failure drops it from the list for the rest of the session.
_BAD_RUNS = {}                      # run id -> why it could not be opened


def try_build_run(run_dir, ground_step=4):
    """Build the scene for a run. Returns "" on success, else the reason it could not be opened.

    `SystemExit` is caught explicitly: the readers signal a malformed or missing run with it, and it
    derives from BaseException, so `except Exception` would let it straight through and end Blender.
    """
    rid = key_for_dir(run_dir)
    try:
        build_scene_for_run(run_dir, ground_step=ground_step)
    except (Exception, SystemExit) as exc:
        why = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}" if str(exc) else type(exc).__name__
        _BAD_RUNS[rid] = why
        print(f"[viewer_app] cannot open {rid}: {why}")
        return why
    _BAD_RUNS.pop(rid, None)
    _STATE["load_error"] = ""
    return ""


def _startup_candidates(runs, limit=25):
    """Indices to try at startup: the head, plus a spread over the rest.

    Runs sort by id, so the head is all one collection. If that collection happens to be the one that
    has not synced, trying only the head would give up while every other collection was fine.
    """
    n = len(runs)
    order = list(range(min(limit, n)))
    if n > limit:
        order += list(range(0, n, max(1, n // limit)))
    seen, out = set(), []
    for i in order:
        if i < n and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def first_buildable(candidates, ground_step=4, limit=25):
    """Open the first candidate that actually loads, skipping (and remembering) the ones that do not.

    A partially synced share used to abort startup outright, because the very first run was built
    unconditionally and its failure propagated out of main().
    """
    for i in _startup_candidates(candidates, limit):
        rid, run_dir = candidates[i]
        if try_build_run(run_dir, ground_step=ground_step) == "":
            return rid, run_dir
    return None, None


def prefer_run(runs, want):
    """(the runs `want` names, the rest) — by full id or folder on disk first, short take name after.

    `--run` at startup opens a chosen take first instead of whatever sorts first across every root;
    the headless selftest uses it to prove a specific dataset opens rather than some other one. A
    short name is ambiguous whenever two collections hold the same take (the other one may sort
    first and be opened instead), so an exact id or path wins.
    """
    want_dir = os.path.normcase(os.path.abspath(want)) if os.path.isdir(want) else None
    exact, by_name, rest = [], [], []
    for rid, run_dir in runs:
        if rid == want or (want_dir and os.path.normcase(os.path.abspath(run_dir)) == want_dir):
            exact.append((rid, run_dir))
        elif run_name_of(rid) == want:
            by_name.append((rid, run_dir))
        else:
            rest.append((rid, run_dir))
    return exact + by_name, rest


# --------------------------------------------------------------------------- live plot
def init_plot(width_px=880, height_px=1120):
    fig, axes = imu_plot.make_figure(width_px, height_px)
    _STATE["fig"], _STATE["axes"] = fig, axes
    img = bpy.data.images.get("IMU_PLOT")
    if img is None:
        img = bpy.data.images.new("IMU_PLOT", width_px, height_px, alpha=True)
    _STATE["img"] = img


PLOT_HZ_DEFAULT = 20                # redraws/s. Raised from 4 once switching runs stopped stalling:
                                    # the plot cursor now tracks playback closely, and the slider is
                                    # still there to trade it back for 3D smoothness on a slow PC.


def _plot_interval():
    """Seconds between automatic plot redraws, or None when the user turned them off (0 Hz).

    Redrawing the matplotlib figure is the expensive part of playback, so this is the knob that trades
    plot liveness for 3D smoothness.
    """
    props = getattr(bpy.context.scene, "soma_viewer", None)
    hz = int(getattr(props, "plot_update_hz", PLOT_HZ_DEFAULT)) if props else PLOT_HZ_DEFAULT
    return (1.0 / hz) if hz > 0 else None


def refresh_plot(force=False):
    st = _STATE
    if st["fig"] is None or st["img"] is None or st["data"] is None:
        return
    now = time.time()
    if not force:
        interval = _plot_interval()
        if interval is None or (now - st["last"]) < interval:
            return
    st["last"] = now
    site = bpy.context.scene.soma_viewer.imu_site
    imu_plot.draw(st["fig"], st["axes"], st["data"], site, bpy.context.scene.frame_current)
    rgba, w, h = imu_plot.figure_to_rgba_float(st["fig"])
    if tuple(st["img"].size) != (w, h):
        st["img"].scale(w, h)
    st["img"].pixels.foreach_set(rgba.ravel())
    st["img"].update()


def _on_frame(scene, depsgraph=None):
    refresh_plot(force=False)


# --------------------------------------------------------------------------- UI properties / panel
MAX_LISTED = 200                    # a Blender enum with thousands of entries is unusable anyway
_CURRENT_GROUP = [""]               # mirror of run_group; see _group_items for why it must exist


def _group_items(self, context):
    """The collection dropdown: one entry per group, with how many runs it holds.

    Find narrows this list too — with 255 collections across the AMASS subjects, a filter that only
    reached the run list would still leave hundreds of names to scroll. The currently selected group
    is always kept present so the enum value never becomes invalid mid-edit.
    """
    try:
        refresh_runs()
    except Exception as exc:
        print("[viewer_app] run rescan skipped:", exc)
    global _GROUP_ITEMS
    needle = (getattr(self, "run_filter", "") or "").strip().lower()
    # NEVER read self.run_group here: resolving an EnumProperty runs its own items callback, so
    # touching it from inside that callback recurses until the stack blows. The selection is
    # mirrored into _CURRENT_GROUP by the update handler instead.
    current = _CURRENT_GROUP[0]
    counts, matched = {}, set()
    for rid, _d in _RUNS:
        g = run_group_of(rid)
        counts[g] = counts.get(g, 0) + 1
        if needle and (needle in rid.lower() or needle in g.lower()):
            matched.add(g)
    keep = [g for g in sorted(counts) if not needle or g in matched or g == current]
    _GROUP_ITEMS = ([(g, f"{g}  ({counts[g]})", f"{counts[g]} run(s)") for g in keep]
                    or [("NONE", "(no runs found)", "")])
    return _GROUP_ITEMS


def _runs_in_group(props):
    """Runs of the selected group, narrowed by the filter box. Returns (shown, total)."""
    group = getattr(props, "run_group", "") or ""
    needle = (getattr(props, "run_filter", "") or "").strip().lower()
    picked = [(rid, d) for rid, d in _RUNS
              if not group or group == "NONE" or run_group_of(rid) == group]
    # Typing a collection name is how you jump to it, so it must not then empty that collection's
    # own run list; the filter only applies to run names once the group itself already matches.
    if needle and needle not in group.lower():
        picked = [(rid, d) for rid, d in picked if needle in rid.lower()]
    return picked[:MAX_LISTED], len(picked)


def _run_items(self, context):
    global _ENUM_ITEMS
    try:
        refresh_runs()              # opening the dropdown re-scans the dataset folder
    except Exception as exc:        # a shared drive can blink; keep the last good list rather than
        print("[viewer_app] run rescan skipped:", exc)   # breaking the panel mid-redraw
    shown, _total = _runs_in_group(self)
    # identifier stays the unique key; the label is the short name, since the collection is already
    # named by the dropdown above
    _ENUM_ITEMS = [(rid, run_name_of(rid), d) for rid, d in shown] or [("NONE", "(no match)", "")]
    return _ENUM_ITEMS


_SUPPRESS_REBUILD = [False]         # set while syncing the enums to an already-built scene


def _on_group_change(self, context):
    """Selecting a collection re-points the run list; keep the scene on its current run if it fits."""
    _CURRENT_GROUP[0] = self.run_group
    shown, _total = _runs_in_group(self)
    ids = [rid for rid, _d in shown]
    if ids and self.run not in ids:
        self.run = ids[0]           # assignment fires _on_run_change, which rebuilds the scene


def _on_run_change(self, context):
    if _SUPPRESS_REBUILD[0]:
        return
    run_dir = dict(_RUNS).get(self.run)
    if not run_dir:
        return
    why = try_build_run(run_dir)
    if why:
        # the scene is untouched (build_scene_for_run reads before it clears), so put the selection
        # back on the run that is actually displayed rather than leaving it on one that will not open
        _STATE["load_error"] = f"{self.run}: {why}"
        refresh_runs(force=True)                       # drop the bad run from the picker
        back = _STATE.get("run_id")
        if back and back != self.run:
            _SUPPRESS_REBUILD[0] = True
            try:
                _CURRENT_GROUP[0] = run_group_of(back)
                self.run_group = _CURRENT_GROUP[0]
                self.run = back
            finally:
                _SUPPRESS_REBUILD[0] = False
        return
    refresh_plot(force=True)


def _on_site_change(self, context):
    refresh_plot(force=True)


def _on_fps_change(self, context):
    context.scene.render.fps = int(self.playback_fps)   # playback speed = scene frame rate (data is 100 Hz)


def _on_plot_hz_change(self, context):
    refresh_plot(force=True)        # redraw once so the new rate is visible even while paused


def _apply_visibility():
    """Show/hide the SMPL skeleton (armature) and the IMU-axis objects per the panel toggles.
    hide_viewport is display-only — the mesh still deforms and the IMU straps still follow."""
    p = bpy.context.scene.soma_viewer
    for o in bpy.data.objects:
        if o.type == "ARMATURE":
            o.hide_viewport = not p.show_skeleton
        elif o.name.startswith(("ax_", "imupiv_", "imudot_")):
            o.hide_viewport = not p.show_imu_axes


def _on_visibility_change(self, context):
    _apply_visibility()


class SOMAViewerProps(bpy.types.PropertyGroup):
    run_group: bpy.props.EnumProperty(
        name="Collection", items=_group_items, update=_on_group_change,
        description="Subject / capture collection. Picking one narrows the run list below")
    run_filter: bpy.props.StringProperty(
        name="Find", default="", options={"TEXTEDIT_UPDATE"},
        description="Show only runs whose name contains this text")
    run: bpy.props.EnumProperty(name="Run", items=_run_items, update=_on_run_change)
    imu_site: bpy.props.EnumProperty(
        name="IMU", items=[(s, s, f"IMU sensor {s}") for s in imu_plot.SITES],
        default="back_T4", update=_on_site_change)
    playback_fps: bpy.props.IntProperty(
        name="Speed (fps)", default=100, min=5, max=240, update=_on_fps_change,
        description="Playback frame rate. Data is 100 Hz, so 100 = real time; lower = slower, higher = faster")
    plot_update_hz: bpy.props.IntProperty(
        name="Plot update (Hz)", default=PLOT_HZ_DEFAULT, min=0, max=30, update=_on_plot_hz_change,
        description="How often the IMU plot redraws during playback. Redrawing it is the expensive "
                    "part, so lower this (or set 0 = only on pause/scrub) for a smoother 3D view")
    show_imu_axes: bpy.props.BoolProperty(name="IMU axes", default=True, update=_on_visibility_change,
                                          description="Show/hide the 8 IMU RGB axis arrows")
    show_skeleton: bpy.props.BoolProperty(name="Skeleton", default=False, update=_on_visibility_change,
                                          description="Show/hide the SMPL joint skeleton (armature bones)")


class SOMA_OT_reload_run(bpy.types.Operator):
    bl_idname = "soma.reload_run"
    bl_label = "Reload run"
    bl_description = "Rebuild the body + IMU animation from the selected run"

    def execute(self, context):
        _on_run_change(context.scene.soma_viewer, context)
        return {"FINISHED"}


class SOMA_OT_rescan_runs(bpy.types.Operator):
    bl_idname = "soma.rescan_runs"
    bl_label = "Rescan dataset"
    bl_description = ("Look for runs in the dataset folder again. The list also re-scans by itself "
                      "whenever you open the dropdown")

    def execute(self, context):
        runs = refresh_runs(force=True)
        roots = run_roots()
        self.report({"INFO"}, f"SOMA: {len(runs)} run(s) in {len(roots)} folder(s)"
                              + (f" — {roots[0]}" if roots else ""))
        return {"FINISHED"}


class SOMA_PT_viewer(bpy.types.Panel):
    bl_label = "SOMA Small/Large Viewer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SOMA"

    def draw(self, context):
        p = context.scene.soma_viewer
        col = self.layout.column(align=True)
        col.label(text=f"Dataset run:  ({len(_RUNS)} found)")
        col.prop(p, "run_group", text="")
        col.prop(p, "run_filter", text="", icon="VIEWZOOM")
        col.prop(p, "run", text="")
        shown, total = _runs_in_group(p)
        if total > len(shown):
            col.label(text=f"showing {len(shown)} of {total} — narrow with Find", icon="INFO")
        if _STATE.get("trans_source") == smpl_recon.IN_PLACE_TRANSLATION:
            col.label(text="no root translation — playing in place", icon="ERROR")
        if _STATE.get("load_error"):
            col.label(text="could not open that run (see console)", icon="CANCEL")
            if _BAD_RUNS:
                col.label(text=f"{len(_BAD_RUNS)} run(s) hidden — not readable here")
        row = col.row(align=True)
        row.operator("soma.reload_run", icon="FILE_REFRESH")
        row.operator("soma.rescan_runs", icon="VIEWZOOM")
        col.separator()
        col.label(text="IMU live plot:")
        col.prop(p, "imu_site", text="")
        data = _STATE.get("data") or {}
        if p.imu_site in (data.get("unbacked") or ()):
            col.label(text="not source-backed in this dataset", icon="ERROR")
        if data and p.imu_site in data.get("codes", ()):
            frac = imu_plot.valid_fraction(data, p.imu_site)
            if frac < 1.0:
                col.label(text=f"signal on {100 * frac:.0f}% of frames (rest masked)", icon="INFO")
        col.separator()
        col.label(text="Display:")
        row = col.row(align=True)
        row.prop(p, "show_imu_axes", toggle=True)
        row.prop(p, "show_skeleton", toggle=True)
        col.separator()
        col.label(text="Playback:")
        col.prop(p, "playback_fps")
        col.prop(p, "plot_update_hz")
        if p.plot_update_hz == 0:
            col.label(text="plot: only on pause / scrub", icon="INFO")
        col.label(text=f"frame {context.scene.frame_current} / {context.scene.frame_end}"
                       f"  ({context.scene.render.fps} fps)")


_CLASSES = (SOMAViewerProps, SOMA_OT_reload_run, SOMA_OT_rescan_runs, SOMA_PT_viewer)


def register():
    for c in _CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.soma_viewer = bpy.props.PointerProperty(type=SOMAViewerProps)
    if _on_frame not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_on_frame)


def unregister():
    if _on_frame in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_on_frame)
    del bpy.types.Scene.soma_viewer
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)


# --------------------------------------------------------------------------- GUI viewport setup
def _open_plot_editor():
    """Split the largest 3D area into 3D-view + an Image Editor showing the IMU plot."""
    try:
        win = bpy.context.window
        scr = win.screen
        area = max((a for a in scr.areas if a.type == "VIEW_3D"),
                   key=lambda a: a.width * a.height, default=None)
        if area is None:
            return
        with bpy.context.temp_override(window=win, area=area):
            bpy.ops.screen.area_split(direction="VERTICAL", factor=0.62)
        newarea = max((a for a in scr.areas if a.type == "VIEW_3D"), key=lambda a: a.x)
        newarea.type = "IMAGE_EDITOR"
        newarea.spaces.active.image = _STATE["img"]
    except Exception as exc:                          # non-fatal: plot image still exists to open manually
        print("[viewer_app] plot editor split skipped:", exc)


def _setup_viewport():
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            sp = area.spaces.active
            sp.shading.type = "MATERIAL"
            sp.region_3d.view_perspective = "CAMERA"
            # clean look: hide the helper displays (IMU-pivot Empty wires, area-light shapes, camera
            # guide, parent-child relationship lines). Body mesh + RGB axis arrows + dots stay visible;
            # MATERIAL shading uses studio light so hiding scene lights doesn't change the look.
            sp.overlay.show_relationship_lines = False
            sp.show_object_viewport_empty = False
            sp.show_object_viewport_light = False
            sp.show_object_viewport_camera = False
    try:
        bpy.context.preferences.view.use_save_prompt = False    # generated scene → no "unsaved" quit prompt
        bpy.context.scene.sync_mode = "FRAME_DROP"              # play at real time (drop frames if the PC is busy)
    except Exception as exc:
        print("[viewer_app] pref/sync setup skipped:", exc)
    _apply_visibility()                                         # apply skeleton / IMU-axis toggles
    _open_plot_editor()
    try:
        bpy.ops.screen.animation_play()
    except Exception as exc:
        print("[viewer_app] animation_play skipped:", exc)


def _deferred_setup():
    """Run viewport/area/playback setup once, AFTER Blender's UI context is ready (timer callback)."""
    _setup_viewport()
    return None                                       # one-shot timer


def _selftest_module_origin():
    """Every viewer module must come from THIS folder, or the selftest is passing someone else's code.

    render_amass puts its own folder at sys.path[0] on import; when that is the deployed bundle,
    which carries older copies of these modules, the imports after it resolved there and a selftest
    run against a checkout exercised the bundle instead. The presentation modules may come from
    anywhere.
    """
    here = os.path.normcase(HERE)
    mods = [viewer_paths, smpl_rig, smpl_model, smpl_recon, viewer_scene, imu_plot, rmi]
    mods += [sys.modules[n] for n in ("generate_prism_faithful", "anthro_smpl") if n in sys.modules]
    for mod in mods:
        origin = os.path.normcase(os.path.dirname(os.path.abspath(mod.__file__)))
        assert origin == here, f"{mod.__name__} was imported from {mod.__file__}, not from {HERE}"
    print(f"SELFTEST modules: viewer modules from {HERE}; render_amass from "
          f"{os.path.dirname(os.path.abspath(ra.__file__))}")


def _selftest_plot_rate():
    """The Plot update slider must actually gate the automatic redraw (0 Hz = off)."""
    p = bpy.context.scene.soma_viewer
    try:
        p.plot_update_hz = 10
        assert abs(_plot_interval() - 0.1) < 1e-9, _plot_interval()
        p.plot_update_hz = 0
        assert _plot_interval() is None
        _STATE["last"] = 0.0                      # a redraw is "due" — but 0 Hz must still skip it
        refresh_plot(force=False)
        assert _STATE["last"] == 0.0, "0 Hz still redrew the plot"
        refresh_plot(force=True)                  # explicit refresh must always work
        assert _STATE["last"] > 0.0, "force=True did not redraw"
        print("SELFTEST plot rate: 10 Hz -> 0.1 s, 0 Hz -> auto-redraw off, force still redraws")
    finally:
        p.plot_update_hz = PLOT_HZ_DEFAULT


def _restore_picker(p, keep):
    """Put the panel back exactly as a selftest found it, and prove the restore took.

    An enum assignment is silently ignored when the value is not among the current items, so a
    selftest could otherwise hand the session back pointing at a different collection than it started
    with — which is what "leaks state" looks like from the outside.
    """
    run, group, needle, current = keep
    _SUPPRESS_REBUILD[0] = True
    try:
        _CURRENT_GROUP[0] = current
        p.run_filter = needle
        p.run_group = group
        p.run = run
    finally:
        _SUPPRESS_REBUILD[0] = False
    assert (p.run, p.run_group, p.run_filter) == (run, group, needle), (
        f"picker not restored: {(p.run, p.run_group, p.run_filter)} != {(run, group, needle)}")


def _write_probe_run(run_dir, frames=2, gender="male", betas=None, openable=False):
    """A minimal run that satisfies the contract: shape, rotations and a root translation.

    `openable` adds the IMU channels the plot reads, which is what separates a run the picker will
    LIST from one it can actually BUILD — discovery tests want the first, the body-model test needs
    the second.
    """
    os.makedirs(run_dir, exist_ok=True)
    rot = np.zeros((frames, 24, 4), np.float32)
    rot[..., 0] = 1.0                                 # identity quaternions
    small = {"pair_id": np.array("probe")}
    if openable:
        small.update(sensor_codes=np.array(imu_plot.SITES),
                     imu_acceleration=np.zeros((frames, len(imu_plot.SITES), 3), np.float32),
                     imu_angular_velocity=np.zeros((frames, len(imu_plot.SITES), 3), np.float32),
                     imu_orientation=np.tile(np.array([1, 0, 0, 0], np.float32),
                                             (frames, len(imu_plot.SITES), 1)))
    np.savez(os.path.join(run_dir, "small_reference.npz"), **small)
    np.savez(os.path.join(run_dir, "large_reference.npz"), pair_id=np.array("probe"),
             smpl_global_orientation_world=rot, frame_count=np.int64(frames))
    np.savez(os.path.join(run_dir, "anthro_reference.npz"), pair_id=np.array("probe"),
             betas=(np.zeros(10, np.float32) if betas is None else np.asarray(betas, np.float32)),
             gender=np.array(gender), smpl_trans=np.zeros((frames, 3), np.float32))


def _selftest_model_by_gender():
    """A female or neutral subject must be drawn on ITS OWN model, end to end through build_scene_for_run.

    Asserted on the mesh Blender actually holds rather than on which path was chosen: the rig builds
    its mesh from the shaped rest vertices, so comparing them against each model's own forward pass
    is what distinguishes "picked the right file" from "picked it and then used the other one".
    Neutral gets the same treatment as female because it is how GAITEX arrives, and a neutral subject
    drawn on the male template was exactly as silent as the female case used to be.

    Leaves the probe scene in place; it runs last in the headless selftest for that reason.
    """
    import shutil
    import tempfile

    assert model_gender("female") == model_gender("F") == model_gender("f") == "female"
    assert model_gender("neutral") == model_gender("N") == model_gender("Neutral") == "neutral"
    assert model_gender("male") == model_gender("M") == model_gender("") == "male"
    assert model_gender(None) == model_gender("None") == "male"          # unrecorded is not neutral
    assert model_gender(b"female") == "female" and model_gender(np.array("neutral")) == "neutral"
    assert len(set(MODEL_FILES.values())) == 3, MODEL_FILES

    betas = np.linspace(-1.2, 1.4, 10).astype(np.float32)      # a shape no model draws by default
    other = fallback_model()                                   # what a subject without its own gets
    assert other, "no SMPL model at all on this machine — nothing to test the rig with"
    other_verts = smpl_model.smpl_shape(smpl_model.load_model_npz(other), betas)[0]
    tmp = tempfile.mkdtemp(prefix="soma_gender_")
    try:
        for gender in ("female", "neutral"):
            own_npz = _resolve_model(gender)
            run_dir = os.path.join(tmp, "coll_gender_v1", f"probe-{gender}-take001")
            _write_probe_run(run_dir, frames=40, gender=gender, betas=betas, openable=True)
            chosen, note = model_for_gender(gender)
            build_scene_for_run(run_dir, ground_step=4)
            mesh = next(o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("SMPL"))
            built = np.empty(len(mesh.data.vertices) * 3, np.float64)
            mesh.data.vertices.foreach_get("co", built)
            built = built.reshape(-1, 3)

            if own_npz:
                assert not note, f"the {gender} model is present but was refused: {note}"
                assert os.path.normcase(chosen) == os.path.normcase(own_npz), (chosen, own_npz)
                want = smpl_model.smpl_shape(smpl_model.load_model_npz(own_npz), betas)[0]
                gap_own = float(np.abs(built - want).max())
                assert gap_own < 1e-6, f"the built body is not the {gender} one ({gap_own:.2e} m)"
                if os.path.normcase(own_npz) != os.path.normcase(other):
                    gap_other = float(np.abs(built - other_verts).max())
                    assert gap_other > 1e-3, (
                        f"{gender} and {os.path.basename(other)} bodies came out identical "
                        f"({gap_other:.2e} m) — the chosen model is not reaching the rig")
                    print(f"SELFTEST body model: {gender} subject -> {os.path.basename(chosen)}; built "
                          f"mesh matches it to {gap_own:.1e} m and differs from "
                          f"{os.path.basename(other)} by {gap_other * 1000:.1f} mm")
                else:
                    print(f"SELFTEST body model: {gender} subject -> {os.path.basename(chosen)}; built "
                          f"mesh matches it to {gap_own:.1e} m (it is the only model here)")
            else:
                # A bundle may ship one model alone. That is allowed, but it must be SAID.
                assert os.path.normcase(chosen) == os.path.normcase(other), (chosen, other)
                assert "NOT THIS SUBJECT" in note, note
                assert float(np.abs(built - other_verts).max()) < 1e-6, "fell back but drew something else"
                print(f"SELFTEST body model: no {gender} model here, fell back to "
                      f"{os.path.basename(other)} and said so")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _selftest_collection_keys():
    """Two collections holding the SAME take name must stay separately listed and selectable.

    Built as a fixture instead of hunting for a duplicate name in _RUNS: on a machine carrying only
    one collection — which is what a teammate has — that search finds nothing and passes without testing
    the regression at all, and _RUNS is already deduplicated by key so it could not have failed.
    """
    import shutil
    import tempfile

    take = "prism-subj001-take002"
    colls = ("coll_faithful_v1", "coll_measured_v1")
    tmp = tempfile.mkdtemp(prefix="soma_colls_")
    prev = os.environ.get("SOMA_VIEWER_DATASET")
    p = bpy.context.scene.soma_viewer
    keep = (p.run, p.run_group, p.run_filter, _CURRENT_GROUP[0])
    displayed = _STATE.get("run_id")
    try:
        for coll in colls:
            _write_probe_run(os.path.join(tmp, coll, take))
            with open(os.path.join(tmp, coll, CATALOG), "w", encoding="utf-8") as fh:
                json.dump({"takes": [{"take": take, "status": "ok"}]}, fh)

        os.environ["SOMA_VIEWER_DATASET"] = tmp
        same_name = sorted(rid for rid, _d in refresh_runs(force=True) if run_name_of(rid) == take)
        assert len(set(same_name)) == len(same_name), f"a take name collapsed onto one key: {same_name}"
        keys = [k for k in same_name if k.split("/", 1)[0] in colls]     # the fixture's own two
        assert keys == [f"{colls[0]}/{take}", f"{colls[1]}/{take}"], keys
        assert len({run_group_of(k) for k in keys}) == 2, "both collections collapsed into one"

        _SUPPRESS_REBUILD[0] = True                   # selecting a collection must not rebuild here
        try:
            for k in keys:
                _CURRENT_GROUP[0] = run_group_of(k)
                p.run_group = _CURRENT_GROUP[0]
                offered = [i[0] for i in _run_items(p, None)]
                assert offered == [k], f"{p.run_group} offered {offered}"
        finally:
            _SUPPRESS_REBUILD[0] = False
        print(f"SELFTEST collections: '{take}' in 2 collections -> 2 keys, each selectable on its own")
    finally:
        os.environ.pop("SOMA_VIEWER_DATASET", None)
        if prev is not None:
            os.environ["SOMA_VIEWER_DATASET"] = prev
        shutil.rmtree(tmp, ignore_errors=True)
        refresh_runs(force=True)
        _restore_picker(p, keep)
        assert _STATE.get("run_id") == displayed, "the collection selftest changed the displayed run"


def _selftest_run_picker():
    """The collection + Find controls must actually narrow the run list."""
    assert run_group_of("prism-subj001-take002") == "prism-subj001"
    assert run_group_of("amass_BMLmovi_Subject_1_F_MoSh__Subject_1_F_1_poses") == "amass_BMLmovi_Subject_1_F_MoSh"
    assert run_group_of("standalone") == "standalone"
    assert run_group_of("coll_a/prism-subj001-take002") == "coll_a/prism-subj001"
    assert run_name_of("coll_b/prism-subj001-take002") == "prism-subj001-take002"
    # --run: a full id names exactly one take, whatever sorts first; a short name offers every
    # collection's take of that name, in list order
    twins = [("_dryrun/take-a", "d1"), ("bundle_v1/take-a", "d2"), ("bundle_v1/take-b", "d3")]
    assert prefer_run(twins, "bundle_v1/take-a") == ([twins[1]], [twins[0], twins[2]])
    assert prefer_run(twins, "take-a") == ([twins[0], twins[1]], [twins[2]])
    assert prefer_run(twins, "take-zzz") == ([], twins)

    p = bpy.context.scene.soma_viewer
    keep = (p.run, p.run_group, p.run_filter, _CURRENT_GROUP[0])
    displayed = _STATE.get("run_id")
    try:
        _selftest_run_picker_body(p)
    finally:                                          # the panel must be left exactly as it was
        _restore_picker(p, keep)
    assert _STATE.get("run_id") == displayed, "the picker selftest changed the displayed run"


def _selftest_run_picker_body(p):
    groups = [g for g, _lbl, _d in _group_items(p, None)]
    assert len(groups) < len(_RUNS) or len(_RUNS) <= 1, (len(groups), len(_RUNS))

    _SUPPRESS_REBUILD[0] = True                      # changing collections here must not rebuild
    try:
        _CURRENT_GROUP[0] = run_group_of(_RUNS[0][0])
        p.run_group = _CURRENT_GROUP[0]
    finally:
        _SUPPRESS_REBUILD[0] = False
    in_group = [i[0] for i in _run_items(p, None)]
    assert all(run_group_of(r) == p.run_group for r in in_group), in_group[:5]
    assert len(in_group) <= len(_RUNS)

    p.run_filter = _RUNS[0][0]                       # Find is a substring match, so a full id may
    only = [i[0] for i in _run_items(p, None)]       # still match longer ids that contain it
    assert _RUNS[0][0] in only and all(p.run_filter in r for r in only), only[:5]
    assert len(only) <= len(in_group)
    p.run_filter = "zzz-no-such-run"
    assert [i[0] for i in _run_items(p, None)] == ["NONE"]

    p.run_filter = p.run_group                       # a collection name narrows the COLLECTION list
    narrowed = [g for g, _lbl, _d in _group_items(p, None)]
    assert p.run_group in narrowed and len(narrowed) <= len(groups), (len(narrowed), len(groups))
    assert len(_run_items(p, None)) == len(in_group), "matching the group must not empty its runs"
    p.run_filter = ""
    print(f"SELFTEST run picker: {len(_RUNS)} runs -> {len(groups)} collection(s), "
          f"{len(in_group)} in '{p.run_group}', Find narrows to 1")


def _selftest_failure_isolation():
    """A run that cannot open must not raise, and must not cost the scene already on screen.

    Covers the SystemExit path specifically: the readers signal a malformed run with it, and it is a
    BaseException, so it slips past `except Exception`.
    """
    import shutil
    import tempfile

    def scene_shape():
        objs = list(bpy.data.objects)
        return (len([o for o in objs if o.type == "MESH" and o.name.startswith("SMPL")]),
                len([o for o in objs if o.name.startswith("imupiv_")]))

    before_shape = scene_shape()
    before_run = _STATE.get("run_id")
    assert before_shape == (1, 8), f"a good scene must be on screen first, got {before_shape}"

    tmp = tempfile.mkdtemp(prefix="soma_isolation_")
    try:
        # npz all present, but the large reference carries no [T,24,4] rotation array -> SystemExit
        bad = os.path.join(tmp, "malformed-take001")
        os.makedirs(bad)
        np.savez(os.path.join(bad, "small_reference.npz"), pair_id=np.array("x"))
        np.savez(os.path.join(bad, "large_reference.npz"), pair_id=np.array("x"),
                 joint_rotation=np.zeros((2, 18, 4), np.float32))
        np.savez(os.path.join(bad, "anthro_reference.npz"), betas=np.zeros(10, np.float32),
                 gender=np.array("male"), smpl_trans=np.zeros((2, 3), np.float32),
                 pair_id=np.array("x"))

        why = try_build_run(bad)                       # must return, not raise
        assert why, "a malformed run reported success"
        assert "SystemExit" in why, f"expected the SystemExit path, got {why}"
        assert scene_shape() == before_shape, f"scene was destroyed: {scene_shape()} != {before_shape}"
        assert _STATE.get("run_id") == before_run, "the displayed run changed on a failed open"
        print(f"SELFTEST isolation: malformed run rejected as {why.split(':')[0]}, "
              f"scene kept ({before_shape[0]} body, {before_shape[1]} IMU)")
    finally:
        _BAD_RUNS.pop("malformed-take001", None)
        _STATE["load_error"] = ""
        shutil.rmtree(tmp, ignore_errors=True)


def _selftest_catalog():
    """A published INDEX.json must drive discovery instead of a walk, and must be authoritative."""
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="soma_catalog_")
    prev = os.environ.get("SOMA_VIEWER_DATASET")
    displayed = _STATE.get("run_id")              # probing bad runs must not disturb what is shown
    try:
        # One entry per field a shipped catalog has used to name its folders. Reading only some of
        # them is how the fast path died once already: the regenerated collections switched from
        # `out_dir`/`take` to `relative_path`/`take_id`, no entry resolved, and every collection
        # silently reverted to a full walk. Each spelling is therefore exercised by its own take.
        listed = {"out_dir": "listed-take001", "take": "listed-take002",
                  "relative_path": "listed-take003", "take_id": "listed-take004"}
        # GAITEX names its takes <subject>_<condition>, which no take-marker rule can split, and its
        # catalog carries `subject_id` instead. Two takes of one subject must land in one group.
        subject = {"listed-take001": "subjA", "listed-take002": "subjA", "listed-take003": "subjB"}
        coll = os.path.join(tmp, "coll_v1")
        for name in list(listed.values()) + ["unlisted-take005"]:
            os.makedirs(os.path.join(coll, name))     # deliberately EMPTY: no npz to walk into
        with open(os.path.join(coll, CATALOG), "w", encoding="utf-8") as fh:
            json.dump({"takes": [{field: name, "status": "ok",
                                  **({"subject_id": subject[name]} if name in subject else {})}
                                 for field, name in listed.items()] +
                                [{"take": "missing-take009", "status": "ok"},
                                 {"take": "failed-take008", "status": "failed"}]}, fh)

        os.environ["SOMA_VIEWER_DATASET"] = tmp
        ids = {run_name_of(rid) for rid, _d in refresh_runs(force=True)}
        for field, name in listed.items():
            assert name in ids, f"a catalog entry naming its folder with {field!r} was not offered"
        assert "unlisted-take005" not in ids, "a folder the catalog omits must not appear"
        assert "missing-take009" not in ids, "a catalog entry with no folder must not appear"
        assert "failed-take008" not in ids, "a catalog entry marked failed must not appear"
        assert all(rid.startswith("coll_v1/") for rid, _d in _RUNS
                   if run_name_of(rid).startswith("listed-take")), "keys must name their collection"
        groups = {run_name_of(rid): run_group_of(rid) for rid, _d in _RUNS if rid.startswith("coll_v1/")}
        assert groups["listed-take001"] == groups["listed-take002"] == "coll_v1/subjA", groups
        assert groups["listed-take003"] == "coll_v1/subjB", groups
        assert groups["listed-take004"] == "coll_v1/listed", groups   # no subject_id: the name rule

        # The catalog says these are published; this machine cannot read them (the folders are
        # empty, as a half-synced share looks). Startup must survive that, and they must leave.
        probed = [(rid, d) for rid, d in _RUNS if run_name_of(rid).startswith("listed-take")]
        assert len(probed) == len(listed), (len(probed), len(listed))
        opened, _dir = first_buildable(probed)
        assert opened is None, "an empty run folder must not report as opened"
        assert len(_BAD_RUNS) >= len(listed), _BAD_RUNS
        after = {run_name_of(rid) for rid, _d in refresh_runs(force=True)}
        assert not (after & set(listed.values())), "unreadable runs stayed in the list"
        assert _STATE.get("run_id") == displayed, "probing unreadable runs changed the displayed run"
        print(f"SELFTEST catalog: INDEX.json drives discovery; unlisted/missing/failed skipped; "
              f"{len(_BAD_RUNS)} unreadable run(s) dropped instead of aborting startup")
    finally:
        os.environ.pop("SOMA_VIEWER_DATASET", None)
        if prev is not None:
            os.environ["SOMA_VIEWER_DATASET"] = prev
        shutil.rmtree(tmp, ignore_errors=True)
        # _BAD_RUNS is keyed by the composite id, not the bare take name — popping the short name
        # left the fixture's rejects behind in a dict that lives for the whole session.
        for rid in [k for k in _BAD_RUNS if run_group_of(k).startswith("coll_v1/")]:
            _BAD_RUNS.pop(rid, None)
        _STATE["load_error"] = ""
        refresh_runs(force=True)


def _selftest_dropdown_rescan():
    """A run dropped into the dataset folder must reach the dropdown without restarting Blender.

    Exercises the real path: write a run, then ask the enum callback for its items (as the UI does
    when the list is opened) and check the new id is offered.
    """
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="soma_rescan_")
    prev = os.environ.get("SOMA_VIEWER_DATASET")
    try:
        os.environ["SOMA_VIEWER_DATASET"] = tmp
        before = len(refresh_runs(force=True))
        _write_probe_run(os.path.join(tmp, "probe-run-rescan"))

        _SCAN["t"] = 0.0                          # the UI would re-scan once the throttle window lapses
        found = [rid for rid, _d in refresh_runs(force=True)]
        probe_key = next((rid for rid in found if run_name_of(rid) == "probe-run-rescan"), None)
        assert probe_key, "rescan missed the new run"
        assert len(found) == before + 1, f"expected {before + 1} runs, got {len(found)}"

        p = bpy.context.scene.soma_viewer         # and it must be reachable through the picker
        keep = p.run_group
        _SUPPRESS_REBUILD[0] = True               # switching collections would rebuild the scene
        try:
            p.run_group = run_group_of(probe_key)
            offered = [i[0] for i in _run_items(p, None) if i[0] == probe_key]
            p.run_group = keep
        finally:
            _SUPPRESS_REBUILD[0] = False
        assert offered == [probe_key], "new run not offered by its own collection"
        print(f"SELFTEST rescan: {before} -> {len(found)} run(s), new one selectable, no restart")
    finally:
        os.environ.pop("SOMA_VIEWER_DATASET", None)
        if prev is not None:
            os.environ["SOMA_VIEWER_DATASET"] = prev
        shutil.rmtree(tmp, ignore_errors=True)
        refresh_runs(force=True)                  # drop the probe from the cached list


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    selftest = "--selftest" in argv
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    want = None                                        # open this take first
    if "--run" in argv:
        i = argv.index("--run") + 1
        if i < len(argv):
            want = argv[i]
        else:
            print("[viewer_app] --run given without a take; ignored")
    for i, arg in enumerate(argv):                     # --folder <dir>, repeatable: scanned first
        if arg == "--folder":
            if i + 1 < len(argv) and os.path.isdir(argv[i + 1]):
                _EXPLICIT_ROOTS.append(argv[i + 1])
            else:
                raise SystemExit(f"[viewer_app] --folder needs an existing folder, got "
                                 f"{argv[i + 1] if i + 1 < len(argv) else 'nothing'}")

    _load_scan_cache()                                 # a first scan of a big collection is costly
    refresh_runs(force=True)
    for i, root in enumerate(run_roots()):             # printed so a user can see where runs came from
        print(f"[viewer_app] run root {i + 1}: {root}")
    print(f"[viewer_app] {len(_RUNS)} run(s): {[r for r, _ in _RUNS]}")
    if not _RUNS:
        raise SystemExit("[viewer_app] no viewable runs found. Searched:\n  "
                         + "\n  ".join(run_roots() or ["(no existing root)"])
                         + f"\nA run folder needs {list(_PAIR_NPZ)} plus "
                         + f"{list(smpl_recon.SHAPE_REFERENCE_NAMES)} carrying betas + smpl_trans.\n"
                         + viewer_paths.no_runs_hint())

    candidates, hit = _RUNS, []
    if want:
        hit, rest = prefer_run(_RUNS, want)
        if not hit:
            msg = f"[viewer_app] --run {want!r} names none of the {len(_RUNS)} discovered run(s)"
            if selftest:
                raise SystemExit(msg)
            print(msg + " — opening the first run instead")
        elif len(hit) > 1:
            # A short take name shared by two collections would let a selftest pass on one while
            # claiming the other. Headless, that is a failure.
            listed = ", ".join(rid for rid, _d in hit)
            if selftest:
                raise SystemExit(f"[viewer_app] --run {want!r} is ambiguous ({listed}); "
                                 f"give <collection>/<take>")
            print(f"[viewer_app] --run {want!r} matches {len(hit)} runs ({listed}); opening the first")
        if hit:
            candidates = hit + rest
    opened_id, _opened_dir = first_buildable(candidates, ground_step=(4 if not selftest else 6))
    if hit and opened_id is not None and opened_id not in {rid for rid, _d in hit}:
        # In the GUI the fallback is the right behaviour; headless it would pass the wrong dataset.
        print(f"[viewer_app] --run {want!r} could not be opened ({_BAD_RUNS.get(hit[0][0], '?')}); "
              f"showing {opened_id} instead")
        if selftest:
            raise SystemExit(f"[viewer_app] selftest asked for {want!r} and got {opened_id}")
    if opened_id is None:
        tried = hit if hit else candidates[:25]       # a requested take is reported as itself
        raise SystemExit("[viewer_app] none of the first runs could be opened. Tried:\n  "
                         + "\n  ".join(f"{rid}: {_BAD_RUNS.get(rid, '?')}" for rid, _d in tried)
                         + "\nOn a synced share the files may not have downloaded yet.")
    if _BAD_RUNS:
        print(f"[viewer_app] skipped {len(_BAD_RUNS)} unreadable run(s) before {opened_id}")
        refresh_runs(force=True)                      # drop them from the picker
    register()
    # Point the collection dropdown at the run that was just built, or its enum would default to the
    # first group and silently disagree with the scene on screen. Suppressed: the scene already IS
    # this run, and rebuilding a 13 k-frame take again would cost another minute for nothing.
    props = bpy.context.scene.soma_viewer
    _SUPPRESS_REBUILD[0] = True
    try:
        _CURRENT_GROUP[0] = run_group_of(opened_id)
        props.run_group = _CURRENT_GROUP[0]
        props.run = opened_id
    finally:
        _SUPPRESS_REBUILD[0] = False
    init_plot()
    refresh_plot(force=True)

    if selftest:
        img = _STATE["img"]
        px = np.empty(len(img.pixels), dtype=np.float32)
        img.pixels.foreach_get(px)
        nonwhite = int(np.count_nonzero(px[:len(px) // 4 * 4].reshape(-1, 4)[:, :3].sum(1) < 2.9))
        print(f"SELFTEST runs={len(_RUNS)} imu_sites={len(imu_plot.SITES)} "
              f"plot_img={tuple(img.size)} plot_nonwhite_px={nonwhite}")
        print(f"SELFTEST opened={opened_id} group={run_group_of(opened_id)} "
              f"body={os.path.basename(_STATE.get('body_model') or '?')} "
              f"unbacked={sorted(_STATE['data'].get('unbacked') or [])} "
              f"root={_STATE.get('trans_source')}")
        objs = list(bpy.data.objects)
        meshes = [o for o in objs if o.type == "MESH" and o.name.startswith("SMPL")]
        pivots = [o for o in objs if o.name.startswith("imupiv_")]
        assert len(meshes) == 1 and len(pivots) == 8, (len(meshes), len(pivots))
        assert nonwhite > 1000, f"plot looks empty ({nonwhite} non-white px)"
        _selftest_module_origin()
        _selftest_failure_isolation()
        _selftest_catalog()
        _selftest_collection_keys()
        _selftest_dropdown_rescan()
        _selftest_plot_rate()
        _selftest_run_picker()
        if out:
            imu_plot.render_png(_STATE["run_dir"], "wrist_l", 500, out)
            print("wrote", out)
        _selftest_model_by_gender()      # last: it replaces the scene with its own probe subject
        print("SELFTEST_OK")
        return

    bpy.app.timers.register(_deferred_setup, first_interval=0.3)   # defer GUI setup until UI is ready


if __name__ == "__main__":
    main()
