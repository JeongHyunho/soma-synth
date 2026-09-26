"""Where the viewer and its tools find things on THIS machine: data, body models, dependencies.

The data plane comes from ``SOMA_DATA_ROOT`` (the bundles under
``<SOMA_DATA_ROOT>/runs/experimental_generation_poc_demo``, the body models under
``SOMA_BODY_MODEL_DIR``, else ``<SOMA_DATA_ROOT>/body_models/smpl``) or from an explicit folder;
a tool that needs the data root and has neither stops with :func:`data_root`'s message naming both.
A tool that writes beside a take refuses a folder inside a bundle, the lineage container, an
evidence folder or the sources (:func:`writable_folder`): only a bundle's generator writes there.

`render_amass.py` / `gear_kit.py` (the presentation package) are not in this repository: they are
distributed separately, with the portable viewer bundle. The viewer, its scene builder and the
PRISM mesh tools import them and do not run without them. A checkout finds them through
``$SOMA_ANIM_DIR``; the portable bundle carries its own copies next to the viewer scripts.
:func:`anim_dir` finds a readable copy or returns "", and the import that needs it then fails by
name. It never opens a file that is not stored on this machine (a cloud client's online-only
placeholder: opening it blocks until the file is downloaded, or forever when it cannot be).
INTERNAL-ONLY.

No bpy here, so the location rules are testable outside Blender (tests/poc/test_viewer_paths.py).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT_ENV = "SOMA_DATA_ROOT"
BODY_MODEL_DIR_ENV = "SOMA_BODY_MODEL_DIR"
#: Where the bundles live under the data root, one folder per lineage.
POC_DEMO = ("runs", "experimental_generation_poc_demo")
#: Evidence and run records beside the lineage folders: never a collection of runs.
EVIDENCE_FOLDERS = ("_superseded", "_runs", "_manifest_backfill")
_PROBE = "render_amass.py"
#: Windows file attributes that say a file's data is not stored locally: OFFLINE (0x1000),
#: RECALL_ON_OPEN (0x40000), RECALL_ON_DATA_ACCESS (0x400000). A Dropbox or OneDrive online-only
#: placeholder carries one of them (a Dropbox placeholder typically reads 0x1620: OFFLINE,
#: REPARSE_POINT, SPARSE, ARCHIVE); opening it waits for the sync client to download it.
NOT_STORED_LOCALLY = 0x1000 | 0x40000 | 0x400000


def _env(name, environ=None):
    value = (os.environ if environ is None else environ).get(name, "")
    return value.strip() or None


def data_root(explicit=None, *, environ=None, hint="or pass the location explicitly"):
    """``explicit``, else ``$SOMA_DATA_ROOT``, as an absolute path; SystemExit naming both when
    neither is given. There is no default: a default is one machine's layout."""
    value = explicit or _env(DATA_ROOT_ENV, environ)
    if not value:
        raise SystemExit(f"{DATA_ROOT_ENV} is not set. Set it to the local data root (the folder "
                         f"holding runs/ and extracted/), {hint}")
    return os.path.abspath(value)


def bundles_root(root):
    """``<root>/runs/experimental_generation_poc_demo``: the lineage container."""
    return os.path.join(root, *POC_DEMO)


def bundles_hint(environ=None):
    """Where the local bundles are, for a message that asks for an explicit folder."""
    root = _env(DATA_ROOT_ENV, environ)
    if root:
        return f"the bundles are under {bundles_root(os.path.abspath(root))}"
    return (f"the bundles are under $SOMA_DATA_ROOT/{'/'.join(POC_DEMO)} ({DATA_ROOT_ENV} is not "
            "set here)")


def require(value, flag, what, *, environ=None, writes=None):
    """``value``, or SystemExit asking for ``flag`` with :func:`bundles_hint`. The PRISM tools have
    no default: name the take folder. ``writes``, for a tool that writes where ``flag`` points,
    says what to name instead of a bundle's own folder (:func:`writable_folder` refuses one)."""
    if value:
        return value
    hint = bundles_hint(environ)
    message = f"name {what} with {flag}; there is no default. {hint[0].upper()}{hint[1:]}"
    if writes:
        message += f". This tool writes there and never into a bundle: {writes}"
    raise SystemExit(message)


#: What a bundle and a corpus hold at their top: a folder under one is its generator's alone.
BUNDLE_INDEX = "INDEX.json"
#: What a writing tool's refusal and its missing-argument message tell the user to do instead.
COPY_THE_TAKE = "copy the take folder to a scratch folder and name the copy"


def _inside(path, folder):
    path = os.path.normcase(os.path.realpath(path))
    folder = os.path.normcase(os.path.realpath(folder))
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:                      # another drive on Windows
        return False


def _folder_names(path):
    """The folder names of ``path`` (absolute, links resolved), case-folded, root first."""
    names = []
    head = os.path.realpath(path)
    while True:
        head, tail = os.path.split(head)
        if not tail:
            return names[::-1]
        names.append(tail.casefold())


def writable_folder(folder, flag, *, instead=COPY_THE_TAKE, environ=None):
    """``folder`` when a tool may write its own files into it; SystemExit otherwise, before anything
    is written.

    Refused: a folder inside a bundle or corpus (it, or a folder above it, holds ``INDEX.json``),
    inside the lineage container ``runs/experimental_generation_poc_demo`` of any data root (matched
    by its folder names), inside an evidence folder (``_superseded``, ``_runs``,
    ``_manifest_backfill``), or inside ``extracted/`` or ``raw_archives/`` of ``$SOMA_DATA_ROOT`` or
    under ``$SOMA_SOURCE_ROOT`` (retention rule 1.4: sources are never written). Only a bundle's
    generator writes into it; a tool that writes beside a take works on a copy (``instead``). The
    rule soma-synth's ``hash-sources`` applies to its list, with the standard library only."""
    def refuse(where):
        raise SystemExit(f"refusing to write into {folder} ({flag}): it is inside {where}; "
                         f"{instead}")

    names = _folder_names(folder)
    evidence = {name.casefold() for name in EVIDENCE_FOLDERS}
    for name in names:
        if name in evidence:
            refuse(f"the evidence folder {name!r}, which holds records only")
    container = tuple(part.casefold() for part in POC_DEMO)
    for index in range(len(names) - len(container) + 1):
        if tuple(names[index:index + len(container)]) == container:
            refuse(f"the lineage container {'/'.join(POC_DEMO)}, which holds bundles and corpora "
                   "only")
    current = os.path.realpath(folder)
    while True:
        if os.path.isfile(os.path.join(current, BUNDLE_INDEX)):
            refuse(f"the bundle or corpus {current} (it holds {BUNDLE_INDEX}), which only its "
                   "generator writes")
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    root = _env(DATA_ROOT_ENV, environ)
    sources = _env("SOMA_SOURCE_ROOT", environ)
    guarded = [(os.path.join(root, name), f"{DATA_ROOT_ENV}/{name}")
               for name in ("extracted", "raw_archives")] if root else []
    guarded += [(sources, "SOMA_SOURCE_ROOT")] if sources else []
    for base, label in guarded:
        if _inside(folder, base):
            refuse(f"{label} ({base}), whose sources are never written")
    return folder


def require_files(folder, names):
    """``folder`` when it holds every file of ``names``; SystemExit naming the missing ones (checked
    before a tool writes anything)."""
    missing = [name for name in names if not os.path.isfile(os.path.join(folder, name))]
    if missing:
        raise SystemExit(f"{folder} does not hold {', '.join(missing)}; name the folder that does")
    return folder


def source_folder(root, folder, *, environ=None):
    """``$SOMA_SOURCE_ROOT/<folder>`` when that is set, else ``<root>/extracted/<folder>``."""
    sources = _env("SOMA_SOURCE_ROOT", environ)
    return os.path.join(sources, folder) if sources else os.path.join(root, "extracted", folder)


def body_model_dir(root=None, *, environ=None):
    """``$SOMA_BODY_MODEL_DIR``, else ``<root or $SOMA_DATA_ROOT>/body_models/smpl``; None when
    neither variable is set and no root is given."""
    override = _env(BODY_MODEL_DIR_ENV, environ)
    if override:
        return os.path.abspath(override)
    base = root or _env(DATA_ROOT_ENV, environ)
    return os.path.join(base, "body_models", "smpl") if base else None


def run_roots(bundle_root, here=HERE, explicit=(), *, environ=None):
    """Existing folders to scan for runs, highest priority first, de-duplicated.

    1. the folders given explicitly (the viewer's ``--folder``), then ``$SOMA_VIEWER_DATASET``;
    2. the bundles under ``$SOMA_DATA_ROOT`` (``runs/experimental_generation_poc_demo``);
    3. a portable bundle's own ``data/runs`` (``$SOMA_VIEWER_DATA/runs``, then beside the scripts).
    """
    roots, seen = [], set()

    def add(path):
        if not path or not os.path.isdir(path):
            return
        absolute = os.path.abspath(path)
        key = os.path.normcase(absolute)
        if key not in seen:
            seen.add(key)
            roots.append(absolute)

    for folder in explicit or ():
        add(folder)
    add(_env("SOMA_VIEWER_DATASET", environ))
    root = _env(DATA_ROOT_ENV, environ)
    add(bundles_root(root) if root else None)
    data_env = _env("SOMA_VIEWER_DATA", environ)
    add(os.path.join(data_env, "runs") if data_env else None)
    add(os.path.join(here, "data", "runs"))
    add(os.path.join(bundle_root, "data", "runs"))
    return roots


def no_runs_hint(environ=None):
    """What to do when no root yielded a run: the data root, or an explicit folder."""
    if not _env(DATA_ROOT_ENV, environ):
        return (f"{DATA_ROOT_ENV} is not set, so no bundle folder was searched. Set it to the "
                "local data root (the viewer lists the bundles under "
                "runs/experimental_generation_poc_demo) or name a folder with --folder <dir> (or "
                "SOMA_VIEWER_DATASET)")
    return ("name another folder with --folder <dir> (or SOMA_VIEWER_DATASET), or generate a bundle "
            "into the data root first")


def model_candidates(fname, gender, bundle_root, here=HERE, *, environ=None):
    """Where the licensed SMPL model ``fname`` (for ``gender``) may be, in order: its override
    variable, the portable bundle's data, then ``$SOMA_BODY_MODEL_DIR`` or
    ``$SOMA_DATA_ROOT/body_models/smpl``."""
    override = _env(f"SOMA_VIEWER_MODEL_{gender.upper()}", environ)
    if not override and gender == "male":
        override = _env("SOMA_VIEWER_MODEL", environ)          # the name this knob shipped under
    data_env = _env("SOMA_VIEWER_DATA", environ)
    models = body_model_dir(environ=environ)
    return [c for c in (override,
                        os.path.join(data_env, fname) if data_env else None,
                        os.path.join(bundle_root, "data", fname),
                        os.path.join(here, "data", fname),
                        os.path.join(models, fname) if models else None) if c]


def option(argv, name, default=None):
    """The value after ``name`` in a Blender script's own arguments (those after ``--``)."""
    if name in argv:
        index = argv.index(name) + 1
        if index < len(argv):
            return argv[index]
    return default


def script_argv(argv=None):
    """A script's own arguments: after ``--`` under Blender, else everything after the script."""
    argv = sys.argv if argv is None else argv
    return argv[argv.index("--") + 1:] if "--" in argv else argv[1:]


# ---------------------------------------------------------------- the presentation package
def stored_locally(path):
    """Whether ``path`` is a file whose data is stored locally. Asked of its attributes
    (``os.stat``, which does not download anything), never by opening it: an online-only
    placeholder lists and stats fine, and opening it blocks until the sync client has fetched it --
    or for good, when it cannot.
    False for a file that is missing or cannot be stat'ed. Outside Windows there are no such
    attributes and every existing file counts as stored."""
    try:
        attributes = getattr(os.stat(path), "st_file_attributes", 0)
    except OSError:
        return False
    return not attributes & NOT_STORED_LOCALLY


def _readable(path):
    """A file stored locally (:func:`stored_locally`) that opens and yields a byte. A placeholder
    is skipped before it is opened."""
    if not stored_locally(path):
        return False
    try:
        with open(path, "rb") as fh:
            fh.read(1)
        return True
    except OSError:
        return False


def _override():
    d = os.environ.get("SOMA_ANIM_DIR")
    return os.path.abspath(d) if d else ""


def _candidates():
    """Folders to try, in order: $SOMA_ANIM_DIR, then this folder (the portable bundle ships its own
    copies beside the viewer scripts)."""
    if _override():
        yield _override()
    yield HERE


def anim_dir():
    """The first folder that actually yields render_amass.py, or "" when none does."""
    for c in _candidates():
        if c and _readable(os.path.join(c, _PROBE)):
            return os.path.abspath(c)
    return ""


def _pin_first(folder):
    while folder in sys.path:
        sys.path.remove(folder)
    sys.path.insert(0, folder)


def add_to_sys_path():
    """Put this folder first and the folder of the presentation code (:func:`anim_dir`) after it,
    so the viewer's own modules win.

    render_amass inserts ITS OWN folder at sys.path[0] when imported. When that folder is a
    portable bundle's scripts folder -- which also carries older copies of imu_plot, smpl_recon
    and the rest -- every module imported after it silently came from the bundle instead of this
    checkout (a headless selftest passed on code it had not run). So the dependency is imported
    here, once, and this folder is pinned first again afterwards. An explicit $SOMA_ANIM_DIR wins
    even over a copy next to this file, for that one import.

    Returns the folder of the presentation code used ("" when none was found; the import that needs it then
    fails with the usual ModuleNotFoundError, naming render_amass).
    """
    _pin_first(HERE)
    d = anim_dir()
    if d and d not in sys.path:
        sys.path.append(d)
    forced = bool(d) and os.path.normcase(d) == os.path.normcase(_override()) \
        and os.path.normcase(d) != os.path.normcase(HERE)
    if forced:
        sys.path.insert(0, d)
    try:
        import render_amass  # noqa: F401
    except ImportError:
        pass                                    # the caller's own import reports it, by name
    _pin_first(HERE)
    return d
