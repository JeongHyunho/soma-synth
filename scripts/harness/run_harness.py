"""Byte-identity harness: rebuild the sample subjects of the five lineages into a scratch directory.

Rebuild the sample subjects of the five lineages from a given checkout into a scratch directory,
so that two checkouts can be compared byte for byte with `hash_outputs.py` and
`compare_hashes.py` beside it. It runs, one source after another, the retarget-corpus stage (hknu,
gaitex, addbiomechanics) and the bundle stage for the sample subjects only, then renders each
bundle's KNOWN_LIMITATIONS.json. It never validates, writes a README, registers or pushes.

Authority: the five records
research/decisions/2026-09-25_<source>_harness_sample_generation_authorization.md, which live in
the parent project (the checkout that consumes soma-synth), not here: pass that checkout's
research/decisions with --decisions-dir (default: the parent project's research/decisions when
soma-synth is mounted there as packages/soma-synth; a standalone soma-synth checkout has no
default and needs the option). They allow the
sample generation only into a scratch directory under <SOMA_DATA_ROOT>/tmp/harness, never
registered, deleted after hashing. The driver refuses a scratch directory anywhere else, refuses a
source whose record is missing, and compares (size, mtime) before and after the run of: every
top-level entry of runs/experimental_generation_poc_demo, every file at the root of each
production bundle and in each of its sample take directories, every file under the evidence
folders _superseded, _runs and _manifest_backfill, the top level of
extracted/ and every file in the sample source folders. It does not stat the production takes
outside the sample, which a harness run has no business naming. Sources are read in place or
copied; they are never written.

    python scripts/harness/run_harness.py --code-root <exported checkout> \\
        --scratch <SOMA_DATA_ROOT>/tmp/harness/<run> [--sources amass,prism,hknu,gaitex,addbiomechanics] \\
        [--python <exe>] [--record-dir <dir>] [--decisions-dir <parent checkout>/research/decisions]

The code root may hold the package as ``soma_synth`` or as ``soma_synthetic_imu``; the driver
finds which one it holds, and passes only flags both layouts accept.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

#: The checkout this driver lives in.
DRIVER_REPO = pathlib.Path(__file__).resolve().parents[2]


def default_decisions(driver_repo: pathlib.Path) -> pathlib.Path:
    """The parent project's research/decisions when this checkout is mounted there as
    packages/soma-synth (the same rule as soma_synth.pipeline.gates.resolve_governance_root);
    otherwise research/decisions beside the driver, which only a parent checkout has."""
    parent = driver_repo.parent.parent
    if ((driver_repo.name, driver_repo.parent.name) == ("soma-synth", "packages")
            and (parent / "configs" / "evaluators").is_dir()
            and (parent / "research" / "decisions").is_dir()):
        return parent / "research" / "decisions"
    return driver_repo / "research" / "decisions"


DECISIONS = default_decisions(DRIVER_REPO)
AUTH_TEMPLATE = "2026-09-25_{source}_harness_sample_generation_authorization.md"
#: The generation package a code root may hold, under either of its two names.
PACKAGES = ("soma_synth", "soma_synthetic_imu")

SOURCES = ("amass", "prism", "hknu", "gaitex", "addbiomechanics")
POC = pathlib.Path("runs") / "experimental_generation_poc_demo"

#: The production bundles the driver must never write, and whose root it checks after the run.
PRODUCTION_BUNDLES = ("amass_faithful_full", "prism_faithful_full", "hknu_unified8",
                      "gaitex_unified8", "addbio_unified8")
#: Evidence and run records beside the bundles (retention rule 1.4), checked file by file.
EVIDENCE_FOLDERS = ("_superseded", "_runs", "_manifest_backfill")


def _hash_outputs():
    """hash_outputs.py beside this file, for its sample filter (SAMPLES / in_sample), loaded by
    path so the driver works as a script and when a test loads it by path."""
    import importlib.util

    name = "_run_harness_hash_outputs"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).with_name(
            "hash_outputs.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]

# ------------------------------------------------------------------ the sample subjects
AMASS_GROUPS = (("TotalCapture", "s4"), ("CMU", "45"), ("KIT", "63"))
PRISM_SUBJECT = "subj005"
HKNU_SUBJECT = "S04"
GAITEX_SUBJECT = "austra"
ADDBIO_SUBJECTS = (("Hamner2013_Formatted_No_Arm", "subject01"),
                   ("Hammer2013_Formatted_With_Arm", "subject01"),
                   ("Tiziana2019_Formatted_With_Arm", "Subject43"))

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
#: Removed from the inherited environment so every code root resolves every location from
#: SOMA_DATA_ROOT alone. The current generators honour SOMA_SOURCE_ROOT / SOMA_BODY_MODEL_DIR and
#: older ones ignore them, so an inherited value would make the two runs read different inputs;
#: ADDBIO_RETARGET_CORPUS, SOMA_POC_OUT_DIR and the per-file SMPL overrides are honoured by both
#: and would redirect an input or an output.
UNPINNED_VARS = ("ADDBIO_RETARGET_CORPUS", "SOMA_SOURCE_ROOT", "SOMA_BODY_MODEL_DIR",
                 "SOMA_POC_OUT_DIR", "SOMA_SMPL_MODEL_MALE", "SOMA_SMPL_MODEL_FEMALE",
                 "PYTHONHOME", "PYTHONSTARTUP")


class HarnessError(RuntimeError):
    """The driver refused to run, or found that something it must not touch changed."""


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _is_within(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------------ snapshots
def _stat_row(path: pathlib.Path) -> list:
    st = path.stat()
    return [st.st_size, st.st_mtime_ns, path.is_dir()]


def _tree_rows(root: pathlib.Path) -> dict[str, list]:
    rows = {}
    if not root.exists():
        return {"<missing>": [0, 0, False]}
    if root.is_file():
        return {root.name: _stat_row(root)}
    for current, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            path = pathlib.Path(current) / name
            rows[path.relative_to(root).as_posix()] = _stat_row(path)
    return rows


def sample_source_paths(data_root: pathlib.Path) -> dict[str, list[pathlib.Path]]:
    """The source folders each source reads, for the before/after check (read-only inputs)."""
    extracted = data_root / "extracted"
    addbio = []
    for study, subject in ADDBIO_SUBJECTS:
        addbio.extend(sorted((extracted / "addbiomechanics").glob(f"*/*/{study}/{subject}")))
    return {
        "amass": [extracted / "amass" / ds / sub for ds, sub in AMASS_GROUPS],
        "prism": sorted(p for p in (extracted / "prism").iterdir() if p.is_dir()),
        "hknu": [extracted / "hknu_fullbody" / "Dataset_Processed" / HKNU_SUBJECT,
                 extracted / "hknu_fullbody" / "MATLAB" / "DatasetInfo.xlsx"],
        "gaitex": [extracted / "gaitex" / GAITEX_SUBJECT],
        "addbiomechanics": addbio,
    }


def sample_take_rows(bundle: str, root: pathlib.Path) -> dict[str, list]:
    """(size, mtime_ns, is_dir) of every file in the bundle's sample take directories
    (hash_outputs.SAMPLES matched against the bundle's INDEX, as hash_outputs does)."""
    index_path = root / "INDEX.json"
    if not index_path.is_file():
        return {"<no INDEX.json>": [0, 0, False]}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    in_sample = _hash_outputs().in_sample
    rows: dict[str, list] = {}
    for entry in index.get("takes", []):
        if not in_sample(bundle, entry):
            continue
        take = root / entry["rel"]
        if not take.is_dir():
            rows[f"{entry['rel']}/<missing>"] = [0, 0, False]
            continue
        for path in sorted(take.iterdir()):
            if path.is_file():
                rows[f"{entry['rel']}/{path.name}"] = _stat_row(path)
    return rows


def snapshot(data_root: pathlib.Path, sources: list[str]) -> dict:
    """(size, mtime_ns, is_dir) of what the run must leave untouched.

    Every top-level entry of runs/experimental_generation_poc_demo (a generator that fell back to
    its production default would add or touch one); every file at the root of each production
    bundle (INDEX.json among them) and in each of its sample take directories (a generator writing
    into a production take would change no root file); every file under the evidence folders
    _superseded, _runs and _manifest_backfill; the top level of extracted/; and every file in the
    sample source folders.
    """
    poc = data_root / POC
    # os.lstat rather than DirEntry.stat: on Windows the latter reports the copy of a directory's
    # timestamps kept in its parent's index, which NTFS updates lazily
    top = {entry.name: [os.lstat(entry.path).st_size, os.lstat(entry.path).st_mtime_ns,
                        entry.is_dir()]
           for entry in sorted(os.scandir(poc), key=lambda e: e.name)}
    bundles, takes = {}, {}
    for name in PRODUCTION_BUNDLES:
        root = poc / name
        if root.is_dir():
            bundles[name] = {p.name: _stat_row(p) for p in sorted(root.iterdir()) if p.is_file()}
            takes.update({f"{name}/{k}": v for k, v in sample_take_rows(name, root).items()})
        else:
            bundles[name] = {"<missing>": [0, 0, False]}
    evidence = {}
    for name in EVIDENCE_FOLDERS:
        evidence.update({f"{name}/{k}": v for k, v in _tree_rows(poc / name).items()})
    source_rows = {}
    for source in sources:
        for path in sample_source_paths(data_root)[source]:
            source_rows[str(path)] = _tree_rows(path)
    # os.lstat for the reason given for poc_demo_top above: DirEntry.stat reads the parent's
    # lazily updated index copy of a directory's timestamps on Windows
    extracted_top = {entry.name: [os.lstat(entry.path).st_mtime_ns, entry.is_dir()]
                     for entry in sorted(os.scandir(data_root / "extracted"), key=lambda e: e.name)}
    return {"poc_demo_top": top, "production_bundle_roots": bundles,
            "production_sample_takes": takes, "evidence_folders": evidence,
            "sample_sources": source_rows, "extracted_top": extracted_top}


def diff_snapshots(before: dict, after: dict) -> list[str]:
    problems = []
    for section, a in before.items():
        b = after.get(section, {})
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                problems.append(f"{section}: {key}: before={a.get(key)} after={b.get(key)}")
    return problems


def tree_bytes(root: pathlib.Path) -> int:
    total = 0
    for current, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += (pathlib.Path(current) / name).stat().st_size
            except OSError:
                pass
    return total


# ---------------------------------------------------------------------- authorization
def authorization_for(source: str, decisions: pathlib.Path = DECISIONS) -> pathlib.Path:
    """The authorization record for this source's sample run, checked for the scope it names."""
    path = decisions / AUTH_TEMPLATE.format(source=source)
    if not path.is_file():
        raise HarnessError(f"{source}: no harness authorization record at {path}; refusing "
                           "(the records live in the parent project: pass its research/decisions "
                           "with --decisions-dir)")
    # Separators are normalised, so a record may spell the scratch folder either way.
    text = path.read_text(encoding="utf-8").replace("\\", "/")
    for needle in (f"source_name: {source}", "tmp/harness", "releases_holds: false",
                   "experimental_non_candidate"):
        if needle not in text:
            raise HarnessError(f"{source}: {path.name} does not state {needle!r}; refusing")
    return path


# ------------------------------------------------------------------------- staging
def _copy_file(src: pathlib.Path, dst: pathlib.Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if dst.stat().st_size != src.stat().st_size:
        raise HarnessError(f"copy of {src} is short")
    return dst.stat().st_size


def stage_amass(data_root: pathlib.Path, scratch: pathlib.Path) -> tuple[pathlib.Path, int]:
    """Whole fit groups: every npz of each sample subject, in the source's own layout."""
    root = scratch / "src" / "amass"
    copied = 0
    for dataset, subject in AMASS_GROUPS:
        folder = data_root / "extracted" / "amass" / dataset / subject
        files = sorted(p for p in folder.iterdir() if p.is_file())
        if not files:
            raise HarnessError(f"amass: no files in {folder}")
        for path in files:
            copied += _copy_file(path, root / dataset / subject / path.name)
    return root, copied


def stage_prism(data_root: pathlib.Path, scratch: pathlib.Path) -> tuple[pathlib.Path, int]:
    """subj005 whole, plus the first take of every other subject (the gender scan reads it)."""
    data = scratch / "prismroot"
    source = data_root / "extracted" / "prism"
    copied = 0
    for subject_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        takes = sorted(subject_dir.glob("*.pkl"))
        if not takes:
            continue
        chosen = takes if subject_dir.name == PRISM_SUBJECT else takes[:1]
        for path in chosen:
            copied += _copy_file(path, data / "extracted" / "prism" / subject_dir.name / path.name)
    if not (data / "extracted" / "prism" / PRISM_SUBJECT).is_dir():
        raise HarnessError(f"prism: {PRISM_SUBJECT} not found under {source}")
    return data, copied


def stage_addbio(data_root: pathlib.Path, scratch: pathlib.Path) -> tuple[pathlib.Path, int]:
    """The sample subjects' b3d files under <split>/<variant>/<study>/<subject>/."""
    source = data_root / "extracted" / "addbiomechanics"
    root = scratch / "src" / "addbiomechanics"
    copied = 0
    for study, subject in ADDBIO_SUBJECTS:
        folders = sorted(source.glob(f"*/*/{study}/{subject}"))
        if len(folders) != 1:
            raise HarnessError(f"addbiomechanics: expected one folder for {study}/{subject}, "
                               f"found {[str(f) for f in folders]}")
        for path in sorted(folders[0].glob("*.b3d")):
            copied += _copy_file(path, root / path.relative_to(source))
    return root, copied


# --------------------------------------------------------------------------- the run
class Driver:
    def __init__(self, args: argparse.Namespace) -> None:
        self.code = pathlib.Path(args.code_root).resolve()
        self.scratch = pathlib.Path(args.scratch).resolve()
        self.data_root = pathlib.Path(args.data_root).resolve()
        self.python = args.python
        self.sources = [s.strip() for s in args.sources.split(",") if s.strip()]
        self.record_dir = pathlib.Path(args.record_dir or (self.scratch / "_harness")).resolve()
        self.decisions = pathlib.Path(getattr(args, "decisions_dir", None) or DECISIONS).resolve()
        self.logs = self.record_dir / "logs"
        self.poc = self.scratch / POC
        self.package = next((name for name in PACKAGES if (self.code / "src" / name).is_dir()),
                            PACKAGES[0])
        self.record: dict = {
            "what": "byte-identity harness run (sample subjects only; no validate/readme/register/push)",
            "started_utc": _now(), "code_root": str(self.code), "scratch": str(self.scratch),
            "data_root": str(self.data_root), "python": self.python, "package": self.package,
            "decisions_dir": str(self.decisions), "sources": {},
            "scratch_bytes_after_step": [], "failures": [],
        }
        self.peak = 0

    def display(self, path: pathlib.Path) -> str:
        """A record path relative to the checkout that holds it (the driver's, or the one the
        decisions directory belongs to), else absolute."""
        bases = [DRIVER_REPO, *self.decisions.parents[1:2]]
        for base in bases:
            try:
                return path.resolve().relative_to(base.resolve()).as_posix()
            except ValueError:
                continue
        return path.resolve().as_posix()

    # -- guards
    def check_layout(self) -> None:
        unknown = sorted(set(self.sources) - set(SOURCES))
        if unknown:
            raise HarnessError(f"unknown sources {unknown}; known: {', '.join(SOURCES)}")
        harness_root = self.data_root / "tmp" / "harness"
        if not _is_within(self.scratch, harness_root) or self.scratch == harness_root.resolve():
            raise HarnessError(f"scratch {self.scratch} is not a run directory under {harness_root}; "
                               "the authorization records allow nothing else")
        if self.scratch.exists() and any(p.name != "_harness" for p in self.scratch.iterdir()):
            raise HarnessError(f"scratch {self.scratch} is not empty; delete it or pick a new run name "
                               "(a resumable generator would skip what is already there)")
        for rel in ("scripts/poc", f"src/{self.package}", "packages/smpl18/src/smpl18",
                    "scripts/write_known_limitations.py"):
            if not (self.code / rel).exists():
                raise HarnessError(f"code root {self.code} has no {rel}")
        if _is_within(self.code, self.scratch):
            raise HarnessError("the code root must not sit inside the scratch directory")
        # the logs are written before the after-snapshot, so a record directory in the production
        # tree would itself change what the snapshot guards
        if _is_within(self.record_dir, self.data_root) and not _is_within(self.record_dir,
                                                                           self.scratch):
            raise HarnessError(f"--record-dir {self.record_dir} is under the data root "
                               f"{self.data_root} but outside the scratch directory; put it in "
                               "the scratch directory (the default) or outside the data root")

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        for name in UNPINNED_VARS:
            env.pop(name, None)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.code / "src"), str(self.code / "packages" / "smpl18" / "src"), str(self.code)])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        for name in THREAD_VARS:
            env[name] = "1"
        env["SOMA_DATA_ROOT"] = str(self.data_root)
        return env

    def preflight(self) -> dict:
        """Which interpreter and which copies of the two packages the generators will import."""
        probe = (
            f"import json, sys, numpy, scipy, smpl18, {self.package}\n"
            "print(json.dumps({'python': sys.version.split()[0], 'executable': sys.executable,"
            " 'numpy': numpy.__version__, 'scipy': scipy.__version__,"
            " 'smpl18': smpl18.__file__, 'smpl18_version': getattr(smpl18, '__version__', None),"
            f" '{self.package}': {self.package}.__file__}}))\n"
        )
        done = subprocess.run([self.python, "-c", probe], env=self.env(), cwd=str(self.code),
                              capture_output=True, text=True, encoding="utf-8", check=False)
        if done.returncode != 0:
            raise HarnessError(f"preflight failed: {done.stderr.strip()}")
        info = json.loads(done.stdout.strip().splitlines()[-1])
        for key in ("smpl18", self.package):
            if not _is_within(pathlib.Path(info[key]), self.code):
                raise HarnessError(f"{key} imports from {info[key]}, not from the code root {self.code}")
        return info

    # -- execution
    def note_size(self, label: str) -> None:
        size = tree_bytes(self.scratch)
        self.peak = max(self.peak, size)
        self.record["scratch_bytes_after_step"].append([label, size])

    def run_step(self, source: str, label: str, argv: list[str]) -> dict:
        self.logs.mkdir(parents=True, exist_ok=True)
        log_path = self.logs / f"{source}__{label}.log"
        print(f"  [{source}] {label}: {' '.join(argv)}", flush=True)
        started = time.perf_counter()
        with open(log_path, "w", encoding="utf-8", newline="\n") as log:
            log.write("argv: " + json.dumps(argv, ensure_ascii=False) + "\n\n")
            log.flush()
            code = subprocess.run(argv, env=self.env(), cwd=str(self.code), stdout=log,
                                  stderr=subprocess.STDOUT, check=False).returncode
        seconds = round(time.perf_counter() - started, 1)
        self.note_size(f"{source}:{label}")
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-3:]
        print(f"      exit {code} in {seconds}s; {' | '.join(t.strip() for t in tail)[:300]}", flush=True)
        step = {"step": label, "argv": argv, "exit": code, "seconds": seconds, "log": str(log_path)}
        if code != 0:
            self.record["failures"].append(f"{source}:{label} exited {code}")
        return step

    def script(self, *parts: str) -> str:
        return str(self.code.joinpath(*parts))

    def limitations(self, source: str, bundle: pathlib.Path) -> dict:
        return self.run_step(source, "known_limitations",
                             [self.python, self.script("scripts", "write_known_limitations.py"),
                              str(bundle)])

    def run_source(self, source: str) -> dict:
        extracted = self.data_root / "extracted"
        py = self.python
        steps: list[dict] = []
        staged: dict = {}
        started = time.perf_counter()
        if source == "amass":
            root, copied = stage_amass(self.data_root, self.scratch)
            staged = {"root": str(root), "bytes": copied}
            bundle = self.poc / "amass_faithful_full"
            steps.append(self.run_step(source, "bundle", [
                py, self.script("scripts", "poc", "generate_amass_faithful_all.py"),
                "--amass-root", str(root), "--out-root", str(bundle)]))
        elif source == "prism":
            data, copied = stage_prism(self.data_root, self.scratch)
            staged = {"root": str(data), "bytes": copied}
            bundle = data / POC / "prism_faithful_full"
            steps.append(self.run_step(source, "bundle", [
                py, self.script("scripts", "poc", "generate_prism_measured_all.py"),
                "--data-root", str(data), "--only", f"prism_{PRISM_SUBJECT}"]))
            # the PRISM bundle's heading companion is written by this post-step
            steps.append(self.run_step(source, "insole_heading", [
                py, self.script("scripts", "poc", "synthesize_insole_heading.py"), str(bundle)]))
        elif source == "hknu":
            corpus = self.poc / "hknu_smpl24_paired"
            bundle = self.poc / "hknu_unified8"
            steps.append(self.run_step(source, "corpus", [
                py, self.script("scripts", "poc", "generate_hknu_faithful.py"),
                "--root", str(extracted / "hknu_fullbody"), "--out", str(corpus),
                "--subjects", HKNU_SUBJECT]))
            steps.append(self.run_step(source, "bundle", [
                py, self.script("scripts", "poc", "generate_hknu_unified8.py"),
                "--out", str(bundle), "--paired", str(corpus), "--subjects", HKNU_SUBJECT]))
        elif source == "gaitex":
            corpus = self.poc / "gaitex_smpl24"
            bundle = self.poc / "gaitex_unified8"
            steps.append(self.run_step(source, "corpus", [
                py, self.script("scripts", "poc", "generate_gaitex_smpl24.py"),
                "--source-root", str(extracted / "gaitex"), "--out", str(corpus),
                "--subjects", GAITEX_SUBJECT]))
            steps.append(self.run_step(source, "bundle", [
                py, self.script("scripts", "poc", "generate_gaitex_unified8.py"),
                "--out", str(bundle), "--retarget", str(corpus),
                "--extracted", str(extracted / "gaitex"), "--subjects", GAITEX_SUBJECT]))
        elif source == "addbiomechanics":
            root, copied = stage_addbio(self.data_root, self.scratch)
            staged = {"root": str(root), "bytes": copied}
            corpus = self.poc / "addbio_smpl24_raw"
            bundle = self.poc / "addbio_unified8"
            steps.append(self.run_step(source, "corpus", [
                py, self.script("scripts", "poc", "generate_addbio_smpl24.py"),
                "--source-root", str(root), "--out", str(corpus)]))
            steps.append(self.run_step(source, "bundle", [
                py, self.script("scripts", "poc", "generate_addbio_unified8.py"),
                "--out", str(bundle), "--raw", str(corpus)]))
        else:  # pragma: no cover - check_layout refuses it first
            raise HarnessError(source)
        if bundle.is_dir():
            steps.append(self.limitations(source, bundle))
        else:
            self.record["failures"].append(f"{source}: bundle {bundle} was not written")
        return {"bundle": str(bundle), "staged": staged, "steps": steps,
                "seconds": round(time.perf_counter() - started, 1)}

    def main(self) -> int:
        self.check_layout()
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        auth = {source: authorization_for(source, self.decisions) for source in self.sources}
        print(f"harness run -> {self.scratch}", flush=True)
        print(f"code root   : {self.code} (package {self.package})", flush=True)
        for source, path in auth.items():
            print(f"authorized  : {source:<16} by {self.display(path)}", flush=True)
        self.record["authorization"] = {s: self.display(p) for s, p in auth.items()}
        self.record["environment"] = {k: v for k, v in self.env().items()
                                      if k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTHONUTF8",
                                               "SOMA_DATA_ROOT", *THREAD_VARS)}
        # which of the removed variables the operator's environment carried, and their values
        self.record["environment"]["removed_from_inherited"] = {
            name: os.environ[name] for name in UNPINNED_VARS if name in os.environ}
        self.record["preflight"] = self.preflight()
        print(f"interpreter : {json.dumps(self.record['preflight'])}", flush=True)

        before = snapshot(self.data_root, self.sources)
        started = time.perf_counter()
        try:
            for source in self.sources:
                print(f"== {source} (authorized by {auth[source].name})", flush=True)
                self.record["sources"][source] = self.run_source(source)
                self.record["sources"][source]["authorization"] = self.record["authorization"][source]
        finally:
            self.record["total_seconds"] = round(time.perf_counter() - started, 1)
            after = snapshot(self.data_root, self.sources)
            problems = diff_snapshots(before, after)
            self.record["untouched_check"] = "unchanged" if not problems else problems
            self.record["scratch_peak_bytes"] = self.peak
            self.record["finished_utc"] = _now()
            out = self.record_dir / "HARNESS_RUN.json"
            out.write_text(json.dumps(self.record, indent=1, ensure_ascii=False) + "\n",
                           encoding="utf-8", newline="\n")
            print(f"record      : {out}", flush=True)
        if problems:
            print("!!! PRODUCTION OR SOURCE STATE CHANGED DURING THE HARNESS RUN !!!", flush=True)
            for line in problems[:50]:
                print("    " + line, flush=True)
            return 3
        print("untouched   : poc_demo top level, production bundle roots and sample takes, "
              "evidence folders and sample sources unchanged", flush=True)
        print(f"total {self.record['total_seconds']}s, scratch peak {self.peak / 2**30:.3f} GiB, "
              f"failures {self.record['failures'] or 'none'}", flush=True)
        return 1 if self.record["failures"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--code-root", required=True,
                        help="checkout whose generators run (an export of a commit, or a worktree)")
    parser.add_argument("--scratch", required=True,
                        help="run directory under <SOMA_DATA_ROOT>/tmp/harness; must be new or empty")
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", default=os.environ.get("SOMA_DATA_ROOT"),
                        help="the real data root (sources and production bundles); default SOMA_DATA_ROOT")
    parser.add_argument("--record-dir", default=None,
                        help="where HARNESS_RUN.json and the step logs go (default <scratch>/_harness)")
    parser.add_argument("--decisions-dir", default=None,
                        help="the parent project's research/decisions, which holds the five "
                             "harness authorization records (default: the parent project's, when "
                             "soma-synth is mounted there as packages/soma-synth)")
    args = parser.parse_args(argv)
    if not args.data_root:
        parser.error("--data-root is required when SOMA_DATA_ROOT is not set")
    try:
        return Driver(args).main()
    except HarnessError as error:
        print(f"REFUSED: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
