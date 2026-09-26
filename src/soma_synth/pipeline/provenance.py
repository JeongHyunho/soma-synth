"""Collect the ten provenance items ``PIPELINE_GOVERNANCE.md`` §10.2 requires of every run.

A take manifest alone does not carry the code revision, the SMPL model hash, the host, or the
input/output artifact hashes, so a bundle cannot be reproduced from its manifests -- they do not
say which code or which body model made it. The run record carries all ten.

Two rules shape this module.

**Nothing is invented.** An item that cannot be measured is recorded as unavailable with a
reason, never guessed. A fabricated provenance field is worse than an absent one, because it
reads as evidence.

**Absence is stated, not left blank.** §10.2 requires seeds; this generation path has no
randomness at all across the production generators and package, excluding the non-generation
``experimental`` namespace. Generation imports of that namespace are refused. The seed entry
says so explicitly, and
``assert_generation_path_is_deterministic`` guards the claim. A blank cannot distinguish "there
is none" from "we did not write it down".

Credentials, tokens, cookies and personal identifiers are never recorded (§10.2).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

#: Declared in pyproject.toml; recorded so a run says what it was built against. openpyxl: the
#: HKNU generators read the subject workbook through it.
_TRACKED_DISTRIBUTIONS = (
    "numpy", "scipy", "lxml", "protobuf", "pypdf", "python-docx", "PyYAML", "openpyxl",
)

#: Anything matching these in the generation path would be an unrecorded source of variation.
_RNG_PATTERNS = (
    re.compile(r"\bnp\.random\b"),
    re.compile(r"\bnumpy\.random\b"),
    re.compile(r"\bdefault_rng\b"),
    re.compile(r"\bRandomState\b"),
    re.compile(r"(?<![\w.])random\.[a-z_]+\("),
)

UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Unavailable:
    """A provenance item that could not be measured, and why. Never a guess."""

    reason: str

    def as_json(self) -> dict[str, str]:
        return {"status": UNAVAILABLE, "reason": self.reason}


def _run_git(*arguments: str, root: Path | None = None) -> str | None:
    """Return git output, or None. A missing or unhappy git is an unavailable item, not a crash."""
    executable = os.environ.get("SOMA_GIT_EXECUTABLE") or "git"
    try:
        completed = subprocess.run(
            [executable, "-C", str(root or _REPOSITORY_ROOT), *arguments],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
    except (OSError, ValueError):
        return None
    if completed.returncode != 0:
        return None
    # Only trailing newlines are stripped: `git status --porcelain` starts a line with a space
    # (" M path"), and stripping that space would shift the first path by one character.
    return completed.stdout.rstrip("\n")


#: What ``patch_sha256`` covers. Untracked files are named in ``dirty_paths`` but not hashed.
PATCH_SCOPE = "git diff HEAD: tracked files only; untracked files are named in dirty_paths, not hashed"


def _git_revision(root: Path | None = None) -> dict[str, object] | None:
    """{commit, dirty[, dirty_paths, patch_sha256]} of the checkout at ``root``, None if git fails."""
    head = _run_git("rev-parse", "HEAD", root=root)
    if head is None:
        return None
    head = head.strip()
    porcelain = _run_git("status", "--porcelain", root=root) or ""
    dirty = bool(porcelain.strip())
    record: dict[str, object] = {"commit": head, "dirty": dirty}
    if dirty:
        diff = _run_git("diff", "HEAD", root=root) or ""
        record["dirty_paths"] = sorted(
            line[3:].strip() for line in porcelain.splitlines() if len(line) > 3
        )
        record["patch_sha256"] = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        record["patch_scope"] = PATCH_SCOPE
    return record


def code_revision() -> dict[str, object]:
    """§10.2 item 1: the commit, and a hash of the working tree's deviation from it.

    A dirty tree is not refused here -- §10.3 leaves that to the caller -- but the patch is
    hashed so a run made from an uncommitted tree can at least be told apart from one made from
    the commit it claims.
    """
    record = _git_revision()
    if record is None:
        return Unavailable("git did not answer; is SOMA_GIT_EXECUTABLE set?").as_json()
    return record


def _files_digest(files: dict[str, object]) -> str:
    lines = "\n".join(f"{name} {files[name]}" for name in sorted(files))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def smpl18_revision(package_dir: Path | str | None = None) -> dict[str, object]:
    """The ``smpl18`` the run imported: the commit of its checkout, else its version and files.

    ``smpl18`` is the second code base a bundle depends on (the SMPL-24 conversion, the model
    selection and the 18-joint reduction), mounted at ``packages/smpl18`` as a submodule. When
    the imported package sits in a git checkout (``<checkout>/src/smpl18`` with
    ``<checkout>/.git``), its HEAD, dirty flag and patch hash are recorded, like
    :func:`code_revision`; an installed copy has no checkout, so its version and the sha256 of
    every file of the package are recorded instead. ADR-0041 asks every run record for it next to
    the code revision, so a withdrawn or corrected conversion can be traced to the bundles it made.
    """
    version: object
    if package_dir is None:
        try:
            import smpl18
        except ImportError as error:  # pragma: no cover - the pipeline cannot run without it
            return Unavailable(f"smpl18 is not importable: {error}").as_json()
        package = Path(smpl18.__file__).resolve().parent
        version = getattr(smpl18, "__version__", None)
    else:
        package = Path(package_dir).resolve()
        version = None
    if version is None:
        import importlib.metadata as md

        try:
            version = md.version("smpl18")
        except md.PackageNotFoundError:
            version = Unavailable("smpl18 declares no version").as_json()

    record: dict[str, object] = {"package": "smpl18", "version": version}
    checkout = package.parent.parent
    if package.parent.name == "src" and (checkout / ".git").exists():
        revision = _git_revision(checkout)
        if revision is not None:
            try:
                record["checkout"] = checkout.relative_to(_REPOSITORY_ROOT.resolve()).as_posix()
            except ValueError:
                record["checkout"] = "outside this repository"
            record["source"] = "git"
            record.update(revision)
            return record
    files = {
        path.relative_to(package).as_posix(): file_sha256(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    record["source"] = "package_files"
    record["files"] = files
    record["files_digest"] = _files_digest(files)
    record["files_digest_algorithm"] = "sha256(sorted '<relative path> <sha256>' lines)"
    return record


def dependency_versions() -> dict[str, object]:
    """§10.2 item 5."""
    import importlib.metadata as md

    versions: dict[str, object] = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
    }
    for name in _TRACKED_DISTRIBUTIONS:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = Unavailable("not installed").as_json()
    return versions


def host_environment() -> dict[str, object]:
    """§10.2 item 8 -- non-sensitive only. No user, no hostname, no paths."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor_count": os.cpu_count(),
    }


def file_sha256(path: Path | str) -> str | dict[str, str]:
    """§10.2 items 3, 6 and 9 all reduce to this."""
    target = Path(path)
    if not target.is_file():
        return Unavailable(f"not a file: {target.name}").as_json()
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(path: Path | str | None, root: Path | str | None) -> str | None:
    """``path`` relative to ``root`` with forward slashes, or None when either is missing or the
    path lies outside the root. Recorded beside an absolute path so that a run record read on
    another PC (another ``SOMA_DATA_ROOT``) still says where under the data root a thing was."""
    if path is None or root is None:
        return None
    target, base = Path(path), Path(root)
    try:
        relative = target.relative_to(base)
    except ValueError:
        try:
            relative = target.resolve().relative_to(base.resolve())
        except (ValueError, OSError):
            return None
    return relative.as_posix()


#: Keys that name a source file in a take manifest, and the keys that carry its hash beside them.
#: Every bundle names its input as
#: ``source.relative_path`` with ``source.source_asset_sha256`` (PRISM:
#: ``source_asset_sha256_take``); GAITEX also names the extracted marker and IMU files it
#: synthesised small from (``small_synthesis.inputs.*``); HKNU names the corpus it paired with
#: (``paired_measured_counterpart``). ``source_asset_id`` is what the npz files call the same
#: reference, accepted here so a manifest that carries it is not missed.
SOURCE_REFERENCE_KEYS = ("relative_path", "source_asset_id")
SOURCE_HASH_KEYS = ("sha256", "source_asset_sha256", "source_asset_sha256_take")
SOURCE_MANIFEST_SCHEMA = "source_manifest_v1"
SOURCE_MANIFEST_AGGREGATE = "sha256 over the sorted lines '<relative_path>\\t<sha256 or empty>', joined by '\\n'"


def take_source_references(manifest: object) -> list[dict[str, object]]:
    """Every source reference in one take manifest: ``{field, relative_path, sha256}``.

    A mapping that carries one of :data:`SOURCE_REFERENCE_KEYS` as a non-empty string is a
    reference; its hash is the first of :data:`SOURCE_HASH_KEYS` it carries (None when none).
    ``field`` is the dotted path of the mapping in the manifest.
    """
    found: list[dict[str, object]] = []

    def walk(node: object, where: str) -> None:
        if isinstance(node, dict):
            reference = next((node[key] for key in SOURCE_REFERENCE_KEYS
                              if isinstance(node.get(key), str) and node.get(key)), None)
            if reference is not None:
                digest = next((node[key] for key in SOURCE_HASH_KEYS
                               if isinstance(node.get(key), str) and node.get(key)), None)
                found.append({"field": where or "<root>", "relative_path": reference,
                              "sha256": digest})
            for key, value in node.items():
                walk(value, f"{where}.{key}" if where else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{where}[{index}]")

    walk(manifest, "")
    return found


def _bundle_take_dirs(dataset_dir: Path) -> list[tuple[str, Path]]:
    """(take name, take directory) for every take the bundle lists, in a stable order."""
    index_path = dataset_dir / "INDEX.json"
    names: list[str] = []
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            index = {}
        takes = index.get("takes") if isinstance(index, dict) else None
        if isinstance(takes, list):
            for entry in takes:
                if isinstance(entry, dict):
                    name = entry.get("rel") or entry.get("relative_path") or entry.get("take_id")
                    if isinstance(name, str) and name:
                        names.append(name)
    if not names and dataset_dir.is_dir():
        names = [child.name for child in dataset_dir.iterdir()
                 if child.is_dir() and (child / "manifest.json").is_file()]
    return [(name, dataset_dir / name) for name in sorted(set(names))]


def source_manifest(dataset_dir: Path | str, data_root: Path | str | None = None) -> dict[str, object]:
    """The source files a bundle's take manifests name, with their hashes, for licence tracing.

    ADR-0041 lets teammates generate on their own PCs without a central catalog; when a source's
    licence is withdrawn, each run record has to say which source files its bundle was made from.
    This reads every take manifest the bundle lists (INDEX.json, else every take directory with a
    manifest), collects the references :func:`take_source_references` finds, and returns them
    de-duplicated by (relative_path, sha256) with the takes and fields that name each one, a count
    and one aggregate digest (:data:`SOURCE_MANIFEST_AGGREGATE`).

    What it can say is what the manifests say. A bundle built from a retarget corpus names the
    corpus file (``runs/.../<corpus>/...npz``) as its source; the corpus's own record names the
    extracted source, and the run record carries the corpus fingerprint beside this list.
    """
    bundle = Path(dataset_dir)
    references: dict[tuple[str, str], dict[str, object]] = {}
    scanned = 0
    unreadable: list[str] = []
    fields: set[str] = set()
    for name, take in _bundle_take_dirs(bundle):
        manifest_path = take / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            unreadable.append(f"{name}: {type(error).__name__}")
            continue
        scanned += 1
        for found in take_source_references(manifest):
            key = (str(found["relative_path"]), str(found["sha256"] or ""))
            entry = references.setdefault(key, {
                "relative_path": key[0], "sha256": found["sha256"], "fields": set(), "takes": set(),
            })
            entry["fields"].add(found["field"])            # type: ignore[union-attr]
            entry["takes"].add(name)                       # type: ignore[union-attr]
            fields.add(str(found["field"]))
    ordered = [references[key] for key in sorted(references)]
    lines = "\n".join(f"{path}\t{digest}" for path, digest in sorted(references))
    relative = relative_to_root(bundle, data_root)
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "note": (
            "Source references read from the take manifests of the bundle this run generated; "
            "kept in the run record so a licence withdrawal can be traced to this bundle "
            "(ADR-0041, PIPELINE_GOVERNANCE.md section 12)."
        ),
        # the absolute path only when the bundle has no data-root-relative form: teammates' run
        # records are collected, and the relative form is what another PC can read
        **({"dataset_dir": str(bundle)} if relative is None else {}),
        "dataset_dir_relative": relative,
        "manifests_scanned": scanned,
        "manifests_unreadable": unreadable,
        "reference_fields": sorted(fields),
        "count": len(ordered),
        "aggregate_sha256": hashlib.sha256(lines.encode("utf-8")).hexdigest(),
        "aggregate_algorithm": SOURCE_MANIFEST_AGGREGATE,
        "references": [
            {
                "relative_path": entry["relative_path"],
                "sha256": entry["sha256"],
                "fields": sorted(entry["fields"]),         # type: ignore[arg-type]
                "takes": sorted(entry["takes"]),           # type: ignore[arg-type]
            }
            for entry in ordered
        ],
    }


def body_model_hash(model_path: Path | str | None) -> dict[str, object]:
    """§10.2 item 6. Absent from every bundle measured, and the heaviest of the gaps: every
    source's `large` and `anthro` come out of this file."""
    if model_path is None:
        return Unavailable("no body model path supplied").as_json()
    return {"path": Path(model_path).name, "sha256": file_sha256(model_path)}


def seeds() -> dict[str, object]:
    """§10.2 item 7 -- and the one item with nothing to fill in.

    See the module docstring. The generation path is deterministic by construction, so this
    records the absence rather than inventing a value or leaving a hole.
    """
    return {"present": False, "reason": "no stochastic step in the generation path"}


def _iter_generation_sources() -> Iterable[Path]:
    scripts = _REPOSITORY_ROOT / "scripts" / "poc"
    if scripts.is_dir():
        yield from sorted(scripts.glob("generate_*.py"))
        emitter = scripts / "unified8_emit.py"
        if emitter.exists():
            yield emitter
    package = _REPOSITORY_ROOT / "src" / "soma_synth"
    if package.is_dir():
        yield from (
            path for path in sorted(package.rglob("*.py"))
            if not path.is_relative_to(package / "experimental")
        )


def _assert_no_experimental_imports() -> None:
    """Keep excluded research modules out of production's static import graph.

    This is a source audit, not a sandbox for arbitrary dynamic code. Absolute
    literal dynamic imports are checked too; other dynamic targets require review.
    Read or parse errors propagate instead of certifying an unaudited file.
    """
    prefix = "soma_synth.experimental"
    for path in _iter_generation_sources():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        parts = path.relative_to(_REPOSITORY_ROOT).parts
        scope = parts[1:-1] if parts[0] == "src" else parts[:-1]
        dynamic_names = {"import_module", "__import__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "importlib":
                dynamic_names.update(
                    alias.asname or alias.name for alias in node.names
                    if alias.name == "import_module"
                )
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                parent = scope[:len(scope) - node.level + 1] if node.level else ()
                module = ".".join((*parent, node.module or "")).rstrip(".")
                modules = [module, *(f"{module}.{alias.name}" for alias in node.names)]
            elif isinstance(node, ast.Call):
                name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                target = node.args[0] if node.args else next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "name"), None,
                )
                if name in dynamic_names and isinstance(target, ast.Constant):
                    value = target.value
                    if isinstance(value, str):
                        modules = [value]
            if any(module == prefix or module.startswith(prefix + ".") for module in modules):
                raise AssertionError(
                    f"generation path imports experimental code: "
                    f"{path.relative_to(_REPOSITORY_ROOT)}:{node.lineno}"
                )


def find_rng_uses() -> list[tuple[str, int, str]]:
    """Every apparent use of randomness in the generation path. Expected to be empty."""
    found: list[tuple[str, int, str]] = []
    for path in _iter_generation_sources():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if path.name == Path(__file__).name:
            continue  # this module names the patterns it looks for
        for number, line in enumerate(text.splitlines(), 1):
            if any(pattern.search(line) for pattern in _RNG_PATTERNS):
                found.append((str(path.relative_to(_REPOSITORY_ROOT)), number, line.strip()))
    return found


def assert_generation_path_is_deterministic() -> None:
    """Fail if randomness entered the generation path, so ``seeds()`` cannot become a lie."""
    _assert_no_experimental_imports()
    uses = find_rng_uses()
    if uses:
        rendered = "\n".join(f"  {p}:{n}  {line}" for p, n, line in uses)
        raise AssertionError(
            "seeds() claims the generation path has no stochastic step, but randomness was "
            f"found:\n{rendered}\nEither remove it or record the seed and revise seeds()."
        )


@dataclass
class ProvenanceRecord:
    """The ten items, assembled. Artifact hashes accumulate as takes are written."""

    source_name: str
    spec_id: str
    spec_version: str
    body_model_path: Path | None = None
    #: The registry's ``BodyModelSet`` for this source and the directory it resolves to. Set by
    #: the runner; a run given both records every model in the set rather than a single file,
    #: because the generators choose one per subject.
    body_model_set: object | None = None
    body_model_dir: Path | None = None
    #: How the generators pick one model out of the set, as the registry declares it.
    body_model_selection: str = ""
    source_assets: dict[str, object] = field(default_factory=dict)
    output_artifacts: dict[str, object] = field(default_factory=dict)
    quality: dict[str, object] = field(default_factory=dict)
    #: The run's data root. When set, recorded paths also carry their data-root-relative form.
    data_root: Path | None = None

    def record_source_asset(self, key: str, path: Path | str) -> None:
        entry: dict[str, object] = {"path": str(path), "sha256": file_sha256(path)}
        if self.data_root is not None:
            entry["relative_path"] = relative_to_root(path, self.data_root)
        self.source_assets[key] = entry

    def record_output_artifact(self, key: str, path: Path | str) -> None:
        self.output_artifacts[key] = file_sha256(path)

    def body_model_record(self) -> dict[str, object]:
        """§10.2 item 6: the model set when the run knows one, else the single path, else why not."""
        if self.body_model_set is not None:
            from soma_synth.pipeline import body_models

            return body_models.set_record(
                self.body_model_dir,
                set_name=self.body_model_set.name,
                selection=self.body_model_selection,
                resolver=self.body_model_set.resolver,
                data_root=self.data_root,
            )
        return body_model_hash(self.body_model_path)

    def as_json(self) -> dict[str, object]:
        return {
            "governance_reference": "PIPELINE_GOVERNANCE.md section 10.2",
            "code_revision": code_revision(),
            # beside item 1: the conversion package is the other half of the code (ADR-0041)
            "smpl18_revision": smpl18_revision(),
            "contract_versions": {"spec_id": self.spec_id, "spec_version": self.spec_version},
            "source_assets": self.source_assets
            or Unavailable("no source asset recorded for this run").as_json(),
            "spec_snapshot": Unavailable(
                "filled by the caller from the take manifest's spec_documents"
            ).as_json(),
            "dependency_versions": dependency_versions(),
            "body_model": self.body_model_record(),
            "seeds": seeds(),
            "host_environment": host_environment(),
            "output_artifacts": self.output_artifacts
            or Unavailable("no output artifact recorded for this run").as_json(),
            "quality": self.quality
            or Unavailable("filled by close_run from the validation report").as_json(),
        }
