"""Where the viewer and its tools find data, models and the presentation package.

Data locations come from ``SOMA_DATA_ROOT`` (a clear error when it is unset) or explicit arguments;
the viewer lists the bundles under ``<SOMA_DATA_ROOT>/runs/experimental_generation_poc_demo`` and a
folder it is given, and scans no sibling folder. ``viewer_app.py`` runs inside Blender, so its
rules live in ``viewer_paths.py``, which is tested here without bpy.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
POC = REPO / "scripts" / "poc"
if str(POC) not in sys.path:
    sys.path.insert(0, str(POC))

import viewer_paths  # noqa: E402

VIEWER_FILES = ("viewer_paths.py", "viewer_app.py", "viewer_launch.py", "enrich_root_translation.py",
                "export_anim_data.py", "blender_prism_anim.py", "prism_to_smplx.py",
                "probe_align.py")
ENV = ("SOMA_DATA_ROOT", "SOMA_SOURCE_ROOT", "SOMA_BODY_MODEL_DIR", "SOMA_VIEWER_DATASET",
       "SOMA_VIEWER_DATA", "SOMA_VIEWER_MODEL", "SOMA_VIEWER_MODEL_MALE",
       "SOMA_VIEWER_MODEL_FEMALE", "SOMA_VIEWER_MODEL_NEUTRAL", "SOMA_ANIM_DIR")
#: A user profile or a drive letter (the Blender install default under "C:\Program Files" is a
#: program location, not data).
HARDCODED_LOCATION = re.compile(r"[\\/]Users[\\/]|(?<![A-Za-z0-9])[A-Za-z]:[\\/](?!Program Files)")


@pytest.fixture
def env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _code_strings(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return [(node.lineno, node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docstrings]


@pytest.mark.parametrize("name", VIEWER_FILES)
def test_no_hardcoded_location_in_viewer_code(name):
    found = [(line, text) for line, text in _code_strings((POC / name).read_text(encoding="utf-8"))
             if HARDCODED_LOCATION.search(text)]
    assert not found, f"{name} carries a hardcoded location: {found}"


def test_the_deploy_script_has_no_default_destination_and_no_drive_default():
    text = (POC / "deploy_viewer.ps1").read_text(encoding="utf-8")
    assert not re.search(r"(?<![A-Za-z0-9])[A-Za-z]:\\", text)       # no drive-letter default
    assert '[string]$Dest   = ""' in text
    assert "there is no default destination" in text
    assert "[char[]]" not in text                     # no drive-letter scan for the shared drive


def test_the_viewer_scans_no_sibling_dataset_folder():
    tree = ast.parse((POC / "viewer_app.py").read_text(encoding="utf-8"))
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "_dataset_dirs_near_bundle" not in names
    assert not any(isinstance(n, ast.Name) and n.id == "DATASET_DIRNAME" for n in ast.walk(tree))


# ---------------------------------------------------------------- the data root
def test_the_data_root_is_explicit_or_the_variable(env, tmp_path):
    assert viewer_paths.data_root(str(tmp_path)) == str(tmp_path)
    env.setenv("SOMA_DATA_ROOT", str(tmp_path / "root"))
    assert viewer_paths.data_root() == str(tmp_path / "root")
    assert viewer_paths.data_root(str(tmp_path)) == str(tmp_path)      # explicit wins
    env.delenv("SOMA_DATA_ROOT")
    with pytest.raises(SystemExit, match="SOMA_DATA_ROOT is not set.*--data-root") as refused:
        viewer_paths.data_root(hint="or pass --data-root <dir>")
    assert not HARDCODED_LOCATION.search(str(refused.value))


def test_sources_and_models_follow_the_variables(env, tmp_path):
    root = str(tmp_path / "root")
    assert viewer_paths.source_folder(root, "prism") == os.path.join(root, "extracted", "prism")
    env.setenv("SOMA_SOURCE_ROOT", str(tmp_path / "src"))
    assert viewer_paths.source_folder(root, "prism") == os.path.join(str(tmp_path / "src"), "prism")
    assert viewer_paths.body_model_dir() is None
    assert viewer_paths.body_model_dir(root) == os.path.join(root, "body_models", "smpl")
    env.setenv("SOMA_BODY_MODEL_DIR", str(tmp_path / "models"))
    assert viewer_paths.body_model_dir(root) == str(tmp_path / "models")
    assert viewer_paths.bundles_root(root) == os.path.join(root, "runs",
                                                          "experimental_generation_poc_demo")


# ---------------------------------------------------------------- where runs are listed from
def _mkdirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def test_run_roots_are_explicit_then_the_data_roots_bundles_then_the_bundles_own(env, tmp_path):
    share = tmp_path / "share"
    viewer = share / "viewer"
    here = viewer / "scripts"
    sibling = share / "datasets"                      # a sibling folder: must not be scanned
    data = tmp_path / "data"
    bundles = data / "runs" / "experimental_generation_poc_demo"
    explicit, named = tmp_path / "mine", tmp_path / "named"
    _mkdirs(here / "data" / "runs", viewer / "data" / "runs", sibling / "hknu_unified8",
            bundles / "hknu_unified8", explicit, named, share / "other_dataset")
    env.setenv("SOMA_VIEWER_DATASET", str(named))
    env.setenv("SOMA_DATA_ROOT", str(data))
    roots = viewer_paths.run_roots(str(viewer), str(here), [str(explicit), str(explicit)])
    assert roots == [str(explicit), str(named), str(bundles), str(here / "data" / "runs"),
                     str(viewer / "data" / "runs")]
    assert not any("datasets" in root or "other_dataset" in root for root in roots)


def test_without_a_data_root_only_given_folders_are_listed_and_the_hint_says_so(env, tmp_path):
    viewer = tmp_path / "viewer"
    _mkdirs(viewer / "scripts", tmp_path / "datasets" / "x")
    assert viewer_paths.run_roots(str(viewer), str(viewer / "scripts")) == []
    assert "SOMA_DATA_ROOT is not set" in viewer_paths.no_runs_hint()
    assert "--folder" in viewer_paths.no_runs_hint()
    env.setenv("SOMA_DATA_ROOT", str(tmp_path))
    assert "SOMA_DATA_ROOT is not set" not in viewer_paths.no_runs_hint()


def test_model_candidates_never_name_a_drive(env, tmp_path):
    bundle, here = str(tmp_path / "b"), str(tmp_path / "b" / "scripts")
    assert viewer_paths.model_candidates("SMPL_MALE_clean.npz", "male", bundle, here) == [
        os.path.join(bundle, "data", "SMPL_MALE_clean.npz"),
        os.path.join(here, "data", "SMPL_MALE_clean.npz")]
    env.setenv("SOMA_DATA_ROOT", str(tmp_path / "root"))
    env.setenv("SOMA_VIEWER_MODEL", str(tmp_path / "m.npz"))
    found = viewer_paths.model_candidates("SMPL_MALE_clean.npz", "male", bundle, here)
    assert found[0] == str(tmp_path / "m.npz")
    assert found[-1] == os.path.join(str(tmp_path / "root"), "body_models", "smpl",
                                     "SMPL_MALE_clean.npz")
    # the male-only legacy knob does not override another gender
    assert viewer_paths.model_candidates("SMPL_FEMALE_clean.npz", "female", bundle, here)[0] == \
        os.path.join(bundle, "data", "SMPL_FEMALE_clean.npz")


# ---------------------------------------------------------------- where a writing tool may write
def test_a_writing_tool_never_writes_into_a_bundle_the_container_the_evidence_or_the_sources(
        env, tmp_path):
    bundle = tmp_path / "anywhere" / "prism_faithful_full"
    _mkdirs(bundle / "prism-subj001-take002")
    (bundle / "INDEX.json").write_text("{}", encoding="utf-8")
    data, sources = tmp_path / "data", tmp_path / "sources"
    env.setenv("SOMA_DATA_ROOT", str(data))
    env.setenv("SOMA_SOURCE_ROOT", str(sources))
    container = tmp_path / "other" / "runs" / "experimental_generation_poc_demo"
    refused = {
        bundle: "INDEX.json",
        bundle / "prism-subj001-take002": "INDEX.json",
        bundle / "prism-subj001-take002" / "figures" / "new": "INDEX.json",   # not made yet
        container / "scratch" / "take": "lineage container",                  # no INDEX.json
        tmp_path / "other" / "RUNS" / "Experimental_Generation_POC_Demo": "lineage container",
        tmp_path / "x" / "_superseded" / "prism_faithful_full_2026-09-01" / "take": "evidence",
        tmp_path / "x" / "_Runs": "evidence",
        tmp_path / "x" / "_manifest_backfill" / "y": "evidence",
        data / "extracted" / "prism": "SOMA_DATA_ROOT/extracted",
        data / "raw_archives": "SOMA_DATA_ROOT/raw_archives",
        sources / "prism" / "subj001": "SOMA_SOURCE_ROOT",
    }
    for folder, reason in refused.items():
        with pytest.raises(SystemExit) as caught:
            viewer_paths.writable_folder(str(folder), "--run")
        message = str(caught.value)
        assert message.startswith(f"refusing to write into {folder} (--run)"), message
        assert reason in message and viewer_paths.COPY_THE_TAKE in message, message
    with pytest.raises(SystemExit, match="name a file in a scratch folder"):
        viewer_paths.writable_folder(str(bundle), "--out", instead="name a file in a scratch folder")
    allowed = (tmp_path / "scratch" / "prism-subj001-take002", data / "tmp" / "take",
               data / "extracted_copy", tmp_path / "x" / "runs" / "take",
               tmp_path / "experimental_generation_poc_demo" / "take", tmp_path / "x" / "my_runs")
    for folder in allowed:
        assert viewer_paths.writable_folder(str(folder), "--run") == str(folder)
    env.delenv("SOMA_DATA_ROOT")
    env.delenv("SOMA_SOURCE_ROOT")
    assert viewer_paths.writable_folder(str(data / "extracted" / "x"), "--run")   # no root known


def test_a_writing_tools_missing_argument_says_where_to_write_instead(env, tmp_path):
    env.setenv("SOMA_DATA_ROOT", str(tmp_path))
    with pytest.raises(SystemExit) as reading:
        viewer_paths.require(None, "--npz", "the npz")
    assert "never into a bundle" not in str(reading.value)
    with pytest.raises(SystemExit) as writing:
        viewer_paths.require(None, "--run", "the take folder", writes=viewer_paths.COPY_THE_TAKE)
    assert str(writing.value).startswith("name the take folder with --run; there is no default. "
                                         f"The bundles are under {tmp_path}")
    assert str(writing.value).endswith(". This tool writes there and never into a bundle: "
                                       + viewer_paths.COPY_THE_TAKE)


def test_blender_script_arguments():
    assert viewer_paths.script_argv(["blender", "-b", "--python", "x.py", "--", "--run", "R"]) == \
        ["--run", "R"]
    assert viewer_paths.script_argv(["x.py", "--run", "R"]) == ["--run", "R"]
    assert viewer_paths.option(["--run", "R"], "--run") == "R"
    assert viewer_paths.option(["--run"], "--run", "d") == "d"


# ---------------------------------------------------------------- the presentation package
def test_the_presentation_code_is_looked_for_in_named_places_only(env, tmp_path):
    """$SOMA_ANIM_DIR and the scripts' own folder only: nothing under the data root and no folder
    above the checkout. A checkout inside a synced folder could find a presentation folder above
    it, an online-only placeholder whose open() never returned."""
    assert list(viewer_paths._candidates()) == [viewer_paths.HERE]
    env.setenv("SOMA_DATA_ROOT", str(tmp_path / "root"))
    env.setenv("SOMA_ANIM_DIR", str(tmp_path / "anim"))
    assert list(viewer_paths._candidates()) == [str(tmp_path / "anim"), viewer_paths.HERE]
    source = (POC / "viewer_paths.py").read_text(encoding="utf-8")
    assert "_MAX_PARENTS" not in source                  # no walk up the parent folders


class _Stat:
    def __init__(self, attributes):
        if attributes is not None:
            self.st_file_attributes = attributes


@pytest.mark.parametrize("attributes, found", [
    (0x1620, False),       # a Dropbox placeholder: OFFLINE | REPARSE_POINT | SPARSE | ARCHIVE
    (0x1000, False),       # OFFLINE
    (0x40000, False),      # RECALL_ON_OPEN
    (0x400000, False),     # RECALL_ON_DATA_ACCESS
    (0x20, True),          # ARCHIVE: a file on this disk
    (0x80020, True),       # PINNED (always keep on this device) and ARCHIVE
    (None, True),          # no such attributes (not Windows)
])
def test_a_file_not_stored_locally_is_skipped_unopened(env, monkeypatch, tmp_path, attributes,
                                                      found):
    anim = tmp_path / "anim"
    anim.mkdir()
    probe = anim / "render_amass.py"
    probe.write_text("# stand-in\n", encoding="utf-8")
    env.setenv("SOMA_ANIM_DIR", str(anim))
    real_stat, real_open, opened = os.stat, open, []

    def stat(path, *args, **kwargs):
        if os.path.normcase(os.fspath(path)) == os.path.normcase(str(probe)):
            real_stat(path, *args, **kwargs)                 # still missing when it is missing
            return _Stat(attributes)
        return real_stat(path, *args, **kwargs)

    def spy_open(path, *args, **kwargs):
        opened.append(os.fspath(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(viewer_paths.os, "stat", stat)
    monkeypatch.setattr(viewer_paths, "open", spy_open, raising=False)
    assert viewer_paths.stored_locally(str(probe)) is found
    assert viewer_paths.anim_dir() == (str(anim) if found else "")
    assert (str(probe) in opened) is found                   # a placeholder is never opened
    assert viewer_paths.stored_locally(str(anim / "missing.py")) is False


# ---------------------------------------------------------------- the launcher
def test_the_launcher_needs_a_run_and_a_data_root(env, monkeypatch, tmp_path, capsys):
    import viewer_launch

    monkeypatch.setattr(sys, "argv", ["viewer_launch.py"])
    with pytest.raises(SystemExit, match="--run"):
        viewer_launch.main()
    monkeypatch.setattr(sys, "argv", ["viewer_launch.py", "--run", str(tmp_path)])
    with pytest.raises(SystemExit, match="SOMA_DATA_ROOT is not set"):
        viewer_launch.main()
    env.setenv("BLENDER_EXE", str(tmp_path / "no-blender"))
    assert viewer_launch.default_blender() == str(tmp_path / "no-blender")
    env.setenv("SOMA_DATA_ROOT", str(tmp_path))
    with pytest.raises(SystemExit, match="Blender not found"):
        viewer_launch.main()
    assert viewer_launch.logical_run_id(str(tmp_path / "runs" / "x"), str(tmp_path)) == "runs/x"
