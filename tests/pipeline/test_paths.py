"""`pipeline.paths`: every location comes from the environment, when asked, or not at all.

These tests pin what each variable resolves to, what is refused and with which error, that
nothing is read at import, and that no absolute path appears in the generation code.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from soma_synth.pipeline import paths

REPO = Path(__file__).resolve().parents[2]
ENV = (paths.DATA_ROOT_ENV, paths.SOURCE_ROOT_ENV, paths.BODY_MODEL_DIR_ENV,
       paths.SHARED_DRIVE_NAMES_ENV)


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def data_root(clean_env, tmp_path):
    root = tmp_path / "data"
    (root / "extracted").mkdir(parents=True)
    clean_env.setenv(paths.DATA_ROOT_ENV, str(root))
    return root


# ---------------------------------------------------------------- data_root


class TestDataRoot:
    def test_unset_is_refused_by_name(self, clean_env):
        with pytest.raises(paths.PathConfigError, match="SOMA_DATA_ROOT is not set"):
            paths.data_root()

    def test_blank_counts_as_unset(self, clean_env):
        clean_env.setenv(paths.DATA_ROOT_ENV, "   ")
        with pytest.raises(paths.PathConfigError, match="not set"):
            paths.data_root()

    def test_a_relative_value_is_refused(self, clean_env):
        clean_env.setenv(paths.DATA_ROOT_ENV, "relative/data")
        with pytest.raises(paths.PathConfigError, match="not an absolute path"):
            paths.data_root()

    def test_a_missing_folder_is_refused(self, clean_env, tmp_path):
        clean_env.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "absent"))
        with pytest.raises(paths.PathConfigError, match="does not exist"):
            paths.data_root()

    def test_a_file_is_not_a_data_root(self, clean_env, tmp_path):
        target = tmp_path / "file"
        target.write_text("x", encoding="utf-8")
        clean_env.setenv(paths.DATA_ROOT_ENV, str(target))
        with pytest.raises(paths.PathConfigError, match="not a directory"):
            paths.data_root()

    def test_the_value_is_returned_as_given(self, data_root):
        assert paths.data_root() == data_root

    def test_the_error_is_a_runtime_error(self):
        assert issubclass(paths.PathConfigError, RuntimeError)
        assert issubclass(paths.OutputLocationRefused, paths.PathConfigError)
        # relative_to used to raise ValueError where data_relative now raises; both still catch it
        assert issubclass(paths.PathOutsideRootError, ValueError)
        assert issubclass(paths.PathOutsideRootError, paths.PathConfigError)


# ---------------------------------------------------------------- source_root / source_dir


class TestSourceRoot:
    def test_defaults_to_extracted_under_the_data_root(self, data_root):
        assert paths.source_root() == data_root / "extracted"

    def test_the_variable_overrides_the_default(self, data_root, clean_env, tmp_path):
        elsewhere = tmp_path / "sources"
        elsewhere.mkdir()
        clean_env.setenv(paths.SOURCE_ROOT_ENV, str(elsewhere))
        assert paths.source_root() == elsewhere

    def test_the_variable_does_not_need_a_data_root(self, clean_env, tmp_path):
        clean_env.setenv(paths.SOURCE_ROOT_ENV, str(tmp_path))
        assert paths.source_root() == tmp_path

    def test_a_missing_source_root_is_refused(self, data_root, clean_env, tmp_path):
        clean_env.setenv(paths.SOURCE_ROOT_ENV, str(tmp_path / "absent"))
        with pytest.raises(paths.PathConfigError, match="does not exist"):
            paths.source_root()

    def test_a_missing_default_extracted_is_refused(self, clean_env, tmp_path):
        clean_env.setenv(paths.DATA_ROOT_ENV, str(tmp_path))
        with pytest.raises(paths.PathConfigError, match="extracted"):
            paths.source_root()

    def test_a_relative_source_root_is_refused(self, data_root, clean_env):
        clean_env.setenv(paths.SOURCE_ROOT_ENV, "sources")
        with pytest.raises(paths.PathConfigError, match="not an absolute path"):
            paths.source_root()

    def test_the_folders_are_the_extracted_names(self):
        assert paths.SOURCE_FOLDERS == {
            "amass": "amass", "prism": "prism", "gaitex": "gaitex",
            "hknu": "hknu_fullbody", "addbiomechanics": "addbiomechanics",
        }

    @pytest.mark.parametrize("source", sorted(paths.SOURCE_FOLDERS))
    def test_source_dir_is_the_folder_under_the_source_root(self, data_root, source):
        assert paths.source_dir(source) == data_root / "extracted" / paths.SOURCE_FOLDERS[source]

    def test_an_unknown_source_is_refused_before_the_environment_is_read(self, clean_env):
        with pytest.raises(paths.PathConfigError, match="unknown source 'weargait'"):
            paths.source_dir("weargait")

    def test_an_existing_source_folder_is_returned(self, data_root, tmp_path):
        (data_root / "extracted" / "hknu_fullbody").mkdir()
        assert paths.existing_source_dir("hknu") == data_root / "extracted" / "hknu_fullbody"
        assert paths.existing_source_dir("amass", tmp_path) == tmp_path

    def test_a_missing_default_source_folder_is_named_with_its_variable(self, data_root, clean_env,
                                                                       tmp_path):
        """A batch generator's glob over a missing folder finds nothing and says nothing."""
        with pytest.raises(paths.PathConfigError, match="SOMA_DATA_ROOT/extracted/amass") as refused:
            paths.existing_source_dir("amass")
        assert str(data_root / "extracted" / "amass") in str(refused.value)
        moved = tmp_path / "moved"
        moved.mkdir()
        clean_env.setenv(paths.SOURCE_ROOT_ENV, str(moved))
        with pytest.raises(paths.PathConfigError, match="SOMA_SOURCE_ROOT/prism"):
            paths.existing_source_dir("prism")

    def test_a_missing_or_file_flag_folder_is_named(self, clean_env, tmp_path):
        with pytest.raises(paths.PathConfigError, match="as given") as refused:
            paths.existing_source_dir("amass", tmp_path / "absent")
        assert str(tmp_path / "absent") in str(refused.value)
        (tmp_path / "file").write_text("x", encoding="utf-8")
        with pytest.raises(paths.PathConfigError, match="not a directory"):
            paths.existing_source_dir("prism", tmp_path / "file")


# ---------------------------------------------------------------- body models


class TestBodyModelDir:
    def test_defaults_under_the_data_root(self, data_root):
        assert paths.body_model_dir() == data_root / "body_models" / "smpl"

    def test_another_folder_is_honoured_now_that_the_runner_records_it(self, data_root, clean_env,
                                                                       tmp_path):
        """The runner resolves the folder through this same function and hashes the one it
        resolved (tests/pipeline/test_runner_v2.py), so the variable is honoured."""
        clean_env.setenv(paths.BODY_MODEL_DIR_ENV, str(tmp_path / "models"))
        assert paths.body_model_dir() == tmp_path / "models"
        assert paths.body_model_override() == tmp_path / "models"

    def test_the_variable_needs_no_data_root(self, clean_env, tmp_path):
        clean_env.setenv(paths.BODY_MODEL_DIR_ENV, str(tmp_path / "models"))
        assert paths.body_model_dir() == tmp_path / "models"

    def test_an_explicit_root_replaces_the_data_root_but_not_the_variable(self, data_root,
                                                                         clean_env, tmp_path):
        assert paths.body_model_dir(root=tmp_path) == tmp_path / "body_models" / "smpl"
        clean_env.setenv(paths.BODY_MODEL_DIR_ENV, str(tmp_path / "models"))
        assert paths.body_model_dir(root=tmp_path / "elsewhere") == tmp_path / "models"

    def test_naming_the_default_folder_is_accepted(self, data_root, clean_env):
        clean_env.setenv(paths.BODY_MODEL_DIR_ENV, str(data_root / "body_models" / "smpl"))
        assert paths.body_model_dir() == data_root / "body_models" / "smpl"

    def test_a_relative_value_is_refused(self, data_root, clean_env):
        clean_env.setenv(paths.BODY_MODEL_DIR_ENV, "models")
        with pytest.raises(paths.PathConfigError, match="not an absolute path"):
            paths.body_model_dir()

    def test_unset_everything_is_refused(self, clean_env):
        with pytest.raises(paths.PathConfigError, match="SOMA_DATA_ROOT"):
            paths.body_model_dir()

    def test_shape_fit_and_anthro_select_inside_it(self, data_root, clean_env, tmp_path):
        from soma_synth.addbio_retarget import shape_fit

        models = data_root / "body_models" / "smpl"
        assert shape_fit.clean_model_path_for_gender("neutral") == models / "SMPL_NEUTRAL_clean.npz"
        # an explicit data root still means <root>/body_models/smpl, as it always did
        assert (shape_fit.clean_model_path_for_gender("male", root=tmp_path)
                == tmp_path / "body_models" / "smpl" / "SMPL_MALE_clean.npz")
        clean_env.setenv(paths.BODY_MODEL_DIR_ENV, str(tmp_path / "elsewhere"))
        assert (shape_fit.clean_model_path_for_gender("neutral")
                == tmp_path / "elsewhere" / "SMPL_NEUTRAL_clean.npz")
        # an explicit root is a data root, as it always was, whatever the variable says
        assert (shape_fit.clean_model_path_for_gender("male", root=tmp_path)
                == tmp_path / "body_models" / "smpl" / "SMPL_MALE_clean.npz")


# ---------------------------------------------------------------- lineage_dir


class TestLineageDir:
    def test_under_the_poc_demo_folder_of_the_data_root(self, data_root):
        assert paths.POC_DEMO == "runs/experimental_generation_poc_demo"
        assert (paths.lineage_dir("hknu_unified8")
                == data_root / "runs" / "experimental_generation_poc_demo" / "hknu_unified8")

    def test_an_explicit_root_needs_no_environment(self, clean_env, tmp_path):
        assert (paths.lineage_dir("x", root=tmp_path)
                == tmp_path / "runs" / "experimental_generation_poc_demo" / "x")

    @pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b"])
    def test_a_name_must_be_one_folder(self, clean_env, tmp_path, name):
        with pytest.raises(paths.PathConfigError, match="lineage directory name"):
            paths.lineage_dir(name, root=tmp_path)


class TestPartialRunIntoLineage:
    def test_the_default_lineage_in_any_spelling(self, data_root):
        lineage = data_root / "runs" / "experimental_generation_poc_demo" / "amass_faithful_full"
        assert paths.is_default_lineage(lineage, "amass_faithful_full")
        assert paths.is_default_lineage(str(lineage), "amass_faithful_full")
        assert paths.is_default_lineage(lineage / ".." / "amass_faithful_full", "amass_faithful_full")
        if os.name == "nt":
            assert paths.is_default_lineage(str(lineage).upper(), "amass_faithful_full")

    def test_another_folder_or_lineage_is_not(self, data_root, tmp_path):
        poc = data_root / "runs" / "experimental_generation_poc_demo"
        assert not paths.is_default_lineage(poc / "prism_faithful_full", "amass_faithful_full")
        assert not paths.is_default_lineage(poc, "amass_faithful_full")
        assert not paths.is_default_lineage(data_root / "tmp" / "amass_faithful_full",
                                            "amass_faithful_full")
        assert not paths.is_default_lineage(
            paths.lineage_dir("amass_faithful_full", root=tmp_path / "scratch_root"),
            "amass_faithful_full")

    def test_no_data_root_means_no_default(self, clean_env, tmp_path):
        assert not paths.is_default_lineage(tmp_path / "x", "amass_faithful_full")
        paths.refuse_partial_run_into_lineage(tmp_path / "x", "amass_faithful_full",
                                              ["--limit 1"], "--out-root")

    def test_a_set_but_missing_data_root_is_an_error(self, clean_env, tmp_path):
        clean_env.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "absent"))
        with pytest.raises(paths.PathConfigError, match="does not exist"):
            paths.is_default_lineage(tmp_path / "x", "amass_faithful_full")

    def test_a_partial_run_into_the_lineage_is_refused(self, data_root):
        lineage = paths.lineage_dir("prism_faithful_full")
        with pytest.raises(paths.OutputLocationRefused) as refused:
            paths.refuse_partial_run_into_lineage(lineage, "prism_faithful_full",
                                                  ["--only prism_subj005", "--limit 3"], "--out",
                                                  note="(or --data-root)")
        text = str(refused.value)
        assert "--only prism_subj005 --limit 3 selects part of the source" in text
        assert "SOMA_DATA_ROOT/runs/experimental_generation_poc_demo/prism_faithful_full" in text
        assert text.endswith(f"--out {data_root / 'tmp' / 'prism_faithful_full'} (or --data-root)")
        assert isinstance(refused.value, paths.PathConfigError)     # the mains' except clause

    def test_a_full_run_or_a_scratch_output_passes(self, data_root):
        lineage = paths.lineage_dir("prism_faithful_full")
        paths.refuse_partial_run_into_lineage(lineage, "prism_faithful_full", [], "--out")
        paths.refuse_partial_run_into_lineage(lineage, "prism_faithful_full", ["", ""], "--out")
        paths.refuse_partial_run_into_lineage(data_root / "tmp" / "prism_sample",
                                              "prism_faithful_full", ["--limit 3"], "--out")


# ---------------------------------------------------------------- relative ids


class TestLogicalSourceId:
    def test_the_id_manifests_carry(self, data_root):
        take = data_root / "extracted" / "gaitex" / "austra" / "gwo" / "markers.csv"
        assert paths.logical_source_id("gaitex", take) == "extracted/gaitex/austra/gwo/markers.csv"

    def test_hknu_uses_its_folder_name(self, data_root):
        take = data_root / "extracted" / "hknu_fullbody" / "MATLAB" / "DatasetInfo.xlsx"
        assert paths.logical_source_id("hknu", take) == "extracted/hknu_fullbody/MATLAB/DatasetInfo.xlsx"

    def test_the_id_does_not_depend_on_where_the_source_is(self, clean_env, tmp_path):
        relocated = tmp_path / "somewhere" / "gaitex_copy"
        take = relocated / "austra" / "gwo" / "markers.csv"
        assert (paths.logical_source_id("gaitex", take, root=relocated)
                == "extracted/gaitex/austra/gwo/markers.csv")

    def test_the_source_root_variable_moves_the_default(self, data_root, clean_env, tmp_path):
        moved = tmp_path / "moved"
        (moved / "prism").mkdir(parents=True)
        clean_env.setenv(paths.SOURCE_ROOT_ENV, str(moved))
        take = moved / "prism" / "subj005" / "take001.pkl"
        assert paths.logical_source_id("prism", take) == "extracted/prism/subj005/take001.pkl"

    def test_a_path_outside_the_source_is_refused(self, data_root):
        with pytest.raises(paths.PathOutsideRootError):
            paths.logical_source_id("gaitex", data_root / "extracted" / "prism" / "x.pkl")

    def test_the_folder_itself_has_no_id(self, data_root):
        with pytest.raises(paths.PathOutsideRootError, match="itself"):
            paths.logical_source_id("gaitex", data_root / "extracted" / "gaitex")

    def test_an_unknown_source_is_refused(self, clean_env, tmp_path):
        with pytest.raises(paths.PathConfigError, match="unknown source"):
            paths.logical_source_id("nope", tmp_path / "x", root=tmp_path)


class TestDataRelative:
    def test_forward_slashes_under_the_data_root(self, data_root):
        take = data_root / "runs" / "experimental_generation_poc_demo" / "gaitex_smpl24" / "a" / "b.npz"
        assert paths.data_relative(take) == "runs/experimental_generation_poc_demo/gaitex_smpl24/a/b.npz"

    def test_the_same_string_relative_to_returned_before(self, data_root):
        take = data_root / "tmp" / "harness" / "run" / "addbio_smpl24_raw" / "S" / "s1" / "t.npz"
        assert paths.data_relative(take) == str(take.relative_to(data_root)).replace("\\", "/")

    def test_an_explicit_root(self, clean_env, tmp_path):
        assert paths.data_relative(tmp_path / "a" / "b", root=tmp_path) == "a/b"

    def test_a_relative_spelling_is_resolved(self, data_root, clean_env):
        (data_root / "runs").mkdir()
        clean_env.chdir(data_root)
        assert paths.data_relative(Path("runs") / "x.npz") == "runs/x.npz"

    def test_outside_is_refused(self, data_root, tmp_path):
        with pytest.raises(paths.PathOutsideRootError, match="not inside the data root"):
            paths.data_relative(tmp_path / "elsewhere" / "x.npz")

    def test_unset_is_refused(self, clean_env, tmp_path):
        with pytest.raises(paths.PathConfigError):
            paths.data_relative(tmp_path / "x")


# ---------------------------------------------------------------- output guard


class TestCheckOutputDir:
    def test_an_ordinary_output_is_returned_unchanged(self, data_root):
        target = data_root / "tmp" / "harness" / "run1" / "hknu_unified8"
        assert paths.check_output_dir(target) == target
        assert paths.check_output_dir(str(target)) == target

    def test_the_source_root_is_refused(self, data_root):
        with pytest.raises(paths.OutputLocationRefused, match="source root"):
            paths.check_output_dir(data_root / "extracted")

    def test_inside_the_source_root_is_refused(self, data_root):
        with pytest.raises(paths.OutputLocationRefused, match="source root"):
            paths.check_output_dir(data_root / "extracted" / "gaitex" / "out")

    def test_a_relocated_source_root_is_guarded_too(self, data_root, clean_env, tmp_path):
        moved = tmp_path / "moved_sources"
        moved.mkdir()
        clean_env.setenv(paths.SOURCE_ROOT_ENV, str(moved))
        with pytest.raises(paths.OutputLocationRefused, match="source root"):
            paths.check_output_dir(moved / "prism" / "bundle")

    def test_the_body_model_directory_is_refused(self, data_root):
        with pytest.raises(paths.OutputLocationRefused, match="body model"):
            paths.check_output_dir(data_root / "body_models" / "smpl" / "out")

    @pytest.mark.parametrize("folder", ["Dropbox", "dropbox", "Dropbox (Team)", "OneDrive",
                                        "OneDrive - Example", "Google Drive", "GoogleDrive",
                                        "My Drive", "Shared drives", "Other computers",
                                        "내 드라이브", "공유 드라이브", "다른 컴퓨터",  # Korean Drive folder names under test
                                        "SynologyDrive", "Synology Drive"])
    def test_a_cloud_synchronised_folder_is_warned_about_not_refused(self, data_root, tmp_path,
                                                                     folder, capsys):
        """Google Drive for desktop names its folders in the Windows display language; a
        Korean-language Windows names them in Korean. Writing there is allowed with a warning
        (owner decision, 2026-09-30)."""
        target = tmp_path / folder / "project" / "bundle"
        assert paths.check_output_dir(target) == target
        err = capsys.readouterr().err
        assert err.count("warning:") == 1
        assert "cloud-synchronised" in err and "not recommended" in err

    def test_a_decomposed_korean_folder_name_is_warned_about_too(self, data_root, tmp_path,
                                                                capsys):
        decomposed = unicodedata.normalize("NFD", "내 드라이브")  # Korean "My Drive", decomposed below
        assert decomposed != "내 드라이브"  # Korean "My Drive": NFD must differ from the composed form
        target = tmp_path / decomposed / "bundle"
        assert paths.check_output_dir(target) == target
        assert "cloud-synchronised" in capsys.readouterr().err

    def test_a_folder_merely_named_like_one_is_not(self, data_root, capsys):
        target = data_root / "tmp" / "dropbox_notes_backup"
        assert paths.check_output_dir(target) == target
        assert "warning:" not in capsys.readouterr().err

    def test_a_named_shared_drive_is_warned_about(self, data_root, clean_env, tmp_path, capsys):
        target = tmp_path / "Team_Share" / "datasets" / "bundle"
        assert paths.check_output_dir(target) == target        # no shared drive is named
        assert "warning:" not in capsys.readouterr().err
        clean_env.setenv(paths.SHARED_DRIVE_NAMES_ENV, "Other, team_share")
        assert paths.check_output_dir(target) == target
        err = capsys.readouterr().err
        assert err.count("warning:") == 1 and "shared drive" in err

    def test_the_warning_is_printed_once_per_path(self, data_root, tmp_path, capsys):
        target = tmp_path / "Dropbox" / "bundle"
        for _ in range(3):
            assert paths.warn_if_synced(target) == target
        assert paths.check_output_dir(target) == target
        assert capsys.readouterr().err.count("warning:") == 1
        other = tmp_path / "Dropbox" / "other"
        paths.warn_if_synced(other)
        assert capsys.readouterr().err.count("warning:") == 1

    def test_the_other_refusals_still_apply_inside_a_synchronised_folder(self, clean_env,
                                                                          tmp_path):
        root = tmp_path / "Dropbox" / "data"
        (root / "extracted").mkdir(parents=True)
        clean_env.setenv(paths.DATA_ROOT_ENV, str(root))
        with pytest.raises(paths.OutputLocationRefused, match="read-only"):
            paths.check_output_dir(root / "extracted" / "prism" / "bundle")
        with pytest.raises(paths.OutputLocationRefused, match="itself"):
            paths.check_output_dir(root)

    def test_the_warning_uses_the_pure_rule(self, data_root, monkeypatch, tmp_path, capsys):
        """warn_if_synced judges the path as written and as resolved with synced_location, so
        the platform rules tested on pure paths below are the ones a real run meets."""
        seen = []
        monkeypatch.setattr(paths, "synced_location",
                            lambda path: seen.append(path) or "Somewhere")
        target = tmp_path / "plain" / "bundle"
        assert paths.warn_if_synced(target) == target
        assert "'Somewhere'" in capsys.readouterr().err
        assert seen and seen[0] == (tmp_path / "plain" / "bundle").absolute()


class TestSyncedLocationsOnEveryPlatform:
    """Teammates may run on macOS or Linux. :func:`paths.synced_location` follows the flavour
    of the path it is given, not the running system's, so the POSIX spellings of the sync
    clients' folders are tested here with pure paths on any PC."""

    @pytest.mark.parametrize("text", [
        "/Users/kim/Dropbox/soma/runs/x",
        "/Users/kim/Dropbox (Personal)/soma",
        "/home/kim/Dropbox/soma",
        "/home/kim/dropbox/soma",                       # a name marker, matched without case
        "/Users/kim/OneDrive - Example University/soma",
        "/home/kim/OneDrive/soma",
        "/Users/kim/Library/CloudStorage/GoogleDrive-kim@example.org/My Drive/soma",
        "/Users/kim/Library/CloudStorage/GoogleDrive-kim@example.org/내 드라이브/soma",  # Korean "My Drive"
        "/Users/kim/Library/CloudStorage/OneDrive-Personal/soma",
        "/Users/kim/Library/CloudStorage/Box-Box/soma",
        "/Users/kim/Library/CloudStorage/Dropbox/soma",
        "/Users/kim/Library/Mobile Documents/com~apple~CloudDocs/soma",
        "/Users/kim/Google Drive/soma",
        "/Volumes/GoogleDrive/My Drive/soma",
        "/Volumes/GoogleDrive-1234567890/Shared drives/team/soma",
        "/Users/kim/My Drive/soma",
        "/home/kim/공유 드라이브/soma",                 # Korean "Shared drives"
        "/run/user/1000/gvfs/google-drive:host=example.org,user=kim/soma",
        "/home/kim/SynologyDrive/soma",
    ])
    def test_posix_synchronised_folders_are_recognised(self, text):
        assert paths.synced_location(PurePosixPath(text)) is not None
        decomposed = PurePosixPath(unicodedata.normalize("NFD", text))   # as macOS may hand it out
        assert paths.synced_location(decomposed) is not None

    @pytest.mark.parametrize("text", [
        "/home/kim/soma/data/runs/experimental_generation_poc_demo/hknu_unified8",
        "/Users/kim/soma_data/extracted/hknu_fullbody",
        "/Users/kim/Library/Application Support/soma",
        "/Users/kim/Library/Caches/soma",
        "/Users/kim/CloudStorage/soma",                 # not under Library
        "/Users/kim/Library/CloudStorageNotes/soma",
        "/Users/kim/Documents/Mobile Documents/soma",   # not under Library
        "/Volumes/Data/SOMA",
        "/Volumes/Drive/soma",
        "/home/kim/dropbox_notes_backup",
        "/home/kim/my drives/soma",
        "/mnt/google/soma",
        "/srv/google-drive-exports/soma",               # not a gvfs mount (no ':')
        "/",
    ])
    def test_ordinary_posix_folders_are_not(self, text):
        assert paths.synced_location(PurePosixPath(text)) is None

    def test_windows_paths_keep_their_rules(self):
        assert paths.synced_location(PureWindowsPath(r"C:\Users\kim\Dropbox\soma")) == "Dropbox"
        assert paths.synced_location(PureWindowsPath(r"X:\내 드라이브\soma")) == "내 드라이브"  # Korean "My Drive"
        assert paths.synced_location(PureWindowsPath(r"C:\Users\kim\iCloudDrive\x")) == "iCloudDrive"
        assert paths.synced_location(PureWindowsPath(r"E:\data\runs")) is None
        # the macOS pair and the gvfs mount are POSIX spellings: a Windows folder so named is not
        assert paths.synced_location(PureWindowsPath(r"E:\Library\CloudStorage\x")) is None
        assert paths.synced_location(PureWindowsPath(r"E:\data\google-drive:x")) is None
        # the anchor (a drive, a share) is never a marker
        assert paths.synced_location(PureWindowsPath("G:/")) is None

    def test_a_drive_letter_is_a_plain_folder_name_on_posix(self):
        assert paths.synced_location(PurePosixPath("/home/kim/G:/x")) is None
        assert paths.synced_location(PurePosixPath("/home/kim/G:/My Drive/x")) == "My Drive"
        assert paths.synced_location(PurePosixPath("G:/x")) is None

    def test_the_marker_names_what_it_found(self):
        mac = PurePosixPath("/Users/kim/Library/CloudStorage/Box-Box/soma")
        assert paths.synced_location(mac) == "Library/CloudStorage"

    def test_the_shared_drive_on_every_platform(self, monkeypatch):
        monkeypatch.delenv(paths.SHARED_DRIVE_NAMES_ENV, raising=False)
        assert paths.shared_drive_names() == ()
        assert not paths.on_shared_drive(PurePosixPath("/home/kim/Team_Share/x"))
        monkeypatch.setenv(paths.SHARED_DRIVE_NAMES_ENV, " Team_Share ,,Lab Data,team_share")
        assert paths.shared_drive_names() == ("team_share", "lab data")
        assert paths.on_shared_drive(PurePosixPath(
            "/Users/kim/Library/CloudStorage/GoogleDrive-x/Shared drives/Team_Share/x"))
        assert paths.on_shared_drive(PurePosixPath("/home/kim/team_share/x"))
        assert paths.on_shared_drive(PureWindowsPath(r"X:\공유 드라이브\LAB DATA"))  # Korean "Shared drives"
        assert not paths.on_shared_drive(PurePosixPath("/home/kim/team/share"))
        assert not paths.on_shared_drive(PurePosixPath("/home/kim/team_share_old/x"))

    def test_logical_ids_are_forward_slash_on_either_flavour(self):
        """Manifests and run records carry ``extracted/<folder>/<path>`` whatever PC wrote them:
        the relative part is rendered with as_posix, never with os.sep."""
        for flavour, root, take in (
                (PurePosixPath, "/home/kim/src/gaitex", "/home/kim/src/gaitex/austra/gwo/m.csv"),
                (PureWindowsPath, r"E:\src\gaitex", r"E:\src\gaitex\austra\gwo\m.csv")):
            relative = flavour(take).relative_to(flavour(root)).as_posix()
            assert relative == "austra/gwo/m.csv"
        source = (REPO / "src" / "soma_synth" / "pipeline" / "paths.py").read_text(encoding="utf-8")
        assert "os.sep" not in source.split("def _relative_posix", 1)[1].split("\ndef ", 1)[0]

    # --- the folders the run itself reads (flags or defaults), whatever the environment says
    def test_inside_an_explicit_input_is_refused(self, data_root, tmp_path):
        amass = tmp_path / "copies" / "amass"
        amass.mkdir(parents=True)
        with pytest.raises(paths.OutputLocationRefused, match="a folder this run reads"):
            paths.check_output_dir(amass / "bundle", inputs=[amass])
        with pytest.raises(paths.OutputLocationRefused, match="a folder this run reads"):
            paths.check_output_dir(amass, inputs=[tmp_path / "other", amass])

    def test_beside_an_input_is_accepted(self, data_root, tmp_path):
        target = tmp_path / "scratch" / "runs" / "bundle"
        assert paths.check_output_dir(target, inputs=[tmp_path / "scratch" / "src", None]) == target

    def test_a_folder_holding_an_input_is_refused(self, data_root, tmp_path):
        """The parent of an input, or a folder further up: INDEX.json and the take directories
        would be written beside the input (or over it, were a take named like it)."""
        source = tmp_path / "scratch" / "src" / "prism"
        source.mkdir(parents=True)
        for target in (source.parent, tmp_path / "scratch"):
            with pytest.raises(paths.OutputLocationRefused, match="the folder holding") as refused:
                paths.check_output_dir(target, inputs=[tmp_path / "elsewhere", source])
            assert str(source) in str(refused.value)
        # an input that does not exist yet is guarded the same way
        with pytest.raises(paths.OutputLocationRefused, match="the folder holding"):
            paths.check_output_dir(tmp_path / "new", inputs=[tmp_path / "new" / "src"])
        # a sibling whose name merely starts like the input's parent is not its holder
        assert (paths.check_output_dir(tmp_path / "scratch" / "sr", inputs=[source])
                == tmp_path / "scratch" / "sr")

    def test_an_input_is_guarded_without_any_environment(self, clean_env, tmp_path):
        with pytest.raises(paths.OutputLocationRefused, match="a folder this run reads"):
            paths.check_output_dir(tmp_path / "src" / "out", inputs=[tmp_path / "src"])

    # --- the data root and the poc_demo folder are containers, never a bundle
    def test_the_data_root_itself_is_refused(self, data_root):
        with pytest.raises(paths.OutputLocationRefused, match="data root itself"):
            paths.check_output_dir(data_root)

    def test_the_poc_demo_folder_itself_is_refused(self, data_root):
        poc = data_root / "runs" / "experimental_generation_poc_demo"
        with pytest.raises(paths.OutputLocationRefused, match="experimental_generation_poc_demo"):
            paths.check_output_dir(poc)
        assert paths.check_output_dir(poc / "amass_faithful_full") == poc / "amass_faithful_full"

    def test_the_runs_folder_itself_is_refused(self, data_root):
        with pytest.raises(paths.OutputLocationRefused, match="runs folder"):
            paths.check_output_dir(data_root / "runs")
        assert paths.check_output_dir(data_root / "runs" / "x") == data_root / "runs" / "x"

    # --- what docs/guides/RETENTION_RULES.md 1.4 protects on the data plane
    def test_the_raw_archives_are_refused(self, data_root):
        with pytest.raises(paths.OutputLocationRefused, match="raw_archives"):
            paths.check_output_dir(data_root / "raw_archives" / "prism" / "bundle")

    @pytest.mark.parametrize("folder", paths.EVIDENCE_FOLDERS)
    def test_the_evidence_folders_are_refused(self, data_root, folder):
        """`--out .../_superseded/hknu_smpl24_paired_2026-09-07` would overwrite the evidence."""
        target = (data_root / "runs" / "experimental_generation_poc_demo" / folder
                  / "hknu_smpl24_paired_2026-09-07")
        with pytest.raises(paths.OutputLocationRefused, match=f"evidence folder .*{folder}"):
            paths.check_output_dir(target)

    def test_a_relocated_source_root_still_guards_the_data_roots_extracted(self, data_root,
                                                                         clean_env, tmp_path):
        moved = tmp_path / "moved_sources"
        moved.mkdir()
        clean_env.setenv(paths.SOURCE_ROOT_ENV, str(moved))
        with pytest.raises(paths.OutputLocationRefused, match="SOMA_DATA_ROOT/extracted"):
            paths.check_output_dir(data_root / "extracted" / "gaitex" / "out")

    def test_a_data_root_that_is_set_but_missing_is_an_error(self, clean_env, tmp_path):
        """Skipping it would switch off every data-root guard without a word."""
        clean_env.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "absent"))
        with pytest.raises(paths.PathConfigError, match="does not exist"):
            paths.check_output_dir(tmp_path / "bundle")

    def test_a_source_root_that_is_set_but_missing_is_an_error(self, data_root, clean_env,
                                                              tmp_path):
        clean_env.setenv(paths.SOURCE_ROOT_ENV, str(tmp_path / "absent"))
        with pytest.raises(paths.PathConfigError, match="does not exist"):
            paths.check_output_dir(data_root / "tmp" / "bundle")

    # --- a source root that holds more than the sources
    def test_only_the_source_folders_of_a_shared_source_root_are_guarded(self, clean_env,
                                                                         tmp_path):
        """A teammate keeps every dataset in one folder and the data root beside the sources in
        it: outputs under the data root are accepted, the source folders and the root itself not."""
        datasets = tmp_path / "datasets"
        for folder in ("amass", "prism", "other_project"):
            (datasets / folder).mkdir(parents=True)
        out = datasets / "soma_out"
        out.mkdir()
        clean_env.setenv(paths.SOURCE_ROOT_ENV, str(datasets))
        clean_env.setenv(paths.DATA_ROOT_ENV, str(out))
        bundle = out / "runs" / "experimental_generation_poc_demo" / "amass_faithful_full"
        assert paths.check_output_dir(bundle) == bundle
        assert paths.check_output_dir(datasets / "other_project" / "x") == datasets / "other_project" / "x"
        for folder in paths.SOURCE_FOLDERS.values():
            with pytest.raises(paths.OutputLocationRefused, match=f"source root's {folder} folder"):
                paths.check_output_dir(datasets / folder / "bundle")
        with pytest.raises(paths.OutputLocationRefused, match="source root itself"):
            paths.check_output_dir(datasets)

    # --- the environment is consulted only where it configures something
    def test_an_unconfigured_pc_needs_no_environment_when_locations_are_explicit(self, clean_env,
                                                                                  tmp_path):
        """The guard used to require SOMA_DATA_ROOT and so broke `--data-root` runs on a PC
        without it. A default location still needs the variable, and says so, where the
        generator resolves it (paths.lineage_dir / source_dir)."""
        assert paths.check_output_dir(tmp_path / "bundle") == tmp_path / "bundle"
        with pytest.raises(paths.PathConfigError, match="SOMA_DATA_ROOT"):
            paths.lineage_dir("amass_faithful_full")

    def test_a_data_root_without_extracted_still_guards_the_body_models(self, clean_env, tmp_path):
        clean_env.setenv(paths.DATA_ROOT_ENV, str(tmp_path))
        assert paths.check_output_dir(tmp_path / "runs" / "x") == tmp_path / "runs" / "x"
        with pytest.raises(paths.OutputLocationRefused, match="body model"):
            paths.check_output_dir(tmp_path / "body_models" / "smpl" / "x")

    def test_a_malformed_variable_is_still_an_error(self, clean_env, tmp_path):
        clean_env.setenv(paths.SOURCE_ROOT_ENV, "relative/sources")
        with pytest.raises(paths.PathConfigError, match="not an absolute path"):
            paths.check_output_dir(tmp_path / "bundle")

    def test_a_moved_body_model_folder_is_guarded_and_so_is_the_default(self, data_root,
                                                                         clean_env, tmp_path):
        """SOMA_BODY_MODEL_DIR is honoured: both the folder it names and the default
        folder under the data root are read-only; an ordinary output passes."""
        clean_env.setenv(paths.BODY_MODEL_DIR_ENV, str(tmp_path / "models"))
        assert paths.check_output_dir(data_root / "tmp" / "bundle") == data_root / "tmp" / "bundle"
        with pytest.raises(paths.OutputLocationRefused, match="SOMA_BODY_MODEL_DIR"):
            paths.check_output_dir(tmp_path / "models" / "x")
        with pytest.raises(paths.OutputLocationRefused, match="body_models/smpl"):
            paths.check_output_dir(data_root / "body_models" / "smpl" / "x")

    def test_a_relative_body_model_folder_is_still_an_error(self, data_root, clean_env):
        clean_env.setenv(paths.BODY_MODEL_DIR_ENV, "models")
        with pytest.raises(paths.PathConfigError, match="not an absolute path"):
            paths.check_output_dir(data_root / "tmp" / "bundle")


POC = "runs/experimental_generation_poc_demo"


class TestTheRunsOwnRoot:
    """``root=``: the runner guards its own data root, which it hands its generators as
    SOMA_DATA_ROOT, before any generator runs (and SOMA_DATA_ROOT's, when that is another)."""

    @pytest.mark.parametrize("folder, label", [
        ("extracted/prism/x", "the source root"),
        ("raw_archives/prism/x", "raw_archives"),
        ("body_models/smpl/x", "body model"),
        (f"{POC}/_superseded/prism_faithful_full_v1", "_superseded"),
        (f"{POC}/_manifest_backfill/x", "_manifest_backfill"),
        (f"{POC}/_runs/x", "_runs"),
    ])
    def test_a_run_root_is_guarded_without_any_environment(self, clean_env, tmp_path, folder,
                                                           label):
        root = tmp_path / "run_root"                 # need not exist
        with pytest.raises(paths.OutputLocationRefused, match=label):
            paths.check_output_dir(root / folder, root=root)
        # without root nothing is configured, so the generators' guard is unchanged
        assert paths.check_output_dir(root / folder) == root / folder

    def test_a_run_root_and_its_containers(self, clean_env, tmp_path):
        root = tmp_path / "run_root"
        target = root / POC / "prism_faithful_full"
        assert paths.check_output_dir(target, root=root) == target
        for container in (root, root / "runs", root / POC):
            with pytest.raises(paths.OutputLocationRefused, match="itself"):
                paths.check_output_dir(container, root=root)

    def test_soma_data_root_is_guarded_beside_another_run_root(self, data_root, tmp_path):
        other = tmp_path / "other"
        with pytest.raises(paths.OutputLocationRefused, match="SOMA_DATA_ROOT/extracted"):
            paths.check_output_dir(data_root / "extracted" / "x", root=other)
        with pytest.raises(paths.OutputLocationRefused, match="the source root"):
            paths.check_output_dir(other / "extracted" / "x", root=other)
        # the same folder named twice is guarded once, by the run's label
        with pytest.raises(paths.OutputLocationRefused, match="run's data root/raw_archives"):
            paths.check_output_dir(data_root / "raw_archives" / "x", root=data_root)

    def test_a_missing_folder_is_guarded_rather_than_an_error(self, clean_env, tmp_path):
        """The runner records the locations its generators will see without requiring them to
        exist; guarding a folder that does not exist switches nothing off."""
        clean_env.setenv(paths.SOURCE_ROOT_ENV, str(tmp_path / "absent_sources"))
        clean_env.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "absent_root"))
        root = tmp_path / "root"
        assert paths.check_output_dir(root / "tmp" / "b", root=root) == root / "tmp" / "b"
        with pytest.raises(paths.OutputLocationRefused, match="source root's prism folder"):
            paths.check_output_dir(tmp_path / "absent_sources" / "prism" / "x", root=root)
        with pytest.raises(paths.OutputLocationRefused, match="SOMA_DATA_ROOT/extracted"):
            paths.check_output_dir(tmp_path / "absent_root" / "extracted" / "x", root=root)
        # a malformed value still is an error
        clean_env.setenv(paths.SOURCE_ROOT_ENV, "relative/sources")
        with pytest.raises(paths.PathConfigError, match="not an absolute path"):
            paths.check_output_dir(root / "tmp" / "b", root=root)


class TestCheckRecordDir:
    def test_run_records_go_to_runs_and_the_catalog_under_the_root(self, clean_env, tmp_path):
        root = tmp_path / "root"
        for target in (root / POC / "_runs", root / POC / "_runs" / "run_x",
                       root / "experimental" / "catalog", root / "tmp" / "sample" / "_runs",
                       tmp_path / "elsewhere" / "_runs"):
            assert paths.check_record_dir(target, root=root) == target

    def test_a_container_itself_is_refused(self, clean_env, tmp_path):
        """A run record folder or catalog at the data root, ``runs``, the lineage container or
        SOMA_SOURCE_ROOT itself fills that folder's top level with run_<identity> folders or
        catalog files."""
        root = tmp_path / "root"
        sources = tmp_path / "sources"
        clean_env.setenv(paths.SOURCE_ROOT_ENV, str(sources))
        for container in (root, root / "runs", root / POC, sources):
            with pytest.raises(paths.OutputLocationRefused, match="itself"):
                paths.check_record_dir(container, root=root)

    @pytest.mark.parametrize("inside", ["hknu_unified8", "hknu_unified8/_runs",
                                        "hknu_smpl24_paired/catalog", "prism_faithful_full/x/y"])
    def test_a_lineage_directory_is_refused(self, clean_env, tmp_path, inside):
        """A lineage directory holds a bundle or a corpus; only ``_runs`` beside them is a place
        for run records."""
        root = tmp_path / "root"
        with pytest.raises(paths.OutputLocationRefused, match="lineage directory"):
            paths.check_record_dir(root / POC / inside, root=root)

    def test_soma_data_roots_containers_and_lineages_are_refused_beside_the_run_root(
            self, data_root, tmp_path):
        other = tmp_path / "other"
        with pytest.raises(paths.OutputLocationRefused, match="itself"):
            paths.check_record_dir(data_root, root=other)
        with pytest.raises(paths.OutputLocationRefused, match="SOMA_DATA_ROOT/" + POC):
            paths.check_record_dir(data_root / POC / "amass_faithful_full" / "_runs", root=other)
        assert paths.check_record_dir(data_root / POC / "_runs", root=other) == \
            data_root / POC / "_runs"
        # without a run root, SOMA_DATA_ROOT's are guarded the same way
        with pytest.raises(paths.OutputLocationRefused, match="lineage directory"):
            paths.check_record_dir(data_root / POC / "amass_faithful_full")

    @pytest.mark.parametrize("folder", ["extracted/prism", "raw_archives", "body_models/smpl",
                                        f"{POC}/_superseded/x", f"{POC}/_manifest_backfill"])
    def test_read_only_and_evidence_folders_are_refused(self, clean_env, tmp_path, folder):
        root = tmp_path / "root"
        with pytest.raises(paths.OutputLocationRefused, match="read-only"):
            paths.check_record_dir(root / folder / "_runs", root=root)

    def test_a_synchronised_folder_is_warned_about(self, clean_env, tmp_path, capsys):
        target = tmp_path / "Dropbox" / "_runs"
        assert paths.check_record_dir(target, root=tmp_path / "root") == target
        assert "cloud-synchronised" in capsys.readouterr().err

    def test_the_configured_data_root_is_guarded_without_a_run_root(self, data_root):
        with pytest.raises(paths.OutputLocationRefused, match="source root"):
            paths.check_record_dir(data_root / "extracted" / "_runs")
        assert paths.check_record_dir(data_root / POC / "_runs") == data_root / POC / "_runs"


class TestCheckRecordFile:
    """A validation report or ledger written outside its bundle (``pipeline --report``)."""

    @pytest.mark.parametrize("relative", list(paths.PROTECTED_DATA_FILES))
    def test_the_data_planes_own_files_are_never_overwritten(self, clean_env, tmp_path,
                                                             relative):
        root = tmp_path / "root"
        with pytest.raises(paths.OutputLocationRefused):
            paths.check_record_file(root / relative, root=root)

    def test_its_folder_gets_the_record_folder_check(self, clean_env, tmp_path):
        root = tmp_path / "root"
        with pytest.raises(paths.OutputLocationRefused, match="lineage directory"):
            paths.check_record_file(root / POC / "prism_faithful_full" / "report.json", root=root)
        with pytest.raises(paths.OutputLocationRefused, match="read-only"):
            paths.check_record_file(root / "extracted" / "prism" / "report.json", root=root)
        target = root / POC / "_runs" / "report.json"
        assert paths.check_record_file(target, root=root) == target
        assert paths.check_record_file(tmp_path / "out" / "r.json", root=root) == \
            tmp_path / "out" / "r.json"


class TestCheckReadmeRoot:
    """``--top-level-root`` writes ``<folder>/README.md``."""

    def test_a_data_root_and_the_source_root_themselves_are_refused(self, clean_env, data_root,
                                                                    tmp_path):
        sources = tmp_path / "sources"
        sources.mkdir()
        other = tmp_path / "other"
        with pytest.raises(paths.OutputLocationRefused, match="never overwritten"):
            paths.check_readme_root(data_root)                     # SOMA_DATA_ROOT
        with pytest.raises(paths.OutputLocationRefused, match="never overwritten"):
            paths.check_readme_root(other, root=other)             # the run's own data root
        with pytest.raises(paths.OutputLocationRefused, match="never overwritten"):
            paths.check_readme_root(data_root, root=other)         # and SOMA_DATA_ROOT beside it
        clean_env.setenv(paths.SOURCE_ROOT_ENV, str(sources))
        with pytest.raises(paths.OutputLocationRefused, match="SOMA_SOURCE_ROOT itself"):
            paths.check_readme_root(sources)

    @pytest.mark.parametrize("folder", ["extracted", "extracted/prism", "raw_archives",
                                        "body_models/smpl", f"{POC}/_superseded/x",
                                        f"{POC}/_runs", f"{POC}/_manifest_backfill"])
    def test_read_only_and_evidence_folders_are_refused(self, data_root, folder):
        with pytest.raises(paths.OutputLocationRefused, match="read-only"):
            paths.check_readme_root(data_root / folder)

    def test_the_lineage_container_and_a_folder_of_ones_own_are_allowed(self, data_root,
                                                                        tmp_path):
        for folder in (data_root / POC, data_root / "runs", tmp_path / "mine"):
            assert paths.check_readme_root(folder) == folder
        synced = tmp_path / "Dropbox" / "x"
        assert paths.check_readme_root(synced) == synced              # warned, not refused


class TestCheckBundleDir:
    """The bundle that validate (writing into it), readme or register take: what run refuses."""

    @pytest.mark.parametrize("name", paths.EVIDENCE_FOLDERS)
    def test_an_evidence_folder_is_found_under_any_root(self, clean_env, tmp_path, name):
        inside = tmp_path / "elsewhere" / POC / name / "gaitex_unified8_2026-09-16" / "take"
        assert paths.evidence_folder(inside) == name
        assert paths.evidence_folder(tmp_path / "elsewhere" / POC / name) == name
        if os.name == "nt":                                   # the file system folds case
            assert paths.evidence_folder(tmp_path / "e" / "RUNS" /
                                         "Experimental_Generation_POC_Demo" / name.upper()) == \
                name.upper()
        for other in (tmp_path / "elsewhere" / POC / "gaitex_unified8",
                      tmp_path / "elsewhere" / POC, tmp_path / name / "x",
                      tmp_path / "elsewhere" / POC / "gaitex_unified8" / name):
            assert paths.evidence_folder(other) is None, other

    @pytest.mark.parametrize("writes", [True, False])
    def test_evidence_is_refused_without_any_root_configured(self, clean_env, tmp_path, writes):
        bundle = tmp_path / "elsewhere" / POC / "_superseded" / "gaitex_unified8_2026-09-16"
        action = "write" if writes else "register"
        with pytest.raises(paths.OutputLocationRefused,
                           match=f"refusing to {action} .*evidence folder .*_superseded"):
            paths.check_bundle_dir(bundle, writes=writes, action=action)

    @pytest.mark.parametrize("folder", ["extracted/gaitex/x", "raw_archives/x",
                                        "body_models/smpl/x"])
    def test_the_read_only_folders_of_the_data_root_are_refused(self, data_root, folder):
        for writes in (True, False):
            with pytest.raises(paths.OutputLocationRefused, match="read-only"):
                paths.check_bundle_dir(data_root / folder, writes=writes)
        other = data_root.parent / "run_root"
        with pytest.raises(paths.OutputLocationRefused, match="read-only"):
            paths.check_bundle_dir(other / folder, root=other, writes=False)

    def test_a_lineage_or_scratch_bundle_is_taken(self, data_root, tmp_path):
        for bundle in (data_root / POC / "gaitex_unified8", data_root / "tmp" / "gaitex_unified8",
                       tmp_path / "mine" / "gaitex_unified8"):
            assert paths.check_bundle_dir(bundle) == bundle
            assert paths.check_bundle_dir(bundle, writes=False) == bundle

    def test_only_a_command_that_writes_into_the_bundle_gets_the_rest_of_the_output_guard(
            self, data_root, tmp_path, capsys):
        container = data_root / POC
        container.mkdir(parents=True)
        with pytest.raises(paths.OutputLocationRefused, match="itself"):
            paths.check_bundle_dir(container)
        assert paths.check_bundle_dir(container, writes=False) == container
        synced = tmp_path / "Dropbox" / "gaitex_unified8"
        assert paths.check_bundle_dir(synced, writes=False) == synced   # register reads it
        assert "warning:" not in capsys.readouterr().err
        assert paths.check_bundle_dir(synced) == synced                 # validate, readme write
        assert "cloud-synchronised" in capsys.readouterr().err


# ---------------------------------------------------------------- import-time behaviour


def _relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


#: The generation closure: every generator (a glob, so a new
#: one is covered the day it lands), the scripts they load by path, and the packages they import.
#: No exclusions; the retired generate_prism_poc_demo.py was parametrized rather than exempted.
GENERATION_SCRIPTS = tuple(sorted(
    [_relative(p) for p in (REPO / "scripts" / "poc").glob("generate_*.py")]
    + ["scripts/poc/anthro_smpl.py", "scripts/poc/smpl_recon.py", "scripts/poc/unified8_emit.py",
       "scripts/diagnostics/hknu_retarget_probe.py",
       "scripts/diagnostics/pose_discontinuity_scan.py"]))
GENERATION_PACKAGES = tuple(sorted(
    _relative(p)
    for package in ("addbio_retarget", "gaitex_retarget", "gaitex_synthesis")
    for p in (REPO / "src" / "soma_synth" / package).rglob("*.py")))
GENERATION_CLOSURE = (GENERATION_SCRIPTS + GENERATION_PACKAGES
                      + ("src/soma_synth/pipeline/paths.py",))
#: A drive-letter path (E:\, X:/ ...; not the "ID:" of ordinary text, which a letter
#: precedes) or a user profile. Shared-drive names come from SOMA_SHARED_DRIVE_NAMES only.
ABSOLUTE_LOCATION = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|[\\/]Users[\\/]")


def test_the_closure_is_the_one_the_audit_scans():
    assert "scripts/poc/generate_prism_poc_demo.py" in GENERATION_SCRIPTS
    assert "scripts/poc/generate_prism_measured.py" in GENERATION_SCRIPTS
    assert len([s for s in GENERATION_SCRIPTS if "/generate_" in s]) >= 14
    assert len(GENERATION_CLOSURE) >= 38
    assert any(p.startswith("src/soma_synth/gaitex_retarget/") for p in GENERATION_PACKAGES)
    assert any(p.startswith("src/soma_synth/gaitex_synthesis/") for p in GENERATION_PACKAGES)
    assert any(p.startswith("src/soma_synth/addbio_retarget/") for p in GENERATION_PACKAGES)


@pytest.mark.parametrize("text, hit", [
    (r"E:\data", True), ("F:/data", True), (r"X:\shared", True),
    ("X:/My Drive/x", True), (r"C:\Users\someone\data", True), ("/Users/kim/data", True),
    ("subject ID: 7", False), ("ID:/a", False), ("https://example.org/x", False),
    ("$SOMA_DATA_ROOT/extracted", False), ("ratio 1:2", False),
])
def test_the_marker_pattern(text, hit):
    assert bool(ABSOLUTE_LOCATION.search(text)) is hit


def _code_strings(source: str) -> list[tuple[int, str]]:
    """Every string constant of a module except its docstrings, with its line."""
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


@pytest.mark.parametrize("relative", GENERATION_CLOSURE)
def test_no_absolute_path_in_generation_code(relative):
    """Zero drive-letter constants. Docstrings may name the variables."""
    source = (REPO / relative).read_text(encoding="utf-8")
    found = [(line, text) for line, text in _code_strings(source) if ABSOLUTE_LOCATION.search(text)]
    assert not found, f"{relative} carries an absolute path: {found}"


def test_the_generation_scripts_import_with_nothing_configured():
    """In a clean interpreter, with none of the three variables set, every script imports and
    only a call that needs a location refuses.

    A subprocess, so the generators' own sys.modules registrations cannot leak into this run."""
    code = (
        "import importlib.util, sys\n"
        "from pathlib import Path\n"
        "bad = []\n"
        "for i, rel in enumerate(sys.argv[1:]):\n"
        "    spec = importlib.util.spec_from_file_location(f'closure_import_{i}', Path(rel))\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    sys.modules[spec.name] = module\n"
        "    try:\n"
        "        spec.loader.exec_module(module)\n"
        "    except Exception as error:\n"
        "        bad.append(f'{rel}: {type(error).__name__}: {error}')\n"
        "from soma_synth.pipeline import paths\n"
        "try:\n"
        "    paths.data_root()\n"
        "    bad.append('data_root() accepted an unset SOMA_DATA_ROOT')\n"
        "except paths.PathConfigError:\n"
        "    pass\n"
        "print('\\n'.join(bad))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ENV + ("ADDBIO_RETARGET_CORPUS",)}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO / "packages" / "smpl18" / "src"), str(REPO / "scripts" / "poc")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    scripts = [f for f in GENERATION_SCRIPTS if f != "scripts/diagnostics/pose_discontinuity_scan.py"]
    result = subprocess.run([sys.executable, "-c", code, *scripts], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
