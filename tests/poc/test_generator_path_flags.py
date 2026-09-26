"""The generators' location flags and path defaults, on synthetic fixtures only.

Nothing here reads a source or generates a take (generation is under a hold). Each test exercises
the enumeration or path resolution a flag changes and checks that, without the flag, the result is
what the generator produced before.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from soma_synth.pipeline import paths

REPO = Path(__file__).resolve().parents[2]
POC = REPO / "scripts" / "poc"
ENV = ("SOMA_DATA_ROOT", "SOMA_SOURCE_ROOT", "SOMA_BODY_MODEL_DIR", "ADDBIO_RETARGET_CORPUS")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, POC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV:
        # set-then-delete so that a value a generator's main() writes is undone after the test
        monkeypatch.setenv(name, "unused")
        monkeypatch.delenv(name)
    return monkeypatch


@pytest.fixture
def data_root(clean_env, tmp_path):
    root = tmp_path / "data"
    (root / "extracted").mkdir(parents=True)
    clean_env.setenv("SOMA_DATA_ROOT", str(root))
    return root


# ================================================================ PRISM --source-root / --out


@pytest.fixture(scope="module")
def prism_all():
    return _load("flags_prism_measured_all", "generate_prism_measured_all.py")


def _prism_tree(folder: Path) -> None:
    for subject, takes in (("subj001", ("take002", "take003")), ("subj005", ("take001",))):
        (folder / subject).mkdir(parents=True)
        for take in takes:
            (folder / subject / f"{take}.pkl").write_bytes(b"x")


class TestPrismInputAndOutputAreSeparate:
    def test_data_root_alone_keeps_its_old_meaning(self, prism_all, clean_env, tmp_path):
        args = prism_all.build_parser().parse_args(["--data-root", str(tmp_path)])
        source, out = prism_all.resolve_locations(args)
        assert source == tmp_path / "extracted" / "prism"
        assert out == tmp_path / "runs" / "experimental_generation_poc_demo" / "prism_faithful_full"

    @pytest.mark.parametrize("extra", [["--source-root", "somewhere"], ["--out", "elsewhere"]])
    def test_data_root_does_not_combine_with_the_new_flags(self, prism_all, tmp_path, extra, capsys):
        with pytest.raises(SystemExit) as stopped:
            prism_all.main(["--data-root", str(tmp_path), *extra])
        assert stopped.value.code == 2
        assert "cannot be combined" in capsys.readouterr().err

    def test_without_flags_both_sides_come_from_the_environment(self, prism_all, data_root):
        args = prism_all.build_parser().parse_args([])
        source, out = prism_all.resolve_locations(args)
        assert source == data_root / "extracted" / "prism"
        assert out == data_root / "runs" / "experimental_generation_poc_demo" / "prism_faithful_full"

    def test_each_side_moves_on_its_own(self, prism_all, data_root, tmp_path):
        parse = prism_all.build_parser().parse_args
        source, out = prism_all.resolve_locations(parse(["--out", str(tmp_path / "bundle")]))
        assert (source, out) == (data_root / "extracted" / "prism", tmp_path / "bundle")
        source, out = prism_all.resolve_locations(parse(["--source-root", str(tmp_path / "copy")]))
        assert source == tmp_path / "copy"
        assert out == data_root / "runs" / "experimental_generation_poc_demo" / "prism_faithful_full"

    def test_a_relocated_source_enumerates_the_same_takes_under_the_same_ids(self, prism_all, tmp_path):
        old_root = tmp_path / "old"
        _prism_tree(old_root / "extracted" / "prism")
        moved = tmp_path / "anywhere" / "prism_copy"
        _prism_tree(moved)
        genders = {"prism_subj001": "M", "prism_subj005": "F"}

        def key(spec):
            return spec.subject_id, spec.take_id, spec.gender, spec.window, spec.relative_path

        before = prism_all.enumerate_specs(str(old_root), gender_by_subject=genders)
        after = prism_all.enumerate_specs(gender_by_subject=genders, source_dir=moved)
        assert [key(s) for s in after] == [key(s) for s in before] == [
            ("prism_subj001", "take002", "M", None, "extracted/prism/subj001/take002.pkl"),
            ("prism_subj001", "take003", "M", None, "extracted/prism/subj001/take003.pkl"),
            ("prism_subj005", "take001", "F", None, "extracted/prism/subj005/take001.pkl"),
        ]
        assert after[0].source_pkl == str(moved / "subj001" / "take002.pkl")

    def test_the_output_is_checked_before_the_source_is_read(self, prism_all, data_root, tmp_path,
                                                              capsys):
        code = prism_all.main(["--source-root", str(tmp_path / "absent"),
                               "--out", str(data_root / "extracted" / "prism_bundle")])
        assert code == 2
        assert "source root" in capsys.readouterr().out
        assert not (data_root / "extracted" / "prism_bundle").exists()

    def test_an_output_inside_an_explicit_source_is_refused(self, prism_all, data_root, tmp_path,
                                                            capsys):
        moved = tmp_path / "prism_copy"
        _prism_tree(moved)
        assert prism_all.main(["--source-root", str(moved), "--out", str(moved / "bundle")]) == 2
        assert "a folder this run reads" in capsys.readouterr().out
        assert not (moved / "bundle").exists()

    def test_data_root_guards_its_own_extracted_folder(self, prism_all, data_root, tmp_path):
        """With --data-root the guard protects <root>/extracted, not only $SOMA_SOURCE_ROOT."""
        other = tmp_path / "other_root"
        (other / "extracted").mkdir(parents=True)
        args = prism_all.build_parser().parse_args(["--data-root", str(other)])
        _source, out = prism_all.resolve_locations(args)
        assert paths.check_output_dir(out, inputs=[other / "extracted"]) == out
        with pytest.raises(paths.OutputLocationRefused, match="a folder this run reads"):
            paths.check_output_dir(other / "extracted" / "x", inputs=[other / "extracted"])

    def test_data_root_still_needs_the_variable_for_the_body_models(self, prism_all, clean_env,
                                                                    tmp_path, capsys):
        """The docstring's promise: --data-root moves the source and the bundle, not the models,
        which come from $SOMA_DATA_ROOT/body_models/smpl. Without the variable nothing is written."""
        root = tmp_path / "root"
        _prism_tree(root / "extracted" / "prism")
        assert prism_all.main(["--data-root", str(root), "--only", "prism_subj005"]) == 2
        assert "SOMA_DATA_ROOT is not set" in capsys.readouterr().out
        assert not (root / "runs").exists()

    @pytest.mark.parametrize("script", ["generate_prism_measured_all.py",
                                        "generate_prism_faithful_all.py"])
    def test_an_unknown_only_subject_is_an_error_not_an_empty_bundle(self, script, data_root,
                                                                    clean_env, tmp_path, capsys):
        """--only needs a scratch output (see TestAPartialSelectionNeedsAScratchOutput), here a
        scratch --data-root, which both mains take."""
        module = _load(f"flags_{script[:-3]}", script)
        scratch = tmp_path / "scratch_root"
        _prism_tree(scratch / "extracted" / "prism")
        genders = {"prism_subj001": "M", "prism_subj005": "F"}
        # enumerate_specs lives in faithful_all; the stub pkl files carry no gender to read
        clean_env.setattr(getattr(module, "faithful_all", module), "_gender_by_subject",
                          lambda *a, **k: genders)
        assert module.main(["--data-root", str(scratch), "--only", "prism_subj05"]) == 2
        out = capsys.readouterr().out
        assert "prism_subj05" in out and "prism_subj001, prism_subj005" in out
        assert not (scratch / "runs").exists()
        assert not (data_root / "runs").exists()

    def test_faithful_all_reads_the_source_root_the_guard_protects(self, data_root, clean_env,
                                                                   tmp_path, capsys):
        """Without --data-root, faithful_all reads $SOMA_SOURCE_ROOT/prism, as measured_all does:
        the refusal of a source folder with no subject names that folder."""
        module = _load("flags_prism_faithful_all", "generate_prism_faithful_all.py")
        moved = tmp_path / "moved_sources"
        (moved / "prism").mkdir(parents=True)
        clean_env.setenv("SOMA_SOURCE_ROOT", str(moved))
        clean_env.setattr(module, "_gender_by_subject", lambda *a, **k: {})
        assert module.main([]) == 2
        assert str(moved / "prism") in capsys.readouterr().out
        assert not (data_root / "runs").exists()


# ================================================================ the retired PRISM demo


class TestPrismPocDemo:
    """generate_prism_poc_demo.py is retired and is parametrized rather than exempted, so the
    check holds over every generate_*.py."""

    def test_it_imports_and_resolves_nothing(self, clean_env):
        module = _load("flags_prism_poc_demo", "generate_prism_poc_demo.py")
        assert not hasattr(module, "SOMA_DATA_ROOT")
        assert not hasattr(module, "OUT_DIR") and not hasattr(module, "SOURCE_PKL")
        assert module.SOURCE_TAKE == ("subj001", "take002.pkl")

    def test_an_unconfigured_pc_is_refused_before_anything_is_read(self, clean_env, capsys):
        module = _load("flags_prism_poc_demo", "generate_prism_poc_demo.py")
        assert module.main() == 2
        assert "SOMA_DATA_ROOT" in capsys.readouterr().out


# ================================================================ AMASS --groups


@pytest.fixture
def amass_all(monkeypatch):
    module = _load("flags_amass_faithful_all", "generate_amass_faithful_all.py")
    from smpl18.skeleton import definition

    rest = np.zeros((24, 3))
    for joint in range(1, 24):
        rest[joint] = rest[definition.PARENTS[joint]] + np.array([0.02 * joint, -0.1, 0.03])
    monkeypatch.setattr(module.gen.anthro_smpl, "load_smpl_model_for_gender", lambda letter: {})
    monkeypatch.setattr(module.gen.anthro_smpl, "rest_joints",
                        lambda model, betas: rest * (1.0 + float(betas[0])))
    return module


def _amass_tree(root: Path) -> str:
    amass = root / "amass"
    rng = np.random.default_rng(7)
    for dataset, subject, name, betas0 in (
        ("CMU", "01", "01_02_poses.npz", 0.0),
        ("CMU", "01", "01_04_poses.npz", 0.2),      # same subject, other betas: a second group
        ("CMU", "02", "02_01_poses.npz", 0.0),
        ("KIT", "10", "Walk_poses.npz", 0.0),
        ("KIT", "100", "Walk_poses.npz", 0.1),      # a name that merely starts like "10"
    ):
        folder = amass / dataset / subject
        folder.mkdir(parents=True, exist_ok=True)
        betas = np.zeros(16)
        betas[0] = betas0
        np.savez(folder / name, poses=rng.normal(0, 0.1, (30, 156)), trans=np.zeros((30, 3)),
                 betas=betas, gender=np.array("male"), mocap_framerate=np.array(100.0))
    (amass / "KIT" / "10" / "shape.npz").write_bytes(b"")
    return str(amass)


def _identity(spec):
    return spec.dataset, spec.subject_id, spec.sequence_id, spec.source_npz, spec.relative_path


class TestAmassGroups:
    def test_the_value_is_dataset_slash_subject(self, amass_all):
        assert amass_all.parse_group("TotalCapture/s4") == ("TotalCapture", "s4")
        for bad in ("TotalCapture", "a/b/c", "/s4", "CMU/"):
            with pytest.raises(argparse.ArgumentTypeError):
                amass_all.parse_group(bad)

    def test_a_group_is_every_file_of_the_subject_in_enumeration_order(self, amass_all, tmp_path):
        root = _amass_tree(tmp_path)
        everything = amass_all.enumerate_specs(root)
        chosen = amass_all.enumerate_specs(root, None, [("KIT", "10"), ("CMU", "01")])
        expected = [_identity(s) for s in everything
                    if (s.dataset, s.subject_id) in {("CMU", "amass_CMU_01"), ("KIT", "amass_KIT_10")}]
        assert [_identity(s) for s in chosen] == expected
        assert [s.sequence_id for s in chosen] == ["01_02_poses", "01_04_poses", "Walk_poses", "shape"]
        assert all(s.subject_id != "amass_KIT_100" for s in chosen)

    def test_without_groups_nothing_changes(self, amass_all, tmp_path):
        root = _amass_tree(tmp_path)
        assert ([_identity(s) for s in amass_all.enumerate_specs(root, None, None)]
                == [_identity(s) for s in amass_all.enumerate_specs(root)])
        assert len(amass_all.enumerate_specs(root)) == 6

    def test_an_unknown_group_is_an_error_not_an_empty_run(self, amass_all, tmp_path):
        root = _amass_tree(tmp_path)
        with pytest.raises(ValueError, match="CMU/99"):
            amass_all.enumerate_specs(root, None, [("CMU", "01"), ("CMU", "99")])
        with pytest.raises(ValueError, match="KIT/10"):
            amass_all.enumerate_specs(root, ["CMU"], [("KIT", "10")])

    def test_the_selected_groups_fit_exactly_as_in_the_full_run(self, amass_all, tmp_path):
        root = _amass_tree(tmp_path)
        _s, full_groups, full_takes, _u = amass_all.fit_all_groups(amass_all.enumerate_specs(root))
        _s, groups, takes, _u = amass_all.fit_all_groups(
            amass_all.enumerate_specs(root, None, [("CMU", "01")]))
        assert len(groups) == 2                       # both betas groups of the subject, whole
        assert {rel: full_takes[rel] for rel in takes} == takes
        assert {name: full_groups[name] for name in groups} == groups

    def test_main_refuses_an_unknown_group_before_writing(self, amass_all, data_root, tmp_path, capsys):
        root = _amass_tree(tmp_path)
        out = tmp_path / "bundle"
        assert amass_all.main(["--amass-root", root, "--out-root", str(out),
                               "--groups", "CMU/99"]) == 2
        assert "CMU/99" in capsys.readouterr().out
        assert not out.exists()

    def test_main_refuses_an_output_inside_the_amass_root(self, amass_all, clean_env, tmp_path,
                                                          capsys):
        """An explicit --amass-root is guarded even on a PC with no environment configured."""
        root = _amass_tree(tmp_path)
        out = Path(root) / "bundle"
        assert amass_all.main(["--amass-root", root, "--out-root", str(out)]) == 2
        assert "a folder this run reads" in capsys.readouterr().out
        assert not out.exists()

    def test_main_refuses_the_poc_demo_folder_itself(self, amass_all, data_root, tmp_path, capsys):
        """2026-09-15: shards once wrote straight into runs/experimental_generation_poc_demo."""
        root = _amass_tree(tmp_path)
        poc = data_root / "runs" / "experimental_generation_poc_demo"
        assert amass_all.main(["--amass-root", root, "--out-root", str(poc)]) == 2
        assert "lineage container" in capsys.readouterr().out
        assert not poc.exists()

    def test_limit_does_not_combine_with_groups(self, amass_all, data_root, tmp_path, capsys):
        """--limit cut the spec list after --groups and before the fits, so a group could be fitted
        on part of its sequences and get other frozen-joint constants than a full run gives it."""
        root = _amass_tree(tmp_path)
        out = tmp_path / "bundle"
        assert amass_all.main(["--amass-root", root, "--out-root", str(out),
                               "--groups", "CMU/01", "--limit", "1"]) == 2
        assert "--limit cannot be combined with --groups" in capsys.readouterr().out
        assert not out.exists()

    def test_explicit_roots_still_need_the_body_models_configured(self, amass_all, clean_env,
                                                                  tmp_path, capsys):
        """Refused before anything is written, not by a traceback from inside the fits."""
        root = _amass_tree(tmp_path)
        out = tmp_path / "bundle"
        assert amass_all.main(["--amass-root", root, "--out-root", str(out)]) == 2
        assert "SOMA_DATA_ROOT is not set" in capsys.readouterr().out
        assert not out.exists()


# ================================================================ an empty selection writes nothing
#
# An existing but empty source folder selected nothing, and the AMASS and PRISM batch mains still
# rewrote INDEX.json, reduced_model_fit.json and the description (PRISM: its policy file too) in
# their output, which defaults to the production lineage. Each test puts a stand-in production
# INDEX.json there first and requires it byte-identical, and alone, afterwards.

PRODUCTION_INDEX = b'{"takes": ["the production bundle an empty run must not touch"]}\n'
PRISM_MAINS = ["generate_prism_measured_all.py", "generate_prism_faithful_all.py"]


def _bundle_with_index(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    index = folder / "INDEX.json"
    index.write_bytes(PRODUCTION_INDEX)
    return index


def _untouched(index: Path) -> bool:
    return (index.read_bytes() == PRODUCTION_INDEX
            and sorted(p.name for p in index.parent.iterdir()) == ["INDEX.json"])


def _lineage(data_root: Path, name: str) -> Path:
    return data_root / "runs" / "experimental_generation_poc_demo" / name


class TestAnEmptySelectionWritesNothing:
    @pytest.mark.parametrize("merge", [[], ["--merge"]], ids=["fit", "merge"])
    def test_amass_an_empty_source_folder(self, amass_all, data_root, capsys, merge):
        source = data_root / "extracted" / "amass"
        source.mkdir()
        index = _bundle_with_index(_lineage(data_root, "amass_faithful_full"))
        assert amass_all.main(merge) == 2
        out = capsys.readouterr().out
        assert "no AMASS file selected" in out and str(source) in out
        assert _untouched(index)

    @pytest.mark.parametrize("extra", [["--datasets", "NoSuchDataset"], ["--limit", "0"],
                                       ["--datasets", "NoSuchDataset", "--merge"]])
    def test_amass_a_selection_of_nothing(self, amass_all, data_root, tmp_path, capsys, extra):
        root = _amass_tree(tmp_path)
        index = _bundle_with_index(tmp_path / "bundle")
        assert amass_all.main(["--amass-root", root, "--out-root", str(index.parent), *extra]) == 2
        out = capsys.readouterr().out
        assert "no AMASS file selected" in out and root in out
        assert _untouched(index)

    @pytest.mark.parametrize("given", [True, False], ids=["flag", "default"])
    def test_amass_a_missing_source_folder_is_named(self, amass_all, data_root, tmp_path, capsys,
                                                    given):
        index = _bundle_with_index(tmp_path / "bundle")
        missing = tmp_path / "absent" if given else data_root / "extracted" / "amass"
        flags = ["--amass-root", str(missing)] if given else []
        assert amass_all.main([*flags, "--out-root", str(index.parent)]) == 2
        out = capsys.readouterr().out
        assert str(missing) in out and "does not exist" in out
        assert ("as given" if given else "SOMA_DATA_ROOT/extracted/amass") in out
        assert _untouched(index)

    @pytest.mark.parametrize("script", PRISM_MAINS)
    def test_prism_an_empty_source_folder(self, script, data_root, capsys):
        module = _load(f"flags_{script[:-3]}", script)
        source = data_root / "extracted" / "prism"
        source.mkdir()
        index = _bundle_with_index(_lineage(data_root, "prism_faithful_full"))
        assert module.main([]) == 2
        out = capsys.readouterr().out
        assert "holds no subject folder" in out and str(source) in out
        assert _untouched(index)

    @pytest.mark.parametrize("script", PRISM_MAINS)
    def test_prism_data_root_with_an_empty_source_folder(self, script, data_root, tmp_path, capsys):
        other = tmp_path / "other_root"
        (other / "extracted" / "prism").mkdir(parents=True)
        index = _bundle_with_index(_lineage(other, "prism_faithful_full"))
        module = _load(f"flags_{script[:-3]}", script)
        assert module.main(["--data-root", str(other)]) == 2
        assert str(other / "extracted" / "prism") in capsys.readouterr().out
        assert _untouched(index)

    @pytest.mark.parametrize("script", PRISM_MAINS)
    def test_prism_a_missing_source_folder_is_named(self, script, data_root, capsys):
        module = _load(f"flags_{script[:-3]}", script)
        index = _bundle_with_index(_lineage(data_root, "prism_faithful_full"))
        assert module.main([]) == 2
        out = capsys.readouterr().out
        assert "SOMA_DATA_ROOT/extracted/prism" in out and "does not exist" in out
        assert _untouched(index)

    @pytest.mark.parametrize("script", PRISM_MAINS)
    def test_prism_a_selection_of_nothing(self, script, data_root, clean_env, tmp_path, capsys):
        """In a scratch --data-root: --limit into the production lineage is refused before this."""
        module = _load(f"flags_{script[:-3]}", script)
        scratch = tmp_path / "scratch_root"
        _prism_tree(scratch / "extracted" / "prism")
        clean_env.setattr(getattr(module, "faithful_all", module), "_gender_by_subject",
                          lambda *a, **k: {"prism_subj001": "M", "prism_subj005": "F"})
        index = _bundle_with_index(_lineage(scratch, "prism_faithful_full"))
        assert module.main(["--data-root", str(scratch), "--limit", "-3"]) == 2  # three takes, all cut
        assert "no PRISM take selected" in capsys.readouterr().out
        assert _untouched(index)

    def test_a_source_root_given_as_the_prism_folder_is_refused(self, prism_all, data_root,
                                                                tmp_path, capsys):
        """--source-root is the PRISM folder itself; the source root that holds it selects
        nothing, and the refusal names the folder that was probably meant."""
        sources = tmp_path / "sources"
        _prism_tree(sources / "prism")
        index = _bundle_with_index(tmp_path / "bundle")
        assert prism_all.main(["--source-root", str(sources), "--out", str(index.parent)]) == 2
        out = " ".join(capsys.readouterr().out.split())
        assert "not SOMA_SOURCE_ROOT" in out
        assert f"did you mean --source-root {sources / 'prism'}?" in out
        assert _untouched(index)

    def test_the_help_says_the_source_root_is_the_prism_folder(self, prism_all):
        text = " ".join(prism_all.build_parser().format_help().split())
        assert "the PRISM folder itself" in text and "not SOMA_SOURCE_ROOT" in text


# ================================================================ AddBio --only


@pytest.fixture(scope="module")
def addbio_faithful():
    return _load("flags_addbio_faithful", "generate_addbio_faithful.py")


@pytest.fixture(scope="module")
def addbio_unified8():
    return _load("flags_addbio_unified8", "generate_addbio_unified8.py")


@pytest.fixture(scope="module")
def addbio_smpl24():
    return _load("flags_addbio_smpl24", "generate_addbio_smpl24.py")


def _corpus(root: Path) -> Path:
    for study, subjects in (("Hamner2013_Formatted_No_Arm", ("subject01", "subject02")),
                            ("Tiziana2019_Formatted_With_Arm", ("Subject43", "Subject44")),
                            ("Zeta2020_Formatted_No_Arm", ("s1",))):
        for subject in subjects:
            (root / study / subject).mkdir(parents=True)
    return root


def _old_enumeration(corpus: Path, studies=None, subjects=None):
    """The loop both AddBio synth generators ran before `--only`."""
    names = sorted(p.name for p in corpus.iterdir() if p.is_dir())
    if studies:
        names = names[:studies]
    pairs = []
    for study in names:
        members = sorted(p.name for p in (corpus / study).iterdir() if p.is_dir())
        if subjects:
            members = members[:subjects]
        pairs.extend((study, s) for s in members)
    return names, pairs


class TestAddbioOnly:
    def test_the_value_is_study_slash_subject(self, addbio_faithful, addbio_smpl24):
        for parse in (addbio_faithful.study_subject, addbio_smpl24.study_subject):
            assert parse("Hamner2013_Formatted_No_Arm/subject01") == (
                "Hamner2013_Formatted_No_Arm", "subject01")
            for bad in ("subject01", "a/b/c", "/x", "x/"):
                with pytest.raises(argparse.ArgumentTypeError):
                    parse(bad)

    @pytest.mark.parametrize("studies, subjects", [(None, None), (2, None), (None, 1), (2, 1)])
    def test_without_only_the_walk_is_the_old_one(self, addbio_faithful, tmp_path, studies, subjects):
        corpus = _corpus(tmp_path / "corpus")
        assert (addbio_faithful.enumerate_subjects(corpus, studies, subjects)
                == _old_enumeration(corpus, studies, subjects))

    def test_only_keeps_the_named_subjects_in_walk_order(self, addbio_faithful, tmp_path):
        corpus = _corpus(tmp_path / "corpus")
        only = [("Tiziana2019_Formatted_With_Arm", "Subject43"),
                ("Hamner2013_Formatted_No_Arm", "subject01")]
        names, pairs = addbio_faithful.enumerate_subjects(corpus, only=only)
        assert names == ["Hamner2013_Formatted_No_Arm", "Tiziana2019_Formatted_With_Arm"]
        assert pairs == [("Hamner2013_Formatted_No_Arm", "subject01"),
                         ("Tiziana2019_Formatted_With_Arm", "Subject43")]

    def test_a_cap_that_keeps_every_named_subject_is_accepted(self, addbio_faithful, tmp_path):
        corpus = _corpus(tmp_path / "corpus")
        only = [("Hamner2013_Formatted_No_Arm", "subject02"),
                ("Tiziana2019_Formatted_With_Arm", "Subject43")]
        _names, pairs = addbio_faithful.enumerate_subjects(corpus, studies=2, subjects=1, only=only)
        assert pairs == only

    @pytest.mark.parametrize("studies, subjects, dropped", [
        (None, 1, "Hamner2013_Formatted_No_Arm/subject02"),
        (1, None, "Tiziana2019_Formatted_With_Arm/Subject43"),
    ])
    def test_a_cap_that_drops_a_named_subject_is_an_error(self, addbio_faithful, tmp_path, studies,
                                                          subjects, dropped):
        """The run converts every subject --only names or none; it used to drop the rest quietly."""
        corpus = _corpus(tmp_path / "corpus")
        only = [("Hamner2013_Formatted_No_Arm", "subject02"),
                ("Hamner2013_Formatted_No_Arm", "subject01"),
                ("Tiziana2019_Formatted_With_Arm", "Subject43")]
        with pytest.raises(ValueError, match="would leave out") as refused:
            addbio_faithful.enumerate_subjects(corpus, studies, subjects, only)
        assert dropped in str(refused.value)

    @pytest.mark.parametrize("generator", ["addbio_faithful", "addbio_unified8"])
    def test_main_refuses_a_cap_that_drops_a_named_subject(self, request, generator, data_root,
                                                           capsys):
        module = request.getfixturevalue(generator)
        corpus = _corpus(data_root / "tmp" / "corpus")
        out = data_root / "tmp" / "out"
        code = module.main(["--out", str(out), "--raw", str(corpus), "--subjects", "1",
                            "--only", "Hamner2013_Formatted_No_Arm/subject01",
                            "--only", "Hamner2013_Formatted_No_Arm/subject02"])
        assert code == 2
        assert "Hamner2013_Formatted_No_Arm/subject02" in capsys.readouterr().out
        assert not out.exists()

    def test_a_subject_the_corpus_lacks_is_an_error(self, addbio_faithful, tmp_path):
        corpus = _corpus(tmp_path / "corpus")
        with pytest.raises(ValueError, match="Hamner2013_Formatted_No_Arm/subject99"):
            addbio_faithful.enumerate_subjects(
                corpus, only=[("Hamner2013_Formatted_No_Arm", "subject99")])

    @pytest.mark.parametrize("only", [
        ("hamner2013_formatted_no_arm", "SUBJECT01"),
        ("Hamner2013_Formatted_No_Arm", "Subject01"),
        ("Tiziana2019_Formatted_With_Arm", "subject43"),
    ])
    def test_a_differently_cased_name_is_an_error_not_an_empty_run(self, addbio_faithful,
                                                                   tmp_path, only):
        """On NTFS `(corpus / study / subject).is_dir()` ignores case while the selection does
        not; the check must read the listing, or the run selects nothing and says nothing."""
        corpus = _corpus(tmp_path / "corpus")
        with pytest.raises(ValueError, match="case-sensitive"):
            addbio_faithful.enumerate_subjects(corpus, only=[only])

    @pytest.mark.parametrize("generator", ["addbio_faithful", "addbio_unified8"])
    def test_main_refuses_a_differently_cased_subject(self, request, generator, data_root,
                                                      tmp_path, capsys):
        module = request.getfixturevalue(generator)
        corpus = _corpus(data_root / "tmp" / "corpus")    # unified8 names takes against the root
        out = tmp_path / "out"
        code = module.main(["--out", str(out), "--raw", str(corpus),
                            "--only", "Tiziana2019_Formatted_With_Arm/subject43"])
        assert code == 2
        assert "Tiziana2019_Formatted_With_Arm/subject43" in capsys.readouterr().out
        assert not out.exists()          # no INDEX.json, SUMMARY.json or fit record over zero takes

    def test_the_corpus_variable_stays_an_alias_for_raw(self, addbio_faithful, addbio_unified8,
                                                        data_root, clean_env, tmp_path):
        lineage = data_root / "runs" / "experimental_generation_poc_demo" / "addbio_smpl24_raw"
        assert addbio_faithful.retarget_corpus() == lineage
        assert addbio_unified8.retarget_corpus() == lineage
        clean_env.setenv("ADDBIO_RETARGET_CORPUS", str(tmp_path / "scratch_corpus"))
        assert addbio_faithful.retarget_corpus() == tmp_path / "scratch_corpus"
        assert addbio_unified8.retarget_corpus() == tmp_path / "scratch_corpus"

    @pytest.mark.parametrize("generator", ["addbio_faithful", "addbio_unified8"])
    def test_main_refuses_a_missing_subject_before_converting(self, request, generator, data_root,
                                                              tmp_path, capsys):
        module = request.getfixturevalue(generator)
        corpus = _corpus(data_root / "tmp" / "corpus")    # unified8 names takes against the root
        code = module.main(["--out", str(tmp_path / "out"), "--raw", str(corpus),
                            "--only", "Hamner2013_Formatted_No_Arm/subject99"])
        assert code == 2
        assert "subject99" in capsys.readouterr().out
        # --raw reaches the pool's workers through the variable, as before
        assert os.environ["ADDBIO_RETARGET_CORPUS"] == str(corpus)
        assert not any((tmp_path / "out").rglob("*.npz"))

    @pytest.mark.parametrize("generator", ["addbio_faithful", "addbio_unified8"])
    def test_main_refuses_an_output_inside_the_sources(self, request, generator, data_root, tmp_path,
                                                       capsys):
        module = request.getfixturevalue(generator)
        corpus = _corpus(tmp_path / "corpus")
        code = module.main(["--out", str(data_root / "extracted" / "out"), "--raw", str(corpus)])
        assert code == 2
        assert "source root" in capsys.readouterr().out
        assert not (data_root / "extracted" / "out").exists()

    def test_unified8_refuses_a_corpus_outside_the_data_root_before_writing(
            self, addbio_unified8, data_root, tmp_path, capsys):
        """Every take's relative_path is written against the data root: a corpus elsewhere failed
        take by take and still left INDEX.json, reduced_model_fit.json and a description."""
        corpus = _corpus(tmp_path / "corpus")
        out = data_root / "tmp" / "out"
        assert addbio_unified8.main(["--out", str(out), "--raw", str(corpus)]) == 2
        assert "not inside the data root" in capsys.readouterr().out
        assert not out.exists()


def _b3d_tree(root: Path) -> Path:
    for split, variant, study, subject, payload in (
        ("train", "No_Arm", "Hamner2013_Formatted_No_Arm", "subject01", b"x"),
        ("train", "No_Arm", "Hamner2013_Formatted_No_Arm", "subject02", b"x"),
        ("train", "No_Arm", "Hamner2013_Formatted_No_Arm", "subject03", b""),     # ADR-0030 empty
        ("test", "No_Arm", "Hamner2013_Formatted_No_Arm", "subject10", b"x"),
        ("train", "With_Arm", "Hammer2013_Formatted_With_Arm", "subject01", b"x"),
        ("train", "With_Arm", "Tiziana2019_Formatted_With_Arm", "Subject43", b"x"),
    ):
        folder = root / split / variant / study / subject
        folder.mkdir(parents=True)
        (folder / f"{subject}.b3d").write_bytes(payload)
    return root


def _picked(files):
    return [(split, variant, study, path.parent.name) for split, variant, study, path in files]


class TestAddbioSmpl24Only:
    def test_without_only_every_payload_as_before(self, addbio_smpl24, tmp_path):
        root = _b3d_tree(tmp_path / "addbiomechanics")
        assert _picked(addbio_smpl24.payload_files(root, None)) == [
            ("test", "No_Arm", "Hamner2013_Formatted_No_Arm", "subject10"),
            ("train", "No_Arm", "Hamner2013_Formatted_No_Arm", "subject01"),
            ("train", "No_Arm", "Hamner2013_Formatted_No_Arm", "subject02"),
            ("train", "With_Arm", "Hammer2013_Formatted_With_Arm", "subject01"),
            ("train", "With_Arm", "Tiziana2019_Formatted_With_Arm", "Subject43"),
        ]
        assert _picked(addbio_smpl24.payload_files(root, 1)) == [
            ("test", "No_Arm", "Hamner2013_Formatted_No_Arm", "subject10"),
            ("train", "No_Arm", "Hamner2013_Formatted_No_Arm", "subject01"),
            ("train", "With_Arm", "Hammer2013_Formatted_With_Arm", "subject01"),
            ("train", "With_Arm", "Tiziana2019_Formatted_With_Arm", "Subject43"),
        ]

    def test_only_picks_the_harness_subjects_across_splits_and_variants(self, addbio_smpl24, tmp_path):
        root = _b3d_tree(tmp_path / "addbiomechanics")
        only = [("Hamner2013_Formatted_No_Arm", "subject01"),
                ("Hammer2013_Formatted_With_Arm", "subject01"),
                ("Tiziana2019_Formatted_With_Arm", "Subject43")]
        assert _picked(addbio_smpl24.payload_files(root, None, only)) == [
            ("train", "No_Arm", "Hamner2013_Formatted_No_Arm", "subject01"),
            ("train", "With_Arm", "Hammer2013_Formatted_With_Arm", "subject01"),
            ("train", "With_Arm", "Tiziana2019_Formatted_With_Arm", "Subject43"),
        ]

    def test_per_study_that_keeps_every_named_subject_is_accepted(self, addbio_smpl24, tmp_path):
        root = _b3d_tree(tmp_path / "addbiomechanics")
        only = [("Hamner2013_Formatted_No_Arm", "subject02"),
                ("Tiziana2019_Formatted_With_Arm", "Subject43")]
        assert _picked(addbio_smpl24.payload_files(root, 1, only)) == [
            ("train", "No_Arm", "Hamner2013_Formatted_No_Arm", "subject02"),
            ("train", "With_Arm", "Tiziana2019_Formatted_With_Arm", "Subject43")]

    def test_per_study_that_drops_a_named_subject_is_an_error(self, addbio_smpl24, tmp_path):
        """It used to convert the first and leave the second out without a word."""
        root = _b3d_tree(tmp_path / "addbiomechanics")
        only = [("Hamner2013_Formatted_No_Arm", "subject02"),
                ("Hamner2013_Formatted_No_Arm", "subject01")]
        with pytest.raises(ValueError, match="would leave out .*Hamner2013_Formatted_No_Arm/subject02"):
            addbio_smpl24.payload_files(root, 1, only)

    def test_main_refuses_an_unconfigured_body_model_folder_before_converting(
            self, addbio_smpl24, clean_env, tmp_path, capsys):
        """Explicit --source-root and --out, no SOMA_DATA_ROOT: the models' folder is unknown, and
        the conversion loop would have crashed on it with a traceback."""
        root = _b3d_tree(tmp_path / "addbiomechanics")
        out = tmp_path / "out"
        assert addbio_smpl24.main(["--source-root", str(root), "--out", str(out)]) == 2
        assert "SOMA_DATA_ROOT is not set" in capsys.readouterr().out
        assert not out.exists()

    @pytest.mark.parametrize("missing", [("Hamner2013_Formatted_No_Arm", "subject03"),   # empty
                                         ("Hamner2013_Formatted_No_Arm", "subject99"),
                                         ("NoSuchStudy", "subject01")])
    def test_a_subject_without_payload_is_an_error(self, addbio_smpl24, tmp_path, missing):
        root = _b3d_tree(tmp_path / "addbiomechanics")
        with pytest.raises(ValueError, match=missing[1]):
            addbio_smpl24.payload_files(root, None, [missing])

    def test_main_refuses_a_missing_subject_before_converting(self, addbio_smpl24, data_root,
                                                              tmp_path, capsys):
        root = _b3d_tree(tmp_path / "addbiomechanics")
        code = addbio_smpl24.main(["--source-root", str(root), "--out", str(tmp_path / "out"),
                                   "--only", "Hamner2013_Formatted_No_Arm/subject99"])
        assert code == 2
        assert "subject99" in capsys.readouterr().out
        assert not (tmp_path / "out").exists()

    def test_the_default_source_comes_from_the_environment(self, addbio_smpl24, data_root, tmp_path,
                                                           capsys):
        code = addbio_smpl24.main(["--out", str(tmp_path / "out")])
        assert code == 2      # the tmp data root has no addbiomechanics folder, and says so
        assert str(data_root / "extracted" / "addbiomechanics") in capsys.readouterr().out


# ================================================================ HKNU --hknu-root


class TestHknuRoot:
    def test_the_workbook_is_read_from_the_folder_given(self, clean_env, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        generator = _load("flags_hknu_unified8", "generate_hknu_unified8.py")
        root = tmp_path / "hknu_copy"
        (root / "MATLAB").mkdir(parents=True)
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.title = "Subject"
        sheet.append(["SubjectID", "Gender", "BodyMass", "BodyHeight"])
        sheet.append(["S04", "M", 71.5, 1.76])
        sheet.append([None, None, None, None])
        book.save(root / "MATLAB" / "DatasetInfo.xlsx")
        assert generator.subject_anthropometry(root) == {
            "S04": {"gender": "male", "mass_kg": 71.5, "height_m": 1.76}}

    def test_the_default_is_the_hknu_source_folder(self, data_root):
        generator = _load("flags_hknu_unified8", "generate_hknu_unified8.py")
        with pytest.raises(FileNotFoundError, match="hknu_fullbody"):
            generator.subject_anthropometry()


# ================================================================ GAITEX logical input ids


@pytest.fixture(scope="module")
def gaitex_unified8():
    return _load("flags_gaitex_unified8", "generate_gaitex_unified8.py")


class TestGaitexInputIds:
    def test_a_relocated_extracted_folder_keeps_the_logical_id(self, gaitex_unified8, clean_env,
                                                                tmp_path):
        generator = gaitex_unified8
        relocated = tmp_path / "copies" / "gaitex_elsewhere"
        csv = relocated / "austra" / "gwo" / "qualisys_marker_data_austra_gwo.csv"
        csv.parent.mkdir(parents=True)
        csv.write_bytes(b"frame,x\n0,1\n")
        record = generator._input_record(csv, 1, relocated)
        assert record == {
            "relative_path": "extracted/gaitex/austra/gwo/qualisys_marker_data_austra_gwo.csv",
            "sha256": hashlib.sha256(b"frame,x\n0,1\n").hexdigest(),
            "frames": 1,
        }

    def test_the_default_folder_gives_the_string_relative_to_gave(self, gaitex_unified8, data_root):
        csv = data_root / "extracted" / "gaitex" / "austra" / "ng" / "imu.csv"
        csv.parent.mkdir(parents=True)
        csv.write_bytes(b"t\n")
        assert (gaitex_unified8._input_record(csv, 3)["relative_path"]
                == csv.relative_to(data_root).as_posix()
                == "extracted/gaitex/austra/ng/imu.csv")

    def test_the_retarget_take_is_named_against_the_data_root(self, gaitex_unified8, data_root):
        take = data_root / "tmp" / "harness" / "h1" / "gaitex_smpl24" / "austra" / "gwo.npz"
        take.parent.mkdir(parents=True)
        take.write_bytes(b"npz")
        settings = SimpleNamespace(filter_order=4, cutoff_hz=6.0)
        identity = gaitex_unified8.identity_for(take, "austra", "gwo", 100, 100.0, settings, {}, "0" * 64)
        assert identity.relative_path == "tmp/harness/h1/gaitex_smpl24/austra/gwo.npz"
        assert identity.relative_path == take.relative_to(data_root).as_posix()

    def test_unified8_refuses_a_corpus_outside_the_data_root_before_writing(
            self, gaitex_unified8, data_root, tmp_path, capsys):
        """Every take's relative_path is written against the data root: a corpus elsewhere failed
        take by take and still left an INDEX, a fit record and a description behind."""
        retarget = tmp_path / "corpus_elsewhere"
        retarget.mkdir()
        out = data_root / "tmp" / "out"
        code = gaitex_unified8.main(["--out", str(out), "--retarget", str(retarget),
                                     "--extracted", str(tmp_path / "gaitex_copy")])
        assert code == 2
        assert "not inside the data root" in capsys.readouterr().out
        assert not out.exists()


# ================================================================ every main calls the guard
#
# One test per generator main the flag tests above do not already cover: an output the guard
# refuses must stop the run with exit 2 before anything is created. Without the check_output_dir
# call each of these would create the directory and then fail on the missing inputs.


class TestEveryMainCallsTheOutputGuard:
    def test_hknu_faithful(self, data_root, tmp_path, capsys):
        """Its own older prefix check refuses --out inside --root first (exit 1), so the guard is
        shown with an output only it refuses: the lineage container itself."""
        module = _load("flags_hknu_faithful", "generate_hknu_faithful.py")
        root = tmp_path / "hknu_copy"
        root.mkdir()
        poc = data_root / "runs" / "experimental_generation_poc_demo"
        assert module.main(["--root", str(root), "--out", str(poc)]) == 2
        assert "lineage container" in capsys.readouterr().out
        assert not poc.exists()

    def test_hknu_unified8(self, data_root, tmp_path, capsys):
        module = _load("flags_hknu_unified8", "generate_hknu_unified8.py")
        paired = data_root / "tmp" / "hknu_smpl24_paired"
        paired.mkdir(parents=True)
        root = tmp_path / "hknu_copy"
        root.mkdir()
        out = paired / "bundle"
        assert module.main(["--out", str(out), "--paired", str(paired),
                            "--hknu-root", str(root)]) == 2
        assert "a folder this run reads" in capsys.readouterr().out
        assert not out.exists()

    def test_gaitex_smpl24(self, data_root, tmp_path, capsys):
        module = _load("flags_gaitex_smpl24", "generate_gaitex_smpl24.py")
        source = tmp_path / "gaitex_copy"
        source.mkdir()
        out = source / "corpus"
        assert module.main(["--source-root", str(source), "--out", str(out)]) == 2
        assert "a folder this run reads" in capsys.readouterr().out
        assert not out.exists()

    def test_gaitex_unified8(self, gaitex_unified8, data_root, tmp_path, capsys):
        retarget = data_root / "tmp" / "gaitex_smpl24"
        retarget.mkdir(parents=True)
        out = retarget / "bundle"
        assert gaitex_unified8.main(["--out", str(out), "--retarget", str(retarget),
                                     "--extracted", str(tmp_path / "gaitex_copy")]) == 2
        assert "a folder this run reads" in capsys.readouterr().out
        assert not out.exists()

    def test_prism_faithful_run_as_a_script(self, data_root, clean_env, capsys):
        """The single-take generator's `__main__` block: SOMA_POC_OUT_DIR inside the source."""
        out = data_root / "extracted" / "prism" / "bundle"
        clean_env.setenv("SOMA_POC_OUT_DIR", str(out))
        with pytest.raises(SystemExit) as stopped:
            runpy.run_path(str(POC / "generate_prism_faithful.py"), run_name="__main__")
        assert stopped.value.code == 2
        assert "source root" in capsys.readouterr().out
        assert not out.exists()


# ================================================================ a partial run needs a scratch output
#
# A selection flag narrows the run to part of the corpus, and the batch mains then rewrite INDEX.json,
# the fit record and the description over that part and exit 0. Into the production lineage (the
# default output) that leaves a bundle describing a sample under the production name, so each main
# requires an output of its own there. Each test puts a stand-in production INDEX.json in the
# lineage first and requires it byte-identical, and alone, afterwards.


class TestAPartialSelectionNeedsAScratchOutput:
    @pytest.mark.parametrize("extra", [["--groups", "CMU/01"], ["--datasets", "CMU"],
                                       ["--limit", "1"], ["--groups", "CMU/01", "--merge"],
                                       ["--datasets", "CMU", "KIT", "--shard", "0/2"]],
                             ids=["groups", "datasets", "limit", "merge", "shard"])
    @pytest.mark.parametrize("spelling", ["omitted", "explicit", "roundabout"])
    def test_amass_a_partial_selection_into_the_lineage(self, amass_all, data_root, tmp_path,
                                                        capsys, extra, spelling):
        root = _amass_tree(tmp_path)
        lineage = _lineage(data_root, "amass_faithful_full")
        index = _bundle_with_index(lineage)
        out = {"omitted": [], "explicit": ["--out-root", str(lineage)],
               "roundabout": ["--out-root", str(lineage / ".." / "amass_faithful_full")]}[spelling]
        assert amass_all.main(["--amass-root", root, *out, *extra]) == 2
        text = " ".join(capsys.readouterr().out.split())
        assert "production lineage directory" in text and "Pass --out-root" in text
        assert str(data_root / "tmp" / "amass_faithful_full") in text     # where to put it instead
        assert _untouched(index)

    def test_amass_a_partial_selection_into_a_scratch_output_passes_the_guard(
            self, amass_all, data_root, tmp_path, capsys):
        """The guard names the lineage only: a scratch --out-root reaches the selection, here an
        unknown group, which is refused on its own account."""
        root = _amass_tree(tmp_path)
        index = _bundle_with_index(_lineage(data_root, "amass_faithful_full"))
        scratch = data_root / "tmp" / "amass_sample"
        assert amass_all.main(["--amass-root", root, "--out-root", str(scratch),
                               "--groups", "CMU/99"]) == 2
        out = capsys.readouterr().out
        assert "CMU/99" in out and "production lineage" not in out
        assert _untouched(index) and not scratch.exists()

    def test_amass_a_full_run_is_not_a_partial_selection(self, amass_all):
        parse = argparse.Namespace
        assert amass_all.partial_selection(parse(datasets=None, groups=None, limit=None)) == []
        assert amass_all.partial_selection(parse(datasets=[], groups=None, limit=None)) == []
        assert amass_all.partial_selection(
            parse(datasets=["CMU", "KIT"], groups=[("CMU", "01")], limit=0)) == [
            "--datasets CMU KIT", "--groups CMU/01", "--limit 0"]

    @pytest.mark.parametrize("script", PRISM_MAINS)
    @pytest.mark.parametrize("extra", [["--only", "prism_subj005"], ["--limit", "1"]],
                             ids=["only", "limit"])
    def test_prism_a_partial_selection_into_the_lineage(self, script, data_root, capsys, extra):
        module = _load(f"flags_{script[:-3]}", script)
        _prism_tree(data_root / "extracted" / "prism")
        index = _bundle_with_index(_lineage(data_root, "prism_faithful_full"))
        assert module.main(extra) == 2
        text = " ".join(capsys.readouterr().out.split())
        assert "production lineage directory" in text
        flag = "--data-root" if script == "generate_prism_faithful_all.py" else "--out"
        assert f"Pass {flag} a scratch location" in text
        assert _untouched(index)

    @pytest.mark.parametrize("script", PRISM_MAINS)
    def test_prism_a_data_root_that_is_the_data_root_is_the_lineage(self, script, data_root,
                                                                    capsys):
        module = _load(f"flags_{script[:-3]}", script)
        _prism_tree(data_root / "extracted" / "prism")
        index = _bundle_with_index(_lineage(data_root, "prism_faithful_full"))
        assert module.main(["--data-root", str(data_root), "--only", "prism_subj005"]) == 2
        assert "Pass --data-root a scratch location" in " ".join(capsys.readouterr().out.split())
        assert _untouched(index)

    def test_prism_measured_an_explicit_out_that_is_the_lineage(self, prism_all, data_root,
                                                               capsys):
        _prism_tree(data_root / "extracted" / "prism")
        lineage = _lineage(data_root, "prism_faithful_full")
        index = _bundle_with_index(lineage)
        assert prism_all.main(["--out", str(lineage), "--limit", "2"]) == 2
        assert "production lineage directory" in capsys.readouterr().out
        assert _untouched(index)

    def test_prism_measured_a_scratch_out_passes_the_guard(self, prism_all, data_root, clean_env,
                                                           tmp_path, capsys):
        _prism_tree(data_root / "extracted" / "prism")
        clean_env.setattr(prism_all.faithful_all, "_gender_by_subject",
                          lambda *a, **k: {"prism_subj001": "M", "prism_subj005": "F"})
        index = _bundle_with_index(_lineage(data_root, "prism_faithful_full"))
        scratch = tmp_path / "scratch_bundle"
        assert prism_all.main(["--out", str(scratch), "--only", "prism_subj99"]) == 2
        out = capsys.readouterr().out
        assert "prism_subj99" in out and "production lineage" not in out
        assert _untouched(index) and not scratch.exists()


# ================================================================ AMASS --merge writes nothing from nothing


class TestAmassMergeOfNothing:
    def test_no_shard_index_in_the_default_lineage(self, amass_all, data_root, tmp_path, capsys):
        """The production lineage holds an INDEX.json and no shard: the merge used to replace it
        with an INDEX over zero takes and rewrite the description."""
        root = _amass_tree(tmp_path)
        index = _bundle_with_index(_lineage(data_root, "amass_faithful_full"))
        assert amass_all.main(["--amass-root", root, "--merge"]) == 2
        assert "no shard index" in capsys.readouterr().out
        assert _untouched(index)

    def test_no_shard_index_in_an_explicit_output(self, amass_all, data_root, tmp_path, capsys):
        root = _amass_tree(tmp_path)
        index = _bundle_with_index(tmp_path / "bundle")
        assert amass_all.main(["--amass-root", root, "--out-root", str(index.parent),
                               "--merge"]) == 2
        assert "no shard index" in capsys.readouterr().out
        assert _untouched(index)

    def test_a_missing_output_is_not_created(self, amass_all, data_root, tmp_path, capsys):
        root = _amass_tree(tmp_path)
        out = tmp_path / "never_sharded"
        assert amass_all.main(["--amass-root", root, "--out-root", str(out), "--merge"]) == 2
        assert "no shard index" in capsys.readouterr().out
        assert not out.exists()

    def test_shards_that_list_no_take(self, amass_all, data_root, tmp_path, capsys):
        root = _amass_tree(tmp_path)
        index = _bundle_with_index(tmp_path / "bundle")
        shards = []
        for i in range(2):
            shard = index.parent / f"INDEX_shard_{i}_of_2.json"
            shard.write_text('{"takes": []}\n', encoding="utf-8")
            shards.append(shard.name)
        assert amass_all.main(["--amass-root", root, "--out-root", str(index.parent),
                               "--merge"]) == 2
        assert "list no take" in capsys.readouterr().out
        assert index.read_bytes() == PRODUCTION_INDEX
        assert sorted(p.name for p in index.parent.iterdir()) == sorted(["INDEX.json", *shards])

    def test_merge_shards_refuses_before_writing(self, amass_all, tmp_path):
        out = tmp_path / "bundle"
        out.mkdir()
        with pytest.raises(ValueError, match="no shard index"):
            amass_all.merge_shards(str(out), total_specs=3)
        (out / "INDEX_shard_0_of_1.json").write_text('{"takes": []}', encoding="utf-8")
        with pytest.raises(ValueError, match="list no take"):
            amass_all.merge_shards(str(out), total_specs=3)
        assert sorted(p.name for p in out.iterdir()) == ["INDEX_shard_0_of_1.json"]


# ================================================================ GAITEX / HKNU unified8: --subjects


def _gaitex_retarget(folder: Path, takes=("austra/gwo.npz", "austra/ng.npz", "carla/gwo.npz")) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for take in takes:
        (folder / take).parent.mkdir(parents=True, exist_ok=True)
        (folder / take).write_bytes(b"npz")
    return folder


def _hknu_paired(folder: Path, takes=("S04/S04_walk01.npz", "S04/S04_walk02.npz",
                                      "S10/S10_walk01.npz")) -> Path:
    return _gaitex_retarget(folder, takes)


class TestUnified8Subjects:
    """Each case leaves an INDEX.json in the output first, as a resumed or production bundle has,
    and requires it byte-identical and alone afterwards; nothing past the refusal is read."""

    @pytest.fixture
    def gaitex(self, gaitex_unified8, data_root, tmp_path):
        retarget = _gaitex_retarget(data_root / "tmp" / "gaitex_smpl24")
        extracted = tmp_path / "gaitex_copy"
        extracted.mkdir()
        index = _bundle_with_index(tmp_path / "out")

        def run(*extra, retarget=retarget, extracted=extracted):
            return gaitex_unified8.main(["--out", str(index.parent), "--retarget", str(retarget),
                                         "--extracted", str(extracted), *extra])
        return SimpleNamespace(run=run, index=index, retarget=retarget, extracted=extracted)

    @pytest.fixture
    def hknu(self, data_root, tmp_path):
        module = _load("flags_hknu_unified8", "generate_hknu_unified8.py")
        paired = _hknu_paired(data_root / "tmp" / "hknu_smpl24_paired")
        hknu_root = tmp_path / "hknu_copy"
        hknu_root.mkdir()
        index = _bundle_with_index(tmp_path / "out")

        def run(*extra, paired=paired, hknu_root=hknu_root):
            return module.main(["--out", str(index.parent), "--paired", str(paired),
                                "--hknu-root", str(hknu_root), *extra])
        return SimpleNamespace(run=run, index=index, paired=paired, module=module)

    @pytest.mark.parametrize("name", ["Austra", "AUSTRA", "nobody"])
    def test_gaitex_an_unknown_or_differently_cased_subject(self, gaitex, capsys, name):
        assert gaitex.run("--subjects", "austra", name) == 2
        out = capsys.readouterr().out
        assert f"--subjects {name}:" in out and "case-sensitive" in out and "austra, carla" in out
        assert _untouched(gaitex.index)

    def test_gaitex_a_selection_of_no_take(self, gaitex, capsys):
        (gaitex.retarget / "dora").mkdir()                    # a subject folder with no take
        assert gaitex.run("--subjects", "dora") == 2
        assert "no take" in capsys.readouterr().out
        assert _untouched(gaitex.index)

    def test_gaitex_a_corpus_of_no_subject(self, gaitex, data_root, capsys):
        empty = data_root / "tmp" / "gaitex_empty"
        (empty / "_logs").mkdir(parents=True)
        (empty / "_run.json").write_text("{}", encoding="utf-8")
        assert gaitex.run(retarget=empty) == 2
        assert "no take" in capsys.readouterr().out
        assert _untouched(gaitex.index)

    def test_gaitex_a_missing_retarget_corpus(self, gaitex, data_root, capsys):
        assert gaitex.run(retarget=data_root / "tmp" / "absent") == 2
        out = capsys.readouterr().out
        assert "retarget corpus" in out and "does not exist" in out and "as given" in out
        assert _untouched(gaitex.index)

    def test_gaitex_the_default_retarget_corpus_is_named(self, gaitex_unified8, data_root,
                                                         tmp_path, capsys):
        extracted = tmp_path / "gaitex_copy"
        extracted.mkdir()
        index = _bundle_with_index(tmp_path / "out")
        assert gaitex_unified8.main(["--out", str(index.parent), "--extracted", str(extracted)]) == 2
        assert "SOMA_DATA_ROOT/runs/experimental_generation_poc_demo/gaitex_smpl24" in (
            capsys.readouterr().out)
        assert _untouched(index)

    @pytest.mark.parametrize("given", [True, False], ids=["flag", "default"])
    def test_gaitex_a_missing_source_folder(self, gaitex_unified8, data_root, tmp_path, capsys,
                                            given):
        retarget = _gaitex_retarget(data_root / "tmp" / "gaitex_smpl24")
        index = _bundle_with_index(tmp_path / "out")
        flags = ["--extracted", str(tmp_path / "absent")] if given else []
        assert gaitex_unified8.main(["--out", str(index.parent), "--retarget", str(retarget),
                                     *flags]) == 2
        out = capsys.readouterr().out
        assert "gaitex source folder" in out and "does not exist" in out
        assert ("as given" if given else "SOMA_DATA_ROOT/extracted/gaitex") in out
        assert _untouched(index)

    def test_gaitex_select_subjects_keeps_the_old_walk(self, gaitex_unified8, tmp_path):
        retarget = _gaitex_retarget(tmp_path / "corpus")
        (retarget / "_run.json").write_text("{}", encoding="utf-8")
        assert gaitex_unified8.select_subjects(retarget) == ["austra", "carla"]
        assert gaitex_unified8.select_subjects(retarget, []) == ["austra", "carla"]
        assert gaitex_unified8.select_subjects(retarget, ["carla", "austra"]) == ["austra", "carla"]

    @pytest.mark.parametrize("name", ["s04", "S4", "nobody"])
    def test_hknu_an_unknown_or_differently_cased_subject(self, hknu, capsys, name):
        assert hknu.run("--subjects", "S04", name) == 2
        out = capsys.readouterr().out
        assert f"--subjects {name}:" in out and "case-sensitive" in out and "S04, S10" in out
        assert _untouched(hknu.index)

    @pytest.mark.parametrize("extra, caps", [(["--subjects", "S99x"], None),
                                             (["--trials", "-2", "--subjects", "S10"],
                                              "--subjects S10 --trials -2")])
    def test_hknu_a_selection_of_no_take(self, hknu, capsys, extra, caps):
        (hknu.paired / "S99x").mkdir()                        # a subject folder with no take
        assert hknu.run(*extra) == 2
        out = capsys.readouterr().out
        assert "no take" in out and (caps is None or caps in out)
        assert _untouched(hknu.index)

    def test_hknu_a_missing_paired_corpus(self, hknu, tmp_path, capsys):
        assert hknu.run(paired=tmp_path / "absent") == 2
        out = capsys.readouterr().out
        assert "paired corpus" in out and "does not exist" in out and "as given" in out
        assert _untouched(hknu.index)

    @pytest.mark.parametrize("given", [True, False], ids=["flag", "default"])
    def test_hknu_a_missing_source_folder(self, data_root, tmp_path, capsys, given):
        module = _load("flags_hknu_unified8", "generate_hknu_unified8.py")
        paired = _hknu_paired(data_root / "tmp" / "hknu_smpl24_paired")
        index = _bundle_with_index(tmp_path / "out")
        flags = ["--hknu-root", str(tmp_path / "absent")] if given else []
        assert module.main(["--out", str(index.parent), "--paired", str(paired), *flags]) == 2
        out = capsys.readouterr().out
        if given:
            assert "hknu source folder" in out and "as given" in out and "does not exist" in out
        else:     # the default source root has no hknu_fullbody folder
            assert "hknu_fullbody" in out and "does not exist" in out
        assert _untouched(index)

    def test_hknu_select_subjects_keeps_the_old_walk(self, hknu):
        assert hknu.module.select_subjects(hknu.paired) == ["S04", "S10"]
        assert hknu.module.select_subjects(hknu.paired, ["S10"]) == ["S10"]
        assert hknu.module.select_subjects(hknu.paired, None, 1) == ["S04", "S10"]


# ================================================================ AddBio: a selection of no trial


def _retarget_corpus(root: Path, trials=True) -> Path:
    _corpus(root)
    if trials:
        for trial in ("Hamner2013_Formatted_No_Arm/subject01/trial_0000.npz",
                      "Tiziana2019_Formatted_With_Arm/Subject43/trial_0000.npz"):
            (root / trial).write_bytes(b"npz")
    return root


class TestAddbioSelectionOfNothing:
    @pytest.mark.parametrize("generator", ["addbio_faithful", "addbio_unified8"])
    @pytest.mark.parametrize("extra, message", [
        ([], "no retarget trial"),
        (["--only", "Zeta2020_Formatted_No_Arm/s1"], "--only Zeta2020_Formatted_No_Arm/s1"),
        (["--studies", "1", "--subjects", "1", "--trials", "-1"],
         "--studies 1 --subjects 1 --trials -1"),
    ], ids=["corpus_of_no_trial", "subject_of_no_trial", "caps_cut_every_trial"])
    def test_synth_mains(self, request, generator, data_root, tmp_path, capsys, extra, message):
        module = request.getfixturevalue(generator)
        corpus = _retarget_corpus(data_root / "tmp" / "corpus", trials=bool(extra))
        index = _bundle_with_index(tmp_path / "out")
        assert module.main(["--out", str(index.parent), "--raw", str(corpus), *extra]) == 2
        out = " ".join(capsys.readouterr().out.split())
        assert message in out and str(corpus) in out and "nothing written" in out
        assert _untouched(index)

    @pytest.mark.parametrize("generator", ["addbio_faithful", "addbio_unified8"])
    def test_synth_mains_a_corpus_of_no_subject(self, request, generator, data_root, tmp_path,
                                                capsys):
        module = request.getfixturevalue(generator)
        corpus = data_root / "tmp" / "corpus"
        corpus.mkdir(parents=True)
        out = tmp_path / "out"
        assert module.main(["--out", str(out), "--raw", str(corpus)]) == 2
        assert "no subject selected" in capsys.readouterr().out
        assert not out.exists()

    def test_require_trials_passes_a_selection_with_a_trial(self, addbio_faithful, tmp_path):
        corpus = _retarget_corpus(tmp_path / "corpus")
        pairs = [("Hamner2013_Formatted_No_Arm", "subject02"),
                 ("Tiziana2019_Formatted_With_Arm", "Subject43")]
        assert addbio_faithful.require_trials(corpus, pairs) is None
        assert addbio_faithful.require_trials(corpus, pairs, trials=1) is None
        with pytest.raises(ValueError, match="no retarget trial"):
            addbio_faithful.require_trials(corpus, pairs[:1])

    @pytest.mark.parametrize("extra, caps", [([], ""), (["--per-study", "1"], "--per-study 1")])
    def test_smpl24_a_source_of_no_payload(self, addbio_smpl24, data_root, tmp_path, capsys,
                                           extra, caps):
        root = tmp_path / "addbiomechanics"
        folder = root / "train" / "No_Arm" / "Hamner2013_Formatted_No_Arm" / "subject03"
        folder.mkdir(parents=True)
        (folder / "subject03.b3d").write_bytes(b"")          # ADR-0030 empty: not a payload
        index = _bundle_with_index(tmp_path / "out")
        assert addbio_smpl24.main(["--source-root", str(root), "--out", str(index.parent),
                                   *extra]) == 2
        out = capsys.readouterr().out
        assert "no payload" in out and str(root) in out and caps in out
        assert _untouched(index)


# ================================================================ GAITEX smpl24 / HKNU faithful


def _gaitex_source(root: Path) -> Path:
    for subject in ("austra", "carla"):
        (root / subject / "gwo").mkdir(parents=True)
    (root / "_docs").mkdir()
    return root


@pytest.fixture(scope="module")
def gaitex_smpl24():
    return _load("flags_gaitex_smpl24", "generate_gaitex_smpl24.py")


class TestCorpusGeneratorSubjects:
    @pytest.mark.parametrize("name", ["Austra", "nobody", "_docs"])
    def test_gaitex_an_unknown_or_differently_cased_subject(self, gaitex_smpl24, data_root,
                                                            tmp_path, capsys, name):
        source = _gaitex_source(tmp_path / "gaitex_copy")
        out = tmp_path / "corpus"
        assert gaitex_smpl24.main(["--source-root", str(source), "--out", str(out),
                                   "--subjects", "austra", name]) == 2
        text = capsys.readouterr().out
        assert f"--subjects {name}:" in text and "austra, carla" in text
        assert not out.exists()

    def test_gaitex_a_source_of_no_subject(self, gaitex_smpl24, data_root, tmp_path, capsys):
        source = tmp_path / "gaitex_copy"
        (source / "austra").mkdir(parents=True)               # no condition folder: not a subject
        index = _bundle_with_index(tmp_path / "corpus")
        assert gaitex_smpl24.main(["--source-root", str(source), "--out", str(index.parent)]) == 2
        assert "no subject" in capsys.readouterr().out
        assert _untouched(index)

    @pytest.mark.parametrize("given", [True, False], ids=["flag", "default"])
    def test_gaitex_a_missing_source_folder(self, gaitex_smpl24, data_root, tmp_path, capsys,
                                            given):
        flags = ["--source-root", str(tmp_path / "absent")] if given else []
        out = tmp_path / "corpus"
        assert gaitex_smpl24.main([*flags, "--out", str(out)]) == 2
        text = capsys.readouterr().out
        assert "does not exist" in text
        assert ("as given" if given else "SOMA_DATA_ROOT/extracted/gaitex") in text
        assert not out.exists()

    def test_gaitex_select_subjects_keeps_the_old_order(self, gaitex_smpl24, tmp_path):
        source = _gaitex_source(tmp_path / "gaitex_copy")
        assert gaitex_smpl24.select_subjects(source) == ["austra", "carla"]
        # as given, in the given order, as `arguments.subjects or discover_subjects(...)` was
        assert gaitex_smpl24.select_subjects(source, ["carla", "austra"]) == ["carla", "austra"]

    @pytest.fixture
    def hknu_root(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        root = tmp_path / "hknu_copy"
        (root / "MATLAB").mkdir(parents=True)
        for subject in ("S04", "S10"):
            (root / "Dataset_Processed" / subject).mkdir(parents=True)
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.title = "Subject"
        sheet.append(["SubjectID", "Gender", "BodyMass", "BodyHeight"])
        for subject in ("S04", "S10", "S11"):                 # S11 has no Dataset_Processed folder
            sheet.append([subject, "M", 70.0, 1.75])
        book.save(root / "MATLAB" / "DatasetInfo.xlsx")
        return root

    @pytest.mark.parametrize("name", ["s04", "S11", "S99"])
    def test_hknu_faithful_an_unknown_subject(self, hknu_root, data_root, tmp_path, capsys, name):
        module = _load("flags_hknu_faithful", "generate_hknu_faithful.py")
        out = tmp_path / "corpus"
        assert module.main(["--root", str(hknu_root), "--out", str(out),
                            "--subjects", "S04", name]) == 2
        text = capsys.readouterr().out
        assert f"--subjects {name}:" in text and "S04, S10" in text
        assert not out.exists()

    def test_hknu_faithful_a_workbook_of_no_subject(self, hknu_root, data_root, tmp_path, capsys):
        openpyxl = pytest.importorskip("openpyxl")
        book = openpyxl.Workbook()
        book.active.title = "Subject"
        book.active.append(["SubjectID", "Gender", "BodyMass", "BodyHeight"])
        book.save(hknu_root / "MATLAB" / "DatasetInfo.xlsx")
        index = _bundle_with_index(tmp_path / "corpus")
        module = _load("flags_hknu_faithful", "generate_hknu_faithful.py")
        assert module.main(["--root", str(hknu_root), "--out", str(index.parent)]) == 2
        assert "no subject" in capsys.readouterr().out
        assert _untouched(index)

    def test_hknu_faithful_a_missing_source_folder(self, data_root, tmp_path, capsys):
        module = _load("flags_hknu_faithful", "generate_hknu_faithful.py")
        out = tmp_path / "corpus"
        assert module.main(["--root", str(tmp_path / "absent"), "--out", str(out)]) == 2
        text = capsys.readouterr().out
        assert "does not exist" in text and "as given" in text
        assert not out.exists()

    def test_hknu_faithful_select_subjects_keeps_the_old_choice(self, hknu_root):
        module = _load("flags_hknu_faithful", "generate_hknu_faithful.py")
        demographics = {"S04": {}, "S10": {}, "S11": {}}
        assert module.select_subjects(hknu_root, demographics) == ["S04", "S10", "S11"]
        assert module.select_subjects(hknu_root, demographics, ["S10", "S04"]) == ["S10", "S04"]
