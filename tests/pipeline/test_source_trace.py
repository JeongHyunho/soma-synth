"""The run record's ``source_files``: the full SHA-256 of every source file a corpus source's
stages could read (closing the gap GENERATION_PIPELINE_STANDARD §4.1 recorded).

The corpora name their sources weakly -- HKNU by subject and trial, GAITEX and AddBiomechanics
by a first-MiB digest -- and no generator output may change, so the runner hashes the source folder
(narrowed to the selected subjects) into the run record instead. These tests build scratch source
trees; nothing reads the data plane.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from soma_synth.pipeline import source_trace as trace_mod

REPO = Path(__file__).resolve().parents[2]


def _write(root: Path, files: dict[str, bytes]) -> None:
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


HKNU = {
    "AUTHORS_README.txt": b"authors",
    "MATLAB/DatasetInfo.xlsx": b"workbook",
    "Dataset_Processed/S01/S01_Npose.mat": b"s01-npose",
    "Dataset_Processed/S01/S01_Walk.mat": b"s01-walk" * 300_000,       # > 1 MiB: a whole-file hash
    "Dataset_Processed/S02/S02_Npose.mat": b"s02-npose",
    "Dataset_Raw/C3D/S02/S02_Walk.c3d": b"s02-c3d",
    "Dataset_Raw/C3D/S01/S01_Walk.c3d": b"s01-c3d",
}
GAITEX = {
    "README.md": b"readme",
    "_provenance/acquisition_record.json": b"{}",
    "austra/gwo/markers.csv": b"austra markers",
    "austra/gwo/imu.csv": b"austra imu",
    "neves/gwo/markers.csv": b"neves markers",
}
ADDBIO = {
    "train/No_Arm/StudyA/s01/s01.b3d": b"a-s01",
    "train/No_Arm/StudyA/s02/s02.b3d": b"a-s02",
    "train/No_Arm/StudyA/s03/s03.b3d": b"",                       # empty: not a payload
    "train/No_Arm/StudyB/t01/t01.b3d": b"b-t01",
    "train/With_Arm/StudyA/s01/s01.b3d": b"wa-s01",
    "test/No_Arm/StudyA/s09/s09.b3d": b"test-s09",
}


@pytest.fixture
def source_root(tmp_path):
    root = tmp_path / "extracted"
    _write(root / "hknu_fullbody", HKNU)
    _write(root / "gaitex", GAITEX)
    _write(root / "addbiomechanics", ADDBIO)
    return root


def _files(block) -> dict[str, dict]:
    return {entry["relative_path"]: entry for entry in block["files"]}


def _quiet(_message):
    pass


def test_a_full_run_hashes_every_file_of_the_folder_whole(source_root):
    block = trace_mod.trace("hknu", source_root, corpus_built=True, log=_quiet)
    files = _files(block)
    assert set(files) == {f"extracted/hknu_fullbody/{name}" for name in HKNU}
    for name, data in HKNU.items():
        entry = files[f"extracted/hknu_fullbody/{name}"]
        assert entry["sha256"] == _sha(data) and entry["bytes"] == len(data)
        assert entry["stages"] == ["bundle", "corpus"]
    big = files["extracted/hknu_fullbody/Dataset_Processed/S01/S01_Walk.mat"]
    assert big["sha256"] != _sha(HKNU["Dataset_Processed/S01/S01_Walk.mat"][: 1 << 20])
    assert block["count"] == len(HKNU) and block["bytes"] == sum(map(len, HKNU.values()))
    lines = "\n".join(sorted(f"{p}\t{e['sha256']}" for p, e in files.items()))
    assert block["aggregate_sha256"] == hashlib.sha256(lines.encode("utf-8")).hexdigest()
    assert block["schema"] == trace_mod.SCHEMA and block["algorithm"].startswith("sha256 of the whole")
    assert block["folder"] == "extracted/hknu_fullbody"


def test_hknu_is_narrowed_to_the_selected_subjects_per_stage(source_root):
    block = trace_mod.trace("hknu", source_root, corpus_built=True,
                            corpus_args=["--subjects", "S01"], bundle_args=["--subjects=S01"],
                            corpus_declared=("--root", "--out", "--subjects", "--trials"),
                            bundle_declared=("--out", "--paired", "--hknu-root", "--subjects",
                                             "--trials"), log=_quiet)
    names = set(_files(block))
    assert "extracted/hknu_fullbody/Dataset_Processed/S02/S02_Npose.mat" not in names
    assert "extracted/hknu_fullbody/Dataset_Raw/C3D/S02/S02_Walk.c3d" not in names
    assert {"extracted/hknu_fullbody/MATLAB/DatasetInfo.xlsx",          # nobody's: kept
            "extracted/hknu_fullbody/AUTHORS_README.txt",
            "extracted/hknu_fullbody/Dataset_Processed/S01/S01_Npose.mat",
            "extracted/hknu_fullbody/Dataset_Raw/C3D/S01/S01_Walk.c3d"} <= names
    stages = {s["stage"]: s for s in block["stages"]}
    assert stages["corpus"]["selection"] == {"--subjects": ["S01"]} and stages["corpus"]["narrowed"]
    # the bundle stage without a selection could read every subject: S02 comes back for it only
    wide = trace_mod.trace("hknu", source_root, corpus_built=True,
                           corpus_args=["--subj", "S01"], log=_quiet,
                           corpus_declared=("--root", "--out", "--subjects", "--trials"))
    s02 = _files(wide)["extracted/hknu_fullbody/Dataset_Processed/S02/S02_Npose.mat"]
    assert s02["stages"] == ["bundle"]


def test_a_reused_corpus_leaves_the_bundle_stage_alone(source_root):
    block = trace_mod.trace("gaitex", source_root, corpus_built=False,
                            bundle_args=["--subjects", "austra"], log=_quiet)
    stages = {s["stage"]: s for s in block["stages"]}
    assert stages["corpus"] == {"stage": "corpus", "reads_source": False, "built_in_run": False}
    assert set(_files(block)) == {"extracted/gaitex/README.md",
                                  "extracted/gaitex/_provenance/acquisition_record.json",
                                  "extracted/gaitex/austra/gwo/markers.csv",
                                  "extracted/gaitex/austra/gwo/imu.csv"}
    assert all(e["stages"] == ["bundle"] for e in block["files"])


def test_the_addbio_bundle_reads_no_source_file(source_root):
    block = trace_mod.trace("addbiomechanics", source_root, corpus_built=False, log=_quiet)
    assert block["count"] == 0 and block["files"] == []
    assert [s["reads_source"] for s in block["stages"]] == [False, False]


def test_addbio_only_narrows_to_study_and_subject(source_root):
    block = trace_mod.trace("addbiomechanics", source_root, corpus_built=True,
                            corpus_args=["--only", "StudyA/s01", "--only=StudyB/t01"], log=_quiet)
    assert set(_files(block)) == {
        "extracted/addbiomechanics/train/No_Arm/StudyA/s01/s01.b3d",
        "extracted/addbiomechanics/train/With_Arm/StudyA/s01/s01.b3d",
        "extracted/addbiomechanics/train/No_Arm/StudyB/t01/t01.b3d"}


def _load_generator():
    path = REPO / "scripts" / "poc" / "generate_addbio_smpl24.py"
    spec = importlib.util.spec_from_file_location("trace_addbio_smpl24_payload", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.mark.parametrize("per_study, only", [(1, None), (2, None), (1, [("StudyA", "s02")]),
                                            (5, [("StudyA", "s01"), ("StudyB", "t01")]),
                                            (0, None), (-1, None), (0, [("StudyA", "s02")])])
def test_addbio_per_study_picks_what_the_corpus_generator_picks(source_root, per_study, only):
    """--per-study N is the first N non-empty .b3d of each study in the generator's order (and the
    first one for N <= 0: the generator takes a file before it compares); the trace uses the
    generator's own selection function as the reference."""
    generator = _load_generator()
    folder = source_root / "addbiomechanics"
    expected = {f"extracted/addbiomechanics/{path.relative_to(folder).as_posix()}"
                for *_, path in generator.payload_files(folder, per_study, only)}
    args = ["--per-study", str(per_study)]
    for study, subject in only or ():
        args += ["--only", f"{study}/{subject}"]
    block = trace_mod.trace("addbiomechanics", source_root, corpus_built=True, corpus_args=args,
                            log=_quiet)
    assert set(_files(block)) == expected
    assert {s["stage"]: s for s in block["stages"]}["corpus"]["narrowed"] is True


def test_metadata_files_are_not_source_files(source_root):
    (source_root / "gaitex" / "austra" / ".DS_Store").write_bytes(b"finder")
    # nor is a copy an interrupted stage-sources left under its temporary name
    (source_root / "gaitex" / "austra" / "gwo" / ".imu.csv.q1w2e3r4.staging").write_bytes(b"part")
    block = trace_mod.trace("gaitex", source_root, corpus_built=True, log=_quiet)
    assert not any(".DS_Store" in name or "staging" in name for name in _files(block))
    assert len(block["files"]) == len(GAITEX)


def test_a_missing_folder_or_a_source_without_a_corpus_is_said_so(tmp_path):
    missing = trace_mod.trace("hknu", tmp_path / "nowhere", corpus_built=True, log=_quiet)
    assert missing["status"] == "unavailable" and "hknu_fullbody" in missing["reason"]
    for source in ("amass", "prism"):
        block = trace_mod.trace(source, tmp_path, corpus_built=False, log=_quiet)
        assert block["status"] == "not_recorded" and "take manifests" in block["reason"]


@pytest.mark.parametrize("args, flag, kind, values", [
    (["--subjects", "a", "b", "--trials", "2"], "--subjects", "star", ["a", "b"]),
    (["--subjects", "a", "--subjects", "b"], "--subjects", "star", ["b"]),
    (["--subjects=a"], "--subjects", "star", ["a"]),
    (["--subj", "a"], "--subjects", "star", ["a"]),
    (["--trials", "2"], "--subjects", "star", None),
    (["--only", "s/a", "--only", "s/b"], "--only", "append", ["s/a", "s/b"]),
    (["--s", "a"], "--subjects", "star", None),                 # ambiguous: --subjects/--source-root
    (["--per-study", "-1"], "--per-study", "single", ["-1"]),    # a negative number is a value
    (["--per-study", "2", "--per-study", "0"], "--per-study", "single", ["0"]),
    (["--per-study=3", "--trials", "2"], "--per-study", "single", ["3"]),
    (["--per-study", "--trials", "2"], "--per-study", "single", []),   # no value (argparse refuses)
])
def test_flag_values_read_the_arguments_as_argparse_would(args, flag, kind, values):
    declared = ("--subjects", "--trials", "--only", "--source-root")
    assert trace_mod.flag_values(args, flag, declared, kind) == values
