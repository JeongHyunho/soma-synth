"""``soma-synth hash-sources`` and ``stage-sources`` over scratch trees.

The source folders reach a teammate's PC by copying from a folder that holds them (for example a
read-only shared copy). The list (SHA256SUMS, GNU coreutils' format) is written outside the
sources; staging copies what is missing through a verified temporary file and never deletes or
overwrites anything.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from soma_synth import cli
from soma_synth.pipeline import paths
from soma_synth.pipeline import source_files as sf

ENV = (paths.DATA_ROOT_ENV, paths.SOURCE_ROOT_ENV, paths.BODY_MODEL_DIR_ENV,
       paths.SHARED_DRIVE_NAMES_ENV)


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree(root: Path, files: dict[str, bytes]) -> Path:
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


FILES = {
    "hknu_fullbody/MATLAB/DatasetInfo.xlsx": b"workbook",
    "hknu_fullbody/Dataset_Processed/S01/S01_Npose.mat": b"s01 npose",
    "hknu_fullbody/Dataset_Processed/S02/S02_Npose.mat": b"s02 npose",
    "gaitex/austra/gwo/markers.csv": b"t,x\n0,1\n",
    "gaitex/README.md": b"readme",
    "prism/subj001/take002.pkl": b"\x00\x01pickle",
}


@pytest.fixture
def sources(tmp_path):
    return _tree(tmp_path / "shared_copy" / "sources", FILES)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file()}


# ---------------------------------------------------------------- the list format
def test_the_list_is_gnu_sha256sum_sorted_one_line_per_file(sources, tmp_path, clean_env):
    (sources / "gaitex" / "austra" / ".DS_Store").write_bytes(b"finder")      # not a source file
    (sources / "gaitex" / "Thumbs.db").write_bytes(b"explorer")
    out = tmp_path / "lists" / "SHA256SUMS"
    before = _snapshot(sources)
    result = sf.hash_sources(sources, ["prism", "gaitex", "hknu_fullbody"], out, log=lambda _: None)
    assert _snapshot(sources) == before                                       # read only
    lines = out.read_text(encoding="utf-8").splitlines()
    expected = sorted(f"{_sha(data)}  {name}" for name, data in FILES.items()
                      if name.split("/")[0] in ("prism", "gaitex", "hknu_fullbody"))
    assert lines == sorted(lines, key=lambda line: line.split("  ", 1)[1])
    assert sorted(lines) == expected and len(lines) == 6
    assert out.read_bytes().endswith(b"\n") and b"\r" not in out.read_bytes()
    assert result.files == 6 and result.bytes == sum(len(d) for d in FILES.values())
    assert sf.read_sums(out) == [sf.parse_sums_line(line) for line in lines]


@pytest.mark.parametrize("name", ["a/b c.txt", "a/back\\slash", "a/new\nline", "a/ret\rurn"])
def test_names_round_trip_the_coreutils_escapes(name):
    line = sf.format_sums_line("a" * 64, name)
    assert ("\n" not in line and "\r" not in line)
    assert line.startswith("\\") == any(c in name for c in "\\\n\r")
    assert sf.parse_sums_line(line) == ("a" * 64, name)
    assert sf.parse_sums_line(f"{'A' * 64} *{name}" if "\\" not in line else line)[1] == name


@pytest.mark.parametrize("line", ["", "abc  x", f"{'g' * 64}  x", f"{'a' * 64} x"])
def test_a_line_that_is_not_sha256sum_is_refused(line):
    with pytest.raises(sf.SourceFilesError):
        sf.parse_sums_line(line)


@pytest.mark.parametrize("name", ["/abs/x", "hknu_fullbody", "hknu_fullbody/../prism/x",
                                  "hknu_fullbody/./x", "C:/x", "other/x", "prism//x",
                                  "prism\\x"])
def test_an_unsafe_or_unknown_listed_name_is_refused(name):
    with pytest.raises(sf.SourceFilesError):
        sf.split_listed_name(name)


def test_sources_are_folder_or_source_names(clean_env):
    assert sf.parse_sources_option("hknu, prism") == ["hknu_fullbody", "prism"]
    assert sf.parse_sources_option(None) == sorted(paths.SOURCE_FOLDERS.values())
    with pytest.raises(sf.SourceFilesError, match="unknown source"):
        sf.parse_sources_option("weargait_pd")


# ---------------------------------------------------------------- where the list may go
def test_the_list_is_refused_inside_the_sources_or_the_data_roots_read_only_folders(
        sources, tmp_path, clean_env):
    data = tmp_path / "data"
    (data / "extracted").mkdir(parents=True)
    clean_env.setenv(paths.DATA_ROOT_ENV, str(data))
    clean_env.setenv(paths.SHARED_DRIVE_NAMES_ENV, "Team_Share")
    for out in (sources / "SHA256SUMS", sources / "prism" / "SUMS",
                data / "extracted" / "SUMS", data / "raw_archives" / "SUMS",
                tmp_path / "Dropbox" / "SUMS", tmp_path / "x" / "Team_Share" / "SUMS",
                data / "README.md", data / "MASTER.md",
                data / "state" / "local_archive_inventory.json"):
        with pytest.raises(paths.OutputLocationRefused):
            sf.hash_sources(sources, ["prism"], out, log=lambda _: None)
        assert not out.exists() or out.name in ("README.md", "MASTER.md")
    assert sf.check_sums_output(tmp_path / "lists" / "SUMS", sources) == tmp_path / "lists" / "SUMS"


def test_the_list_is_refused_in_evidence_the_lineage_container_or_a_bundle(sources, tmp_path,
                                                                           clean_env):
    data = tmp_path / "data"
    (data / "extracted").mkdir(parents=True)
    clean_env.setenv(paths.DATA_ROOT_ENV, str(data))
    bundle = tmp_path / "somewhere" / "amass_faithful_full"
    (bundle / "take000").mkdir(parents=True)
    (bundle / "INDEX.json").write_text("{}", encoding="utf-8")
    poc = data / "runs" / "experimental_generation_poc_demo"
    for out in (tmp_path / "x" / "_superseded" / "SUMS", poc / "_runs" / "run_x" / "SUMS",
                tmp_path / "_Manifest_Backfill" / "SUMS",           # matched case-insensitively
                poc / "SUMS", poc / "lists" / "SUMS",
                # another data root's container, known by its folder names
                tmp_path / "other" / "runs" / "experimental_generation_poc_demo" / "SUMS",
                bundle / "SUMS", bundle / "take000" / "SUMS"):
        with pytest.raises(paths.OutputLocationRefused):
            sf.hash_sources(sources, ["prism"], out, log=lambda _: None)
        assert not out.exists()
    # a folder of one's own beside them is fine, and so is the data root's runs folder itself
    for out in (tmp_path / "somewhere" / "SUMS", data / "runs" / "SUMS"):
        assert sf.hash_sources(sources, ["prism"], out, log=lambda _: None).files == 1


def test_an_existing_file_that_is_not_a_list_is_kept_unless_forced(sources, tmp_path, clean_env,
                                                                  capsys):
    out = tmp_path / "lists" / "SHA256SUMS"
    sf.hash_sources(sources, ["prism"], out, log=lambda _: None)
    listed = out.read_bytes()
    assert sf.is_sums_list(out)
    sf.hash_sources(sources, ["prism", "gaitex"], out, log=lambda _: None)   # a list: replaced
    assert out.read_bytes() != listed and sf.is_sums_list(out)

    line = f"{_sha(b'a')}  prism/a.pkl\n"
    for content in (b"my notes\n", b"", line.encode() + b"not a sums line\n",
                    f"{_sha(b'b')}  prism/b.pkl\n{line}".encode(),        # not sorted
                    f"{_sha(b'a')}  weargait/a.csv\n".encode(),           # not a source folder
                    line.rstrip("\n").encode(), b"\xff\xfe binary"):
        other = tmp_path / "other.txt"
        other.write_bytes(content)
        assert not sf.is_sums_list(other)
        with pytest.raises(paths.OutputLocationRefused, match="--force"):
            sf.hash_sources(sources, ["prism"], other, log=lambda _: None)
        assert other.read_bytes() == content
    assert cli.main(["hash-sources", "--source-root", str(sources), "--sources", "prism",
                     "--out", str(other)]) == 3
    assert other.read_bytes() == b"\xff\xfe binary"
    assert cli.main(["hash-sources", "--source-root", str(sources), "--sources", "prism",
                     "--out", str(other), "--force"]) == 0
    assert sf.is_sums_list(other)
    # --force does not lift a location refusal
    assert cli.main(["hash-sources", "--source-root", str(sources), "--sources", "prism",
                     "--out", str(sources / "SUMS"), "--force"]) == 3
    capsys.readouterr()


def test_a_missing_source_folder_is_named(sources, tmp_path, clean_env):
    with pytest.raises(sf.SourceFilesError, match="amass"):
        sf.hash_sources(sources, ["amass"], tmp_path / "SUMS", log=lambda _: None)


# ---------------------------------------------------------------- staging
@pytest.fixture
def listed(sources, tmp_path, clean_env):
    out = tmp_path / "lists" / "SHA256SUMS"
    sf.hash_sources(sources, sf.resolve_folders(["hknu", "gaitex", "prism"]), out,
                    log=lambda _: None)
    return out


def test_staging_copies_what_is_missing_and_skips_what_is_there(sources, listed, tmp_path):
    to = tmp_path / "pc" / "extracted"
    _tree(to, {"prism/subj001/take002.pkl": FILES["prism/subj001/take002.pkl"]})
    result = sf.stage_sources(to, listed, from_root=sources, log=lambda _: None)
    assert result.exit_code == 0
    assert result.present == ["prism/subj001/take002.pkl"]
    assert len(result.copied) == 5
    assert _snapshot(to) == FILES
    assert not list(to.rglob("*" + sf.STAGING_SUFFIX))
    again = sf.stage_sources(to, listed, from_root=sources, log=lambda _: None)
    assert again.exit_code == 0 and again.copied == [] and len(again.present) == 6


def test_a_destination_that_differs_is_refused_and_nothing_is_copied(sources, listed, tmp_path):
    to = tmp_path / "pc" / "extracted"
    _tree(to, {"gaitex/README.md": b"someone edited this"})
    result = sf.stage_sources(to, listed, from_root=sources, log=lambda _: None)
    assert result.exit_code == sf.EXIT_REFUSED
    assert result.conflicts == ["gaitex/README.md"] and result.copied == []
    assert _snapshot(to) == {"gaitex/README.md": b"someone edited this"}     # not overwritten


def test_a_corrupt_or_absent_copy_is_reported_and_leaves_nothing(sources, listed, tmp_path):
    (sources / "gaitex" / "README.md").write_bytes(b"half-synced")
    (sources / "prism" / "subj001" / "take002.pkl").unlink()
    to = tmp_path / "pc" / "extracted"
    result = sf.stage_sources(to, listed, from_root=sources, log=lambda _: None)
    assert result.exit_code == sf.EXIT_INCOMPLETE
    assert result.corrupt == ["gaitex/README.md"]
    assert result.unavailable == ["prism/subj001/take002.pkl"]
    assert len(result.copied) == 4
    assert not (to / "gaitex" / "README.md").exists()
    assert not [p for p in to.rglob("*") if p.is_file() and p.name.endswith(sf.STAGING_SUFFIX)]


def test_verify_only_copies_nothing_and_reports(sources, listed, tmp_path):
    to = tmp_path / "pc" / "extracted"
    _tree(to, {"gaitex/README.md": b"different", "prism/subj001/take002.pkl":
               FILES["prism/subj001/take002.pkl"]})
    before = _snapshot(to)
    result = sf.stage_sources(to, listed, verify_only=True, log=lambda _: None)
    assert result.exit_code == sf.EXIT_INCOMPLETE
    assert result.conflicts == ["gaitex/README.md"] and len(result.missing) == 4
    assert result.present == ["prism/subj001/take002.pkl"]
    assert _snapshot(to) == before
    full = tmp_path / "full"
    sf.stage_sources(full, listed, from_root=sources, log=lambda _: None)
    assert sf.stage_sources(full, listed, verify_only=True, log=lambda _: None).exit_code == 0


def test_sources_narrow_what_is_staged(sources, listed, tmp_path):
    to = tmp_path / "pc" / "extracted"
    result = sf.stage_sources(to, listed, from_root=sources, folders=["prism"],
                              log=lambda _: None)
    assert result.copied == ["prism/subj001/take002.pkl"] and result.listed == 1
    assert set(_snapshot(to)) == {"prism/subj001/take002.pkl"}


def test_the_destination_gets_the_source_root_guards(sources, listed, tmp_path, clean_env):
    data = tmp_path / "data"
    data.mkdir()
    clean_env.setenv(paths.DATA_ROOT_ENV, str(data))
    poc = data / "runs" / "experimental_generation_poc_demo"
    for to in (tmp_path / "Dropbox" / "extracted", tmp_path / "Google Drive" / "src",
               poc / "_superseded" / "x", tmp_path / "any" / "_runs" / "x", data,
               data / "runs" / "x", data / "raw_archives" / "x", data / "body_models" / "smpl",
               sources / "inner", sources.parent):
        with pytest.raises(paths.OutputLocationRefused):
            sf.stage_sources(to, listed, from_root=sources, log=lambda _: None)
        assert not to.exists() or to in (data, sources.parent)
    # the data root's extracted folder is where sources go by default
    result = sf.stage_sources(data / "extracted", listed, from_root=sources, log=lambda _: None)
    assert result.exit_code == 0


def test_the_source_may_be_on_a_synchronised_or_shared_drive(tmp_path, clean_env):
    clean_env.setenv(paths.SHARED_DRIVE_NAMES_ENV, "Team_Share")
    shared = _tree(tmp_path / "team_drive" / "Shared drives" / "Team_Share" / "sources",
                   {"prism/a.pkl": b"a"})
    sums = tmp_path / "SUMS"
    sums.write_text(f"{_sha(b'a')}  prism/a.pkl\n", encoding="utf-8")
    result = sf.stage_sources(tmp_path / "local" / "extracted", sums, from_root=shared,
                              log=lambda _: None)
    assert result.exit_code == 0 and result.copied == ["prism/a.pkl"]


def test_nothing_is_overwritten_even_when_a_file_appears_during_staging(sources, listed,
                                                                       tmp_path, monkeypatch):
    to = tmp_path / "pc" / "extracted"
    original = sf._rename_without_overwriting

    def racing(temporary, target):
        if target.name == "README.md":
            target.write_bytes(b"written meanwhile")
        return original(temporary, target)

    monkeypatch.setattr(sf, "_rename_without_overwriting", racing)
    result = sf.stage_sources(to, listed, from_root=sources, log=lambda _: None)
    assert (to / "gaitex" / "README.md").read_bytes() == b"written meanwhile"
    assert "gaitex/README.md" in result.conflicts and result.exit_code == sf.EXIT_REFUSED
    assert not [p for p in to.rglob("*") if p.name.endswith(sf.STAGING_SUFFIX)]


def test_staging_leftovers_are_reported_never_deleted_and_never_source_files(sources, listed,
                                                                             tmp_path, capsys):
    """What an interrupted stage-sources leaves (``.<name>.<random>.staging``) is not a source
    file: neither listed nor traced. Staging, and --verify-only, report it and leave it."""
    leftover_name = ".take002.pkl.k3x_9q2a.staging"
    assert sf.is_staging_leftover(leftover_name)
    assert not any(sf.is_staging_leftover(name) for name in
                   ("take002.pkl", "a.staging", ".staging", "notes.staging.txt", ".x.staging"))
    # a leftover in the sources is not listed
    (sources / "prism" / "subj001" / leftover_name).write_bytes(b"partial")
    assert [p.name for p in sf.iter_source_files(sources / "prism")] == ["take002.pkl"]
    again = tmp_path / "again" / "SHA256SUMS"
    sf.hash_sources(sources, ["prism"], again, log=lambda _: None)
    assert "staging" not in again.read_text(encoding="utf-8")

    to = tmp_path / "pc" / "extracted"
    stray = to / "gaitex" / "austra" / ".markers.csv.abcd1234.staging"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"half a copy")
    messages: list[str] = []
    for verify_only in (True, False, True):
        result = sf.stage_sources(to, listed, from_root=sources, verify_only=verify_only,
                                  log=messages.append)
        assert result.leftovers == ["gaitex/austra/.markers.csv.abcd1234.staging"]
        assert stray.read_bytes() == b"half a copy"               # never deleted
    assert result.exit_code == 0 and len(result.present) == 6    # the leftover changes no code
    assert any("LEFTOVER" in m and "Remove it by hand" in m for m in messages)
    # a folder --sources leaves out is not looked at
    narrowed = sf.stage_sources(to, listed, verify_only=True, folders=["prism"], log=lambda _: None)
    assert narrowed.leftovers == []
    assert cli.main(["stage-sources", "--to", str(to), "--sums", str(listed), "--verify-only"]) == 0
    assert "1 leftover staging file(s)" in capsys.readouterr().out


# ---------------------------------------------------------------- the commands
def test_the_commands_exit_codes(sources, tmp_path, clean_env, capsys):
    sums = tmp_path / "lists" / "SHA256SUMS"
    assert cli.main(["hash-sources", "--source-root", str(sources), "--sources",
                     "prism,gaitex,hknu", "--out", str(sums)]) == 0
    assert cli.main(["hash-sources", "--source-root", str(sources), "--sources", "prism",
                     "--out", str(sources / "SUMS")]) == 3
    assert cli.main(["hash-sources", "--source-root", str(sources), "--sources", "amass",
                     "--out", str(tmp_path / "x")]) == 2
    to = tmp_path / "pc" / "extracted"
    assert cli.main(["stage-sources", "--to", str(to), "--sums", str(sums), "--verify-only"]) == 1
    assert cli.main(["stage-sources", "--from", str(sources), "--to", str(to),
                     "--sums", str(sums)]) == 0
    assert cli.main(["stage-sources", "--to", str(to), "--sums", str(sums), "--verify-only"]) == 0
    (to / "prism" / "subj001" / "take002.pkl").write_bytes(b"changed")
    assert cli.main(["stage-sources", "--from", str(sources), "--to", str(to),
                     "--sums", str(sums)]) == 3
    assert cli.main(["stage-sources", "--to", str(to), "--sums", str(sums)]) == 2   # no --from
    assert cli.main(["stage-sources", "--from", str(sources), "--to", str(to),
                     "--sums", str(tmp_path / "absent")]) == 2
    capsys.readouterr()


def test_the_defaults_resolve_through_the_environment(sources, tmp_path, clean_env):
    data = tmp_path / "data"
    (data / "extracted").mkdir(parents=True)
    clean_env.setenv(paths.DATA_ROOT_ENV, str(data))
    sums = tmp_path / "SUMS"
    # hash-sources reads SOMA_SOURCE_ROOT (else <data root>/extracted)
    clean_env.setenv(paths.SOURCE_ROOT_ENV, str(sources))
    assert cli.main(["hash-sources", "--sources", "prism", "--out", str(sums)]) == 0
    # stage-sources fills SOMA_SOURCE_ROOT, else <data root>/extracted
    clean_env.delenv(paths.SOURCE_ROOT_ENV)
    assert cli.main(["stage-sources", "--from", str(sources), "--sums", str(sums)]) == 0
    assert (data / "extracted" / "prism" / "subj001" / "take002.pkl").is_file()
    clean_env.delenv(paths.DATA_ROOT_ENV)
    assert cli.main(["stage-sources", "--from", str(sources), "--sums", str(sums)]) == 2


def test_help_documents_both_commands(capsys):
    for command in ("hash-sources", "stage-sources"):
        with pytest.raises(SystemExit) as done:
            cli.main([command, "--help"])
        assert done.value.code == 0
    text = capsys.readouterr().out
    assert "--verify-only" in text and "--source-root" in text


@pytest.mark.skipif(os.name != "nt", reason="the POSIX branch is exercised on POSIX")
def test_windows_rename_refuses_an_existing_target(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    with pytest.raises(FileExistsError):
        sf._rename_without_overwriting(a, b)
    assert b.read_bytes() == b"b"
