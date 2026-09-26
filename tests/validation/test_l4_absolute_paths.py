"""L4 governance: a take manifest names no absolute filesystem path, Windows or POSIX.

A manifest carries logical ids (``extracted/...``, ``runs/...``) relative to the data root. The
check caught a drive letter only; teammates may generate on macOS or Linux, so it also catches a
POSIX path under any root (a Linux data root may be /data, /opt, /srv, /scratch, /nfs, /gpfs,
/workspace, ...) -- without taking a logical id that happens to contain such a folder name, a unit,
or a URL, for one. Changing the check changes the validator digest,
so every validation ledger re-checks its takes once (GENERATION_PIPELINE_STANDARD section 7).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soma_synth.validation import checks

from test_validation_checks import make_take

FLAGGED = [
    r"E:\data\runs\x",
    "C:/Users/someone/x",
    "/Users/someone/data/x.npz",
    "/users/someone/x",                        # macOS volumes are case-insensitive
    "/home/someone/x",
    "/Volumes/External/x",
    "/mnt/data/x",
    "/media/usb/x",
    "/run/media/someone/x",
    "/tmp/x.npz",
    "/var/folders/ab/c1d2/T/x",
    "/private/var/tmp/x",
    "/root/x",
    "/data/someone/soma/extracted/x.npz",      # a Linux data root: any root counts
    "/opt/soma/runs/x",
    "/srv/soma/x",
    "/scratch/someone/x",
    "/nfs/lab/soma/x",
    "/workspace/soma/x",
    "/gpfs/project/x",
    "python gen.py --out /home/someone/out",   # after whitespace
    "cwd=/Users/someone",                      # after '='
    "cwd:/home/someone/x",                     # after ':'
    "SOMA_DATA_ROOT=/data/soma/runs",
    "'/tmp/x'",
    "(/mnt/x)",
    "a;/media/x",
]
NOT_FLAGGED = [
    "extracted/amass/CMU/01/01_01_poses.npz",
    "runs/experimental_generation_poc_demo/_runs/run_x/tmp/y",   # a folder named tmp, logically
    "extracted/hknu_fullbody/home/x.mat",
    "https://github.com/org/repo/tree/main/home/x",
    "https://example.invalid/tmp/data",
    "http://host/Users/x",
    "file:///home/someone/x",                  # a URL, not a path value
    "file:///data/soma/x",
    "s3://bucket/home/x",
    "ftp://host/data/x",
    "x/tmp/y",
    "left/right",
    "m/s^2",
    "rad/s, m/s^2",
    "deg / s",
    "/",
    "/tmp",                                    # no path under it
    "/data",
    "N/A",
    "https://example.invalid/fixture",
]


def _governance(tmp_path: Path, value: str) -> list:
    take = make_take(tmp_path / "amass_x", "t0")
    manifest = json.loads((take / "manifest.json").read_text(encoding="utf-8"))
    manifest["provenance_note"] = {"nested": [value]}
    (take / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return [f for f in checks.check_take_l4_governance(take, "t0") if f.severity == "FAIL"]


@pytest.mark.parametrize("value", FLAGGED)
def test_an_absolute_path_fails_l4(tmp_path, value):
    failures = _governance(tmp_path, value)
    assert [f.message.split(":")[0] for f in failures] == ["absolute path in a value"], failures


@pytest.mark.parametrize("value", NOT_FLAGGED)
def test_a_logical_id_or_a_url_passes(tmp_path, value):
    assert _governance(tmp_path, value) == []


def test_the_fixture_take_passes_as_it_is(tmp_path):
    take = make_take(tmp_path / "amass_x", "t0")
    assert [f for f in checks.check_take_l4_governance(take, "t0") if f.severity == "FAIL"] == []
