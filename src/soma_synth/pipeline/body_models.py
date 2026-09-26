"""The body-model set a run used, resolved and hashed through ``smpl18``.

The run records the model set it generated from (``PIPELINE_GOVERNANCE.md`` §10.2 item 6). The
directory is the one the generators load from, :func:`soma_synth.pipeline.paths.body_model_dir`:
``SOMA_BODY_MODEL_DIR`` when set, else the registry's ``body_model_sets`` relative path under the
run's data root (``body_models/smpl``, the same folder). Which file a gender selects comes from
:mod:`smpl18.model.select` -- the package that owns the SMPL-24 conversion and its model
convention. The record therefore names what the generators actually read.

The whole set is hashed, not one file: every generator picks per subject
(``body_model_selection: by_subject_gender``), so a bundle of mixed-sex subjects came from more
than one of these files. For the run identity the set collapses to a single digest over those
hashes, so a changed or added model gives a different identity.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from smpl18.model import select as smpl18_select

from soma_synth.pipeline import paths as paths_mod
from soma_synth.pipeline import provenance as provenance_mod

__all__ = [
    "SET_DIGEST_ALGORITHM",
    "path_for_gender",
    "set_digest",
    "set_directory",
    "set_directory_source",
    "set_hashes",
    "set_record",
]

#: How the per-file hashes collapse into the one value the run identity carries.
SET_DIGEST_ALGORITHM = "sha256(sorted 'name sha256' lines)"


def set_directory(relative_path: str, data_root: Path | str | None) -> Path | None:
    """Where the set lives for this run, or None when there is no set or nowhere to look.

    ``SOMA_BODY_MODEL_DIR`` wins when it is set, because the generators then load from it
    (:func:`paths.body_model_dir`) and the record has to name what they loaded. Otherwise the
    registry's relative path under ``data_root``, which for ``body_models/smpl`` is exactly
    ``paths.body_model_dir(root=data_root)``.
    """
    if not relative_path:
        return None
    if paths_mod.body_model_override() is not None:
        return paths_mod.body_model_dir(root=data_root)
    if data_root is None:
        return None
    return Path(data_root) / relative_path


def set_directory_source() -> str:
    """Which setting chose the directory :func:`set_directory` returns, for the run record."""
    if paths_mod.body_model_override() is not None:
        return paths_mod.BODY_MODEL_DIR_ENV
    return "data_root"


def set_hashes(directory: Path | str | None) -> dict[str, str] | None:
    """``{file name: sha256}`` for every model in the set, or None when there is no set to read.

    Reads the set through :func:`smpl18.model.select.set_hashes`, so the pipeline and the
    conversion package cannot disagree about which files belong to it.
    """
    if directory is None:
        return None
    root = Path(directory)
    if not root.is_dir():
        return None
    found = {name: digest for name, digest in smpl18_select.set_hashes(root).items() if digest}
    return found or None


def set_digest(hashes: dict[str, str] | None) -> str | None:
    """One value standing for the whole set, for the run identity. None when the set is absent."""
    if not hashes:
        return None
    lines = "\n".join(f"{name} {hashes[name]}" for name in sorted(hashes))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def set_record(
    directory: Path | str | None, *, set_name: str, selection: str, resolver: str,
    data_root: Path | str | None = None,
) -> dict[str, object]:
    """The §10.2 item 6 record: the set, how a file is chosen from it, and every file's hash.

    With ``data_root``, the directory is also given relative to it (None when it lies outside,
    as a ``SOMA_BODY_MODEL_DIR`` elsewhere may).
    """
    hashes = set_hashes(directory)
    if hashes is None:
        reason = (
            "no data root, so the body model set was not located"
            if directory is None
            else f"no model set under {Path(directory)}"
        )
        return provenance_mod.Unavailable(reason).as_json()
    record: dict[str, object] = {
        "set": set_name,
        "directory": str(directory),
        "selection": selection,
        "resolver": resolver,
        "models": hashes,
        "set_digest": set_digest(hashes),
        "set_digest_algorithm": SET_DIGEST_ALGORITHM,
    }
    if data_root is not None:
        record["directory_relative"] = provenance_mod.relative_to_root(directory, data_root)
        record["directory_from"] = set_directory_source()
    return record


def path_for_gender(gender: str, directory: Path | str) -> Path:
    """The model file a gender selects, as the generators select it."""
    return smpl18_select.model_path_for_gender(gender, directory)
