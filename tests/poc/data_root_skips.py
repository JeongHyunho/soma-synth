"""The data root the real-data tests read, and a skip marker that says why they did not run.

These tests resolve the data root as the generators do (``pipeline.paths``; there is no
default), so a shell without ``SOMA_DATA_ROOT`` skips them. That skip and the one for a machine
that lacks a source are different news: the first is coverage lost to configuration (a CI job, a
shell without the variable), the second is missing data. The reasons start differently so
``pytest -rs`` tells them apart:

* ``SOMA_DATA_ROOT unset: ...`` -- the variable is unset or blank;
* ``SOMA_DATA_ROOT unusable: ...`` -- set, but not an absolute path to an existing folder;
* ``SOMA_DATA_ROOT set, data missing: ...`` -- the root resolves and the file the test reads is
  not under it.

Imported by basename: pytest puts this folder on ``sys.path`` for the test modules beside it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from soma_synth.pipeline import paths

#: Where DATA_ROOT points when the variable does not resolve: a relative name no file exists under,
#: so every "is the source there" probe is False and the test skips with the reason below.
UNRESOLVED = Path("<SOMA_DATA_ROOT does not resolve>")


def resolve() -> tuple[Path | None, str | None]:
    """(data root, None), or (None, the skip reason's prefix and cause) when it does not resolve."""
    try:
        return paths.data_root(), None
    except paths.PathConfigError as error:
        value = os.environ.get(paths.DATA_ROOT_ENV)
        if value is None or not value.strip():
            return None, (f"{paths.DATA_ROOT_ENV} unset: this data test did not run; set it to the "
                          "data root to cover it")
        return None, f"{paths.DATA_ROOT_ENV} unusable: {error}"


def skip_reason(what: str, status: tuple[Path | None, str | None] | None = None) -> str:
    """Why a test that needs ``what`` under the data root is skipped."""
    root, problem = status if status is not None else (_ROOT, _PROBLEM)
    if root is None:
        return f"{problem} (needs {what})"
    return f"{paths.DATA_ROOT_ENV} set, data missing: {what} not found under {root}"


def needs_data(present: bool, what: str) -> pytest.MarkDecorator:
    """``skipif(not present)`` with :func:`skip_reason` for ``what``."""
    return pytest.mark.skipif(not present, reason=skip_reason(what))


_ROOT, _PROBLEM = resolve()
#: The data root, or :data:`UNRESOLVED` (under which nothing exists) when it does not resolve.
DATA_ROOT = _ROOT if _ROOT is not None else UNRESOLVED
