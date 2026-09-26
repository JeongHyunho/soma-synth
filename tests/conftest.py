"""Refuse the suite early on an interpreter older than the package supports, and say why.

``pyproject.toml`` requires Python 3.12 or newer; the canonical environment is CPython 3.13.5
(``constraints.txt``). A run on an older interpreter would fail in scattered, misleading ways
rather than with one line naming the cause.
"""

from __future__ import annotations

import sys

import pytest

MINIMUM = (3, 12)


def pytest_configure(config) -> None:  # noqa: ANN001 - pytest hook signature
    if sys.version_info >= MINIMUM:
        return
    running = ".".join(str(part) for part in sys.version_info[:3])
    required = ".".join(str(part) for part in MINIMUM)
    # UsageError rather than a bare exception: pytest renders it as one ERROR line.
    raise pytest.UsageError(
        f"this suite requires Python {required} or newer; running {running}. "
        "The canonical environment is CPython 3.13.5 (constraints.txt)."
    )
