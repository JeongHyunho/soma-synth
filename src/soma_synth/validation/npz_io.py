"""Pickle-free npz introspection: read every member's shape+dtype WITHOUT unpickling.

``read_npz_schema`` reads .npy headers straight from the zip, so object-dtype arrays (which
``np.load(allow_pickle=False)[key]`` raises on) are reported without executing any pickle.
``load_numeric`` loads named numeric arrays with ``allow_pickle=False``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as _npformat


def read_npz_schema(path: str | Path) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Return ``{key: (shape, dtype)}`` for every array in the npz, header-only (no pickle)."""
    out: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            key = name[:-4]
            with zf.open(name) as fp:
                major, _minor = _npformat.read_magic(fp)
                if major == 1:
                    shape, _fortran, dtype = _npformat.read_array_header_1_0(fp)
                else:
                    shape, _fortran, dtype = _npformat.read_array_header_2_0(fp)
            out[key] = (tuple(shape), dtype)
    return out


def load_numeric(path: str | Path, keys) -> dict[str, np.ndarray]:
    """Load named arrays with ``allow_pickle=False`` (numeric/unicode only)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in keys if k in z.files}
