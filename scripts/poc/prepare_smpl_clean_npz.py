"""Fail-closed converter: licensed SMPL .pkl (chumpy) -> chumpy-free .npz.

The published SMPL pickles wrap arrays as chumpy objects and use protocol-0 globals
that would execute on load. This converter maps every ``chumpy.*`` class to a passive
capture stub and allows ONLY numpy / scipy.sparse reconstruction globals; any other
global is refused, so a tampered pickle cannot import or execute code. The licensed
original is read read-only and never modified.

    python scripts/poc/prepare_smpl_clean_npz.py --src <basicmodel_f_...pkl> --out <SMPL_FEMALE_clean.npz>
"""
from __future__ import annotations

import argparse
import os
import pickle
import warnings

import numpy as np

# Standard SMPL shape-space width (PRISM betas are 10-dim; the male clean npz is also 10).
DEFAULT_NUM_BETAS = 10

_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("scipy.sparse._csr", "csr_matrix"),
    ("scipy.sparse.csr", "csr_matrix"),
    ("scipy.sparse._csc", "csc_matrix"),
    ("scipy.sparse.csc", "csc_matrix"),
    # numpy's own ndarray.__reduce__ round-trips raw buffer bytes through a
    # latin1-encoded str for pickle protocols < 3, then calls _codecs.encode()
    # to turn that str back into bytes. This is part of numpy's reconstruction
    # chain (always invoked as _codecs.encode(<str>, "latin1")), not an
    # attacker-controlled code path, so it belongs in the whitelist alongside
    # the other numpy globals above.
    ("_codecs", "encode"),
    # Generic object reconstruction + base types/containers used by the licensed SMPL pickles
    # (enumerated by a no-execution pickletools disassembly of the basicmodel pickles). All are
    # safe: copy_reg._reconstructor calls base.__new__(cls) where `cls` is itself gated through
    # find_class (chumpy -> stub; anything unlisted is refused), and object/set run no code.
    # Dangerous globals (os.system, builtins.eval/exec, subprocess, ...) are absent and stay
    # rejected. Py2 names appear in the pickle; Python 3 maps copy_reg->copyreg and
    # __builtin__->builtins via _compat_pickle when super().find_class resolves them.
    ("copy_reg", "_reconstructor"),
    ("copyreg", "_reconstructor"),
    ("__builtin__", "object"),
    ("builtins", "object"),
    ("__builtin__", "set"),
    ("builtins", "set"),
}


class _Captured:
    """Passive stand-in for any chumpy class: keeps the pickled state, runs nothing."""

    def __init__(self, *args, **kwargs):
        self._args = args

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state

    def to_array(self):
        for key in ("x", "_x", "r"):
            value = self.__dict__.get(key)
            if isinstance(value, np.ndarray):
                return value
        for value in self.__dict__.values():
            if isinstance(value, np.ndarray):
                return value
        raise TypeError("no array payload in captured object: %s" % list(self.__dict__))


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] == "chumpy":
            return _Captured
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError("BLOCKED non-whitelisted global: %s.%s" % (module, name))


def _plain(value):
    if isinstance(value, _Captured):
        return np.asarray(value.to_array())
    if hasattr(value, "toarray"):  # scipy sparse J_regressor
        return np.asarray(value.toarray())
    if isinstance(value, np.ndarray):
        return value
    return value


def convert(src: str, out: str, num_betas: int = DEFAULT_NUM_BETAS) -> None:
    src = os.path.expanduser(src)
    out = os.path.expanduser(out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(src, "rb") as handle:
        # Old SMPL pickles reconstruct numpy dtypes with the legacy `align=0` idiom, which numpy
        # emits a VisibleDeprecationWarning for. It is harmless (numpy handles it) and outside our
        # control (it is the licensed asset's encoding), so suppress just that noise around the load.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = _StubUnpickler(handle, encoding="latin1").load()

    converted = {}
    for key, value in raw.items():
        plain = _plain(value)
        if isinstance(plain, np.ndarray):
            converted[key] = plain
        elif isinstance(plain, (str, bytes, int, float)):
            converted[key] = np.asarray(plain)

    if "kintree_parents" not in converted and "kintree_table" in converted:
        kt = np.asarray(converted["kintree_table"])
        parents = kt[0].astype(np.int64).copy()
        parents[0] = -1
        converted["kintree_parents"] = parents

    # Truncate to the standard SMPL 10-shape space so the clean model matches PRISM's 10-dim betas
    # and the male clean npz (some basicmodel pickles ship the full 300-PC shape space).
    sd = converted.get("shapedirs")
    if sd is not None and sd.ndim == 3 and sd.shape[2] > num_betas:
        converted["shapedirs"] = np.ascontiguousarray(sd[:, :, :num_betas])

    required = ("v_template", "shapedirs", "J_regressor", "kintree_parents")
    missing = [k for k in required if k not in converted]
    if missing:
        raise SystemExit("model missing required fields: %s" % missing)

    np.savez(out, **converted)
    print("wrote %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-betas", type=int, default=DEFAULT_NUM_BETAS,
                    help="truncate shapedirs to this many shape PCs (default 10)")
    args = ap.parse_args()
    convert(args.src, args.out, num_betas=args.num_betas)


if __name__ == "__main__":
    main()
