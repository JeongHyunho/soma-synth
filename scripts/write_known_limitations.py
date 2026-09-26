"""Render each bundle's KNOWN_LIMITATIONS.json from the repository's one config.

The config ``configs/datasets/known_limitations_v1.yaml`` is the source of truth; the JSON in
the bundle root is what a consumer reads without the repository. The validator checks that the
two agree, so this script is the only sanctioned way to change a bundle's copy.

    python scripts/write_known_limitations.py <bundle_dir> [<bundle_dir> ...] [--check]

``--check`` writes nothing and exits non-zero if any bundle's file is missing, malformed, or
differs from the config.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from soma_synth.contracts import dataset_limitations as dl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundles", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    config = dl.load_config(args.config)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    failed = 0
    for bundle in args.bundles:
        name = bundle.name
        target = bundle / dl.FILE_NAME
        if name not in config.bundles:
            print(f"{name}: no block in {dl.CONFIG_RELPATH}")
            failed += 1
            continue
        if args.check:
            problems = dl.problems_in_file(target, name, config) if target.is_file() \
                else [f"{dl.FILE_NAME} missing"]
            print(f"{name}: {'ok' if not problems else '; '.join(problems)}")
            failed += bool(problems)
            continue
        # The same renderer the pipeline runner uses after a generator returns; an unchanged
        # file keeps its timestamp so a no-op re-render is byte-identical.
        dl.write_rendered(bundle, config, generated_utc=stamp)
        print(f"{name}: wrote {len(config.bundles[name])} entries")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
