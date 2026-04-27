#!/usr/bin/env python3

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SETS = {
    "encoder":    "scripts/generate_encoder_figures.py",
    "temporal":   "scripts/generate_temporal_figures.py",
    "taxonomy":   "scripts/generate_taxonomy_figures.py",
    "trajectory": "scripts/generate_trajectory_figures.py",
}


_USAGE = """Usage: python scripts/generate.py <set> [args...]

Per-set help: python scripts/generate.py <set> --help
See docs/scripts_overview.md for what each set produces.
"""


def _usage(exit_code: int = 1):
    stream = sys.stderr if exit_code else sys.stdout
    print(_USAGE, file=stream)
    available = sorted(k for k, v in SETS.items() if (PROJECT_ROOT / v).exists())
    missing = sorted(k for k, v in SETS.items() if not (PROJECT_ROOT / v).exists())
    print(f"Available on this checkout: {', '.join(available)}", file=stream)
    if missing:
        print(f"Defined but script missing (gitignored?): {', '.join(missing)}",
              file=stream)
    sys.exit(exit_code)


def main():
    if len(sys.argv) < 2:
        _usage(1)
    set_name = sys.argv[1]
    if set_name in ("-h", "--help"):
        _usage(0)
    if set_name not in SETS:
        print(f"Error: unknown figure set '{set_name}'\n", file=sys.stderr)
        _usage(1)

    target = PROJECT_ROOT / SETS[set_name]
    if not target.exists():
        print(f"Error: target script not present at {target}", file=sys.stderr)
        print("(This set may be local-only and gitignored.)", file=sys.stderr)
        sys.exit(2)

    # Make the dispatched script see its own argv — argparse inside the target
    # will behave as if it was invoked directly.
    sys.argv = [str(target)] + sys.argv[2:]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
