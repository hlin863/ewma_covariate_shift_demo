"""Export figures from an existing TWIN-ADAPT run without changing its evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptation.experiments.run_figures import export_run_figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Run directory containing accuracy_windows.csv or final_evaluation/",
    )
    args = parser.parse_args(argv)
    manifest = export_run_figures(args.run_dir)
    print(json.dumps(manifest, indent=2))
    print(f"Saved plots in {args.run_dir / 'figures'}")


if __name__ == "__main__":
    main()
