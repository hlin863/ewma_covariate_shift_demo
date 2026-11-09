"""Run TWIN-ADAPT on BCI Competition IV 2A/2B using existing GDF/MAT loaders."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptation.experiments.twin_adapt_bci import run_bci_experiment
from src.adaptation.twin_adapt.config import TwinAdaptConfig


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["2a", "2b"], required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments/twin_adapt_bci"))
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--win-size", type=int, default=10)
    parser.add_argument("--win-size-max", type=int, default=120)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--delta", type=float, default=0.85)
    parser.add_argument("--label-delay", type=int, default=0, help="-1 withholds labels entirely; units are batches")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    config = TwinAdaptConfig(
        batch_size=args.batch_size, win_size=args.win_size,
        win_size_max=args.win_size_max, gamma=args.gamma, delta=args.delta,
        label_delay=None if args.label_delay == -1 else args.label_delay, seed=args.seed,
    )
    destination, summary = run_bci_experiment(
        dataset=args.dataset, subject=args.subject, config=config,
        data_root=args.data_root, output_root=args.output_root,
    )
    print(f"Saved TWIN-ADAPT run: {destination}")
    print(f"Prequential accuracy: {summary['accuracy']:.4f}")
    print(f"Updates: {summary['updates']}")
    return destination


if __name__ == "__main__":
    main()
