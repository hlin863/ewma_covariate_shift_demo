# scripts/run_matched_bci_comparison.py

"""Run a matched frozen/adaptive BCI comparison."""

from argparse import ArgumentParser
from pathlib import Path

from src.adaptation.experiments.matched_bci_comparison import (
    run_matched_bci_comparison,
)
from src.adaptation.twin_adapt.config import TwinAdaptConfig


def main():
    parser = ArgumentParser(description=__doc__)

    parser.add_argument("--dataset", required=True, choices=("2a", "2b"))
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--win-size", type=int, default=10)
    parser.add_argument("--win-size-max", type=int, default=120)

    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--delta", type=float, default=0.85)
    parser.add_argument("--label-delay", type=int, default=0)

    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))

    args = parser.parse_args()

    config = TwinAdaptConfig(
        batch_size=args.batch_size,
        win_size=args.win_size,
        win_size_max=args.win_size_max,
        gamma=args.gamma,
        delta=args.delta,
        label_delay=(None if args.label_delay == -1 else args.label_delay),
        seed=args.seed,
    )

    destination, summary = run_matched_bci_comparison(
        dataset=args.dataset,
        subject=args.subject,
        config=config,
        data_root=args.data_root,
    )

    print(f"Saved comparison to: {destination}")
    print(f"Frozen accuracy: {summary['frozen_accuracy']:.4f}")
    print(f"Adaptive accuracy: {summary['adaptive_accuracy']:.4f}")
    print("Accuracy difference (pp): " f"{summary['accuracy_difference_pp']:+.2f}")
    print(f"Adaptive updates: {summary['adaptive_updates']}")


if __name__ == "__main__":
    main()
