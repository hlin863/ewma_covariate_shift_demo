"""Optimise TWIN-ADAPT on development-only BCI features and evaluate on held-out sessions."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptation.experiments.twin_adapt_bci import run_optimized_bci_experiment
from src.adaptation.twin_adapt.objective import CostObjectiveConfig
from src.adaptation.twin_adapt.optimization import TwinAdaptSearchSpace


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("2a", "2b"))
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("outputs/experiments/twin_adapt_optimization"))
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--promotion-fraction", type=float, default=1 / 3)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--label-delay", type=int, default=0,
                        help="-1 withholds labels from the adaptation controller")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--reference-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)

    space = TwinAdaptSearchSpace(
        batch_size=args.batch_size,
        label_delay=None if args.label_delay == -1 else args.label_delay,
        seed=args.seed,
    )
    objective = CostObjectiveConfig(
        alpha=args.alpha, beta=args.beta,
        reference_seconds=args.reference_seconds,
    )
    destination, summary = run_optimized_bci_experiment(
        dataset=args.dataset,
        subject=args.subject,
        data_root=args.data_root,
        output_root=args.output_root,
        search_space=space,
        budgets=args.budgets,
        objective=objective,
        initial_candidates=args.candidates,
        promotion_fraction=args.promotion_fraction,
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved optimisation evidence to {destination}")


if __name__ == "__main__":
    main()
