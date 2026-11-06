"""Run full online adaptation on repository's Diethe-inspired synthetic stream."""
from argparse import ArgumentParser
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptation.experiments.diethe import DietheConfig, synthetic_features
from src.adaptation.policies import (
    NeverUpdate, PeriodicRetrain, RetrainOnWarning, RetrainOnValidatedShift,
)
from src.online.pipeline import OnlineAdaptiveBCI, OnlineAdaptationConfig


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["none", "feature_shift", "relationship_shift"],
                        default="feature_shift")
    parser.add_argument("--policy", choices=["never", "periodic", "warning", "validated"],
                        default="validated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-delay", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/online/synthetic"))
    args = parser.parse_args()
    dataset, change_at = synthetic_features(DietheConfig(
        seed=args.seed, scenario=args.scenario
    ))
    policies = dict(never=NeverUpdate(), periodic=PeriodicRetrain(10),
                    warning=RetrainOnWarning(), validated=RetrainOnValidatedShift())
    monitor = OnlineAdaptiveBCI.from_calibration(
        dataset["training_features"], dataset["training_labels"],
        policy=policies[args.policy],
        config=OnlineAdaptationConfig(window_size=args.window_size,
                                      label_delay=None if args.label_delay < 0 else args.label_delay),
        pca_components=3, lambda_override=0.2,
        control_limit_multiplier=2.5,
    )
    result = monitor.run_replay(
        dataset["evaluation_features"],
        dataset["evaluation_times"],
        dataset["evaluation_labels"],
    )
    folder = args.output / args.scenario / args.policy
    folder.mkdir(parents=True, exist_ok=True)
    result.trials.to_csv(folder / "trials.csv", index=False)
    result.validations.to_csv(folder / "stage2.csv", index=False)
    result.updates.to_csv(folder / "updates.csv", index=False)
    summary = dict(scenario=args.scenario, policy=args.policy, seed=args.seed,
                   shift_at=change_at, trials=len(result.trials),
                   accuracy=float(result.trials["correct"].mean()),
                   warnings=int(result.trials["stage1_warning"].sum()),
                   updates=len(result.updates),
                   confirmed=int(result.validations["confirmed_shift"].sum())
                   if not result.validations.empty else 0)
    pd.DataFrame([summary]).to_csv(folder / "summary.csv", index=False)
    print(pd.Series(summary).to_string())
    print("Artifacts:", folder)


if __name__ == "__main__":
    main()
