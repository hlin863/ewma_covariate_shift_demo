"""Compare Park (2023) first-order binary decisions against CPSS on paired streams.

Run: python -m scripts.run_park_cpss_benchmark --evaluation-seeds 100
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.experiments.park2023.cpss_benchmark import (
    BenchmarkConfig, export_benchmark, run_benchmark,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", type=int, default=160)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--calibration-seeds", type=int, default=12)
    p.add_argument("--evaluation-seeds", type=int, default=24)
    p.add_argument("--perturbations", type=int, default=8)
    p.add_argument("--jitter-sd", type=float, default=0.12)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--mean-shift-size", type=float, default=2.0)
    p.add_argument("--variance-factor", type=float, default=1.8)
    p.add_argument("--min-segment", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", type=Path, default=Path("outputs/park_cpss") /
                   datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args = p.parse_args()
    try:
        config = BenchmarkConfig(**{key: getattr(args, key)
                                    for key in BenchmarkConfig.__dataclass_fields__})
        result = run_benchmark(config)
        saved = export_benchmark(result, args.output)
    except (ValueError, OSError) as error:
        p.error(str(error))
    print("Park binary versus proposed CPSS — exploratory synthetic comparison")
    print(f"Calibration threshold: {result['calibration']['threshold']:.6f}")
    for method, stats in result["classification"].items():
        print(f"{method}: accuracy={stats['accuracy']:.3f}, "
              f"balanced_accuracy={stats['balanced_accuracy']:.3f}, "
              f"F1={stats['f1']}, AUC={stats['roc_auc']:.3f}, "
              f"stationary_FPR={stats['stationary_positive_rate']:.3f}, "
              f"variance_only_alarm_rate={stats['variance_only_positive_rate']:.3f}")
    print(saved)


if __name__ == "__main__":
    main()
