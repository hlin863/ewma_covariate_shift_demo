"""Command-line runner for the 2015 paper's synthetic datasets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.paper2015.d2 import (
    D2ExperimentConfig,
    PAPER_D2_REFERENCE,
    run_d2_experiment,
)
from src.simulation.jumping_mean import JumpingMeanConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["d2"], default="d2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument(
        "--lambda-mode",
        choices=["estimate", "configured"],
        default="estimate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics/paper2015/d2"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_d2_experiment(
        D2ExperimentConfig(
            dataset=JumpingMeanConfig(random_seed=args.seed),
            train_size=args.train_size,
            lambda_mode=args.lambda_mode,
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.stage_1_evaluation.events.to_csv(
        args.output_dir / "stage1_events.csv", index=False
    )
    result.stage_2_evaluation.events.to_csv(
        args.output_dir / "stage2_events.csv", index=False
    )
    result.detector.stage_1_results.to_csv(
        args.output_dir / "stage1_trace.csv", index=False
    )
    result.detector.stage_2_results.to_csv(
        args.output_dir / "stage2_validations.csv", index=False
    )
    summary = {
        "dataset": "D2",
        "evaluation_scope": "all testing-stream shifts",
        "estimated_lambda": result.detector.training_result.lambda_value,
        "stage_1": asdict(result.stage_1_evaluation.metrics),
        "stage_2": asdict(result.stage_2_evaluation.metrics),
        "paper_table_3_reference": PAPER_D2_REFERENCE,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote D2 results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
