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
    run_d2_table_3_experiment,
)
from src.experiments.paper2015.table4 import (
    PAPER_TABLE_4_REFERENCE,
    Table4ExperimentConfig,
    run_table_4_experiment,
)
from src.simulation.jumping_mean import JumpingMeanConfig
from src.simulation.multivariate_shift import MultivariateShiftConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["d2", "d3", "d4", "table4"],
        default="d2",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument(
        "--lambda-mode",
        choices=["estimate", "configured"],
        default="configured",
    )
    parser.add_argument(
        "--evaluation-scope",
        choices=["table3", "full-stream"],
        default="table3",
    )
    parser.add_argument("--ici-block-size", type=int, default=10)
    parser.add_argument(
        "--ici-confidence-parameter",
        type=float,
        default=2.0,
    )
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--lambda-value", type=float, default=0.10)
    parser.add_argument("--stage2-window-size", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def _markdown_table(table) -> str:
    rendered = table.reset_index()
    columns = list(rendered.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for values in rendered.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            formatted.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, separator, *rows]) + "\n"


def main() -> int:
    args = parse_args()
    if args.dataset != "d2":
        dataset_names = {
            "d3": ("D3",),
            "d4": ("D4",),
            "table4": ("D3", "D4"),
        }[args.dataset]
        output_dir = args.output_dir or Path("outputs/metrics/paper2015/table4")
        output_dir.mkdir(parents=True, exist_ok=True)
        table_4 = run_table_4_experiment(
            Table4ExperimentConfig(
                dataset=MultivariateShiftConfig(random_seed=args.seed),
                repetitions=args.repetitions,
                lambda_value=args.lambda_value,
                stage_2_window_size=args.stage2_window_size,
            ),
            datasets=dataset_names,
        )
        table_4.computed.to_csv(output_dir / "table4_computed.csv")
        table_4.comparison.to_csv(output_dir / "table4_comparison.csv")
        (output_dir / "table4_computed.md").write_text(
            "# Computed multivariate Table IV results\n\n"
            + _markdown_table(table_4.computed),
            encoding="utf-8",
        )
        for name, stream in table_4.representative_streams.items():
            stream.to_csv(output_dir / f"{name.lower()}_stream.csv", index=False)
        for name, trace in table_4.representative_traces.items():
            safe_name = name.lower().replace("-", "_")
            trace.to_csv(output_dir / f"{safe_name}_trace.csv", index=False)
        summary = {
            "datasets": list(dataset_names),
            "metadata": table_4.metadata,
            "computed": table_4.computed.reset_index().to_dict(orient="records"),
            "paper_table_4_reference": {
                name: PAPER_TABLE_4_REFERENCE[name] for name in dataset_names
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        print(f"Wrote Table IV results to {output_dir}")
        return 0

    config = D2ExperimentConfig(
        dataset=JumpingMeanConfig(random_seed=args.seed),
        train_size=args.train_size,
        lambda_mode=args.lambda_mode,
        ici_block_size=args.ici_block_size,
        ici_confidence_parameter=args.ici_confidence_parameter,
    )
    output_dir = args.output_dir or Path("outputs/metrics/paper2015/d2")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluation_scope == "table3":
        table_3 = run_d2_table_3_experiment(config)
        result = table_3.experiment
        table_3.computed.to_csv(output_dir / "table3_computed.csv")
        table_3.comparison.to_csv(output_dir / "table3_comparison.csv")
        (output_dir / "table3_computed.md").write_text(
            "# Computed D2 Table III results\n\n"
            + _markdown_table(table_3.computed),
            encoding="utf-8",
        )
        table_3.ici_cdt.trace.to_csv(
            output_dir / "ici_cdt_trace.csv", index=False
        )
        for method, evaluation in table_3.evaluations.items():
            evaluation.events.to_csv(
                output_dir / f"table3_{method}_events.csv", index=False
            )
        summary = {
            "dataset": "D2",
            "evaluation_scope": "Table III inferred final 1,000-point window",
            "table_3_shift_times": list(table_3.table_3_shift_times),
            "estimated_lambda": result.detector.training_result.lambda_value,
            "computed": table_3.computed.reset_index().to_dict(orient="records"),
            "paper_table_3_reference": PAPER_D2_REFERENCE,
        }
    else:
        result = run_d2_experiment(config)
        summary = {
            "dataset": "D2",
            "evaluation_scope": "all testing-stream shifts",
            "estimated_lambda": result.detector.training_result.lambda_value,
            "stage_1": asdict(result.stage_1_evaluation.metrics),
            "stage_2": asdict(result.stage_2_evaluation.metrics),
            "paper_table_3_reference": PAPER_D2_REFERENCE,
        }

    result.stage_1_evaluation.events.to_csv(
        output_dir / "stage1_events.csv", index=False
    )
    result.stage_2_evaluation.events.to_csv(
        output_dir / "stage2_events.csv", index=False
    )
    result.detector.stage_1_results.to_csv(
        output_dir / "stage1_trace.csv", index=False
    )
    result.detector.stage_2_results.to_csv(
        output_dir / "stage2_validations.csv", index=False
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote D2 results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
