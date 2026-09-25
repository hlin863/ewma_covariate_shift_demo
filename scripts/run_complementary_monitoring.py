"""Run with python -m scripts.run_complementary_monitoring."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.detection.stage1.complementary import (
    fit_complementary,
    monitor,
)
from src.simulation.pca_subspace import (
    SCENARIOS,
    generate_reference,
    generate_evaluation,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--lambda-value", type=float, default=0.2)
    parser.add_argument("--magnitude", type=float, default=3.0)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--output", default="outputs/complementary"
    )
    args = parser.parse_args()

    if args.seeds < 1 or not 1 <= args.horizon <= 500:
        parser.error(
            "seeds must be positive; horizon must be from 1 to 500."
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    records, fits = [], []
    methods = {
        "score_only": "score_only_alarm",
        "residual_only": "residual_only_alarm",
        "combined": "stage_1_alarm",
    }

    for seed in range(args.seeds):
        training, calibration = generate_reference(seed)

        start = perf_counter()
        model = fit_complementary(
            training,
            calibration,
            args.alpha,
            args.lambda_value,
        )
        fit_seconds = perf_counter() - start

        fits.append({
            "seed": seed,
            "fit_seconds": fit_seconds,
            "pc1_variance_ratio":
                model.pca.explained_variance_ratio[0],
            "score_limit": model.score_limit,
            "residual_limit": model.residual_limit,
            "joint_limit": model.joint_limit,
        })

        for scenario in SCENARIOS:
            features, shift_at, delta = generate_evaluation(
                100000 + seed,
                model.pca,
                scenario,
                args.magnitude,
            )

            start = perf_counter()
            trace = monitor(features, model)
            elapsed = perf_counter() - start

            if seed == 0:
                trace.to_csv(
                    output / f"trace_{scenario}.csv",
                    index=False,
                )

            for method, column in methods.items():
                alarms = trace[column].to_numpy(dtype=bool)

                null_alarms = (
                    alarms if shift_at is None
                    else alarms[:shift_at]
                )

                detected, delay = np.nan, np.nan

                if shift_at is not None:
                    hits = np.flatnonzero(
                        alarms[
                            shift_at:shift_at + args.horizon
                        ]
                    )
                    detected = float(hits.size > 0)
                    if hits.size:
                        delay = int(hits[0])

                records.append({
                    "seed": seed,
                    "scenario": scenario,
                    "method": method,
                    "false_alarm_rate": null_alarms.mean(),
                    "detected": detected,
                    "delay": delay,
                    # Same window in the matched no-change stream.
                    "window_alarm": float(
                        alarms[500:500 + args.horizon].any()
                    ),
                    "total_alarms": int(alarms.sum()),
                    "delta_1": delta[0],
                    "delta_2": delta[1],
                    # Shared runtime, not per-method runtime.
                    "all_routes_seconds": elapsed,
                })

    results = pd.DataFrame(records)
    results.to_csv(output / "runs.csv", index=False)

    pd.DataFrame(fits).to_csv(
        output / "calibration.csv", index=False
    )

    summary = results.groupby(
        ["scenario", "method"]
    ).agg(
        mean_false_alarm_rate=("false_alarm_rate", "mean"),
        window_alarm_rate=("window_alarm", "mean"),
        detection_rate=("detected", "mean"),
        mean_delay_detected_only=("delay", "mean"),
        mean_alarm_count=("total_alarms", "mean"),
    )

    summary.to_csv(output / "summary.csv")
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    print(summary.to_string())


if __name__ == "__main__":
    main()