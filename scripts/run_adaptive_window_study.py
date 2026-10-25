"""Small controlled study: fixed H versus adaptive H on identical alarm events.

Synthetic mode uses an artificially supplied alarm, NOT detected EWMA alarms.
Real-input mode accepts PCA-transformed features and saved Stage-I alarms.
Neither mode reproduces the published Table 1 by itself.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.detection.stage2.adaptive_window_hotelling import (
    AdaptiveHotellingConfig,
    validate_adaptive_window_alarms,
)
from src.detection.stage2.paper_two_sample_hotelling import (
    PaperTwoSampleHotellingConfig,
    validate_paper_two_sample_alarms,
)
from src.detection.stage2.window_feasibility import WindowSelectionPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-npy", type=Path, default=None)
    p.add_argument("--times-npy", type=Path, default=None)
    p.add_argument("--alarms-csv", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fixed-h", type=int, default=10)
    p.add_argument("--min-window", type=int, default=2)
    p.add_argument("--max-window", type=int, default=40)
    p.add_argument("--target-df2", type=int, default=10)
    p.add_argument("--max-condition-number", type=float, default=1e6)
    p.add_argument("--require-full-rank-empirical", action="store_true")
    p.add_argument(
        "--covariance-method", choices=["empirical", "shrinkage"], default="shrinkage"
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--alarm-gap", type=int, default=10)
    p.add_argument(
        "--output-root", type=Path, default=Path("outputs/experiments/adaptive_window")
    )
    return p.parse_args()


def load_case(args: argparse.Namespace):
    supplied = (args.features_npy is not None, args.alarms_csv is not None)
    if supplied.count(True) == 1:
        raise ValueError("Supply BOTH --features-npy and --alarms-csv, or neither")
    if all(supplied):
        features = np.load(args.features_npy)
        times = np.load(args.times_npy) if args.times_npy else np.arange(len(features))
        warnings = pd.read_csv(args.alarms_csv)
        if not {"time", "stage_1_alarm"}.issubset(warnings.columns):
            raise ValueError("alarms CSV must have time,stage_1_alarm")
        return features, times, warnings, "supplied_pca_features_and_alarms"
    if args.times_npy:
        raise ValueError("--times-npy requires --features-npy and --alarms-csv")
    rng = np.random.default_rng(args.seed)
    split, n, d = 80, 160, 3
    features = rng.normal(0.0, 0.4, size=(n, d))
    features[split:] += 2.0
    times = np.arange(n)
    warnings = pd.DataFrame(
        {"time": times, "stage_1_alarm": (times == split - 1).astype(int)}
    )
    return features, times, warnings, "synthetic_manual_warning_not_EWMA"


def summarise(mode: str, results: pd.DataFrame, n_warnings: int) -> dict:
    evaluated = results[results["status"].isin(["confirmed", "rejected"])]
    return dict(
        method=mode,
        stage1_warnings=n_warnings,
        stage2_tested=len(evaluated),
        confirmed=int(results["confirmed_shift"].fillna(False).astype(bool).sum()),
        pending=int(results["status"].eq("pending_current_window").sum()),
        mean_window_size=(
            float(evaluated["sample_size"].mean()) if len(evaluated) else None
        ),
        statuses=json.dumps(results["status"].value_counts().to_dict()),
    )


def main() -> None:
    args = parse_args()
    features, times, warnings, source = load_case(args)
    n_warnings = int(warnings["stage_1_alarm"].fillna(False).astype(bool).sum())
    policy = WindowSelectionPolicy(
        min_window=args.min_window,
        max_window=args.max_window,
        target_df2=args.target_df2,
        covariance_method=args.covariance_method,
        max_condition_number=args.max_condition_number,
        require_full_rank_empirical=args.require_full_rank_empirical,
    )
    fixed = validate_paper_two_sample_alarms(
        features,
        times,
        warnings,
        PaperTwoSampleHotellingConfig(
            window_size=args.fixed_h,
            alpha=args.alpha,
            covariance_method=args.covariance_method,
            minimum_alarm_gap=args.alarm_gap,
        ),
    )
    adaptive = validate_adaptive_window_alarms(
        features,
        times,
        warnings,
        AdaptiveHotellingConfig(
            window_policy=policy,
            alpha=args.alpha,
            minimum_alarm_gap=args.alarm_gap,
        ),
    )
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    )
    dest = args.output_root / run_id
    dest.mkdir(parents=True, exist_ok=False)
    fixed.to_csv(dest / "fixed_events.csv", index=False)
    adaptive.to_csv(dest / "adaptive_events.csv", index=False)
    warnings.to_csv(dest / "stage1_alarm_input.csv", index=False)
    comparison = pd.DataFrame(
        [
            summarise("fixed", fixed, n_warnings),
            summarise("adaptive", adaptive, n_warnings),
        ]
    )
    comparison.to_csv(dest / "comparison.csv", index=False)
    manifest = {
        "run_id": run_id,
        "source": source,
        "note": "Diagnostic comparison, not Table 1 paper reproduction",
        "feature_shape": list(np.shape(features)),
        "fixed_h": args.fixed_h,
        "adaptive_policy": policy.__dict__,
        "alpha": args.alpha,
        "minimum_alarm_gap": args.alarm_gap,
        "seed": args.seed if source.startswith("synthetic") else None,
        "features_file": str(args.features_npy) if args.features_npy else None,
        "alarms_file": str(args.alarms_csv) if args.alarms_csv else None,
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8"
    )
    print(comparison.to_string(index=False))
    print("Study saved to:", dest)


if __name__ == "__main__":
    main()
