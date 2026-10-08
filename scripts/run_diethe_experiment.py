"""Run with python -m scripts.run_diethe_experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from src.adaptation.experiments.diethe import (
    DietheConfig, run_synthetic, run_policy_comparison, export_experiment,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['none', 'feature_shift', 'relationship_shift'], default='feature_shift')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trials', type=int, default=240)
    parser.add_argument('--magnitude', type=float, default=2.0)
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--performance-window', type=int, default=30)
    parser.add_argument('--accuracy-drop', type=float, default=0.10)
    parser.add_argument('--update-scope', choices=['current_trial', 'since_last_update'], default='since_last_update')
    parser.add_argument('--evidence-samples', type=int, default=10)
    parser.add_argument('--evidence-sample-cost', type=float, default=0.001)
    parser.add_argument('--evidence-delay-cost', type=float, default=0.001)
    parser.add_argument('--evidence-consequence-weight', type=float, default=1.0)
    parser.add_argument('--features', type=Path, help='NPZ of prepared training/validation/evaluation arrays; no pickle')
    parser.add_argument('--provenance', type=Path, help='JSON dataset, subject, split and preprocessing metadata required for NPZ')
    parser.add_argument('--output', type=Path, default=Path('outputs/diethe') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = parser.parse_args()
    try:
        config = DietheConfig(**{key: getattr(args, key) for key in DietheConfig.__dataclass_fields__})
        if args.features:
            if not args.provenance:
                parser.error('--features requires --provenance')
            metadata = json.loads(args.provenance.read_text(encoding='utf-8'))
            if not isinstance(metadata, dict) or not metadata:
                raise ValueError('Provenance must be a nonempty JSON object.')
            keys = ['training_features', 'training_labels', 'validation_features', 'validation_labels',
                    'evaluation_features', 'evaluation_labels', 'evaluation_times']
            with np.load(args.features, allow_pickle=False) as data:
                arrays = {key: data[key] for key in keys}
            result = run_policy_comparison(**arrays, config=config,
                source=f'prepared features: {args.features.name}', provenance=metadata)
        else:
            result = run_synthetic(config)
        print(export_experiment(result, args.output))
        for row in result['summary']:
            print(f"{row['policy']}: accuracy={row['accuracy']:.3f}, updates={row['update_count']}, retrain_seconds={row['retrain_seconds']:.4f}")
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
