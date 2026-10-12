"""Run: python -m scripts.run_gpr_kernel_validation --all-scenarios."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import numpy as np
from src.adaptation.experiments.gpr_kernel import SCENARIOS, run_prepared, run_synthetic
from src.detection.stage2.gpr_kernel_hotelling import GPRKernelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=SCENARIOS, default='mean_shift')
    parser.add_argument('--all-scenarios', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--permutations', type=int, default=199)
    parser.add_argument('--features', type=Path, help='Prepared NPZ, loaded without pickle')
    parser.add_argument('--provenance', type=Path, help='Required JSON split/preprocessing metadata')
    parser.add_argument('--output', type=Path, default=Path('outputs/gpr_kernel') /
                        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = parser.parse_args()
    try:
        if args.features:
            if not args.provenance or args.all_scenarios:
                parser.error('--features requires --provenance and cannot use --all-scenarios')
            source = json.loads(args.provenance.read_text())
            required = {'dataset', 'subject', 'split', 'preprocessing', 'target_definition'}
            if not isinstance(source, dict) or not required.issubset(source):
                raise ValueError('Provenance needs dataset, subject, split, preprocessing, target_definition.')
            with np.load(args.features, allow_pickle=False) as data:
                arrays = {key: data[key] for key in ('training_features', 'training_targets',
                    'calibration_features', 'evaluation_features', 'evaluation_times')}
            result = run_prepared(**arrays, config=GPRKernelConfig(seed=args.seed, permutations=args.permutations))
            result['source'] = source
            results = [result]
        else:
            scenarios = SCENARIOS if args.all_scenarios else (args.scenario,)
            results = [run_synthetic(s, args.seed, args.permutations) for s in scenarios]
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
        manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'revision': revision,
                    'working_tree_dirty': dirty, 'command': sys.argv, 'results': results,
                    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(__file__).resolve().relative_to(Path.cwd()),
                            Path('src/adaptation/experiments/gpr_kernel.py'),
                            Path('src/detection/stage2/gpr_kernel_hotelling.py'),
                            Path('src/detection/stage2/paper_two_sample_hotelling.py'),
                            Path('src/detection/stage2/retrospective_hotelling.py'),
                            Path('src/detection/stage1/sd_ewma.py')]},
                    'versions': {name: __import__(name).__version__ for name in ('numpy', 'scipy', 'sklearn', 'pandas')}}
        args.output.mkdir(parents=True, exist_ok=False)
        path = args.output / 'validation.json'
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        print(path)
        for result in results:
            print(result['source'], result['summary'])
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
