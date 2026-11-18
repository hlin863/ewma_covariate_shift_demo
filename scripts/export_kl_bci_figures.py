"""Export evidence-grounded KL Slide-12 PNG/SVG figures for BCI Competition IV.

Reuses existing 2A EDA FBCSP and 2B Raza-2019 development/evaluation paths.
Requires locally installed official GDF files; NEVER substitutes synthetic BCI data.

Examples:
 python -m scripts.export_kl_bci_figures --dataset 2a --subject 1 \
   --data-2a data/raw/bci_competition_iv_2a \
   --labels-2a data/raw/bci_competition_iv_2a_labels
 python -m scripts.export_kl_bci_figures --dataset 2b --subject 1 \
   --data-2b data/raw/bci_competition_iv_2b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.kl_figures import export_kl_figures
from src.web.bci_shift_views import detection_event_payload


def _revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unrecorded"


def _source_snapshot(sources: list[Path]) -> list[dict]:
    snapshot = []
    for p in sources:
        if not p.is_file():
            raise FileNotFoundError(f"Required real BCI input missing: {p}")
        stats = p.stat()
        snapshot.append({"path": str(p.resolve()), "size_bytes": stats.st_size,
                         "mtime_ns": stats.st_mtime_ns})
    return snapshot


def _run_2a(subject: int, args):
    from flask import Flask
    from src.bci.data import dataset_2a_filename
    from src.web.bci_eda import _load_dataset_2a_feature_timelines

    paths = [
        args.data_2a / dataset_2a_filename(subject, "T"),
        args.data_2a / dataset_2a_filename(subject, "E"),
        args.labels_2a / f"A{subject:02d}E.mat",
    ]
    source = _source_snapshot(paths)
    app = Flask(__name__)
    app.config.update(
        DATASET_2A_PATH=str(args.data_2a),
        DATASET_2A_LABELS_PATH=str(args.labels_2a),
    )
    with app.app_context():
        payload, reason = _load_dataset_2a_feature_timelines(subject)
    if payload is None:
        raise RuntimeError(f"Unable to derive real Dataset 2A feature evidence: {reason}")
    return payload["cse_events"], source, int(payload["development_count"]), (
        int(payload["evaluation_count"])
    ), int(payload["seed"])


def _run_2b(subject: int, args):
    from src.bci.data import dataset_2b_filename, load_bci_competition_iv_2b_session
    from src.bci.datasets.dataset2b.development_split import prepare_cse_uael_2019_protocol
    from src.bci.datasets.dataset2b.development_pipeline import (
        build_dataset_2b_development_fbcsp_features,
    )
    from src.bci.datasets.dataset2b.experiment import extract_dataset_2b_trials

    paths = [args.data_2b / dataset_2b_filename(subject, session)
             for session in range(1, 6)]
    source = _source_snapshot(paths)
    sessions = [
        extract_dataset_2b_trials(
            load_bci_competition_iv_2b_session(args.data_2b, subject, session)
        )
        for session in range(1, 6)
    ]
    protocol = prepare_cse_uael_2019_protocol(
        sessions, validation_fraction=0.30, random_state=42,
    )
    pipeline = build_dataset_2b_development_fbcsp_features(
        protocol.development, protocol.evaluation,
    )
    # Unlike 2A, the concatenated 2B trial index is not physical cue-onset
    # seconds. Omit onset metadata rather than display fabricated timestamps.
    events = detection_event_payload(
        pipeline.training.features, pipeline.evaluation.features, None,
    )
    return events, source, len(pipeline.training.features), (
        len(pipeline.evaluation.features)
    ), 42


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["2a", "2b"], required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--selected-alarm", type=int, default=None,
                        help="Use a specific eligible warning; defaults to highest forward KL")
    parser.add_argument("--data-2a", type=Path,
                        default=ROOT / "data/raw/bci_competition_iv_2a")
    parser.add_argument("--labels-2a", type=Path,
                        default=ROOT / "data/raw/bci_competition_iv_2a_labels")
    parser.add_argument("--data-2b", type=Path,
                        default=ROOT / "data/raw/bci_competition_iv_2b")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs/experiments/kl_stage2")
    args = parser.parse_args(argv)
    if not 1 <= args.subject <= 9:
        parser.error("--subject must be 1..9")

    if args.dataset == "2a":
        events, inputs, n_train, n_eval, seed = _run_2a(args.subject, args)
    else:
        events, inputs, n_train, n_eval, seed = _run_2b(args.subject, args)

    subject = f"{'A' if args.dataset == '2a' else 'B'}{args.subject:02d}"
    version = _revision()
    run_spec = {
        "dataset": args.dataset, "subject": subject, "inputs": inputs,
        "config": events["config"], "kl_config": {
            "bins": events["kl_diagnostics"]["n_bins"],
            "pseudocount": events["kl_diagnostics"]["pseudocount"],
        },
        "revision": version, "selected_alarm": args.selected_alarm,
    }
    run_directory = hashlib.sha256(
        json.dumps(run_spec, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    destination = args.output_root / args.dataset / subject / run_directory
    manifest = export_kl_figures(
        events["kl_diagnostics"], destination,
        dataset=args.dataset, subject=subject,
        detector_config=events["config"],
        input_sources=[f"{x['path']} (bytes={x['size_bytes']}, mtime_ns={x['mtime_ns']})"
                       for x in inputs],
        calibration_count=n_train, evaluation_count=n_eval, seed=seed,
        code_revision=version, selected_alarm=args.selected_alarm,
    )
    print(json.dumps({"directory": str(destination),
                      "status": manifest["status"],
                      "figures": [x for x in manifest["artifacts"]
                                  if x.endswith(".png")],
                      "manifest": str(destination / "manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
