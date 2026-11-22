"""Populate manifest-backed KL Stage-II output folders for selected real BCI subjects.

This is an orchestration wrapper around export_kl_bci_figures; it never creates
synthetic BCI observations and leaves existing detector/reproduction code intact.

Example:
  python -m scripts.export_kl_bci_batch --dataset both --subjects 1 2
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import sys

from scripts.export_kl_bci_figures import ROOT, main as export_subject


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("2a", "2b", "both"), default="both")
    parser.add_argument("--subjects", nargs="+", type=int, default=[1],
                        help="One or more BCI subject numbers from 1 to 9")
    parser.add_argument("--data-2a", type=Path,
                        default=ROOT / "data/raw/bci_competition_iv_2a")
    parser.add_argument("--labels-2a", type=Path,
                        default=ROOT / "data/raw/bci_competition_iv_2a_labels")
    parser.add_argument("--data-2b", type=Path,
                        default=ROOT / "data/raw/bci_competition_iv_2b")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs/experiments/kl_stage2")
    args = parser.parse_args(argv)
    if any(not 1 <= n <= 9 for n in args.subjects):
        parser.error("--subjects must be in 1..9")
    if len(set(args.subjects)) != len(args.subjects):
        parser.error("--subjects must not contain duplicates")

    datasets = ("2a", "2b") if args.dataset == "both" else (args.dataset,)
    reports = []
    for dataset in datasets:
        for number in args.subjects:
            subject = f"{'A' if dataset == '2a' else 'B'}{number:02d}"
            subject_args = [
                "--dataset", dataset, "--subject", str(number),
                "--data-2a", str(args.data_2a),
                "--labels-2a", str(args.labels_2a),
                "--data-2b", str(args.data_2b),
                "--output-root", str(args.output_root),
            ]
            captured = StringIO()
            try:
                with redirect_stdout(captured):
                    code = export_subject(subject_args)
                if code != 0:
                    raise RuntimeError(f"Subject exporter returned exit code {code}")
                payload = json.loads(captured.getvalue())
                reports.append({
                    "dataset": dataset, "subject": subject,
                    "status": payload["status"],
                    "directory": payload["directory"],
                    "manifest": payload["manifest"],
                    "figures": payload["figures"],
                })
            except (FileNotFoundError, RuntimeError, ValueError, KeyError,
                    json.JSONDecodeError) as error:
                reports.append({
                    "dataset": dataset, "subject": subject,
                    "status": "failed",
                    "reason": str(error),
                    "figures": [],
                })

    args.output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = args.output_root / f"batch_report_{timestamp}.json"
    report = {
        "purpose": "Real BCI KL Stage-II figure-export batch; no synthetic fallback",
        "requested_dataset": args.dataset,
        "requested_subjects": args.subjects,
        "runs": reports,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        "report": str(report_path),
        "figures_generated": sum(r["status"] == "figures_generated" for r in reports),
        "no_eligible_windows": sum(r["status"] == "no_eligible_windows" for r in reports),
        "failed": sum(r["status"] == "failed" for r in reports),
    }
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
