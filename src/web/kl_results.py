"""Saved BCI KL research figures, separated from live EDA recomputation."""
from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, current_app, render_template, request

kl_results_bp = Blueprint("kl_results", __name__)


def discover_saved_kl_runs(outputs_root: str | Path, *,
                           dataset: str | None = None, subject: str | None = None) -> list[dict]:
    """Return only real, manifest-backed BCI KL runs from canonical output folders."""
    root = Path(outputs_root)
    if dataset not in (None, "2a", "2b"):
        raise ValueError("unknown dataset")
    if subject is not None and (len(subject) != 3 or subject[0] not in "AB"
                                or not subject[1:].isdigit()
                                or not 1 <= int(subject[1:]) <= 9):
        raise ValueError("invalid BCI subject")
    found = []
    for manifest_path in (root / "experiments" / "kl_stage2").glob("*/*/*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ds, sub = str(manifest.get("dataset")), str(manifest.get("subject"))
        if ds not in {"2a", "2b"} or sub[:1] != ("A" if ds == "2a" else "B"):
            continue
        if ds != manifest_path.parent.parent.parent.name or sub != manifest_path.parent.parent.name:
            continue
        if dataset is not None and ds != dataset:
            continue
        if subject is not None and sub != subject:
            continue
        artifacts = []
        for file_name in manifest.get("artifacts", []):
            if (not isinstance(file_name, str)
                    or Path(file_name).name != file_name
                    or Path(file_name).suffix.lower() not in {".png", ".svg", ".csv", ".json"}
                    or not (manifest_path.parent / file_name).is_file()):
                continue
            artifacts.append({
                "name": file_name,
                "relative_path": str((manifest_path.parent / file_name).relative_to(root))
                    .replace("\\", "/"),
                "kind": Path(file_name).suffix.lower().lstrip("."),
            })
        found.append({
            "dataset": ds, "subject": sub,
            "run_directory": manifest_path.parent.name,
            "run_id": manifest.get("run_id"),
            "status": manifest.get("status", "unknown"),
            "eligible_count": manifest.get("eligible_count", 0),
            "event_count": manifest.get("event_count", 0),
            "selected_alarm": manifest.get("selected_alarm"),
            "selected_component": manifest.get("selected_component"),
            "selected_hotelling_status": manifest.get("selected_hotelling_status"),
            "detector_config": manifest.get("detector_config", {}),
            "kl_bins": manifest.get("kl_bins"),
            "kl_pseudocount": manifest.get("kl_pseudocount"),
            "code_revision": manifest.get("code_revision"),
            "artifacts": artifacts,
            "mtime": manifest_path.stat().st_mtime,
        })
    return sorted(found, key=lambda x: x["mtime"], reverse=True)


@kl_results_bp.get("/results/kl-stage2")
def results():
    dataset = request.args.get("dataset", "").lower()
    subject_num = request.args.get("subject", "")
    if dataset not in {"2a", "2b"}:
        dataset = ""
    if not subject_num.isdigit() or not 1 <= int(subject_num) <= 9:
        subject_num = ""
    subject = ((("A" if dataset == "2a" else "B") + f"{int(subject_num):02d}")
               if dataset and subject_num else None)
    runs = discover_saved_kl_runs(
        current_app.config["RESULTS_ROOT"],
        dataset=dataset or None, subject=subject,
    )
    return render_template(
        "kl_results.html",
        runs=runs,
        dataset=dataset,
        subject_num=subject_num,
        count=len(runs),
    )
