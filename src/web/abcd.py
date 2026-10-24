"""ABCD equation-level learning page; no external downloads or EEG uploads."""
from dataclasses import asdict

from flask import Blueprint, render_template, request
from src.experiments.abcd_lab import ABCDLabConfig, run_abcd_lab

abcd_bp = Blueprint("abcd", __name__)


@abcd_bp.route("/results/abcd", methods=["GET", "POST"])
def laboratory():
    values = asdict(ABCDLabConfig())
    result = error = None
    if request.method == "POST":
        try:
            values["scenario"] = request.form.get("scenario", values["scenario"])
            for key in (
                "seed", "segment_size", "dimensions", "changed_dimensions",
                "bottleneck", "candidate_count",
            ):
                values[key] = int(request.form.get(key, values[key]))
            for key in ("shift", "M", "delta"):
                values[key] = float(request.form.get(key, values[key]))
            result = run_abcd_lab(ABCDLabConfig(**values))
        except (ValueError, OverflowError, TypeError, FloatingPointError) as exc:
            error = str(exc)
    return render_template(
        "abcd.html", values=values, result=result, error=error
    ), (400 if error else 200)
