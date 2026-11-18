"""Export reproducible slide-ready KL figures from a *real* BCI event payload.

No EEG estimation happens here: upstream detector/diagnostic code owns the data.
Missing or ineligible windows never become invented zero-divergence points.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURE_NAMES = (
    "01_kl_event_magnitude",
    "02_kl_per_pca_component",
    "03_kl_before_after_probabilities",
    "04_kl_slide12_composite",
)


def _save(fig, base: Path) -> list[str]:
    paths = []
    for extension in ("png", "svg"):
        output = base.with_suffix("." + extension)
        fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
        paths.append(output.name)
    plt.close(fig)
    return paths


def _plot_timeline(ax, events, chosen):
    eligible = [e for e in events if e["kl_status"] == "diagnostic_only"]
    xs = [int(e["alarm_index"]) for e in eligible]
    ax.scatter(xs, [e["forward"] for e in eligible], s=55, color="#2C6E9F",
               label=r"Before $\Vert$ after", zorder=3)
    ax.scatter(xs, [e["reverse"] for e in eligible], s=55, color="#CD4D58",
               label=r"After $\Vert$ before", zorder=3)
    ax.axvline(chosen["alarm_index"], linestyle="--", linewidth=1.1, color="#60738a",
               alpha=.7, label="Selected warning")
    ax.set_xlabel("EWMA warning trial index (not confirmation time)")
    ax.set_ylabel("Mean marginal KL")
    ax.set_ylim(bottom=0)
    ax.set_xlim(min(xs) - 1, max(xs) + 1 if max(xs) != min(xs) else max(xs) + 2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3,
              frameon=False, fontsize=9)
    ax.grid(alpha=.16)


def _plot_components(ax, selected, feature_names):
    values = selected["per_feature"]
    ax.bar(feature_names, values, color="#2C6E9F", width=.55)
    ax.set_ylabel("Forward marginal KL")
    ax.set_xlabel("Retained PCA component")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.16)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)


def _plot_probabilities(ax, selected, component):
    histogram = selected["histograms"][component]
    p, q = histogram["reference"], histogram["current"]
    x = np.arange(1, len(p) + 1)
    ax.bar(x-.2, p, width=.4, label="Before", color="#2C6E9F")
    ax.bar(x+.2, q, width=.4, label="After", color="#CD4D58")
    ax.set_xticks(x)
    ax.set_xlabel("Shared calibration-fitted histogram bin")
    ax.set_ylabel("Smoothed bin probability")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.16)


def export_kl_figures(
    kl_payload: dict,
    destination: str | Path,
    *,
    dataset: str,
    subject: str,
    detector_config: dict,
    input_sources: list[str],
    calibration_count: int,
    evaluation_count: int,
    seed: int,
    code_revision: str = "unrecorded",
    selected_alarm: int | None = None,
) -> dict:
    """Save PNG/SVG + machine-readable evidence under a subject-specific folder.

    Plot A: all completed eligible warning windows.
    Plot B/C: the same selected event, with PC giving largest forward KL.
    Plot D: slide 12-ready panel composition from the identical event.
    """
    if dataset not in {"2a", "2b"}:
        raise ValueError("dataset must be '2a' or '2b'")
    if not subject.startswith("A" if dataset == "2a" else "B"):
        raise ValueError("subject must match dataset")
    if not kl_payload or "events" not in kl_payload:
        raise ValueError("real run KL event evidence required; no synthetic fallback")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    events = kl_payload["events"]
    eligible = [e for e in events if e["kl_status"] == "diagnostic_only"]
    feature_names = kl_payload["feature_names"]
    selected = None
    if selected_alarm is not None:
        selected = next((e for e in eligible if e["alarm_index"] == selected_alarm), None)
        if selected is None:
            raise ValueError("Selected alarm must identify an eligible KL event")
    elif eligible:
        selected = max(eligible, key=lambda e: e["forward"])
    selected_component = (
        int(np.argmax(selected["per_feature"])) if selected is not None else None
    )
    event_columns = [
        "alarm_index", "validation_index", "hotelling_status", "kl_status",
        "forward", "reverse", "reference_start", "reference_end",
        "current_start", "current_end",
    ]
    rows = []
    for e in events:
        ref = e.get("reference_window") or [None, None]
        cur = e.get("current_window") or [None, None]
        rows.append(dict(zip(event_columns, [
            e["alarm_index"], e.get("validation_index"), e["hotelling_status"],
            e["kl_status"], e.get("forward"), e.get("reverse"),
            *ref, *cur,
        ])))
    with (target / "kl_events.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=event_columns)
        writer.writeheader()
        writer.writerows(rows)
    (target / "kl_events.json").write_text(
        json.dumps({"events": events, "feature_names": feature_names},
                   indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    files = ["kl_events.csv", "kl_events.json"]
    if selected is not None:
        for title, name, plot in [
            ("Event-level KL magnitude", FIGURE_NAMES[0],
             lambda a: _plot_timeline(a, events, selected)),
            ("PCA component contributions", FIGURE_NAMES[1],
             lambda a: _plot_components(a, selected, feature_names)),
            ("Before / after PCA distributions", FIGURE_NAMES[2],
             lambda a: _plot_probabilities(a, selected, selected_component)),
        ]:
            fig, ax = plt.subplots(figsize=(11, 4.7))
            fig.suptitle(f"{dataset.upper()} · {subject} · {title}",
                         x=.12, ha="left", fontweight="bold", fontsize=16)
            plot(ax)
            fig.subplots_adjust(top=.78)
            files += _save(fig, target / name)

        fig = plt.figure(figsize=(14, 8), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0])
        ax0 = fig.add_subplot(grid[0, :])
        ax1 = fig.add_subplot(grid[1, 0])
        ax2 = fig.add_subplot(grid[1, 1])
        _plot_timeline(ax0, events, selected)
        _plot_components(ax1, selected, feature_names)
        _plot_probabilities(ax2, selected, selected_component)
        fig.suptitle(f"{dataset.upper()} · {subject} | KL divergence around EWMA warnings",
                     ha="left", x=.06, fontsize=17, fontweight="bold")
        fig.text(.06, .005, (
            f"Selected warning {selected['alarm_index']} · KL available at trial "
            f"{selected.get('validation_index')} · Hotelling {selected['hotelling_status']} "
            f"· H={detector_config.get('window_size')} · PCA={detector_config.get('pca_components')} "
            f"· {kl_payload.get('n_bins')} bins · pseudocount {kl_payload.get('pseudocount')}"
        ), fontsize=9, color="#4b5563")
        files += _save(fig, target / FIGURE_NAMES[3])

    identity = {
        "dataset": dataset, "subject": subject, "detector_config": detector_config,
        "kl_method": kl_payload.get("method"), "kl_bins": kl_payload.get("n_bins"),
        "kl_pseudocount": kl_payload.get("pseudocount"),
        "input_sources": input_sources, "calibration_count": calibration_count,
        "evaluation_count": evaluation_count, "seed": seed,
        "code_revision": code_revision,
    }
    run_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    manifest = {
        **identity, "run_id": run_key, "event_count": len(events),
        "eligible_count": len(eligible),
        "selected_alarm": selected["alarm_index"] if selected else None,
        "selected_validation_index": selected.get("validation_index") if selected else None,
        "selected_hotelling_status": selected["hotelling_status"] if selected else None,
        "selected_component": (feature_names[selected_component]
                               if selected_component is not None else None),
        "status": "figures_generated" if selected is not None else "no_eligible_windows",
        "artifacts": files,
        "interpretation": (
            "Unsupervised descriptive mean marginal histogram KL across retained PCA "
            "features; Stage-II Hotelling decisions unchanged. No calibrated KL "
            "significance test, no SVM adaptation claim, no published-count resolution."
        ),
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest
