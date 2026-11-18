"""Figure-export contract: tests use synthetic fixtures, never label them real EEG."""
import json

import numpy as np
import pytest

from src.reporting.kl_figures import FIGURE_NAMES, export_kl_figures
from src.web.bci_shift_views import detection_event_payload


def synthetic_kl_payload():
    rng = np.random.default_rng(1701)
    train = rng.normal(size=(100, 5))
    evaluation = rng.normal(size=(140, 5))
    evaluation[50:] += 3
    evidence = detection_event_payload(
        train, evaluation, 2 + 5*np.arange(len(evaluation)),
    )
    return evidence


def test_exports_four_figures_and_exact_provenance(tmp_path):
    evidence = synthetic_kl_payload()
    kl = evidence["kl_diagnostics"]
    if not kl["eligible_count"]:
        pytest.skip("detector fixture has no eligible warnings under its config")
    dest = tmp_path / "experiments" / "kl_stage2" / "2a" / "A01" / "test-run"
    manifest = export_kl_figures(
        kl, dest, dataset="2a", subject="A01",
        detector_config=evidence["config"], input_sources=["synthetic-test-fixture"],
        calibration_count=100, evaluation_count=140, seed=1701,
        code_revision="test-fixture",
    )
    assert manifest["eligible_count"] > 0
    assert manifest["selected_hotelling_status"] in {"confirmed", "rejected"}
    assert manifest["selected_alarm"] is not None
    for name in FIGURE_NAMES:
        for ext in ("png", "svg"):
            image = dest / f"{name}.{ext}"
            assert image.is_file() and image.stat().st_size > 500
    assert (dest / "kl_events.csv").is_file()
    saved = json.loads((dest / "manifest.json").read_text())
    assert saved["code_revision"] == "test-fixture"
    assert saved["kl_bins"] == 8
    assert saved["status"] == "figures_generated"
    assert "no calibrated KL" in saved["interpretation"].lower() or "no calibrated" in saved["interpretation"].lower()


def test_does_not_fabricate_figures_for_ineligible_windows(tmp_path):
    dest = tmp_path / "no-evidence"
    empty = {"events": [{"alarm_index": 11, "validation_index": None,
                         "hotelling_status": "pending_current_window",
                         "kl_status": "ineligible_pending_current_window",
                         "forward": None, "reverse": None, "per_feature": [],
                         "histograms": []}],
             "feature_names": ["PC1", "PC2"], "n_bins": 8,
             "pseudocount": 0.5, "method": "marginal histogram KL"}
    result = export_kl_figures(
        empty, dest, dataset="2a", subject="A01",
        detector_config={"window_size": 10, "pca_components": 2},
        input_sources=["synthetic-test-fixture"], calibration_count=60,
        evaluation_count=40, seed=2,
    )
    assert result["status"] == "no_eligible_windows"
    assert not list(dest.glob("*.png"))
    assert (dest / "kl_events.csv").exists()
    assert (dest / "manifest.json").exists()


def test_rejects_invalid_subject_and_unverified_alarm(tmp_path):
    payload = synthetic_kl_payload()["kl_diagnostics"]
    args = dict(dataset="2a", subject="B01",
                detector_config={}, input_sources=[], calibration_count=80,
                evaluation_count=140, seed=2)
    with pytest.raises(ValueError, match="subject"):
        export_kl_figures(payload, tmp_path, **args)
    with pytest.raises(ValueError, match="Selected alarm"):
        export_kl_figures(payload, tmp_path, dataset="2a", subject="A01",
                          detector_config={}, input_sources=[], calibration_count=80,
                          evaluation_count=140, seed=2, selected_alarm=-9999)
