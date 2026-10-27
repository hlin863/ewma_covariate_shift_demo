"""Saved BCI figure regression tests."""
import json
import pandas as pd
from src.reporting.bci_comparison_figures import build_bci_comparison_figures


def test_existing_metrics_generate_png_svg_without_reconstruction(tmp_path):
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame([
        {"dataset": "2A", "subject": "A01", "published_csw": 12,
         "computed_csw": 4, "published_csv": 6, "computed_csv": 0},
        {"dataset": "2B", "subject": "B01", "published_csw": 14,
         "computed_csw": 17, "published_csv": 10, "computed_csv": 0},
    ]).to_csv(metrics, index=False)
    output = tmp_path / "figures"
    paths = build_bci_comparison_figures(
        metrics, tmp_path / "not_generated.json", output
    )
    assert len(paths) == 2
    assert (output / "bci_previous_method_comparison.png").stat().st_size > 0
    assert (output / "bci_previous_method_comparison.svg").stat().st_size > 0
    assert not (output / "bci_reconstruction_comparison.png").exists()


def test_reconstruction_results_generate_additional_backups(tmp_path):
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame([
        {"dataset": "2A", "subject": "A01", "published_csw": 12,
         "computed_csw": 4, "published_csv": 6, "computed_csv": 0},
    ]).to_csv(metrics, index=False)
    reconstruction = tmp_path / "manifest.json"
    reconstruction.write_text(json.dumps({"subjects": [
        {"dataset": "2A", "subject": 1,
         "mean_training_loss": 0.03, "mean_evaluation_loss": 0.09}
    ]}), encoding="utf-8")
    output = tmp_path / "figures"
    paths = build_bci_comparison_figures(metrics, reconstruction, output)
    assert len(paths) == 5
    assert (output / "bci_reconstruction_comparison.png").stat().st_size > 0
    assert (output / "bci_reconstruction_comparison.svg").stat().st_size > 0
    frame = pd.read_csv(output / "bci_reconstruction_comparison.csv")
    assert abs(frame.iloc[0]["delta_mse"] - 0.06) < 1e-12
