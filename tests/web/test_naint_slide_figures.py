"""Regression checks for provenance-aware, presentation-ready figure export."""
from pathlib import Path
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from scripts.export_naint_slide_figures import export_2a_payload, export_d2_saved


def _payload():
    x = np.arange(30, dtype=float)
    return {
        "timelines": [{
            "name": "8_12Hz_csp_high_1",
            "development": [{"trial_index": int(i), "feature_value": float(v)}
                            for i, v in enumerate(x[:15])],
            "evaluation": [{"trial_index": int(i), "feature_value": float(v)}
                           for i, v in enumerate(x)],
        }],
        "heatmap": {
            "feature_names": ["f1", "f2"],
            "windows": [
                {"values": [0.4, -0.7]},
                {"values": [1.3, 0.1]},
                {"values": [-0.5, 0.2]},
            ],
        },
        "cse_events": {
            "config": {"validation_mode": "paper_two_sample", "window_size": 10},
            "warnings": [
                {"index": int(i), "feature_value": float(np.sin(i / 5)),
                 "prediction": 0.0, "lcl": -0.7, "ucl": 0.7,
                 "warning": bool(i == 12)}
                for i in range(30)
            ],
            "validations": [
                {"alarm_index": 12, "validation_index": 22,
                 "status": "confirmed", "confirmed": True},
                {"alarm_index": 23, "validation_index": None,
                 "status": "pending_current_window", "confirmed": False},
            ],
        },
        "development_count": 15, "evaluation_count": 30, "seed": 42,
    }


def test_2a_exports_four_figures_with_provenance(tmp_path: Path):
    destination = tmp_path / "A01"
    paths = export_2a_payload(_payload(), destination, "A01")
    assert len(paths) == 8
    assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths)
    manifest = (destination / "manifest.json").read_text(encoding="utf-8")
    assert '"experiment_type": "real_eeg_exploratory"' in manifest
    assert '"window_size": 10' in manifest


def test_2a_rejects_incomplete_trace_instead_of_inventing_values(tmp_path: Path):
    payload = _payload()
    payload["cse_events"]["warnings"][0]["lcl"] = None
    with pytest.raises(ValueError, match="lacks recorded"):
        export_2a_payload(payload, tmp_path, "A01")


def test_d2_export_preserves_synthetic_identity_and_completion(tmp_path: Path):
    source = tmp_path / "saved"
    source.mkdir()
    pd.DataFrame({
        "time": [501, 502, 503, 504, 505],
        "x": [.1, .2, 2., 1.5, .3],
        "lcl": [-1.] * 5, "ucl": [1.] * 5,
        "stage_1_alarm": [0, 0, 1, 1, 0],
    }).to_csv(source / "stage1_trace.csv", index=False)
    pd.DataFrame({
        "alarm_time": [503, 504],
        "validation_time": [505, None],
        "status": ["confirmed", "skipped_nearby_alarm"],
    }).to_csv(source / "stage2_validations.csv", index=False)
    destination = tmp_path / "exports"
    paths = export_d2_saved(destination, source)
    assert len(paths) == 2
    assert all(Path(path).is_file() for path in paths)
    assert "synthetic_d2_archived" in (destination / "manifest.json").read_text(encoding="utf-8")


def test_d2_missing_sources_fail_without_placeholder_figures(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        export_d2_saved(tmp_path / "out", tmp_path / "missing")
    assert not (tmp_path / "out").exists()



def test_direct_script_invocation_finds_repository_src(tmp_path: Path):
    """Reproduce the Windows-style direct command without relying on PYTHONPATH."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "export_naint_slide_figures.py"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable, str(script), "--source", "2a", "--subject", "1",
            "--data-2a", str(tmp_path / "missing_gdf"),
            "--labels-2a", str(tmp_path / "missing_labels"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    # The data are intentionally absent: the expected failure is a clear
    # input-data error, not ModuleNotFoundError for the repository src package.
    assert result.returncode != 0
    assert "Official Session-II MAT labels are required" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
