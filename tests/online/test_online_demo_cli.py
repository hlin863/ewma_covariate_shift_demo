"""Exercise the actual command-line entry point without external BCI files."""
import sys

from scripts.run_online_demo import main


def test_synthetic_online_cli_creates_auditable_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_online_demo.py", "--scenario", "feature_shift",
        "--policy", "never", "--output", str(tmp_path),
    ])
    main()
    target = tmp_path / "feature_shift" / "never"
    assert (target / "trials.csv").is_file()
    assert (target / "stage2.csv").is_file()
    assert (target / "updates.csv").is_file()
    assert (target / "summary.csv").is_file()
