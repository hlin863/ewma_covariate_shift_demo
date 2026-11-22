"""Batch-export orchestration tests; no real or simulated EEG result claims."""
import json
from pathlib import Path

import pytest

from scripts import export_kl_bci_batch as batch


def test_batch_exports_subjects_to_canonical_paths(tmp_path, monkeypatch, capsys):
    seen = []

    def fake_export(args):
        seen.append(args)
        dataset = args[args.index("--dataset") + 1]
        subject = args[args.index("--subject") + 1]
        code = f"{'A' if dataset == '2a' else 'B'}{int(subject):02d}"
        dest = tmp_path / dataset / code / "fake-run"
        print(json.dumps({
            "directory": str(dest),
            "status": "figures_generated",
            "figures": [str(dest / "04_kl_slide12_composite.png")],
            "manifest": str(dest / "manifest.json"),
        }))
        return 0

    monkeypatch.setattr(batch, "export_subject", fake_export)
    assert batch.main(["--dataset", "both", "--subjects", "1", "2",
                       "--output-root", str(tmp_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    report = json.loads(Path(summary["report"]).read_text())
    assert summary["figures_generated"] == 4
    assert [(r["dataset"], r["subject"]) for r in report["runs"]] == [
        ("2a", "A01"), ("2a", "A02"), ("2b", "B01"), ("2b", "B02")]
    assert len(seen) == 4
    assert all("--output-root" in args for args in seen)


def test_missing_real_inputs_reported_without_figures(tmp_path, monkeypatch, capsys):
    def missing_inputs(args):
        raise FileNotFoundError("Required real BCI input missing: A01E.mat")

    monkeypatch.setattr(batch, "export_subject", missing_inputs)
    assert batch.main(["--dataset", "2a", "--output-root", str(tmp_path)]) == 1
    summary = json.loads(capsys.readouterr().out)
    report = json.loads(Path(summary["report"]).read_text())
    assert summary["failed"] == 1
    assert report["runs"][0]["figures"] == []
    assert report["runs"][0]["status"] == "failed"
    assert not list(tmp_path.rglob("*.png"))


def test_rejects_invalid_subjects(tmp_path):
    with pytest.raises(SystemExit):
        batch.main(["--subjects", "0", "--output-root", str(tmp_path)])
    with pytest.raises(SystemExit):
        batch.main(["--subjects", "1", "1", "--output-root", str(tmp_path)])
