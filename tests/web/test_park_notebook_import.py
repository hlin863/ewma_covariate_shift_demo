"""Notebook 07 import cell must work from both common Jupyter launch directories."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "07_park_unimodality_visualisation.ipynb"


@pytest.mark.parametrize("working_dir", [ROOT, ROOT / "notebooks"])
def test_park_notebook_imports_from_working_directory(working_dir):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][1]["source"])
    compile(source, str(NOTEBOOK), "exec")
    run = subprocess.run(
        [sys.executable, "-c", source],
        cwd=working_dir,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert f"Repository root: {ROOT}" in run.stdout
