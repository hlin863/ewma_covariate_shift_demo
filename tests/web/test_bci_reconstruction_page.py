import json
import pandas as pd
from src.web import app


def test_bci_reconstruction_empty_and_saved(tmp_path):
    app.config["BCI_RECONSTRUCTION_PATH"] = str(tmp_path / "results")
    app.config["BCI_METRICS_PATH"] = str(tmp_path / "metrics")
    client = app.test_client()
    empty = client.get("/results/abcd/bci")
    assert empty.status_code == 200
    assert b"Run the pipeline" in empty.data
    root = tmp_path / "results"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"subjects": [
        {"dataset":"2B","subject":1,"pca_components":1,
         "mean_training_loss":0.2,"mean_evaluation_loss":0.3,
         "training_trials":40,"evaluation_trials":30}
    ]}))
    pd.DataFrame({"trial_index":[0,1],"reconstruction_mse":[0.2,0.3],"pc1_score":[0.1,0.2]}).to_csv(root/"2B_01_loss_trace.csv",index=False)
    response = client.get("/results/abcd/bci?subject=B01")
    assert response.status_code == 200
    assert b"B01" in response.data
    assert b"bci-loss-chart" in response.data
    assert b"0.3" in response.data


def test_bci_reconstruction_no_baseline_claim(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"subjects": [
        {"dataset":"2A","subject":2,"pca_components":1,"mean_training_loss":0.1,"mean_evaluation_loss":0.1}
    ]}))
    app.config["BCI_RECONSTRUCTION_PATH"] = str(root)
    app.config["BCI_METRICS_PATH"] = str(tmp_path / "missing")
    response = app.test_client().get("/results/abcd/bci")
    assert response.status_code == 200
    assert b"Previous EWMA warnings" in response.data
    assert b"N/A" in response.data
