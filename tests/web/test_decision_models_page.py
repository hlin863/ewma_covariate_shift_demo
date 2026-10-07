import json
from pathlib import Path
import pandas as pd
from flask import Blueprint, Flask
from src.web.decision_models import decision_models_bp
from src.web.data_structures import data_structures_bp

def _app(project_root: Path) -> Flask:
    app=Flask(__name__,template_folder=str(project_root/"templates"),static_folder=str(project_root/"static"))
    app.add_url_rule("/",endpoint="home",view_func=lambda:"home")
    app.add_url_rule("/results",endpoint="results_catalog",view_func=lambda:"results")
    stub=Blueprint("bci_eda",__name__)
    stub.add_url_rule("/data-distributions",endpoint="data_distributions",view_func=lambda:"eda")
    app.register_blueprint(stub)
    app.register_blueprint(decision_models_bp)
    app.register_blueprint(data_structures_bp)
    return app

def test_decision_model_page_reads_saved_evidence(tmp_path: Path):
    metrics=pd.DataFrame([{"dataset":"2A","subject":"A01","evaluation_scope":"Session II","method":"decision_tree","role":"classifier","training_trials":10,"testing_trials":4,"feature_count":2,"accuracy":.75,"correct_predictions":3,"fit_seconds":.01,"predict_seconds":.001}])
    metrics_path=tmp_path/"metrics.csv"; metrics.to_csv(metrics_path,index=False)
    details=tmp_path/"details"; details.mkdir()
    payload={"metadata":{"dataset":"2A","subject":"A01"},"metrics":[],"tree":{"node_id":0,"kind":"leaf","samples":10,"class_values":[5,5],"prediction":0,"prediction_label":"left"},"examples":[{"trial_index":0,"truth":0,"truth_label":"left","feature_values":[],"decision_tree":{"prediction":0,"prediction_label":"left","leaf_id":0,"path":[{"node_id":0,"kind":"leaf","samples":10,"class_values":[5,5],"prediction":0,"prediction_label":"left"}]},"knn":{"prediction":0,"prediction_label":"left","neighbours":[],"vote_counts":{}},"pwknn":{"prediction":0,"prediction_label":"left","confidence":.8,"class_scores":{"left":.8,"right":.2},"neighbours":[]},"linear_svm":{"prediction":0,"prediction_label":"left","decision_score":-1.,"intercept":0.,"negative_class":"left","positive_class":"right","top_contributions":[]}}]}
    (details/"2a_A01.json").write_text(json.dumps(payload),encoding="utf-8")
    app=_app(Path(__file__).resolve().parents[2]); app.config.update(TESTING=True,BCI_DECISION_METRICS_PATH=str(metrics_path),BCI_DECISION_DETAILS_PATH=str(details))
    response=app.test_client().get("/results/decision-models?dataset=2A&subject=A01&trial=0")
    assert response.status_code==200
    page=response.get_data(as_text=True)
    assert "How does each model make a BCI decision?" in page
    assert "The datasets are not the tree nodes." in page
    assert "Trial 0 threshold path" in page

def test_decision_model_page_has_empty_state(tmp_path: Path):
    app=_app(Path(__file__).resolve().parents[2]); app.config.update(TESTING=True,BCI_DECISION_METRICS_PATH=str(tmp_path/"missing.csv"),BCI_DECISION_DETAILS_PATH=str(tmp_path/"details"))
    response=app.test_client().get("/results/decision-models")
    assert response.status_code==200
    assert b"No decision-model evidence has been generated yet" in response.data
