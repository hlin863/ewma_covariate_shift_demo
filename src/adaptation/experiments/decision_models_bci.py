"""Interpretable Decision Tree / KNN / PWKNN / SVM comparison on real BCI features."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd

from src.adaptation.classifier import DecisionTreeClassifier, KNNClassifier, LinearSVMClassifier
from src.adaptation.decision_support import explain_decision_tree, explain_knn, explain_linear_svm, explain_pwknn, serialise_decision_tree
from src.adaptation.pseudo_labelling import PWKNNPseudoLabeler
from src.bci.data import load_bci_competition_iv_2a_session, load_bci_competition_iv_2b_session
from src.bci.datasets.dataset2a import build_dataset_2a_fbcsp_features, extract_dataset_2a_trials
from src.bci.datasets.dataset2a.evaluation_labels import load_dataset_2a_evaluation_labels, resolve_dataset_2a_evaluation_label_path
from src.bci.datasets.dataset2b import build_dataset_2b_fbcsp_features, concatenate_trial_signals, extract_dataset_2b_trials


@dataclass(frozen=True)
class PreparedDecisionDataset:
    dataset: str
    subject: str
    evaluation_scope: str
    training_features: np.ndarray
    training_labels: np.ndarray
    testing_features: np.ndarray
    testing_labels: np.ndarray
    feature_names: tuple[str, ...]


def prepare_dataset_2a(*, data_directory: str | Path, labels_directory: str | Path, subject: int) -> PreparedDecisionDataset:
    training_session = load_bci_competition_iv_2a_session(data_directory, subject, "T")
    evaluation_session = load_bci_competition_iv_2a_session(data_directory, subject, "E")
    label_path = resolve_dataset_2a_evaluation_label_path(subject, data_directory=Path(data_directory), labels_directory=Path(labels_directory))
    training = extract_dataset_2a_trials(training_session)
    testing = extract_dataset_2a_trials(evaluation_session, evaluation_labels=load_dataset_2a_evaluation_labels(label_path))
    pipeline = build_dataset_2a_fbcsp_features(training, testing)
    return PreparedDecisionDataset("2A", f"A{subject:02d}", "Session II · official left/right labels", np.asarray(pipeline.training.features,float), np.asarray(training.labels,int), np.asarray(pipeline.testing.features,float), np.asarray(testing.labels,int), tuple(pipeline.training.feature_names))


def prepare_dataset_2b(*, data_directory: str | Path, subject: int) -> PreparedDecisionDataset:
    sessions = {i: extract_dataset_2b_trials(load_bci_competition_iv_2b_session(data_directory, subject, i)) for i in (1,2,3)}
    training, testing = concatenate_trial_signals([sessions[1], sessions[2]]), sessions[3]
    if np.any(testing.labels < 0):
        raise ValueError(f"B{subject:02d}: Session III must contain labelled left/right cues.")
    pipeline = build_dataset_2b_fbcsp_features(training, testing)
    return PreparedDecisionDataset("2B", f"B{subject:02d}", "Session III · GDF-only labelled holdout", np.asarray(pipeline.training.features,float), np.asarray(training.labels,int), np.asarray(pipeline.testing.features,float), np.asarray(testing.labels,int), tuple(pipeline.training.feature_names))


def _metric(p, *, method, role, predictions, fit_seconds, predict_seconds):
    predictions = np.asarray(predictions)
    return {"dataset":p.dataset,"subject":p.subject,"evaluation_scope":p.evaluation_scope,"method":method,"role":role,"training_trials":int(p.training_features.shape[0]),"testing_trials":int(p.testing_features.shape[0]),"feature_count":int(p.training_features.shape[1]),"accuracy":float(np.mean(predictions==p.testing_labels)),"correct_predictions":int(np.count_nonzero(predictions==p.testing_labels)),"fit_seconds":float(fit_seconds),"predict_seconds":float(predict_seconds)}


def evaluate_prepared_decision_models(prepared: PreparedDecisionDataset, *, random_state=42, tree_depth=4, min_samples_leaf=5, n_neighbors=5, rbf_sigma=1.0, explain_trials=12):
    x_train,y_train,x_test=prepared.training_features,prepared.training_labels,prepared.testing_features
    tree=DecisionTreeClassifier(max_depth=tree_depth,min_samples_leaf=min_samples_leaf,random_state=random_state); t=perf_counter(); tree.fit(x_train,y_train); tree_fit=perf_counter()-t; t=perf_counter(); tree_pred=tree.predict(x_test); tree_p=perf_counter()-t
    knn=KNNClassifier(n_neighbors=n_neighbors); t=perf_counter(); knn.fit(x_train,y_train); knn_fit=perf_counter()-t; t=perf_counter(); knn_pred=knn.predict(x_test); knn_p=perf_counter()-t
    svm=LinearSVMClassifier(random_state=random_state); t=perf_counter(); svm.fit(x_train,y_train); svm_fit=perf_counter()-t; t=perf_counter(); svm_pred=svm.predict(x_test); svm_p=perf_counter()-t
    pw=PWKNNPseudoLabeler(n_neighbors=n_neighbors,rbf_sigma=rbf_sigma); t=perf_counter(); pw_batch=pw.predict_with_confidence(reference_features=x_train,reference_labels=y_train,query_features=x_test); pw_p=perf_counter()-t
    metrics=[
      _metric(prepared,method="decision_tree",role="supervised classifier · threshold path",predictions=tree_pred,fit_seconds=tree_fit,predict_seconds=tree_p),
      _metric(prepared,method="knn",role="supervised classifier · majority neighbour vote",predictions=knn_pred,fit_seconds=knn_fit,predict_seconds=knn_p),
      _metric(prepared,method="pwknn",role="pseudo-labeller · RBF-weighted neighbour evidence",predictions=pw_batch.labels,fit_seconds=0.0,predict_seconds=pw_p),
      _metric(prepared,method="linear_svm",role="supervised classifier · maximum-margin decision",predictions=svm_pred,fit_seconds=svm_fit,predict_seconds=svm_p),
    ]
    count=min(max(int(explain_trials),1),int(x_test.shape[0]))
    indices=np.unique(np.linspace(0,x_test.shape[0]-1,num=count,dtype=int))
    examples=[]
    for index in indices:
        sample=x_test[index]
        examples.append({"trial_index":int(index),"truth":int(prepared.testing_labels[index]),"truth_label":"left" if prepared.testing_labels[index]==0 else "right","feature_values":[{"feature":n,"value":float(v)} for n,v in zip(prepared.feature_names,sample)],"decision_tree":explain_decision_tree(tree,sample,prepared.feature_names),"knn":explain_knn(knn,sample),"pwknn":explain_pwknn(pw,reference_features=x_train,reference_labels=y_train,sample=sample),"linear_svm":explain_linear_svm(svm,sample,prepared.feature_names)})
    detail={"metadata":{"dataset":prepared.dataset,"subject":prepared.subject,"evaluation_scope":prepared.evaluation_scope,"training_trials":int(x_train.shape[0]),"testing_trials":int(x_test.shape[0]),"feature_count":int(x_train.shape[1]),"feature_names":list(prepared.feature_names),"representation":"FBCSP log-normalised variance features","detector_boundary":"No EWMA/Hotelling decision is made here; detection remains in src/detection.","pwknn_boundary":"PWKNN is a pseudo-labeller; its accuracy here is post-run diagnostic evidence."},"metrics":metrics,"tree":serialise_decision_tree(tree,prepared.feature_names),"examples":examples}
    return metrics,detail


def run_bci_decision_model_experiment(*,data_2a,labels_2a,data_2b,datasets=("2a","2b"),subjects=tuple(range(1,10)),random_state=42,tree_depth=4,min_samples_leaf=5,n_neighbors=5,rbf_sigma=1.0,explain_trials=12):
    records=[]; details={}
    for subject in subjects:
        for key in datasets:
            prepared=prepare_dataset_2a(data_directory=data_2a,labels_directory=labels_2a,subject=subject) if key=="2a" else prepare_dataset_2b(data_directory=data_2b,subject=subject)
            rows,detail=evaluate_prepared_decision_models(prepared,random_state=random_state,tree_depth=tree_depth,min_samples_leaf=min_samples_leaf,n_neighbors=n_neighbors,rbf_sigma=rbf_sigma,explain_trials=explain_trials)
            records.extend(rows); details[f"{key}_{prepared.subject}"]=detail
    return pd.DataFrame.from_records(records),details
