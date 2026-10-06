"""Model-specific evidence for interpretable BCI decision making."""

from __future__ import annotations
from typing import Iterable
import numpy as np

from src.adaptation.classifier import DecisionTreeClassifier, KNNClassifier, LinearSVMClassifier
from src.adaptation.pseudo_labelling import PWKNNPseudoLabeler


def _one_sample(values: np.ndarray, *, expected_features: int) -> np.ndarray:
    sample = np.asarray(values, dtype=float)
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    if sample.ndim != 2 or sample.shape[0] != 1:
        raise ValueError("decision evidence requires exactly one feature vector.")
    if sample.shape[1] != expected_features:
        raise ValueError(f"expected {expected_features} features, received {sample.shape[1]}.")
    if not np.isfinite(sample).all():
        raise ValueError("feature vector must contain only finite values.")
    return sample


def _names(feature_names: Iterable[str], n_features: int) -> tuple[str, ...]:
    names = tuple(str(name) for name in feature_names)
    if len(names) != n_features:
        raise ValueError("feature_names must match the feature dimension.")
    return names


def _python_label(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def class_name(value: object) -> str:
    return "left" if value == 0 else "right" if value == 1 else str(value)


def explain_decision_tree(model: DecisionTreeClassifier, sample: np.ndarray, feature_names: Iterable[str]) -> dict[str, object]:
    if model.n_features is None:
        raise RuntimeError("Decision tree must be fitted before explanation.")
    x = _one_sample(sample, expected_features=model.n_features)
    names = _names(feature_names, model.n_features)
    tree = model.tree_
    leaf_id = int(model.apply(x)[0])
    path = []
    for node_id in model.decision_path(x).indices.tolist():
        values = np.asarray(tree.value[node_id]).reshape(-1)
        predicted = model.classes_[int(np.argmax(values))]
        if node_id == leaf_id or int(tree.feature[node_id]) < 0:
            path.append({"node_id": int(node_id), "kind": "leaf", "samples": int(tree.n_node_samples[node_id]), "class_values": values.round(6).tolist(), "prediction": _python_label(predicted), "prediction_label": class_name(predicted)})
            continue
        feature_index = int(tree.feature[node_id])
        threshold = float(tree.threshold[node_id])
        observed = float(x[0, feature_index])
        go_left = observed <= threshold
        path.append({"node_id": int(node_id), "kind": "split", "feature_index": feature_index, "feature": names[feature_index], "value": observed, "threshold": threshold, "operator": "<=" if go_left else ">", "branch": "left" if go_left else "right", "samples": int(tree.n_node_samples[node_id]), "class_values": values.round(6).tolist()})
    prediction = model.predict(x)[0]
    return {"prediction": _python_label(prediction), "prediction_label": class_name(prediction), "leaf_id": leaf_id, "path": path}


def serialise_decision_tree(model: DecisionTreeClassifier, feature_names: Iterable[str]) -> dict[str, object]:
    if model.n_features is None:
        raise RuntimeError("Decision tree must be fitted before serialisation.")
    names = _names(feature_names, model.n_features)
    tree, classes = model.tree_, model.classes_
    def visit(node_id: int) -> dict[str, object]:
        values = np.asarray(tree.value[node_id]).reshape(-1)
        predicted = classes[int(np.argmax(values))]
        feature_index = int(tree.feature[node_id])
        node = {"node_id": int(node_id), "samples": int(tree.n_node_samples[node_id]), "class_values": values.round(6).tolist(), "prediction": _python_label(predicted), "prediction_label": class_name(predicted)}
        if feature_index < 0:
            node["kind"] = "leaf"
            return node
        node.update({"kind": "split", "feature_index": feature_index, "feature": names[feature_index], "threshold": float(tree.threshold[node_id]), "left": visit(int(tree.children_left[node_id])), "right": visit(int(tree.children_right[node_id]))})
        return node
    return visit(0)


def explain_knn(model: KNNClassifier, sample: np.ndarray) -> dict[str, object]:
    if model.n_features is None:
        raise RuntimeError("KNN must be fitted before explanation.")
    x = _one_sample(sample, expected_features=model.n_features)
    distances, indices = model.kneighbors(x)
    labels = model.training_labels
    neighbours = [{"training_index": int(index), "label": _python_label(labels[index]), "label_name": class_name(labels[index]), "distance": float(distance)} for distance, index in zip(distances[0], indices[0])]
    prediction = model.predict(x)[0]
    classes, counts = np.unique(labels[indices[0]], return_counts=True)
    return {"prediction": _python_label(prediction), "prediction_label": class_name(prediction), "neighbours": neighbours, "vote_counts": {class_name(label): int(count) for label, count in zip(classes, counts)}}


def explain_pwknn(labeler: PWKNNPseudoLabeler, *, reference_features: np.ndarray, reference_labels: np.ndarray, sample: np.ndarray) -> dict[str, object]:
    x_ref, y_ref = np.asarray(reference_features, dtype=float), np.asarray(reference_labels)
    if x_ref.ndim != 2 or y_ref.ndim != 1 or x_ref.shape[0] != y_ref.size:
        raise ValueError("reference features/labels are inconsistent.")
    x = _one_sample(sample, expected_features=x_ref.shape[1])
    batch = labeler.predict_with_confidence(reference_features=x_ref, reference_labels=y_ref, query_features=x)
    squared = np.sum((x_ref - x[0]) ** 2, axis=1)
    k = min(labeler.n_neighbors, x_ref.shape[0])
    nearest = np.argpartition(squared, k - 1)[:k]
    nearest = nearest[np.argsort(squared[nearest])]
    stabilized = squared[nearest] - squared[nearest].min()
    weights = np.exp(-stabilized / (2.0 * labeler.rbf_sigma**2))
    total = float(weights.sum())
    scores = {class_name(label): float(np.sum(weights[y_ref[nearest] == label]) / total) for label in np.unique(y_ref)}
    neighbours = [{"training_index": int(index), "label": _python_label(y_ref[index]), "label_name": class_name(y_ref[index]), "distance": float(np.sqrt(squared[index])), "weight": float(weight)} for index, weight in zip(nearest, weights)]
    prediction = batch.labels[0]
    return {"prediction": _python_label(prediction), "prediction_label": class_name(prediction), "confidence": float(batch.confidence[0]), "class_scores": scores, "neighbours": neighbours}


def explain_linear_svm(model: LinearSVMClassifier, sample: np.ndarray, feature_names: Iterable[str], *, top_features: int = 8) -> dict[str, object]:
    if model.n_features is None:
        raise RuntimeError("Linear SVM must be fitted before explanation.")
    x = _one_sample(sample, expected_features=model.n_features)
    names = _names(feature_names, model.n_features)
    coef = model.coef_
    if coef.shape[0] != 1:
        raise ValueError("This explanation currently supports binary linear SVMs only.")
    contributions = x[0] * coef[0]
    order = np.argsort(np.abs(contributions))[::-1][:max(1, int(top_features))]
    prediction = model.predict(x)[0]
    return {"prediction": _python_label(prediction), "prediction_label": class_name(prediction), "decision_score": float(np.asarray(model.decision_function(x)).reshape(-1)[0]), "intercept": float(model.intercept_[0]), "negative_class": class_name(model.classes_[0]), "positive_class": class_name(model.classes_[-1]), "top_contributions": [{"feature_index": int(index), "feature": names[index], "value": float(x[0,index]), "weight": float(coef[0,index]), "contribution": float(contributions[index])} for index in order]}
