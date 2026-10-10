"""Paper-linked learning milestones with local, explicit progress persistence.

Conversation-derived activity is labelled 'studied', never inferred as mastered.
User-selected progress is saved separately from empirical experiment outputs.
"""
import json
import os
from pathlib import Path

TOPICS = (
    {"id":"park-segment-means","title":"Two-window CUSUM mean comparison","source":"Park et al. (2023), Section 2.1 / Eq. (2.2)","stage":"Studied in discussion","detail":"Partition ordered observations and compute left and right segment means.","url":"park_worked_example.example"},
    {"id":"park-normalisation","title":"Equation (2.2): normalisation and weighting","source":"Park et al. (2023), Section 2.1","stage":"Practised in discussion","detail":"Reconstruct the exact worked result ν₄ = √2 for the eight-observation example.","url":"park_worked_example.example"},
    {"id":"park-aggregation","title":"Maximum and average CUSUM aggregation","source":"Park et al. (2023), Section 2.1","stage":"Studied; review next","detail":"Differentiate feature-wise, maximum and average candidate curves.","url":"data_structures.inspect_dataset","args":{"dataset":"park"}},
    {"id":"park-multiple-peaks","title":"Multiple peaks and candidate-location ambiguity","source":"Park et al. (2023); Fearnhead & Rigaill (2020)","stage":"Literature writing updated","detail":"Explain why local peaks need not correspond to separate confirmed changes.","url":"data_structures.inspect_dataset","args":{"dataset":"park"}},
    {"id":"park-confirmation","title":"Unimodality and peak-agreement decision criteria","source":"Park et al. (2023), Section 2.1","stage":"Next reading","detail":"Check the original decision rules before implementing confirmation.","url":"data_structures.inspect_dataset","args":{"dataset":"park"}},
    {"id":"park-second-order","title":"Second-order structural changes","source":"Park et al. (2023), following mean-change analysis","stage":"Upcoming","detail":"Study wavelet representations and dynamic PCA after mean-change validation.","url":"data_structures.inspect_dataset","args":{"dataset":"park"}},
)

VALID_STATUS = ("not_started","in_progress","completed")


def progress_file(config):
    return Path(config.get("LEARNING_PROGRESS_PATH", Path(__file__).resolve().parents[2] / "outputs" / "learning" / "progress.json"))


def read_progress(config):
    path = progress_file(config)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    allowed = {topic["id"] for topic in TOPICS}
    if not isinstance(payload, dict):
        return {}
    return {key:value for key,value in payload.items() if key in allowed and value in VALID_STATUS}


def write_progress(config, values):
    path = progress_file(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def learning_model(config):
    progress = read_progress(config)
    topics = [dict(topic, status=progress.get(topic["id"], "not_started")) for topic in TOPICS]
    return {"topics":topics, "completed":sum(t["status"]=="completed" for t in topics),
            "in_progress":sum(t["status"]=="in_progress" for t in topics),
            "total":len(topics)}
