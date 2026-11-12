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

ADDITIONAL_TOPICS = (
    {"id":"bci-cue-provenance","title":"EEG trial provenance and chronological FBCSP features","source":"Dataset 2A code; Souza et al. (2020), temporal dependence","stage":"Code implemented · tests and explanation to review","detail":"Trace original cue index and onset through trial filtering, development splits, FBCSP and the feature timeline. Explain retained index vs original time.","objective":"Write and verify a chronological-order and row-alignment test.","evidence":"https://github.com/hlin863/ewma_covariate_shift_demo/blob/main/src/bci/datasets/dataset2a/experiment.py","evidence_label":"Inspect Dataset 2A feature provenance"},
    {"id":"bci-visual-evidence","title":"Interpret BCI EDA and Stage-I/II event visualisations","source":"BCI EDA main merge · 12 Nov 2026","stage":"Code integrated · interpretation pending","detail":"Review FBCSP feature heatmaps, EWMA timeline, alarms and validation events without conflating exploratory plots with confirmed shifts.","objective":"Explain one event from cue to feature, Stage-I warning and Stage-II result.","evidence":"https://github.com/hlin863/ewma_covariate_shift_demo/commit/6cafc5ac9071d7483812d32810af138c1e28ea95","evidence_label":"Review integrated BCI EDA merge"},
    {"id":"moabb-benchmark","title":"MOABB: BCI paradigm, pipeline and evaluation","source":"Jayaram & Barachant (2018), Sections 2–3","stage":"Paper discussed · ChatGPT quiz practised; application assessment not imported","detail":"Distinguish trial preprocessing, model transformations and split/scoring protocol. Assess leakage and matched evaluations on 2A/2B.","objective":"Create a protocol manifest and test held-out fitting boundaries.","evidence":"https://github.com/hlin863/ewma_covariate_shift_demo/blob/main/src/bci/datasets/dataset2a/experiment.py","evidence_label":"Inspect BCI paradigm and feature pipeline"},
    {"id":"cpd-benchmark","title":"Change-point evaluation and event matching","source":"van den Burg & Williams (2022), Sections 3–5","stage":"Recommended reading · coding extension proposed","detail":"Compare known change timestamps to alarms with explicit tolerance, one-to-one matching, precision, recall and delay.","objective":"Write synthetic tests for TP/FP/FN event matching; do not claim EEG change ground truth.","evidence":"https://github.com/hlin863/ewma_covariate_shift_demo/blob/main/tests/reporting/test_evaluation.py","evidence_label":"Inspect current detection evaluation tests"},
    {"id":"stream-benchmark","title":"Stream-learning benchmarks and temporal dependence","source":"Souza et al. (2020), Sections 2 and 4","stage":"Recommended reading · diagnostic proposed","detail":"Investigate chronological versus shuffled EWMA monitoring and distinguish controlled injected drift from unlabeled real EEG nonstationarity.","objective":"Create an order-ablation trace using identical features and document the causal online protocol.","evidence":"https://github.com/hlin863/ewma_covariate_shift_demo/blob/main/scripts/run_complementary_monitoring.py","evidence_label":"Inspect synthetic monitoring runner"},
    {"id":"stage2-replay","title":"B01/D2 Stage-II discrepancy and eligible-window replay","source":"Raza et al. (2015, 2019); B01/D2 review","stage":"Investigation ongoing · numerical discrepancy unresolved","detail":"Preserve published B01 14 warnings/10 confirmations versus computed 17/0; examine fixed H=10 versus adaptive windows and 2H-d-1>0 eligibility. Wider Stage-I 255/191 and Stage-II 143/9 remain unresolved.","objective":"Freeze warning timestamps and compare candidate-window and rejection logs before claiming replication.","evidence":"https://github.com/hlin863/ewma_covariate_shift_demo/blob/main/docs/table1_algorithm_gap_analysis.md","evidence_label":"Inspect reproduction gap analysis"},
    {"id":"cost-aware-bci","title":"TWIN-ADAPT held-out evaluation and update cost","source":"Gupta et al. (2024); 10 Nov 2026 repository work","stage":"Implementation and single-subject pilot evidence · not superiority validation","detail":"Follow optimisation runners, candidate history, held-out data isolation, update count, retraining cost and seed-specific claims.","objective":"Compare fixed and cost-aware runs using matched splits, seeds and attached output provenance.","evidence":"https://github.com/hlin863/ewma_covariate_shift_demo/commit/f5b00e2d18a1d578cd790f2ef2da1bde8f846d30","evidence_label":"Inspect archived A01/B01 integration"},
)

RECENT_LEARNING_EVIDENCE = (
    {"date":"12 Nov 2026","title":"BCI feature provenance and detector charts","status":"Implemented in code","detail":"Cue indices/onsets are carried through Dataset 2A feature handling; BCI distribution, EWMA and Stage-II chart work merged.","url":"https://github.com/hlin863/ewma_covariate_shift_demo/commit/6cafc5ac9071d7483812d32810af138c1e28ea95","topic_id":"bci-cue-provenance"},
    {"date":"10 Nov 2026","title":"Cost-aware adaptation reference integration","status":"Implementation / exploratory evidence","detail":"Archived A01/B01 reference views and held-out optimisation flow are linked; broader performance claims still require matched repeated runs.","url":"https://github.com/hlin863/ewma_covariate_shift_demo/commit/f5b00e2d18a1d578cd790f2ef2da1bde8f846d30","topic_id":"cost-aware-bci"},
    {"date":"11–12 Nov 2026","title":"Three new benchmarking papers added to reading plan","status":"Studied / reading proposed","detail":"MOABB, change-point evaluation, and stream-learning benchmarking inform reproducibility, detection metrics and temporal assumptions; exercises are not yet verified in the application.","url":"https://github.com/hlin863/ewma_covariate_shift_demo/blob/main/papers/README.md","topic_id":"moabb-benchmark"},
    {"date":"Current review","title":"B01/D2 Stage-II reproduction discrepancy","status":"Unresolved","detail":"Published B01 14/10 versus computed 17/0; wider Stage-I 255/191 and Stage-II 143/9 are not resolved by learning or visualisation work.","url":"https://github.com/hlin863/ewma_covariate_shift_demo/blob/main/docs/table1_algorithm_gap_analysis.md","topic_id":"stage2-replay"},
)

TOPICS = TOPICS + ADDITIONAL_TOPICS

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
    return {"topics":topics, "recent_evidence":RECENT_LEARNING_EVIDENCE, "completed":sum(t["status"]=="completed" for t in topics),
            "in_progress":sum(t["status"]=="in_progress" for t in topics),
            "total":len(topics)}
