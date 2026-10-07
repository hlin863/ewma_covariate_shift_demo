"""Method navigation kept separate from saved experiment evidence."""

from flask import Blueprint, abort, current_app, render_template, request, url_for


process_bp = Blueprint("process", __name__)

PROCESS_PAGES = (
    {
        "key": "covariate-shift", "number": "01", "title": "Covariate shift",
        "question": "Has the feature distribution changed?",
        "description": "Follow feature preparation and Stage-I EWMA warnings.",
        "input": "Training-fitted features and a chronological evaluation stream.",
        "output": "Candidate warning events passed to Stage-II validation.",
        "steps": (
            ("Prepare the representation", "Load cue-aligned trials, fit FBCSP and PCA on training data, and transform evaluation observations with those fitted components."),
            ("Monitor the stream", "Use SD/MSD-EWMA to compare incoming observations with control limits. Complementary monitoring also examines PCA scores and reconstruction residuals."),
            ("Record a warning", "Preserve the warning index and detector configuration. A warning identifies a candidate change; Stage II determines whether it is confirmed."),
        ),
        "note": "An input-distribution change does not by itself establish reduced classification performance or justify retraining.",
        "links": (("data_processing_page", "Feature preparation and protocols", "processing"),),
        "next": "validation",
    },
    {
        "key": "validation", "number": "02", "title": "Validation",
        "question": "Is the warning supported by a second test?",
        "description": "Distinguish candidate warnings from confirmed shifts.",
        "input": "Stage-I warnings plus the reference observations required by the chosen test.",
        "output": "Confirmed shifts, rejected candidates and ineligible tests, recorded separately.",
        "steps": (
            ("Choose the test", "Keep the K–S, training-reference Hotelling and two-sample Hotelling interpretations explicit. They use different reference data and are not interchangeable."),
            ("Check the available evidence", "Record feature dimension, sample counts and reference/window boundaries. A retrospective before/after window uses observations after the warning and must be identified as retrospective."),
            ("Confirm or reject", "Apply the configured significance threshold and retain the outcome. A test that lacks sufficient data or valid degrees of freedom is ineligible, rather than evidence that no shift occurred."),
        ),
        "note": "Stage-II drift validation is separate from a development validation split, held-out model evaluation and automated software tests.",
        "links": (("data_processing_page", "Training, validation and evaluation roles", "protocols"),),
        "next": "adaptive-learning",
    },
    {
        "key": "adaptive-learning", "number": "03", "title": "Adaptive learning",
        "question": "When and how should the model update?",
        "description": "Use Hurtado et al. (2023)'s active/passive adaptation distinction inside four explicit dimensions: update timing, label acquisition, retention and model evolution.",
        "input": "Validated drift evidence, current predictions and the labels or memory allowed by the protocol.",
        "output": "An update decision, an updated knowledge base/model and a separate evaluation record.",
        "steps": (
            ("Choose the adaptation family", "Separate non-adaptive, passive and active update rules before comparing accuracy or cost."),
            ("Choose the knowledge boundary", "Supervised labels, active-learning queries and PWKNN pseudo-labels solve different problems and must remain distinct."),
            ("Choose retention and model evolution", "State which arrived observations can enter training and whether one classifier or an ensemble evolves."),
            ("Evaluate the change", "Compare prediction performance, update burden, training growth and retention under the same chronological protocol."),
        ),
        "dimensions": (
            {
                "label": "01 · When to update",
                "title": "Update timing",
                "question": "Does adaptation require detected evidence, or does it occur independently of drift detection?",
                "items": (
                    {"name": "NeverUpdate", "detail": "Static non-adaptive control.", "status": "implemented"},
                    {"name": "PeriodicRetrain", "detail": "Passive update after a configured number of new trials since the preceding update.", "status": "implemented"},
                    {"name": "ContinuousRetrain", "detail": "Passive per-observation or mini-batch update controlled by batch_size.", "status": "implemented"},
                    {"name": "RetrainOnWarning", "detail": "Active update on a Stage-I warning.", "status": "implemented"},
                    {"name": "RetrainOnValidatedShift", "detail": "Active update only after Stage-II confirmation.", "status": "implemented"},
                    {"name": "RetrainOnPerformanceDrop", "detail": "Active labelled-performance trigger with a rolling window and cooldown.", "status": "implemented"},
                ),
            },
            {
                "label": "02 · How knowledge is acquired",
                "title": "Label and query regime",
                "question": "Where does the information used for adaptation come from?",
                "items": (
                    {"name": "Supervised labels", "detail": "Prediction happens first; the permitted true label can then enter a supervised update.", "status": "implemented"},
                    {"name": "Active-learning query selection", "detail": "Uncertainty, diversity and committee-disagreement utilities rank observations for label requests.", "status": "utility implemented"},
                    {"name": "PWKNN pseudo-labels", "detail": "RBF-weighted neighbour evidence supplies a pseudo-label and confidence without exposing evaluation truth.", "status": "implemented"},
                ),
            },
            {
                "label": "03 · What data are retained",
                "title": "Training horizon",
                "question": "Which arrived observations are eligible to become training knowledge?",
                "items": (
                    {"name": "current_trial", "detail": "Add only the observation at the update event.", "status": "implemented"},
                    {"name": "since_last_update", "detail": "Add observations accumulated since the previous update.", "status": "implemented"},
                    {"name": "all_seen", "detail": "Transductive path may reconsider all arrived, not-yet-admitted observations.", "status": "implemented"},
                    {"name": "bounded memory / reservoir", "detail": "Explicit fixed-memory continual-learning policy.", "status": "planned"},
                ),
            },
            {
                "label": "04 · How the model evolves",
                "title": "Model evolution",
                "question": "What predictive object changes after knowledge is admitted?",
                "items": (
                    {"name": "Single retrainable classifier", "detail": "Linear SVM, KNN or another classifier satisfying the retraining contract.", "status": "implemented"},
                    {"name": "Static bagging baseline", "detail": "Bootstrap ensemble comparison kept separate from drift-triggered growth.", "status": "implemented"},
                    {"name": "Dynamic CSE-UAEL ensemble", "detail": "CSE-triggered LDA growth and weighted voting from the later Raza lineage.", "status": "planned"},
                ),
            },
        ),
        "active_learning_boundary": (
            "Active adaptation asks when a model should update because the environment changed. "
            "Active learning asks which observations are worth querying for labels. "
            "The repository implements both concepts as separate mechanisms."
        ),
        "note": "Confidence-gated transductive refitting and passive/active policy primitives are implemented. The first full 2A/2B transductive adaptive experiment, bounded-memory learning and dynamic CSE-UAEL ensemble growth remain open.",
        "links": (
            ("diethe.laboratory", "Continual-learning policy laboratory", None),
            ("process.transductive_results", "Transductive adaptation evidence", None),
            ("decision_models.explorer", "BCI decision model explorer", None),
        ),
        "next": None,
    },
)

RESULT_PAGES = (
    {"title": "Covariate-shift evidence", "pages": (
        ("figure_1", "Figure 1 · A07", "Cue-aligned signals and feature visualisations."),
        ("complementary_results", "Complementary monitoring", "Saved PCA score and residual experiments."),
    )},
    {"title": "Validation and reproduction", "pages": (
        ("dashboard", "Table 1 dashboard", "Published and computed warning/confirmation counts."),
        ("ks_validation_results", "K–S validation diagnostics", "Per-warning Stage-II test evidence."),
    )},
    {"title": "Continual learning and classifier evidence", "pages": (
        ("diethe.laboratory", "Continual-learning policy laboratory", "Matched non-adaptive, passive and active update-policy comparisons."),
        ("process.transductive_results", "Transductive adaptation evidence", "Online event sequence, pseudo-label confidence, knowledge-base growth and current evidence boundary."),
        ("decision_models.explorer", "BCI decision model explorer", "Saved classifier decisions and PWKNN pseudo-label confidence."),
        ("process.processing_results", "Data-processing and bagging results", "Diagnostic counts and single-versus-bagged accuracy."),
    )},
    {"title": "Implementation checks", "pages": (
        ("test_results.test_results", "Automated tests", "Software checks, separate from scientific reproduction."),
    )},
)


def result_navigation_groups():
    """Resolve only endpoints available to the current Flask application."""
    return [dict(group, pages=[
        {"url": url_for(endpoint), "label": label, "description": description,
         "active": request.endpoint == endpoint}
        for endpoint, label, description in group["pages"]
        if endpoint in current_app.view_functions
    ]) for group in RESULT_PAGES]


@process_bp.app_context_processor
def research_navigation():
    return {
        "process_navigation": [dict(page, url=url_for("process.overview", stage=page["key"]),
                                    active=request.endpoint == "process.overview" and
                                    request.view_args.get("stage") == page["key"])
                               for page in PROCESS_PAGES],
        "result_navigation_groups": result_navigation_groups(),
    }


@process_bp.get("/process/<stage>")
def overview(stage):
    page = next((page for page in PROCESS_PAGES if page["key"] == stage), None)
    if page is None:
        abort(404)
    links = [{"url": url_for(endpoint, _anchor=anchor), "label": label}
             for endpoint, label, anchor in page["links"]]
    return render_template("process.html", process=page, process_links=links)


@process_bp.get("/results/transductive-adaptation")
def transductive_results():
    event_groups = (
        {
            "title": "Trial records",
            "purpose": "What the deployed classifier knew and predicted before any same-trial update.",
            "fields": (
                "trial_index", "time", "predicted_label", "stage1_warning",
                "stage2_status", "stage2_p_value", "validated_shift",
                "classifier_version", "training_size_before", "trials_since_update",
                "adaptation_triggered", "pseudo_label_candidates",
                "pseudo_labels_accepted", "retrained_after_trial",
            ),
        },
        {
            "title": "Pseudo-label events",
            "purpose": "Which arrived observations were considered for knowledge-base admission.",
            "fields": (
                "trigger_trial_index", "candidate_trial_index", "pseudo_label",
                "confidence", "confidence_threshold", "accepted",
                "classifier_version_before",
            ),
        },
        {
            "title": "Update events",
            "purpose": "What changed when enough confidence-gated pseudo-labels were admitted.",
            "fields": (
                "trigger_trial_index", "trials_since_update", "validated_shift", "candidate_samples",
                "accepted_samples", "rejected_samples", "mean_accepted_confidence",
                "training_size_before", "training_size_after",
                "knowledge_base_size_after", "retrain_seconds", "classifier_version",
            ),
        },
    )
    return render_template(
        "transductive_results.html",
        event_groups=event_groups,
        implemented=(
            "Evaluation truth is excluded from the online learner API.",
            "PWKNN returns pseudo-labels with confidence and applies a strict confidence gate.",
            "Accepted observations grow both the classifier training set and the PWKNN knowledge base.",
            "Prediction occurs before an update triggered at the same trial.",
            "Ground-truth evaluation is isolated to separate post-run scoring.",
        ),
        future=(
            "Generate and preserve the first full BCI Competition IV 2A/2B transductive adaptive run.",
            "Add bounded-memory or reservoir retention experiments.",
            "Add dynamic CSE-UAEL classifier growth and weighted voting.",
            "Compare the same active/passive policy families on a predictive-maintenance stream.",
        ),
    )


@process_bp.get("/results/data-processing")
def processing_results():
    from pathlib import Path

    from src.web.data_processing import build_data_processing_view

    return render_template("processing_results.html", **build_data_processing_view(
        Path(current_app.config["RESULTS_ROOT"])))
