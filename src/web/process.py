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
        "description": "Connect confirmed drift with supervised, transductive and policy decisions.",
        "input": "Validated drift evidence, current predictions and the labels or memory allowed by the protocol.",
        "output": "An update decision, an updated knowledge base/model and a separate evaluation record.",
        "steps": (
            ("Select the update policy", "Compare remaining static, periodic updates and evidence-triggered updates. The Diethe laboratory provides an interactive policy experiment."),
            ("Respect the label boundary", "Supervised updates use permitted labels. Transductive updates use PWKNN pseudo-labels with confidence gating; hidden evaluation labels are reserved for post-run scoring."),
            ("Evaluate the change", "Compare prediction performance, update cost and retention under a stated protocol. Active-learning queries choose labels; memory selection decides which examples to retain."),
        ),
        "note": "Confidence-gated transductive refitting is implemented. Dynamic CSE-UAEL ensemble growth and weighted voting remain planned; the Turbofan query helper is not yet a completed adaptation experiment.",
        "links": (("diethe.laboratory", "Diethe policy laboratory", None),),
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
    {"title": "Learning and classifier evidence", "pages": (
        ("decision_models.explorer", "BCI decision model explorer", "Saved classifier decisions and pseudo-label confidence."),
        ("process.processing_results", "Data-processing run results", "Diagnostic counts and single-versus-bagged accuracy."),
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


@process_bp.get("/results/data-processing")
def processing_results():
    from pathlib import Path

    from src.web.data_processing import build_data_processing_view

    return render_template("processing_results.html", **build_data_processing_view(
        Path(current_app.config["RESULTS_ROOT"])))
