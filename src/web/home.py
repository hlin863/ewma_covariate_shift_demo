"""Paper-grounded content model for the research home page."""

from __future__ import annotations


def build_home_page_model() -> dict[str, object]:
    """Return the research narrative, implementation status and page catalogue."""

    return {
        "pages": (
            {
                "endpoint": "dashboard",
                "label": "Table 1 dashboard",
                "kind": "Primary reproduction",
                "description": (
                    "Compare published and computed CSW/CSV counts for BCI "
                    "Competition IV Datasets 2A and 2B."
                ),
            },
            {
                "endpoint": "results_catalog",
                "label": "Research results",
                "kind": "Evidence catalogue",
                "description": (
                    "Inspect generated tables, detector diagnostics, sensitivity "
                    "studies and synthetic reproductions."
                ),
            },
            {
                "endpoint": "figure_1",
                "label": "Figure 1 · A07",
                "kind": "Signal visualisation",
                "description": (
                    "Follow Dataset 2A subject A07 from cue-aligned trials through "
                    "FBCSP features and the training/evaluation shift view."
                ),
            },
            {
                "endpoint": "chowdhury_demographics",
                "label": "Chowdhury cohort",
                "kind": "Clinical extension",
                "description": (
                    "Explore connected stroke-cohort metadata with distributions "
                    "and participant-level Cleveland dot plots."
                ),
            },
            {
                "endpoint": "test_results.test_results",
                "label": "Automated tests",
                "kind": "Implementation verification",
                "description": (
                    "Review the current pytest/JUnit evidence and drill into each "
                    "test's observed and expected behaviour."
                ),
            },
        ),
        "pipeline": (
            {
                "number": "01",
                "title": "EEG trials",
                "detail": "Cue-aligned BCI IV 2A/2B session loading",
                "status": "implemented",
            },
            {
                "number": "02",
                "title": "FBCSP",
                "detail": "Ten overlapping 8–30 Hz bands and CSP features",
                "status": "implemented",
            },
            {
                "number": "03",
                "title": "PCA",
                "detail": "PC1 monitoring with retained components for validation",
                "status": "implemented",
            },
            {
                "number": "04",
                "title": "Stage I",
                "detail": "SD-EWMA covariate-shift warnings (CSW)",
                "status": "implemented",
            },
            {
                "number": "05",
                "title": "Stage II",
                "detail": "Multivariate validation and eligibility diagnostics",
                "status": "implemented",
            },
            {
                "number": "06",
                "title": "Paper agreement",
                "detail": "Subject-level numerical reproduction and audit",
                "status": "in progress",
            },
            {
                "number": "07",
                "title": "UAEL adaptation",
                "detail": "PWKNN pseudo-labelling and dynamic ensembles",
                "status": "planned",
            },
        ),
        "lineage": (
            {
                "year": "2015",
                "title": "Detector foundations",
                "detail": "SD-EWMA/TSSD-EWMA shift detection and D2 synthetic evidence.",
            },
            {
                "year": "2018",
                "title": "Online adaptive BCI",
                "detail": "Clinical and online-learning context for non-stationary EEG.",
            },
            {
                "year": "2019",
                "title": "CSE-UAEL",
                "detail": "Two-stage estimation connected to unsupervised adaptive learning.",
            },
            {
                "year": "Current",
                "title": "Auditable reconstruction",
                "detail": "Explicit alternatives expose decisions hidden by methodological ambiguity.",
            },
        ),
        "contributions": (
            "A modular reconstruction of cue extraction, FBCSP, PCA, EWMA warning and multivariate validation.",
            "Published-versus-computed subject results with generated diagnostics rather than hard-coded claims.",
            "Methodological ambiguity treated as an experimental variable: split, PCA retention, control limits and Stage-II interpretation.",
        ),
        "limitations": (
            "The default H=10 and d=20 Stage-II configuration is dimensionally ineligible because 2H − d − 1 is negative.",
            "Passing implementation tests do not establish exact agreement with published EEG warning and validation counts.",
            "Confirmed shifts do not yet trigger refitting, detector reinitialisation or classifier adaptation.",
        ),
    }
