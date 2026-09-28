"""Paper-grounded content model for the research home page."""

from __future__ import annotations


def build_home_page_model() -> dict[str, object]:
    """Return the research narrative, implementation status and page catalogue."""

    return {
        "application_map": {
            "root": {
                "title": "CSE research application",
                "detail": (
                    "The home page is the navigation root. Evidence flows from "
                    "dataset/protocol inspection into experiment analysis and then "
                    "into generated results and implementation verification."
                ),
            },
            "layers": (
                {
                    "number": "01",
                    "title": "Data and protocol foundation",
                    "detail": (
                        "Inspect source datasets, processing boundaries and cohort "
                        "context before interpreting detector or classifier outputs."
                    ),
                    "relationship": "feeds experiments",
                    "pages": (
                        {
                            "endpoint": "data_processing_page",
                            "label": "Data processing atlas",
                            "kind": "Protocols and paper evidence",
                            "description": (
                                "Trace development splits, FBCSP boundaries and "
                                "paper-to-implementation decisions."
                            ),
                        },
                        {
                            "endpoint": "figure_1",
                            "label": "Figure 1 · A07",
                            "kind": "Signal visualisation",
                            "description": (
                                "Follow Dataset 2A subject A07 from cue-aligned "
                                "trials through FBCSP and the shift view."
                            ),
                        },
                        {
                            "endpoint": "chowdhury_demographics",
                            "label": "Chowdhury cohort",
                            "kind": "Clinical extension",
                            "description": (
                                "Inspect stroke-cohort metadata and participant-level "
                                "distributions used by the clinical extension."
                            ),
                        },
                    ),
                },
                {
                    "number": "02",
                    "title": "Detection and experiment analysis",
                    "detail": (
                        "Run and inspect the primary reproduction and complementary "
                        "monitoring experiments built on the data-processing layer."
                    ),
                    "relationship": "produces evidence",
                    "pages": (
                        {
                            "endpoint": "dashboard",
                            "label": "Table 1 dashboard",
                            "kind": "Primary reproduction",
                            "description": (
                                "Compare published and computed CSW/CSV counts for "
                                "BCI Competition IV Datasets 2A and 2B."
                            ),
                        },
                        {
                            "endpoint": "complementary_results",
                            "label": "Complementary monitoring",
                            "kind": "BCI experiment",
                            "description": (
                                "Explore PCA score and reconstruction-residual "
                                "warnings across BCI Competition IV 2A and 2B."
                            ),
                        },
                    ),
                },
                {
                    "number": "03",
                    "title": "Evidence and implementation assurance",
                    "detail": (
                        "Consolidate generated artifacts and verify that the code "
                        "contracts supporting those artifacts continue to pass."
                    ),
                    "relationship": "supports interpretation",
                    "pages": (
                        {
                            "endpoint": "results_catalog",
                            "label": "Research results",
                            "kind": "Evidence catalogue",
                            "description": (
                                "Inspect generated tables, detector diagnostics, "
                                "sensitivity studies and experiment visualisations."
                            ),
                        },
                        {
                            "endpoint": "test_results.test_results",
                            "label": "Automated tests",
                            "kind": "Implementation verification",
                            "description": (
                                "Review pytest/JUnit evidence and drill into each "
                                "test's observed and expected behaviour."
                            ),
                        },
                    ),
                },
            ),
            "support": {
                "endpoint": "support.support_layer",
                "label": "Local RAG support layer",
                "kind": "Cross-cutting repository assistant",
                "description": (
                    "Ask local Llama 3.2 about code, dataset protocols, generated "
                    "results and proposal scope using retrieved project evidence."
                ),
                "boundary": (
                    "Reads across all application layers for explanation and "
                    "navigation; it does not alter the CSE scientific pipeline."
                ),
            },
        },
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
                "paper_file": "2015-raza-ewma-covariate-shift.pdf",
                "paper_title": (
                    "EWMA model based shift-detection methods for detecting "
                    "covariate shifts in non-stationary environments"
                ),
                "citation": "Raza, Prasad & Li · Pattern Recognition",
            },
            {
                "year": "2018",
                "title": "Online adaptive BCI",
                "detail": "Clinical and online-learning context for non-stationary EEG.",
                "paper_file": "2018-chowdhury-online-adaptive-bci.pdf",
                "paper_title": (
                    "Online Covariate Shift Detection-Based Adaptive "
                    "Brain-Computer Interface to Trigger Hand Exoskeleton "
                    "Feedback for Neuro-Rehabilitation"
                ),
                "citation": "Chowdhury et al. · IEEE TCDS",
            },
            {
                "year": "2019",
                "title": "CSE-UAEL",
                "detail": "Two-stage estimation connected to unsupervised adaptive learning.",
                "paper_file": "2019-raza-cse-uael.pdf",
                "paper_title": (
                    "Covariate shift estimation based adaptive ensemble learning "
                    "for handling non-stationarity in motor imagery related "
                    "EEG-based brain-computer interface"
                ),
                "citation": "Raza et al. · Neurocomputing",
            },
            {
                "year": "Current",
                "title": "Auditable reconstruction",
                "detail": "Explicit alternatives expose decisions hidden by methodological ambiguity.",
                "paper_file": None,
                "paper_title": None,
                "citation": "Current repository stage",
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
