"""Paper-grounded content model for the research home page."""

from __future__ import annotations


def build_home_page_model() -> dict[str, object]:
    """Return the research narrative, implementation status and page catalogue."""

    return {
        "scope": {
            "headline": (
                "Detection through confidence-gated transductive adaptation is "
                "implemented and testable."
            ),
            "detail": (
                "The repository now separates Stage-I warnings, explicit Stage-II "
                "validation variants, supervised adaptation, PWKNN pseudo-labelling, "
                "transductive knowledge-base updates and hidden-label post-run scoring. "
                "Dynamic CSE-UAEL ensemble growth, weighted voting and the first full "
                "2A/2B adaptive reproduction remain open milestones."
            ),
        },
        "application_map": {
            "root": {
                "title": "CSE research application",
                "detail": (
                    "The home page is the navigation root. Evidence flows from "
                    "dataset/protocol inspection through detection and adaptation "
                    "analysis into generated results and implementation verification."
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
                            "endpoint": "data_structures.overview",
                            "label": "Data structures",
                            "kind": "Exploratory data analysis",
                            "description": (
                                "Choose BCI 2A/2B, Turbofan, algae, clinical metadata "
                                "or synthetic data to inspect loaded structures and distributions."
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
                    "title": "Detection and representation analysis",
                    "detail": (
                        "Inspect the primary reproduction, Stage-I/Stage-II behaviour "
                        "and complementary score/residual monitoring experiments."
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
                    "title": "Adaptation and continual learning",
                    "detail": (
                        "Inspect when detector evidence should trigger model change and "
                        "how the project separates supervised from transductive updates."
                    ),
                    "relationship": "guides model updates",
                    "pages": (
                        {
                            "endpoint": "diethe.laboratory",
                            "label": "Diethe policy laboratory",
                            "kind": "Continual-learning experiment",
                            "description": (
                                "Compare update policies, evidence and adaptation cost "
                                "without conflating policy choice with detector logic."
                            ),
                        },
                        {
                            "endpoint": "decision_models.explorer",
                            "label": "BCI decision model explorer",
                            "kind": "Classifier and pseudo-labelling evidence",
                            "description": (
                                "Trace Decision Tree, KNN, PWKNN and linear-SVM "
                                "decisions on the same saved FBCSP trial evidence."
                            ),
                        },
                    ),
                },
                {
                    "number": "04",
                    "title": "Evidence and implementation assurance",
                    "detail": (
                        "Consolidate generated artifacts and verify that the code "
                        "contracts supporting those artifacts continue to pass."
                    ),
                    "relationship": "produces auditable evidence",
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
                "detail": "PCA score monitoring plus complementary residual-space evidence",
                "status": "implemented",
            },
            {
                "number": "04",
                "title": "Stage I",
                "detail": "SD/MSD-EWMA warnings with complementary score/residual monitoring",
                "status": "implemented",
            },
            {
                "number": "05",
                "title": "Stage II",
                "detail": "K-S plus explicit training-reference, two-sample and retrospective Hotelling paths",
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
                "title": "Transductive adaptation",
                "detail": (
                    "PWKNN pseudo-labelling, confidence-gated knowledge-base growth "
                    "and hidden-label post-run evaluation"
                ),
                "status": "implemented",
            },
            {
                "number": "08",
                "title": "Dynamic ensemble",
                "detail": (
                    "CSE-triggered classifier growth and weighted CSE-UAEL voting"
                ),
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
                "detail": "Two-stage shift evidence connected to synchronous supervised BCI adaptation.",
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
                "detail": "CSE, PWKNN transduction and adaptive ensemble learning for non-stationary MI-BCI.",
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
                "title": "Auditable adaptive framework",
                "detail": (
                    "Explicit Stage-II alternatives, complementary monitoring, "
                    "policy experiments and supervised/transductive adaptation."
                ),
                "paper_file": None,
                "paper_title": None,
                "citation": "Current repository stage",
            },
        ),
        "contributions": (
            "A modular reconstruction of cue extraction, FBCSP, PCA, EWMA warning and multivariate validation.",
            "Published-versus-computed subject results with generated diagnostics rather than hard-coded claims.",
            "Methodological ambiguity treated as an experimental variable: split, PCA retention, control limits and Stage-II interpretation.",
            "Evaluation-label-free transductive adaptation with PWKNN confidence gating and separate offline ground-truth scoring.",
            "A living computational literature review linking source studies to algorithmic, implementation and evidence lineage.",
        ),
        "limitations": (
            "The default H=10 and d=20 Stage-II configuration is dimensionally ineligible because 2H − d − 1 is negative.",
            "Passing implementation tests do not establish exact agreement with published EEG warning and validation counts.",
            "PWKNN-triggered transductive refitting is implemented and regression-tested, but dynamic ensemble growth, weighted voting and the first end-to-end 2A/2B adaptive reproduction remain future work.",
        ),
    }
