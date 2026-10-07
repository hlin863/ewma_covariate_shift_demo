"""Paper-grounded content model for the research home page."""

from __future__ import annotations

from src.web.process import PROCESS_PAGES


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
                "validation variants, non-adaptive/passive/active update policies, "
                "supervised adaptation, PWKNN pseudo-labelling, transductive knowledge-base "
                "updates and hidden-label post-run scoring. Dynamic CSE-UAEL ensemble growth, "
                "bounded-memory continual learning and the first full 2A/2B adaptive "
                "reproduction remain open milestones."
            ),
        },
        "processes": PROCESS_PAGES,
        "support": {
            "endpoint": "support.support_layer",
            "label": "Local RAG support layer",
            "description": "Ask about methods, code and saved evidence using the local repository assistant.",
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
                "title": "Update policy",
                "detail": (
                    "Never-update control; periodic/continuous passive adaptation; "
                    "warning, validated-shift and performance-drop active triggers"
                ),
                "status": "implemented",
            },
            {
                "number": "08",
                "title": "Label acquisition",
                "detail": (
                    "Supervised labels and confidence-bearing PWKNN pseudo-labels; "
                    "active-learning query scoring remains a separate utility"
                ),
                "status": "implemented",
            },
            {
                "number": "09",
                "title": "Transductive adaptation",
                "detail": (
                    "Confidence-gated knowledge-base growth, classifier versioning "
                    "and hidden-label post-run evaluation"
                ),
                "status": "implemented",
            },
            {
                "number": "10",
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
