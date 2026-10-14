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
        "wp_updated": "14 October 2026",
        "work_packages": (
            {
                "id": "wp1", "number": "WP1", "title": "Theory & architecture",
                "status": "in progress", "stage": "Foundation",
                "evidence": "Modular BCI, Stage-I/II CPD and adaptation code; research process and evidence navigation.",
                "next": "Formalise the CPD–continual-learning–digital-twin interface and shared event contracts.",
                "links": (("Explore process", "/process/covariate-shift"),),
                "notion_url": "https://app.notion.com/p/3f1410db31c88182b4a7ca2acf760a85",
            },
            {
                "id": "wp2", "number": "WP2", "title": "CPD benchmarking",
                "status": "in progress", "stage": "Experiments",
                "evidence": "EWMA/Hotelling variants, Park CUSUM smoothing and models A/B/C, isolated structural-change plots and GPR/MMD experiments.",
                "next": "Resolve BCI reproduction discrepancies, verify Park confirmation and calibrate matched multi-seed detector tests.",
                "links": (("View results", "/results"), ("Park inspection", "/data-structures/park")),
                "notion_url": "https://app.notion.com/p/3f3410db31c881c5b292dc502b23953b",
            },
            {
                "id": "wp3", "number": "WP3", "title": "Continual learning",
                "status": "in progress", "stage": "Prototypes",
                "evidence": "Six-policy laboratory, supervised and PWKNN transductive updates, versioned evaluation evidence.",
                "next": "Add bounded replay, forgetting and resource-cost experiments; validate across repeated seeds.",
                "links": (("Policy laboratory", "/results/diethe"), ("Adaptive learning", "/process/adaptive-learning")),
                "notion_url": "https://app.notion.com/p/3f3410db31c88104ad54dc6803532add",
            },
            {
                "id": "wp4", "number": "WP4", "title": "Digital twin co-evolution",
                "status": "planned", "stage": "Integration",
                "evidence": "Synthetic stream and monitoring infrastructure provide preliminary components, not an operational twin.",
                "next": "Specify twin state, synchronisation, disagreement and safety/rollback behaviour; build an initial testbed.",
                "links": (("Review architecture", "/process/adaptive-learning"),),
                "notion_url": "https://app.notion.com/p/3f1410db31c8817a97a0e8c553bfed77",
            },
            {
                "id": "wp5", "number": "WP5", "title": "Cross-domain evaluation",
                "status": "planned", "stage": "Transfer",
                "evidence": "BCI 2A/2B pipelines and NASA C-MAPSS data inspection exist; comparative transfer is not verified.",
                "next": "Define matched BCI/maintenance benchmarks and assess availability of photonic-domain data.",
                "links": (("Inspect datasets", "/data-structures"),),
                "notion_url": "https://app.notion.com/p/3f1410db31c881bbb473ca4e295311f2",
            },
            {
                "id": "wp6", "number": "WP6", "title": "Synthesis & engagement",
                "status": "in progress", "stage": "Research record",
                "evidence": "Living literature review, dated Notion diary, saved reproducibility outputs and research presentation.",
                "next": "Reconcile numerical claims and prepare reproducible release, manuscripts and stakeholder review.",
                "links": (("Explore results", "/results"),),
                "notion_url": "https://app.notion.com/p/3e2410db31c881fb82c5c85cc0eb637c",
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
                "year": "2023",
                "title": "Continual learning for predictive maintenance",
                "detail": (
                    "Hurtado et al. frame classical adaptation as active drift-triggered "
                    "updates versus passive continual updates and motivate realistic "
                    "non-stationary predictive-maintenance benchmarks."
                ),
                "paper_file": None,
                "paper_title": None,
                "citation": "Hurtado et al. · Intelligent Systems with Applications",
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
