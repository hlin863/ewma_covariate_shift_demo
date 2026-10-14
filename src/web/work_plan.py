"""Evidence-first WP1–WP6 research work-plan pages.

Statements represent recorded implementation and research plans, not live benchmark results.
"""
from __future__ import annotations

from flask import Blueprint, abort, render_template
from src.web.home import build_home_page_model

work_plan_bp = Blueprint("work_plan", __name__, url_prefix="/work-plan")

DETAILS = {
 "wp1": {
  "goal": "Design the CPD–continual-learning–digital-twin architecture and research interfaces.",
  "accomplished": [
   ("Canonical separation of EEG processing, Stage-I monitoring, Stage-II validators and supervised/transductive adaptation", "src/detection/, src/bci/, src/adaptation/; Notion canonical architecture diary, 3 October"),
   ("Research process and results pages, architecture map, local repository-grounded RAG assistance", "src/web/process.py, src/web/support.py; Notion research architecture diary, 28 September"),
   ("Explicit experimental variants rather than treating algorithmic ambiguity as established reproduction", "docs/table1_algorithm_gap_analysis.md"),
  ],
  "next_tasks": [
   ("Define common ShiftEvidence, ValidationDecision and AdaptationEvent schemas", "Specify timing, status, configuration IDs and provenance; add compatibility tests"),
   ("Document CPD-to-adaptation interfaces", "Trace an actual event from warning through validation and model update"),
   ("Specify a digital-twin boundary", "State, synchronisation, disagreement, safety and rollback contracts; do not describe it as implemented"),
  ],
  "boundary": "The modular web application and research maps exist; a unified CPD–CL–twin runtime does not.",
  "notion": "https://app.notion.com/p/3f1410db31c88182b4a7ca2acf760a85",
  "code": [("Architecture process", "src/web/process.py"), ("Detector namespace", "src/detection/")],
 },
 "wp2": {
  "goal": "Benchmark covariate and structural-change detectors under controlled, reproducible protocols.",
  "accomplished": [
   ("BCI 2A/2B preprocessing and separate Stage-I warning / Stage-II validation variants", "src/bci/, src/detection/; Notion WP2 audit, 8 October"),
   ("Park channelwise mean-CUSUM, smoothed curves and candidate peaks", "src/detection/stage1/park_mean_cusum.py; Notion diary, 14 October"),
   ("Park A/B(i)/C synthetic families, notebook explorations and isolated shift visualisations", "src/simulation/park_structural_models.py; scripts/generate_slide22_structural_visuals.py"),
   ("GPR uncertainty and kernel MMD diagnostic prototypes", "src/detection/stage2/gpr_kernel_hotelling.py"),
  ],
  "next_tasks": [
   ("Reconcile Raza EEG Table I numerical gaps", "Compare per-warning Stage-I/II inputs, windows, p-values and run manifests against the published protocol"),
   ("Complete Park confirmation and second-order comparisons", "Implement unimodality / smoothed-peak agreement; benchmark mean, variance and AR shifts"),
   ("Calibrate stream-level false alarms and detector comparisons", "Matched false alarm budgets, repeated seeds, delay and power with serial-dependence-aware tests"),
   ("Restore full web CI", "Investigate custom Park CSV channel-label regression without weakening test assertions"),
  ],
  "boundary": "Existing simulations and argmax peaks are not confirmed change-point detections; published numerical agreement remains unresolved.",
  "notion": "https://app.notion.com/p/3f9410db31c88134a639d37451ff5e9a",
  "code": [("Park CUSUM", "src/detection/stage1/park_mean_cusum.py"), ("Synthetic generator", "src/simulation/park_structural_models.py"), ("Reproduction audit", "docs/table1_algorithm_gap_analysis.md")],
 },
 "wp3": {
  "goal": "Evaluate continual-learning strategies for nonstationary classification with realistic adaptation costs.",
  "accomplished": [
   ("Six-policy laboratory comparing non-adaptive, passive and event-triggered updates", "src/adaptation/; docs/diethe_experiments.md"),
   ("ContinuousRetrain and trials-since-last-update event bookkeeping", "Notion continual-learning policy integration, 7 October"),
   ("Supervised and confidence-gated PWKNN transductive adaptation prototypes", "src/adaptation/; results/transductive-adaptation"),
  ],
  "next_tasks": [
   ("Run matched, repeated passive-versus-active policy experiments", "Compare no shift, covariate shift and changed conditional relationships"),
   ("Implement bounded replay and retention measurements", "Measure forgetting, memory budget and historical/new-task accuracy"),
   ("Measure costs and update quality", "Timing, update frequency, computational or energy costs and acceptance/rollback gates"),
   ("Address delayed or unavailable labels", "Make feedback availability explicit in each policy comparison"),
  ],
  "boundary": "Adaptation policies run as prototypes; bounded replay and cross-policy generalisation have not been established.",
  "notion": "https://app.notion.com/p/3f3410db31c88104ad54dc6803532add",
  "code": [("Diethe laboratory", "docs/diethe_experiments.md"), ("Adaptation package", "src/adaptation/")],
 },
 "wp4": {
  "goal": "Study safe joint adaptation between an observed system and its digital twin.",
  "accomplished": [
   ("Synthetic nonstationary-stream generators and monitoring components that can support a future twin testbed", "src/simulation/ and src/detection/"),
   ("Architectural separation between shift evidence and model update decisions", "src/detection/, src/adaptation/; Notion canonical architecture diary"),
  ],
  "next_tasks": [
   ("Specify twin state and versioned observation exchange", "Define observable signals, latent state assumptions and update cadence"),
   ("Prototype synchronisation and disagreement experiments", "Measure state mismatch, monitoring lag and stale-model effects"),
   ("Specify safe update and rollback criteria", "Test simulated interventions before introducing joint updates"),
  ],
  "boundary": "Supporting simulations exist, but there is no verified operational digital-twin co-evolution subsystem.",
  "notion": "https://app.notion.com/p/3f1410db31c8817a97a0e8c553bfed77",
  "code": [("Simulation package", "src/simulation/")],
 },
 "wp5": {
  "goal": "Test whether detection and adaptation insights transfer beyond EEG to maintenance and photonic streams.",
  "accomplished": [
   ("BCI Competition IV 2A/2B input and preprocessing pathways", "src/bci/"),
   ("Maintenance data inspection and synthetic shift scenarios", "src/web/data_structures.py, src/simulation/"),
  ],
  "next_tasks": [
   ("Define comparable cross-domain benchmarks", "Specify labels, features, shifts and equivalent metrics by domain"),
   ("Evaluate on maintenance streams", "Keep temporal splits leakage-free and report calibration, delay and accuracy"),
   ("Assess photonic data readiness", "Do not claim photonic transfer unless dataset, feature protocol and results exist"),
  ],
  "boundary": "Dataset access and exploration do not prove cross-domain transfer or photonic validation.",
  "notion": "https://app.notion.com/p/3f1410db31c881bbb473ca4e295311f2",
  "code": [("Data structures", "src/web/data_structures.py"), ("BCI processing", "src/bci/")],
 },
 "wp6": {
  "goal": "Convert the audited literature-to-code trail into reproducible academic outputs and engagement.",
  "accomplished": [
   ("Living computational review and methods-to-evidence navigation", "docs/living_literature_review/, src/web/"),
   ("Dated Notion project diary, academic progress slides and committed synthetic figures", "Notion project diary; presentations/; outputs/structural_changes/slide22/"),
   ("Published-versus-computed diagnostics with explicit evidence boundaries", "outputs/metrics/; docs/table1_algorithm_gap_analysis.md"),
  ],
  "next_tasks": [
   ("Produce a verified results ledger", "Link every numerical claim to code version, configuration, dataset and saved outputs"),
   ("Resolve discrepancies before publication claims", "Distinguish implemented method, validated benchmark and numerical reproduction"),
   ("Prepare reproducible research release and manuscript sections", "Freeze benchmark configurations, archive figures and record limitations"),
  ],
  "boundary": "Documentation and presentation outputs exist; peer-reviewed publication or validated reproduction is not claimed.",
  "notion": "https://app.notion.com/p/3e2410db31c881fb82c5c85cc0eb637c",
  "code": [("Living review", "docs/living_literature_review/"), ("Gap register", "docs/table1_algorithm_gap_analysis.md")],
 },
}


@work_plan_bp.get("/")
def index():
    return render_template("work_plan_index.html", packages=build_home_page_model()["work_packages"])


@work_plan_bp.get("/<wp_id>")
def detail(wp_id: str):
    if wp_id not in DETAILS:
        abort(404)
    items = build_home_page_model()["work_packages"]
    package = next(item for item in items if item["id"] == wp_id)
    return render_template("work_plan_detail.html", package=package, detail=DETAILS[wp_id], packages=items)
