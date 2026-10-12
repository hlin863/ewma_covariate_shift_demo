"""Bounded interactive Hotelling/GPR/kernel diagnostic laboratory."""
from flask import Blueprint, render_template, request
from src.adaptation.experiments.gpr_kernel import SCENARIOS, run_synthetic

gpr_kernel_bp = Blueprint('gpr_kernel', __name__)


@gpr_kernel_bp.route('/results/gpr-kernel', methods=['GET', 'POST'])
def laboratory():
    result, error = None, None
    scenario, seed = 'mean_shift', 42
    if request.method == 'POST':
        try:
            scenario = request.form.get('scenario', scenario)
            seed = int(request.form.get('seed', seed))
            result = run_synthetic(scenario, seed)
        except (ValueError, OverflowError) as exc:
            error = str(exc)
    return render_template('gpr_kernel.html', scenarios=SCENARIOS,
        scenario=scenario, seed=seed, result=result, error=error), (400 if error else 200)
