"""Diethe-inspired policy laboratory within the existing Flask application."""
from dataclasses import asdict
from flask import Blueprint, render_template, request
from src.adaptation.experiments.diethe import DietheConfig, run_synthetic

diethe_bp = Blueprint('diethe', __name__)


@diethe_bp.route('/results/diethe', methods=['GET', 'POST'])
def laboratory():
    values = asdict(DietheConfig())
    result, error = None, None
    if request.method == 'POST':
        try:
            for key in ('seed', 'trials', 'interval', 'performance_window'):
                values[key] = int(request.form.get(key, values[key]))
            for key in ('magnitude', 'accuracy_drop'):
                values[key] = float(request.form.get(key, values[key]))
            for key in ('scenario', 'update_scope'):
                values[key] = request.form.get(key, values[key])
            result = run_synthetic(DietheConfig(**values))
        except (ValueError, OverflowError) as exc:
            error = str(exc)
    return render_template('diethe.html', values=values, result=result, error=error), (400 if error else 200)
