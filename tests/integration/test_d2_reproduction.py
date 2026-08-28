from src.experiments.paper2015.d2 import (
    D2ExperimentConfig,
    run_d2_experiment,
    run_d2_table_3_experiment,
)
from src.simulation.jumping_mean import JumpingMeanConfig


def test_d2_experiment_runs_full_two_stage_pipeline() -> None:
    result = run_d2_experiment(
        D2ExperimentConfig(
            dataset=JumpingMeanConfig(random_seed=42),
            train_size=500,
        )
    )

    assert len(result.stream) == 5000
    assert result.detector.training_result.lambda_value > 0.0
    assert result.stage_1_evaluation.metrics.true_shift_count == 45
    assert result.stage_2_evaluation.metrics.true_shift_count == 45
    assert not result.detector.stage_1_results.empty
    assert "status" in result.detector.stage_2_results.columns


def test_d2_table_3_runs_all_three_methods_on_nine_shifts() -> None:
    result = run_d2_table_3_experiment(
        D2ExperimentConfig(
            dataset=JumpingMeanConfig(random_seed=42),
            train_size=500,
            lambda_mode="configured",
        )
    )

    assert result.table_3_shift_times == tuple(range(4101, 4902, 100))
    assert list(result.computed.index) == ["SD-EWMA", "TSSD-EWMA", "ICI-CDT"]
    assert set(result.computed.columns) == {
        "fp_percent",
        "fn_percent",
        "rci",
        "ct_seconds",
    }
    assert result.comparison.loc["ICI-CDT", "published_fn_percent"] == 55.4
    assert all(
        evaluation.metrics.true_shift_count == 9
        for evaluation in result.evaluations.values()
    )
