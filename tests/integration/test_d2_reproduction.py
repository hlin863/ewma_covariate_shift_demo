from src.experiments.paper2015.d2 import D2ExperimentConfig, run_d2_experiment
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
