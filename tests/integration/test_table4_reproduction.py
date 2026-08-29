from src.experiments.paper2015.table4 import (
    TABLE_4_METHODS,
    Table4ExperimentConfig,
    run_table_4_experiment,
)


def test_table4_pipeline_runs_d3_and_d4_all_five_methods() -> None:
    result = run_table_4_experiment(Table4ExperimentConfig(repetitions=2))

    assert list(result.computed.index.get_level_values("dataset").unique()) == ["D3", "D4"]
    assert list(result.computed.loc["D3"].index) == list(TABLE_4_METHODS)
    assert set(result.computed.columns) == {
        "fp_percent", "fn_percent", "rci", "ct_seconds"
    }
    assert result.comparison.loc[("D3", "MSD-EWMA"), "published_fp_percent"] == 5.0
    assert result.metadata["raw_control_limit"] == 22.67
    assert result.metadata["pca_control_limit"] == 10.58
    assert result.metadata["pca_components"] == 2
    assert result.metadata["stage_2_window_size"] == 25
    for stream in result.representative_streams.values():
        assert stream.loc[stream["true_shift"] == 1, "time"].tolist() == [101, 201]
