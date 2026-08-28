import numpy as np

from src.detection.stage1.sd_ewma import SD_EWMA_Config
from src.detection.stage2 import Stage2Config
from src.detection.two_stage import TSSDEWMAConfig, run_tssd_ewma
from src.stage_2 import Stage2Config as LegacyStage2Config
from src.tssd_ewma import run_tssd_ewma as legacy_run_tssd_ewma


def test_legacy_imports_resolve_to_canonical_implementations() -> None:
    assert LegacyStage2Config is Stage2Config
    assert legacy_run_tssd_ewma is run_tssd_ewma


def test_configured_lambda_and_variance_mode_are_preserved() -> None:
    rng = np.random.default_rng(4)
    training = rng.normal(0.0, 1.0, 100)
    testing = np.concatenate([rng.normal(0.0, 1.0, 50), rng.normal(4.0, 1.0, 50)])
    result = run_tssd_ewma(
        training,
        testing,
        np.arange(100, 200),
        TSSDEWMAConfig(
            stage_1=SD_EWMA_Config(
                lambda_value=0.4,
                variance_update_mode="frozen",
            ),
            stage_2=Stage2Config(before_size=10, after_size=10),
            lambda_mode="configured",
        ),
    )

    assert result.training_result.lambda_value == 0.4
    assert set(result.stage_1_results["variance_update_mode"]) == {"frozen"}
