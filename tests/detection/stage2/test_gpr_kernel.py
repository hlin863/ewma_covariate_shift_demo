import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import rbf_kernel
from src.detection.stage2.gpr_kernel_hotelling import (
    GPRKernelConfig, fit_gpr_kernel, kernel_mmd, validate_gpr_kernel_alarms,
)
from src.detection.stage2.paper_two_sample_hotelling import (
    PaperTwoSampleHotellingConfig, validate_paper_two_sample_alarms,
)
from src.adaptation.policies import AdaptationContext, RetrainOnGPRKernelRecommendation


def fitted(seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(80, 2))
    calibration = rng.normal(size=(100, 2))
    return fit_gpr_kernel(x, np.sin(x[:, 0]), calibration,
                          GPRKernelConfig(window_size=10, permutations=99))


def test_mmd_matches_direct_unbiased_formula_and_is_reproducible():
    a = np.array([[0.], [1.], [2.]])
    b = np.array([[3.], [4.]])
    aa, bb, ab = [rbf_kernel(x, y, gamma=0.5) for x, y in [(a,a),(b,b),(a,b)]]
    expected = (aa.sum()-np.trace(aa))/6 + (bb.sum()-np.trace(bb))/2 - 2*ab.mean()
    observed = kernel_mmd(a,b,gamma=0.5,permutations=99)
    assert observed[0] == pytest.approx(expected)
    assert observed == kernel_mmd(a,b,gamma=0.5,permutations=99)
    assert 0 < observed[1] <= 1


def test_kernel_detects_equal_mean_variance_change():
    a = np.tile([[-1.], [1.]], (30,1))
    b = np.tile([[-4.], [4.]], (30,1))
    assert a.mean() == b.mean()
    assert kernel_mmd(a,b,gamma=0.25,permutations=199)[1] < 0.05


def test_gpr_extrapolation_increases_std_without_refitting():
    model = fitted()
    kernel_before = str(model.regressor.kernel_)
    _, near = model.regressor.predict([[0,0]], return_std=True)
    _, far = model.regressor.predict([[20,20]], return_std=True)
    assert far[0] > near[0]
    assert far[0] > model.uncertainty_limit
    assert str(model.regressor.kernel_) == kernel_before


def test_baseline_unchanged_and_future_window_not_used_early():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60,2)); x[30:] += 5
    times = np.arange(60)
    warnings = pd.DataFrame({'time':[2,29,58], 'stage_1_alarm':True})
    model = fitted()
    baseline = validate_paper_two_sample_alarms(x,times,warnings,PaperTwoSampleHotellingConfig(window_size=10))
    result = validate_gpr_kernel_alarms(x,times,warnings,model)
    pd.testing.assert_frame_equal(result[baseline.columns],baseline)
    assert result.iloc[0].diagnostic_status == 'unavailable'
    assert result.iloc[2].diagnostic_status == 'unavailable'
    assert result.iloc[1].validation_time == 39
    assert result.iloc[1].adaptation_recommended
    truncated = validate_gpr_kernel_alarms(x[:39],times[:39],warnings.iloc[:2],model)
    assert truncated.iloc[1].status == 'pending_current_window'
    assert not truncated.iloc[1].adaptation_recommended
    # Future observations outside both test windows cannot alter this event.
    changed = x.copy(); changed[40:] += 100
    stable = validate_gpr_kernel_alarms(changed,times,warnings,model)
    assert stable.iloc[1].mmd_p_value == result.iloc[1].mmd_p_value
    assert stable.iloc[1].gpr_mean_std == result.iloc[1].gpr_mean_std


def test_policy_respects_recorded_availability():
    policy = RetrainOnGPRKernelRecommendation()
    record = {'diagnostic_status':'evaluated','validation_time':39,'adaptation_recommended':True}
    assert not policy.should_update(AdaptationContext(29,29,validation_record=record))
    assert policy.should_update(AdaptationContext(39,39,validation_record=record))
    assert not policy.should_update(AdaptationContext(40,40))


@pytest.mark.parametrize('kwargs',[{'noise_variance':float('nan')},{'length_scale':0},
                                  {'seed':-1},{'permutations':1},{'window_size':1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        GPRKernelConfig(**kwargs)


def test_training_and_calibration_contracts():
    with pytest.raises(ValueError,match='five complete windows'):
        fit_gpr_kernel(np.arange(40).reshape(20,2),np.arange(20),np.ones((10,2)))
    with pytest.raises(ValueError,match='training_targets'):
        fit_gpr_kernel(np.arange(40).reshape(20,2),[np.nan]*20,np.ones((100,2)))
