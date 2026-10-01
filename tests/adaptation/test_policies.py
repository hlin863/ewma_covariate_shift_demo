from src.adaptation.policies import AdaptationContext, RetrainOnPerformanceDrop


def test_performance_rule_requires_full_window_and_cooldown():
    rule = RetrainOnPerformanceDrop(.9, .1, 20)
    assert not rule.should_update(AdaptationContext(20, 20, recent_accuracy=None, trials_since_update=21))
    assert not rule.should_update(AdaptationContext(20, 20, recent_accuracy=.5, trials_since_update=1))
    assert not rule.should_update(AdaptationContext(20, 20, recent_accuracy=.8, trials_since_update=21))
    assert rule.should_update(AdaptationContext(20, 20, recent_accuracy=.7, trials_since_update=21))
