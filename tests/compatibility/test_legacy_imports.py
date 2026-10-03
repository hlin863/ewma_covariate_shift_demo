"""Legacy imports must resolve to the same canonical objects."""

def test_legacy_flat_imports_remain_compatible() -> None:
    from src.bci_data import BCISessionData as LegacySession
    from src.bci.data import BCISessionData as CanonicalSession
    from src.fbcsp import FBCSPModel as LegacyFBCSP
    from src.bci.fbcsp import FBCSPModel as CanonicalFBCSP
    from src.table1_reproduction import Table1Row as LegacyTable1Row
    from src.reporting.table1 import Table1Row as CanonicalTable1Row

    assert LegacySession is CanonicalSession
    assert LegacyFBCSP is CanonicalFBCSP
    assert LegacyTable1Row is CanonicalTable1Row



def test_two_stage_legacy_imports_resolve_to_canonical_implementations() -> None:
    from src.detection.stage2 import Stage2Config
    from src.detection.two_stage import run_tssd_ewma
    from src.stage_2 import Stage2Config as LegacyStage2Config
    from src.tssd_ewma import run_tssd_ewma as legacy_run_tssd_ewma

    assert LegacyStage2Config is Stage2Config
    assert legacy_run_tssd_ewma is run_tssd_ewma



def test_adaptation_history_imports_resolve_to_canonical_modules() -> None:
    from src.adaptation.evaluation import evaluate_unsupervised_adaptation as canonical_eval
    from src.adaptation.pseudo_labelling import PWKNNPseudoLabeler as canonical_pwknn
    from src.adaptation.supervised import (
        PWKNNPseudoLabeler as historical_pwknn,
        evaluate_unsupervised_adaptation as historical_eval,
        run_unsupervised_adaptation as historical_run,
    )
    from src.adaptation.transductive import run_unsupervised_adaptation as canonical_run

    assert historical_pwknn is canonical_pwknn
    assert historical_eval is canonical_eval
    assert historical_run is canonical_run


def test_stage2_history_imports_resolve_to_canonical_modules() -> None:
    from src.cse_algorithm1_stage_2 import calculate_training_reference_hotelling as legacy_training
    from src.cse_paper_stage_2 import calculate_paper_two_sample_hotelling as legacy_paper
    from src.multivariate_stage_2 import calculate_hotelling_t_squared as legacy_retro
    from src.detection.stage2.paper_two_sample_hotelling import calculate_paper_two_sample_hotelling as canonical_paper
    from src.detection.stage2.retrospective_hotelling import calculate_hotelling_t_squared as canonical_retro
    from src.detection.stage2.training_reference_hotelling import calculate_training_reference_hotelling as canonical_training

    assert legacy_training is canonical_training
    assert legacy_paper is canonical_paper
    assert legacy_retro is canonical_retro
