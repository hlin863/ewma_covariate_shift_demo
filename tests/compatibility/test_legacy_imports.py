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
