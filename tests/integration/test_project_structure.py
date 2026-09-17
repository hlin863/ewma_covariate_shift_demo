"""Architecture-level smoke tests for the refactored package structure."""


def test_canonical_bci_namespace_exposes_core_components() -> None:
    from src.bci.data import BCISessionData
    from src.bci.fbcsp import (
        CSE_UAEL_FILTER_BANK,
        ONLINE_BCI_2018_FILTER_BANK,
        FBCSPModel,
        PAPER_FILTER_BANK,
    )

    assert BCISessionData.__name__ == "BCISessionData"
    assert FBCSPModel.__name__ == "FBCSPModel"
    assert len(CSE_UAEL_FILTER_BANK) == 10
    assert ONLINE_BCI_2018_FILTER_BANK == ((8.0, 12.0), (16.0, 24.0))
    assert PAPER_FILTER_BANK is CSE_UAEL_FILTER_BANK


def test_dataset_namespaces_expose_reproduction_entry_points() -> None:
    from src.bci.datasets.dataset2a import (
        build_dataset_2a_development_fbcsp_features,
        extract_dataset_2a_trials,
        split_dataset_2a_session1,
    )
    from src.bci.datasets.dataset2b import (
        build_dataset_2b_fbcsp_features,
        extract_dataset_2b_trials,
    )

    assert callable(build_dataset_2a_development_fbcsp_features)
    assert callable(extract_dataset_2a_trials)
    assert callable(split_dataset_2a_session1)
    assert callable(build_dataset_2b_fbcsp_features)
    assert callable(extract_dataset_2b_trials)


def test_detection_namespace_groups_cse_stages() -> None:
    from src.detection import CSEConfig, run_cse
    from src.detection.stage1 import fit_sd_ewma
    from src.detection.stage2 import validate_algorithm1_alarms

    assert CSEConfig.__name__ == "CSEConfig"
    assert callable(run_cse)
    assert callable(fit_sd_ewma)
    assert callable(validate_algorithm1_alarms)


def test_reporting_and_web_namespaces_are_importable() -> None:
    from src.reporting.table1 import Table1Row
    from src.web.dashboard import app

    assert Table1Row.__name__ == "Table1Row"
    assert app.name == "src.web.dashboard"


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
