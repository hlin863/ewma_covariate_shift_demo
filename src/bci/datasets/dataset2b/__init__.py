"""BCI Competition IV Dataset 2B reproduction helpers."""

from src.bci.datasets.dataset2b.diagnostics import (
    Dataset2BDiagnosticResult,
    run_dataset_2b_diagnostic,
)
from src.bci.datasets.dataset2b.evaluation_labels import (
    dataset_2b_evaluation_label_filename,
    load_dataset_2b_evaluation_labels,
    resolve_dataset_2b_evaluation_label_path,
)
from src.bci.datasets.dataset2b.development_pipeline import (
    Dataset2BDevelopmentFeaturePipelineResult,
    build_dataset_2b_development_fbcsp_features,
)
from src.bci.datasets.dataset2b.development_split import (
    Dataset2BCSEUAELProtocol,
    Dataset2BDevelopmentSplit,
    prepare_cse_uael_2019_protocol,
    split_dataset_2b_training_pool,
)
from src.bci.datasets.dataset2b.experiment import (
    PUBLISHED_2B_RESULTS,
    Dataset2BExperimentResult,
    Dataset2BFeaturePipelineResult,
    TrialFeatureResult,
    TrialSignalResult,
    build_dataset_2b_fbcsp_features,
    concatenate_trial_signals,
    extract_dataset_2b_trials,
    run_dataset_2b_subject,
)
from src.bci.datasets.dataset2b.reference import (
    BCI_2B_TABLE1_ROWS,
    bci_2b_table1_reference,
)

__all__ = [
    "BCI_2B_TABLE1_ROWS",
    "PUBLISHED_2B_RESULTS",
    "Dataset2BDiagnosticResult",
    "Dataset2BCSEUAELProtocol",
    "Dataset2BDevelopmentSplit",
    "Dataset2BDevelopmentFeaturePipelineResult",
    "Dataset2BExperimentResult",
    "Dataset2BFeaturePipelineResult",
    "TrialFeatureResult",
    "TrialSignalResult",
    "bci_2b_table1_reference",
    "build_dataset_2b_fbcsp_features",
    "build_dataset_2b_development_fbcsp_features",
    "concatenate_trial_signals",
    "dataset_2b_evaluation_label_filename",
    "load_dataset_2b_evaluation_labels",
    "resolve_dataset_2b_evaluation_label_path",
    "extract_dataset_2b_trials",
    "prepare_cse_uael_2019_protocol",
    "split_dataset_2b_training_pool",
    "run_dataset_2b_diagnostic",
    "run_dataset_2b_subject",
]
