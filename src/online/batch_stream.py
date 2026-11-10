from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np

from src.online.replay import replay_feature_rows


@dataclass(frozen=True)
class BCIBatch:
    batch_index: int
    times: np.ndarray
    features: np.ndarray
    labels: np.ndarray | None = None


def make_validation_stream(
    x_validation,
    y_validation,
    batch_size,
    budget,
):
    x = np.asarray(x_validation, dtype=float)
    y = np.asarray(y_validation)

    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("Expected a 2-D feature matrix and 1-D labels")
    if len(x) != len(y):
        raise ValueError("Feature and label counts must match")
    if not 0 < budget <= len(x):
        raise ValueError(
            "budget must be positive and no more than the number of samples"
        )
    if budget < batch_size:
        raise ValueError("budget must be at least as large as batch_size")

    return (
        make_bci_dataflow(
            x[:budget],
            batch_size=batch_size,
        ),
        y[:budget],
    )


def make_bci_dataflow(
    features: np.ndarray,
    *,
    batch_size: int,
    labels: np.ndarray | None = None,
) -> Iterator[BCIBatch]:
    """Replay prepared BCI feature rows as ordered batches.

    Labels, if supplied, are for offline supervised replay.
    The consumer must predict before releasing them.
    """
    x = np.asarray(features, dtype=float)
    y = None if labels is None else np.asarray(labels)

    if x.ndim != 2 or len(x) == 0:
        raise ValueError("Expected a nonempty 2-D feature matrix")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if y is not None and (y.ndim != 1 or len(y) != len(x)):
        raise ValueError("Feature and label counts must match")

    times = np.arange(len(x), dtype=int)

    batch_features = []
    batch_times = []
    batch_index = 0

    for time, feature_row in replay_feature_rows(x, times):
        batch_features.append(feature_row)
        batch_times.append(time)

        if len(batch_features) == batch_size:
            indices = np.asarray(batch_times, dtype=int)

            yield BCIBatch(
                batch_index=batch_index,
                times=indices,
                features=np.stack(batch_features),
                labels=None if y is None else y[indices].copy(),
            )

            batch_features = []
            batch_times = []
            batch_index += 1

    # Final incomplete batch, if any.
    if batch_features:
        indices = np.asarray(batch_times, dtype=int)

        yield BCIBatch(
            batch_index=batch_index,
            times=indices,
            features=np.stack(batch_features),
            labels=None if y is None else y[indices].copy(),
        )
