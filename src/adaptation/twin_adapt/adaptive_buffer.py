"""Maintain selected, released-label observations with explicit max sample count."""
from dataclasses import dataclass, field
import numpy as np
from src.online.batch_stream import BCIBatch


@dataclass
class AdaptiveBuffer:
    max_trials: int
    _entries: list[tuple[int, np.ndarray, object]] = field(default_factory=list)
    _seen: set[int] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, batch: BCIBatch, labels: np.ndarray) -> int:
        """Add without duplicate trials; bounded by capacity (latest accepted trials kept)."""
        y = np.asarray(labels)
        if y.ndim != 1 or len(y) != len(batch.times):
            raise ValueError("labels must align to batch times")
        for t, x, label in zip(batch.times, batch.features, y):
            index = int(t)
            if index in self._seen:
                raise ValueError(f"duplicate trial in adaptive buffer: {index}")
            self._entries.append((index, x.copy(), label))
            self._seen.add(index)
        overflow = max(0, len(self._entries) - self.max_trials)
        for t, _, _ in self._entries[:overflow]:
            self._seen.remove(t)
        if overflow:
            self._entries = self._entries[overflow:]
        return overflow

    def arrays(self) -> tuple[np.ndarray, np.ndarray, list[int]]:
        if not self._entries:
            raise ValueError("Cannot retrain on an empty buffer")
        return (np.stack([row[1] for row in self._entries]),
                np.asarray([row[2] for row in self._entries]),
                [row[0] for row in self._entries])

    def clear(self) -> None:
        self._entries.clear()
        self._seen.clear()
