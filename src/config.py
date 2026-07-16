from dataclasses import dataclass


@dataclass(frozen=True)
class GaussianConfig:
    random_seed: int = 42

    n_train: int = 500
    n_normal_test: int = 200
    n_shifted_test: int = 300

    mean_before: float = 0.0
    mean_after: float = 2.0

    std_before: float = 1.0
    std_after: float = 1.0

    @property
    def train_end(self) -> int:
        return self.n_train

    @property
    def shift_point(self) -> int:
        return self.n_train + self.n_normal_test