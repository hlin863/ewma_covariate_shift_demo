"""Park et al. (2023) section 3.1 synthetic models A, B(i), C.

Reproducible new simulations from paper distributions; not the original
random realisations in Figure 3. Observations by channels output.
"""
import numpy as np


def simulate_park_model(model, *, n=1024, channels=50, change_at=512,
                        rho=0.5, seed=2023):
    """Generate AR mean-change (A), AR-dependence (B), or MA-dependence (C).

    The first changed observation has zero-based index change_at.
    None means no change. rho selects floor(rho * channels) series.
    """
    model = model.upper()
    if model not in ("A", "B", "C"):
        raise ValueError("model must be A, B, or C")
    if not isinstance(n, int) or n < 4 or not isinstance(channels, int) or channels < 2:
        raise ValueError("n >= 4 and channels >= 2 are required")
    if change_at is not None and (not isinstance(change_at, int) or not 1 <= change_at < n):
        raise ValueError("change_at must be None or between 1 and n-1")
    if not np.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("rho must be in [0, 1]")

    rng = np.random.default_rng(seed)
    count = int(np.floor(rho * channels)) if change_at is not None else 0
    affected = np.sort(rng.choice(channels, size=count, replace=False))
    flag = np.zeros(channels, dtype=bool)
    flag[affected] = True

    if model == "A":
        before_alpha = rng.uniform(-0.5, 0.5, channels)
        before_sigma = rng.uniform(0.5, 2, channels)
        after_alpha = before_alpha.copy()
        after_sigma = before_sigma.copy()
        mu = np.zeros(channels)
        after_alpha[affected] = rng.uniform(-0.5, 0.5, count)
        after_sigma[affected] = rng.uniform(0.5, 2, count)
        mu[affected] = rng.uniform(2, 5, count)
        eps = 2 * rng.normal(size=(n, channels))  # innovations N(0,4)
        if change_at is not None and count:
            eps[change_at:, affected] += mu[affected]
    else:
        before_alpha = rng.uniform(0.5, 1.0, channels)
        after_alpha = before_alpha.copy()
        after_alpha[affected] = rng.uniform(-0.5, 0.3, count)
        eps = rng.normal(size=(n, channels))

    x = np.empty_like(eps)
    x[0] = before_sigma * eps[0] if model == "A" else eps[0]
    for t in range(1, n):
        alpha = before_alpha.copy()
        if change_at is not None and t >= change_at:
            alpha[flag] = after_alpha[flag]
        if model == "A":
            sigma = before_sigma.copy()
            if change_at is not None and t >= change_at:
                sigma[flag] = after_sigma[flag]
            x[t] = alpha * x[t-1] + sigma * eps[t]
        elif model == "B":
            x[t] = alpha * x[t-1] + eps[t]
        else:
            x[t] = eps[t] + alpha * eps[t-1]
    return {"values": x, "model": model, "n": n, "channels": channels,
            "change_at": change_at, "rho": rho, "seed": seed,
            "affected_channels": affected.tolist()}
