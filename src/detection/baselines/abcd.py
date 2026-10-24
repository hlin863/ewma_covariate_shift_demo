"""Annotated ABCD teaching baseline based on Heyden et al. (2024).

Paper: 'Adaptive Bernstein change detector for high-dimensional data streams'
Section 4.2: Eq. (5) two-sample Bernstein bound, Eq. (8) split search.
Section 4.2.1: Eq. (12) recommended approximate kappa selection.

This module deliberately implements a *batch teaching calculation* only:
- It does not implement encoder-decoder reconstruction or online window resets.
- It plugs empirical variances into an inequality originally stated with
  population variances. The result is a heuristic score, not a rigorously
  calibrated p-value or a stream-wide false-alarm guarantee.
"""

import numpy as np


def bernstein_two_sample_upper_bound(n1, n2, v1, v2, eps, M, kappa):
    """Compute the two-sample Bernstein upper-bound expression (paper Eq. 5).

    Under H0: mu_1 = mu_2, the paper bounds

        P(|mu_hat_1 - mu_hat_2| >= eps)
          <= 2 exp{-n1 (kappa*eps)^2 / [2(v1 + kappa*M*eps/3)]}
           + 2 exp{-n2 ((1-kappa)*eps)^2
                   / [2(v2 + (1-kappa)*M*eps/3)]}.

    Args:
        n1, n2: Sizes of the two loss subwindows (positive integers).
        v1, v2: Variances of reconstruction losses in each subwindow.
            The theorem uses population variances; scan_candidate_splits()
            supplies empirical sample-variance estimates as a plug-in.
        eps: Absolute difference between the two observed sample means.
        M: Bound on |L - E[L]| required by Bernstein's inequality; choosing
            an empirically small M without this property forfeits the bound.
        kappa: Share of eps allocated to subwindow 1 (0 < kappa < 1).

    Returns:
        The *uncapped* sum of two upper-bound terms, in (0, 4] for the
        inputs used here. This is not an exact probability and may exceed 1.

    Assumptions for the original bound include independent, bounded losses
    and equality of their population means under H0.
    """
    if min(n1, n2) < 2 or M <= 0 or not 0 < kappa < 1:
        raise ValueError("Need two observations per side, positive M, 0<kappa<1")
    if min(v1, v2, eps) < 0:
        raise ValueError("Variances and epsilon must be non-negative")

    # If the observed difference eps is zero, each exponential in Eq. (5)
    # equals exp(0) = 1 (for positive variances); the expression is 2 + 2 = 4.
    # Do NOT interpret this loose upper bound as a probability of 4.
    if eps == 0:
        return 4.0  # limiting loose upper bound (NOT an exact probability)

    # Eq. (7): apply the union bound after splitting the deviation threshold:
    # eps = kappa*eps + (1-kappa)*eps.
    e1, e2 = kappa * eps, (1.0 - kappa) * eps

    # Eq. (6), applied to subwindow 1 with deviation threshold e1:
    # P(|mu_hat_1 - mu_1| >= e1) <= 2 exp[-n1*e1^2 / (2*(v1+M*e1/3))].
    # The leading 2 accounts for the positive and negative deviation tails.
    b1 = 2 * np.exp(-n1 * e1**2 / (2 * (v1 + M * e1 / 3)))

    # Eq. (6), applied to subwindow 2 with deviation threshold e2:
    # P(|mu_hat_2 - mu_2| >= e2) <= 2 exp[-n2*e2^2 / (2*(v2+M*e2/3))].
    b2 = 2 * np.exp(-n2 * e2**2 / (2 * (v2 + M * e2 / 3)))

    # Eq. (5): sum b1 and b2 by the union bound; return the raw bound.
    return float(b1 + b2)


def scan_candidate_splits(losses, *, min_side=2, M=1., delta=.05):
    """Scan candidate split points for the smallest Eq. (5) score (Eq. 8).

    Let the *current* (batch) window be W_t = (L_1, ..., L_t).
    For each eligible k, split it into

        W_before(k) = (L_1, ..., L_k),      n1 = k
        W_after(k)  = (L_{k+1}, ..., L_t), n2 = t-k.

    Eq. (4) gives the two sample means; the observed deviation is

        eps_k = |mean(W_before(k)) - mean(W_after(k))|.

    Evaluate Eq. (5) for each k using eps_k, and implement Eq. (8) as

        p_score = min_k B(k),     k_star = argmin_k B(k).

    The paper's Eq. (12) recommends approximately
        kappa = n2 / (n1+n2), restricted to [0.05, 0.95].

    IMPORTANT: 'illustrative_flag' is only the numerical rule p_score < delta.
    It is *not* a rigorously calibrated family-wise or sequential test: the
    same observations are reused over candidate splits, and the procedure
    does not adjust for repeated evaluations over time. The function also
    does not implement ABCD Algorithm 2's adaptive window and retraining.
    """
    losses = np.asarray(losses, dtype=float)
    if not np.all(np.isfinite(losses)) or len(losses) < 2 * min_side:
        raise ValueError("Need finite losses and two eligible subwindows")

    t = len(losses)
    rows = []
    # k denotes a candidate split index; it is NOT the Greek kappa.
    # min_side=2 permits ddof=1 sample variances on both sides.
    for k in range(min_side, t - min_side + 1):
        a, b = losses[:k], losses[k:]

        # Paper Eq. (4): mean losses before and after candidate k.
        # Paper Eq. (8): set eps to the observed absolute difference.
        eps = abs(a.mean() - b.mean())

        # Paper Eq. (12): kappa ~= n2/(n1+n2), capped to [0.05, 0.95].
        # This is an approximation motivated by loss changes dominating
        # variances; it need not minimise the bound for all samples.
        kappa = float(np.clip(len(b) / t, .05, .95))

        # Paper Eq. (5): use sample variances (ddof=1) as plug-in estimates.
        # Thus score is not automatically a finite-sample valid p-value.
        score = bernstein_two_sample_upper_bound(
            len(a), len(b), a.var(ddof=1), b.var(ddof=1), eps, M, kappa
        )
        rows.append((k, score, eps, kappa))

    # Paper Eq. (8): choose k* = argmin_k B(k), with p_score = min_k B(k).
    best = min(rows, key=lambda row: row[1])
    return {
        "split_after": best[0],          # k*: 1-based count within this window
        "upper_bound_score": best[1],   # min_k B(k): bound-based score, not p-value
        "epsilon": best[2],             # eps at the selected split
        "kappa": best[3],               # Eq. (12) allocation at the selected split
        "illustrative_flag": best[1] < delta,  # nominal comparison only
    }