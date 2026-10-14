"""Slide 22: controlled structural changes and statistical signatures.

Run: python -m scripts.generate_slide22_structural_visuals
Figures are descriptive simulations, NOT detector performance measurements.
This generator deliberately isolates mean, variance, and AR dependence changes;
src/simulation/structural_changes.py currently allows confounding shifts.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCENARIOS = ("mean_shift", "variance_shift", "dependence_shift")


def generate_isolated_stream(scenario, *, seed=42, n=1024, change_at=512):
    """Stationary AR(1) in each regime, no hidden co-changing parameters.

    X_t = mu + phi*(X_(t-1)-mu) + sqrt(v*(1-phi**2))*z_t, z~N(0,1).
    v is stationary marginal variance, phi is lag-one correlation.
    """
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    if n < 32 or not 1 < change_at < n - 1:
        raise ValueError("Need n>=32 and an interior breakpoint")
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    # Pre/post stationary target parameters: (mean, marginal variance, AR1).
    baseline = (0.0, 1.0, 0.25)
    changed = {
        "mean_shift": (1.5, 1.0, 0.25),
        "variance_shift": (0.0, 4.0, 0.25),
        "dependence_shift": (0.0, 1.0, 0.75),
    }[scenario]
    x[0] = rng.normal(loc=baseline[0], scale=np.sqrt(baseline[1]))
    for t in range(1, n):
        mu, var, phi = baseline if t < change_at else changed
        noise = rng.normal() * np.sqrt(var * (1 - phi * phi))
        # Regime switch is transient; parameters specify stationary targets.
        x[t] = mu + phi * (x[t - 1] - mu) + noise
    return x, {"scenario": scenario, "seed": seed, "n": n,
               "change_at": change_at, "baseline": baseline, "after": changed,
               "model": "mean-centred AR1 with invariant non-target population moments"}


def rolling_signatures(x, window=96):
    """Rolling descriptive signatures; lag correlation measured within window."""
    if window < 4 or window > len(x):
        raise ValueError("window must be in [4,n]")
    times = np.arange(window - 1, len(x))
    means, variances, autocorr = [], [], []
    for t in times:
        y = x[t - window + 1:t + 1]
        means.append(float(np.mean(y)))
        variances.append(float(np.var(y, ddof=1)))
        a, b = y[:-1], y[1:]
        sd = np.std(a) * np.std(b)
        autocorr.append(float(np.mean((a-a.mean()) * (b-b.mean())) / sd) if sd > 1e-12 else np.nan)
    return times, np.array([means, variances, autocorr])


def make_figures(out, *, seed=42, n=1024, change_at=512, window=96):
    out.mkdir(parents=True, exist_ok=True)
    xs = {name: generate_isolated_stream(name, seed=seed + j, n=n, change_at=change_at)
          for j, name in enumerate(SCENARIOS)}

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.2), sharex=True)
    for ax, (name, (x, _)) in zip(axes, xs.items()):
        ax.plot(np.arange(n), x, lw=0.65)
        ax.axvline(change_at, color="black", ls="--", label="Known change")
        ax.set_ylabel(name.replace("_", " ").title())
        ax.grid(alpha=0.15)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Observation index")
    fig.suptitle("Controlled structural changes — synthetic examples")
    fig.tight_layout()
    fig.savefig(out / "slide22_scenarios.png", dpi=200)
    fig.savefig(out / "slide22_scenarios.svg")
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True)
    metrics = ("Rolling mean", "Rolling variance", "Rolling lag-1 correlation")
    for i, (name, (x, meta)) in enumerate(xs.items()):
        times, values = rolling_signatures(x, window=window)
        for j, (metric, vec) in enumerate(zip(metrics, values)):
            ax = axes[i, j]
            ax.plot(times, vec, lw=1.1)
            ax.axvline(change_at, color="black", ls="--")
            ax.grid(alpha=0.15)
            if i == 0: ax.set_title(metric)
            if j == 0: ax.set_ylabel(name.replace("_", " ").title())
            if i == 2: ax.set_xlabel("Observation index")
    fig.suptitle("Descriptive statistical signatures — no detector outcomes")
    fig.tight_layout()
    fig.savefig(out / "slide22_rolling_signatures.png", dpi=200)
    fig.savefig(out / "slide22_rolling_signatures.svg")
    plt.close(fig)

    # This table is a conceptual method-to-target map, NOT an empirical scorecard.
    rows = ["Mean contrast / Hotelling T²", "Variance comparison", "Lag/correlation test"]
    cols = ["Mean shift", "Variance shift", "AR dependence shift"]
    coverage = [["Primary target", "Not direct test", "Not direct test"],
                ["Not direct test", "Primary target", "Not direct test"],
                ["Not direct test", "Not direct test", "Primary target"]]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")
    table = ax.table(cellText=coverage, rowLabels=rows, colLabels=cols,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.1)
    ax.set_title("Detector design targets (conceptual) — evaluation outcomes NOT TESTED", pad=20)
    fig.tight_layout()
    fig.savefig(out / "slide22_method_targets.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "slide22_method_targets.svg", bbox_inches="tight")
    plt.close(fig)

    config = {"seed": seed, "n": n, "change_at": change_at,
              "rolling_window": window, "scenarios": list(SCENARIOS),
              "interpretation": "descriptive simulations; no measured detector performance",
              "generator": "isolated stationary-target AR(1) scenarios"}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("Wrote figures and config to", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/structural_changes/slide22"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--change-at", type=int, default=512)
    parser.add_argument("--window", type=int, default=96)
    args = parser.parse_args()
    make_figures(args.output, seed=args.seed, n=args.n,
                 change_at=args.change_at, window=args.window)


if __name__ == "__main__":
    main()
