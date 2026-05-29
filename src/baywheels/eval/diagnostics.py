"""Optimization and convergence diagnostics.

check_marginal_balance  – verify that at the optimum, predicted origin
                          and destination totals match observed totals
                          (a necessary condition from the score equations).
plot_convergence        – NLL vs L-BFGS iteration.
plot_residuals          – observed vs predicted scatter.
"""

from __future__ import annotations

import numpy as np

from baywheels.model.fit import FitResult
from baywheels.model.poisson import PoissonData, predict_mu_obs


def check_marginal_balance(
    result: FitResult,
    data: PoissonData,
    tol: float = 1e-2,
) -> dict[str, float]:
    """Check the score-equation invariant at the reported optimum.

    At the exact maximum:
      Σ_jt (N_ijt − μ_ijt) = 0  for all i   (origin balance)
      Σ_it (N_ijt − μ_ijt) = 0  for all j   (destination balance)

    Returns max absolute imbalance for origins, destinations, and
    temporal strata as a diagnostic dict.
    """
    theta = result.theta
    mu = predict_mu_obs(theta, data)
    residual = data.counts - mu

    orig_resid = np.abs(
        np.bincount(data.orig, weights=residual, minlength=data.layout.n_stations)
    )
    dest_resid = np.abs(
        np.bincount(data.dest, weights=residual, minlength=data.layout.n_stations)
    )
    hour_resid = np.abs(
        np.bincount(data.hour_obs.astype(np.int32), weights=residual, minlength=24)
    )

    diag = {
        "max_origin_imbalance": float(orig_resid.max()),
        "max_dest_imbalance": float(dest_resid.max()),
        "max_hour_imbalance": float(hour_resid.max()),
        "grad_norm": result.grad_norm,
    }

    # Relative to total observed count for easier interpretation
    total = data.counts.sum()
    diag["max_origin_imbalance_rel"] = diag["max_origin_imbalance"] / total
    diag["max_dest_imbalance_rel"] = diag["max_dest_imbalance"] / total

    return diag


def plot_convergence(result: FitResult, ax=None):
    """Plot negative log-likelihood vs L-BFGS function evaluation."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    history = result.nll_history
    ax.plot(np.arange(1, len(history) + 1), history, lw=1.5)
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("Negative log-likelihood")
    ax.set_title("L-BFGS convergence")
    ax.set_yscale("log")
    return ax


def plot_residuals(
    result: FitResult,
    data: PoissonData,
    max_points: int = 50_000,
    ax=None,
):
    """Scatter plot of observed vs predicted counts (log scale)."""
    import matplotlib.pyplot as plt

    mu = predict_mu_obs(result.theta, data)
    y = data.counts

    if len(y) > max_points:
        idx = np.random.default_rng(0).choice(len(y), max_points, replace=False)
        y, mu = y[idx], mu[idx]

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(mu, y, s=2, alpha=0.3, rasterized=True)
    lim = max(mu.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("Predicted μ")
    ax.set_ylabel("Observed N")
    ax.set_title("Observed vs predicted")
    return ax
