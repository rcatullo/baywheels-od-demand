"""L-BFGS fitting for the Poisson OD model.

fit_baseline  – optimise all parameters jointly.
profile_ll    – profile likelihood ℓ_p(γ) = max_{α,β,η} ℓ(α,β,η,γ).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from baywheels.model.params import ParamLayout
from baywheels.model.poisson import PoissonData, neg_log_likelihood


@dataclass
class FitResult:
    theta: np.ndarray          # fitted parameter vector
    layout: ParamLayout
    nll: float                 # negative log-likelihood at optimum
    grad_norm: float           # ‖∇ nll‖ at optimum (should be ~0)
    n_iter: int
    converged: bool
    nll_history: list[float] = field(default_factory=list)
    ridge: float = 1e-3


def fit_baseline(
    data: PoissonData,
    ridge: float = 1e-3,
    maxiter: int = 500,
    gtol: float = 1e-5,
    verbose: bool = True,
) -> FitResult:
    """Fit all parameters via L-BFGS-B.

    Returns a FitResult with the optimised θ and diagnostics.
    """
    layout = data.layout
    theta0 = layout.zeros()

    nll_history: list[float] = []

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        nll, grad = neg_log_likelihood(theta, data, ridge=ridge)
        nll_history.append(nll)
        return nll, grad

    if verbose:
        print(f"Fitting baseline model: {layout.n_params} parameters, "
              f"{len(data.counts):,} observed cells, ridge={ridge}")

    # Pin four reference categories to remove the 4D null space.
    # The null space is spanned by adding a constant to any one of
    # {α, β, η_hour, η_dow, η_month} and subtracting it from another.
    # Fixing α[0]=η_hour[0]=η_dow[0]=η_month[0]=0 identifies the model.
    bounds: list[tuple] = [(None, None)] * layout.n_params
    for fixed_idx in [
        layout.sl_alpha.start,   # α[0] = 0  (reference origin)
        layout.sl_hour.start,    # η_hour[0] = 0  (reference hour)
        layout.sl_dow.start,     # η_dow[0] = 0   (reference dow)
        layout.sl_month.start,   # η_month[0] = 0 (reference month)
    ]:
        bounds[fixed_idx] = (0.0, 0.0)

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "gtol": gtol, "iprint": 10 if verbose else -1},
    )

    grad_norm = float(np.linalg.norm(result.jac)) if result.jac is not None else float("nan")

    if verbose:
        status = "CONVERGED" if result.success else "NOT CONVERGED"
        print(f"  [{status}] {result.message}")
        print(f"  iterations={result.nit}  nll={result.fun:.4f}  ‖grad‖={grad_norm:.2e}")

    return FitResult(
        theta=result.x,
        layout=layout,
        nll=float(result.fun),
        grad_norm=grad_norm,
        n_iter=result.nit,
        converged=result.success,
        nll_history=nll_history,
        ridge=ridge,
    )


def profile_ll(
    gamma_fixed: np.ndarray,
    data: PoissonData,
    ridge: float = 1e-3,
    maxiter: int = 300,
) -> tuple[float, np.ndarray]:
    """Compute the profile log-likelihood ℓ_p(γ) for fixed γ.

    Fixes γ entries (dist, holiday, and any extras) and optimises over
    α, β, η only.  Returns (profile_nll, theta_opt).
    """
    layout = data.layout
    theta0 = layout.zeros()
    theta0[layout.sl_gamma_holiday] = gamma_fixed[0]
    theta0[layout.sl_gamma_dist] = gamma_fixed[1]
    if len(gamma_fixed) > 2:
        theta0[layout.sl_gamma_extra] = gamma_fixed[2:]

    # Build a mask: free parameters = all except gamma
    free_mask = np.ones(layout.n_params, dtype=bool)
    free_mask[layout.sl_gamma_holiday] = False
    free_mask[layout.sl_gamma_dist] = False
    free_mask[layout.sl_gamma_extra] = False

    free_idx = np.where(free_mask)[0]
    theta_fixed = theta0.copy()

    def obj_free(x_free: np.ndarray) -> tuple[float, np.ndarray]:
        theta = theta_fixed.copy()
        theta[free_idx] = x_free
        nll, grad = neg_log_likelihood(theta, data, ridge=ridge)
        return nll, grad[free_idx]

    result = minimize(
        obj_free,
        theta0[free_idx],
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "gtol": 1e-5},
    )
    theta_opt = theta_fixed.copy()
    theta_opt[free_idx] = result.x
    return float(result.fun), theta_opt
