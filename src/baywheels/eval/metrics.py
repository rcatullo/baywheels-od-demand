"""Predictive accuracy metrics for Poisson OD counts."""

from __future__ import annotations

import numpy as np


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def pearson_r(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.corrcoef(y, yhat)[0, 1])


def poisson_deviance(y: np.ndarray, yhat: np.ndarray) -> float:
    """Residual deviance D = 2 Σ [y log(y/ŷ) − (y − ŷ)].

    Cells with y=0 contribute 2*(ŷ) to the deviance.
    """
    eps = 1e-12
    mask = y > 0
    d = np.zeros_like(y, dtype=np.float64)
    d[mask] = 2.0 * (y[mask] * np.log(y[mask] / (yhat[mask] + eps)) - (y[mask] - yhat[mask]))
    d[~mask] = 2.0 * yhat[~mask]
    return float(d.sum())


def summary(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "pearson_r": pearson_r(y, yhat),
        "poisson_deviance": poisson_deviance(y, yhat),
        "n_obs": float(len(y)),
        "total_observed": float(y.sum()),
        "total_predicted": float(yhat.sum()),
    }
