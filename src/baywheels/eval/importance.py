"""Model-agnostic permutation feature importance.

For each feature group, replace its values with a random permutation of
the training values and measure the increase in negative log-likelihood.
A larger increase means the feature contributes more to predictive fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baywheels.model.poisson import PoissonData, neg_log_likelihood


@dataclass
class ImportanceResult:
    feature: str
    delta_nll: float
    delta_nll_std: float


def _permuted_data(data: PoissonData, feature: str, rng: np.random.Generator) -> PoissonData:
    """Return a shallow copy of data with one feature group permuted."""
    import copy
    d = copy.copy(data)
    perm = rng.permutation(len(data.counts))

    if feature == "dist":
        d.dist_obs = data.dist_obs[perm]
        d.N_dist   = float((data.counts * d.dist_obs).sum())
    elif feature == "elevation":
        if data.elev_obs is not None:
            d.elev_obs = data.elev_obs[perm]
            d.N_elev   = float((data.counts * d.elev_obs).sum())
    elif feature == "weather":
        if data.weather_obs is not None:
            d.weather_obs = data.weather_obs[perm]
            d.N_weather   = (data.counts[:, None] * d.weather_obs).sum(axis=0)
    elif feature == "hour":
        d.hour_obs = data.hour_obs[perm]
        d.N_hour   = np.bincount(
            d.hour_obs.astype(np.int32), weights=data.counts, minlength=24
        )
    elif feature == "dow":
        d.dow_obs = data.dow_obs[perm]
        d.N_dow   = np.bincount(
            d.dow_obs.astype(np.int32), weights=data.counts, minlength=7
        )
    elif feature == "month":
        d.month_obs = data.month_obs[perm]
        d.N_month   = np.bincount(
            d.month_obs.astype(np.int32), weights=data.counts, minlength=12
        )
    elif feature == "holiday":
        d.hol_obs = data.hol_obs[perm]
        d.N_hol   = float(data.counts[d.hol_obs.astype(bool)].sum())
    elif feature == "origin":
        d.orig   = data.orig[perm]
        d.N_orig = np.bincount(d.orig, weights=data.counts,
                               minlength=data.layout.n_stations)
    elif feature == "destination":
        d.dest   = data.dest[perm]
        d.N_dest = np.bincount(d.dest, weights=data.counts,
                               minlength=data.layout.n_stations)
    else:
        raise ValueError(f"Unknown feature '{feature}'")
    return d


def permutation_importance(
    theta: np.ndarray,
    data: PoissonData,
    features: list[str] | None = None,
    n_repeats: int = 5,
    ridge: float = 1e-3,
    seed: int = 0,
) -> list[ImportanceResult]:
    """Compute permutation importance for each feature group.

    Parameters
    ----------
    theta     : fitted parameter vector
    data      : PoissonData (train or test)
    features  : feature names; default = all available
    n_repeats : number of random permutations to average over
    ridge     : must match value used during fitting
    """
    if features is None:
        features = ["dist", "hour", "dow", "month", "holiday",
                    "origin", "destination"]
        if data.elev_obs is not None:
            features.append("elevation")
        if data.weather_obs is not None:
            features.append("weather")

    rng      = np.random.default_rng(seed)
    base_nll, _ = neg_log_likelihood(theta, data, ridge=ridge)

    results: list[ImportanceResult] = []
    for feat in features:
        deltas: list[float] = []
        for _ in range(n_repeats):
            d_perm   = _permuted_data(data, feat, rng)
            nll_perm, _ = neg_log_likelihood(theta, d_perm, ridge=ridge)
            deltas.append(nll_perm - base_nll)
        results.append(
            ImportanceResult(
                feature=feat,
                delta_nll=float(np.mean(deltas)),
                delta_nll_std=float(np.std(deltas)),
            )
        )

    results.sort(key=lambda r: r.delta_nll, reverse=True)
    return results
