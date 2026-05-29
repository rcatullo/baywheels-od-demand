"""Unit tests for the Poisson model.

1. Gradient check: analytic gradient matches finite-difference approximation.
2. Convergence invariant: at the optimum, predicted marginals match observed.
3. Model sanity: γ_dist < 0 (demand decays with distance).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baywheels.model.params import ParamLayout
from baywheels.model.poisson import PoissonData, neg_log_likelihood, predict_mu_obs
from baywheels.model.fit import fit_baseline


def _toy_data(
    n_stations: int = 6,
    n_obs: int = 80,
    T: int = 336,   # 2 full weeks → all 24 hours × 7 dow values are covered
    seed: int = 0,
) -> tuple[PoissonData, ParamLayout]:
    rng = np.random.default_rng(seed)
    layout = ParamLayout(n_stations=n_stations)

    # Fake distance matrix
    lats = rng.uniform(37.7, 37.9, n_stations)
    lons = rng.uniform(-122.5, -122.3, n_stations)
    from baywheels.data.aggregator import haversine_matrix
    dist_matrix = haversine_matrix(lats, lons)

    # Calendar: T consecutive hourly bins covering all (hour, dow) combinations.
    # Temporal features are derived deterministically so every (h, d, m) in the
    # calendar is also reachable by observations (sampled from calendar indices).
    hour_cal = (np.arange(T) % 24).astype(np.int8)
    dow_cal  = (np.arange(T) // 24 % 7).astype(np.int8)
    month_cal = np.zeros(T, dtype=np.int8)   # all January for simplicity
    hol_cal   = np.zeros(T, dtype=np.int8)

    # Sample observations by drawing random calendar-bin indices, then random
    # station pairs.  This guarantees every (h, d, m) in the observations also
    # appears in the calendar, preventing one-sided η gradients from diverging.
    cal_idx = rng.integers(0, T, n_obs * 2)   # over-sample then trim
    orig_raw = rng.integers(0, n_stations, n_obs * 2).astype(np.int32)
    dest_raw = rng.integers(0, n_stations, n_obs * 2).astype(np.int32)
    valid = orig_raw != dest_raw
    cal_idx, orig_raw, dest_raw = cal_idx[valid], orig_raw[valid], dest_raw[valid]
    # Trim to n_obs (or use all if fewer)
    n = min(n_obs, len(orig_raw))
    cal_idx, orig, dest = cal_idx[:n], orig_raw[:n], dest_raw[:n]

    hour_obs  = hour_cal[cal_idx]
    dow_obs   = dow_cal[cal_idx]
    month_obs = month_cal[cal_idx]
    hol_obs   = hol_cal[cal_idx]
    counts    = rng.poisson(2, n).astype(np.float64) + 1
    dist_obs  = dist_matrix[orig, dest]

    N_orig  = np.bincount(orig, weights=counts, minlength=n_stations)
    N_dest  = np.bincount(dest, weights=counts, minlength=n_stations)
    N_hour  = np.bincount(hour_obs.astype(np.int32), weights=counts, minlength=24)
    N_dow   = np.bincount(dow_obs.astype(np.int32),  weights=counts, minlength=7)
    N_month = np.bincount(month_obs.astype(np.int32), weights=counts, minlength=12)
    N_hol   = float(counts[hol_obs.astype(bool)].sum())
    N_dist  = float((counts * dist_obs).sum())

    data = PoissonData(
        orig=orig, dest=dest,
        hour_obs=hour_obs, dow_obs=dow_obs, month_obs=month_obs, hol_obs=hol_obs,
        counts=counts, dist_obs=dist_obs,
        dist_matrix=dist_matrix,
        hour_cal=hour_cal, dow_cal=dow_cal, month_cal=month_cal, hol_cal=hol_cal,
        layout=layout,
        N_orig=N_orig, N_dest=N_dest, N_hour=N_hour, N_dow=N_dow,
        N_month=N_month, N_hol=N_hol, N_dist=N_dist,
    )
    return data, layout


class TestGradient:
    def test_fd_vs_analytic(self):
        data, layout = _toy_data(seed=42)
        rng = np.random.default_rng(7)
        theta = rng.normal(0, 0.1, layout.n_params)
        ridge = 1e-2

        nll, grad = neg_log_likelihood(theta, data, ridge=ridge)

        eps = 1e-5
        fd_grad = np.empty_like(theta)
        for i in range(layout.n_params):
            tp = theta.copy(); tp[i] += eps
            tm = theta.copy(); tm[i] -= eps
            fd_grad[i] = (neg_log_likelihood(tp, data, ridge=ridge)[0]
                          - neg_log_likelihood(tm, data, ridge=ridge)[0]) / (2 * eps)

        rel_err = np.abs(grad - fd_grad) / (np.abs(fd_grad) + 1e-8)
        assert rel_err.max() < 1e-4, f"Max relative FD error: {rel_err.max():.2e}"

    def test_nll_decreases_along_neg_grad(self):
        data, layout = _toy_data(seed=13)
        theta = layout.zeros()
        nll0, grad0 = neg_log_likelihood(theta, data)
        # Normalize step so it is proportional to 1/||grad|| (Cauchy step)
        step = 1e-4 / (np.linalg.norm(grad0) + 1e-8)
        theta1 = theta - step * grad0
        nll1, _ = neg_log_likelihood(theta1, data)
        assert nll1 < nll0, "NLL should decrease along the negative gradient"


class TestConvergenceInvariant:
    """At the optimum, predicted marginals must equal observed marginals."""

    def test_origin_balance(self):
        data, layout = _toy_data(n_stations=4, n_obs=200, seed=99)
        result = fit_baseline(data, ridge=1e-3, maxiter=1000, gtol=1e-6, verbose=False)

        mu = predict_mu_obs(result.theta, data)
        residual = data.counts - mu
        orig_imbalance = np.abs(
            np.bincount(data.orig, weights=residual, minlength=layout.n_stations)
        )
        # Allow tolerance proportional to total count
        tol = 0.05 * data.counts.sum() / layout.n_stations
        assert orig_imbalance.max() < tol, (
            f"Origin imbalance too large: {orig_imbalance.max():.2f} > {tol:.2f}"
        )

    def test_dest_balance(self):
        data, layout = _toy_data(n_stations=4, n_obs=200, seed=99)
        result = fit_baseline(data, ridge=1e-3, maxiter=1000, gtol=1e-6, verbose=False)

        mu = predict_mu_obs(result.theta, data)
        residual = data.counts - mu
        dest_imbalance = np.abs(
            np.bincount(data.dest, weights=residual, minlength=layout.n_stations)
        )
        tol = 0.05 * data.counts.sum() / layout.n_stations
        assert dest_imbalance.max() < tol


class TestSanity:
    def test_param_count(self):
        layout = ParamLayout(n_stations=500)
        assert layout.n_params == 2 * 500 + 24 + 7 + 12 + 1 + 1

    def test_slices_partition(self):
        layout = ParamLayout(n_stations=10)
        all_idx: set[int] = set()
        for sl in (
            layout.sl_alpha, layout.sl_beta, layout.sl_hour, layout.sl_dow,
            layout.sl_month, layout.sl_gamma_holiday, layout.sl_gamma_dist,
        ):
            new = set(range(*sl.indices(layout.n_params)))
            assert not new & all_idx, "Parameter slices overlap"
            all_idx |= new
        assert len(all_idx) == layout.n_params

    def test_dist_decay(self):
        """Fit on simulated data where demand decays with distance."""
        import dataclasses
        data, layout = _toy_data(n_stations=5, n_obs=300, seed=0)
        # Reweight counts by inverse distance so closer pairs have more trips
        w = data.counts / (data.dist_obs + 0.1)
        data = dataclasses.replace(
            data,
            counts=w,
            N_orig=np.bincount(data.orig, weights=w, minlength=layout.n_stations),
            N_dest=np.bincount(data.dest, weights=w, minlength=layout.n_stations),
            N_dist=float((w * data.dist_obs).sum()),
        )
        result = fit_baseline(data, ridge=1e-3, maxiter=300, verbose=False)
        # γ_dist should be negative (farther stations have lower demand)
        assert layout.gamma_dist(result.theta) < 0
