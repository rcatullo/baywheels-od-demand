"""Poisson log-likelihood and analytic gradient for the OD demand model.

Full factorisation
------------------
log μ_ijt = (α_i + β_j + γ_dist·d_ij + Σ_k γ_sk·x_sk_ij)
          + (η_hour + η_dow + η_month + γ_hol·hol_t + Σ_k γ_tk·w_tk_t)

Where γ_sk are *spatial* extra gammas (per OD-pair, e.g. elevation gain)
and γ_tk are *temporal* extra gammas (per hour, e.g. weather).

The sum over all cells factorises as  M · S  where

    M = Σ_{i,j} exp(α_i + β_j + γ_dist·d_ij + Σ_k γ_sk·x_sk_ij)
    S = Σ_t     exp(η_hour[h(t)] + η_dow[d(t)] + η_month[m(t)]
                    + γ_hol·hol_t + Σ_k γ_tk·w_tk_t)

reducing each gradient step to O(I²) + O(T) instead of O(I²·T).

All computations use logsumexp for numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from baywheels.model.params import ParamLayout


@dataclass
class PoissonData:
    # ── Observed cells (non-zero counts) ─────────────────────────────────
    orig: np.ndarray       # int32  (n_obs,)
    dest: np.ndarray       # int32
    hour_obs: np.ndarray   # int8
    dow_obs: np.ndarray    # int8
    month_obs: np.ndarray  # int8
    hol_obs: np.ndarray    # int8
    counts: np.ndarray     # float64
    dist_obs: np.ndarray   # float64 – dist_km per observed cell

    # ── Full spatial matrices ─────────────────────────────────────────────
    dist_matrix: np.ndarray    # (I, I) haversine km

    # ── Calendar for the training period (every hour, including zeros) ───
    hour_cal: np.ndarray   # int8  (T,)
    dow_cal: np.ndarray    # int8
    month_cal: np.ndarray  # int8
    hol_cal: np.ndarray    # int8

    layout: ParamLayout

    # ── Sufficient statistics (constant across iterations) ────────────────
    N_orig: np.ndarray     # (I,)
    N_dest: np.ndarray     # (I,)
    N_hour: np.ndarray     # (24,)
    N_dow: np.ndarray      # (7,)
    N_month: np.ndarray    # (12,)
    N_hol: float
    N_dist: float

    # ── Optional: elevation (spatial extra covariate) ─────────────────────
    elev_matrix: np.ndarray | None = None   # (I, I) dest_elev - orig_elev [m]
    elev_obs: np.ndarray | None = None      # (n_obs,) elevation gain per trip
    N_elev: float = 0.0                     # Σ count·Δelev

    # ── Optional: weather (temporal extra covariates) ─────────────────────
    weather_cal: np.ndarray | None = None   # (T, n_weather)
    weather_obs: np.ndarray | None = None   # (n_obs, n_weather)
    N_weather: np.ndarray | None = None     # (n_weather,)

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        obs: "pd.DataFrame",
        dist_matrix: np.ndarray,
        layout: ParamLayout,
        cal: "pd.DataFrame",
        elev_matrix: np.ndarray | None = None,
    ) -> "PoissonData":
        """Construct PoissonData from processed DataFrames.

        obs must have columns: orig_idx, dest_idx, hour, dow, month,
            is_holiday, count, dist_km.
            Optional: temperature_2m, precipitation, wind_speed_10m,
                      relative_humidity_2m  (area-wide weather).
        cal must have columns: hour, dow, month, is_holiday.
            Optional: same weather columns.
        elev_matrix: (I, I) float64 elevation gain matrix [m], or None.
        """
        import pandas as pd

        n_stations = layout.n_stations
        counts = obs["count"].to_numpy(dtype=np.float64)
        orig   = obs["orig_idx"].to_numpy(dtype=np.int32)
        dest   = obs["dest_idx"].to_numpy(dtype=np.int32)
        hour_obs  = obs["hour"].to_numpy(dtype=np.int8)
        dow_obs   = obs["dow"].to_numpy(dtype=np.int8)
        month_obs = obs["month"].to_numpy(dtype=np.int8)
        hol_obs   = obs["is_holiday"].to_numpy(dtype=np.int8)
        dist_obs  = obs["dist_km"].to_numpy(dtype=np.float64)

        N_orig  = np.bincount(orig, weights=counts, minlength=n_stations)
        N_dest  = np.bincount(dest, weights=counts, minlength=n_stations)
        N_hour  = np.bincount(hour_obs.astype(np.int32), weights=counts, minlength=24)
        N_dow   = np.bincount(dow_obs.astype(np.int32),  weights=counts, minlength=7)
        N_month = np.bincount(month_obs.astype(np.int32), weights=counts, minlength=12)
        N_hol   = float(counts[hol_obs.astype(bool)].sum())
        N_dist  = float((counts * dist_obs).sum())

        # ── Elevation (spatial extra) ─────────────────────────────────────
        elev_obs_arr: np.ndarray | None = None
        N_elev = 0.0
        if elev_matrix is not None:
            elev_obs_arr = elev_matrix[orig, dest]
            N_elev = float((counts * elev_obs_arr).sum())

        # ── Weather (temporal extra) ──────────────────────────────────────
        WEATHER_COLS = [
            "temperature_2m", "precipitation",
            "wind_speed_10m", "relative_humidity_2m",
        ]
        obs_weather_cols = [c for c in WEATHER_COLS if c in obs.columns]
        cal_weather_cols = [c for c in WEATHER_COLS if c in cal.columns]

        weather_obs_arr: np.ndarray | None = None
        weather_cal_arr: np.ndarray | None = None
        N_weather_arr:  np.ndarray | None = None

        if obs_weather_cols:
            # Fill NaN with column mean so missing hours don't propagate NaN
            # into the dot product (nan * 0 = nan in IEEE 754).
            weather_obs_df = obs[obs_weather_cols].copy()
            weather_obs_df.fillna(weather_obs_df.mean(), inplace=True)
            weather_obs_arr = weather_obs_df.to_numpy(dtype=np.float64)
            N_weather_arr = (counts[:, None] * weather_obs_arr).sum(axis=0)

        if cal_weather_cols:
            weather_cal_df = cal[cal_weather_cols].copy()
            weather_cal_df.fillna(weather_cal_df.mean(), inplace=True)
            weather_cal_arr = weather_cal_df.to_numpy(dtype=np.float64)

        return cls(
            orig=orig, dest=dest,
            hour_obs=hour_obs, dow_obs=dow_obs,
            month_obs=month_obs, hol_obs=hol_obs,
            counts=counts, dist_obs=dist_obs,
            dist_matrix=dist_matrix,
            hour_cal=cal["hour"].to_numpy(dtype=np.int8),
            dow_cal=cal["dow"].to_numpy(dtype=np.int8),
            month_cal=cal["month"].to_numpy(dtype=np.int8),
            hol_cal=cal["is_holiday"].to_numpy(dtype=np.int8),
            layout=layout,
            N_orig=N_orig, N_dest=N_dest,
            N_hour=N_hour, N_dow=N_dow,
            N_month=N_month, N_hol=N_hol, N_dist=N_dist,
            elev_matrix=elev_matrix,
            elev_obs=elev_obs_arr,
            N_elev=N_elev,
            weather_obs=weather_obs_arr,
            weather_cal=weather_cal_arr,
            N_weather=N_weather_arr,
        )


# ── Core numerics ──────────────────────────────────────────────────────────────

def neg_log_likelihood(
    theta: np.ndarray,
    data: PoissonData,
    ridge: float = 1e-3,
) -> tuple[float, np.ndarray]:
    """Return (nll, grad_nll) for L-BFGS-B minimisation.

    Uses the M·S factorisation with logsumexp for overflow safety.
    """
    from scipy.special import logsumexp

    ly   = data.layout
    alpha  = ly.alpha(theta)
    beta   = ly.beta(theta)
    eta_h  = ly.eta_hour(theta)
    eta_d  = ly.eta_dow(theta)
    eta_m  = ly.eta_month(theta)
    g_hol  = ly.gamma_holiday(theta)
    g_dist = ly.gamma_dist(theta)

    n_spatial  = ly.n_spatial_gamma
    n_temporal = ly.n_temporal_gamma
    g_spatial  = ly.gamma_spatial_extra(theta)   # shape (n_spatial,)
    g_temporal = ly.gamma_temporal_extra(theta)  # shape (n_temporal,)

    # ------------------------------------------------------------------ #
    # Spatial kernel  log_K[i,j] = γ_dist·d_ij + Σ_k γ_sk·x_ij_k        #
    # ------------------------------------------------------------------ #
    log_K = g_dist * data.dist_matrix
    if n_spatial > 0 and data.elev_matrix is not None:
        log_K = log_K + g_spatial[0] * data.elev_matrix   # elevation gain

    log_mat = alpha[:, None] + log_K + beta[None, :]      # (I, I)

    log_M   = float(logsumexp(log_mat))
    M       = np.exp(log_M)
    log_vKu = logsumexp(log_mat, axis=1)   # (I,)
    vKu     = np.exp(log_vKu)
    log_uKv = logsumexp(log_mat, axis=0)   # (I,)
    uKv     = np.exp(log_uKv)

    # ------------------------------------------------------------------ #
    # Temporal sum  η_t = η_h + η_d + η_m + γ_hol·hol + Σ_k γ_tk·w_k   #
    # ------------------------------------------------------------------ #
    eta_cal = (
        eta_h[data.hour_cal.astype(np.int32)]
        + eta_d[data.dow_cal.astype(np.int32)]
        + eta_m[data.month_cal.astype(np.int32)]
        + g_hol * data.hol_cal
    )
    if n_temporal > 0 and data.weather_cal is not None:
        eta_cal = eta_cal + data.weather_cal @ g_temporal  # (T,)

    log_S       = float(logsumexp(eta_cal))
    S           = np.exp(log_S)
    exp_eta_cal = np.exp(eta_cal)

    S_hour  = np.bincount(data.hour_cal.astype(np.int32), weights=exp_eta_cal, minlength=24)
    S_dow   = np.bincount(data.dow_cal.astype(np.int32),  weights=exp_eta_cal, minlength=7)
    S_month = np.bincount(data.month_cal.astype(np.int32), weights=exp_eta_cal, minlength=12)
    S_hol   = float((exp_eta_cal * data.hol_cal).sum())

    MS = M * S
    if not np.isfinite(MS):
        return 1e30, np.zeros_like(theta)

    # ------------------------------------------------------------------ #
    # Log-likelihood  Σ_obs count · log μ_obs  −  M·S                    #
    # ------------------------------------------------------------------ #
    log_mu_obs = (
        alpha[data.orig]
        + beta[data.dest]
        + eta_h[data.hour_obs.astype(np.int32)]
        + eta_d[data.dow_obs.astype(np.int32)]
        + eta_m[data.month_obs.astype(np.int32)]
        + g_hol * data.hol_obs
        + g_dist * data.dist_obs
    )
    if n_spatial > 0 and data.elev_obs is not None:
        log_mu_obs = log_mu_obs + g_spatial[0] * data.elev_obs
    if n_temporal > 0 and data.weather_obs is not None:
        log_mu_obs = log_mu_obs + data.weather_obs @ g_temporal

    ll_obs = float((data.counts * log_mu_obs).sum())
    ll     = ll_obs - MS

    # Ridge on γ parameters only — station and temporal fixed effects are
    # identified by the data and should not be shrunk toward zero.
    gamma_start = ly.sl_gamma_holiday.start
    theta_gamma = theta[gamma_start:]
    penalty = 0.5 * ridge * float(np.dot(theta_gamma, theta_gamma))
    nll     = -(ll - penalty)

    # ------------------------------------------------------------------ #
    # Gradient                                                             #
    # ------------------------------------------------------------------ #
    grad = np.empty_like(theta)

    grad[ly.sl_alpha] = data.N_orig - vKu * S
    grad[ly.sl_beta]  = data.N_dest - uKv * S

    grad[ly.sl_hour]  = data.N_hour  - M * S_hour
    grad[ly.sl_dow]   = data.N_dow   - M * S_dow
    grad[ly.sl_month] = data.N_month - M * S_month

    grad[ly.sl_gamma_holiday] = data.N_hol  - M * S_hol

    DKM = float(np.sum(data.dist_matrix * np.exp(log_mat)))
    grad[ly.sl_gamma_dist] = data.N_dist - DKM * S

    # Spatial extra (elevation gain)
    if n_spatial > 0 and data.elev_matrix is not None:
        DKM_elev = float(np.sum(data.elev_matrix * np.exp(log_mat)))
        grad[ly.sl_gamma_spatial_extra] = data.N_elev - DKM_elev * S
    elif n_spatial > 0:
        grad[ly.sl_gamma_spatial_extra] = 0.0

    # Temporal extra (weather)
    if n_temporal > 0 and data.weather_cal is not None:
        S_weather = data.weather_cal.T @ exp_eta_cal   # (n_temporal,)
        grad[ly.sl_gamma_temporal_extra] = data.N_weather - M * S_weather
    elif n_temporal > 0:
        grad[ly.sl_gamma_temporal_extra] = 0.0

    grad = -grad
    grad[gamma_start:] += ridge * theta_gamma
    return nll, grad


def predict_mu_obs(theta: np.ndarray, data: PoissonData) -> np.ndarray:
    """Return μ_ijt for every observed cell (in-sample or out-of-sample)."""
    ly = data.layout
    n_spatial  = ly.n_spatial_gamma
    n_temporal = ly.n_temporal_gamma

    log_mu = (
        ly.alpha(theta)[data.orig]
        + ly.beta(theta)[data.dest]
        + ly.eta_hour(theta)[data.hour_obs.astype(np.int32)]
        + ly.eta_dow(theta)[data.dow_obs.astype(np.int32)]
        + ly.eta_month(theta)[data.month_obs.astype(np.int32)]
        + ly.gamma_holiday(theta) * data.hol_obs
        + ly.gamma_dist(theta) * data.dist_obs
    )
    if n_spatial > 0 and data.elev_obs is not None:
        log_mu = log_mu + ly.gamma_spatial_extra(theta)[0] * data.elev_obs
    if n_temporal > 0 and data.weather_obs is not None:
        log_mu = log_mu + data.weather_obs @ ly.gamma_temporal_extra(theta)

    return np.exp(log_mu)
