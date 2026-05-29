"""Parameter vector layout for the Poisson OD model.

Baseline layout (extra_gamma=()):

    [α_0…α_{I-1} | β_0…β_{J-1} | η_hour(24) | η_dow(7) | η_month(12)
     | γ_holiday | γ_dist]

Full model with elevation + weather (n_spatial_gamma=1,
extra_gamma=("elev_gain", "temperature_2m", "precipitation",
             "wind_speed_10m", "relative_humidity_2m")):

    [… | γ_dist | γ_elev | γ_temp | γ_precip | γ_wind | γ_humidity]
                  ↑ spatial ↑       ↑       temporal        ↑

The first `n_spatial_gamma` entries of extra_gamma are spatial covariates
(per OD-pair, enter the kernel K_ij alongside distance).  The remaining
entries are temporal covariates (per calendar hour, enter η_t alongside
the fixed-effects strata).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ParamLayout:
    n_stations: int
    extra_gamma: Sequence[str] = ()  # names of all extra covariates, in order
    n_spatial_gamma: int = 0         # how many of extra_gamma are spatial
    use_zip: bool = False            # add hurdle activity model (δ_intercept, δ_dist)

    # ------------------------------------------------------------------ #
    # Core slices                                                          #
    # ------------------------------------------------------------------ #

    @property
    def sl_alpha(self) -> slice:
        return slice(0, self.n_stations)

    @property
    def sl_beta(self) -> slice:
        return slice(self.n_stations, 2 * self.n_stations)

    @property
    def sl_hour(self) -> slice:
        base = 2 * self.n_stations
        return slice(base, base + 24)

    @property
    def sl_dow(self) -> slice:
        base = 2 * self.n_stations + 24
        return slice(base, base + 7)

    @property
    def sl_month(self) -> slice:
        base = 2 * self.n_stations + 31
        return slice(base, base + 12)

    @property
    def sl_gamma_holiday(self) -> slice:
        base = 2 * self.n_stations + 43
        return slice(base, base + 1)

    @property
    def sl_gamma_dist(self) -> slice:
        base = 2 * self.n_stations + 44
        return slice(base, base + 1)

    # ------------------------------------------------------------------ #
    # Extra gamma slices                                                   #
    # ------------------------------------------------------------------ #

    @property
    def sl_gamma_extra(self) -> slice:
        """Full slice over all extra gammas (backward-compat alias)."""
        base = 2 * self.n_stations + 45
        return slice(base, base + len(self.extra_gamma))

    @property
    def sl_gamma_spatial_extra(self) -> slice:
        """Spatial extra gammas (e.g. elevation gain) — enter kernel K_ij."""
        base = 2 * self.n_stations + 45
        return slice(base, base + self.n_spatial_gamma)

    @property
    def sl_gamma_temporal_extra(self) -> slice:
        """Temporal extra gammas (e.g. weather) — enter η_t."""
        base = 2 * self.n_stations + 45 + self.n_spatial_gamma
        return slice(base, base + self.n_temporal_gamma)

    @property
    def n_temporal_gamma(self) -> int:
        return len(self.extra_gamma) - self.n_spatial_gamma

    # ------------------------------------------------------------------ #
    # ZIP activity model slices (only present when use_zip=True)         #
    # ------------------------------------------------------------------ #

    @property
    def sl_delta_intercept(self) -> slice:
        """Log-odds intercept for OD-pair activity (ZIP only)."""
        base = 2 * self.n_stations + 45 + len(self.extra_gamma)
        return slice(base, base + 1)

    @property
    def sl_delta_dist(self) -> slice:
        """Distance coefficient for OD-pair activity log-odds (ZIP only)."""
        base = 2 * self.n_stations + 45 + len(self.extra_gamma) + 1
        return slice(base, base + 1)

    def delta_intercept(self, theta: np.ndarray) -> float:
        return float(theta[self.sl_delta_intercept][0])

    def delta_dist(self, theta: np.ndarray) -> float:
        return float(theta[self.sl_delta_dist][0])

    @property
    def n_params(self) -> int:
        base = 2 * self.n_stations + 45 + len(self.extra_gamma)
        return base + (2 if self.use_zip else 0)

    # ------------------------------------------------------------------ #
    # Named views into an existing theta array                            #
    # ------------------------------------------------------------------ #

    def alpha(self, theta: np.ndarray) -> np.ndarray:
        return theta[self.sl_alpha]

    def beta(self, theta: np.ndarray) -> np.ndarray:
        return theta[self.sl_beta]

    def eta_hour(self, theta: np.ndarray) -> np.ndarray:
        return theta[self.sl_hour]

    def eta_dow(self, theta: np.ndarray) -> np.ndarray:
        return theta[self.sl_dow]

    def eta_month(self, theta: np.ndarray) -> np.ndarray:
        return theta[self.sl_month]

    def gamma_holiday(self, theta: np.ndarray) -> float:
        return float(theta[self.sl_gamma_holiday][0])

    def gamma_dist(self, theta: np.ndarray) -> float:
        return float(theta[self.sl_gamma_dist][0])

    def gamma_extra(self, theta: np.ndarray) -> np.ndarray:
        return theta[self.sl_gamma_extra]

    def gamma_spatial_extra(self, theta: np.ndarray) -> np.ndarray:
        return theta[self.sl_gamma_spatial_extra]

    def gamma_temporal_extra(self, theta: np.ndarray) -> np.ndarray:
        return theta[self.sl_gamma_temporal_extra]

    # ------------------------------------------------------------------ #

    def zeros(self) -> np.ndarray:
        return np.zeros(self.n_params, dtype=np.float64)
