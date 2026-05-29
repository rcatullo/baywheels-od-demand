#!/usr/bin/env python3
"""Fit the Poisson OD model (with elevation and weather covariates) and save.

Model
-----
log μ_ijt = α_i + β_j                         (station fixed effects)
          + η_hour[h] + η_dow[d] + η_month[m]  (temporal fixed effects)
          + γ_holiday · hol_t                   (holiday)
          + γ_dist · d_ij                       (haversine distance)
          + γ_elev · Δelev_ij                   (elevation gain, if available)
          + γ_temp · temp_t + γ_precip · precip_t
          + γ_wind · wind_t + γ_hum · hum_t    (area-wide weather, if available)

Usage
-----
python scripts/train_baseline.py \\
    --data-dir data/processed \\
    --out-dir  models \\
    --ridge    1e-3 \\
    --maxiter  500
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from baywheels.model.fit import fit_baseline
from baywheels.model.params import ParamLayout
from baywheels.model.poisson import PoissonData


WEATHER_VARS = (
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit Poisson OD model")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out-dir",  default="models")
    p.add_argument("--ridge",    type=float, default=1e-3)
    p.add_argument("--maxiter",  type=int,   default=500)
    p.add_argument("--gtol",     type=float, default=1e-5)
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed data …")
    obs_train  = pd.read_parquet(data_dir / "obs_train.parquet")
    stations   = pd.read_parquet(data_dir / "stations.parquet")
    dist_matrix = np.load(data_dir / "dist_matrix.npy")
    cal_train  = pd.read_parquet(data_dir / "cal_train.parquet")

    # Load elevation matrix (optional — baseline falls back if absent)
    elev_matrix_path = data_dir / "elev_matrix.npy"
    elev_matrix = np.load(elev_matrix_path) if elev_matrix_path.exists() else None

    n_stations = len(stations)

    # Determine which extra covariates are present in the data
    wx_present  = [c for c in WEATHER_VARS if c in obs_train.columns
                   and obs_train[c].notna().any()]
    has_elev    = elev_matrix is not None and stations["elevation_m"].abs().sum() > 0

    extra_gamma: tuple = ()
    n_spatial   = 0
    if has_elev:
        extra_gamma = ("elev_gain",)
        n_spatial   = 1
    if wx_present:
        extra_gamma = extra_gamma + tuple(wx_present)

    layout = ParamLayout(
        n_stations=n_stations,
        extra_gamma=extra_gamma,
        n_spatial_gamma=n_spatial,
    )

    print(f"  {n_stations} stations  |  {layout.n_params} parameters")
    print(f"  Spatial extras  : {list(extra_gamma[:n_spatial])}")
    print(f"  Temporal extras : {list(extra_gamma[n_spatial:])}")

    data = PoissonData.build(obs_train, dist_matrix, layout, cal_train, elev_matrix)

    result = fit_baseline(
        data,
        ridge=args.ridge,
        maxiter=args.maxiter,
        gtol=args.gtol,
        verbose=True,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    bundle = {
        "result":      result,
        "stations":    stations,
        "dist_matrix": dist_matrix,
        "elev_matrix": elev_matrix,
    }
    out_path = out_dir / "model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f, protocol=5)
    print(f"\nSaved model → {out_path}")

    # ── Covariate summary ─────────────────────────────────────────────────
    ly = result.layout
    print(f"\n  γ_dist    = {ly.gamma_dist(result.theta):+.4f}"
          f"  (negative = demand decays with distance)")
    if has_elev:
        g_elev = ly.gamma_spatial_extra(result.theta)[0]
        print(f"  γ_elev    = {g_elev:+.4f}"
              f"  (negative = fewer trips uphill)")
    if wx_present:
        g_wx = ly.gamma_temporal_extra(result.theta)
        for name, val in zip(wx_present, g_wx):
            print(f"  γ_{name:<26} = {val:+.6f}")
    print(f"  γ_holiday = {ly.gamma_holiday(result.theta):+.4f}")


if __name__ == "__main__":
    main()
