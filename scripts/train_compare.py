#!/usr/bin/env python3
"""Train null and full Poisson OD models for comparison.

Null model : α_i + β_j + η_t + γ_dist + γ_holiday
Full model : + γ_elev (spatial) + γ_temp/precip/wind/humidity (temporal)

Ridge penalty applies to γ parameters only (not station / temporal FEs).

Usage
-----
python scripts/train_compare.py \\
    --data-dir data/processed \\
    --out-dir  models \\
    --ridge    1e-3 \\
    --maxiter  10000
"""

import argparse
import pickle
import sys
import time
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
    p = argparse.ArgumentParser(description="Train null and full Poisson OD models")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out-dir",  default="models")
    p.add_argument("--ridge",    type=float, default=1e-3)
    p.add_argument("--maxiter",  type=int,   default=10000)
    p.add_argument("--gtol",     type=float, default=1e-5)
    return p.parse_args()


def load_data(data_dir: Path) -> dict:
    print("Loading processed data …")
    obs_train   = pd.read_parquet(data_dir / "obs_train.parquet")
    stations    = pd.read_parquet(data_dir / "stations.parquet")
    dist_matrix = np.load(data_dir / "dist_matrix.npy")
    cal_train   = pd.read_parquet(data_dir / "cal_train.parquet")

    elev_path = data_dir / "elev_matrix.npy"
    elev_matrix = np.load(elev_path) if elev_path.exists() else None

    n_stations = len(stations)
    wx_present = [c for c in WEATHER_VARS if c in obs_train.columns
                  and obs_train[c].notna().any()]
    has_elev   = (elev_matrix is not None
                  and stations["elevation_m"].abs().sum() > 0)

    print(f"  {n_stations} stations  |  "
          f"{len(obs_train):,} training OD cells")
    print(f"  Elevation available: {has_elev}")
    print(f"  Weather vars:        {wx_present}")

    return dict(
        obs_train=obs_train, stations=stations,
        dist_matrix=dist_matrix, cal_train=cal_train,
        elev_matrix=elev_matrix,
        n_stations=n_stations,
        wx_present=wx_present, has_elev=has_elev,
    )


def build_layout_and_data(
    d: dict,
    include_extras: bool,
) -> tuple[ParamLayout, PoissonData]:
    n_stations = d["n_stations"]
    if include_extras:
        extra_gamma: tuple = ()
        n_spatial = 0
        if d["has_elev"]:
            extra_gamma = ("elev_gain",)
            n_spatial   = 1
        if d["wx_present"]:
            extra_gamma = extra_gamma + tuple(d["wx_present"])
    else:
        extra_gamma = ()
        n_spatial   = 0

    layout = ParamLayout(
        n_stations=n_stations,
        extra_gamma=extra_gamma,
        n_spatial_gamma=n_spatial,
    )
    elev = d["elev_matrix"] if include_extras else None
    data = PoissonData.build(
        d["obs_train"], d["dist_matrix"], layout, d["cal_train"], elev
    )
    return layout, data


def train_one(
    label: str,
    data: PoissonData,
    ridge: float,
    maxiter: int,
    gtol: float,
    theta0: np.ndarray | None = None,
) -> object:
    print(f"\n{'='*60}")
    print(f"  Training: {label}")
    print(f"  Parameters : {data.layout.n_params}")
    print(f"  Extra γ    : {list(data.layout.extra_gamma)}")
    print(f"{'='*60}")
    t0 = time.time()

    from baywheels.model.fit import FitResult
    from scipy.optimize import minimize
    from baywheels.model.poisson import neg_log_likelihood

    layout = data.layout
    if theta0 is None:
        theta0 = layout.zeros()
    else:
        # theta0 may come from null model with fewer params; expand if needed
        if len(theta0) < layout.n_params:
            new_theta = layout.zeros()
            new_theta[:len(theta0)] = theta0
            theta0 = new_theta

    nll_history: list[float] = []

    def objective(theta):
        nll, grad = neg_log_likelihood(theta, data, ridge=ridge)
        nll_history.append(nll)
        return nll, grad

    bounds = [(None, None)] * layout.n_params
    for fixed_idx in [
        layout.sl_alpha.start,
        layout.sl_hour.start,
        layout.sl_dow.start,
        layout.sl_month.start,
    ]:
        bounds[fixed_idx] = (0.0, 0.0)

    result = minimize(
        objective, theta0,
        method="L-BFGS-B", jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "gtol": gtol},
    )

    elapsed = time.time() - t0
    grad_norm = float(np.linalg.norm(result.jac)) if result.jac is not None else float("nan")
    status = "CONVERGED" if result.success else "NOT CONVERGED"
    print(f"\n  [{status}] {result.message}")
    print(f"  iterations={result.nit}  nll={result.fun:.4f}  "
          f"‖grad‖={grad_norm:.2e}  elapsed={elapsed:.1f}s")

    return FitResult(
        theta=result.x, layout=layout,
        nll=float(result.fun), grad_norm=grad_norm,
        n_iter=result.nit, converged=result.success,
        nll_history=nll_history, ridge=ridge,
    )


def print_coefficients(result, label: str) -> None:
    print(f"\n--- {label} coefficients ---")
    ly = result.layout
    theta = result.theta
    print(f"  γ_dist    = {ly.gamma_dist(theta):+.4f}")
    print(f"  γ_holiday = {ly.gamma_holiday(theta):+.4f}")
    if ly.n_spatial_gamma > 0:
        g = ly.gamma_spatial_extra(theta)
        for name, val in zip(ly.extra_gamma[:ly.n_spatial_gamma], g):
            print(f"  γ_{name:<26} = {val:+.6f}")
    if ly.n_temporal_gamma > 0:
        g = ly.gamma_temporal_extra(theta)
        names = ly.extra_gamma[ly.n_spatial_gamma:]
        for name, val in zip(names, g):
            print(f"  γ_{name:<26} = {val:+.6f}")


def main() -> None:
    args     = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load_data(data_dir)

    # ── 1. Null model ─────────────────────────────────────────────────────
    null_layout, null_data = build_layout_and_data(d, include_extras=False)
    null_result = train_one(
        "Null model (no elevation / weather)",
        null_data, args.ridge, args.maxiter, args.gtol,
    )
    null_bundle = {
        "result":      null_result,
        "stations":    d["stations"],
        "dist_matrix": d["dist_matrix"],
        "elev_matrix": None,
        "label":       "null",
    }
    null_path = out_dir / "null_model.pkl"
    with open(null_path, "wb") as f:
        pickle.dump(null_bundle, f, protocol=5)
    print(f"\nSaved null model → {null_path}")
    print_coefficients(null_result, "Null")

    # ── 2. Full model (warm-started from null θ) ──────────────────────────
    full_layout, full_data = build_layout_and_data(d, include_extras=True)
    full_result = train_one(
        "Full model (+ elevation + weather)",
        full_data, args.ridge, args.maxiter, args.gtol,
        theta0=null_result.theta,   # warm start from null solution
    )
    full_bundle = {
        "result":      full_result,
        "stations":    d["stations"],
        "dist_matrix": d["dist_matrix"],
        "elev_matrix": d["elev_matrix"],
        "label":       "full",
    }
    full_path = out_dir / "full_model.pkl"
    with open(full_path, "wb") as f:
        pickle.dump(full_bundle, f, protocol=5)
    print(f"\nSaved full model → {full_path}")
    print_coefficients(full_result, "Full")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  {'Model':<8}  {'Params':>7}  {'NLL':>14}  {'‖grad‖':>10}  {'Converged'}")
    for lbl, res in [("Null", null_result), ("Full", full_result)]:
        print(f"  {lbl:<8}  {res.layout.n_params:>7}  "
              f"{res.nll:>14.2f}  {res.grad_norm:>10.2e}  {res.converged}")


if __name__ == "__main__":
    main()
