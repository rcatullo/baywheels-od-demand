#!/usr/bin/env python3
"""Evaluate a fitted Poisson OD model on train and test sets.

Outputs
-------
  - Predictive accuracy table (MAE, RMSE, Pearson R, Poisson deviance)
  - Fitted covariate coefficients with interpretation
  - Permutation feature importance
  - Marginal balance check (station-level convergence invariant)
  - diagnostics.png: convergence curve + observed-vs-predicted scatter

Usage
-----
python scripts/evaluate.py \\
    --model    models/model.pkl \\
    --data-dir data/processed \\
    --out-dir  models
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from baywheels.eval.diagnostics import check_marginal_balance, plot_convergence, plot_residuals
from baywheels.eval.importance import permutation_importance
from baywheels.eval.metrics import summary
from baywheels.model.params import ParamLayout
from baywheels.model.poisson import PoissonData, predict_mu_obs


WEATHER_VARS = (
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Poisson OD model")
    p.add_argument("--model",      default="models/model.pkl")
    p.add_argument("--data-dir",   default="data/processed")
    p.add_argument("--out-dir",    default="models")
    p.add_argument("--n-repeats",  type=int, default=5)
    return p.parse_args()


def load_split(data_dir: Path, split: str,
               dist_matrix: np.ndarray,
               elev_matrix: np.ndarray | None,
               layout: ParamLayout) -> PoissonData:
    obs = pd.read_parquet(data_dir / f"obs_{split}.parquet")
    cal = pd.read_parquet(data_dir / f"cal_{split}.parquet")
    return PoissonData.build(obs, dist_matrix, layout, cal, elev_matrix)


def print_metrics(label: str, met: dict) -> None:
    print(f"\n  {label}")
    print(f"    Observations : {int(met['n_obs']):>12,}")
    print(f"    Total obs N  : {int(met['total_observed']):>12,}")
    print(f"    Total pred μ : {met['total_predicted']:>12.0f}")
    print(f"    MAE          : {met['mae']:>12.4f}")
    print(f"    RMSE         : {met['rmse']:>12.4f}")
    print(f"    Pearson R    : {met['pearson_r']:>12.4f}")
    print(f"    Poisson Dev  : {met['poisson_deviance']:>12.0f}")


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model} …")
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)

    result      = bundle["result"]
    dist_matrix = bundle["dist_matrix"]
    elev_matrix = bundle.get("elev_matrix")
    layout      = result.layout
    data_dir    = Path(args.data_dir)
    theta       = result.theta

    # ── Predictive accuracy ───────────────────────────────────────────────
    print("\n=== Predictive accuracy ===")
    train_data = load_split(data_dir, "train", dist_matrix, elev_matrix, layout)
    for split in ("train", "test"):
        try:
            data = load_split(data_dir, split, dist_matrix, elev_matrix, layout)
        except FileNotFoundError:
            print(f"  {split}: not found, skipping")
            continue
        mu  = predict_mu_obs(theta, data)
        met = summary(data.counts, mu)
        print_metrics(split.upper(), met)

    # ── Fitted coefficients ───────────────────────────────────────────────
    print("\n=== Fitted covariate coefficients ===")
    ly = layout
    print(f"  γ_dist    = {ly.gamma_dist(theta):+.4f}  "
          f"(demand decays with distance; expected < 0)")
    print(f"  γ_holiday = {ly.gamma_holiday(theta):+.4f}")

    if ly.n_spatial_gamma > 0:
        g_sp = ly.gamma_spatial_extra(theta)
        names_sp = list(ly.extra_gamma[:ly.n_spatial_gamma])
        for name, val in zip(names_sp, g_sp):
            print(f"  γ_{name:<26} = {val:+.6f}"
                  f"  (uphill trips are {'rarer' if val < 0 else 'more common'})")

    if ly.n_temporal_gamma > 0:
        g_tmp  = ly.gamma_temporal_extra(theta)
        names_tmp = list(ly.extra_gamma[ly.n_spatial_gamma:])
        wx_interp = {
            "temperature_2m":       "warmer → more trips",
            "precipitation":        "more rain → fewer trips",
            "wind_speed_10m":       "stronger wind → fewer trips",
            "relative_humidity_2m": "more humid → effect TBD",
        }
        for name, val in zip(names_tmp, g_tmp):
            interp = wx_interp.get(name, "")
            print(f"  γ_{name:<26} = {val:+.6f}  ({interp})")

    # ── Fitted temporal effects ───────────────────────────────────────────
    print("\n=== Fitted temporal fixed effects ===")
    eta_h = ly.eta_hour(theta) - ly.eta_hour(theta).mean()
    eta_d = ly.eta_dow(theta)  - ly.eta_dow(theta).mean()
    eta_m = ly.eta_month(theta)- ly.eta_month(theta).mean()

    print("  η_hour (demeaned):")
    for h, v in enumerate(eta_h):
        bar = "█" * int(abs(v) * 8) if abs(v) > 0.05 else ""
        print(f"    {h:02d}h  {v:+.3f}  {bar}")

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print("  η_dow (demeaned):")
    for d, v in enumerate(eta_d):
        print(f"    {dow_names[d]}  {v:+.3f}")

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    print("  η_month (demeaned):")
    for m, v in enumerate(eta_m):
        print(f"    {month_names[m]}  {v:+.3f}")

    # ── Marginal balance ─────────────────────────────────────────────────
    print("\n=== Marginal balance (score-equation check) ===")
    balance = check_marginal_balance(result, train_data)
    for k, v in balance.items():
        status = "✓" if abs(v) < 1.0 else "✗"
        print(f"    {k:30s}: {v:.4e}  {status}")

    # ── Feature importance ────────────────────────────────────────────────
    print("\n=== Permutation feature importance (train) ===")
    imp = permutation_importance(
        theta, train_data,
        n_repeats=args.n_repeats,
        ridge=result.ridge,
    )
    print(f"    {'Feature':<20}  {'ΔNLL (mean)':>12}  {'ΔNLL (std)':>12}")
    for r in imp:
        print(f"    {r.feature:<20}  {r.delta_nll:>12.2f}  {r.delta_nll_std:>12.2f}")

    # ── Plots ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        plot_convergence(result, ax=axes[0])
        plot_residuals(result, train_data, ax=axes[1])
        fig.tight_layout()
        fig.savefig(out_dir / "diagnostics.png", dpi=120)
        print(f"\nSaved → {out_dir}/diagnostics.png")

        # Hour-of-day demand profile
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.bar(range(24), np.exp(eta_h))
        ax2.set_xlabel("Hour of day")
        ax2.set_ylabel("Relative demand (exp η_hour)")
        ax2.set_title("Fitted hourly demand profile")
        ax2.set_xticks(range(24))
        fig2.tight_layout()
        fig2.savefig(out_dir / "hourly_profile.png", dpi=120)
        print(f"Saved → {out_dir}/hourly_profile.png")
    except Exception as exc:
        print(f"\n[Warning] Could not save plots: {exc}")


if __name__ == "__main__":
    main()
