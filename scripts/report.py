#!/usr/bin/env python3
"""Generate all report figures and tables for the Bay Wheels OD demand model.

Expects both null_model.pkl and full_model.pkl in --model-dir.

Outputs (saved to --out-dir):
  fig1_convergence.png      NLL training curves, null vs full
  fig2_obs_vs_pred.png      Observed-vs-predicted scatter, null vs full
  fig3_temporal_effects.png Hour / DoW / Month fixed effects (full model)
  fig4_distance_decay.png   Fitted distance-decay curve
  fig5_importance.png       Permutation feature importance (full model)
  fig6_station_effects.png  Station attractiveness (α + β histograms)
  metrics_table.txt         Printed metrics for LaTeX table

Usage
-----
python scripts/report.py \\
    --model-dir  models \\
    --data-dir   data/processed \\
    --out-dir    figures
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from baywheels.eval.diagnostics import check_marginal_balance
from baywheels.eval.importance import permutation_importance
from baywheels.eval.metrics import summary, poisson_deviance
from baywheels.model.params import ParamLayout
from baywheels.model.poisson import PoissonData, predict_mu_obs


WEATHER_VARS = (
    "temperature_2m", "precipitation",
    "wind_speed_10m", "relative_humidity_2m",
)

STYLE = {
    "null": dict(color="#4e79a7", ls="--", lw=1.5, label="Null model"),
    "full": dict(color="#e15759", ls="-",  lw=1.5, label="Full model"),
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 120,
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_bundle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_split(data_dir: Path, split: str,
               dist_matrix: np.ndarray,
               elev_matrix,
               layout: ParamLayout) -> PoissonData:
    obs = pd.read_parquet(data_dir / f"obs_{split}.parquet")
    cal = pd.read_parquet(data_dir / f"cal_{split}.parquet")
    return PoissonData.build(obs, dist_matrix, layout, cal, elev_matrix)


def get_metrics(theta, data) -> dict:
    mu = predict_mu_obs(theta, data)
    return summary(data.counts, mu)


def savefig(fig, path: Path, **kw):
    fig.savefig(path, bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Convergence curves
# ─────────────────────────────────────────────────────────────────────────────

def fig_convergence(null_result, full_result, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: absolute NLL (shows the two models' final levels)
    ax = axes[0]
    for lbl, res in [("null", null_result), ("full", full_result)]:
        h = np.array(res.nll_history)
        ax.plot(np.arange(1, len(h) + 1), h / 1e6, **STYLE[lbl])
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("NLL (×10⁶)")
    ax.set_title("Training NLL (absolute)")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Right: convergence gap  NLL_t - NLL_final  on log scale
    ax = axes[1]
    for lbl, res in [("null", null_result), ("full", full_result)]:
        h = np.array(res.nll_history)
        gap = h - h.min()
        # Keep only points where gap > 0 to avoid log(0)
        pos = gap > 0
        ax.plot(np.where(pos)[0] + 1, gap[pos], **STYLE[lbl])
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("NLL − NLL* (convergence gap)")
    ax.set_title("Convergence rate (log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.suptitle("L-BFGS-B training convergence", fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir / "fig1_convergence.png", dpi=150)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Observed vs predicted
# ─────────────────────────────────────────────────────────────────────────────

def fig_obs_vs_pred(null_result, full_result,
                    null_test: PoissonData, full_test: PoissonData,
                    out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    rng = np.random.default_rng(42)
    MAX = 30_000

    for ax, lbl, result, data in [
        (axes[0], "Null", null_result, null_test),
        (axes[1], "Full", full_result, full_test),
    ]:
        mu = predict_mu_obs(result.theta, data)
        y  = data.counts
        if len(y) > MAX:
            idx = rng.choice(len(y), MAX, replace=False)
            y, mu = y[idx], mu[idx]

        met = summary(data.counts, predict_mu_obs(result.theta, data))
        r2  = met["pearson_r"] ** 2

        ax.scatter(mu, y, s=3, alpha=0.25, rasterized=True,
                   color=STYLE[lbl.lower()]["color"])
        lim = max(mu.max(), y.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
        ax.set_xlabel("Predicted μ")
        ax.set_ylabel("Observed N")
        ax.set_title(f"{lbl} model — test set\n"
                     f"R²={r2:.4f}  RMSE={met['rmse']:.2f}")
        ax.legend(fontsize=9)

    fig.suptitle("Observed vs Predicted counts (OD cell–hour aggregates)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    savefig(fig, out_dir / "fig2_obs_vs_pred.png", dpi=150)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Temporal fixed effects
# ─────────────────────────────────────────────────────────────────────────────

def fig_temporal_effects(full_result, out_dir: Path):
    ly    = full_result.layout
    theta = full_result.theta

    eta_h = ly.eta_hour(theta);  eta_h = eta_h - eta_h.mean()
    eta_d = ly.eta_dow(theta);   eta_d = eta_d - eta_d.mean()
    eta_m = ly.eta_month(theta); eta_m = eta_m - eta_m.mean()

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    # Hour of day
    ax = axes[0]
    ax.bar(range(24), np.exp(eta_h), color="#e15759", edgecolor="white", lw=0.4)
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Relative demand factor")
    ax.set_title("Intraday demand profile")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], rotation=30)

    # Day of week
    ax = axes[1]
    dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colors = ["#e15759" if i < 5 else "#76b7b2" for i in range(7)]
    ax.bar(dow, np.exp(eta_d), color=colors, edgecolor="white", lw=0.4)
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Relative demand factor")
    ax.set_title("Day-of-week profile")

    # Month
    ax = axes[2]
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.bar(months, np.exp(eta_m), color="#4e79a7", edgecolor="white", lw=0.4)
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Month")
    ax.set_ylabel("Relative demand factor")
    ax.set_title("Seasonal (monthly) profile")
    ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Fitted temporal fixed effects (full model, demeaned)",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir / "fig3_temporal_effects.png", dpi=150)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Distance decay
# ─────────────────────────────────────────────────────────────────────────────

def fig_distance_decay(full_result, full_train: PoissonData, out_dir: Path):
    """Station-effect-adjusted trip volume vs distance, with fitted line.

    Under the model: log(N_ij) - α_i - β_j ≈ γ_dist·d_ij + log(S)
    so binned medians of the left-hand side should lie on the fitted line.
    """
    from scipy.special import logsumexp

    ly    = full_result.layout
    theta = full_result.theta
    alpha = ly.alpha(theta)
    beta  = ly.beta(theta)

    # ── Sum counts per (orig, dest) pair ─────────────────────────────────
    pair_key = full_train.orig.astype(np.int64) * 10_000 + full_train.dest.astype(np.int64)
    unique_pairs, inverse = np.unique(pair_key, return_inverse=True)
    N_ij   = np.bincount(inverse, weights=full_train.counts,
                         minlength=len(unique_pairs))
    # representative distance per pair (same for all hours)
    D_ij   = np.bincount(inverse, weights=full_train.dist_obs,
                         minlength=len(unique_pairs)) / \
             np.bincount(inverse, minlength=len(unique_pairs))
    orig_u = (unique_pairs // 10_000).astype(np.int32)
    dest_u = (unique_pairs %  10_000).astype(np.int32)

    # ── log(N_ij) - α_i - β_j ────────────────────────────────────────────
    log_residual = (np.log(N_ij + 1e-9)
                    - alpha[orig_u] - beta[dest_u])

    # ── Compute log(S) from training calendar ────────────────────────────
    eta_cal = (ly.eta_hour(theta)[full_train.hour_cal.astype(np.int32)]
               + ly.eta_dow(theta)[full_train.dow_cal.astype(np.int32)]
               + ly.eta_month(theta)[full_train.month_cal.astype(np.int32)]
               + ly.gamma_holiday(theta) * full_train.hol_cal)
    if full_train.weather_cal is not None:
        eta_cal = eta_cal + full_train.weather_cal @ ly.gamma_temporal_extra(theta)
    log_S = float(logsumexp(eta_cal))

    # ── Bin by distance (0.5 km bins, 0–15 km) ───────────────────────────
    mask = (D_ij > 0.1) & (D_ij <= 15)
    d_filt   = D_ij[mask]
    res_filt = log_residual[mask]

    bin_size = 0.5
    d_bin = np.floor(d_filt / bin_size).astype(int)
    bins  = np.arange(0, int(15 / bin_size))
    bin_centers = (bins + 0.5) * bin_size
    bin_med = np.array([
        np.median(res_filt[d_bin == b]) if (d_bin == b).sum() >= 3 else np.nan
        for b in bins
    ])
    valid = np.isfinite(bin_med)

    # ── Fitted line: γ_dist · d + log(S) ─────────────────────────────────
    g_dist  = ly.gamma_dist(theta)
    d_range = np.linspace(0, 15, 300)
    fitted  = g_dist * d_range + log_S

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(bin_centers[valid], bin_med[valid], s=30,
               color="#aaaaaa", zorder=3,
               label="Empirical (binned median per OD pair)")
    style = {**STYLE["full"], "label": f"Model fit  (γ_dist = {g_dist:.3f})"}
    ax.plot(d_range, fitted, **style)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel(r"$\log N_{ij} - \alpha_i - \beta_j$")
    ax.set_title("Distance decay: station-adjusted trip volume vs distance")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, out_dir / "fig4_distance_decay.png", dpi=150)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def fig_importance(full_result, full_train: PoissonData,
                   null_result, null_train: PoissonData,
                   out_dir: Path, n_repeats: int = 3):
    print("  Computing permutation importance (full model, train) …")
    imp_full = permutation_importance(
        full_result.theta, full_train, n_repeats=n_repeats, ridge=full_result.ridge
    )
    print("  Computing permutation importance (null model, train) …")
    imp_null = permutation_importance(
        null_result.theta, null_train, n_repeats=n_repeats, ridge=null_result.ridge
    )

    fig, axes = plt.subplots(2, 1, figsize=(7, 9))

    for ax, imp, lbl in [(axes[0], imp_null, "Null"), (axes[1], imp_full, "Full")]:
        features = [r.feature for r in imp]
        deltas   = [r.delta_nll for r in imp]
        stds     = [r.delta_nll_std for r in imp]
        color    = STYLE[lbl.lower()]["color"]
        ax.barh(features[::-1], deltas[::-1],
                xerr=stds[::-1], color=color,
                error_kw=dict(ecolor="gray", capsize=3), edgecolor="white")
        ax.set_xlabel("ΔNLL (higher = more important)")
        ax.set_title(f"{lbl} model — permutation importance\n(train set, {n_repeats} repeats)")
        ax.axvline(0, color="black", lw=0.8)

    fig.tight_layout()
    savefig(fig, out_dir / "fig5_importance.png", dpi=150)

    return imp_full, imp_null


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Station effects
# ─────────────────────────────────────────────────────────────────────────────

def fig_station_effects(full_result, stations: pd.DataFrame, out_dir: Path):
    ly    = full_result.layout
    theta = full_result.theta
    alpha = ly.alpha(theta)
    beta  = ly.beta(theta)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    axes[0].hist(alpha, bins=40, color="#e15759", edgecolor="white", lw=0.3)
    axes[0].set_xlabel("Origin fixed effect α_i")
    axes[0].set_ylabel("Number of stations")
    axes[0].set_title("Distribution of origin attractiveness")
    axes[0].axvline(0, color="black", lw=1, ls="--")

    axes[1].hist(beta, bins=40, color="#4e79a7", edgecolor="white", lw=0.3)
    axes[1].set_xlabel("Destination fixed effect β_j")
    axes[1].set_ylabel("Number of stations")
    axes[1].set_title("Distribution of destination attractiveness")
    axes[1].axvline(0, color="black", lw=1, ls="--")

    fig.suptitle("Station-level fixed effects (full model)", fontsize=12, y=1.01)
    fig.tight_layout()
    savefig(fig, out_dir / "fig6_station_effects.png", dpi=150)

    # Also print top/bottom 10 stations by α + β
    combined = alpha + beta
    df = stations[["station_id", "name"]].copy() if "name" in stations.columns \
         else stations[["station_id"]].copy()
    df["alpha"]    = alpha
    df["beta"]     = beta
    df["combined"] = combined
    print("\n--- Top 10 stations by α+β (most popular overall) ---")
    print(df.nlargest(10, "combined").to_string(index=False))
    print("\n--- Bottom 10 stations by α+β (least popular overall) ---")
    print(df.nsmallest(10, "combined").to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Covariate coefficient summary (visual table)
# ─────────────────────────────────────────────────────────────────────────────

def fig_coeff_summary(full_result, out_dir: Path):
    ly    = full_result.layout
    theta = full_result.theta

    names, vals = [], []
    names.append("γ_dist");    vals.append(ly.gamma_dist(theta))
    names.append("γ_holiday"); vals.append(ly.gamma_holiday(theta))
    if ly.n_spatial_gamma > 0:
        for name, v in zip(ly.extra_gamma[:ly.n_spatial_gamma],
                           ly.gamma_spatial_extra(theta)):
            names.append(f"γ_{name}"); vals.append(v)
    if ly.n_temporal_gamma > 0:
        for name, v in zip(ly.extra_gamma[ly.n_spatial_gamma:],
                           ly.gamma_temporal_extra(theta)):
            names.append(f"γ_{name}"); vals.append(v)

    vals  = np.array(vals)
    colors = ["#e15759" if v < 0 else "#4e79a7" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="white", lw=0.3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coefficient value")
    ax.set_title("Fitted covariate coefficients γ (full model)")
    fig.tight_layout()
    savefig(fig, out_dir / "fig7_coefficients.png", dpi=150)

    return names, vals


# ─────────────────────────────────────────────────────────────────────────────
# Metrics table
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_table(
    models: list[tuple],   # [(label, result, train_data, test_data), ...]
    out_dir: Path,
):
    rows = []
    for lbl, res, tr, te in models:
        for split_lbl, data in [("Train", tr), ("Test", te)]:
            mu = predict_mu_obs(res.theta, data)
            m  = summary(data.counts, mu)
            rows.append({
                "Model":    lbl,
                "Split":    split_lbl,
                "N cells":  int(m["n_obs"]),
                "Total obs":int(m["total_observed"]),
                "RMSE":     round(m["rmse"], 4),
                "Pearson R":round(m["pearson_r"], 4),
                "Pois Dev": round(m["poisson_deviance"], 0),
                "NLL":      round(res.nll, 2),
            })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE")
    print("=" * 80)
    print(df.to_string(index=False))

    txt_path = out_dir / "metrics_table.txt"
    df.to_csv(txt_path.with_suffix(".csv"), index=False)
    with open(txt_path, "w") as fh:
        fh.write("MODEL COMPARISON TABLE\n")
        fh.write("=" * 80 + "\n")
        fh.write(df.to_string(index=False) + "\n\n")

        null_test_dev = df[(df.Model=="Null") & (df.Split=="Test")]["Pois Dev"].values[0]
        for lbl, *_ in models:
            if lbl == "Null":
                continue
            row_dev = df[(df.Model==lbl) & (df.Split=="Test")]["Pois Dev"].values[0]
            imp = (null_test_dev - row_dev) / null_test_dev * 100
            fh.write(f"Test deviance improvement ({lbl} vs Null): {imp:.2f}%\n")

    print(f"\nSaved metrics → {txt_path} + .csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate Bay Wheels OD model report figures")
    p.add_argument("--model-dir",  default="models")
    p.add_argument("--data-dir",   default="data/processed")
    p.add_argument("--out-dir",    default="figures")
    p.add_argument("--n-repeats",  type=int, default=3,
                   help="Permutation importance repeats (3–5 recommended)")
    return p.parse_args()


def main():
    args      = parse_args()
    model_dir = Path(args.model_dir)
    data_dir  = Path(args.data_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model bundles …")
    null_bundle = load_bundle(model_dir / "null_model.pkl")
    full_bundle = load_bundle(model_dir / "full_model.pkl")

    null_result = null_bundle["result"]
    full_result = full_bundle["result"]

    dm_null = null_bundle["dist_matrix"]
    dm_full = full_bundle["dist_matrix"]
    em_null = null_bundle.get("elev_matrix")
    em_full = full_bundle.get("elev_matrix")
    stations = full_bundle["stations"]

    print("Loading train / test splits …")
    null_train = load_split(data_dir, "train", dm_null, em_null, null_result.layout)
    null_test  = load_split(data_dir, "test",  dm_null, em_null, null_result.layout)
    full_train = load_split(data_dir, "train", dm_full, em_full, full_result.layout)
    full_test  = load_split(data_dir, "test",  dm_full, em_full, full_result.layout)

    print("\nGenerating figures …")

    print("  fig1 – convergence curves")
    fig_convergence(null_result, full_result, out_dir)

    print("  fig2 – observed vs predicted")
    fig_obs_vs_pred(null_result, full_result, null_test, full_test, out_dir)

    print("  fig3 – temporal fixed effects")
    fig_temporal_effects(full_result, out_dir)

    print("  fig4 – distance decay vs empirical")
    fig_distance_decay(full_result, full_train, out_dir)

    print("  fig5 – feature importance")
    fig_importance(full_result, full_train, null_result, null_train,
                   out_dir, n_repeats=args.n_repeats)

    print("  fig6 – station effects")
    fig_station_effects(full_result, stations, out_dir)

    print("  fig7 – coefficient summary")
    names, vals = fig_coeff_summary(full_result, out_dir)

    print("\n  Metrics table")
    df_metrics = print_metrics_table(
        [
            ("Null", null_result, null_train, null_test),
            ("Full", full_result, full_train, full_test),
        ],
        out_dir,
    )

    # ── Convergence summary ───────────────────────────────────────────────
    print("\n=== Convergence summary ===")
    for lbl, res in [("Null", null_result), ("Full", full_result)]:
        status = "CONVERGED" if res.converged else "not converged"
        print(f"  {lbl}: nll={res.nll:.2f}  ‖grad‖={res.grad_norm:.2e}  "
              f"iter={res.n_iter}  [{status}]")

    print("\n=== Fitted γ coefficients (full model) ===")
    for n, v in zip(names, vals):
        print(f"  {n:<30} = {v:+.6f}")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
