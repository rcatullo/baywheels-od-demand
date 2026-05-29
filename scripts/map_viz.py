#!/usr/bin/env python3
"""Visualise Bay Wheels stations and model results on a map.

Outputs
-------
figures/map_static.png     4-panel static map (report quality)
figures/map_interactive.html  Folium interactive map (exploration)

Usage
-----
python scripts/map_viz.py --model-dir models --data-dir data/processed --out-dir figures
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
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from baywheels.model.params import ParamLayout
from baywheels.model.poisson import PoissonData, predict_mu_obs

# Web-Mercator projection helpers (no geopandas needed)
def to_mercator(lat, lon):
    """Convert WGS-84 lat/lon to Web Mercator (EPSG:3857) metres."""
    R = 6_378_137.0
    x = np.radians(lon) * R
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R
    return x, y


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",  default="models")
    p.add_argument("--data-dir",   default="data/processed")
    p.add_argument("--out-dir",    default="figures")
    p.add_argument("--top-flows",  type=int, default=300,
                   help="How many busiest OD flows to draw on the map")
    return p.parse_args()


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Static map (matplotlib + contextily basemap)
# ─────────────────────────────────────────────────────────────────────────────

def static_map(stations, full_bundle, obs_train, out_path, top_flows=300):
    import contextily as ctx

    full_result = full_bundle["result"]
    full_ly     = full_result.layout

    # Filter to valid station coords
    st = stations[stations.lat > 30].copy()
    st["x"], st["y"] = to_mercator(st.lat.values, st.lng.values)

    # Station-level attractiveness from full model (α + β, relative)
    alpha = full_ly.alpha(full_result.theta)
    beta  = full_ly.beta(full_result.theta)
    st["attract"] = alpha[st.index] + beta[st.index]

    # Station-level origin volume (total observed trips departing)
    orig_counts = obs_train.groupby("orig_idx")["count"].sum()
    st["volume"] = orig_counts.reindex(st.index).fillna(0)

    # Top OD flows (by trip volume) for flow arrows
    flow = (obs_train.groupby(["orig_idx", "dest_idx"])["count"]
            .sum().reset_index().rename(columns={"count": "trips"}))
    flow = flow.nlargest(top_flows, "trips")

    # Mercator bounds with a small margin
    xmin, xmax = st.x.min() - 500, st.x.max() + 500
    ymin, ymax = st.y.min() - 500, st.y.max() + 500

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Bay Wheels OD Demand Model — Spatial Results", fontsize=15, y=1.01)

    titles = [
        "Station locations (coloured by total departures)",
        "Model attractiveness α+β (origin + destination FE)",
        f"Top {top_flows} busiest OD flows",
        "Model attractiveness vs observed departure volume",
    ]

    for ax in axes.flat[:3]:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.axis("off")

    def add_basemap(ax):
        try:
            ctx.add_basemap(ax, crs="EPSG:3857",
                            source=ctx.providers.CartoDB.Positron,
                            zoom="auto", attribution=False)
        except Exception:
            ax.set_facecolor("#e8e8e8")

    # ── Panel 1: Station locations by volume ──────────────────────────────
    ax = axes[0, 0]
    add_basemap(ax)
    vol_norm = plt.Normalize(st.volume.quantile(0.05), st.volume.quantile(0.95))
    sc = ax.scatter(st.x, st.y, c=st.volume, cmap="YlOrRd",
                    norm=vol_norm, s=30, alpha=0.85, linewidths=0, zorder=3)
    plt.colorbar(sc, ax=ax, shrink=0.6, label="Total departures")
    ax.set_title(titles[0], fontsize=11)

    # ── Panel 2: Model attractiveness ─────────────────────────────────────
    ax = axes[0, 1]
    add_basemap(ax)
    att_max = np.percentile(np.abs(st.attract), 95)
    att_norm = plt.Normalize(-att_max, att_max)
    sc2 = ax.scatter(st.x, st.y, c=st.attract, cmap="RdYlGn",
                     norm=att_norm, s=30, alpha=0.85, linewidths=0, zorder=3)
    plt.colorbar(sc2, ax=ax, shrink=0.6, label="α + β (green = more attractive)")
    ax.set_title(titles[1], fontsize=11)

    # ── Panel 3: Top OD flows ─────────────────────────────────────────────
    ax = axes[1, 0]
    add_basemap(ax)
    flow_norm = plt.Normalize(flow.trips.min(), flow.trips.max())
    flow_cmap = cm.get_cmap("Blues")
    for _, row in flow.iterrows():
        oi, di = int(row.orig_idx), int(row.dest_idx)
        if oi not in st.index or di not in st.index:
            continue
        ox, oy = st.loc[oi, "x"], st.loc[oi, "y"]
        dx, dy = st.loc[di, "x"], st.loc[di, "y"]
        c = flow_cmap(flow_norm(row.trips))
        ax.plot([ox, dx], [oy, dy], color=c, alpha=0.4,
                lw=0.5 + 2 * flow_norm(row.trips), zorder=2)
    ax.scatter(st.x, st.y, c="white", s=12, linewidths=0.4,
               edgecolors="gray", zorder=3, alpha=0.9)
    sm = cm.ScalarMappable(norm=flow_norm, cmap=flow_cmap)
    plt.colorbar(sm, ax=ax, shrink=0.6, label="Trips (train period)")
    ax.set_title(titles[2], fontsize=11)

    # ── Panel 4: Model attractiveness vs observed volume scatter ──────────
    ax = axes[1, 1]
    ax.axis("on")
    valid = st[st.volume > 0].copy()
    ax.scatter(valid.attract, np.log1p(valid.volume),
               c="#e15759", s=18, alpha=0.55, linewidths=0)
    # Trend line
    m, b = np.polyfit(valid.attract, np.log1p(valid.volume), 1)
    x_fit = np.linspace(valid.attract.min(), valid.attract.max(), 100)
    ax.plot(x_fit, m * x_fit + b, color="#4e79a7", lw=2, ls="--",
            label=f"slope={m:.2f}")
    ax.set_xlabel("Model attractiveness α + β")
    ax.set_ylabel("log(1 + observed departures)")
    ax.set_title(titles[3], fontsize=11)
    ax.legend(fontsize=9)
    ax.set_facecolor("#f5f5f5")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive folium map
# ─────────────────────────────────────────────────────────────────────────────

def interactive_map(stations, full_bundle, obs_train, out_path, top_flows=500):
    import folium
    from folium.plugins import MarkerCluster

    full_result = full_bundle["result"]
    full_ly     = full_result.layout

    st = stations[stations.lat > 30].copy()
    alpha    = full_ly.alpha(full_result.theta)
    beta     = full_ly.beta(full_result.theta)
    st["attract"] = alpha[st.index] + beta[st.index]

    orig_counts = obs_train.groupby("orig_idx")["count"].sum()
    st["volume"] = orig_counts.reindex(st.index).fillna(0)

    # Centre the map
    centre = [st.lat.mean(), st.lng.mean()]
    m = folium.Map(location=centre, zoom_start=12,
                   tiles="CartoDB positron")

    # ── Layer 1: Station attractiveness ───────────────────────────────────
    attract_layer = folium.FeatureGroup(name="Station attractiveness (α+β)", show=True)
    a_min = st.attract.quantile(0.05)
    a_max = st.attract.quantile(0.95)
    colormap_a = folium.LinearColormap(
        ["#d73027", "#fee090", "#91cf60", "#1a9850"],
        vmin=a_min, vmax=a_max,
        caption="Station attractiveness α+β"
    )
    for idx, row in st.iterrows():
        color = colormap_a(np.clip(row.attract, a_min, a_max))
        folium.CircleMarker(
            location=[row.lat, row.lng],
            radius=6,
            color="white", weight=0.5,
            fill=True, fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>{row.station_id}</b><br>"
                f"α = {alpha[idx]:+.3f}<br>"
                f"β = {beta[idx]:+.3f}<br>"
                f"α+β = {row.attract:+.3f}<br>"
                f"Departures = {int(row.volume):,}<br>"
                f"Elevation = {row.elevation_m:.0f} m",
                max_width=200
            ),
            tooltip=row.station_id,
        ).add_to(attract_layer)
    attract_layer.add_to(m)
    colormap_a.add_to(m)

    # ── Layer 2: Station volume (bubble size) ─────────────────────────────
    volume_layer = folium.FeatureGroup(name="Station departure volume", show=False)
    vol_max = st.volume.quantile(0.95)
    for idx, row in st.iterrows():
        r = 3 + 14 * np.clip(row.volume / vol_max, 0, 1)
        folium.CircleMarker(
            location=[row.lat, row.lng],
            radius=float(r),
            color="white", weight=0.5,
            fill=True, fill_color="#e15759", fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{row.station_id}</b><br>"
                f"Departures = {int(row.volume):,}",
                max_width=180),
            tooltip=f"{row.station_id}: {int(row.volume):,} trips",
        ).add_to(volume_layer)
    volume_layer.add_to(m)

    # ── Layer 3: Top OD flows ─────────────────────────────────────────────
    flow = (obs_train.groupby(["orig_idx", "dest_idx"])["count"]
            .sum().reset_index().rename(columns={"count": "trips"}))
    flow = flow.nlargest(top_flows, "trips")
    flow_max = flow.trips.max()

    flow_layer = folium.FeatureGroup(name=f"Top {top_flows} OD flows", show=False)
    colormap_f = folium.LinearColormap(
        ["#4e79a7", "#e15759"], vmin=flow.trips.min(), vmax=flow_max,
        caption="Trip volume"
    )
    for _, row in flow.iterrows():
        oi, di = int(row.orig_idx), int(row.dest_idx)
        if oi not in st.index or di not in st.index:
            continue
        olat, olng = st.loc[oi, "lat"], st.loc[oi, "lng"]
        dlat, dlng = st.loc[di, "lat"], st.loc[di, "lng"]
        weight = 1 + 4 * (row.trips / flow_max)
        folium.PolyLine(
            locations=[[olat, olng], [dlat, dlng]],
            color=colormap_f(row.trips),
            weight=float(weight), opacity=0.5,
            tooltip=f"{int(row.trips):,} trips",
        ).add_to(flow_layer)
    flow_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    m.save(str(out_path))
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    model_dir = Path(args.model_dir)
    data_dir  = Path(args.data_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data and models …")
    full_bundle = load_model(model_dir / "full_model.pkl")
    stations    = pd.read_parquet(data_dir / "stations.parquet")
    obs_train   = pd.read_parquet(data_dir / "obs_train.parquet")

    print("Building static map …")
    static_map(stations, full_bundle, obs_train,
               out_dir / "map_static.png", top_flows=args.top_flows)

    print("Building interactive map …")
    interactive_map(stations, full_bundle, obs_train,
                    out_dir / "map_interactive.html", top_flows=args.top_flows)

    print("\nDone.")


if __name__ == "__main__":
    main()
