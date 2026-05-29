#!/usr/bin/env python3
"""Prepare augmented Bay Wheels trip data for the Poisson OD model.

Reads from the weather- and elevation-augmented CSV directory, aggregates to
hourly OD counts, joins area-wide hourly weather from the SQLite cache, and
saves all arrays needed by train_model.py.

Outputs (in --out-dir)
----------------------
obs_train.parquet      hourly OD observations, training period
obs_test.parquet       hourly OD observations, test period
stations.parquet       station index table (with elevation_m)
cal_train.parquet      training calendar with area-wide weather
cal_test.parquet       test calendar with area-wide weather
dist_matrix.npy        (I, I) haversine distance matrix [km]
elev_matrix.npy        (I, I) elevation gain matrix [m] (dest − orig)
weather_stats.parquet  mean/std of weather variables (for reporting)

Usage
-----
python scripts/prepare_data.py \\
    --data-dir  ~/baywheels-tripdata-augmented \\
    --weather-db ~/weather_cache.db \\
    --out-dir   data/processed \\
    --train-years 2023 \\
    --test-years  2024
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import holidays as hol_lib

from baywheels.data.aggregator import aggregate
from baywheels.data.calendar import build_calendar
from baywheels.data.loader import load_files


WEATHER_COLS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare augmented Bay Wheels OD data")
    p.add_argument("--data-dir",    default="~/baywheels-tripdata-augmented",
                   help="Directory of weather+elevation-augmented CSVs")
    p.add_argument("--weather-db",  default="~/weather_cache.db",
                   help="SQLite DB from augment_weather.py")
    p.add_argument("--out-dir",     default="data/processed")
    p.add_argument("--train-years", nargs="+", type=int, default=[2023])
    p.add_argument("--test-years",  nargs="+", type=int, default=[2024])
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def us_holidays_set(years: list[int]) -> set:
    dates: set = set()
    for y in years:
        dates |= set(hol_lib.country_holidays("US", subdiv="CA", years=y).keys())
    return dates


def year_bounds(years: list[int]) -> tuple[str, str]:
    return f"{min(years)}-01-01", f"{max(years) + 1}-01-01"


def load_area_weather(weather_db: Path) -> pd.DataFrame | None:
    """Load hourly area-wide weather averaged across all valid ERA5 cells.

    Returns a DataFrame with columns [_hour, temperature_2m, precipitation,
    wind_speed_10m, relative_humidity_2m] where _hour is a string in format
    "YYYY-MM-DDTHH:00" (local Pacific time) matching the calendar/obs keys.
    """
    if not weather_db.exists():
        print(f"  [WARNING] Weather DB not found: {weather_db}")
        return None
    try:
        con = sqlite3.connect(weather_db)
        sql = """
            SELECT hour                        AS _hour,
                   AVG(temperature_2m)         AS temperature_2m,
                   AVG(precipitation)          AS precipitation,
                   AVG(wind_speed_10m)         AS wind_speed_10m,
                   AVG(relative_humidity_2m)   AS relative_humidity_2m
            FROM weather
            WHERE temperature_2m IS NOT NULL
            GROUP BY hour
        """
        df = pd.read_sql(sql, con)
        con.close()
        if df.empty:
            return None
        print(f"  Area-wide weather: {len(df):,} hours, "
              f"{df['_hour'].min()} → {df['_hour'].max()}")
        return df
    except Exception as exc:
        print(f"  [WARNING] Could not load weather DB: {exc}")
        return None


def join_obs_weather(obs: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Left-join area-wide weather to obs by matching the trip's floor-hour."""
    obs = obs.copy()
    obs["_hour"] = pd.to_datetime(obs["hour_bin"]).dt.strftime("%Y-%m-%dT%H:00")
    obs = obs.merge(weather, on="_hour", how="left")
    obs.drop(columns=["_hour"], inplace=True)
    return obs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args     = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    wx_db    = Path(args.weather_db).expanduser()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_years = sorted(set(args.train_years) | set(args.test_years))
    holidays  = us_holidays_set(all_years)

    # ── Trips ─────────────────────────────────────────────────────────────
    print(f"Loading trips from {data_dir} …")
    trips = load_files(data_dir, year_range=(min(all_years), max(all_years)))
    print(f"  {len(trips):,} named-station trips")
    print(f"  has elevation : {'start_elevation_m' in trips.columns}")

    # ── Aggregate ─────────────────────────────────────────────────────────
    print("Aggregating to hourly OD counts …")
    obs_all, stations, dist_matrix, elev_matrix = aggregate(trips, holidays)
    n_stations = len(stations)
    print(f"  {len(obs_all):,} observed (i,j,t) cells  |  {n_stations} stations")
    print(f"  Elevation range: "
          f"{stations['elevation_m'].min():.1f} – "
          f"{stations['elevation_m'].max():.1f} m  "
          f"(0 m = missing/sea-level default)")

    # ── Area-wide weather ─────────────────────────────────────────────────
    print("Loading area-wide hourly weather …")
    weather = load_area_weather(wx_db)

    # ── Train / test split ────────────────────────────────────────────────
    obs_all["_year"] = pd.to_datetime(obs_all["hour_bin"]).dt.year
    obs_train = obs_all[obs_all["_year"].isin(args.train_years)].drop(columns="_year").reset_index(drop=True)
    obs_test  = obs_all[obs_all["_year"].isin(args.test_years)].drop(columns="_year").reset_index(drop=True)
    print(f"  Train: {len(obs_train):,}  |  Test: {len(obs_test):,}")

    if weather is not None:
        obs_train = join_obs_weather(obs_train, weather)
        obs_test  = join_obs_weather(obs_test,  weather)
        for split, df in (("train", obs_train), ("test", obs_test)):
            miss = df["temperature_2m"].isna().mean()
            if miss > 0.001:
                print(f"  [WARNING] {split} weather miss rate: {miss*100:.1f}%")

    # ── Calendars ─────────────────────────────────────────────────────────
    print("Building calendars …")
    cal_train = build_calendar(*year_bounds(args.train_years), holidays, weather)
    cal_test  = build_calendar(*year_bounds(args.test_years),  holidays, weather)
    if weather is not None:
        miss = cal_train["temperature_2m"].isna().mean()
        print(f"  Calendar weather miss: {miss*100:.2f}%")

    # ── Weather statistics ────────────────────────────────────────────────
    wx_stats = pd.DataFrame()
    wx_cols_present = [c for c in WEATHER_COLS if c in obs_train.columns]
    if wx_cols_present:
        wx_stats = pd.DataFrame({
            "mean": obs_train[wx_cols_present].mean(),
            "std":  obs_train[wx_cols_present].std(),
            "min":  obs_train[wx_cols_present].min(),
            "max":  obs_train[wx_cols_present].max(),
        })
        print("\nWeather statistics (training observations):")
        print(wx_stats.round(3).to_string())

    # ── Save ─────────────────────────────────────────────────────────────
    print(f"\nSaving to {out_dir}/ …")
    obs_train.to_parquet(out_dir / "obs_train.parquet", index=False)
    obs_test.to_parquet(out_dir  / "obs_test.parquet",  index=False)
    stations.to_parquet(out_dir  / "stations.parquet",  index=True)
    cal_train.to_parquet(out_dir / "cal_train.parquet", index=False)
    cal_test.to_parquet(out_dir  / "cal_test.parquet",  index=False)
    np.save(out_dir / "dist_matrix.npy", dist_matrix)
    np.save(out_dir / "elev_matrix.npy", elev_matrix)
    if not wx_stats.empty:
        wx_stats.to_parquet(out_dir / "weather_stats.parquet")

    print("Done.")


if __name__ == "__main__":
    main()
