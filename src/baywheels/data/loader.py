"""Load and normalise Bay Wheels trip CSVs (raw or weather/elevation-augmented).

Two CSV schemas exist in the archive:
  - Pre-2021 ("old"): start_time, start_station_latitude/longitude, ...
  - 2021+    ("new"): started_at, start_lat, start_lng, ...

Both are normalised to a common column set.  If weather and elevation columns
are present (from augment_weather.py / augment_elevation.py) they are kept.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_COMMON_COLS = [
    # Core trip fields
    "started_at",
    "start_station_id",
    "start_lat",
    "start_lng",
    "end_station_id",
    "end_lat",
    "end_lng",
    # Elevation (augment_elevation.py)
    "start_elevation_m",
    "end_elevation_m",
    "elevation_gain_m",
    # Weather (augment_weather.py)
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
]

_OLD_RENAME = {
    "start_time":              "started_at",
    "start_station_latitude":  "start_lat",
    "start_station_longitude": "start_lng",
    "end_station_latitude":    "end_lat",
    "end_station_longitude":   "end_lng",
}


def _is_old_schema(cols: list[str]) -> bool:
    return "start_time" in cols


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    rename = _OLD_RENAME if _is_old_schema(list(df.columns)) else {}
    df = df.rename(columns=rename)
    available = [c for c in _COMMON_COLS if c in df.columns]
    df = df[available].copy()
    # format="mixed" handles files where some rows carry sub-second precision
    # (e.g. "2019-07-31 09:25:37.341") while others don't — pandas ≥2.0
    # infers a single format from the first rows and silently coerces the rest
    # to NaT unless told to handle mixed formats explicitly.
    df["started_at"] = pd.to_datetime(df["started_at"], format="mixed", errors="coerce")
    for col in (
        "start_lat", "start_lng", "end_lat", "end_lng",
        "start_elevation_m", "end_elevation_m", "elevation_gain_m",
        "temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_files(
    data_dir: str | Path,
    year_range: tuple[int, int] = (2023, 2024),
) -> pd.DataFrame:
    """Return normalised trips for *year_range* (inclusive).

    Works with both raw baywheels-tripdata/ and the augmented
    baywheels-tripdata-augmented/ directories.
    """
    data_dir = Path(data_dir)
    frames: list[pd.DataFrame] = []
    for path in sorted(data_dir.glob("*.csv")):
        stem = path.stem
        try:
            year = int(stem[:4])
        except ValueError:
            continue
        if not (year_range[0] <= year <= year_range[1]):
            continue
        df = pd.read_csv(path, low_memory=False)
        df = _normalise(df)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir} for years {year_range}"
        )

    out = pd.concat(frames, ignore_index=True)

    # Drop rows without station IDs or coordinates.  Dockless trips (null
    # station_id) are excluded because the OD model requires named stations.
    out = out.dropna(subset=["started_at", "start_station_id", "end_station_id"])
    out = out.dropna(subset=["start_lat", "start_lng", "end_lat", "end_lng"])

    # Normalise station IDs: remove trailing ".0" from numeric IDs read as
    # float64 (old-schema CSVs store integer station numbers).
    for col in ("start_station_id", "end_station_id"):
        out[col] = (
            out[col].astype(str)
                    .str.strip()
                    .str.replace(r"\.0$", "", regex=True)
        )

    out = out[out["start_station_id"] != out["end_station_id"]]
    return out.reset_index(drop=True)
