"""Aggregate normalised trip rows to hourly OD counts.

Output
------
obs : pd.DataFrame
    One row per observed (orig_idx, dest_idx, hour_bin) triple.
    Columns: orig_idx, dest_idx, hour_bin, hour, dow, month, is_holiday,
             count, dist_km.
stations : pd.DataFrame
    Integer-indexed station table with columns: station_id, lat, lng,
    elevation_m (median start_elevation_m across departing trips; 0 if absent).
dist_matrix : (I, I) float64 haversine distances in km.
elev_matrix : (I, I) float64 elevation gain [m] = dest_elev − orig_elev.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Return (n, n) symmetric matrix of haversine distances in km."""
    R = 6_371.0
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    )
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_station_table(trips: pd.DataFrame) -> pd.DataFrame:
    """Derive canonical station coordinates and elevations.

    Uses median lat/lng per station_id across all trips (start and end).
    Elevation is the median start_elevation_m for departing trips; stations
    with no elevation data get 0.0 m.
    """
    src = pd.concat(
        [
            trips[["start_station_id", "start_lat", "start_lng"]].rename(
                columns={
                    "start_station_id": "station_id",
                    "start_lat": "lat",
                    "start_lng": "lng",
                }
            ),
            trips[["end_station_id", "end_lat", "end_lng"]].rename(
                columns={
                    "end_station_id": "station_id",
                    "end_lat": "lat",
                    "end_lng": "lng",
                }
            ),
        ],
        ignore_index=True,
    )
    coords = src.groupby("station_id")[["lat", "lng"]].median().reset_index()
    coords = coords.sort_values("station_id").reset_index(drop=True)
    coords.index.name = "station_idx"

    # Station elevations from departing trips
    if "start_elevation_m" in trips.columns:
        elev = (
            trips.groupby("start_station_id")["start_elevation_m"]
                 .median()
                 .reindex(coords["station_id"])
                 .fillna(0.0)
                 .values
        )
    else:
        elev = np.zeros(len(coords))
    coords["elevation_m"] = elev

    return coords


def aggregate(
    trips: pd.DataFrame,
    us_holidays: set,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Aggregate trips to hourly OD counts.

    Parameters
    ----------
    trips       : normalised trip DataFrame from loader.load_files
    us_holidays : set of datetime.date objects treated as holidays

    Returns
    -------
    obs          : hourly OD observation DataFrame
    stations     : station index table (with elevation_m)
    dist_matrix  : (I, I) float64 haversine km
    elev_matrix  : (I, I) float64 elevation gain = dest_elev − orig_elev [m]
    """
    stations   = build_station_table(trips)
    id_to_idx  = {sid: i for i, sid in enumerate(stations["station_id"])}

    trips = trips.copy()
    trips["orig_idx"] = trips["start_station_id"].map(id_to_idx)
    trips["dest_idx"] = trips["end_station_id"].map(id_to_idx)
    trips = trips.dropna(subset=["orig_idx", "dest_idx"])
    trips["orig_idx"] = trips["orig_idx"].astype(int)
    trips["dest_idx"] = trips["dest_idx"].astype(int)

    trips["hour_bin"]   = trips["started_at"].dt.floor("h")
    trips["hour"]       = trips["started_at"].dt.hour.astype(np.int8)
    trips["dow"]        = trips["started_at"].dt.dayofweek.astype(np.int8)
    trips["month"]      = (trips["started_at"].dt.month - 1).astype(np.int8)
    trips["is_holiday"] = trips["started_at"].dt.date.isin(us_holidays).astype(np.int8)

    obs = (
        trips.groupby(
            ["orig_idx", "dest_idx", "hour_bin", "hour", "dow", "month", "is_holiday"],
            observed=True,
        )
        .size()
        .reset_index(name="count")
    )

    lats = stations["lat"].to_numpy()
    lons = stations["lng"].to_numpy()
    dist_matrix = haversine_matrix(lats, lons)
    obs["dist_km"] = dist_matrix[obs["orig_idx"].to_numpy(), obs["dest_idx"].to_numpy()]

    # Elevation gain matrix: elev_matrix[i,j] = elev[j] - elev[i]
    elev = stations["elevation_m"].to_numpy(dtype=np.float64)
    elev_matrix = elev[np.newaxis, :] - elev[:, np.newaxis]   # (I, I)

    return obs.reset_index(drop=True), stations, dist_matrix, elev_matrix
