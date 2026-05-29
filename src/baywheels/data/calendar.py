"""Build the full training-period calendar with optional area-wide weather.

The log-likelihood factorises as  M · S  where  S = Σ_t exp(η_t).  The
calendar enumerates every hour in the training window so that S and its
per-stratum sub-sums can be computed with a single np.bincount pass.

When an area-wide `weather` DataFrame is provided, its columns are joined to
the calendar so that temporal weather covariates enter η_t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_calendar(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    us_holidays: set,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with one row per hour in [start, end).

    Parameters
    ----------
    start, end  : range boundaries (end is exclusive)
    us_holidays : set of datetime.date objects
    weather     : optional DataFrame with column '_hour' (format
                  "YYYY-MM-DDTHH:00") and weather variable columns.
                  Joined left to the calendar by matching the hour string.

    Columns always present
    ----------------------
    ts          : hourly timestamp (timezone-naive, local time)
    hour        : int8  0-23
    dow         : int8  0-6  (0 = Monday)
    month       : int8  0-11
    is_holiday  : int8  0/1
    """
    hours = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    df = pd.DataFrame({"ts": hours})
    df["hour"]       = df["ts"].dt.hour.astype(np.int8)
    df["dow"]        = df["ts"].dt.dayofweek.astype(np.int8)
    df["month"]      = (df["ts"].dt.month - 1).astype(np.int8)
    df["is_holiday"] = df["ts"].dt.date.isin(us_holidays).astype(np.int8)

    if weather is not None:
        df["_hour"] = df["ts"].dt.strftime("%Y-%m-%dT%H:00")
        df = df.merge(weather, on="_hour", how="left")
        df.drop(columns=["_hour"], inplace=True)

    return df.reset_index(drop=True)
