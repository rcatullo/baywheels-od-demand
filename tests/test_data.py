"""Unit tests for data loading and preprocessing."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baywheels.data.aggregator import haversine_matrix, build_station_table
from baywheels.data.loader import _normalise, _is_old_schema
from baywheels.data.calendar import build_calendar


class TestLoader:
    def _old_row(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "duration_sec": 300,
            "start_time": "2019-05-01 08:00:00",
            "start_station_id": "101",
            "start_station_name": "A",
            "start_station_latitude": 37.7,
            "start_station_longitude": -122.4,
            "end_station_id": "202",
            "end_station_name": "B",
            "end_station_latitude": 37.8,
            "end_station_longitude": -122.3,
            "bike_id": 1,
            "user_type": "Subscriber",
        }])

    def _new_row(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ride_id": "ABC",
            "rideable_type": "classic_bike",
            "started_at": "2024-01-10 17:51:51",
            "ended_at": "2024-01-10 17:57:03",
            "start_station_name": "A",
            "start_station_id": "EM-D4",
            "end_station_name": "B",
            "end_station_id": "OK-D2",
            "start_lat": 37.83,
            "start_lng": -122.28,
            "end_lat": 37.84,
            "end_lng": -122.27,
            "member_casual": "member",
        }])

    def test_old_schema_detected(self):
        df = self._old_row()
        assert _is_old_schema(list(df.columns))

    def test_new_schema_detected(self):
        df = self._new_row()
        assert not _is_old_schema(list(df.columns))

    def test_old_normalise_columns(self):
        df = _normalise(self._old_row())
        assert "started_at" in df.columns
        assert "start_lat" in df.columns

    def test_new_normalise_columns(self):
        df = _normalise(self._new_row())
        assert "started_at" in df.columns
        assert "start_lat" in df.columns

    def test_started_at_is_datetime(self):
        for df in (self._old_row(), self._new_row()):
            out = _normalise(df)
            assert pd.api.types.is_datetime64_any_dtype(out["started_at"])


class TestHaversine:
    def test_zero_distance(self):
        D = haversine_matrix(np.array([37.7]), np.array([-122.4]))
        assert D[0, 0] == pytest.approx(0.0)

    def test_known_distance(self):
        # SF City Hall to Oakland City Hall: approx 13 km
        lats = np.array([37.7793, 37.8047])
        lons = np.array([-122.4193, -122.2721])
        D = haversine_matrix(lats, lons)
        assert D[0, 1] == pytest.approx(13.0, abs=1.5)

    def test_symmetry(self):
        rng = np.random.default_rng(0)
        lats = rng.uniform(37.7, 37.9, 5)
        lons = rng.uniform(-122.5, -122.3, 5)
        D = haversine_matrix(lats, lons)
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        lats = rng.uniform(37.0, 38.0, 10)
        lons = rng.uniform(-123.0, -122.0, 10)
        D = haversine_matrix(lats, lons)
        assert (D >= 0).all()


class TestCalendar:
    def test_shape(self):
        holidays = set()
        cal = build_calendar("2023-01-01", "2024-01-01", holidays)
        assert len(cal) == 365 * 24

    def test_hour_range(self):
        cal = build_calendar("2023-01-01", "2023-02-01", set())
        assert cal["hour"].min() == 0
        assert cal["hour"].max() == 23

    def test_holiday_flag(self):
        import datetime
        hols = {datetime.date(2023, 1, 1)}  # New Year's Day
        cal = build_calendar("2023-01-01", "2023-01-02", hols)
        assert cal["is_holiday"].sum() == 24  # all 24 hours of Jan 1 flagged
