from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from busseat_ai.storage.database import rows_to_dicts


@dataclass(frozen=True)
class DataQualityReport:
    total_rows: int
    date_range: dict[str, str | None]
    null_columns: list[dict[str, Any]]
    imputed_rates: dict[str, float]
    route_rows: list[dict[str, Any]]
    time_period_rows: list[dict[str, Any]]
    recommended_features: list[str]
    excluded_features: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "totalRows": self.total_rows,
            "dateRange": self.date_range,
            "nullColumns": self.null_columns,
            "imputedRates": self.imputed_rates,
            "routeRows": self.route_rows,
            "timePeriodRows": self.time_period_rows,
            "recommendedFeatures": self.recommended_features,
            "excludedFeatures": self.excluded_features,
        }


IMPUTED_THRESHOLDS = {
    "weather": 0.30,
    "airQuality": 0.50,
    "traffic": 0.50,
    "event": 0.80,
}

FEATURE_GROUPS = {
    "weather": ["temperature", "precipitation", "humidity", "wind_speed", "cloud_amount", "weather_text"],
    "airQuality": ["pm10", "pm25", "o3", "khai", "air_quality_grade"],
    "traffic": ["avg_speed", "traffic_volume", "delay_time", "congestion_level"],
    "event": ["event_count", "event_nearby_count"],
}


def build_data_quality_report(conn: sqlite3.Connection) -> DataQualityReport:
    total_rows = _count(conn, "model_hourly_features")
    date_row = conn.execute(
        "SELECT MIN(service_date) AS start_date, MAX(service_date) AS end_date FROM model_hourly_features"
    ).fetchone()
    date_range = {
        "startDate": date_row["start_date"] if date_row else None,
        "endDate": date_row["end_date"] if date_row else None,
    }
    null_columns = _null_columns(conn, "model_hourly_features")
    imputed_rates = _imputed_rates(conn, total_rows)
    route_rows = rows_to_dicts(
        conn.execute(
            """
            SELECT canonical_route_name, COUNT(*) AS row_count
            FROM model_hourly_features
            GROUP BY canonical_route_name
            ORDER BY canonical_route_name
            """
        ).fetchall()
    )
    time_period_rows = rows_to_dicts(
        conn.execute(
            """
            SELECT time_period, COUNT(*) AS row_count
            FROM model_hourly_features
            GROUP BY time_period
            ORDER BY row_count DESC
            """
        ).fetchall()
    )
    recommended, excluded = _feature_recommendations(imputed_rates)
    return DataQualityReport(
        total_rows=total_rows,
        date_range=date_range,
        null_columns=null_columns,
        imputed_rates=imputed_rates,
        route_rows=route_rows,
        time_period_rows=time_period_rows,
        recommended_features=recommended,
        excluded_features=excluded,
    )


def _count(conn: sqlite3.Connection, table_name: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


def _null_columns(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]
    nulls = []
    for column in columns:
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {column} IS NULL").fetchone()[0]
        if count:
            nulls.append({"column": column, "nullCount": count})
    return nulls


def _imputed_rates(conn: sqlite3.Connection, total_rows: int) -> dict[str, float]:
    if total_rows <= 0:
        return {"weather": 1.0, "airQuality": 1.0, "traffic": 1.0, "event": 1.0}
    row = conn.execute(
        """
        SELECT
            SUM(weather_imputed) AS weather,
            SUM(air_quality_imputed) AS air_quality,
            SUM(traffic_imputed) AS traffic,
            SUM(event_imputed) AS event
        FROM model_hourly_features
        """
    ).fetchone()
    return {
        "weather": round((row["weather"] or 0) / total_rows, 4),
        "airQuality": round((row["air_quality"] or 0) / total_rows, 4),
        "traffic": round((row["traffic"] or 0) / total_rows, 4),
        "event": round((row["event"] or 0) / total_rows, 4),
    }


def _feature_recommendations(imputed_rates: dict[str, float]) -> tuple[list[str], list[str]]:
    recommended: list[str] = []
    excluded: list[str] = []
    for group, features in FEATURE_GROUPS.items():
        threshold = IMPUTED_THRESHOLDS[group]
        target = recommended if imputed_rates.get(group, 1.0) <= threshold else excluded
        target.extend(features)
    return recommended, excluded
