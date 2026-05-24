from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from busseat_ai.storage.database import rows_to_dicts, utc_now_iso


DEFAULT_VEHICLES_PER_HOUR = 6.0
KST = timezone(timedelta(hours=9))

# Sums to 1.0. Conservative initial distribution until hourly demand data exists.
HOUR_DEMAND_RATIOS = {
    0: 0.010,
    1: 0.006,
    2: 0.004,
    3: 0.003,
    4: 0.004,
    5: 0.015,
    6: 0.045,
    7: 0.090,
    8: 0.105,
    9: 0.065,
    10: 0.040,
    11: 0.035,
    12: 0.040,
    13: 0.035,
    14: 0.035,
    15: 0.045,
    16: 0.060,
    17: 0.090,
    18: 0.105,
    19: 0.075,
    20: 0.045,
    21: 0.035,
    22: 0.025,
    23: 0.015,
}


@dataclass(frozen=True)
class ExpectedBoardingsSummary:
    processed: int
    stations: int
    routes: int
    exported_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "stations": self.stations,
            "routes": self.routes,
            "exportedPath": self.exported_path,
        }


def import_station_demand_csv(conn: sqlite3.Connection, path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return insert_station_demand_rows(conn, rows)


def insert_station_demand_rows(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    values = []
    for row in rows:
        service_date = _date_string(row.get("service_date") or row.get("일자") or row.get("use_date"))
        station_id = _to_int(row.get("station_id") or row.get("정류소아이디") or row.get("stationId"))
        if not service_date or station_id is None:
            continue
        values.append(
            (
                service_date,
                station_id,
                _str_or_none(row.get("mobile_no") or row.get("정류소ARS") or row.get("mobileNo")),
                _str_or_none(row.get("station_name") or row.get("정류소명") or row.get("stationName")),
                _str_or_none(row.get("city_name") or row.get("시군명") or row.get("cityName")),
                _to_int(row.get("boarding_total") or row.get("승차인원합계") or row.get("boardingTotal")) or 0,
                _to_int(row.get("first_boarding_total") or row.get("최초승차인원수") or row.get("firstBoardingTotal")) or 0,
                _to_int(row.get("transfer_total") or row.get("환승인원수") or row.get("transferTotal")) or 0,
                _to_int(row.get("alighting_total") or row.get("하차인원합계") or row.get("alightingTotal")) or 0,
            )
        )
    conn.executemany(
        """
        INSERT INTO station_demand_daily (
            service_date, station_id, mobile_no, station_name, city_name,
            boarding_total, first_boarding_total, transfer_total, alighting_total
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(service_date, station_id) DO UPDATE SET
            mobile_no = excluded.mobile_no,
            station_name = excluded.station_name,
            city_name = excluded.city_name,
            boarding_total = excluded.boarding_total,
            first_boarding_total = excluded.first_boarding_total,
            transfer_total = excluded.transfer_total,
            alighting_total = excluded.alighting_total
        """,
        values,
    )
    conn.commit()
    return len(values)


def build_expected_boardings(conn: sqlite3.Connection, export_csv_path: str | Path | None = None) -> ExpectedBoardingsSummary:
    demand = _station_daily_averages(conn)
    route_shares = _station_route_shares(conn)
    vehicles = _vehicles_per_hour_by_route(conn)
    now = utc_now_iso()
    rows = []
    for (station_id, route_id), route_share in route_shares.items():
        station_avg = demand.get(station_id)
        if station_avg is None:
            continue
        for day_of_week in range(7):
            for hour, hour_ratio in HOUR_DEMAND_RATIOS.items():
                vehicle_count = vehicles.get((route_id, day_of_week, hour), DEFAULT_VEHICLES_PER_HOUR)
                expected = station_avg * hour_ratio * route_share / max(vehicle_count, 1.0)
                rows.append(
                    (
                        station_id,
                        route_id,
                        day_of_week,
                        hour,
                        round(station_avg, 4),
                        hour_ratio,
                        round(route_share, 6),
                        round(vehicle_count, 4),
                        round(expected, 6),
                        "station_daily_avg*hour_ratio*route_share/vehicles_per_hour",
                        now,
                    )
                )
    conn.executemany(
        """
        INSERT INTO station_expected_boardings_hourly (
            station_id, route_id, day_of_week, hour, station_daily_boarding_avg,
            hour_demand_ratio, route_share, vehicles_per_hour, expected_boardings_at_stop,
            source, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(station_id, route_id, day_of_week, hour) DO UPDATE SET
            station_daily_boarding_avg = excluded.station_daily_boarding_avg,
            hour_demand_ratio = excluded.hour_demand_ratio,
            route_share = excluded.route_share,
            vehicles_per_hour = excluded.vehicles_per_hour,
            expected_boardings_at_stop = excluded.expected_boardings_at_stop,
            source = excluded.source,
            created_at = excluded.created_at
        """,
        rows,
    )
    conn.commit()
    exported_path = None
    if export_csv_path:
        exported_path = str(export_expected_boardings_csv(conn, export_csv_path))
    return ExpectedBoardingsSummary(
        processed=len(rows),
        stations=len(demand),
        routes=len({route_id for _, route_id in route_shares}),
        exported_path=exported_path,
    )


def export_expected_boardings_csv(conn: sqlite3.Connection, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = rows_to_dicts(
        conn.execute(
            """
            SELECT *
            FROM station_expected_boardings_hourly
            ORDER BY route_id, station_id, day_of_week, hour
            """
        ).fetchall()
    )
    _write_rows_csv(rows, output)
    return output


def lookup_expected_boardings(
    conn: sqlite3.Connection,
    station_id: int | str,
    route_id: int | str,
    when: datetime | None = None,
) -> float | None:
    observed = when or datetime.now(KST)
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=KST)
    observed = observed.astimezone(KST)
    row = conn.execute(
        """
        SELECT expected_boardings_at_stop
        FROM station_expected_boardings_hourly
        WHERE station_id = ?
          AND route_id = ?
          AND day_of_week = ?
          AND hour = ?
        """,
        (_to_int(station_id), _to_int(route_id), observed.weekday(), observed.hour),
    ).fetchone()
    if not row:
        return None
    return float(row["expected_boardings_at_stop"])


def _station_daily_averages(conn: sqlite3.Connection) -> dict[int, float]:
    rows = conn.execute(
        """
        SELECT station_id, AVG(boarding_total) AS avg_boarding
        FROM station_demand_daily
        GROUP BY station_id
        """
    ).fetchall()
    return {row["station_id"]: float(row["avg_boarding"] or 0) for row in rows}


def _station_route_shares(conn: sqlite3.Connection) -> dict[tuple[int, int], float]:
    rows = conn.execute(
        """
        SELECT station_id, route_id
        FROM route_station
        WHERE station_id IS NOT NULL
          AND route_id IS NOT NULL
        """
    ).fetchall()
    routes_by_station: dict[int, set[int]] = {}
    for row in rows:
        routes_by_station.setdefault(row["station_id"], set()).add(row["route_id"])
    shares: dict[tuple[int, int], float] = {}
    for station_id, routes in routes_by_station.items():
        share = 1.0 / len(routes) if routes else 0.0
        for route_id in routes:
            shares[(station_id, route_id)] = share
    return shares


def _vehicles_per_hour_by_route(conn: sqlite3.Connection) -> dict[tuple[int, int, int], float]:
    rows = conn.execute(
        """
        SELECT route_id, day_of_week, hour, COUNT(DISTINCT veh_id) AS vehicles
        FROM model_hourly_features
        GROUP BY route_id, day_of_week, hour
        """
    ).fetchall()
    return {
        (row["route_id"], row["day_of_week"], row["hour"]): max(float(row["vehicles"] or 0), DEFAULT_VEHICLES_PER_HOUR)
        for row in rows
    }


def _write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    try:
        file = path.open("w", encoding="utf-8-sig", newline="")
    except PermissionError:
        fallback = path.with_name(f"{path.stem}.new{path.suffix}")
        file = fallback.open("w", encoding="utf-8-sig", newline="")
    with file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _date_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text[:10]


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)
