from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable

from busseat_ai.services.risk import CROWDED_LABELS, calculate_seat_scarcity_score, estimate_capacity
from busseat_ai.storage.database import rows_to_dicts, utc_now_iso


TARGET_ROUTE_NAMES = (
    "5000",
    "5001A",
    "5001B",
    "5001-1A",
    "5001-1B",
    "5003A",
    "5003B",
    "5005",
)

TARGET_ROUTE_QUERIES = ("5000", "5001", "5001-1", "5003", "5005")
KST = timezone(timedelta(hours=9))


@dataclass(frozen=True)
class PreprocessSummary:
    processed: int
    exported_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"processed": self.processed, "exportedPath": self.exported_path}


def normalize_route_name(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.upper().replace("번", "")
    text = re.sub(r"\s+", "", text)
    text = text.replace("(", "").replace(")", "")
    text = text.replace("[", "").replace("]", "")
    text = text.replace("－", "-").replace("–", "-").replace("—", "-")
    return text


def is_target_route(route_item: dict[str, Any], region_keyword: str | None = None) -> bool:
    canonical = normalize_route_name(route_item.get("routeName"))
    if canonical not in TARGET_ROUTE_NAMES:
        return False
    if not region_keyword:
        return True

    region_text = " ".join(
        str(route_item.get(key) or "")
        for key in ("regionName", "districtCd", "startStationName", "endStationName", "adminName")
    )
    return not region_text.strip() or region_keyword in region_text


def upsert_target_routes(conn: sqlite3.Connection, routes: Iterable[dict[str, Any]]) -> int:
    now = utc_now_iso()
    rows = []
    for route in routes:
        route_id = _to_int(route.get("routeId"))
        if route_id is None:
            continue
        rows.append(
            (
                route_id,
                normalize_route_name(route.get("routeName")),
                route.get("routeName"),
                _to_int(route.get("routeTypeCd")),
                route.get("routeTypeName"),
                route.get("regionName"),
                route.get("startStationName"),
                route.get("endStationName"),
                json.dumps(route, ensure_ascii=False),
                now,
            )
        )
    conn.executemany(
        """
        INSERT INTO target_route (
            route_id, canonical_route_name, route_name, route_type_cd, route_type_name,
            region_name, start_station_name, end_station_name, raw_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(route_id) DO UPDATE SET
            canonical_route_name = excluded.canonical_route_name,
            route_name = excluded.route_name,
            route_type_cd = excluded.route_type_cd,
            route_type_name = excluded.route_type_name,
            region_name = excluded.region_name,
            start_station_name = excluded.start_station_name,
            end_station_name = excluded.end_station_name,
            raw_json = excluded.raw_json,
            updated_at = excluded.updated_at
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def upsert_route_stations(
    conn: sqlite3.Connection,
    route_id: int | str,
    canonical_route_name: str,
    stations: Iterable[dict[str, Any]],
) -> int:
    now = utc_now_iso()
    rows = []
    for station in stations:
        station_seq = _to_int(station.get("stationSeq") or station.get("staOrder"))
        if station_seq is None:
            continue
        rows.append(
            (
                _to_int(route_id),
                station_seq,
                _to_int(station.get("stationId")),
                canonical_route_name,
                station.get("stationName"),
                station.get("mobileNo"),
                _to_float(station.get("x")),
                _to_float(station.get("y")),
                station.get("regionName"),
                station.get("centerYn"),
                station.get("turnYn"),
                json.dumps(station, ensure_ascii=False),
                now,
            )
        )
    conn.executemany(
        """
        INSERT INTO route_station (
            route_id, station_seq, station_id, canonical_route_name, station_name,
            mobile_no, x, y, region_name, center_yn, turn_yn, raw_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(route_id, station_seq) DO UPDATE SET
            station_id = excluded.station_id,
            canonical_route_name = excluded.canonical_route_name,
            station_name = excluded.station_name,
            mobile_no = excluded.mobile_no,
            x = excluded.x,
            y = excluded.y,
            region_name = excluded.region_name,
            center_yn = excluded.center_yn,
            turn_yn = excluded.turn_yn,
            raw_json = excluded.raw_json,
            updated_at = excluded.updated_at
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def get_target_routes(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT route_id, canonical_route_name, route_name, route_type_cd, route_type_name,
               region_name, start_station_name, end_station_name
        FROM target_route
        ORDER BY canonical_route_name, route_id
        """
    ).fetchall()
    return rows_to_dicts(rows)


def count_route_stations(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT tr.route_id, tr.canonical_route_name, tr.route_name, COUNT(rs.station_seq) AS station_count
        FROM target_route tr
        LEFT JOIN route_station rs ON rs.route_id = tr.route_id
        GROUP BY tr.route_id, tr.canonical_route_name, tr.route_name
        ORDER BY tr.canonical_route_name, tr.route_id
        """
    ).fetchall()
    return rows_to_dicts(rows)


def materialize_location_features(conn: sqlite3.Connection, export_csv_path: str | Path | None = None) -> PreprocessSummary:
    capacity_by_vehicle = _observed_capacities(conn)
    rows = conn.execute(
        """
        SELECT
            l.id AS snapshot_id,
            l.collected_at,
            l.route_id,
            COALESCE(tr.route_name, tr.canonical_route_name) AS route_name,
            tr.canonical_route_name,
            COALESCE(l.route_type_cd, tr.route_type_cd) AS route_type_cd,
            tr.route_type_name,
            l.veh_id,
            l.plate_no,
            l.station_id,
            l.station_seq,
            rs.station_name,
            rs.mobile_no,
            rs.x,
            rs.y,
            l.remain_seat_cnt,
            l.crowded,
            l.state_cd,
            l.low_plate,
            l.tagless_cd,
            l.raw_json
        FROM bus_location_snapshot l
        JOIN target_route tr ON tr.route_id = l.route_id
        LEFT JOIN route_station rs
          ON rs.route_id = l.route_id
         AND rs.station_seq = l.station_seq
        ORDER BY l.collected_at, l.route_id, l.station_seq, l.veh_id
        """
    ).fetchall()

    feature_rows = [_build_location_feature(dict(row), capacity_by_vehicle) for row in rows]
    conn.executemany(
        """
        INSERT INTO preprocessed_location_features (
            snapshot_id, collected_at_utc, collected_at_kst, service_date, day_of_week,
            day_name_ko, is_weekend, hour, minute, time_bucket_10m, time_period,
            route_id, route_name, canonical_route_name, route_type_cd, route_type_name,
            veh_id, plate_no, station_id, station_seq, station_name, mobile_no, x, y,
            remain_seat_cnt, crowded, crowded_label, state_cd, low_plate, tagless_cd,
            estimated_capacity, seat_scarcity_score, is_no_seat, is_low_seat_2,
            is_low_seat_5, raw_json, created_at
        ) VALUES (
            :snapshot_id, :collected_at_utc, :collected_at_kst, :service_date, :day_of_week,
            :day_name_ko, :is_weekend, :hour, :minute, :time_bucket_10m, :time_period,
            :route_id, :route_name, :canonical_route_name, :route_type_cd, :route_type_name,
            :veh_id, :plate_no, :station_id, :station_seq, :station_name, :mobile_no, :x, :y,
            :remain_seat_cnt, :crowded, :crowded_label, :state_cd, :low_plate, :tagless_cd,
            :estimated_capacity, :seat_scarcity_score, :is_no_seat, :is_low_seat_2,
            :is_low_seat_5, :raw_json, :created_at
        )
        ON CONFLICT(snapshot_id) DO UPDATE SET
            collected_at_utc = excluded.collected_at_utc,
            collected_at_kst = excluded.collected_at_kst,
            service_date = excluded.service_date,
            day_of_week = excluded.day_of_week,
            day_name_ko = excluded.day_name_ko,
            is_weekend = excluded.is_weekend,
            hour = excluded.hour,
            minute = excluded.minute,
            time_bucket_10m = excluded.time_bucket_10m,
            time_period = excluded.time_period,
            route_id = excluded.route_id,
            route_name = excluded.route_name,
            canonical_route_name = excluded.canonical_route_name,
            route_type_cd = excluded.route_type_cd,
            route_type_name = excluded.route_type_name,
            veh_id = excluded.veh_id,
            plate_no = excluded.plate_no,
            station_id = excluded.station_id,
            station_seq = excluded.station_seq,
            station_name = excluded.station_name,
            mobile_no = excluded.mobile_no,
            x = excluded.x,
            y = excluded.y,
            remain_seat_cnt = excluded.remain_seat_cnt,
            crowded = excluded.crowded,
            crowded_label = excluded.crowded_label,
            state_cd = excluded.state_cd,
            low_plate = excluded.low_plate,
            tagless_cd = excluded.tagless_cd,
            estimated_capacity = excluded.estimated_capacity,
            seat_scarcity_score = excluded.seat_scarcity_score,
            is_no_seat = excluded.is_no_seat,
            is_low_seat_2 = excluded.is_low_seat_2,
            is_low_seat_5 = excluded.is_low_seat_5,
            raw_json = excluded.raw_json,
            created_at = excluded.created_at
        """,
        feature_rows,
    )
    conn.commit()

    exported_path = None
    if export_csv_path:
        exported_path = str(export_location_features_csv(conn, export_csv_path))

    return PreprocessSummary(processed=len(feature_rows), exported_path=exported_path)


def export_location_features_csv(conn: sqlite3.Connection, export_csv_path: str | Path) -> Path:
    path = Path(export_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows_to_dicts(
        conn.execute(
            """
            SELECT *
            FROM preprocessed_location_features
            ORDER BY collected_at_kst, canonical_route_name, station_seq, veh_id
            """
        ).fetchall()
    )
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _build_location_feature(row: dict[str, Any], capacity_by_vehicle: dict[tuple[int | None, str | None], int]) -> dict[str, Any]:
    collected_at_utc = _parse_datetime(row["collected_at"])
    collected_at_kst = collected_at_utc.astimezone(KST)
    route_id = _to_int(row.get("route_id"))
    plate_no = row.get("plate_no")
    route_type_cd = _to_int(row.get("route_type_cd"))
    remain = _to_int(row.get("remain_seat_cnt"))
    observed_capacity = capacity_by_vehicle.get((route_id, plate_no))
    default_capacity = estimate_capacity(route_type_cd)
    capacity_candidates = [value for value in (observed_capacity, default_capacity) if value and value > 0]
    capacity = max(capacity_candidates) if capacity_candidates else None
    scarcity_score = calculate_seat_scarcity_score(remain, capacity) if remain is not None and remain >= 0 else None
    minute_bucket = (collected_at_kst.minute // 10) * 10

    return {
        "snapshot_id": row["snapshot_id"],
        "collected_at_utc": collected_at_utc.isoformat(timespec="seconds"),
        "collected_at_kst": collected_at_kst.isoformat(timespec="seconds"),
        "service_date": collected_at_kst.date().isoformat(),
        "day_of_week": collected_at_kst.weekday(),
        "day_name_ko": _day_name_ko(collected_at_kst.weekday()),
        "is_weekend": 1 if collected_at_kst.weekday() >= 5 else 0,
        "hour": collected_at_kst.hour,
        "minute": collected_at_kst.minute,
        "time_bucket_10m": f"{collected_at_kst.hour:02d}:{minute_bucket:02d}",
        "time_period": _time_period(collected_at_kst.hour),
        "route_id": route_id,
        "route_name": row.get("route_name"),
        "canonical_route_name": row.get("canonical_route_name"),
        "route_type_cd": route_type_cd,
        "route_type_name": row.get("route_type_name"),
        "veh_id": _to_int(row.get("veh_id")),
        "plate_no": plate_no,
        "station_id": _to_int(row.get("station_id")),
        "station_seq": _to_int(row.get("station_seq")),
        "station_name": row.get("station_name"),
        "mobile_no": row.get("mobile_no"),
        "x": _to_float(row.get("x")),
        "y": _to_float(row.get("y")),
        "remain_seat_cnt": remain,
        "crowded": _to_int(row.get("crowded")),
        "crowded_label": CROWDED_LABELS.get(_to_int(row.get("crowded"))),
        "state_cd": _to_int(row.get("state_cd")),
        "low_plate": _to_int(row.get("low_plate")),
        "tagless_cd": _to_int(row.get("tagless_cd")),
        "estimated_capacity": capacity,
        "seat_scarcity_score": scarcity_score,
        "is_no_seat": 1 if remain == 0 else 0,
        "is_low_seat_2": 1 if remain is not None and 0 <= remain <= 2 else 0,
        "is_low_seat_5": 1 if remain is not None and 0 <= remain <= 5 else 0,
        "raw_json": row.get("raw_json") or "{}",
        "created_at": utc_now_iso(),
    }


def _observed_capacities(conn: sqlite3.Connection) -> dict[tuple[int | None, str | None], int]:
    rows = conn.execute(
        """
        SELECT route_id, plate_no, MAX(remain_seat_cnt) AS capacity
        FROM bus_location_snapshot
        WHERE remain_seat_cnt IS NOT NULL
          AND remain_seat_cnt >= 0
          AND plate_no IS NOT NULL
        GROUP BY route_id, plate_no
        """
    ).fetchall()
    return {(_to_int(row["route_id"]), row["plate_no"]): row["capacity"] for row in rows if row["capacity"] is not None}


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _day_name_ko(day_of_week: int) -> str:
    return ("월", "화", "수", "목", "금", "토", "일")[day_of_week]


def _time_period(hour: int) -> str:
    if 7 <= hour < 10:
        return "morning_peak"
    if 17 <= hour < 20:
        return "evening_peak"
    if 5 <= hour < 7:
        return "early_morning"
    if 10 <= hour < 17:
        return "daytime"
    if 20 <= hour < 24:
        return "night"
    return "late_night"


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
