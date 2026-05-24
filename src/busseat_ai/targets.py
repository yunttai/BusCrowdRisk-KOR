from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from busseat_ai.storage.database import rows_to_dicts, utc_now_iso

MAX_NEXT_STATION_MINUTES = 30


@dataclass(frozen=True)
class TargetBuildSummary:
    processed: int
    has_future_5min: int
    has_future_10min: int
    has_next_station: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "hasFuture5Min": self.has_future_5min,
            "hasFuture10Min": self.has_future_10min,
            "hasNextStation": self.has_next_station,
        }


def build_target_labels(conn: sqlite3.Connection) -> TargetBuildSummary:
    rows = rows_to_dicts(
        conn.execute(
            """
            SELECT snapshot_id, route_id, canonical_route_name, veh_id, plate_no,
                   station_seq, collected_at_kst, remain_seat_cnt, is_no_seat,
                   is_low_seat_2, is_low_seat_5
            FROM model_hourly_features
            ORDER BY route_id, veh_id, plate_no, collected_at_kst, station_seq
            """
        ).fetchall()
    )
    grouped = _group_vehicle_rows(rows)
    by_snapshot = {}
    now = utc_now_iso()

    for group_rows in grouped.values():
        for index, row in enumerate(group_rows):
            future_5 = _future_within(group_rows, index, minutes=5)
            future_10 = _future_within(group_rows, index, minutes=10)
            next_station = _next_station(group_rows, index)
            by_snapshot[row["snapshot_id"]] = {
                "snapshot_id": row["snapshot_id"],
                "route_id": row["route_id"],
                "canonical_route_name": row["canonical_route_name"],
                "veh_id": row["veh_id"],
                "plate_no": row["plate_no"],
                "station_seq": row["station_seq"],
                "collected_at_kst": row["collected_at_kst"],
                "target_no_seat_now": row["is_no_seat"],
                "target_low_seat_2_now": row["is_low_seat_2"],
                "target_low_seat_5_now": row["is_low_seat_5"],
                "target_no_seat_next_5min": _target_value(future_5, "is_no_seat"),
                "target_low_seat_2_next_5min": _target_value(future_5, "is_low_seat_2"),
                "target_no_seat_next_10min": _target_value(future_10, "is_no_seat"),
                "target_low_seat_2_next_10min": _target_value(future_10, "is_low_seat_2"),
                "target_no_seat_next_station": _target_value(next_station, "is_no_seat"),
                "target_low_seat_2_next_station": _target_value(next_station, "is_low_seat_2"),
                "has_future_5min": 1 if future_5 else 0,
                "has_future_10min": 1 if future_10 else 0,
                "has_next_station": 1 if next_station else 0,
                "created_at": now,
            }

    label_rows = list(by_snapshot.values())
    conn.executemany(
        """
        INSERT INTO model_target_labels (
            snapshot_id, route_id, canonical_route_name, veh_id, plate_no, station_seq,
            collected_at_kst, target_no_seat_now, target_low_seat_2_now,
            target_low_seat_5_now, target_no_seat_next_5min,
            target_low_seat_2_next_5min, target_no_seat_next_10min,
            target_low_seat_2_next_10min, target_no_seat_next_station,
            target_low_seat_2_next_station, has_future_5min, has_future_10min,
            has_next_station, created_at
        ) VALUES (
            :snapshot_id, :route_id, :canonical_route_name, :veh_id, :plate_no, :station_seq,
            :collected_at_kst, :target_no_seat_now, :target_low_seat_2_now,
            :target_low_seat_5_now, :target_no_seat_next_5min,
            :target_low_seat_2_next_5min, :target_no_seat_next_10min,
            :target_low_seat_2_next_10min, :target_no_seat_next_station,
            :target_low_seat_2_next_station, :has_future_5min, :has_future_10min,
            :has_next_station, :created_at
        )
        ON CONFLICT(snapshot_id) DO UPDATE SET
            route_id = excluded.route_id,
            canonical_route_name = excluded.canonical_route_name,
            veh_id = excluded.veh_id,
            plate_no = excluded.plate_no,
            station_seq = excluded.station_seq,
            collected_at_kst = excluded.collected_at_kst,
            target_no_seat_now = excluded.target_no_seat_now,
            target_low_seat_2_now = excluded.target_low_seat_2_now,
            target_low_seat_5_now = excluded.target_low_seat_5_now,
            target_no_seat_next_5min = excluded.target_no_seat_next_5min,
            target_low_seat_2_next_5min = excluded.target_low_seat_2_next_5min,
            target_no_seat_next_10min = excluded.target_no_seat_next_10min,
            target_low_seat_2_next_10min = excluded.target_low_seat_2_next_10min,
            target_no_seat_next_station = excluded.target_no_seat_next_station,
            target_low_seat_2_next_station = excluded.target_low_seat_2_next_station,
            has_future_5min = excluded.has_future_5min,
            has_future_10min = excluded.has_future_10min,
            has_next_station = excluded.has_next_station,
            created_at = excluded.created_at
        """,
        label_rows,
    )
    conn.commit()
    return TargetBuildSummary(
        processed=len(label_rows),
        has_future_5min=sum(row["has_future_5min"] for row in label_rows),
        has_future_10min=sum(row["has_future_10min"] for row in label_rows),
        has_next_station=sum(row["has_next_station"] for row in label_rows),
    )


def _group_vehicle_rows(rows: list[dict[str, Any]]) -> dict[tuple[Any, Any, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        row["_dt"] = datetime.fromisoformat(row["collected_at_kst"])
        grouped.setdefault((row["route_id"], row["veh_id"], row["plate_no"]), []).append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: (row["_dt"], row["station_seq"]))
    return grouped


def _future_within(rows: list[dict[str, Any]], index: int, minutes: int) -> dict[str, Any] | None:
    current = rows[index]
    deadline = current["_dt"] + timedelta(minutes=minutes)
    candidates = [row for row in rows[index + 1 :] if current["_dt"] < row["_dt"] <= deadline]
    if not candidates:
        return None
    return candidates[-1]


def _next_station(rows: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    current = rows[index]
    deadline = current["_dt"] + timedelta(minutes=MAX_NEXT_STATION_MINUTES)
    for row in rows[index + 1 :]:
        if row["_dt"] > deadline:
            return None
        if row["_dt"] >= current["_dt"] and row["station_seq"] > current["station_seq"]:
            return row
    return None


def _target_value(row: dict[str, Any] | None, key: str) -> int:
    if row is None:
        return 0
    return int(row[key])
