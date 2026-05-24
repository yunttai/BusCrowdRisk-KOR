from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def init_db(db_path: str | Path) -> None:
    conn = connect(db_path)
    try:
        schema_path = Path(__file__).with_name("schema.sql")
        conn.executescript(schema_path.read_text(encoding="utf-8"))
        conn.commit()
    finally:
        conn.close()


def insert_arrival_snapshots(conn: sqlite3.Connection, items: Iterable[dict[str, Any]], collected_at: str | None = None) -> int:
    collected_at = collected_at or utc_now_iso()
    rows = [
        (
            collected_at,
            _to_int(item.get("stationId")),
            _to_int(item.get("routeId")),
            item.get("routeName"),
            _to_int(item.get("routeTypeCd")),
            _to_int(item.get("staOrder")),
            _to_int(item.get("predictTime1")),
            _to_int(item.get("predictTime2")),
            _to_int(item.get("predictTimeSec1")),
            _to_int(item.get("predictTimeSec2")),
            _to_int(item.get("remainSeatCnt1")),
            _to_int(item.get("remainSeatCnt2")),
            _to_int(item.get("crowded1")),
            _to_int(item.get("crowded2")),
            item.get("plateNo1"),
            item.get("plateNo2"),
            _to_int(item.get("vehId1")),
            _to_int(item.get("vehId2")),
            json.dumps(item, ensure_ascii=False),
        )
        for item in items
    ]
    conn.executemany(
        """
        INSERT INTO bus_arrival_snapshot (
            collected_at, station_id, route_id, route_name, route_type_cd, sta_order,
            predict_time1, predict_time2, predict_time_sec1, predict_time_sec2,
            remain_seat_cnt1, remain_seat_cnt2, crowded1, crowded2,
            plate_no1, plate_no2, veh_id1, veh_id2, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def insert_location_snapshots(conn: sqlite3.Connection, items: Iterable[dict[str, Any]], collected_at: str | None = None) -> int:
    collected_at = collected_at or utc_now_iso()
    rows = [
        (
            collected_at,
            _to_int(item.get("routeId")),
            _to_int(item.get("vehId")),
            item.get("plateNo"),
            _to_int(item.get("stationId")),
            _to_int(item.get("stationSeq")),
            _to_int(item.get("remainSeatCnt")),
            _to_int(item.get("crowded")),
            _to_int(item.get("routeTypeCd")),
            _to_int(item.get("lowPlate")),
            _to_int(item.get("stateCd")),
            _to_int(item.get("taglessCd")),
            json.dumps(item, ensure_ascii=False),
        )
        for item in items
    ]
    conn.executemany(
        """
        INSERT INTO bus_location_snapshot (
            collected_at, route_id, veh_id, plate_no, station_id, station_seq,
            remain_seat_cnt, crowded, route_type_cd, low_plate, state_cd,
            tagless_cd, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def recent_observed_capacity(conn: sqlite3.Connection, route_id: int | str, plate_no: str | None = None) -> int | None:
    params: list[Any] = [_to_int(route_id)]
    where = "route_id = ? AND remain_seat_cnt IS NOT NULL AND remain_seat_cnt >= 0"
    if plate_no:
        where += " AND plate_no = ?"
        params.append(plate_no)
    row = conn.execute(
        f"SELECT MAX(remain_seat_cnt) AS capacity FROM bus_location_snapshot WHERE {where}",
        params,
    ).fetchone()
    return row["capacity"] if row and row["capacity"] is not None else None


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
