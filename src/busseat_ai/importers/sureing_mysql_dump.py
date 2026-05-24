from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from busseat_ai.preprocessing import TARGET_ROUTE_NAMES, normalize_route_name, upsert_route_stations, upsert_target_routes
from busseat_ai.storage.database import utc_now_iso

KST = timezone(timedelta(hours=9))


@dataclass(frozen=True)
class SureingImportSummary:
    routes_imported: int
    route_stations_imported: int
    snapshots_seen: int
    snapshots_imported: int
    snapshots_skipped_existing: int
    snapshots_skipped_non_target: int
    minus_one_as_zero: int
    target_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "routesImported": self.routes_imported,
            "routeStationsImported": self.route_stations_imported,
            "snapshotsSeen": self.snapshots_seen,
            "snapshotsImported": self.snapshots_imported,
            "snapshotsSkippedExisting": self.snapshots_skipped_existing,
            "snapshotsSkippedNonTarget": self.snapshots_skipped_non_target,
            "minusOneAsZero": self.minus_one_as_zero,
            "targetOnly": self.target_only,
        }


def import_sureing_mysql_dump(
    conn: sqlite3.Connection,
    path: str | Path,
    *,
    target_only: bool = True,
    minus_one_as_zero: bool = True,
    batch_size: int = 1000,
) -> SureingImportSummary:
    dump = _MySqlDump(Path(path))
    routes_by_pk = _load_routes(dump)
    stations_by_pk = _load_stations(dump)
    target_external_route_ids = _target_external_route_ids(routes_by_pk, target_only)

    route_count = _upsert_routes(conn, routes_by_pk.values(), target_external_route_ids)
    route_station_count = _upsert_route_stations(conn, dump, routes_by_pk, stations_by_pk, target_external_route_ids)
    imported, skipped_existing, skipped_non_target, minus_one_count, seen = _import_snapshots(
        conn,
        dump,
        routes_by_pk,
        target_external_route_ids,
        minus_one_as_zero=minus_one_as_zero,
        batch_size=batch_size,
    )
    return SureingImportSummary(
        routes_imported=route_count,
        route_stations_imported=route_station_count,
        snapshots_seen=seen,
        snapshots_imported=imported,
        snapshots_skipped_existing=skipped_existing,
        snapshots_skipped_non_target=skipped_non_target,
        minus_one_as_zero=minus_one_count,
        target_only=target_only,
    )


class _MySqlDump:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.text = path.read_text(encoding="utf-8", errors="replace")
        self.tables = self._load_table_columns()

    def rows(self, table_name: str) -> Iterable[dict[str, Any]]:
        columns = self.tables.get(table_name)
        if not columns:
            return
        for chunk in self._insert_chunks(table_name):
            for body in _iter_tuples(chunk):
                fields = [_clean_mysql_value(value) for value in _split_fields(body)]
                if len(fields) != len(columns):
                    continue
                yield dict(zip(columns, fields))

    def _load_table_columns(self) -> dict[str, list[str]]:
        create_pattern = re.compile(r"CREATE TABLE `([^`]+)` \((.*?)\) ENGINE=", re.S)
        tables: dict[str, list[str]] = {}
        for name, body in create_pattern.findall(self.text):
            columns = []
            for line in body.splitlines():
                line = line.strip()
                if line.startswith("`"):
                    columns.append(line.split("`", 2)[1])
            tables[name] = columns
        return tables

    def _insert_chunks(self, table_name: str) -> Iterable[str]:
        pattern = f"INSERT INTO `{table_name}` VALUES "
        index = 0
        while True:
            start = self.text.find(pattern, index)
            if start < 0:
                break
            values_start = start + len(pattern)
            end = self.text.find(";", values_start)
            if end < 0:
                break
            yield self.text[values_start:end]
            index = end + 1


def _load_routes(dump: _MySqlDump) -> dict[str, dict[str, Any]]:
    routes = {}
    for row in dump.rows("bus_route") or []:
        external_route_id = _str_or_none(row.get("external_route_id"))
        if not external_route_id:
            continue
        row["canonical_route_name"] = normalize_route_name(row.get("route_name"))
        routes[str(row["id"])] = row
    return routes


def _load_stations(dump: _MySqlDump) -> dict[str, dict[str, Any]]:
    stations = {}
    for row in dump.rows("station") or []:
        if row.get("id") is not None:
            stations[str(row["id"])] = row
    return stations


def _target_external_route_ids(routes: dict[str, dict[str, Any]], target_only: bool) -> set[str]:
    if not target_only:
        return {str(route["external_route_id"]) for route in routes.values() if route.get("external_route_id")}
    return {
        str(route["external_route_id"])
        for route in routes.values()
        if route.get("external_route_id") and route.get("canonical_route_name") in TARGET_ROUTE_NAMES
    }


def _upsert_routes(conn: sqlite3.Connection, routes: Iterable[dict[str, Any]], target_external_route_ids: set[str]) -> int:
    target_routes = []
    for route in routes:
        external_route_id = _to_int(route.get("external_route_id"))
        if external_route_id is None or str(external_route_id) not in target_external_route_ids:
            continue
        target_routes.append(
            {
                "routeId": external_route_id,
                "routeName": route.get("route_name"),
                "routeTypeCd": _route_type_code(route.get("route_type")),
                "routeTypeName": _route_type_name(route.get("route_type")),
                "regionName": "import:sureing_occupancy",
                "startStationName": route.get("start_station_name"),
                "endStationName": route.get("end_station_name"),
                "source": "sureing_occupancy_backup",
                "defaultCapacity": _to_int(route.get("default_capacity")),
                "destinationZone": route.get("destination_zone"),
            }
        )
    return upsert_target_routes(conn, target_routes)


def _upsert_route_stations(
    conn: sqlite3.Connection,
    dump: _MySqlDump,
    routes_by_pk: dict[str, dict[str, Any]],
    stations_by_pk: dict[str, dict[str, Any]],
    target_external_route_ids: set[str],
) -> int:
    by_route: dict[int, list[dict[str, Any]]] = {}
    route_names: dict[int, str] = {}
    for row in dump.rows("route_station") or []:
        route = routes_by_pk.get(str(row.get("route_id")))
        if not route:
            continue
        external_route_id = _to_int(route.get("external_route_id"))
        if external_route_id is None or str(external_route_id) not in target_external_route_ids:
            continue
        station = stations_by_pk.get(str(row.get("station_id")), {})
        by_route.setdefault(external_route_id, []).append(
            {
                "stationSeq": _to_int(row.get("station_sequence")),
                "stationId": _to_int(station.get("external_station_id")),
                "stationName": station.get("station_name"),
                "mobileNo": station.get("station_number"),
                "x": _to_float(station.get("longitude")),
                "y": _to_float(station.get("latitude")),
                "regionName": station.get("city_name"),
                "centerYn": None,
                "turnYn": None,
                "direction": row.get("direction"),
                "towardName": row.get("toward_name"),
                "source": "sureing_occupancy_backup",
            }
        )
        route_names[external_route_id] = route.get("canonical_route_name")

    count = 0
    for route_id, stations in by_route.items():
        count += upsert_route_stations(conn, route_id, route_names.get(route_id, ""), stations)
    return count


def _import_snapshots(
    conn: sqlite3.Connection,
    dump: _MySqlDump,
    routes_by_pk: dict[str, dict[str, Any]],
    target_external_route_ids: set[str],
    *,
    minus_one_as_zero: bool,
    batch_size: int,
) -> tuple[int, int, int, int, int]:
    existing_keys = _load_existing_snapshot_keys(conn)
    rows = []
    imported = 0
    skipped_existing = 0
    skipped_non_target = 0
    minus_one_count = 0
    seen = 0
    now = utc_now_iso()

    for row in dump.rows("bus_snapshot") or []:
        seen += 1
        external_route_id = _to_int(row.get("external_route_id"))
        if external_route_id is None or str(external_route_id) not in target_external_route_ids:
            skipped_non_target += 1
            continue

        collected_at = _mysql_kst_datetime_to_utc_iso(row.get("collected_at"))
        veh_id = _to_int(row.get("external_vehicle_id"))
        station_id = _to_int(row.get("external_station_id"))
        station_seq = _to_int(row.get("station_sequence"))
        plate_no = _str_or_none(row.get("plate_no"))
        key = (collected_at, external_route_id, veh_id, station_id, station_seq, plate_no)
        if key in existing_keys:
            skipped_existing += 1
            continue
        existing_keys.add(key)

        original_remain = _to_int(row.get("remaining_seat_count"))
        remain = original_remain
        normalized_from_minus_one = False
        if remain == -1 and minus_one_as_zero:
            remain = 0
            normalized_from_minus_one = True
            minus_one_count += 1

        route = _route_for_external_id(routes_by_pk, external_route_id)
        raw = dict(row)
        raw["source"] = "sureing_occupancy_backup"
        raw["original_remaining_seat_count"] = original_remain
        raw["normalized_remaining_seat_count"] = remain
        raw["normalized_from_minus_one"] = normalized_from_minus_one

        rows.append(
            (
                collected_at,
                external_route_id,
                veh_id,
                plate_no,
                station_id,
                station_seq,
                remain,
                _to_int(row.get("crowded_code")),
                _route_type_code(route.get("route_type") if route else None),
                _to_int(row.get("low_plate")),
                _to_int(row.get("state_code")),
                _to_int(row.get("tagless_code")),
                json.dumps(raw, ensure_ascii=False),
            )
        )

        if len(rows) >= batch_size:
            imported += _insert_snapshot_rows(conn, rows)
            rows.clear()

    if rows:
        imported += _insert_snapshot_rows(conn, rows)
    conn.commit()
    return imported, skipped_existing, skipped_non_target, minus_one_count, seen


def _insert_snapshot_rows(conn: sqlite3.Connection, rows: list[tuple[Any, ...]]) -> int:
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
    return len(rows)


def _load_existing_snapshot_keys(conn: sqlite3.Connection) -> set[tuple[Any, ...]]:
    rows = conn.execute(
        """
        SELECT collected_at, route_id, veh_id, station_id, station_seq, plate_no
        FROM bus_location_snapshot
        """
    ).fetchall()
    return {
        (row["collected_at"], row["route_id"], row["veh_id"], row["station_id"], row["station_seq"], row["plate_no"])
        for row in rows
    }


def _route_for_external_id(routes_by_pk: dict[str, dict[str, Any]], external_route_id: int) -> dict[str, Any] | None:
    external = str(external_route_id)
    for route in routes_by_pk.values():
        if str(route.get("external_route_id")) == external:
            return route
    return None


def _iter_tuples(chunk: str) -> Iterable[str]:
    in_quote = False
    escape = False
    depth = 0
    buffer: list[str] = []
    for char in chunk:
        if in_quote:
            buffer.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "'":
                in_quote = False
            continue
        if char == "'":
            in_quote = True
            if depth > 0:
                buffer.append(char)
        elif char == "(":
            if depth > 0:
                buffer.append(char)
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                yield "".join(buffer)
                buffer = []
            else:
                buffer.append(char)
        else:
            if depth > 0:
                buffer.append(char)


def _split_fields(tuple_body: str) -> list[str]:
    fields = []
    in_quote = False
    escape = False
    buffer: list[str] = []
    for char in tuple_body:
        if in_quote:
            buffer.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "'":
                in_quote = False
            continue
        if char == "'":
            in_quote = True
            buffer.append(char)
        elif char == ",":
            fields.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
    fields.append("".join(buffer).strip())
    return fields


def _clean_mysql_value(value: str) -> str | None:
    value = value.strip()
    if value.upper() == "NULL":
        return None
    if value.startswith("_binary "):
        value = value[len("_binary ") :].strip()
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1].replace("\\'", "'").replace("\\\\", "\\")
    return value


def _mysql_kst_datetime_to_utc_iso(value: Any) -> str:
    if value is None or value == "":
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    text = str(value)
    try:
        parsed = datetime.strptime(text[:26], "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        parsed = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
    return parsed.replace(tzinfo=KST).astimezone(timezone.utc).isoformat(timespec="seconds")


def _route_type_code(route_type: Any) -> int | None:
    if route_type in {"EXPRESS_SEAT", "METROPOLITAN"}:
        return 11
    return None


def _route_type_name(route_type: Any) -> str:
    if route_type == "METROPOLITAN":
        return "광역급행형시내버스"
    if route_type == "EXPRESS_SEAT":
        return "직행좌석형시내버스"
    return "unknown"


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)
