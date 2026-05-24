from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable

from busseat_ai.preprocessing import materialize_location_features
from busseat_ai.storage.database import rows_to_dicts, utc_now_iso


KST = timezone(timedelta(hours=9))
MAX_NEAREST_TIME_DELTA = timedelta(hours=3)


@dataclass(frozen=True)
class ModelFeatureSummary:
    processed: int
    weather_imputed: int
    air_quality_imputed: int
    traffic_imputed: int
    event_imputed: int
    exported_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "weatherImputed": self.weather_imputed,
            "airQualityImputed": self.air_quality_imputed,
            "trafficImputed": self.traffic_imputed,
            "eventImputed": self.event_imputed,
            "exportedPath": self.exported_path,
        }


def infer_snapshot_date_range(conn: sqlite3.Connection) -> tuple[str | None, str | None]:
    row = conn.execute(
        """
        SELECT MIN(service_date) AS start_date, MAX(service_date) AS end_date
        FROM preprocessed_location_features
        """
    ).fetchone()
    if row and row["start_date"] and row["end_date"]:
        return row["start_date"], row["end_date"]

    materialize_location_features(conn)
    row = conn.execute(
        """
        SELECT MIN(service_date) AS start_date, MAX(service_date) AS end_date
        FROM preprocessed_location_features
        """
    ).fetchone()
    if not row:
        return None, None
    return row["start_date"], row["end_date"]


def insert_weather_hourly(conn: sqlite3.Connection, items: Iterable[dict[str, Any]], source: str = "KMA_ASOS") -> int:
    now = utc_now_iso()
    rows = []
    for item in items:
        observed = _parse_kma_time(item.get("tm"))
        if observed is None:
            continue
        temperature = _float_or_default(item.get("ta"), 0.0)
        precipitation = _float_or_default(item.get("rn"), 0.0)
        humidity = _float_or_default(item.get("hm"), 0.0)
        wind_speed = _float_or_default(item.get("ws"), 0.0)
        cloud_amount = _float_or_default(item.get("dc10Tca"), 0.0)
        weather_text = _weather_text(precipitation, cloud_amount)
        rows.append(
            (
                str(item.get("stnId") or "unknown"),
                str(item.get("stnNm") or "unknown"),
                observed.isoformat(timespec="seconds"),
                observed.date().isoformat(),
                observed.hour,
                temperature,
                precipitation,
                humidity,
                wind_speed,
                cloud_amount,
                weather_text,
                source,
                json.dumps(item, ensure_ascii=False),
                now,
            )
        )
    conn.executemany(
        """
        INSERT INTO external_weather_hourly (
            weather_station_id, weather_station_name, observed_at_kst, service_date, hour,
            temperature, precipitation, humidity, wind_speed, cloud_amount, weather_text,
            source, raw_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(weather_station_id, observed_at_kst) DO UPDATE SET
            weather_station_name = excluded.weather_station_name,
            service_date = excluded.service_date,
            hour = excluded.hour,
            temperature = excluded.temperature,
            precipitation = excluded.precipitation,
            humidity = excluded.humidity,
            wind_speed = excluded.wind_speed,
            cloud_amount = excluded.cloud_amount,
            weather_text = excluded.weather_text,
            source = excluded.source,
            raw_json = excluded.raw_json,
            created_at = excluded.created_at
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def insert_air_quality_hourly(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]], source: str = "external_import") -> int:
    now = utc_now_iso()
    values = []
    for row in rows:
        observed = _parse_any_kst(row.get("observed_at_kst") or row.get("dataTime") or row.get("tm"))
        if observed is None:
            continue
        pm10 = _float_or_default(row.get("pm10") or row.get("pm10Value"), 0.0)
        pm25 = _float_or_default(row.get("pm25") or row.get("pm25Value"), 0.0)
        o3 = _float_or_default(row.get("o3") or row.get("o3Value"), 0.0)
        khai = _float_or_default(row.get("khai") or row.get("khaiValue"), 0.0)
        values.append(
            (
                str(row.get("air_station_name") or row.get("stationName") or "unknown"),
                observed.isoformat(timespec="seconds"),
                observed.date().isoformat(),
                observed.hour,
                pm10,
                pm25,
                o3,
                khai,
                str(row.get("air_quality_grade") or row.get("khaiGrade") or "unknown"),
                source,
                json.dumps(row, ensure_ascii=False),
                now,
            )
        )
    conn.executemany(
        """
        INSERT INTO external_air_quality_hourly (
            air_station_name, observed_at_kst, service_date, hour, pm10, pm25, o3, khai,
            air_quality_grade, source, raw_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(air_station_name, observed_at_kst) DO UPDATE SET
            service_date = excluded.service_date,
            hour = excluded.hour,
            pm10 = excluded.pm10,
            pm25 = excluded.pm25,
            o3 = excluded.o3,
            khai = excluded.khai,
            air_quality_grade = excluded.air_quality_grade,
            source = excluded.source,
            raw_json = excluded.raw_json,
            created_at = excluded.created_at
        """,
        values,
    )
    conn.commit()
    return len(values)


def insert_traffic_hourly(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]], source: str = "external_import") -> int:
    now = utc_now_iso()
    values = []
    for row in rows:
        observed = _parse_any_kst(row.get("observed_at_kst") or row.get("tm"))
        if observed is None:
            continue
        values.append(
            (
                str(row.get("traffic_context_key") or row.get("roadSectionId") or "global"),
                observed.isoformat(timespec="seconds"),
                observed.date().isoformat(),
                observed.hour,
                _float_or_default(row.get("avg_speed") or row.get("speed"), 0.0),
                _float_or_default(row.get("traffic_volume") or row.get("volume"), 0.0),
                _float_or_default(row.get("delay_time") or row.get("delayTime"), 0.0),
                _int_or_default(row.get("congestion_level") or row.get("congestionLevel"), 0),
                source,
                json.dumps(row, ensure_ascii=False),
                now,
            )
        )
    conn.executemany(
        """
        INSERT INTO external_traffic_hourly (
            traffic_context_key, observed_at_kst, service_date, hour, avg_speed,
            traffic_volume, delay_time, congestion_level, source, raw_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(traffic_context_key, observed_at_kst) DO UPDATE SET
            service_date = excluded.service_date,
            hour = excluded.hour,
            avg_speed = excluded.avg_speed,
            traffic_volume = excluded.traffic_volume,
            delay_time = excluded.delay_time,
            congestion_level = excluded.congestion_level,
            source = excluded.source,
            raw_json = excluded.raw_json,
            created_at = excluded.created_at
        """,
        values,
    )
    conn.commit()
    return len(values)


def insert_holidays(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]], source: str = "external_import") -> int:
    now = utc_now_iso()
    values = []
    for row in rows:
        service_date = _date_string(row.get("service_date") or row.get("locdate"))
        if not service_date:
            continue
        values.append(
            (
                service_date,
                _int_or_default(row.get("is_holiday") or row.get("isHoliday"), 1),
                str(row.get("holiday_name") or row.get("dateName") or ""),
                source,
                json.dumps(row, ensure_ascii=False),
                now,
            )
        )
    conn.executemany(
        """
        INSERT INTO external_holiday_daily (
            service_date, is_holiday, holiday_name, source, raw_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(service_date) DO UPDATE SET
            is_holiday = excluded.is_holiday,
            holiday_name = excluded.holiday_name,
            source = excluded.source,
            raw_json = excluded.raw_json,
            created_at = excluded.created_at
        """,
        values,
    )
    conn.commit()
    return len(values)


def insert_events_daily(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]], source: str = "external_import") -> int:
    now = utc_now_iso()
    values = []
    for row in rows:
        service_date = _date_string(row.get("service_date") or row.get("event_date") or row.get("beginDate"))
        if not service_date:
            continue
        values.append(
            (
                str(row.get("area_key") or row.get("regionName") or "global"),
                service_date,
                _int_or_default(row.get("event_count") or row.get("eventCount"), 1),
                _int_or_default(row.get("event_nearby_count") or row.get("eventNearbyCount"), 1),
                source,
                json.dumps(row, ensure_ascii=False),
                now,
            )
        )
    conn.executemany(
        """
        INSERT INTO external_event_daily (
            area_key, service_date, event_count, event_nearby_count, source, raw_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(area_key, service_date) DO UPDATE SET
            event_count = excluded.event_count,
            event_nearby_count = excluded.event_nearby_count,
            source = excluded.source,
            raw_json = excluded.raw_json,
            created_at = excluded.created_at
        """,
        values,
    )
    conn.commit()
    return len(values)


def build_model_hourly_features(conn: sqlite3.Connection, export_csv_path: str | Path | None = None) -> ModelFeatureSummary:
    materialize_location_features(conn)
    base_rows = rows_to_dicts(conn.execute("SELECT * FROM preprocessed_location_features ORDER BY snapshot_id").fetchall())
    weather_rows = _load_time_rows(conn, "external_weather_hourly", "observed_at_kst")
    air_rows = _load_time_rows(conn, "external_air_quality_hourly", "observed_at_kst")
    traffic_rows = _load_time_rows(conn, "external_traffic_hourly", "observed_at_kst")
    holidays = {row["service_date"]: dict(row) for row in conn.execute("SELECT * FROM external_holiday_daily").fetchall()}
    events = _load_event_rows(conn)

    feature_rows = []
    impute_counts = {"weather": 0, "air": 0, "traffic": 0, "event": 0}
    for row in base_rows:
        feature = _build_model_feature_row(row, weather_rows, air_rows, traffic_rows, holidays, events)
        impute_counts["weather"] += feature["weather_imputed"]
        impute_counts["air"] += feature["air_quality_imputed"]
        impute_counts["traffic"] += feature["traffic_imputed"]
        impute_counts["event"] += feature["event_imputed"]
        feature_rows.append(feature)

    conn.executemany(
        """
        INSERT INTO model_hourly_features (
            snapshot_id, collected_at_kst, observed_hour_kst, service_date, day_of_week,
            day_name_ko, is_weekend, is_holiday, holiday_name, hour, time_bucket_10m,
            time_period, route_id, route_name, canonical_route_name, route_type_cd,
            route_type_name, veh_id, plate_no, station_id, station_seq, station_name,
            mobile_no, x, y, remain_seat_cnt, crowded, crowded_label, estimated_capacity,
            seat_scarcity_score, is_no_seat, is_low_seat_2, is_low_seat_5, temperature,
            precipitation, humidity, wind_speed, cloud_amount, weather_text, weather_imputed,
            pm10, pm25, o3, khai, air_quality_grade, air_quality_imputed, avg_speed,
            traffic_volume, delay_time, congestion_level, traffic_imputed, event_count,
            event_nearby_count, event_imputed, created_at
        ) VALUES (
            :snapshot_id, :collected_at_kst, :observed_hour_kst, :service_date, :day_of_week,
            :day_name_ko, :is_weekend, :is_holiday, :holiday_name, :hour, :time_bucket_10m,
            :time_period, :route_id, :route_name, :canonical_route_name, :route_type_cd,
            :route_type_name, :veh_id, :plate_no, :station_id, :station_seq, :station_name,
            :mobile_no, :x, :y, :remain_seat_cnt, :crowded, :crowded_label, :estimated_capacity,
            :seat_scarcity_score, :is_no_seat, :is_low_seat_2, :is_low_seat_5, :temperature,
            :precipitation, :humidity, :wind_speed, :cloud_amount, :weather_text, :weather_imputed,
            :pm10, :pm25, :o3, :khai, :air_quality_grade, :air_quality_imputed, :avg_speed,
            :traffic_volume, :delay_time, :congestion_level, :traffic_imputed, :event_count,
            :event_nearby_count, :event_imputed, :created_at
        )
        ON CONFLICT(snapshot_id) DO UPDATE SET
            collected_at_kst = excluded.collected_at_kst,
            observed_hour_kst = excluded.observed_hour_kst,
            service_date = excluded.service_date,
            day_of_week = excluded.day_of_week,
            day_name_ko = excluded.day_name_ko,
            is_weekend = excluded.is_weekend,
            is_holiday = excluded.is_holiday,
            holiday_name = excluded.holiday_name,
            hour = excluded.hour,
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
            estimated_capacity = excluded.estimated_capacity,
            seat_scarcity_score = excluded.seat_scarcity_score,
            is_no_seat = excluded.is_no_seat,
            is_low_seat_2 = excluded.is_low_seat_2,
            is_low_seat_5 = excluded.is_low_seat_5,
            temperature = excluded.temperature,
            precipitation = excluded.precipitation,
            humidity = excluded.humidity,
            wind_speed = excluded.wind_speed,
            cloud_amount = excluded.cloud_amount,
            weather_text = excluded.weather_text,
            weather_imputed = excluded.weather_imputed,
            pm10 = excluded.pm10,
            pm25 = excluded.pm25,
            o3 = excluded.o3,
            khai = excluded.khai,
            air_quality_grade = excluded.air_quality_grade,
            air_quality_imputed = excluded.air_quality_imputed,
            avg_speed = excluded.avg_speed,
            traffic_volume = excluded.traffic_volume,
            delay_time = excluded.delay_time,
            congestion_level = excluded.congestion_level,
            traffic_imputed = excluded.traffic_imputed,
            event_count = excluded.event_count,
            event_nearby_count = excluded.event_nearby_count,
            event_imputed = excluded.event_imputed,
            created_at = excluded.created_at
        """,
        feature_rows,
    )
    conn.commit()

    exported_path = None
    if export_csv_path:
        exported_path = str(export_model_features_csv(conn, export_csv_path))

    return ModelFeatureSummary(
        processed=len(feature_rows),
        weather_imputed=impute_counts["weather"],
        air_quality_imputed=impute_counts["air"],
        traffic_imputed=impute_counts["traffic"],
        event_imputed=impute_counts["event"],
        exported_path=exported_path,
    )


def export_model_features_csv(conn: sqlite3.Connection, export_csv_path: str | Path) -> Path:
    path = Path(export_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows_to_dicts(
        conn.execute(
            """
            SELECT *
            FROM model_hourly_features
            ORDER BY observed_hour_kst, canonical_route_name, station_seq, veh_id
            """
        ).fetchall()
    )
    _write_rows_csv(rows, path)
    return path


def build_live_model_feature_row(conn: sqlite3.Connection, base_row: dict[str, Any]) -> dict[str, Any]:
    """Build one model feature row from a live GBIS-derived base row."""
    weather_rows = _load_time_rows(conn, "external_weather_hourly", "observed_at_kst")
    air_rows = _load_time_rows(conn, "external_air_quality_hourly", "observed_at_kst")
    traffic_rows = _load_time_rows(conn, "external_traffic_hourly", "observed_at_kst")
    holidays = {row["service_date"]: dict(row) for row in conn.execute("SELECT * FROM external_holiday_daily").fetchall()}
    events = _load_event_rows(conn)
    return _build_model_feature_row(
        _with_live_temporal_defaults(base_row),
        weather_rows,
        air_rows,
        traffic_rows,
        holidays,
        events,
    )


def import_external_csv(conn: sqlite3.Connection, kind: str, path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if kind == "weather":
        return insert_weather_hourly(conn, rows, source=f"csv:{path}")
    if kind == "air":
        return insert_air_quality_hourly(conn, rows, source=f"csv:{path}")
    if kind == "traffic":
        return insert_traffic_hourly(conn, rows, source=f"csv:{path}")
    if kind == "holiday":
        return insert_holidays(conn, rows, source=f"csv:{path}")
    if kind == "event":
        return insert_events_daily(conn, rows, source=f"csv:{path}")
    raise ValueError(f"지원하지 않는 외부 CSV 종류입니다: {kind}")


def _build_model_feature_row(
    row: dict[str, Any],
    weather_rows: list[dict[str, Any]],
    air_rows: list[dict[str, Any]],
    traffic_rows: list[dict[str, Any]],
    holidays: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    collected = _parse_any_kst(row["collected_at_kst"]) or datetime.now(KST)
    observed_hour = collected.replace(minute=0, second=0, microsecond=0)
    service_date = row.get("service_date") or observed_hour.date().isoformat()

    weather, weather_imputed = _nearest_time_row(weather_rows, observed_hour, _default_weather(observed_hour))
    air, air_imputed = _nearest_time_row(air_rows, observed_hour, _default_air(observed_hour))
    traffic, traffic_imputed = _nearest_time_row(traffic_rows, observed_hour, _default_traffic(observed_hour))
    event, event_imputed = _nearest_event_row(events, service_date, _default_event(service_date))
    holiday = holidays.get(service_date, {"is_holiday": 0, "holiday_name": ""})

    return {
        "snapshot_id": _int_or_default(row.get("snapshot_id"), 0),
        "collected_at_kst": row.get("collected_at_kst") or observed_hour.isoformat(timespec="seconds"),
        "observed_hour_kst": observed_hour.isoformat(timespec="seconds"),
        "service_date": service_date,
        "day_of_week": _int_or_default(row.get("day_of_week"), observed_hour.weekday()),
        "day_name_ko": _str_or_default(row.get("day_name_ko"), ""),
        "is_weekend": _int_or_default(row.get("is_weekend"), 0),
        "is_holiday": _int_or_default(holiday.get("is_holiday"), 0),
        "holiday_name": _str_or_default(holiday.get("holiday_name"), ""),
        "hour": _int_or_default(row.get("hour"), observed_hour.hour),
        "time_bucket_10m": _str_or_default(row.get("time_bucket_10m"), f"{observed_hour.hour:02d}:00"),
        "time_period": _str_or_default(row.get("time_period"), "unknown"),
        "route_id": _int_or_default(row.get("route_id"), 0),
        "route_name": _str_or_default(row.get("route_name"), "unknown"),
        "canonical_route_name": _str_or_default(row.get("canonical_route_name"), "unknown"),
        "route_type_cd": _int_or_default(row.get("route_type_cd"), 0),
        "route_type_name": _str_or_default(row.get("route_type_name"), "unknown"),
        "veh_id": _int_or_default(row.get("veh_id"), 0),
        "plate_no": _str_or_default(row.get("plate_no"), "unknown"),
        "station_id": _int_or_default(row.get("station_id"), 0),
        "station_seq": _int_or_default(row.get("station_seq"), 0),
        "station_name": _str_or_default(row.get("station_name"), "unknown"),
        "mobile_no": _str_or_default(row.get("mobile_no"), "unknown"),
        "x": _float_or_default(row.get("x"), 0.0),
        "y": _float_or_default(row.get("y"), 0.0),
        "remain_seat_cnt": _int_or_default(row.get("remain_seat_cnt"), -1),
        "crowded": _int_or_default(row.get("crowded"), 0),
        "crowded_label": _str_or_default(row.get("crowded_label"), "unknown"),
        "estimated_capacity": _int_or_default(row.get("estimated_capacity"), 45),
        "seat_scarcity_score": _int_or_default(row.get("seat_scarcity_score"), 0),
        "is_no_seat": _int_or_default(row.get("is_no_seat"), 0),
        "is_low_seat_2": _int_or_default(row.get("is_low_seat_2"), 0),
        "is_low_seat_5": _int_or_default(row.get("is_low_seat_5"), 0),
        "temperature": _float_or_default(weather.get("temperature"), 0.0),
        "precipitation": _float_or_default(weather.get("precipitation"), 0.0),
        "humidity": _float_or_default(weather.get("humidity"), 0.0),
        "wind_speed": _float_or_default(weather.get("wind_speed"), 0.0),
        "cloud_amount": _float_or_default(weather.get("cloud_amount"), 0.0),
        "weather_text": _str_or_default(weather.get("weather_text"), "unknown"),
        "weather_imputed": weather_imputed,
        "pm10": _float_or_default(air.get("pm10"), 0.0),
        "pm25": _float_or_default(air.get("pm25"), 0.0),
        "o3": _float_or_default(air.get("o3"), 0.0),
        "khai": _float_or_default(air.get("khai"), 0.0),
        "air_quality_grade": _str_or_default(air.get("air_quality_grade"), "unknown"),
        "air_quality_imputed": air_imputed,
        "avg_speed": _float_or_default(traffic.get("avg_speed"), 0.0),
        "traffic_volume": _float_or_default(traffic.get("traffic_volume"), 0.0),
        "delay_time": _float_or_default(traffic.get("delay_time"), 0.0),
        "congestion_level": _int_or_default(traffic.get("congestion_level"), 0),
        "traffic_imputed": traffic_imputed,
        "event_count": _int_or_default(event.get("event_count"), 0),
        "event_nearby_count": _int_or_default(event.get("event_nearby_count"), 0),
        "event_imputed": event_imputed,
        "created_at": utc_now_iso(),
    }


def _with_live_temporal_defaults(row: dict[str, Any]) -> dict[str, Any]:
    collected = _parse_any_kst(row.get("collected_at_kst")) or datetime.now(KST)
    minute_bucket = (collected.minute // 10) * 10
    result = dict(row)
    result.setdefault("snapshot_id", 0)
    result.setdefault("collected_at_kst", collected.isoformat(timespec="seconds"))
    result.setdefault("service_date", collected.date().isoformat())
    result.setdefault("day_of_week", collected.weekday())
    result.setdefault("day_name_ko", _day_name_ko(collected.weekday()))
    result.setdefault("is_weekend", 1 if collected.weekday() >= 5 else 0)
    result.setdefault("hour", collected.hour)
    result.setdefault("time_bucket_10m", f"{collected.hour:02d}:{minute_bucket:02d}")
    result.setdefault("time_period", _time_period(collected.hour))
    return result


def _load_time_rows(conn: sqlite3.Connection, table_name: str, time_column: str) -> list[dict[str, Any]]:
    rows = rows_to_dicts(conn.execute(f"SELECT * FROM {table_name}").fetchall())
    for row in rows:
        row["_dt"] = _parse_any_kst(row[time_column])
    return [row for row in rows if row["_dt"] is not None]


def _load_event_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = rows_to_dicts(conn.execute("SELECT * FROM external_event_daily").fetchall())
    for row in rows:
        row["_date"] = datetime.fromisoformat(row["service_date"]).date()
    return rows


def _nearest_time_row(rows: list[dict[str, Any]], target: datetime, default: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if not rows:
        return default, 1
    exact = [row for row in rows if row["_dt"] == target]
    if exact:
        return exact[0], 0
    nearest = min(rows, key=lambda row: abs((row["_dt"] - target).total_seconds()))
    if abs(nearest["_dt"] - target) > MAX_NEAREST_TIME_DELTA:
        return default, 1
    return nearest, 1


def _nearest_event_row(rows: list[dict[str, Any]], service_date: str, default: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if not rows:
        return default, 1
    target = datetime.fromisoformat(service_date).date()
    exact = [row for row in rows if row["_date"] == target]
    if exact:
        return exact[0], 0
    return default, 1


def _write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    clean_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(clean_rows[0].keys()))
        writer.writeheader()
        writer.writerows(clean_rows)


def _parse_kma_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M").replace(tzinfo=KST)
    except ValueError:
        return _parse_any_kst(value)


def _parse_any_kst(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    compact_formats = {8: "%Y%m%d", 10: "%Y%m%d%H", 12: "%Y%m%d%H%M"}
    if text.isdigit() and len(text) in compact_formats:
        try:
            return datetime.strptime(text, compact_formats[len(text)]).replace(tzinfo=KST)
        except ValueError:
            return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=KST)
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=KST)
    return parsed.astimezone(KST)


def _date_string(value: Any) -> str | None:
    parsed = _parse_any_kst(value)
    if parsed is None:
        return None
    return parsed.date().isoformat()


def _weather_text(precipitation: float, cloud_amount: float) -> str:
    if precipitation > 0:
        return "rain"
    if cloud_amount >= 8:
        return "cloudy"
    if cloud_amount >= 3:
        return "partly_cloudy"
    return "clear"


def _default_weather(target: datetime) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "precipitation": 0.0,
        "humidity": 0.0,
        "wind_speed": 0.0,
        "cloud_amount": 0.0,
        "weather_text": "unknown",
        "_dt": target,
    }


def _default_air(target: datetime) -> dict[str, Any]:
    return {"pm10": 0.0, "pm25": 0.0, "o3": 0.0, "khai": 0.0, "air_quality_grade": "unknown", "_dt": target}


def _default_traffic(target: datetime) -> dict[str, Any]:
    return {"avg_speed": 0.0, "traffic_volume": 0.0, "delay_time": 0.0, "congestion_level": 0, "_dt": target}


def _default_event(service_date: str) -> dict[str, Any]:
    return {"service_date": service_date, "event_count": 0, "event_nearby_count": 0}


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


def _float_or_default(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_or_default(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _str_or_default(value: Any, default: str) -> str:
    if value is None or value == "":
        return default
    return str(value)
