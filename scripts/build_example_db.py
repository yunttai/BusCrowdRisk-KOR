from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from busseat_ai.preprocessing import (  # noqa: E402
    export_location_features_csv,
    materialize_location_features,
    upsert_route_stations,
    upsert_target_routes,
)
from busseat_ai.demand import build_expected_boardings, insert_station_demand_rows  # noqa: E402
from busseat_ai.external_context import (  # noqa: E402
    build_model_hourly_features,
    export_model_features_csv,
    insert_air_quality_hourly,
    insert_events_daily,
    insert_holidays,
    insert_traffic_hourly,
    insert_weather_hourly,
)
from busseat_ai.modeling import evaluate_historical_rate_baseline, export_training_dataset_csv  # noqa: E402
from busseat_ai.quality import build_data_quality_report  # noqa: E402
from busseat_ai.storage.database import connect, init_db, insert_location_snapshots  # noqa: E402
from busseat_ai.targets import build_target_labels  # noqa: E402


SAMPLE_ROUTES = [
    {
        "routeId": 5000001,
        "routeName": "5000",
        "routeTypeCd": 11,
        "routeTypeName": "직행좌석형시내버스",
        "regionName": "용인",
        "startStationName": "명지대",
        "endStationName": "서울역",
    },
    {
        "routeId": 5001001,
        "routeName": "5001(A)",
        "routeTypeCd": 11,
        "routeTypeName": "직행좌석형시내버스",
        "regionName": "용인",
        "startStationName": "명지대",
        "endStationName": "강남역",
    },
    {
        "routeId": 5001002,
        "routeName": "5001(B)",
        "routeTypeCd": 11,
        "routeTypeName": "직행좌석형시내버스",
        "regionName": "용인",
        "startStationName": "명지대",
        "endStationName": "강남역",
    },
    {
        "routeId": 5001100,
        "routeName": "5001-1(A)",
        "routeTypeCd": 11,
        "routeTypeName": "직행좌석형시내버스",
        "regionName": "용인",
        "startStationName": "용인터미널",
        "endStationName": "강남역",
    },
    {
        "routeId": 5001101,
        "routeName": "5001-1(B)",
        "routeTypeCd": 11,
        "routeTypeName": "직행좌석형시내버스",
        "regionName": "용인",
        "startStationName": "용인터미널",
        "endStationName": "강남역",
    },
    {
        "routeId": 5003001,
        "routeName": "5003(A)",
        "routeTypeCd": 11,
        "routeTypeName": "직행좌석형시내버스",
        "regionName": "용인",
        "startStationName": "명지대",
        "endStationName": "양재역",
    },
    {
        "routeId": 5003002,
        "routeName": "5003(B)",
        "routeTypeCd": 11,
        "routeTypeName": "직행좌석형시내버스",
        "regionName": "용인",
        "startStationName": "명지대",
        "endStationName": "양재역",
    },
    {
        "routeId": 5005001,
        "routeName": "5005",
        "routeTypeCd": 11,
        "routeTypeName": "직행좌석형시내버스",
        "regionName": "용인",
        "startStationName": "명지대",
        "endStationName": "서울역",
    },
]


SAMPLE_STATIONS = [
    {"stationSeq": 1, "stationId": 228000101, "stationName": "명지대", "mobileNo": "47610", "x": 127.1901, "y": 37.2221, "regionName": "용인"},
    {"stationSeq": 2, "stationId": 228000102, "stationName": "명지대입구", "mobileNo": "47611", "x": 127.1885, "y": 37.2244, "regionName": "용인"},
    {"stationSeq": 3, "stationId": 228000103, "stationName": "용인시청", "mobileNo": "47612", "x": 127.1774, "y": 37.2402, "regionName": "용인"},
    {"stationSeq": 4, "stationId": 228000104, "stationName": "기흥역", "mobileNo": "47613", "x": 127.1151, "y": 37.2757, "regionName": "용인"},
    {"stationSeq": 5, "stationId": 228000105, "stationName": "신갈오거리", "mobileNo": "47614", "x": 127.1067, "y": 37.2863, "regionName": "용인"},
    {"stationSeq": 6, "stationId": 228000106, "stationName": "강남역", "mobileNo": "22009", "x": 127.0276, "y": 37.4979, "regionName": "서울"},
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an example BusSeat SQLite DB and CSV exports.")
    parser.add_argument("--out-dir", default="example-db", help="Output directory for example DB and CSV files.")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "busseat-example.db"
    if db_path.exists():
        db_path.unlink()

    init_db(db_path)
    with connect(db_path) as conn:
        build_sample_data(conn)
        build_sample_station_demand(conn)
        materialize_location_features(conn)
        build_external_context(conn)
        build_model_hourly_features(conn)
        build_target_labels(conn)
        build_expected_boardings(conn)
        export_table_csv(conn, "target_route", out_dir / "target_route.csv")
        export_table_csv(conn, "route_station", out_dir / "route_station.csv")
        export_table_csv(conn, "bus_location_snapshot", out_dir / "bus_location_snapshot.csv")
        export_table_csv(conn, "station_demand_daily", out_dir / "station_demand_daily.csv")
        export_table_csv(conn, "station_expected_boardings_hourly", out_dir / "station_expected_boardings_hourly.csv")
        export_table_csv(conn, "external_weather_hourly", out_dir / "external_weather_hourly.csv")
        export_table_csv(conn, "external_air_quality_hourly", out_dir / "external_air_quality_hourly.csv")
        export_table_csv(conn, "external_traffic_hourly", out_dir / "external_traffic_hourly.csv")
        export_table_csv(conn, "external_holiday_daily", out_dir / "external_holiday_daily.csv")
        export_table_csv(conn, "external_event_daily", out_dir / "external_event_daily.csv")
        export_location_features_csv(conn, out_dir / "preprocessed_location_features.csv")
        export_model_features_csv(conn, out_dir / "model_hourly_features.csv")
        export_table_csv(conn, "model_target_labels", out_dir / "model_target_labels.csv")
        export_training_dataset_csv(conn, out_dir / "training_dataset.csv")
        evaluate_historical_rate_baseline(
            conn,
            target_column="target_no_seat_next_station",
            output_json_path=out_dir / "baseline_metrics.json",
        )
        export_table_csv(conn, "baseline_model_metrics", out_dir / "baseline_model_metrics.csv")
        write_json(build_data_quality_report(conn).to_dict(), out_dir / "data_quality_report.json")
        export_query_csv(
            conn,
            """
            SELECT
                tr.canonical_route_name,
                tr.route_name,
                tr.route_id,
                COUNT(rs.station_seq) AS station_count
            FROM target_route tr
            LEFT JOIN route_station rs ON rs.route_id = tr.route_id
            GROUP BY tr.canonical_route_name, tr.route_name, tr.route_id
            ORDER BY tr.canonical_route_name
            """,
            out_dir / "route_station_summary.csv",
        )
        export_query_csv(
            conn,
            """
            SELECT
                service_date,
                day_name_ko,
                time_period,
                canonical_route_name,
                COUNT(*) AS observation_count,
                ROUND(AVG(remain_seat_cnt), 2) AS avg_remain_seat_cnt,
                MIN(remain_seat_cnt) AS min_remain_seat_cnt,
                MAX(remain_seat_cnt) AS max_remain_seat_cnt,
                MAX(seat_scarcity_score) AS max_seat_scarcity_score,
                SUM(is_no_seat) AS no_seat_count,
                SUM(is_low_seat_2) AS low_seat_2_count,
                SUM(is_low_seat_5) AS low_seat_5_count
            FROM preprocessed_location_features
            GROUP BY service_date, day_name_ko, time_period, canonical_route_name
            ORDER BY service_date, time_period, canonical_route_name
            """,
            out_dir / "feature_summary_by_route_time.csv",
        )
        export_query_csv(
            conn,
            """
            SELECT
                service_date,
                day_name_ko,
                time_period,
                canonical_route_name,
                COUNT(*) AS observation_count,
                ROUND(AVG(remain_seat_cnt), 2) AS avg_remain_seat_cnt,
                ROUND(AVG(temperature), 2) AS avg_temperature,
                ROUND(AVG(precipitation), 2) AS avg_precipitation,
                ROUND(AVG(pm10), 2) AS avg_pm10,
                ROUND(AVG(avg_speed), 2) AS avg_traffic_speed,
                MAX(event_count) AS max_event_count,
                MAX(is_holiday) AS is_holiday,
                SUM(weather_imputed) AS weather_imputed_count,
                SUM(air_quality_imputed) AS air_quality_imputed_count,
                SUM(traffic_imputed) AS traffic_imputed_count,
                SUM(event_imputed) AS event_imputed_count
            FROM model_hourly_features
            GROUP BY service_date, day_name_ko, time_period, canonical_route_name
            ORDER BY service_date, time_period, canonical_route_name
            """,
            out_dir / "model_summary_by_route_time.csv",
        )

    print(
        json.dumps(
            {
                "dbPath": str(db_path.relative_to(ROOT)),
                "csvFiles": [
                    str((out_dir / "target_route.csv").relative_to(ROOT)),
                    str((out_dir / "route_station.csv").relative_to(ROOT)),
                    str((out_dir / "bus_location_snapshot.csv").relative_to(ROOT)),
                    str((out_dir / "station_demand_daily.csv").relative_to(ROOT)),
                    str((out_dir / "station_expected_boardings_hourly.csv").relative_to(ROOT)),
                    str((out_dir / "external_weather_hourly.csv").relative_to(ROOT)),
                    str((out_dir / "external_air_quality_hourly.csv").relative_to(ROOT)),
                    str((out_dir / "external_traffic_hourly.csv").relative_to(ROOT)),
                    str((out_dir / "external_holiday_daily.csv").relative_to(ROOT)),
                    str((out_dir / "external_event_daily.csv").relative_to(ROOT)),
                    str((out_dir / "preprocessed_location_features.csv").relative_to(ROOT)),
                    str((out_dir / "model_hourly_features.csv").relative_to(ROOT)),
                    str((out_dir / "model_target_labels.csv").relative_to(ROOT)),
                    str((out_dir / "training_dataset.csv").relative_to(ROOT)),
                    str((out_dir / "baseline_metrics.json").relative_to(ROOT)),
                    str((out_dir / "baseline_model_metrics.csv").relative_to(ROOT)),
                    str((out_dir / "data_quality_report.json").relative_to(ROOT)),
                    str((out_dir / "route_station_summary.csv").relative_to(ROOT)),
                    str((out_dir / "feature_summary_by_route_time.csv").relative_to(ROOT)),
                    str((out_dir / "model_summary_by_route_time.csv").relative_to(ROOT)),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_sample_data(conn: sqlite3.Connection) -> None:
    upsert_target_routes(conn, SAMPLE_ROUTES)
    for route in SAMPLE_ROUTES:
        upsert_route_stations(conn, route["routeId"], normalize_sample_route_name(route["routeName"]), SAMPLE_STATIONS)

    collected_times = [
        "2026-05-20T22:50:00+00:00",  # 07:50 KST morning peak
        "2026-05-20T23:10:00+00:00",  # 08:10 KST morning peak
        "2026-05-21T08:30:00+00:00",  # 17:30 KST evening peak
    ]
    seats_by_time = [
        [18, 9, 4, 2, 0],
        [15, 6, 3, 1, 0],
        [22, 12, 7, 4, 2],
    ]
    crowded_by_seats = [(1, 18), (2, 8), (3, 4), (3, 2), (4, 0)]

    for time_index, collected_at in enumerate(collected_times):
        snapshots = []
        for route_index, route in enumerate(SAMPLE_ROUTES):
            for station_index, station in enumerate(SAMPLE_STATIONS[:5]):
                base_seat = seats_by_time[time_index][station_index]
                remain_seat = max(0, base_seat - route_index)
                crowded = crowded_from_remain(remain_seat)
                snapshots.append(
                    {
                        "routeId": route["routeId"],
                        "vehId": route["routeId"] + time_index,
                        "plateNo": f"경기70바{route_index + 1}{time_index + 1:02d}",
                        "stationId": station["stationId"],
                        "stationSeq": station["stationSeq"],
                        "remainSeatCnt": remain_seat,
                        "crowded": crowded,
                        "routeTypeCd": route["routeTypeCd"],
                        "lowPlate": 0,
                        "stateCd": 1,
                        "taglessCd": 0,
                    }
                )
        insert_location_snapshots(conn, snapshots, collected_at=collected_at)

    late_snapshots = []
    for route_index, route in enumerate(SAMPLE_ROUTES):
        late_snapshots.append(
            {
                "routeId": route["routeId"],
                "vehId": route["routeId"] + 99,
                "plateNo": f"경기70바{route_index + 1}99",
                "stationId": SAMPLE_STATIONS[1]["stationId"],
                "stationSeq": SAMPLE_STATIONS[1]["stationSeq"],
                "remainSeatCnt": 35 - route_index,
                "crowded": 1,
                "routeTypeCd": route["routeTypeCd"],
                "lowPlate": 0,
                "stateCd": 1,
                "taglessCd": 0,
            }
        )
    insert_location_snapshots(conn, late_snapshots, collected_at="2026-05-21T16:20:00+00:00")


def build_sample_station_demand(conn: sqlite3.Connection) -> None:
    demand_rows = []
    base_boardings = [1850, 1240, 980, 1460, 1320, 2100]
    for date_index, service_date in enumerate(("2026-05-20", "2026-05-21")):
        for station, base in zip(SAMPLE_STATIONS, base_boardings):
            boarding_total = base + date_index * 80
            demand_rows.append(
                {
                    "service_date": service_date,
                    "station_id": station["stationId"],
                    "mobile_no": station["mobileNo"],
                    "station_name": station["stationName"],
                    "city_name": station["regionName"],
                    "boarding_total": boarding_total,
                    "first_boarding_total": int(boarding_total * 0.82),
                    "transfer_total": int(boarding_total * 0.18),
                    "alighting_total": int(boarding_total * 0.91),
                }
            )
    insert_station_demand_rows(conn, demand_rows)


def build_external_context(conn: sqlite3.Connection) -> None:
    insert_weather_hourly(
        conn,
        [
            {"stnId": "119", "stnNm": "수원", "tm": "2026-05-21 07:00", "ta": "15.8", "rn": "0", "hm": "62", "ws": "1.8", "dc10Tca": "2"},
            {"stnId": "119", "stnNm": "수원", "tm": "2026-05-21 08:00", "ta": "17.1", "rn": "0", "hm": "58", "ws": "2.1", "dc10Tca": "4"},
            {"stnId": "119", "stnNm": "수원", "tm": "2026-05-21 17:00", "ta": "22.4", "rn": "1.2", "hm": "72", "ws": "3.4", "dc10Tca": "9"},
            {"stnId": "119", "stnNm": "수원", "tm": "2026-05-22 01:00", "ta": "13.6", "rn": "0", "hm": "66", "ws": "1.0", "dc10Tca": "1"},
        ],
        source="example:KMA_ASOS",
    )
    insert_air_quality_hourly(
        conn,
        [
            {"air_station_name": "용인", "observed_at_kst": "2026-05-21T07:00:00+09:00", "pm10": 38, "pm25": 18, "o3": 0.031, "khai": 62, "air_quality_grade": "보통"},
            {"air_station_name": "용인", "observed_at_kst": "2026-05-21T08:00:00+09:00", "pm10": 42, "pm25": 20, "o3": 0.034, "khai": 67, "air_quality_grade": "보통"},
            {"air_station_name": "용인", "observed_at_kst": "2026-05-21T17:00:00+09:00", "pm10": 56, "pm25": 31, "o3": 0.041, "khai": 82, "air_quality_grade": "나쁨"},
        ],
        source="example:AirKorea",
    )
    insert_traffic_hourly(
        conn,
        [
            {"traffic_context_key": "yongin_seoul_corridor", "observed_at_kst": "2026-05-21T07:00:00+09:00", "avg_speed": 41.5, "traffic_volume": 820, "delay_time": 6.0, "congestion_level": 2},
            {"traffic_context_key": "yongin_seoul_corridor", "observed_at_kst": "2026-05-21T08:00:00+09:00", "avg_speed": 28.2, "traffic_volume": 1140, "delay_time": 13.5, "congestion_level": 4},
            {"traffic_context_key": "yongin_seoul_corridor", "observed_at_kst": "2026-05-21T17:00:00+09:00", "avg_speed": 24.7, "traffic_volume": 1310, "delay_time": 18.0, "congestion_level": 4},
        ],
        source="example:GyeonggiTraffic",
    )
    insert_holidays(
        conn,
        [
            {"service_date": "2026-05-21", "is_holiday": 0, "holiday_name": ""},
            {"service_date": "2026-05-22", "is_holiday": 0, "holiday_name": ""},
        ],
        source="example:KASI",
    )
    insert_events_daily(
        conn,
        [
            {"area_key": "global", "service_date": "2026-05-21", "event_count": 2, "event_nearby_count": 1},
            {"area_key": "global", "service_date": "2026-05-22", "event_count": 0, "event_nearby_count": 0},
        ],
        source="example:GyeonggiEvent",
    )


def export_table_csv(conn: sqlite3.Connection, table_name: str, path: Path) -> None:
    rows = [dict(row) for row in conn.execute(f"SELECT * FROM {table_name}").fetchall()]
    write_rows_csv(rows, path)


def export_query_csv(conn: sqlite3.Connection, query: str, path: Path) -> None:
    rows = [dict(row) for row in conn.execute(query).fetchall()]
    write_rows_csv(rows, path)


def write_rows_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    import csv

    try:
        file = path.open("w", encoding="utf-8-sig", newline="")
    except PermissionError:
        fallback = path.with_name(f"{path.stem}.new{path.suffix}")
        file = fallback.open("w", encoding="utf-8-sig", newline="")

    with file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(value: dict, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def crowded_from_remain(remain_seat: int) -> int:
    if remain_seat <= 0:
        return 4
    if remain_seat <= 5:
        return 3
    if remain_seat <= 12:
        return 2
    return 1


def normalize_sample_route_name(route_name: str) -> str:
    return route_name.upper().replace("(", "").replace(")", "")


if __name__ == "__main__":
    raise SystemExit(main())
