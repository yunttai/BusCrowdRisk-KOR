import tempfile
import unittest
from pathlib import Path

from busseat_ai.external_context import build_model_hourly_features, insert_holidays, insert_weather_hourly
from busseat_ai.preprocessing import upsert_route_stations, upsert_target_routes
from busseat_ai.quality import build_data_quality_report
from busseat_ai.storage.database import connect, init_db, insert_location_snapshots
from busseat_ai.targets import build_target_labels


class ExternalContextTests(unittest.TestCase):
    def test_insert_holidays_keeps_yyyymmdd_dates_intact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            init_db(db_path)
            conn = connect(db_path)
            try:
                inserted = insert_holidays(
                    conn,
                    [
                        {"locdate": "20260524", "dateName": "부처님오신날"},
                        {"locdate": "20260525", "dateName": "대체공휴일"},
                    ],
                    source="test",
                )
                rows = conn.execute(
                    "SELECT service_date, holiday_name FROM external_holiday_daily ORDER BY service_date"
                ).fetchall()
            finally:
                conn.close()

            self.assertEqual(inserted, 2)
            self.assertEqual([row["service_date"] for row in rows], ["2026-05-24", "2026-05-25"])

    def test_build_model_features_with_nearest_weather_and_no_nulls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            init_db(db_path)
            conn = connect(db_path)
            try:
                upsert_target_routes(
                    conn,
                    [
                        {
                            "routeId": 1,
                            "routeName": "5000",
                            "routeTypeCd": 11,
                            "routeTypeName": "직행좌석형시내버스",
                            "regionName": "용인",
                        }
                    ],
                )
                upsert_route_stations(
                    conn,
                    1,
                    "5000",
                    [
                        {
                            "stationSeq": 1,
                            "stationId": 100,
                            "stationName": "명지대",
                            "mobileNo": "12345",
                            "x": "127.1",
                            "y": "37.2",
                        }
                    ],
                )
                insert_location_snapshots(
                    conn,
                    [
                        {
                            "routeId": 1,
                            "vehId": 10,
                            "plateNo": "경기70바1234",
                            "stationId": 100,
                            "stationSeq": 1,
                            "remainSeatCnt": 2,
                            "crowded": 3,
                            "routeTypeCd": 11,
                        }
                    ],
                    collected_at="2026-05-20T23:05:00+00:00",
                )
                insert_weather_hourly(
                    conn,
                    [
                        {
                            "stnId": "119",
                            "stnNm": "수원",
                            "tm": "2026-05-21 07:00",
                            "ta": "16.0",
                            "rn": "0.5",
                            "hm": "70",
                            "ws": "2.0",
                            "dc10Tca": "8",
                        }
                    ],
                )

                summary = build_model_hourly_features(conn)
                row = conn.execute("SELECT * FROM model_hourly_features").fetchone()
                null_count = 0
                for column in [info[1] for info in conn.execute("PRAGMA table_info(model_hourly_features)").fetchall()]:
                    null_count += conn.execute(f"SELECT COUNT(*) FROM model_hourly_features WHERE {column} IS NULL").fetchone()[0]
            finally:
                conn.close()

            self.assertEqual(summary.processed, 1)
            self.assertEqual(row["observed_hour_kst"], "2026-05-21T08:00:00+09:00")
            self.assertEqual(row["temperature"], 16.0)
            self.assertEqual(row["weather_imputed"], 1)
            self.assertEqual(row["air_quality_imputed"], 1)
            self.assertEqual(row["traffic_imputed"], 1)
            self.assertEqual(row["event_imputed"], 1)
            self.assertEqual(null_count, 0)

    def test_stale_weather_is_not_reused_as_nearest_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            init_db(db_path)
            conn = connect(db_path)
            try:
                upsert_target_routes(
                    conn,
                    [
                        {
                            "routeId": 1,
                            "routeName": "5000",
                            "routeTypeCd": 11,
                            "routeTypeName": "직행좌석형시내버스",
                            "regionName": "용인",
                        }
                    ],
                )
                upsert_route_stations(
                    conn,
                    1,
                    "5000",
                    [{"stationSeq": 1, "stationId": 100, "stationName": "A", "mobileNo": "1", "x": "127.1", "y": "37.2"}],
                )
                insert_location_snapshots(
                    conn,
                    [
                        {
                            "routeId": 1,
                            "vehId": 10,
                            "plateNo": "경기70바1234",
                            "stationId": 100,
                            "stationSeq": 1,
                            "remainSeatCnt": 10,
                            "crowded": 1,
                            "routeTypeCd": 11,
                        }
                    ],
                    collected_at="2026-05-21T10:00:00+00:00",
                )
                insert_weather_hourly(
                    conn,
                    [
                        {
                            "stnId": "119",
                            "stnNm": "수원",
                            "tm": "2026-05-21 07:00",
                            "ta": "16.0",
                            "rn": "0",
                            "hm": "70",
                            "ws": "2.0",
                            "dc10Tca": "8",
                        }
                    ],
                )
                build_model_hourly_features(conn)
                row = conn.execute("SELECT temperature, weather_text, weather_imputed FROM model_hourly_features").fetchone()
            finally:
                conn.close()

            self.assertEqual(row["temperature"], 0.0)
            self.assertEqual(row["weather_text"], "unknown")
            self.assertEqual(row["weather_imputed"], 1)

    def test_quality_and_target_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            init_db(db_path)
            conn = connect(db_path)
            try:
                upsert_target_routes(
                    conn,
                    [
                        {
                            "routeId": 1,
                            "routeName": "5000",
                            "routeTypeCd": 11,
                            "routeTypeName": "직행좌석형시내버스",
                            "regionName": "용인",
                        }
                    ],
                )
                upsert_route_stations(
                    conn,
                    1,
                    "5000",
                    [
                        {"stationSeq": 1, "stationId": 100, "stationName": "A", "mobileNo": "1", "x": "127.1", "y": "37.2"},
                        {"stationSeq": 2, "stationId": 101, "stationName": "B", "mobileNo": "2", "x": "127.2", "y": "37.3"},
                    ],
                )
                insert_location_snapshots(
                    conn,
                    [
                        {
                            "routeId": 1,
                            "vehId": 10,
                            "plateNo": "경기70바1234",
                            "stationId": 100,
                            "stationSeq": 1,
                            "remainSeatCnt": 3,
                            "crowded": 3,
                            "routeTypeCd": 11,
                        }
                    ],
                    collected_at="2026-05-20T23:00:00+00:00",
                )
                insert_location_snapshots(
                    conn,
                    [
                        {
                            "routeId": 1,
                            "vehId": 10,
                            "plateNo": "경기70바1234",
                            "stationId": 101,
                            "stationSeq": 2,
                            "remainSeatCnt": 0,
                            "crowded": 4,
                            "routeTypeCd": 11,
                        }
                    ],
                    collected_at="2026-05-20T23:04:00+00:00",
                )
                build_model_hourly_features(conn)
                quality = build_data_quality_report(conn)
                labels = build_target_labels(conn)
                first = conn.execute("SELECT * FROM model_target_labels WHERE station_seq = 1").fetchone()
            finally:
                conn.close()

            self.assertEqual(quality.total_rows, 2)
            self.assertIn("temperature", quality.excluded_features)
            self.assertEqual(labels.processed, 2)
            self.assertEqual(first["target_no_seat_next_5min"], 1)
            self.assertEqual(first["target_no_seat_next_station"], 1)


if __name__ == "__main__":
    unittest.main()
