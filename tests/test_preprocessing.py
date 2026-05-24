import sqlite3
import tempfile
import unittest
from pathlib import Path

from busseat_ai.preprocessing import (
    is_target_route,
    materialize_location_features,
    normalize_route_name,
    upsert_route_stations,
    upsert_target_routes,
)
from busseat_ai.storage.database import connect, init_db


class PreprocessingTests(unittest.TestCase):
    def test_normalize_route_name_variants(self):
        self.assertEqual(normalize_route_name("5001(A)"), "5001A")
        self.assertEqual(normalize_route_name("5001-1 (b)"), "5001-1B")
        self.assertEqual(normalize_route_name("5000번"), "5000")

    def test_target_route_matching(self):
        self.assertTrue(is_target_route({"routeName": "5003(B)"}))
        self.assertFalse(is_target_route({"routeName": "5002"}))

    def test_materialize_location_features(self):
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
                            "stationSeq": 10,
                            "stationId": 100,
                            "stationName": "명지대입구",
                            "mobileNo": "12345",
                            "x": "127.1",
                            "y": "37.2",
                        }
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO bus_location_snapshot (
                        collected_at, route_id, veh_id, plate_no, station_id, station_seq,
                        remain_seat_cnt, crowded, route_type_cd, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "2026-05-20T23:05:00+00:00",
                        1,
                        77,
                        "경기70바1234",
                        100,
                        10,
                        2,
                        3,
                        11,
                        "{}",
                    ),
                )
                conn.commit()

                summary = materialize_location_features(conn)
                row = conn.execute("SELECT * FROM preprocessed_location_features").fetchone()
            finally:
                conn.close()

            self.assertEqual(summary.processed, 1)
            self.assertIsInstance(row, sqlite3.Row)
            self.assertEqual(row["collected_at_kst"], "2026-05-21T08:05:00+09:00")
            self.assertEqual(row["day_name_ko"], "목")
            self.assertEqual(row["time_bucket_10m"], "08:00")
            self.assertEqual(row["time_period"], "morning_peak")
            self.assertEqual(row["station_name"], "명지대입구")
            self.assertEqual(row["seat_scarcity_score"], 93)
            self.assertEqual(row["is_low_seat_2"], 1)


if __name__ == "__main__":
    unittest.main()
