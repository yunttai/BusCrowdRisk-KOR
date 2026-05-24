import tempfile
import unittest
from pathlib import Path

from busseat_ai.external_context import build_model_hourly_features
from busseat_ai.preprocessing import upsert_route_stations, upsert_target_routes
from busseat_ai.storage.database import connect, init_db, insert_location_snapshots
from busseat_ai.targets import build_target_labels


class TargetLabelTests(unittest.TestCase):
    def test_next_station_label_ignores_unrealistic_time_gap(self):
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
                            "remainSeatCnt": 5,
                            "crowded": 1,
                            "routeTypeCd": 11,
                        }
                    ],
                    collected_at="2026-05-21T00:00:00+00:00",
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
                    collected_at="2026-05-21T02:00:00+00:00",
                )
                build_model_hourly_features(conn)
                build_target_labels(conn)
                first = conn.execute("SELECT * FROM model_target_labels WHERE station_seq = 1").fetchone()
            finally:
                conn.close()

            self.assertEqual(first["has_next_station"], 0)
            self.assertEqual(first["target_no_seat_next_station"], 0)


if __name__ == "__main__":
    unittest.main()
