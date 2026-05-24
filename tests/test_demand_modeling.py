import tempfile
import unittest
from pathlib import Path

from busseat_ai.demand import build_expected_boardings, insert_station_demand_rows
from busseat_ai.external_context import build_model_hourly_features
from busseat_ai.modeling import evaluate_historical_rate_baseline, export_training_dataset_csv
from busseat_ai.preprocessing import upsert_route_stations, upsert_target_routes
from busseat_ai.storage.database import connect, init_db, insert_location_snapshots
from busseat_ai.targets import build_target_labels


class DemandModelingTests(unittest.TestCase):
    def test_expected_boardings_training_dataset_and_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            training_csv = Path(temp_dir) / "training.csv"
            metrics_json = Path(temp_dir) / "metrics.json"
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
                        {"stationSeq": 3, "stationId": 102, "stationName": "C", "mobileNo": "3", "x": "127.3", "y": "37.4"},
                    ],
                )
                for station_seq, station_id, remain in ((1, 100, 4), (2, 101, 1), (3, 102, 0)):
                    insert_location_snapshots(
                        conn,
                        [
                            {
                                "routeId": 1,
                                "vehId": 10,
                                "plateNo": "경기70바1234",
                                "stationId": station_id,
                                "stationSeq": station_seq,
                                "remainSeatCnt": remain,
                                "crowded": 4 if remain == 0 else 3,
                                "routeTypeCd": 11,
                            }
                        ],
                        collected_at="2026-05-20T23:00:00+00:00",
                    )
                insert_station_demand_rows(
                    conn,
                    [
                        {"service_date": "2026-05-21", "station_id": 100, "boarding_total": 1200},
                        {"service_date": "2026-05-21", "station_id": 101, "boarding_total": 800},
                        {"service_date": "2026-05-21", "station_id": 102, "boarding_total": 600},
                    ],
                )
                build_model_hourly_features(conn)
                build_target_labels(conn)

                expected = build_expected_boardings(conn)
                dataset = export_training_dataset_csv(conn, training_csv)
                baseline = evaluate_historical_rate_baseline(
                    conn,
                    target_column="target_no_seat_next_station",
                    output_json_path=metrics_json,
                )
                expected_row = conn.execute(
                    """
                    SELECT expected_boardings_at_stop
                    FROM station_expected_boardings_hourly
                    WHERE station_id = 100 AND route_id = 1 AND day_of_week = 3 AND hour = 8
                    """
                ).fetchone()
            finally:
                conn.close()

            self.assertGreater(expected.processed, 0)
            self.assertIsNotNone(expected_row)
            self.assertGreater(expected_row["expected_boardings_at_stop"], 0)
            self.assertEqual(dataset.rows, 3)
            self.assertTrue(training_csv.exists())
            self.assertEqual(baseline.target_column, "target_no_seat_next_station")
            self.assertEqual(baseline.total_rows, 2)
            self.assertTrue(metrics_json.exists())


if __name__ == "__main__":
    unittest.main()
