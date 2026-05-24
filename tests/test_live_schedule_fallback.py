import tempfile
import unittest
from pathlib import Path

from busseat_ai.cli import _scheduled_proxy_rows
from busseat_ai.preprocessing import upsert_route_stations, upsert_target_routes
from busseat_ai.storage.database import connect, init_db


class FakeGbisClient:
    def route_info(self, route_id):
        return {
            "routeId": route_id,
            "routeName": "5005",
            "weUpFirstTime": "05:00",
            "weUpLastTime": "23:00",
            "weDownFirstTime": "06:50",
            "weDownLastTime": "00:10",
            "wePeekAlloc": "22",
            "weNPeekAlloc": "26",
        }


class LiveScheduleFallbackTests(unittest.TestCase):
    def test_route_name_only_fallback_uses_db_route_and_station_direction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            init_db(db_path)
            conn = connect(db_path)
            try:
                upsert_target_routes(
                    conn,
                    [
                        {
                            "routeId": 228000175,
                            "routeName": "5005",
                            "routeTypeCd": 11,
                            "routeTypeName": "직행좌석형시내버스",
                            "regionName": "용인",
                        }
                    ],
                )
                upsert_route_stations(
                    conn,
                    228000175,
                    "5005",
                    [
                        {
                            "stationSeq": 63,
                            "stationId": 102000070,
                            "stationName": "순천향대학병원",
                            "mobileNo": None,
                            "x": "127.0054",
                            "y": "37.5359",
                            "direction": "DOWN",
                        }
                    ],
                )
                rows = _scheduled_proxy_rows(
                    conn,
                    FakeGbisClient(),
                    [],
                    requested_station_id=102000070,
                    route_id=None,
                    route_name="5005",
                    at="2026-05-20 08:00:00",
                )
            finally:
                conn.close()

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["route_id"], 228000175)
            self.assertEqual(rows[0]["schedule_direction"], "down")
            self.assertEqual(rows[0]["schedule_first_time"], "06:50")
            self.assertEqual(rows[0]["schedule_proxy_time_basis"], "requested_time")


if __name__ == "__main__":
    unittest.main()
