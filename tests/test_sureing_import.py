import tempfile
import unittest
from pathlib import Path

from busseat_ai.importers.sureing_mysql_dump import import_sureing_mysql_dump
from busseat_ai.storage.database import connect, init_db


class SureingImportTests(unittest.TestCase):
    def test_import_target_snapshot_and_convert_minus_one_to_zero(self):
        dump = """
CREATE TABLE `bus_route` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `created_at` datetime(6) NOT NULL,
  `updated_at` datetime(6) NOT NULL,
  `active` bit(1) NOT NULL,
  `default_capacity` int NOT NULL,
  `end_station_name` varchar(100) DEFAULT NULL,
  `external_route_id` varchar(50) NOT NULL,
  `route_name` varchar(50) NOT NULL,
  `route_type` enum('EXPRESS_SEAT','METROPOLITAN','UNKNOWN') NOT NULL,
  `start_station_name` varchar(100) DEFAULT NULL,
  `destination_zone` enum('GANGBYEON','GANGNAM','JONGNO','OTHER','SEOUL_STATION','UNKNOWN') NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
INSERT INTO `bus_route` VALUES (1,'2026-05-13 00:00:00.000000','2026-05-13 00:00:00.000000',_binary '',45,'서울역','228000174','5000','EXPRESS_SEAT','남동차고지','SEOUL_STATION'),(2,'2026-05-13 00:00:00.000000','2026-05-13 00:00:00.000000',_binary '',45,'기타','999','999','EXPRESS_SEAT','기타','OTHER');
CREATE TABLE `station` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `created_at` datetime(6) NOT NULL,
  `updated_at` datetime(6) NOT NULL,
  `city_name` varchar(50) DEFAULT NULL,
  `external_station_id` varchar(50) NOT NULL,
  `latitude` double DEFAULT NULL,
  `longitude` double DEFAULT NULL,
  `station_name` varchar(100) NOT NULL,
  `station_number` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
INSERT INTO `station` VALUES (1,'2026-05-13 00:00:00.000000','2026-05-13 00:00:00.000000','용인','228000191',37.22425,127.1884667,'명지대','47632');
CREATE TABLE `route_station` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `created_at` datetime(6) NOT NULL,
  `updated_at` datetime(6) NOT NULL,
  `direction` enum('DOWN','UNKNOWN','UP') NOT NULL,
  `station_sequence` int NOT NULL,
  `toward_name` varchar(100) DEFAULT NULL,
  `route_id` bigint NOT NULL,
  `station_id` bigint NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
INSERT INTO `route_station` VALUES (1,'2026-05-13 00:00:00.000000','2026-05-13 00:00:00.000000','UP',1,NULL,1,1);
CREATE TABLE `bus_snapshot` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `created_at` datetime(6) NOT NULL,
  `updated_at` datetime(6) NOT NULL,
  `collected_at` datetime(6) NOT NULL,
  `crowded_code` int DEFAULT NULL,
  `estimated_occupancy_rate` decimal(5,1) DEFAULT NULL,
  `external_route_id` varchar(50) NOT NULL,
  `external_station_id` varchar(50) DEFAULT NULL,
  `external_vehicle_id` varchar(50) DEFAULT NULL,
  `low_plate` int DEFAULT NULL,
  `plate_no` varchar(50) DEFAULT NULL,
  `remaining_seat_count` int DEFAULT NULL,
  `route_type_code` int DEFAULT NULL,
  `state_code` int DEFAULT NULL,
  `station_sequence` int DEFAULT NULL,
  `tagless_code` int DEFAULT NULL,
  `route_id` bigint NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
INSERT INTO `bus_snapshot` VALUES (1,'2026-05-13 00:00:00.000000','2026-05-13 00:00:00.000000','2026-05-13 01:18:43.108223',3,NULL,'228000174','228000191','228000001',0,'경기70바1234',-1,11,1,1,0,1),(2,'2026-05-13 00:00:00.000000','2026-05-13 00:00:00.000000','2026-05-13 01:18:43.108223',1,20.0,'999','228000191','999',0,'기타',10,11,1,1,0,2);
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            dump_path = Path(temp_dir) / "dump.sql"
            dump_path.write_text(dump, encoding="utf-8")
            init_db(db_path)
            conn = connect(db_path)
            try:
                summary = import_sureing_mysql_dump(conn, dump_path)
                snapshot = conn.execute("SELECT * FROM bus_location_snapshot").fetchone()
                route_station = conn.execute("SELECT * FROM route_station").fetchone()
            finally:
                conn.close()

            self.assertEqual(summary.routes_imported, 1)
            self.assertEqual(summary.route_stations_imported, 1)
            self.assertEqual(summary.snapshots_imported, 1)
            self.assertEqual(summary.snapshots_skipped_non_target, 1)
            self.assertEqual(summary.minus_one_as_zero, 1)
            self.assertEqual(snapshot["remain_seat_cnt"], 0)
            self.assertEqual(snapshot["route_id"], 228000174)
            self.assertEqual(snapshot["collected_at"], "2026-05-12T16:18:43+00:00")
            self.assertEqual(route_station["station_name"], "명지대")


if __name__ == "__main__":
    unittest.main()
