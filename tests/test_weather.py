import unittest

from busseat_ai.services.weather import lonlat_to_kma_grid


class WeatherGridTests(unittest.TestCase):
    def test_seoul_city_hall_grid(self):
        grid = lonlat_to_kma_grid(126.9780, 37.5665)

        self.assertEqual((grid.nx, grid.ny), (60, 127))

    def test_gangnam_sample_grid_is_reasonable(self):
        grid = lonlat_to_kma_grid(127.0284667, 37.49545)

        self.assertTrue(60 <= grid.nx <= 62)
        self.assertTrue(124 <= grid.ny <= 126)


if __name__ == "__main__":
    unittest.main()
