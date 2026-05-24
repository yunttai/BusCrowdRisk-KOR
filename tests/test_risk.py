import unittest

from busseat_ai.services.risk import calculate_seat_scarcity_score, estimate_capacity


class RiskUtilityTests(unittest.TestCase):
    def test_seat_scarcity_score_is_log_scaled(self):
        one = calculate_seat_scarcity_score(1, 45)
        two = calculate_seat_scarcity_score(2, 45)
        three = calculate_seat_scarcity_score(3, 45)
        ten = calculate_seat_scarcity_score(10, 45)
        eleven = calculate_seat_scarcity_score(11, 45)

        self.assertEqual(one, 97)
        self.assertEqual(two, 93)
        self.assertEqual(three, 89)
        self.assertGreater(one, two)
        self.assertGreater(two, three)
        self.assertGreater(three - ten, ten - eleven)

    def test_zero_remaining_seats_is_max_scarcity(self):
        self.assertEqual(calculate_seat_scarcity_score(0, 45), 100)

    def test_capacity_prefers_observed_value(self):
        self.assertEqual(estimate_capacity(11, observed_capacity=39), 39)

    def test_capacity_uses_route_type_default(self):
        self.assertEqual(estimate_capacity(11), 45)
        self.assertIsNone(estimate_capacity(13))


if __name__ == "__main__":
    unittest.main()
