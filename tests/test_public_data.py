import unittest

from busseat_ai.clients.public_data import build_public_data_url, extract_items


class PublicDataTests(unittest.TestCase):
    def test_extract_gbis_list_response(self):
        payload = {
            "response": {
                "msgHeader": {"resultCode": 0},
                "msgBody": {"busArrivalList": [{"routeId": 1}, {"routeId": 2}]},
            }
        }

        self.assertEqual(extract_items(payload, "busArrivalList"), [{"routeId": 1}, {"routeId": 2}])

    def test_extract_gbis_singleton_response(self):
        payload = {
            "response": {
                "msgHeader": {"resultCode": "0"},
                "msgBody": {"busArrivalList": {"routeId": 1}},
            }
        }

        self.assertEqual(extract_items(payload, "busArrivalList"), [{"routeId": 1}])

    def test_build_url_preserves_encoded_key(self):
        url = build_public_data_url("https://example.test/api", "abc%2Fdef", {"keyword": "명지대"})

        self.assertIn("serviceKey=abc%2Fdef", url)
        self.assertIn("keyword=", url)
        self.assertIn("format=json", url)

    def test_gbis_no_result_code_is_empty_list(self):
        payload = {
            "response": {
                "msgHeader": {"resultCode": "4", "resultMessage": "결과가 존재하지 않습니다."},
                "msgBody": {},
            }
        }

        self.assertEqual(extract_items(payload, "busRouteList"), [])


if __name__ == "__main__":
    unittest.main()
