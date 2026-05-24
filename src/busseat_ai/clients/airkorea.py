from __future__ import annotations

from typing import Any

from busseat_ai.clients.public_data import extract_items, get_json


class AirKoreaClient:
    """한국환경공단 에어코리아 대기오염정보 client."""

    MEASURE_URL = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"

    def __init__(self, service_key: str, timeout: int = 10) -> None:
        self.service_key = service_key
        self.timeout = timeout

    def station_measurements(
        self,
        station_name: str,
        *,
        data_term: str = "DAILY",
        page_no: int = 1,
        num_of_rows: int = 100,
        version: str = "1.3",
    ) -> list[dict[str, Any]]:
        payload = get_json(
            self.MEASURE_URL,
            self.service_key,
            {
                "returnType": "json",
                "numOfRows": num_of_rows,
                "pageNo": page_no,
                "stationName": station_name,
                "dataTerm": data_term,
                "ver": version,
            },
            timeout=self.timeout,
        )
        rows = extract_items(payload)
        for row in rows:
            row.setdefault("stationName", station_name)
        return rows
