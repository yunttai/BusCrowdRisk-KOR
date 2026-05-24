from __future__ import annotations

from typing import Any

from busseat_ai.clients.public_data import extract_items, get_json


class KasiClient:
    """한국천문연구원 특일 정보 client."""

    HOLIDAY_URL = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"

    def __init__(self, service_key: str, timeout: int = 10) -> None:
        self.service_key = service_key
        self.timeout = timeout

    def holidays(self, year: int | str, month: int | str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "pageNo": 1,
            "numOfRows": 100,
            "solYear": str(year),
            "_type": "json",
        }
        if month is not None:
            params["solMonth"] = str(month).zfill(2)
        payload = get_json(
            self.HOLIDAY_URL,
            self.service_key,
            params,
            timeout=self.timeout,
            service_key_param="ServiceKey",
        )
        return extract_items(payload)
