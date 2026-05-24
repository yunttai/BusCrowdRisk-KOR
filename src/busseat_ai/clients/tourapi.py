from __future__ import annotations

from typing import Any
from urllib.parse import quote

from busseat_ai.clients.public_data import extract_items, get_json


class TourApiClient:
    """한국관광공사 국문 관광정보 서비스 client."""

    FESTIVAL_URL = "https://apis.data.go.kr/B551011/KorService2/searchFestival2"

    def __init__(self, service_key: str, timeout: int = 10) -> None:
        self.service_key = service_key
        self.timeout = timeout

    def festivals(
        self,
        start_date: str,
        end_date: str | None = None,
        *,
        area_code: str | int | None = None,
        sigungu_code: str | int | None = None,
        l_dong_regn_cd: str | int | None = None,
        l_dong_signgu_cd: str | int | None = None,
        page_no: int = 1,
        num_of_rows: int = 100,
    ) -> list[dict[str, Any]]:
        payload = get_json(
            self.FESTIVAL_URL,
            quote(self.service_key, safe=""),
            {
                "format": None,
                "MobileOS": "ETC",
                "MobileApp": "busseat-ai",
                "_type": "json",
                "numOfRows": num_of_rows,
                "pageNo": page_no,
                "eventStartDate": start_date.replace("-", ""),
                "eventEndDate": end_date.replace("-", "") if end_date else None,
                "areaCode": area_code,
                "sigunguCode": sigungu_code,
                "lDongRegnCd": l_dong_regn_cd,
                "lDongSignguCd": l_dong_signgu_cd,
            },
            timeout=self.timeout,
        )
        return extract_items(payload)
