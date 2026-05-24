from __future__ import annotations

from typing import Any

from busseat_ai.clients.public_data import extract_items, get_json


class KmaClient:
    """KMA hourly observation client.

    Uses 기상청_지상(종관, ASOS) 시간자료 조회서비스.
    """

    ASOS_HOURLY_URL = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"

    def __init__(self, service_key: str, timeout: int = 10) -> None:
        self.service_key = service_key
        self.timeout = timeout

    def asos_hourly(
        self,
        station_id: str | int,
        start_date: str,
        end_date: str,
        *,
        start_hour: str = "00",
        end_hour: str = "23",
        page_no: int = 1,
        num_of_rows: int = 999,
    ) -> list[dict[str, Any]]:
        payload = get_json(
            self.ASOS_HOURLY_URL,
            self.service_key,
            {
                "pageNo": page_no,
                "numOfRows": num_of_rows,
                "dataType": "JSON",
                "dataCd": "ASOS",
                "dateCd": "HR",
                "startDt": start_date.replace("-", ""),
                "startHh": start_hour.zfill(2),
                "endDt": end_date.replace("-", ""),
                "endHh": end_hour.zfill(2),
                "stnIds": station_id,
            },
            timeout=self.timeout,
            service_key_param="ServiceKey",
        )
        return extract_items(payload)
