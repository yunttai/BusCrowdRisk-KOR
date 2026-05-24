from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


KEY_ENV_NAMES = (
    "PUBLIC_DATA_SERVICE_KEY",
    "ITS_API_KEY",
    "DATA_GG_SERVICE_KEY",
)


@dataclass(frozen=True)
class Probe:
    name: str
    base_url: str
    key_param: str
    params: dict[str, Any]
    expected_kind: str


PROBES = (
    Probe(
        name="data.go.kr/경기버스 GBIS",
        base_url="https://apis.data.go.kr/6410000/busrouteservice/v2/getBusRouteListv2",
        key_param="serviceKey",
        params={"keyword": "5000", "format": "json"},
        expected_kind="공공데이터포털 경기버스 API 키",
    ),
    Probe(
        name="data.go.kr/기상청 ASOS",
        base_url="https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList",
        key_param="ServiceKey",
        params={
            "pageNo": 1,
            "numOfRows": 1,
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "HR",
            "startDt": "20260522",
            "startHh": "00",
            "endDt": "20260522",
            "endHh": "00",
            "stnIds": "119",
        },
        expected_kind="공공데이터포털 기상청 API 키",
    ),
    Probe(
        name="data.go.kr/에어코리아",
        base_url="https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty",
        key_param="serviceKey",
        params={
            "returnType": "json",
            "numOfRows": 1,
            "pageNo": 1,
            "stationName": "수지",
            "dataTerm": "DAILY",
            "ver": "1.3",
        },
        expected_kind="공공데이터포털 에어코리아 API 키",
    ),
    Probe(
        name="data.go.kr/한국관광공사 TourAPI",
        base_url="https://apis.data.go.kr/B551011/KorService2/searchFestival2",
        key_param="serviceKey",
        params={
            "format": None,
            "MobileOS": "ETC",
            "MobileApp": "busseat-ai",
            "_type": "json",
            "numOfRows": 1,
            "pageNo": 1,
            "eventStartDate": "20260501",
            "eventEndDate": "20260531",
        },
        expected_kind="공공데이터포털 한국관광공사 API 키",
    ),
    Probe(
        name="its.go.kr/국가교통정보센터 교통소통",
        base_url="https://openapi.its.go.kr:9443/trafficInfo",
        key_param="apiKey",
        params={
            "type": "all",
            "minX": "126.8",
            "maxX": "127.9",
            "minY": "37.0",
            "maxY": "37.8",
            "getType": "json",
        },
        expected_kind="ITS 국가교통정보센터 API 키",
    ),
    Probe(
        name="data.gg.go.kr/경기데이터드림",
        base_url="https://openapi.gg.go.kr/RegionMnyPublctUse",
        key_param="KEY",
        params={"Type": "json", "pIndex": 1, "pSize": 1},
        expected_kind="경기데이터드림 API 키",
    ),
)


def collect_key_candidates(env_names: tuple[str, ...] = KEY_ENV_NAMES) -> list[dict[str, Any]]:
    seen_values: set[str] = set()
    candidates = []
    for env_name in env_names:
        value = os.getenv(env_name, "").strip()
        if not value or value in seen_values:
            continue
        seen_values.add(value)
        candidates.append(
            {
                "envName": env_name,
                "keyLength": len(value),
                "keyFingerprint": hashlib.sha256(value.encode("utf-8")).hexdigest()[:10],
                "value": value,
            }
        )
    return candidates


def identify_api_keys(timeout: int = 10) -> dict[str, Any]:
    results = []
    for candidate in collect_key_candidates():
        value = candidate["value"]
        probes = [_run_probe(probe, value, timeout=timeout) for probe in PROBES]
        matched = sorted({probe["expectedKind"] for probe in probes if probe["ok"]})
        results.append(
            {
                "envName": candidate["envName"],
                "keyLength": candidate["keyLength"],
                "keyFingerprint": candidate["keyFingerprint"],
                "matchedKinds": matched,
                "probes": probes,
            }
        )

    return {
        "checkedEnvNames": list(KEY_ENV_NAMES),
        "candidateCount": len(results),
        "note": "키 원문은 출력하지 않습니다. keyFingerprint는 같은 키 중복 여부 확인용 SHA-256 앞 10자리입니다.",
        "results": results,
    }


def _run_probe(probe: Probe, service_key: str, timeout: int) -> dict[str, Any]:
    url = _build_url(probe, service_key)
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "busseat-ai/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            raw = response.read(4096).decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - network dependent
        return {
            "name": probe.name,
            "expectedKind": probe.expected_kind,
            "ok": False,
            "status": "request_failed",
            "reason": str(exc)[:180],
        }

    ok, reason = _classify_response(raw, status)
    return {
        "name": probe.name,
        "expectedKind": probe.expected_kind,
        "ok": ok,
        "status": status,
        "reason": reason,
    }


def _build_url(probe: Probe, service_key: str) -> str:
    params = urlencode({k: v for k, v in probe.params.items() if v is not None})
    encoded_key = quote(service_key, safe="%")
    separator = "&" if params else ""
    return f"{probe.base_url}?{probe.key_param}={encoded_key}{separator}{params}"


def _classify_response(raw: str, http_status: int) -> tuple[bool, str]:
    text = raw.strip()
    upper = text.upper()
    if http_status >= 400:
        return False, f"HTTP {http_status}"

    if any(
        marker in upper
        for marker in (
            "SERVICE_KEY_IS_NOT_REGISTERED",
            "SERVICE_KEY_IS_NOT_REGISTERED_ERROR",
            "INVALID_SERVICE_KEY",
            "INVALID_REQUEST_PARAMETER_ERROR",
            "AUTHENTICATION_FAILED",
            "인증키",
            "등록되지",
            "유효하지",
        )
    ):
        return False, _short_reason(text)

    parsed = _try_json(text)
    if parsed is not None:
        json_reason = _classify_json(parsed)
        if json_reason:
            return json_reason

    if "INFO-000" in upper or "NORMAL SERVICE" in upper or "정상" in text:
        return True, "정상 응답"
    if '"RESULTCODE":"0"' in upper or '"RESULTCODE": "0"' in upper:
        return True, "정상 응답"
    if "<RESULTCODE>0</RESULTCODE>" in upper:
        return True, "정상 응답"
    if '"RESULTCODE":"00"' in upper or '"RESULTCODE": "00"' in upper:
        return True, "정상 응답"

    return False, _short_reason(text)


def _classify_json(value: Any) -> tuple[bool, str] | None:
    dumped = json.dumps(value, ensure_ascii=False)
    upper = dumped.upper()
    if "INFO-000" in upper or "정상" in dumped:
        return True, "정상 응답"
    if any(marker in upper for marker in ("SERVICE_KEY_IS_NOT_REGISTERED", "INVALID_SERVICE_KEY", "ERROR-300")):
        return False, _short_reason(dumped)

    response = value.get("response") if isinstance(value, dict) else None
    if isinstance(response, dict):
        header = response.get("header") or response.get("msgHeader") or {}
        result_code = str(header.get("resultCode") or header.get("resultCd") or "")
        result_message = str(header.get("resultMsg") or header.get("resultMessage") or "")
        if result_code in {"0", "00", "0000"}:
            return True, result_message or "정상 응답"
        if result_code:
            return False, f"{result_code}: {result_message}"[:180]

    header = value.get("header") if isinstance(value, dict) else None
    if isinstance(header, dict):
        result_code = str(header.get("resultCode") or "")
        result_message = str(header.get("resultMsg") or "")
        if result_code == "0":
            return True, result_message or "정상 응답"
        if result_code:
            return False, f"{result_code}: {result_message}"[:180]

    return None


def _try_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _short_reason(text: str) -> str:
    compact = " ".join(text.replace("\n", " ").replace("\r", " ").split())
    return compact[:180] if compact else "응답 분류 실패"
