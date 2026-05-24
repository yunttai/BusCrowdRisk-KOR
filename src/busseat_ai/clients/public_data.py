from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class PublicDataError(RuntimeError):
    pass


def build_public_data_url(
    base_url: str,
    service_key: str,
    params: dict[str, Any] | None = None,
    service_key_param: str = "serviceKey",
) -> str:
    params = dict(params or {})
    params.setdefault("format", "json")

    encoded_params = urlencode({k: v for k, v in params.items() if v is not None})
    # 공공데이터포털 키는 encoded/decoded 형태가 섞여 쓰인다. 기존 % 인코딩은 보존한다.
    encoded_key = quote(service_key, safe="%")
    separator = "&" if encoded_params else ""
    return f"{base_url}?{service_key_param}={encoded_key}{separator}{encoded_params}"


def get_json(
    base_url: str,
    service_key: str,
    params: dict[str, Any] | None = None,
    timeout: int = 10,
    service_key_param: str = "serviceKey",
) -> dict[str, Any]:
    url = build_public_data_url(base_url, service_key, params, service_key_param=service_key_param)
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "busseat-ai/0.1"})

    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except Exception as exc:  # pragma: no cover - network dependent
        raise PublicDataError(f"공공데이터 API 호출 실패: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PublicDataError(f"JSON 응답 파싱 실패: {raw[:300]}") from exc


def extract_items(payload: dict[str, Any], item_key: str | None = None) -> list[dict[str, Any]]:
    """Normalize common data.go.kr/GBIS response shapes into a list of dicts."""
    response = payload.get("response", payload)

    header = response.get("msgHeader") or response.get("header") or {}
    result_code = header.get("resultCode") or header.get("resultCd")
    if result_code is not None and str(result_code) not in {"0", "00", "0000"}:
        message = header.get("resultMessage") or header.get("resultMsg") or "unknown error"
        if str(result_code) == "4" and "결과" in str(message) and "존재" in str(message):
            return []
        raise PublicDataError(f"공공데이터 API 오류 {result_code}: {message}")

    body = response.get("msgBody") or response.get("body") or response

    if item_key and item_key in body:
        return _as_list(body[item_key])

    if "items" in body:
        items = body["items"]
        if isinstance(items, dict) and "item" in items:
            return _as_list(items["item"])
        return _as_list(items)

    if "item" in body:
        return _as_list(body["item"])

    if item_key:
        return []

    return _as_list(body)


def extract_item(payload: dict[str, Any], item_key: str) -> dict[str, Any] | None:
    items = extract_items(payload, item_key)
    return items[0] if items else None


def _as_list(value: Any) -> list[dict[str, Any]]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []
