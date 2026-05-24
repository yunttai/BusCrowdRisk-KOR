from __future__ import annotations

from typing import Any

from busseat_ai.clients.public_data import extract_item, extract_items, get_json


class GbisClient:
    BASE = "https://apis.data.go.kr/6410000"

    ENDPOINTS = {
        "arrival_list": f"{BASE}/busarrivalservice/v2/getBusArrivalListv2",
        "arrival_item": f"{BASE}/busarrivalservice/v2/getBusArrivalItemv2",
        "location_list": f"{BASE}/buslocationservice/v2/getBusLocationListv2",
        "route_list": f"{BASE}/busrouteservice/v2/getBusRouteListv2",
        "route_info": f"{BASE}/busrouteservice/v2/getBusRouteInfoItemv2",
        "route_stations": f"{BASE}/busrouteservice/v2/getBusRouteStationListv2",
        "route_line": f"{BASE}/busrouteservice/v2/getBusRouteLineListv2",
        "station_list": f"{BASE}/busstationservice/v2/getBusStationListv2",
        "station_around": f"{BASE}/busstationservice/v2/getBusStationAroundListv2",
        "station_routes": f"{BASE}/busstationservice/v2/getBusStationViaRouteListv2",
        "station_info": f"{BASE}/busstationservice/v2/getBusStationInfoItemv2",
        "base_info": f"{BASE}/baseinfoservice/v2/getBaseInfoItemv2",
    }

    def __init__(self, service_key: str, timeout: int = 10) -> None:
        self.service_key = service_key
        self.timeout = timeout

    def search_stations(self, keyword: str) -> list[dict[str, Any]]:
        payload = self._get("station_list", {"keyword": keyword})
        return extract_items(payload, "busStationList")

    def nearby_stations(self, x: float, y: float) -> list[dict[str, Any]]:
        payload = self._get("station_around", {"x": x, "y": y})
        return extract_items(payload, "busStationList")

    def station_routes(self, station_id: str | int) -> list[dict[str, Any]]:
        payload = self._get("station_routes", {"stationId": station_id})
        return extract_items(payload, "busRouteList")

    def station_info(self, station_id: str | int) -> dict[str, Any] | None:
        payload = self._get("station_info", {"stationId": station_id})
        return extract_item(payload, "busStationInfoItem")

    def search_routes(self, keyword: str) -> list[dict[str, Any]]:
        payload = self._get("route_list", {"keyword": keyword})
        return extract_items(payload, "busRouteList")

    def route_info(self, route_id: str | int) -> dict[str, Any] | None:
        payload = self._get("route_info", {"routeId": route_id})
        return extract_item(payload, "busRouteInfoItem")

    def route_stations(self, route_id: str | int) -> list[dict[str, Any]]:
        payload = self._get("route_stations", {"routeId": route_id})
        return extract_items(payload, "busRouteStationList")

    def route_line(self, route_id: str | int) -> list[dict[str, Any]]:
        payload = self._get("route_line", {"routeId": route_id})
        return extract_items(payload, "busRouteLineList")

    def arrivals(self, station_id: str | int) -> list[dict[str, Any]]:
        payload = self._get("arrival_list", {"stationId": station_id})
        return extract_items(payload, "busArrivalList")

    def arrival_item(self, station_id: str | int, route_id: str | int, sta_order: str | int) -> dict[str, Any] | None:
        payload = self._get("arrival_item", {"stationId": station_id, "routeId": route_id, "staOrder": sta_order})
        return extract_item(payload, "busArrivalItem")

    def locations(self, route_id: str | int) -> list[dict[str, Any]]:
        payload = self._get("location_list", {"routeId": route_id})
        return extract_items(payload, "busLocationList")

    def base_info(self) -> dict[str, Any] | None:
        payload = self._get("base_info", {})
        return extract_item(payload, "baseInfoItem")

    def _get(self, endpoint_name: str, params: dict[str, Any]) -> dict[str, Any]:
        return get_json(self.ENDPOINTS[endpoint_name], self.service_key, params, timeout=self.timeout)
