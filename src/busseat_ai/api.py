from __future__ import annotations

from fastapi import FastAPI, HTTPException

from busseat_ai.clients.gbis import GbisClient
from busseat_ai.cli import _live_feature_rows_from_arrivals, _scheduled_proxy_rows
from busseat_ai.config import load_settings, require_service_key
from busseat_ai.modeling import predict_feature_rows_with_artifact
from busseat_ai.preprocessing import normalize_route_name
from busseat_ai.services.weather import lonlat_to_kma_grid
from busseat_ai.storage.database import connect, init_db, insert_arrival_snapshots


app = FastAPI(title="BusSeat AI", version="0.1.0")


def client() -> GbisClient:
    settings = load_settings()
    try:
        key = require_service_key(settings)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return GbisClient(key, timeout=settings.timeout_seconds)


@app.get("/health")
def health() -> dict[str, object]:
    settings = load_settings()
    return {"ok": True, "hasServiceKey": settings.has_service_key, "dbPath": str(settings.db_path)}


@app.get("/stations/search")
def search_stations(keyword: str) -> list[dict]:
    return client().search_stations(keyword)


@app.get("/stations/nearby")
def nearby_stations(x: float, y: float) -> list[dict]:
    return client().nearby_stations(x, y)


@app.get("/routes/search")
def search_routes(keyword: str) -> list[dict]:
    return client().search_routes(keyword)


@app.get("/routes/{route_id}/stations")
def route_stations(route_id: str) -> list[dict]:
    return client().route_stations(route_id)


@app.get("/arrivals/{station_id}")
def arrivals(station_id: str, save: bool = False) -> dict[str, object]:
    settings = load_settings()
    items = client().arrivals(station_id)
    saved = 0
    if save:
        init_db(settings.db_path)
        with connect(settings.db_path) as conn:
            saved = insert_arrival_snapshots(conn, items)
    return {"saved": saved, "items": items}


@app.get("/locations/{route_id}")
def locations(route_id: str) -> list[dict]:
    return client().locations(route_id)


@app.get("/predict/live-risk")
def predict_live_risk(
    station_id: str,
    route_id: str | None = None,
    route_name: str | None = None,
    at: str | None = None,
    artifact: str = "data/risk_model_lightgbm.pkl",
) -> dict[str, object]:
    settings = load_settings()
    gbis = client()
    items = gbis.arrivals(station_id)
    if route_id:
        items = [item for item in items if str(item.get("routeId")) == str(route_id)]
    if route_name:
        canonical = normalize_route_name(route_name)
        items = [item for item in items if normalize_route_name(item.get("routeName")) == canonical]

    init_db(settings.db_path)
    with connect(settings.db_path) as conn:
        live_rows = _live_feature_rows_from_arrivals(conn, gbis, items, requested_station_id=station_id)
        scheduled_rows = []
        if not live_rows:
            scheduled_rows = _scheduled_proxy_rows(
                conn,
                gbis,
                items,
                requested_station_id=station_id,
                route_id=route_id,
                route_name=route_name,
                at=at,
            )

    prediction_rows = live_rows or scheduled_rows
    result = predict_feature_rows_with_artifact(artifact, prediction_rows)
    if live_rows:
        result["predictionMode"] = "live_vehicle"
    elif scheduled_rows:
        result["predictionMode"] = "schedule_prior"
        result["message"] = "실시간 차량 정보가 없어 시간표/과거 이력 기반 proxy 차량 row를 같은 LightGBM artifact에 넣어 예측했습니다."
    else:
        result["predictionMode"] = "none"
        result["message"] = "GBIS 실시간 응답과 시간표 fallback 모두에서 예측 가능한 row를 만들지 못했습니다."
    return result


@app.get("/weather/grid")
def weather_grid(lon: float, lat: float) -> dict[str, float | int]:
    grid = lonlat_to_kma_grid(lon, lat)
    return {"lon": lon, "lat": lat, "nx": grid.nx, "ny": grid.ny}
