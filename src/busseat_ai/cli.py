from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from busseat_ai.clients.airkorea import AirKoreaClient
from busseat_ai.clients.kma import KmaClient
from busseat_ai.clients.kasi import KasiClient
from busseat_ai.clients.gbis import GbisClient
from busseat_ai.clients.key_probe import identify_api_keys
from busseat_ai.clients.tourapi import TourApiClient
from busseat_ai.config import load_settings, require_service_key
from busseat_ai.demand import build_expected_boardings, import_station_demand_csv
from busseat_ai.external_context import (
    KST,
    build_live_model_feature_row,
    build_model_hourly_features,
    import_external_csv,
    infer_snapshot_date_range,
    insert_air_quality_hourly,
    insert_events_daily,
    insert_holidays,
    insert_weather_hourly,
)
from busseat_ai.importers.sureing_mysql_dump import import_sureing_mysql_dump
from busseat_ai.modeling import (
    backtest_schedule_prior,
    evaluate_historical_rate_baseline,
    export_training_dataset_csv,
    predict_feature_risk_with_artifact,
    predict_feature_rows_with_artifact,
    run_model_diagnostics,
    train_lightgbm_risk_artifact,
)
from busseat_ai.preprocessing import (
    TARGET_ROUTE_NAMES,
    TARGET_ROUTE_QUERIES,
    count_route_stations,
    get_target_routes,
    is_target_route,
    materialize_location_features,
    normalize_route_name,
    upsert_route_stations,
    upsert_target_routes,
)
from busseat_ai.quality import build_data_quality_report
from busseat_ai.services.risk import (
    CROWDED_LABELS,
    calculate_seat_scarcity_score,
    estimate_capacity,
)
from busseat_ai.services.weather import lonlat_to_kma_grid
from busseat_ai.storage.database import connect, init_db, insert_arrival_snapshots, insert_location_snapshots, recent_observed_capacity
from busseat_ai.targets import build_target_labels


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()

    try:
        if args.command == "init-db":
            init_db(settings.db_path)
            print_json({"ok": True, "dbPath": str(settings.db_path)})
            return 0

        if args.command == "kma-grid":
            grid = lonlat_to_kma_grid(args.lon, args.lat)
            print_json({"lon": args.lon, "lat": args.lat, "nx": grid.nx, "ny": grid.ny})
            return 0

        if args.command == "identify-api-keys":
            print_json(identify_api_keys(timeout=settings.timeout_seconds))
            return 0

        if args.command == "import-external-csv":
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                imported = import_external_csv(conn, args.kind, args.path)
            print_json({"imported": imported, "kind": args.kind, "path": args.path})
            return 0

        if args.command == "build-model-hourly-features":
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                summary = build_model_hourly_features(conn, args.export_csv)
            print_json(summary.to_dict())
            return 0

        if args.command == "data-quality":
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                report = build_data_quality_report(conn)
            print_json(report.to_dict())
            return 0

        if args.command == "build-target-labels":
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                summary = build_target_labels(conn)
            print_json(summary.to_dict())
            return 0

        if args.command == "import-sureing-dump":
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                summary = import_sureing_mysql_dump(
                    conn,
                    args.path,
                    target_only=not args.all_routes,
                    minus_one_as_zero=not args.keep_minus_one,
                )
            print_json(summary.to_dict())
            return 0

        if args.command == "import-station-demand-csv":
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                imported = import_station_demand_csv(conn, args.path)
            print_json({"imported": imported, "path": args.path})
            return 0

        if args.command == "build-expected-boardings":
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                summary = build_expected_boardings(conn, args.export_csv)
            print_json(summary.to_dict())
            return 0

        if args.command == "export-training-dataset":
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                summary = export_training_dataset_csv(conn, args.export_csv, exact_weather_only=args.exact_weather_only)
            print_json(summary.to_dict())
            return 0

        if args.command == "evaluate-baseline":
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                summary = evaluate_historical_rate_baseline(
                    conn,
                    target_column=args.target,
                    test_ratio=args.test_ratio,
                    threshold=args.threshold,
                    output_json_path=args.output_json,
                    exact_weather_only=args.exact_weather_only,
                )
            print_json(summary.to_dict())
            return 0

        if args.command == "fetch-holidays":
            init_db(settings.db_path)
            client = KasiClient(require_service_key(settings), timeout=settings.timeout_seconds)
            years = range(args.start_year, args.end_year + 1)
            total = 0
            with connect(settings.db_path) as conn:
                for year in years:
                    rows = client.holidays(year)
                    total += insert_holidays(conn, rows, source=f"KASI:{year}")
            print_json({"inserted": total, "startYear": args.start_year, "endYear": args.end_year})
            return 0

        if args.command == "fetch-air-quality-hourly":
            init_db(settings.db_path)
            client = AirKoreaClient(require_service_key(settings), timeout=settings.timeout_seconds)
            total = 0
            per_station = []
            with connect(settings.db_path) as conn:
                for station_name in (args.station_name or ["수지"]):
                    rows = client.station_measurements(
                        station_name,
                        data_term=args.data_term,
                        num_of_rows=args.num_of_rows,
                    )
                    inserted = insert_air_quality_hourly(conn, rows, source=f"AIRKOREA:{station_name}")
                    total += inserted
                    per_station.append({"stationName": station_name, "fetched": len(rows), "inserted": inserted})
            print_json({"inserted": total, "stations": per_station})
            return 0

        if args.command == "fetch-events-daily":
            init_db(settings.db_path)
            client = TourApiClient(require_service_key(settings), timeout=settings.timeout_seconds)
            items = client.festivals(
                args.start_date,
                args.end_date,
                area_code=args.area_code,
                sigungu_code=args.sigungu_code,
                l_dong_regn_cd=args.ldong_regn_cd,
                l_dong_signgu_cd=args.ldong_signgu_cd,
                num_of_rows=args.num_of_rows,
            )
            rows = _tour_event_daily_rows(items, args.area_key, args.start_date, args.end_date)
            with connect(settings.db_path) as conn:
                inserted = insert_events_daily(conn, rows, source=f"TOURAPI:{args.area_key}")
            print_json({"fetched": len(items), "expandedDailyRows": len(rows), "inserted": inserted, "areaKey": args.area_key})
            return 0

        if args.command == "diagnose-model":
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                report = run_model_diagnostics(
                    conn,
                    target_column=args.target,
                    test_ratio=args.test_ratio,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    l2=args.l2,
                    threshold=args.threshold,
                    output_json_path=args.output_json,
                    output_md_path=args.output_md,
                    exact_weather_only=args.exact_weather_only,
                )
            print_json(report)
            return 0

        if args.command == "backtest-schedule-prior":
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                report = backtest_schedule_prior(
                    conn,
                    target_column=args.target,
                    test_ratio=args.test_ratio,
                    threshold=args.threshold,
                    exact_weather_only=args.exact_weather_only,
                    max_test_rows=args.max_test_rows,
                    output_json_path=args.output_json,
                    output_md_path=args.output_md,
                )
            print_json(report)
            return 0

        if args.command == "train-lightgbm-artifact":
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                summary = train_lightgbm_risk_artifact(
                    conn,
                    args.output_artifact,
                    target_column=args.target,
                    threshold=args.threshold,
                    exact_weather_only=args.exact_weather_only,
                    output_json_path=args.output_json,
                )
            print_json(summary)
            return 0

        if args.command == "predict-feature-risk":
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                result = predict_feature_risk_with_artifact(
                    conn,
                    args.artifact,
                    limit=args.limit,
                    snapshot_id=args.snapshot_id,
                    at=args.at,
                    route_name=args.route_name,
                    station_id=args.station_id,
                    station_name=args.station_name,
                    remain_seat_cnt=args.remain_seat_cnt,
                    output_json_path=args.output_json,
                )
            print_json(result)
            return 0

        gbis_commands = {
            "search-station",
            "search-route",
            "arrivals",
            "locations",
            "prepare-target-routes",
            "collect-target-locations",
            "collect-loop",
            "validate-api",
            "predict-live-risk",
        }
        client = None
        if args.command in gbis_commands:
            client = GbisClient(require_service_key(settings), timeout=settings.timeout_seconds)

        if args.command == "search-station":
            assert client is not None
            print_json(client.search_stations(args.keyword))
            return 0

        if args.command == "search-route":
            assert client is not None
            print_json(client.search_routes(args.keyword))
            return 0

        if args.command == "arrivals":
            assert client is not None
            items = client.arrivals(args.station_id)
            if args.save:
                init_db(settings.db_path)
                with connect(settings.db_path) as conn:
                    saved = insert_arrival_snapshots(conn, items)
                print_json({"saved": saved, "items": items})
            else:
                print_json(items)
            return 0

        if args.command == "predict-live-risk":
            assert client is not None
            init_db(settings.db_path)
            arrivals = client.arrivals(args.station_id)
            if args.route_id:
                arrivals = [item for item in arrivals if str(item.get("routeId")) == str(args.route_id)]
            if args.route_name:
                canonical = normalize_route_name(args.route_name)
                arrivals = [item for item in arrivals if normalize_route_name(item.get("routeName")) == canonical]

            with connect(settings.db_path) as conn:
                if args.save_arrivals:
                    insert_arrival_snapshots(conn, arrivals)
                live_rows = _live_feature_rows_from_arrivals(conn, client, arrivals, requested_station_id=args.station_id)
                scheduled_rows = []
                if not live_rows and not args.no_scheduled_fallback:
                    scheduled_rows = _scheduled_proxy_rows(
                        conn,
                        client,
                        arrivals,
                        requested_station_id=args.station_id,
                        route_id=args.route_id,
                        route_name=args.route_name,
                        at=args.at,
                    )
            prediction_rows = live_rows or scheduled_rows
            result = predict_feature_rows_with_artifact(args.artifact, prediction_rows)
            if live_rows:
                result["predictionMode"] = "live_vehicle"
            elif scheduled_rows:
                result["predictionMode"] = "schedule_prior"
                result["message"] = "실시간 차량 정보가 없어 시간표/과거 이력 기반 proxy 차량 row를 같은 LightGBM artifact에 넣어 예측했습니다."
            else:
                result.update(
                    {
                        "predictionMode": "none",
                        "message": "GBIS 실시간 응답과 시간표 fallback 모두에서 예측 가능한 row를 만들지 못했습니다. rows=0은 만차 예측이 아니라 입력 없음입니다.",
                        "liveInput": {
                            "stationId": args.station_id,
                            "routeId": args.route_id,
                            "routeName": args.route_name,
                            "at": args.at,
                            "arrivalsAfterFilter": len(arrivals),
                            **_arrival_diagnostics(arrivals),
                        },
                    }
                )
            if args.output_json:
                output = Path(args.output_json)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                result["exportedJsonPath"] = str(output)
            print_json(result)
            return 0

        if args.command == "locations":
            assert client is not None
            items = client.locations(args.route_id)
            if args.save:
                init_db(settings.db_path)
                with connect(settings.db_path) as conn:
                    saved = insert_location_snapshots(conn, items)
                print_json({"saved": saved, "items": items})
            else:
                print_json(items)
            return 0

        if args.command == "prepare-target-routes":
            assert client is not None
            init_db(settings.db_path)
            candidates: dict[int, dict[str, Any]] = {}
            for query in TARGET_ROUTE_QUERIES:
                for route in client.search_routes(query):
                    route_id = _to_int(route.get("routeId"))
                    if route_id is None:
                        continue
                    if is_target_route(route, region_keyword=args.region_keyword):
                        info = client.route_info(route_id) or {}
                        candidates[route_id] = {**route, **info}

            with connect(settings.db_path) as conn:
                route_count = upsert_target_routes(conn, candidates.values())
                station_count = 0
                routes = get_target_routes(conn)
                for route in routes:
                    stations = client.route_stations(route["route_id"])
                    station_count += upsert_route_stations(
                        conn,
                        route["route_id"],
                        route["canonical_route_name"],
                        stations,
                    )
                station_summary = count_route_stations(conn)

            found_names = {normalize_route_name(route.get("routeName")) for route in candidates.values()}
            missing = [name for name in TARGET_ROUTE_NAMES if name not in found_names]
            print_json(
                {
                    "targetRouteNames": list(TARGET_ROUTE_NAMES),
                    "matchedRoutes": route_count,
                    "storedRouteStations": station_count,
                    "missingRouteNames": missing,
                    "stationSummary": station_summary,
                }
            )
            return 0

        if args.command == "target-summary":
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                print_json({"routes": get_target_routes(conn), "stationSummary": count_route_stations(conn)})
            return 0

        if args.command == "collect-target-locations":
            assert client is not None
            init_db(settings.db_path)
            total = 0
            per_route = []
            with connect(settings.db_path) as conn:
                routes = get_target_routes(conn)
                for route in routes:
                    items = client.locations(route["route_id"])
                    saved = insert_location_snapshots(conn, items)
                    total += saved
                    per_route.append(
                        {
                            "routeId": route["route_id"],
                            "routeName": route["route_name"],
                            "canonicalRouteName": route["canonical_route_name"],
                            "saved": saved,
                        }
                    )
            print_json({"saved": total, "routes": per_route})
            return 0

        if args.command == "collect-loop":
            assert client is not None
            init_db(settings.db_path)
            iteration = 0
            while args.iterations is None or iteration < args.iterations:
                started = time.monotonic()
                total = 0
                per_route = []
                with connect(settings.db_path) as conn:
                    routes = get_target_routes(conn)
                    if not routes:
                        raise RuntimeError("저장된 대상 노선이 없습니다. prepare-target-routes를 먼저 실행하세요.")
                    for route in routes:
                        items = client.locations(route["route_id"])
                        saved = insert_location_snapshots(conn, items)
                        total += saved
                        per_route.append(
                            {
                                "routeId": route["route_id"],
                                "canonicalRouteName": route["canonical_route_name"],
                                "saved": saved,
                            }
                        )
                    preprocess_summary = None
                    model_summary = None
                    if args.preprocess:
                        preprocess_summary = materialize_location_features(conn).to_dict()
                    if args.build_model_features:
                        model_summary = build_model_hourly_features(conn).to_dict()
                iteration += 1
                print_json(
                    {
                        "iteration": iteration,
                        "saved": total,
                        "routes": per_route,
                        "preprocess": preprocess_summary,
                        "modelFeatures": model_summary,
                    }
                )
                if args.iterations is not None and iteration >= args.iterations:
                    break
                elapsed = time.monotonic() - started
                time.sleep(max(0.0, args.interval - elapsed))
            return 0

        if args.command == "validate-api":
            assert client is not None
            candidates: dict[int, dict[str, Any]] = {}
            for query in TARGET_ROUTE_QUERIES:
                for route in client.search_routes(query):
                    route_id = _to_int(route.get("routeId"))
                    if route_id is None:
                        continue
                    if is_target_route(route, region_keyword=args.region_keyword):
                        info = client.route_info(route_id) or {}
                        candidates[route_id] = {**route, **info}

            validations = []
            for route_id, route in sorted(candidates.items(), key=lambda item: normalize_route_name(item[1].get("routeName"))):
                stations = client.route_stations(route_id)
                locations = client.locations(route_id)
                validations.append(
                    {
                        "routeId": route_id,
                        "routeName": route.get("routeName"),
                        "canonicalRouteName": normalize_route_name(route.get("routeName")),
                        "routeTypeCd": route.get("routeTypeCd"),
                        "stationCount": len(stations),
                        "locationCount": len(locations),
                        "hasRemainSeatCnt": any(item.get("remainSeatCnt") not in (None, "") for item in locations),
                        "hasCrowded": any(item.get("crowded") not in (None, "") for item in locations),
                        "hasStationSeq": any(item.get("stationSeq") not in (None, "") for item in locations),
                        "sampleStation": stations[0] if stations else None,
                        "sampleLocation": locations[0] if locations else None,
                    }
                )

            found_names = {normalize_route_name(route.get("routeName")) for route in candidates.values()}
            missing = [name for name in TARGET_ROUTE_NAMES if name not in found_names]
            print_json({"matchedRoutes": len(candidates), "missingRouteNames": missing, "routes": validations})
            return 0

        if args.command == "preprocess-location-features":
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                summary = materialize_location_features(conn, args.export_csv)
            print_json(summary.to_dict())
            return 0

        if args.command == "fetch-weather-hourly":
            kma_client = KmaClient(require_service_key(settings), timeout=settings.timeout_seconds)
            init_db(settings.db_path)
            with connect(settings.db_path) as conn:
                start_date = args.start_date
                end_date = args.end_date
                if not start_date or not end_date:
                    inferred_start, inferred_end = infer_snapshot_date_range(conn)
                    start_date = start_date or inferred_start
                    end_date = end_date or inferred_end
                if not start_date or not end_date:
                    raise RuntimeError("날씨를 받을 날짜 범위가 없습니다. --start-date/--end-date를 주거나 버스 스냅샷을 먼저 수집하세요.")

                total = 0
                for day in _date_range(start_date, end_date):
                    items = kma_client.asos_hourly(args.station_id, day, day)
                    total += insert_weather_hourly(conn, items, source=f"KMA_ASOS:{args.station_id}")
            print_json({"inserted": total, "stationId": args.station_id, "startDate": start_date, "endDate": end_date})
            return 0

    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="busseat", description="BusSeat AI MVP CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="SQLite 스키마를 생성합니다.")

    sub.add_parser("identify-api-keys", help=".env에 넣은 후보 API 키가 data.go.kr/경기데이터드림 중 어디서 동작하는지 확인합니다.")

    station = sub.add_parser("search-station", help="정류소명 또는 정류소번호로 검색합니다.")
    station.add_argument("keyword")

    route = sub.add_parser("search-route", help="노선번호로 검색합니다.")
    route.add_argument("keyword")

    arrivals = sub.add_parser("arrivals", help="정류소 도착정보를 조회합니다.")
    arrivals.add_argument("station_id")
    arrivals.add_argument("--save", action="store_true", help="조회 결과를 SQLite에 저장합니다.")

    locations = sub.add_parser("locations", help="노선 위치정보를 조회합니다.")
    locations.add_argument("route_id")
    locations.add_argument("--save", action="store_true", help="조회 결과를 SQLite에 저장합니다.")

    prepare = sub.add_parser("prepare-target-routes", help="용인 대상 노선 5000/5001A,B/5001-1A,B/5003A,B/5005와 전체 경유정류소를 저장합니다.")
    prepare.add_argument("--region-keyword", help="노선 검색 결과를 특정 지역명으로 추가 필터링합니다. 기본은 필터링하지 않습니다.")

    sub.add_parser("target-summary", help="저장된 대상 노선과 경유정류소 수를 확인합니다.")

    sub.add_parser("collect-target-locations", help="저장된 대상 노선 전체의 현재 위치/잔여좌석 스냅샷을 수집합니다.")

    loop = sub.add_parser("collect-loop", help="저장된 대상 노선 위치정보를 지정 주기로 반복 수집합니다.")
    loop.add_argument("--interval", type=float, default=60.0, help="수집 주기(초). 기본 60초.")
    loop.add_argument("--iterations", type=int, help="반복 횟수. 생략하면 중지할 때까지 계속 실행합니다.")
    loop.add_argument("--preprocess", action="store_true", help="각 반복 후 위치 feature 전처리까지 실행합니다.")
    loop.add_argument("--build-model-features", action="store_true", help="각 반복 후 외부요인 결합 feature까지 갱신합니다.")

    validate = sub.add_parser("validate-api", help="실제 API 응답의 대상 노선/정류소/잔여좌석 필드 제공 여부를 저장 없이 검증합니다.")
    validate.add_argument("--region-keyword", default="용인", help="노선 검색 결과 지역 필터. 기본값은 용인입니다.")

    preprocess = sub.add_parser("preprocess-location-features", help="위치 스냅샷을 요일/시간대/잔여좌석 feature 테이블로 전처리합니다.")
    preprocess.add_argument("--export-csv", help="전처리 결과를 CSV로 내보냅니다.")

    weather = sub.add_parser("fetch-weather-hourly", help="버스 스냅샷 전체 기간 또는 지정 기간의 기상청 ASOS 시간자료를 저장합니다.")
    weather.add_argument("--station-id", default="119", help="기상청 ASOS 지점 ID. 기본값 119는 수원 지점 기준입니다.")
    weather.add_argument("--start-date", help="YYYY-MM-DD")
    weather.add_argument("--end-date", help="YYYY-MM-DD")

    external = sub.add_parser("import-external-csv", help="대기질/교통/공휴일/행사 외부요인 CSV를 표준 테이블에 적재합니다.")
    external.add_argument("kind", choices=["weather", "air", "traffic", "holiday", "event"])
    external.add_argument("path")

    model_features = sub.add_parser("build-model-hourly-features", help="버스 전처리 feature에 시간 단위 외부요인을 결합한 최종 학습 테이블을 만듭니다.")
    model_features.add_argument("--export-csv", help="최종 모델 feature를 CSV로 내보냅니다.")

    sub.add_parser("data-quality", help="최종 feature 테이블의 NULL, imputed 비율, 추천/제외 feature를 보고합니다.")

    sub.add_parser("build-target-labels", help="현재/5분/10분/다음정류소 기준 학습 target 라벨을 생성합니다.")

    holidays = sub.add_parser("fetch-holidays", help="한국천문연구원 특일 정보에서 공휴일을 적재합니다.")
    holidays.add_argument("--start-year", type=int, required=True)
    holidays.add_argument("--end-year", type=int, required=True)

    air = sub.add_parser("fetch-air-quality-hourly", help="에어코리아 측정소별 실시간/최근 대기질을 시간 단위 외부요인으로 적재합니다.")
    air.add_argument("--station-name", action="append", help="대기질 측정소명. 여러 번 지정할 수 있습니다. 기본값은 수지.")
    air.add_argument("--data-term", default="DAILY", help="에어코리아 dataTerm. 기본 DAILY.")
    air.add_argument("--num-of-rows", type=int, default=100)

    events = sub.add_parser("fetch-events-daily", help="한국관광공사 행사정보를 일 단위 event_count로 적재합니다.")
    events.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    events.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    events.add_argument("--area-code", help="TourAPI 구 지역코드. KorService2에서는 법정동 코드 사용을 권장하므로 기본값 없음.")
    events.add_argument("--sigungu-code", help="TourAPI 구 시군구 코드. 모르면 생략.")
    events.add_argument("--ldong-regn-cd", default="41", help="법정동 시도 코드. 기본 41은 경기도.")
    events.add_argument("--ldong-signgu-cd", help="법정동 시군구 코드. 모르면 생략.")
    events.add_argument("--area-key", default="gyeonggi", help="내부 event area key.")
    events.add_argument("--num-of-rows", type=int, default=200)

    sureing = sub.add_parser("import-sureing-dump", help="sureing_occupancy MySQL dump를 현재 SQLite 학습 파이프라인에 적재합니다.")
    sureing.add_argument("path")
    sureing.add_argument("--all-routes", action="store_true", help="기본 대상 5000계열 외 모든 노선도 적재합니다.")
    sureing.add_argument("--keep-minus-one", action="store_true", help="remaining_seat_count=-1을 0으로 바꾸지 않고 그대로 둡니다.")

    demand = sub.add_parser("import-station-demand-csv", help="정류소별 일별 승하차 CSV를 station_demand_daily에 적재합니다.")
    demand.add_argument("path")

    expected = sub.add_parser("build-expected-boardings", help="정류소 일평균 수요와 운행 빈도로 expected_boardings_at_stop을 생성합니다.")
    expected.add_argument("--export-csv", help="생성 결과를 CSV로 내보냅니다.")

    training = sub.add_parser("export-training-dataset", help="최종 feature, target, expected_boardings를 결합한 학습 CSV를 내보냅니다.")
    training.add_argument("--export-csv", required=True)
    training.add_argument("--exact-weather-only", action="store_true", help="weather_imputed=0인 정확 날씨 매칭 row만 내보냅니다.")

    baseline = sub.add_parser("evaluate-baseline", help="시간순 train/test split으로 이력 만차율 베이스라인을 평가합니다.")
    baseline.add_argument("--target", default="target_no_seat_next_station")
    baseline.add_argument("--test-ratio", type=float, default=0.2)
    baseline.add_argument("--threshold", type=float, default=0.5)
    baseline.add_argument("--output-json", help="평가 결과 JSON 저장 경로.")
    baseline.add_argument("--exact-weather-only", action="store_true", help="weather_imputed=0인 정확 날씨 매칭 row만 평가합니다.")

    diagnose = sub.add_parser("diagnose-model", help="Ablation, 노선별/시간대별 성능, 확률 calibration을 한 번에 생성합니다.")
    diagnose.add_argument("--target", default="target_no_seat_next_station")
    diagnose.add_argument("--test-ratio", type=float, default=0.2)
    diagnose.add_argument("--epochs", type=int, default=8)
    diagnose.add_argument("--learning-rate", type=float, default=0.035)
    diagnose.add_argument("--l2", type=float, default=0.0005)
    diagnose.add_argument("--threshold", type=float, default=0.5)
    diagnose.add_argument("--output-json", help="진단 결과 JSON 저장 경로.")
    diagnose.add_argument("--output-md", help="진단 결과 Markdown 저장 경로.")
    diagnose.add_argument("--exact-weather-only", action="store_true", help="weather_imputed=0인 정확 날씨 매칭 row만 진단합니다.")

    schedule_backtest = sub.add_parser(
        "backtest-schedule-prior",
        help="실시간 차량이 없을 때 쓰는 시간표/과거이력 proxy 예측을 시간순 holdout으로 검증합니다.",
    )
    schedule_backtest.add_argument("--target", default="target_no_seat_next_station")
    schedule_backtest.add_argument("--test-ratio", type=float, default=0.2)
    schedule_backtest.add_argument("--threshold", type=float, default=0.9)
    schedule_backtest.add_argument("--max-test-rows", type=int, help="검증할 test row 수 제한. 생략하면 전체 test row를 씁니다.")
    schedule_backtest.add_argument("--output-json", default="data/schedule_prior_backtest.json")
    schedule_backtest.add_argument("--output-md", default="data/schedule_prior_backtest.md")
    schedule_backtest.add_argument("--exact-weather-only", action="store_true", help="weather_imputed=0인 정확 날씨 매칭 row만 검증합니다.")

    artifact = sub.add_parser("train-lightgbm-artifact", help="현재 학습 데이터 전체로 LightGBM 만차위험 예측 artifact를 저장합니다.")
    artifact.add_argument("--target", default="target_no_seat_next_station")
    artifact.add_argument("--threshold", type=float, default=0.9)
    artifact.add_argument("--output-artifact", default="data/risk_model_lightgbm.pkl")
    artifact.add_argument("--output-json", default="data/risk_model_lightgbm_artifact.json")
    artifact.add_argument("--exact-weather-only", action="store_true", help="weather_imputed=0인 정확 날씨 매칭 row만 최종 모델 학습에 사용합니다.")

    feature_predict = sub.add_parser("predict-feature-risk", help="저장된 LightGBM artifact로 model_hourly_features row의 만차확률을 계산합니다.")
    feature_predict.add_argument("--artifact", default="data/risk_model_lightgbm.pkl")
    feature_predict.add_argument("--limit", type=int, default=20)
    feature_predict.add_argument("--snapshot-id", type=int)
    feature_predict.add_argument("--at", help='KST 관측시각. 예: "2026-05-25 04:03:33"')
    feature_predict.add_argument("--route-name", help="canonical route name 예: 5005")
    feature_predict.add_argument("--station-id", type=int)
    feature_predict.add_argument("--station-name", help="정류소명 exact match. 예: 순천향대학병원")
    feature_predict.add_argument("--remain-seat-cnt", type=int, help="잔여좌석 수 exact match.")
    feature_predict.add_argument("--output-json", help="예측 결과 JSON 저장 경로.")

    live_predict = sub.add_parser("predict-live-risk", help="GBIS 실시간 도착/위치정보를 LightGBM feature로 변환해 만차확률을 계산합니다.")
    live_predict.add_argument("station_id", help="사용자가 탈 정류소 ID.")
    live_predict.add_argument("--route-id", help="특정 노선 ID만 예측합니다.")
    live_predict.add_argument("--route-name", help="특정 노선번호만 예측합니다. 예: 5005")
    live_predict.add_argument("--at", help='실시간 차량이 없을 때 사용할 KST 기준 시간표 예측 시각. 예: "2026-05-25 08:00:00"')
    live_predict.add_argument("--artifact", default="data/risk_model_lightgbm.pkl")
    live_predict.add_argument("--save-arrivals", action="store_true", help="조회한 도착정보 원본을 DB에 저장합니다.")
    live_predict.add_argument("--no-scheduled-fallback", action="store_true", help="실시간 차량이 없을 때 시간표/이력 기반 proxy 예측을 하지 않습니다.")
    live_predict.add_argument("--output-json", help="예측 결과 JSON 저장 경로.")

    grid = sub.add_parser("kma-grid", help="WGS84 좌표를 기상청 격자로 변환합니다.")
    grid.add_argument("lon", type=float)
    grid.add_argument("lat", type=float)

    return parser


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _live_feature_rows_from_arrivals(
    conn: sqlite3.Connection,
    client: GbisClient,
    arrivals: list[dict[str, Any]],
    *,
    requested_station_id: str | int,
) -> list[dict[str, Any]]:
    collected_at_kst = datetime.now(KST).isoformat(timespec="seconds")
    route_locations: dict[int, list[dict[str, Any]]] = {}
    rows = []
    for arrival in arrivals:
        route_id = _to_int(arrival.get("routeId"))
        if route_id is None:
            continue
        if route_id not in route_locations:
            try:
                route_locations[route_id] = client.locations(route_id)
            except Exception:
                route_locations[route_id] = []
        location_lookup = _location_lookup(route_locations[route_id])
        route_meta = _target_route_meta(conn, route_id, arrival)

        for slot in (1, 2):
            candidate = _arrival_candidate(arrival, slot)
            if not candidate:
                continue
            matched = None
            if candidate.get("veh_id") is not None:
                matched = location_lookup["veh_id"].get(candidate["veh_id"])
            if matched is None and candidate.get("plate_no"):
                matched = location_lookup["plate_no"].get(candidate["plate_no"])

            station_seq = _first_int(
                matched.get("stationSeq") if matched else None,
                matched.get("stationSeq") if matched else None,
                arrival.get("staOrder"),
            )
            station_id = _first_int(
                matched.get("stationId") if matched else None,
                arrival.get("stationId"),
                requested_station_id,
            )
            station_meta = _route_station_meta(conn, route_id, station_seq, station_id)
            remain = _first_int(candidate.get("remain_seat_cnt"), matched.get("remainSeatCnt") if matched else None, -1)
            crowded = _first_int(candidate.get("crowded"), matched.get("crowded") if matched else None, 0)
            capacity = recent_observed_capacity(conn, route_id, candidate.get("plate_no")) or estimate_capacity(route_meta["route_type_cd"])
            scarcity_score = calculate_seat_scarcity_score(remain, capacity) if remain is not None and remain >= 0 else 0

            live_context = {
                "requested_station_id": _to_int(requested_station_id),
                "requested_station_name": _requested_station_name(conn, route_id, _to_int(requested_station_id)),
                "arrival_slot": slot,
                "predict_time": candidate.get("predict_time"),
                "predict_time_sec": candidate.get("predict_time_sec"),
                "feature_source": "gbis_location_match" if matched else "gbis_arrival_fallback",
                "matched_location": 1 if matched else 0,
            }
            base_row = {
                "snapshot_id": -slot,
                "collected_at_kst": collected_at_kst,
                "route_id": route_id,
                "route_name": route_meta["route_name"],
                "canonical_route_name": route_meta["canonical_route_name"],
                "route_type_cd": route_meta["route_type_cd"],
                "route_type_name": route_meta["route_type_name"],
                "veh_id": candidate.get("veh_id") or _first_int(matched.get("vehId") if matched else None, 0),
                "plate_no": candidate.get("plate_no") or (matched.get("plateNo") if matched else "unknown"),
                "station_id": station_id or 0,
                "station_seq": station_seq or 0,
                "station_name": station_meta["station_name"],
                "mobile_no": station_meta["mobile_no"],
                "x": station_meta["x"],
                "y": station_meta["y"],
                "remain_seat_cnt": remain if remain is not None else -1,
                "crowded": crowded if crowded is not None else 0,
                "crowded_label": CROWDED_LABELS.get(crowded or 0, "unknown"),
                "estimated_capacity": capacity or 45,
                "seat_scarcity_score": scarcity_score or 0,
                "is_no_seat": 1 if remain == 0 else 0,
                "is_low_seat_2": 1 if remain is not None and 0 <= remain <= 2 else 0,
                "is_low_seat_5": 1 if remain is not None and 0 <= remain <= 5 else 0,
            }
            feature = build_live_model_feature_row(conn, base_row)
            feature.update(live_context)
            rows.append(feature)
    return rows


def _scheduled_proxy_rows(
    conn: sqlite3.Connection,
    client: GbisClient,
    arrivals: list[dict[str, Any]],
    *,
    requested_station_id: str | int,
    route_id: str | int | None,
    route_name: str | None,
    at: str | None,
) -> list[dict[str, Any]]:
    at_kst = _parse_cli_kst(at) if at else datetime.now(KST)
    candidates = list(arrivals)
    if not candidates:
        route_ids = [_to_int(route_id)] if route_id else _route_ids_for_name(conn, route_name)
        for current_route_id in [value for value in route_ids if value is not None]:
            route_info = client.route_info(current_route_id) or {}
            candidates.append(
                {
                    **route_info,
                    "routeId": current_route_id,
                    "routeName": route_info.get("routeName") or route_name or current_route_id,
                    "stationId": requested_station_id,
                }
            )
    if not candidates:
        return []

    rows = []
    for index, arrival in enumerate(candidates, start=1):
        current_route_id = _to_int(arrival.get("routeId") or route_id)
        if current_route_id is None:
            continue
        route_info = client.route_info(current_route_id) or {}
        route_meta = _target_route_meta(conn, current_route_id, {**arrival, **route_info})
        station_id = _first_int(arrival.get("stationId"), requested_station_id)
        station_seq = _first_int(
            arrival.get("staOrder"),
            _route_station_seq(conn, current_route_id, station_id),
        )
        station_meta = _route_station_meta(conn, current_route_id, station_seq, station_id)
        schedule_arrival = {
            **arrival,
            "_stationDirection": _route_station_direction(conn, current_route_id, station_seq, station_id),
        }
        schedule = _route_schedule_context(conn, route_info, schedule_arrival, at_kst)
        proxy_at_kst = _parse_cli_kst(schedule["schedule_proxy_at_kst"]) if schedule.get("schedule_proxy_at_kst") else at_kst
        proxy = _historical_station_proxy(
            conn,
            route_meta["canonical_route_name"],
            station_id,
            station_seq,
            proxy_at_kst,
        )
        capacity = (
            proxy.get("proxyCapacity")
            or recent_observed_capacity(conn, current_route_id)
            or estimate_capacity(route_meta["route_type_cd"])
        )
        remain = proxy.get("proxyRemainSeatCnt")
        if remain is None:
            remain = capacity
        crowded = proxy.get("proxyCrowded")
        if crowded is None:
            crowded = 0
        scarcity_score = calculate_seat_scarcity_score(remain, capacity) if remain >= 0 else 0
        base_row = {
            "snapshot_id": -1000 - index,
            "collected_at_kst": proxy_at_kst.isoformat(timespec="seconds"),
            "route_id": current_route_id,
            "route_name": route_meta["route_name"],
            "canonical_route_name": route_meta["canonical_route_name"],
            "route_type_cd": route_meta["route_type_cd"],
            "route_type_name": route_meta["route_type_name"],
            "veh_id": 0,
            "plate_no": "scheduled_proxy",
            "station_id": station_id or 0,
            "station_seq": station_seq or 0,
            "station_name": station_meta["station_name"],
            "mobile_no": station_meta["mobile_no"],
            "x": station_meta["x"],
            "y": station_meta["y"],
            "remain_seat_cnt": int(remain),
            "crowded": int(crowded),
            "crowded_label": CROWDED_LABELS.get(int(crowded), "unknown"),
            "estimated_capacity": int(capacity or 45),
            "seat_scarcity_score": scarcity_score or 0,
            "is_no_seat": 1 if remain == 0 else 0,
            "is_low_seat_2": 1 if 0 <= remain <= 2 else 0,
            "is_low_seat_5": 1 if 0 <= remain <= 5 else 0,
        }
        feature = build_live_model_feature_row(conn, base_row)
        feature.update(
            {
                "prediction_mode": "schedule_prior",
                "feature_source": "schedule_historical_proxy",
                "requested_station_id": _to_int(requested_station_id),
                "requested_station_name": _requested_station_name(conn, current_route_id, _to_int(requested_station_id)),
                "input_at_kst": at_kst.isoformat(timespec="seconds"),
                **schedule,
                **proxy,
            }
        )
        rows.append(feature)
    return rows


def _route_schedule_context(
    conn: sqlite3.Connection,
    route_info: dict[str, Any],
    arrival: dict[str, Any],
    at_kst: datetime,
) -> dict[str, Any]:
    sta_order = _to_int(arrival.get("staOrder"))
    turn_seq = _to_int(arrival.get("turnSeq"))
    direction = _normalize_direction(arrival.get("_stationDirection") or arrival.get("direction"))
    if direction is None:
        direction = "down" if sta_order is not None and turn_seq is not None and sta_order > turn_seq else "up"
    service_date, day_type = _schedule_service_day(conn, route_info, direction, at_kst)
    schedule_fields = _schedule_fields(route_info, direction, day_type)
    first = schedule_fields["first"]
    last = schedule_fields["last"]
    first_min = _time_minutes(first)
    last_min = _time_minutes(last)
    operating = _within_schedule_window(at_kst.hour * 60 + at_kst.minute, first_min, last_min)
    proxy_at = at_kst if operating else _next_scheduled_datetime(at_kst, first_min)
    allocation_hour = proxy_at.hour if proxy_at else at_kst.hour
    peak_alloc = schedule_fields["peak_alloc"]
    npeak_alloc = schedule_fields["npeak_alloc"]
    allocation = peak_alloc if _is_peak_hour(allocation_hour) and peak_alloc else npeak_alloc or peak_alloc
    proxy_basis = "requested_time" if operating else "next_first_time"
    return {
        "schedule_source": "gbis_route_info",
        "schedule_service_date": service_date.isoformat(),
        "schedule_day_type": day_type,
        "schedule_direction": direction,
        "schedule_first_time": str(first) if first not in (None, "") else None,
        "schedule_last_time": str(last) if last not in (None, "") else None,
        "schedule_allocation_minutes": allocation,
        "scheduled_operating": 1 if operating else 0,
        "schedule_proxy_at_kst": proxy_at.isoformat(timespec="seconds") if proxy_at else None,
        "schedule_proxy_time_basis": proxy_basis,
    }


def _route_ids_for_name(conn: sqlite3.Connection, route_name: str | None) -> list[int]:
    if not route_name:
        return []
    canonical = normalize_route_name(route_name)
    rows = conn.execute(
        """
        SELECT route_id
        FROM target_route
        WHERE canonical_route_name = ?
        ORDER BY route_id
        """,
        (canonical,),
    ).fetchall()
    return [_to_int(row["route_id"]) for row in rows if _to_int(row["route_id"]) is not None]


def _route_station_direction(
    conn: sqlite3.Connection,
    route_id: int,
    station_seq: int | None,
    station_id: int | None,
) -> str | None:
    row = None
    if station_seq is not None:
        row = conn.execute(
            """
            SELECT raw_json
            FROM route_station
            WHERE route_id = ? AND station_seq = ?
            """,
            (route_id, station_seq),
        ).fetchone()
    if row is None and station_id is not None:
        row = conn.execute(
            """
            SELECT raw_json
            FROM route_station
            WHERE route_id = ? AND station_id = ?
            ORDER BY station_seq
            LIMIT 1
            """,
            (route_id, station_id),
        ).fetchone()
    if row is None or not row["raw_json"]:
        return None
    try:
        raw = json.loads(row["raw_json"])
    except json.JSONDecodeError:
        return None
    return _normalize_direction(raw.get("direction"))


def _normalize_direction(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"up", "u", "상행"}:
        return "up"
    if text in {"down", "d", "하행"}:
        return "down"
    return None


def _schedule_service_day(
    conn: sqlite3.Connection,
    route_info: dict[str, Any],
    direction: str,
    at_kst: datetime,
) -> tuple[Any, str]:
    target_min = at_kst.hour * 60 + at_kst.minute
    previous = at_kst - timedelta(days=1)
    previous_day_type = _schedule_day_type(conn, previous)
    previous_fields = _schedule_fields(route_info, direction, previous_day_type)
    previous_first = _time_minutes(previous_fields["first"])
    previous_last = _time_minutes(previous_fields["last"])
    if previous_first is not None and previous_last is not None:
        if previous_first > previous_last and target_min <= previous_last:
            return previous.date(), previous_day_type
    return at_kst.date(), _schedule_day_type(conn, at_kst)


def _schedule_fields(route_info: dict[str, Any], direction: str, day_type: str) -> dict[str, Any]:
    direction_title = "Down" if direction == "down" else "Up"
    prefix = {"weekday": "we", "saturday": "sat", "sunday_holiday": "sun"}[day_type]
    return {
        "first": _first_present(
            route_info.get(f"{prefix}{direction_title}FirstTime"),
            route_info.get(f"{direction}FirstTime"),
        ),
        "last": _first_present(
            route_info.get(f"{prefix}{direction_title}LastTime"),
            route_info.get(f"{direction}LastTime"),
        ),
        "peak_alloc": _first_int(route_info.get(f"{prefix}PeekAlloc"), route_info.get("peekAlloc")),
        "npeak_alloc": _first_int(route_info.get(f"{prefix}NPeekAlloc"), route_info.get("nPeekAlloc")),
    }


def _historical_station_proxy(
    conn: sqlite3.Connection,
    canonical_route_name: str,
    station_id: int | None,
    station_seq: int | None,
    at_kst: datetime,
) -> dict[str, Any]:
    is_weekend = 1 if at_kst.weekday() >= 5 else 0
    is_holiday = _is_holiday(conn, at_kst)
    attempts = [
        (
            "route_station_hour_daytype",
            "mf.canonical_route_name = :route AND mf.station_id = :station_id AND mf.hour = :hour AND mf.is_weekend = :is_weekend AND mf.is_holiday = :is_holiday",
        ),
        (
            "route_station_adjacent_hour_daytype",
            "mf.canonical_route_name = :route AND mf.station_id = :station_id AND mf.hour BETWEEN :hour_low AND :hour_high AND mf.is_weekend = :is_weekend AND mf.is_holiday = :is_holiday",
        ),
        (
            "route_station_hour",
            "mf.canonical_route_name = :route AND mf.station_id = :station_id AND mf.hour = :hour",
        ),
        (
            "route_station_daytype",
            "mf.canonical_route_name = :route AND mf.station_id = :station_id AND mf.is_weekend = :is_weekend AND mf.is_holiday = :is_holiday",
        ),
        (
            "route_station",
            "mf.canonical_route_name = :route AND mf.station_id = :station_id",
        ),
        (
            "route_station_seq_hour",
            "mf.canonical_route_name = :route AND mf.station_seq = :station_seq AND mf.hour = :hour",
        ),
        ("route_hour", "mf.canonical_route_name = :route AND mf.hour = :hour"),
        ("route", "mf.canonical_route_name = :route"),
        ("hour", "mf.hour = :hour"),
    ]
    params = {
        "route": canonical_route_name,
        "station_id": station_id,
        "station_seq": station_seq,
        "hour": at_kst.hour,
        "hour_low": max(0, at_kst.hour - 1),
        "hour_high": min(23, at_kst.hour + 1),
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
    }
    for group_name, where in attempts:
        if ":station_id" in where and station_id is None:
            continue
        if ":station_seq" in where and station_seq is None:
            continue
        rows = conn.execute(
            f"""
            SELECT
                mf.remain_seat_cnt,
                mf.crowded,
                mf.estimated_capacity,
                mf.is_no_seat,
                tl.target_no_seat_next_station,
                tl.has_next_station
            FROM model_hourly_features mf
            LEFT JOIN model_target_labels tl ON tl.snapshot_id = mf.snapshot_id
            WHERE {where}
            """,
            params,
        ).fetchall()
        if rows:
            return _proxy_from_rows(group_name, rows)
    return {
        "historical_proxy_group": "none",
        "historical_proxy_rows": 0,
        "historical_next_station_full_rate": None,
        "historical_now_full_rate": None,
        "proxy_remain_source": "default_capacity",
        "proxyRemainSeatCnt": None,
        "proxyCrowded": None,
        "proxyCapacity": None,
    }


def _proxy_from_rows(group_name: str, rows: list[sqlite3.Row]) -> dict[str, Any]:
    remains = [value for value in (_to_int(row["remain_seat_cnt"]) for row in rows) if value is not None and value >= 0]
    crowded_values = [value for value in (_to_int(row["crowded"]) for row in rows) if value is not None]
    capacities = [value for value in (_to_int(row["estimated_capacity"]) for row in rows) if value is not None and value > 0]
    target_rows = [row for row in rows if _to_int(row["has_next_station"]) == 1]
    target_positives = sum(_to_int(row["target_no_seat_next_station"]) or 0 for row in target_rows)
    now_positives = sum(_to_int(row["is_no_seat"]) or 0 for row in rows)
    return {
        "historical_proxy_group": group_name,
        "historical_proxy_rows": len(rows),
        "historical_next_station_full_rate": round(target_positives / len(target_rows), 6) if target_rows else None,
        "historical_now_full_rate": round(now_positives / len(rows), 6) if rows else None,
        "proxy_remain_source": "historical_median",
        "proxyRemainSeatCnt": _median_int(remains),
        "proxyCrowded": _mode_int(crowded_values),
        "proxyCapacity": _median_int(capacities),
    }


def _arrival_candidate(arrival: dict[str, Any], slot: int) -> dict[str, Any] | None:
    plate_no = arrival.get(f"plateNo{slot}")
    predict_time = _to_int(arrival.get(f"predictTime{slot}"))
    predict_time_sec = _to_int(arrival.get(f"predictTimeSec{slot}"))
    if not plate_no and predict_time is None and predict_time_sec is None:
        return None
    return {
        "slot": slot,
        "veh_id": _to_int(arrival.get(f"vehId{slot}")),
        "plate_no": plate_no,
        "remain_seat_cnt": _to_int(arrival.get(f"remainSeatCnt{slot}")),
        "crowded": _to_int(arrival.get(f"crowded{slot}")),
        "predict_time": predict_time,
        "predict_time_sec": predict_time_sec,
    }


def _arrival_diagnostics(arrivals: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = []
    candidate_slots = 0
    for arrival in arrivals:
        slots = []
        for slot in (1, 2):
            has_candidate = _arrival_candidate(arrival, slot) is not None
            if has_candidate:
                candidate_slots += 1
            slots.append(
                {
                    "slot": slot,
                    "hasCandidate": has_candidate,
                    "vehId": arrival.get(f"vehId{slot}") or None,
                    "plateNo": arrival.get(f"plateNo{slot}") or None,
                    "predictTime": arrival.get(f"predictTime{slot}") or None,
                    "remainSeatCnt": arrival.get(f"remainSeatCnt{slot}") or None,
                    "crowded": arrival.get(f"crowded{slot}") or None,
                }
            )
        summaries.append(
            {
                "routeId": arrival.get("routeId"),
                "routeName": arrival.get("routeName"),
                "staOrder": arrival.get("staOrder"),
                "flag": arrival.get("flag"),
                "slots": slots,
            }
        )
    reason = "arrival_vehicle_fields_present"
    if arrivals and candidate_slots == 0:
        reason = "arrival_exists_but_no_vehicle_fields"
    elif not arrivals:
        reason = "no_arrival_after_filter"
    return {
        "arrivalCandidateSlots": candidate_slots,
        "emptyReason": reason,
        "arrivalDiagnostics": summaries,
    }


def _location_lookup(locations: list[dict[str, Any]]) -> dict[str, dict[Any, dict[str, Any]]]:
    by_veh_id = {}
    by_plate_no = {}
    for location in locations:
        veh_id = _to_int(location.get("vehId"))
        plate_no = location.get("plateNo")
        if veh_id is not None:
            by_veh_id[veh_id] = location
        if plate_no:
            by_plate_no[str(plate_no)] = location
    return {"veh_id": by_veh_id, "plate_no": by_plate_no}


def _target_route_meta(conn: sqlite3.Connection, route_id: int, arrival: dict[str, Any]) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT route_name, canonical_route_name, route_type_cd, route_type_name
        FROM target_route
        WHERE route_id = ?
        """,
        (route_id,),
    ).fetchone()
    if row:
        return dict(row)
    route_name = arrival.get("routeName") or "unknown"
    route_type_cd = _to_int(arrival.get("routeTypeCd")) or 0
    return {
        "route_name": route_name,
        "canonical_route_name": normalize_route_name(route_name),
        "route_type_cd": route_type_cd,
        "route_type_name": "unknown",
    }


def _route_station_meta(
    conn: sqlite3.Connection,
    route_id: int,
    station_seq: int | None,
    station_id: int | None,
) -> dict[str, Any]:
    row = None
    if station_seq:
        row = conn.execute(
            """
            SELECT station_id, station_name, mobile_no, x, y
            FROM route_station
            WHERE route_id = ? AND station_seq = ?
            """,
            (route_id, station_seq),
        ).fetchone()
    if row is None and station_id:
        row = conn.execute(
            """
            SELECT station_id, station_name, mobile_no, x, y
            FROM route_station
            WHERE route_id = ? AND station_id = ?
            """,
            (route_id, station_id),
        ).fetchone()
    if row:
        data = dict(row)
        return {
            "station_name": data.get("station_name") or "unknown",
            "mobile_no": data.get("mobile_no") or "unknown",
            "x": data.get("x") or 0.0,
            "y": data.get("y") or 0.0,
        }
    return {
        "station_name": "unknown",
        "mobile_no": "unknown",
        "x": 0.0,
        "y": 0.0,
    }


def _requested_station_name(conn: sqlite3.Connection, route_id: int, station_id: int | None) -> str:
    if station_id is None:
        return "unknown"
    row = conn.execute(
        """
        SELECT station_name
        FROM route_station
        WHERE route_id = ? AND station_id = ?
        """,
        (route_id, station_id),
    ).fetchone()
    if row and row["station_name"]:
        return row["station_name"]
    return "unknown"


def _route_station_seq(conn: sqlite3.Connection, route_id: int, station_id: int | None) -> int | None:
    if station_id is None:
        return None
    row = conn.execute(
        """
        SELECT station_seq
        FROM route_station
        WHERE route_id = ? AND station_id = ?
        ORDER BY station_seq
        LIMIT 1
        """,
        (route_id, station_id),
    ).fetchone()
    return _to_int(row["station_seq"]) if row else None


def _parse_cli_kst(value: str) -> datetime:
    text = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=KST)
        except ValueError:
            pass
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=KST)
    return parsed.astimezone(KST)


def _schedule_day_type(conn: sqlite3.Connection, at_kst: datetime) -> str:
    if _is_holiday(conn, at_kst) or at_kst.weekday() == 6:
        return "sunday_holiday"
    if at_kst.weekday() == 5:
        return "saturday"
    return "weekday"


def _is_holiday(conn: sqlite3.Connection, at_kst: datetime) -> int:
    row = conn.execute(
        "SELECT is_holiday FROM external_holiday_daily WHERE service_date = ?",
        (at_kst.date().isoformat(),),
    ).fetchone()
    return 1 if row and _to_int(row["is_holiday"]) == 1 else 0


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _to_int(value)
        if parsed is not None:
            return parsed
    return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", 0, "0"):
            return value
    return None


def _time_minutes(value: Any) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    text = str(value)
    if ":" not in text:
        return None
    hour_text, minute_text = text.split(":", 1)
    try:
        return int(hour_text) * 60 + int(minute_text[:2])
    except ValueError:
        return None


def _within_schedule_window(target: int, first: int | None, last: int | None) -> bool:
    if first is None or last is None:
        return False
    if first <= last:
        return first <= target <= last
    return target >= first or target <= last


def _next_scheduled_datetime(at_kst: datetime, first_minutes: int | None) -> datetime | None:
    if first_minutes is None:
        return None
    target = at_kst.hour * 60 + at_kst.minute
    target_date = at_kst.date()
    if target > first_minutes:
        target_date = target_date + timedelta(days=1)
    return datetime.combine(target_date, datetime.min.time(), tzinfo=KST) + timedelta(minutes=first_minutes)


def _is_peak_hour(hour: int) -> bool:
    return 7 <= hour <= 9 or 17 <= hour <= 20


def _median_int(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return int(round(ordered[len(ordered) // 2]))


def _mode_int(values: list[int]) -> int | None:
    if not values:
        return None
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _date_range(start_date: str, end_date: str) -> list[str]:
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    days = []
    current = start
    while current <= end:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _tour_event_daily_rows(items: list[dict[str, Any]], area_key: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    lower = datetime.fromisoformat(start_date).date()
    upper = datetime.fromisoformat(end_date).date()
    for item in items:
        start = _tour_date(item.get("eventstartdate") or item.get("eventStartDate"))
        end = _tour_date(item.get("eventenddate") or item.get("eventEndDate")) or start
        if start is None:
            continue
        current = max(start, lower)
        limited_end = min(end, upper)
        while current <= limited_end:
            key = current.isoformat()
            counts[key] = counts.get(key, 0) + 1
            current += timedelta(days=1)
    return [
        {
            "area_key": area_key,
            "service_date": service_date,
            "event_count": count,
            "event_nearby_count": count,
        }
        for service_date, count in sorted(counts.items())
    ]


def _tour_date(value: Any):
    if value is None or value == "":
        return None
    try:
        return datetime.strptime(str(value), "%Y%m%d").date()
    except ValueError:
        try:
            return datetime.fromisoformat(str(value)).date()
        except ValueError:
            return None


if __name__ == "__main__":
    raise SystemExit(main())
