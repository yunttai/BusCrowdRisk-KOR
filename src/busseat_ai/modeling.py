from __future__ import annotations

import csv
import importlib.util
import json
import math
import pickle
import random
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from busseat_ai.services.risk import CROWDED_LABELS, calculate_seat_scarcity_score
from busseat_ai.storage.database import rows_to_dicts, utc_now_iso


ALLOWED_TARGET_COLUMNS = {
    "target_no_seat_now": None,
    "target_low_seat_2_now": None,
    "target_low_seat_5_now": None,
    "target_no_seat_next_5min": "has_future_5min",
    "target_low_seat_2_next_5min": "has_future_5min",
    "target_no_seat_next_10min": "has_future_10min",
    "target_low_seat_2_next_10min": "has_future_10min",
    "target_no_seat_next_station": "has_next_station",
    "target_low_seat_2_next_station": "has_next_station",
}


@dataclass(frozen=True)
class TrainingDatasetSummary:
    rows: int
    exported_path: str

    def to_dict(self) -> dict[str, Any]:
        return {"rows": self.rows, "exportedPath": self.exported_path}


@dataclass(frozen=True)
class BaselineEvaluation:
    target_column: str
    total_rows: int
    train_rows: int
    test_rows: int
    positive_rate: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    brier_score: float
    top_decile_precision: float
    strategy: str
    exported_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "targetColumn": self.target_column,
            "totalRows": self.total_rows,
            "trainRows": self.train_rows,
            "testRows": self.test_rows,
            "positiveRate": self.positive_rate,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "rocAuc": self.roc_auc,
            "brierScore": self.brier_score,
            "topDecilePrecision": self.top_decile_precision,
            "strategy": self.strategy,
            "exportedPath": self.exported_path,
        }


@dataclass(frozen=True)
class LogisticModelEvaluation:
    model_name: str
    target_column: str
    total_rows: int
    train_rows: int
    test_rows: int
    positive_rate: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    brier_score: float
    top_decile_precision: float
    strategy: str
    included_feature_groups: list[str]
    excluded_feature_groups: list[str]
    feature_count: int
    intercept: float
    feature_weights: list[dict[str, Any]]
    top_features: list[dict[str, Any]]
    exported_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "modelName": self.model_name,
            "targetColumn": self.target_column,
            "totalRows": self.total_rows,
            "trainRows": self.train_rows,
            "testRows": self.test_rows,
            "positiveRate": self.positive_rate,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "rocAuc": self.roc_auc,
            "brierScore": self.brier_score,
            "topDecilePrecision": self.top_decile_precision,
            "strategy": self.strategy,
            "includedFeatureGroups": self.included_feature_groups,
            "excludedFeatureGroups": self.excluded_feature_groups,
            "featureCount": self.feature_count,
            "intercept": self.intercept,
            "featureWeights": self.feature_weights,
            "topFeatures": self.top_features,
            "exportedPath": self.exported_path,
        }


def train_lightgbm_risk_artifact(
    conn: sqlite3.Connection,
    output_artifact_path: str | Path,
    *,
    target_column: str = "target_no_seat_next_station",
    threshold: float = 0.9,
    exact_weather_only: bool = False,
    output_json_path: str | Path | None = None,
) -> dict[str, Any]:
    _validate_target_column(target_column)
    rows = _training_dataset_rows(conn, target_column, exact_weather_only=exact_weather_only)
    if len(rows) < 10:
        raise RuntimeError("LightGBM artifact 학습에는 target이 있는 행이 최소 10개 필요합니다.")

    selector = _FeatureSelector(rows)
    encoder = _FeatureEncoder(selector)
    encoder.fit(rows)
    x_train = [encoder.transform(row) for row in rows]
    y_train = [int(row[target_column]) for row in rows]
    sample_weight = _balanced_sample_weight(y_train)
    model_info = _fit_advanced_classifier(x_train, y_train, sample_weight)
    if model_info["status"] != "ok":
        raise RuntimeError(model_info.get("note") or "LightGBM artifact 학습에 실패했습니다.")

    output = Path(output_artifact_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    max_station_seq_by_route = _max_station_seq_by_route(rows)
    artifact = {
        "version": "lightgbm_full_risk_v1",
        "createdAt": utc_now_iso(),
        "targetColumn": target_column,
        "threshold": threshold,
        "exactWeatherOnly": exact_weather_only,
        "modelName": model_info["modelName"],
        "model": model_info["model"],
        "selector": selector,
        "encoder": encoder,
        "featureNames": encoder.feature_names,
        "includedFeatureGroups": selector.included_groups,
        "excludedFeatureGroups": selector.excluded_groups,
        "trainingRows": len(rows),
        "positiveRate": round(sum(y_train) / len(y_train), 6),
        "maxStationSeqByRoute": max_station_seq_by_route,
        "note": model_info["note"],
    }
    with output.open("wb") as file:
        pickle.dump(artifact, file)

    summary = {
        "artifactPath": str(output),
        "version": artifact["version"],
        "createdAt": artifact["createdAt"],
        "targetColumn": target_column,
        "threshold": threshold,
        "exactWeatherOnly": exact_weather_only,
        "modelName": model_info["modelName"],
        "trainingRows": len(rows),
        "positiveRate": artifact["positiveRate"],
        "featureCount": len(encoder.feature_names),
        "includedFeatureGroups": selector.included_groups,
        "excludedFeatureGroups": selector.excluded_groups,
        "topFeatures": _artifact_top_features(model_info, encoder.feature_names),
    }
    if output_json_path:
        json_output = Path(output_json_path)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["exportedJsonPath"] = str(json_output)
    return summary


def predict_feature_risk_with_artifact(
    conn: sqlite3.Connection,
    artifact_path: str | Path,
    *,
    limit: int = 20,
    snapshot_id: int | None = None,
    at: str | None = None,
    route_name: str | None = None,
    station_id: int | None = None,
    station_name: str | None = None,
    remain_seat_cnt: int | None = None,
    output_json_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = _feature_rows_for_prediction(
        conn,
        limit=limit,
        snapshot_id=snapshot_id,
        at=at,
        route_name=route_name,
        station_id=station_id,
        station_name=station_name,
        remain_seat_cnt=remain_seat_cnt,
    )
    result = predict_feature_rows_with_artifact(artifact_path, rows)
    if not rows:
        result.update(
            _empty_prediction_context(
                conn,
                limit=limit,
                snapshot_id=snapshot_id,
                at=at,
                route_name=route_name,
                station_id=station_id,
                station_name=station_name,
                remain_seat_cnt=remain_seat_cnt,
            )
        )
    if output_json_path:
        output = Path(output_json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["exportedJsonPath"] = str(output)
    return result


def predict_feature_rows_with_artifact(
    artifact_path: str | Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    artifact = _load_model_artifact(artifact_path)
    encoder = artifact["encoder"]
    model = artifact["model"]
    threshold = float(artifact.get("threshold", 0.9))
    max_station_seq_by_route = artifact.get("maxStationSeqByRoute", {})
    predictions = []
    for row in rows:
        features = encoder.transform(row)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            probability = float(model.predict_proba([features])[0][1])
        route = str(row.get("canonical_route_name") or "unknown")
        station_seq = int(_to_float(row.get("station_seq"), 0.0))
        max_seq = int(max_station_seq_by_route.get(route, 0))
        segment = _station_seq_segment(station_seq, max_seq)
        predictions.append(
            {
                "snapshotId": row["snapshot_id"],
                "collectedAtKst": row["collected_at_kst"],
                "observedHourKst": row["observed_hour_kst"],
                "routeId": row["route_id"],
                "routeName": row["canonical_route_name"],
                "plateNo": row["plate_no"],
                "stationId": row["station_id"],
                "stationName": row["station_name"],
                "stationSeq": row["station_seq"],
                "stationSeqSegment": segment,
                "remainSeatCnt": row["remain_seat_cnt"],
                "crowded": row["crowded"],
                "crowdedLabel": row["crowded_label"],
                "fullSeatProbability": round(probability, 6),
                "fullSeatRiskScore": round(probability * 100, 2),
                "threshold": threshold,
                "isFullRisk": probability >= threshold,
                "riskLevel": _probability_level(probability, threshold),
                "modelName": artifact.get("modelName"),
                "modelVersion": artifact.get("version"),
                "featureGroups": artifact.get("includedFeatureGroups", []),
                **_live_prediction_context(row),
            }
        )

    return {
        "artifactPath": str(artifact_path),
        "modelName": artifact.get("modelName"),
        "modelVersion": artifact.get("version"),
        "threshold": threshold,
        "rows": len(predictions),
        "predictions": predictions,
    }


def backtest_schedule_prior(
    conn: sqlite3.Connection,
    *,
    target_column: str = "target_no_seat_next_station",
    test_ratio: float = 0.2,
    threshold: float = 0.9,
    exact_weather_only: bool = False,
    max_test_rows: int | None = None,
    output_json_path: str | Path | None = None,
    output_md_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate schedule_prior fallback with chronological holdout.

    The model is trained on the chronological train split. For each test row,
    the observed bus state is hidden and replaced by a proxy bus state built
    only from train-split history, using the same fallback order as live
    schedule_prior prediction.
    """
    _validate_target_column(target_column)
    rows = _training_dataset_rows(conn, target_column, exact_weather_only=exact_weather_only)
    rows.sort(key=lambda row: (row["collected_at_kst"], row["snapshot_id"]))
    if len(rows) < 20:
        raise RuntimeError("schedule_prior backtest에는 target이 있는 row가 최소 20개 필요합니다.")

    train_size = int(len(rows) * (1.0 - test_ratio))
    train_size = max(1, min(len(rows) - 1, train_size))
    train_rows = rows[:train_size]
    test_rows = rows[train_size:]
    if max_test_rows is not None and max_test_rows > 0:
        test_rows = test_rows[:max_test_rows]

    selector = _FeatureSelector(train_rows)
    encoder = _FeatureEncoder(selector)
    encoder.fit(train_rows)
    x_train = [encoder.transform(row) for row in train_rows]
    y_train = [int(row[target_column]) for row in train_rows]
    model_info = _fit_advanced_classifier(x_train, y_train, _balanced_sample_weight(y_train))
    if model_info["status"] != "ok":
        raise RuntimeError(model_info.get("note") or "schedule_prior backtest 모델 학습에 실패했습니다.")

    proxy_index = _schedule_proxy_index(train_rows, target_column)
    predictions: list[dict[str, Any]] = []
    for actual_row in test_rows:
        proxy_row, proxy_meta = _schedule_proxy_row_from_index(proxy_index, actual_row)
        features = encoder.transform(proxy_row)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            probability = float(model_info["model"].predict_proba([features])[0][1])
        actual = int(actual_row[target_column])
        predicted = 1 if probability >= threshold else 0
        predictions.append(
            {
                "snapshotId": actual_row["snapshot_id"],
                "collectedAtKst": actual_row["collected_at_kst"],
                "routeName": actual_row["canonical_route_name"],
                "stationId": actual_row["station_id"],
                "stationName": actual_row["station_name"],
                "stationSeq": actual_row["station_seq"],
                "hour": actual_row["hour"],
                "isWeekend": actual_row["is_weekend"],
                "isHoliday": actual_row["is_holiday"],
                "actualRemainSeatCnt": actual_row["remain_seat_cnt"],
                "proxyRemainSeatCnt": proxy_row["remain_seat_cnt"],
                "actualCrowded": actual_row["crowded"],
                "proxyCrowded": proxy_row["crowded"],
                "historicalProxyGroup": proxy_meta["historical_proxy_group"],
                "historicalProxyRows": proxy_meta["historical_proxy_rows"],
                "historicalNextStationFullRate": proxy_meta["historical_next_station_full_rate"],
                "historicalNowFullRate": proxy_meta["historical_now_full_rate"],
                "proxyConfidence": _schedule_proxy_confidence(
                    proxy_meta["historical_proxy_group"],
                    proxy_meta["historical_proxy_rows"],
                ),
                "_actual": actual,
                "_probability": probability,
                "_predicted": predicted,
            }
        )

    actuals = [int(row["_actual"]) for row in predictions]
    predicted = [int(row["_predicted"]) for row in predictions]
    probabilities = [float(row["_probability"]) for row in predictions]
    metrics = _classification_metrics(actuals, predicted, probabilities)
    result = {
        "targetColumn": target_column,
        "strategy": "chronological_train_lightgbm_schedule_prior_proxy_backtest",
        "modelName": model_info["modelName"],
        "modelNote": model_info["note"],
        "threshold": threshold,
        "exactWeatherOnly": exact_weather_only,
        "totalRows": len(rows),
        "trainRows": len(train_rows),
        "testRows": len(test_rows),
        "positiveRate": round(sum(int(row[target_column]) for row in rows) / len(rows), 6),
        "includedFeatureGroups": selector.included_groups,
        "excludedFeatureGroups": selector.excluded_groups,
        "featureCount": len(encoder.feature_names),
        "metrics": {_camel_metric_name(key): value for key, value in metrics.items()},
        "thresholdCurve": _threshold_curve(predictions),
        "calibration": _calibration_bins(predictions),
        "fallbackGroupPerformance": _group_prediction_metrics(predictions, "historicalProxyGroup"),
        "confidencePerformance": _group_prediction_metrics(predictions, "proxyConfidence"),
        "topRiskSamples": _schedule_proxy_sample_rows(predictions, reverse=True),
        "missedFullSamples": _schedule_proxy_missed_samples(predictions),
        "generatedAt": utc_now_iso(),
        "notes": [
            "실제 test row의 remain/crowded를 숨기고 train split 이력으로 만든 proxy row를 예측했습니다.",
            "이 검증은 실시간 차량이 없을 때 schedule_prior fallback이 어느 정도 맞는지 보기 위한 것입니다.",
        ],
    }
    if output_json_path:
        output = Path(output_json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["exportedJsonPath"] = str(output)
    if output_md_path:
        output = Path(output_md_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_schedule_prior_backtest_markdown(result), encoding="utf-8")
        result["exportedMarkdownPath"] = str(output)
    return result


def export_training_dataset_csv(
    conn: sqlite3.Connection,
    path: str | Path,
    *,
    exact_weather_only: bool = False,
) -> TrainingDatasetSummary:
    rows = _training_dataset_rows(conn, exact_weather_only=exact_weather_only)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows_csv(rows, output)
    return TrainingDatasetSummary(rows=len(rows), exported_path=str(output))


def evaluate_historical_rate_baseline(
    conn: sqlite3.Connection,
    target_column: str = "target_no_seat_next_station",
    test_ratio: float = 0.2,
    threshold: float = 0.5,
    output_json_path: str | Path | None = None,
    exact_weather_only: bool = False,
) -> BaselineEvaluation:
    _validate_target_column(target_column)
    rows = _training_dataset_rows(conn, target_column, exact_weather_only=exact_weather_only)
    if len(rows) < 2:
        raise RuntimeError("baseline 평가에는 target이 있는 행이 최소 2개 필요합니다.")

    rows.sort(key=lambda row: (row["collected_at_kst"], row["snapshot_id"]))
    train_size = int(len(rows) * (1.0 - test_ratio))
    train_size = max(1, min(len(rows) - 1, train_size))
    train_rows = rows[:train_size]
    test_rows = rows[train_size:]

    model = _HistoricalRateModel(target_column)
    model.fit(train_rows)
    probabilities = [model.predict_proba(row) for row in test_rows]
    actuals = [int(row[target_column]) for row in test_rows]
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    metrics = _classification_metrics(actuals, predictions, probabilities)
    total_positive_rate = round(sum(int(row[target_column]) for row in rows) / len(rows), 6)

    evaluation = BaselineEvaluation(
        target_column=target_column,
        total_rows=len(rows),
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        positive_rate=total_positive_rate,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        roc_auc=metrics["roc_auc"],
        brier_score=metrics["brier_score"],
        top_decile_precision=metrics["top_decile_precision"],
        strategy=f"{model.strategy}|filter=weather_imputed=0" if exact_weather_only else model.strategy,
        exported_path=str(output_json_path) if output_json_path else None,
    )

    _insert_baseline_metrics(conn, evaluation)
    if output_json_path:
        output = Path(output_json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evaluation.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return evaluation


def train_logistic_risk_model(
    conn: sqlite3.Connection,
    target_column: str = "target_no_seat_next_station",
    test_ratio: float = 0.2,
    epochs: int = 8,
    learning_rate: float = 0.035,
    l2: float = 0.0005,
    threshold: float = 0.5,
    output_json_path: str | Path | None = None,
    exact_weather_only: bool = False,
) -> LogisticModelEvaluation:
    _validate_target_column(target_column)
    rows = _training_dataset_rows(conn, target_column, exact_weather_only=exact_weather_only)
    if len(rows) < 10:
        raise RuntimeError("risk model 학습에는 target이 있는 행이 최소 10개 필요합니다.")

    evaluation, _ = _train_logistic_on_rows(
        rows,
        target_column=target_column,
        test_ratio=test_ratio,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        threshold=threshold,
        exact_weather_only=exact_weather_only,
    )

    if output_json_path:
        evaluation = LogisticModelEvaluation(**{**evaluation.__dict__, "exported_path": str(output_json_path)})
    _insert_trained_model_metrics(conn, evaluation)
    if output_json_path:
        output = Path(output_json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evaluation.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return evaluation


def run_model_diagnostics(
    conn: sqlite3.Connection,
    target_column: str = "target_no_seat_next_station",
    test_ratio: float = 0.2,
    epochs: int = 8,
    learning_rate: float = 0.035,
    l2: float = 0.0005,
    threshold: float = 0.5,
    output_json_path: str | Path | None = None,
    output_md_path: str | Path | None = None,
    exact_weather_only: bool = False,
) -> dict[str, Any]:
    _validate_target_column(target_column)
    rows = _training_dataset_rows(conn, target_column, exact_weather_only=exact_weather_only)
    if len(rows) < 10:
        raise RuntimeError("diagnostics에는 target이 있는 행이 최소 10개 필요합니다.")

    variants = [
        ("full", []),
        ("no_bus_state", ["bus_state"]),
        ("no_weather", ["weather"]),
        ("no_route", ["route"]),
        ("no_time", ["time"]),
        ("no_station_seq", ["station_seq"]),
    ]
    ablations = []
    full_predictions: list[dict[str, Any]] = []
    for variant_name, disabled_groups in variants:
        evaluation, predictions = _train_logistic_on_rows(
            rows,
            target_column=target_column,
            test_ratio=test_ratio,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            threshold=threshold,
            exact_weather_only=exact_weather_only,
            disabled_groups=disabled_groups,
            model_name=f"logistic_risk_v1_{variant_name}",
        )
        result = evaluation.to_dict()
        result["variant"] = variant_name
        result["disabledFeatureGroups"] = disabled_groups
        ablations.append(result)
        if variant_name == "full":
            full_predictions = predictions

    calibration = _calibration_bins(full_predictions)
    station_seq_segment_performance = _station_seq_segment_metrics(full_predictions, rows)
    station_seq_segment_thresholds = _station_seq_segment_thresholds(full_predictions, rows)
    route_stratified = _route_stratified_diagnostics(
        rows,
        target_column=target_column,
        test_ratio=test_ratio,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        threshold=threshold,
        exact_weather_only=exact_weather_only,
    )
    calibration_model = _calibration_model_comparison(
        rows,
        target_column=target_column,
        test_ratio=test_ratio,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        threshold=threshold,
        exact_weather_only=exact_weather_only,
    )
    advanced_model = _advanced_model_comparison(
        rows,
        target_column=target_column,
        test_ratio=test_ratio,
        threshold=threshold,
        exact_weather_only=exact_weather_only,
    )
    threshold_curve = _threshold_curve(full_predictions)
    report = {
        "targetColumn": target_column,
        "exactWeatherOnly": exact_weather_only,
        "totalRows": len(rows),
        "testRows": len(full_predictions),
        "positiveRate": round(sum(int(row[target_column]) for row in rows) / len(rows), 6),
        "ablations": ablations,
        "routePerformance": _group_prediction_metrics(full_predictions, "canonical_route_name"),
        "routeStratifiedEvaluation": route_stratified,
        "timePeriodPerformance": _group_prediction_metrics(full_predictions, "time_period"),
        "stationSeqSegmentPerformance": station_seq_segment_performance,
        "stationSeqSegmentThresholds": station_seq_segment_thresholds,
        "thresholdCurve": threshold_curve,
        "operatingPolicy": _operating_policy(threshold_curve, station_seq_segment_thresholds),
        "calibration": calibration,
        "calibrationModel": calibration_model,
        "advancedModelComparison": advanced_model,
        "lightgbmComparison": _lightgbm_status(),
    }

    if output_json_path:
        output = Path(output_json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["exportedJsonPath"] = str(output)
    if output_md_path:
        output = Path(output_md_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_diagnostics_markdown(report), encoding="utf-8")
        report["exportedMarkdownPath"] = str(output)
    return report


class _HistoricalRateModel:
    def __init__(self, target_column: str) -> None:
        self.target_column = target_column
        self.global_rate = 0.5
        self.group_rates: dict[str, dict[tuple[Any, ...], float]] = {}
        self.strategy = (
            "time_split_historical_rate_with_laplace_smoothing:"
            "route_station_hour->route_station->route_hour->hour->global"
        )

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self.global_rate = _smoothed_rate(rows, self.target_column)
        self.group_rates = {
            "route_station_hour": _group_rate(rows, self.target_column, ("canonical_route_name", "station_seq", "hour")),
            "route_station": _group_rate(rows, self.target_column, ("canonical_route_name", "station_seq")),
            "route_hour": _group_rate(rows, self.target_column, ("canonical_route_name", "hour")),
            "hour": _group_rate(rows, self.target_column, ("hour",)),
        }

    def predict_proba(self, row: dict[str, Any]) -> float:
        keys = (
            ("route_station_hour", (row["canonical_route_name"], row["station_seq"], row["hour"])),
            ("route_station", (row["canonical_route_name"], row["station_seq"])),
            ("route_hour", (row["canonical_route_name"], row["hour"])),
            ("hour", (row["hour"],)),
        )
        for group_name, key in keys:
            group = self.group_rates.get(group_name, {})
            if key in group:
                return group[key]
        return self.global_rate


def _train_logistic_on_rows(
    rows: list[dict[str, Any]],
    *,
    target_column: str,
    test_ratio: float,
    epochs: int,
    learning_rate: float,
    l2: float,
    threshold: float,
    exact_weather_only: bool,
    disabled_groups: list[str] | None = None,
    model_name: str = "logistic_risk_v1",
) -> tuple[LogisticModelEvaluation, list[dict[str, Any]]]:
    rows = list(rows)
    rows.sort(key=lambda row: (row["collected_at_kst"], row["snapshot_id"]))
    train_size = int(len(rows) * (1.0 - test_ratio))
    train_size = max(1, min(len(rows) - 1, train_size))
    train_rows = rows[:train_size]
    test_rows = rows[train_size:]
    return _train_logistic_on_split(
        rows,
        train_rows,
        test_rows,
        target_column=target_column,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        threshold=threshold,
        exact_weather_only=exact_weather_only,
        disabled_groups=disabled_groups,
        model_name=model_name,
    )


def _train_logistic_on_split(
    all_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    target_column: str,
    epochs: int,
    learning_rate: float,
    l2: float,
    threshold: float,
    exact_weather_only: bool,
    disabled_groups: list[str] | None = None,
    model_name: str = "logistic_risk_v1",
) -> tuple[LogisticModelEvaluation, list[dict[str, Any]]]:
    all_rows = list(all_rows)
    if not train_rows or not test_rows:
        raise RuntimeError("logistic model 학습/평가에는 train/test row가 모두 필요합니다.")

    selector = _FeatureSelector(all_rows, disabled_groups=disabled_groups)
    encoder = _FeatureEncoder(selector)
    encoder.fit(train_rows)
    x_train = [encoder.transform(row) for row in train_rows]
    y_train = [int(row[target_column]) for row in train_rows]
    x_test = [encoder.transform(row) for row in test_rows]
    y_test = [int(row[target_column]) for row in test_rows]

    weights = _fit_logistic_sgd(x_train, y_train, epochs=epochs, learning_rate=learning_rate, l2=l2)
    probabilities = [_predict_logistic(weights, features) for features in x_test]
    binary_predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    metrics = _classification_metrics(y_test, binary_predictions, probabilities)
    total_positive_rate = round(sum(int(row[target_column]) for row in all_rows) / len(all_rows), 6)
    feature_weights = [
        {"feature": name, "weight": round(weight, 6), "absWeight": round(abs(weight), 6)}
        for name, weight in zip(encoder.feature_names, weights[1:])
    ]
    top_features = [
        item
        for item in sorted(feature_weights, key=lambda item: item["absWeight"], reverse=True)[:30]
    ]
    disabled = sorted(set(disabled_groups or []))
    strategy = (
        "time_split_balanced_logistic_regression:"
        "bus_state+route+station_seq+time+available_external_factors"
        + ("|filter=weather_imputed=0" if exact_weather_only else "")
        + (f"|disabled={','.join(disabled)}" if disabled else "")
    )

    evaluation = LogisticModelEvaluation(
        model_name=model_name,
        target_column=target_column,
        total_rows=len(all_rows),
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        positive_rate=total_positive_rate,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        roc_auc=metrics["roc_auc"],
        brier_score=metrics["brier_score"],
        top_decile_precision=metrics["top_decile_precision"],
        strategy=strategy,
        included_feature_groups=selector.included_groups,
        excluded_feature_groups=selector.excluded_groups,
        feature_count=len(encoder.feature_names),
        intercept=round(weights[0], 6),
        feature_weights=feature_weights,
        top_features=top_features,
    )
    predictions = []
    for row, actual, probability, predicted in zip(test_rows, y_test, probabilities, binary_predictions):
        predictions.append(
            {
                **row,
                "_actual": actual,
                "_probability": probability,
                "_predicted": predicted,
            }
        )
    return evaluation, predictions


class _FeatureSelector:
    def __init__(self, rows: list[dict[str, Any]], disabled_groups: list[str] | None = None) -> None:
        disabled = set(disabled_groups or [])
        self.included_groups = [group for group in ["bus_state", "route", "station_seq", "time"] if group not in disabled]
        self.excluded_groups: list[str] = []
        self.excluded_groups.extend(group for group in ["bus_state", "route", "station_seq", "time"] if group in disabled)
        self.include_expected_boardings = _missing_rate(rows, "expected_boardings_missing") <= 0.5 and "expected_boardings" not in disabled
        if self.include_expected_boardings:
            self.included_groups.append("expected_boardings")
        else:
            self.excluded_groups.append("expected_boardings")

        for group, flag, threshold in (
            ("weather", "weather_imputed", 0.30),
            ("air_quality", "air_quality_imputed", 0.50),
            ("traffic", "traffic_imputed", 0.50),
            ("event", "event_imputed", 0.80),
        ):
            if group not in disabled and _missing_rate(rows, flag) <= threshold:
                self.included_groups.append(group)
            else:
                self.excluded_groups.append(group)

    def numeric_features(self) -> list[str]:
        features = []
        if "bus_state" in self.included_groups:
            features.extend(
                [
                    "remain_seat_cnt",
                    "seat_scarcity_score",
                    "crowded",
                    "estimated_capacity",
                    "is_no_seat",
                    "is_low_seat_2",
                    "is_low_seat_5",
                ]
            )
        if "station_seq" in self.included_groups:
            features.append("station_seq")
        if "time" in self.included_groups:
            features.extend(["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend", "is_holiday"])
        if self.include_expected_boardings:
            features.append("expected_boardings_at_stop")
        if "weather" in self.included_groups:
            features.extend(["temperature", "precipitation", "humidity", "wind_speed", "cloud_amount", "weather_imputed"])
        if "air_quality" in self.included_groups:
            features.extend(["pm10", "pm25", "o3", "khai", "air_quality_imputed"])
        if "traffic" in self.included_groups:
            features.extend(["avg_speed", "traffic_volume", "delay_time", "congestion_level", "traffic_imputed"])
        if "event" in self.included_groups:
            features.extend(["event_count", "event_nearby_count", "event_imputed"])
        return features

    def categorical_features(self) -> list[str]:
        features = []
        if "route" in self.included_groups:
            features.append("canonical_route_name")
        if "time" in self.included_groups:
            features.append("time_period")
        if "bus_state" in self.included_groups:
            features.append("crowded_label")
        if "weather" in self.included_groups:
            features.append("weather_text")
        if "air_quality" in self.included_groups:
            features.append("air_quality_grade")
        return features


class _FeatureEncoder:
    def __init__(self, selector: _FeatureSelector) -> None:
        self.selector = selector
        self.numeric_names = selector.numeric_features()
        self.categorical_names = selector.categorical_features()
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.categories: dict[str, list[str]] = {}
        self.feature_names: list[str] = []

    def fit(self, rows: list[dict[str, Any]]) -> None:
        for name in self.numeric_names:
            values = [_numeric_value(row, name) for row in rows]
            mean = sum(values) / len(values) if values else 0.0
            variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
            std = math.sqrt(variance) or 1.0
            self.means[name] = mean
            self.stds[name] = std
        for name in self.categorical_names:
            values = sorted({_categorical_value(row, name) for row in rows})
            self.categories[name] = values
        self.feature_names = list(self.numeric_names)
        for name in self.categorical_names:
            self.feature_names.extend(f"{name}={value}" for value in self.categories[name])

    def transform(self, row: dict[str, Any]) -> list[float]:
        features = []
        for name in self.numeric_names:
            value = _numeric_value(row, name)
            features.append((value - self.means[name]) / self.stds[name])
        for name in self.categorical_names:
            value = _categorical_value(row, name)
            features.extend(1.0 if value == category else 0.0 for category in self.categories[name])
        return features


def _route_stratified_diagnostics(
    rows: list[dict[str, Any]],
    *,
    target_column: str,
    test_ratio: float,
    epochs: int,
    learning_rate: float,
    l2: float,
    threshold: float,
    exact_weather_only: bool,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("canonical_route_name") or "unknown"), []).append(row)

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    skipped_routes = []
    route_counts = []
    for route, route_rows in sorted(groups.items()):
        route_rows.sort(key=lambda row: (row["collected_at_kst"], row["snapshot_id"]))
        if len(route_rows) < 2:
            skipped_routes.append({"route": route, "rows": len(route_rows)})
            continue
        train_size = int(len(route_rows) * (1.0 - test_ratio))
        train_size = max(1, min(len(route_rows) - 1, train_size))
        route_train = route_rows[:train_size]
        route_test = route_rows[train_size:]
        train_rows.extend(route_train)
        test_rows.extend(route_test)
        route_counts.append(
            {
                "route": route,
                "totalRows": len(route_rows),
                "trainRows": len(route_train),
                "testRows": len(route_test),
                "positiveRate": round(sum(int(row[target_column]) for row in route_rows) / len(route_rows), 6),
            }
        )

    if not train_rows or not test_rows:
        return {
            "status": "skipped",
            "reason": "route-stratified split을 만들 수 있는 row가 부족합니다.",
            "routes": route_counts,
            "skippedRoutes": skipped_routes,
        }

    train_rows.sort(key=lambda row: (row["collected_at_kst"], row["snapshot_id"]))
    test_rows.sort(key=lambda row: (row["collected_at_kst"], row["snapshot_id"]))
    evaluation, predictions = _train_logistic_on_split(
        rows,
        train_rows,
        test_rows,
        target_column=target_column,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        threshold=threshold,
        exact_weather_only=exact_weather_only,
        model_name="logistic_risk_v1_route_stratified",
    )
    return {
        "status": "ok",
        "split": "per_route_chronological_last_test_ratio",
        "overall": evaluation.to_dict(),
        "routes": route_counts,
        "skippedRoutes": skipped_routes,
        "routePerformance": _group_prediction_metrics(predictions, "canonical_route_name"),
    }


def _station_seq_segment_metrics(predictions: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segmented = _with_station_seq_segments(predictions, rows)
    order = {
        "origin_0_25": 0,
        "early_mid_25_50": 1,
        "late_mid_50_75": 2,
        "terminal_75_100": 3,
        "unknown": 4,
    }
    return sorted(
        _group_prediction_metrics(segmented, "_station_seq_segment"),
        key=lambda item: order.get(item["group"], 99),
    )


def _with_station_seq_segments(predictions: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_seq_by_route: dict[str, int] = {}
    for row in rows:
        route = str(row.get("canonical_route_name") or "unknown")
        max_seq_by_route[route] = max(max_seq_by_route.get(route, 0), int(_to_float(row.get("station_seq"), 0.0)))

    segmented = []
    for row in predictions:
        route = str(row.get("canonical_route_name") or "unknown")
        max_seq = max_seq_by_route.get(route, 0)
        segment = _station_seq_segment(int(_to_float(row.get("station_seq"), 0.0)), max_seq)
        segmented.append({**row, "_station_seq_segment": segment})
    return segmented


def _station_seq_segment(station_seq: int, max_seq: int) -> str:
    if station_seq <= 0 or max_seq <= 0:
        return "unknown"
    progress = station_seq / max_seq
    if progress <= 0.25:
        return "origin_0_25"
    if progress <= 0.50:
        return "early_mid_25_50"
    if progress <= 0.75:
        return "late_mid_50_75"
    return "terminal_75_100"


def _threshold_curve(predictions: list[dict[str, Any]], thresholds: list[float] | None = None) -> list[dict[str, Any]]:
    thresholds = thresholds or _DEFAULT_THRESHOLDS
    actuals = [int(row["_actual"]) for row in predictions]
    probabilities = [float(row["_probability"]) for row in predictions]
    results = []
    for threshold in thresholds:
        binary_predictions = [1 if probability >= threshold else 0 for probability in probabilities]
        metrics = _classification_metrics(actuals, binary_predictions, probabilities)
        predicted_positive = sum(binary_predictions)
        results.append(
            {
                "threshold": round(threshold, 3),
                "predictedPositive": predicted_positive,
                "alertRate": round(predicted_positive / len(predictions), 6) if predictions else 0.0,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
    return results


_DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
_SEGMENT_THRESHOLDS = [0.01, 0.02, 0.05, *_DEFAULT_THRESHOLDS]


def _station_seq_segment_thresholds(predictions: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segmented = _with_station_seq_segments(predictions, rows)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in segmented:
        groups.setdefault(str(row["_station_seq_segment"]), []).append(row)

    order = {
        "origin_0_25": 0,
        "early_mid_25_50": 1,
        "late_mid_50_75": 2,
        "terminal_75_100": 3,
        "unknown": 4,
    }
    results = []
    for segment, segment_rows in groups.items():
        curve = _threshold_curve(segment_rows, thresholds=_SEGMENT_THRESHOLDS)
        best = max(curve, key=lambda item: (item["f1"], item["precision"], item["threshold"])) if curve else {}
        default_09 = next((item for item in curve if item["threshold"] == 0.9), None)
        results.append(
            {
                "segment": segment,
                "rows": len(segment_rows),
                "positiveRate": round(sum(int(row["_actual"]) for row in segment_rows) / len(segment_rows), 6)
                if segment_rows
                else 0.0,
                "recommendedThreshold": best.get("threshold"),
                "recommendedPrecision": best.get("precision"),
                "recommendedRecall": best.get("recall"),
                "recommendedF1": best.get("f1"),
                "recommendedAlertRate": best.get("alertRate"),
                "default09Precision": default_09.get("precision") if default_09 else None,
                "default09Recall": default_09.get("recall") if default_09 else None,
                "default09F1": default_09.get("f1") if default_09 else None,
                "curve": curve,
            }
        )
    return sorted(results, key=lambda item: order.get(item["segment"], 99))


def _operating_policy(
    threshold_curve: list[dict[str, Any]],
    station_seq_segment_thresholds: list[dict[str, Any]],
) -> dict[str, Any]:
    balanced = _curve_item(threshold_curve, 0.9)
    precision_mode = _curve_item(threshold_curve, 0.95)
    fallback = max(threshold_curve, key=lambda item: item["f1"]) if threshold_curve else {}
    terminal = next((item for item in station_seq_segment_thresholds if item["segment"] == "terminal_75_100"), None)
    return {
        "defaultThreshold": 0.9 if balanced else fallback.get("threshold"),
        "defaultReason": (
            "0.9에서 precision과 recall이 균형을 이루고 alert rate가 약 15% 수준이라 초기 운영 기준으로 적합합니다."
            if balanced
            else "threshold curve에서 F1이 가장 높은 값을 기본 후보로 선택했습니다."
        ),
        "precisionModeThreshold": 0.95 if precision_mode else fallback.get("threshold"),
        "balancedThresholdMetrics": balanced,
        "precisionModeMetrics": precision_mode,
        "terminalSegmentPolicy": {
            "segment": "terminal_75_100",
            "recommendedThreshold": terminal.get("recommendedThreshold") if terminal else None,
            "reason": (
                "종점부는 positive가 희소하므로 전체 threshold를 그대로 쓰면 경고가 거의 발생하지 않습니다. "
                "구간별 threshold는 참고값으로만 쓰고, 기본 화면에서는 낮은 위험도 prior와 calibration 경고 문구를 함께 둡니다."
                if terminal
                else "terminal_75_100 구간 데이터가 없어 별도 정책을 만들 수 없습니다."
            ),
        },
    }


def _curve_item(curve: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    return next((item for item in curve if item["threshold"] == round(threshold, 3)), None)


def _calibration_model_comparison(
    rows: list[dict[str, Any]],
    *,
    target_column: str,
    test_ratio: float,
    epochs: int,
    learning_rate: float,
    l2: float,
    threshold: float,
    exact_weather_only: bool,
) -> dict[str, Any]:
    rows = list(rows)
    rows.sort(key=lambda row: (row["collected_at_kst"], row["snapshot_id"]))
    test_start = int(len(rows) * (1.0 - test_ratio))
    test_start = max(2, min(len(rows) - 1, test_start))
    train_pool = rows[:test_start]
    test_rows = rows[test_start:]
    calibration_size = max(1, int(len(train_pool) * 0.2))
    if len(train_pool) - calibration_size < 10 or len(test_rows) < 2:
        return {
            "status": "skipped",
            "reason": "calibration train/calibration/test split을 만들 row가 부족합니다.",
        }

    model_train_rows = train_pool[:-calibration_size]
    calibration_rows = train_pool[-calibration_size:]
    _, calibration_predictions = _train_logistic_on_split(
        train_pool,
        model_train_rows,
        calibration_rows,
        target_column=target_column,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        threshold=threshold,
        exact_weather_only=exact_weather_only,
        model_name="logistic_risk_v1_calibration_fit",
    )
    _, raw_test_predictions = _train_logistic_on_split(
        train_pool,
        model_train_rows,
        test_rows,
        target_column=target_column,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        threshold=threshold,
        exact_weather_only=exact_weather_only,
        model_name="logistic_risk_v1_calibration_eval",
    )
    scaler = _fit_platt_scaler(calibration_predictions)
    calibrated_predictions = []
    for row in raw_test_predictions:
        probability = _apply_platt_scaler(float(row["_probability"]), scaler)
        calibrated_predictions.append(
            {
                **row,
                "_probability": probability,
                "_predicted": 1 if probability >= threshold else 0,
            }
        )

    raw_actuals = [int(row["_actual"]) for row in raw_test_predictions]
    raw_probabilities = [float(row["_probability"]) for row in raw_test_predictions]
    raw_binary = [int(row["_predicted"]) for row in raw_test_predictions]
    calibrated_probabilities = [float(row["_probability"]) for row in calibrated_predictions]
    calibrated_binary = [int(row["_predicted"]) for row in calibrated_predictions]
    raw_calibration = _calibration_bins(raw_test_predictions)
    calibrated_calibration = _calibration_bins(calibrated_predictions)
    return {
        "status": "ok",
        "method": "platt_scaling",
        "split": "chronological_train_64_calibration_16_test_20",
        "trainRows": len(model_train_rows),
        "calibrationRows": len(calibration_rows),
        "testRows": len(test_rows),
        "parameters": {"a": round(scaler["a"], 6), "b": round(scaler["b"], 6)},
        "rawMetrics": _classification_metrics(raw_actuals, raw_binary, raw_probabilities),
        "calibratedMetrics": _classification_metrics(raw_actuals, calibrated_binary, calibrated_probabilities),
        "rawExpectedCalibrationError": raw_calibration["expectedCalibrationError"],
        "calibratedExpectedCalibrationError": calibrated_calibration["expectedCalibrationError"],
        "rawCalibration": raw_calibration,
        "calibratedCalibration": calibrated_calibration,
    }


def _advanced_model_comparison(
    rows: list[dict[str, Any]],
    *,
    target_column: str,
    test_ratio: float,
    threshold: float,
    exact_weather_only: bool,
) -> dict[str, Any]:
    rows = list(rows)
    rows.sort(key=lambda row: (row["collected_at_kst"], row["snapshot_id"]))
    train_size = int(len(rows) * (1.0 - test_ratio))
    train_size = max(1, min(len(rows) - 1, train_size))
    train_rows = rows[:train_size]
    test_rows = rows[train_size:]

    selector = _FeatureSelector(rows)
    encoder = _FeatureEncoder(selector)
    encoder.fit(train_rows)
    x_train = [encoder.transform(row) for row in train_rows]
    y_train = [int(row[target_column]) for row in train_rows]
    x_test = [encoder.transform(row) for row in test_rows]
    y_test = [int(row[target_column]) for row in test_rows]

    positives = sum(y_train)
    negatives = len(y_train) - positives
    pos_weight = len(y_train) / (2 * positives) if positives else 1.0
    neg_weight = len(y_train) / (2 * negatives) if negatives else 1.0
    sample_weight = [pos_weight if actual else neg_weight for actual in y_train]

    model_info = _fit_advanced_classifier(x_train, y_train, sample_weight)
    if model_info["status"] != "ok":
        return model_info

    model = model_info["model"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        probabilities = [float(pair[1]) for pair in model.predict_proba(x_test)]
    binary_predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    metrics = _classification_metrics(y_test, binary_predictions, probabilities)
    predictions = [
        {**row, "_actual": actual, "_probability": probability, "_predicted": predicted}
        for row, actual, probability, predicted in zip(test_rows, y_test, probabilities, binary_predictions)
    ]
    result = {
        "status": "ok",
        "modelName": model_info["modelName"],
        "note": model_info["note"],
        "exactWeatherOnly": exact_weather_only,
        "trainRows": len(train_rows),
        "testRows": len(test_rows),
        "featureCount": len(encoder.feature_names),
        "includedFeatureGroups": selector.included_groups,
        "excludedFeatureGroups": selector.excluded_groups,
        "metrics": metrics,
        "thresholdCurve": _threshold_curve(predictions),
        "calibration": _calibration_bins(predictions),
        "routePerformance": _group_prediction_metrics(predictions, "canonical_route_name"),
        "timePeriodPerformance": _group_prediction_metrics(predictions, "time_period"),
    }
    if model_info.get("featureImportances") is not None:
        feature_importances = [
            {
                "feature": feature,
                "importance": round(float(importance), 6),
            }
            for feature, importance in zip(encoder.feature_names, model_info["featureImportances"])
        ]
        result["topFeatures"] = sorted(feature_importances, key=lambda item: item["importance"], reverse=True)[:30]
    return result


def _fit_advanced_classifier(x_train: list[list[float]], y_train: list[int], sample_weight: list[float]) -> dict[str, Any]:
    if importlib.util.find_spec("lightgbm") is not None:
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:  # pragma: no cover - optional dependency guard
            return {
                "status": "import_failed",
                "modelName": "lightgbm_lgbm_classifier",
                "note": str(exc),
            }

        model = LGBMClassifier(
            n_estimators=220,
            learning_rate=0.045,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.1,
            objective="binary",
            random_state=20260524,
            verbose=-1,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        return {
            "status": "ok",
            "modelName": "lightgbm_lgbm_classifier",
            "note": "LightGBM LGBMClassifier로 Logistic Regression 기준 모델과 동일 split/feature에서 비교했습니다.",
            "model": model,
            "featureImportances": list(model.feature_importances_),
        }

    if importlib.util.find_spec("sklearn") is None:
        return {
            "status": "not_installed",
            "modelName": "tree_boosting",
            "note": "LightGBM/scikit-learn이 없어 tree boosting 비교 모델을 실행하지 않았습니다.",
        }

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception as exc:  # pragma: no cover - optional dependency guard
        return {
            "status": "import_failed",
            "modelName": "sklearn_hist_gradient_boosting",
            "note": str(exc),
        }

    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.001,
        random_state=20260524,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return {
        "status": "ok",
        "modelName": "sklearn_hist_gradient_boosting",
        "note": "LightGBM 대신 현재 환경에 설치된 scikit-learn의 HistGradientBoostingClassifier로 비선형 비교 모델을 실행했습니다.",
        "model": model,
        "featureImportances": None,
    }


def _balanced_sample_weight(y_train: list[int]) -> list[float]:
    positives = sum(y_train)
    negatives = len(y_train) - positives
    pos_weight = len(y_train) / (2 * positives) if positives else 1.0
    neg_weight = len(y_train) / (2 * negatives) if negatives else 1.0
    return [pos_weight if actual else neg_weight for actual in y_train]


def _load_model_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"모델 artifact가 없습니다: {artifact_path}")
    with artifact_path.open("rb") as file:
        artifact = pickle.load(file)
    required = {"model", "encoder", "threshold", "version"}
    missing = sorted(required - set(artifact))
    if missing:
        raise RuntimeError(f"모델 artifact 형식이 올바르지 않습니다. 누락: {', '.join(missing)}")
    return artifact


def _artifact_top_features(model_info: dict[str, Any], feature_names: list[str], limit: int = 30) -> list[dict[str, Any]]:
    importances = model_info.get("featureImportances")
    if importances is None:
        return []
    rows = [
        {"feature": feature, "importance": round(float(importance), 6)}
        for feature, importance in zip(feature_names, importances)
    ]
    return sorted(rows, key=lambda item: item["importance"], reverse=True)[:limit]


def _max_station_seq_by_route(rows: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        route = str(row.get("canonical_route_name") or "unknown")
        result[route] = max(result.get(route, 0), int(_to_float(row.get("station_seq"), 0.0)))
    return result


def _probability_level(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "high"
    if probability >= max(0.7, threshold - 0.2):
        return "watch"
    if probability >= 0.4:
        return "medium"
    return "low"


def _live_prediction_context(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "prediction_mode",
        "service_date",
        "day_name_ko",
        "is_holiday",
        "holiday_name",
        "requested_station_id",
        "requested_station_name",
        "input_at_kst",
        "arrival_slot",
        "predict_time",
        "predict_time_sec",
        "feature_source",
        "matched_location",
        "schedule_source",
        "schedule_service_date",
        "schedule_day_type",
        "schedule_direction",
        "schedule_first_time",
        "schedule_last_time",
        "schedule_allocation_minutes",
        "scheduled_operating",
        "schedule_proxy_at_kst",
        "schedule_proxy_time_basis",
        "historical_proxy_group",
        "historical_proxy_rows",
        "historical_next_station_full_rate",
        "historical_now_full_rate",
        "proxy_remain_source",
        "proxyRemainSeatCnt",
        "proxyCrowded",
        "proxyCapacity",
    )
    return {_camel_case(key): row[key] for key in keys if key in row}


def _schedule_proxy_index(rows: list[dict[str, Any]], target_column: str) -> dict[str, Any]:
    groups: dict[str, dict[tuple[Any, ...], list[dict[str, Any]]]] = {
        "route_station_hour_daytype": {},
        "route_station_daytype": {},
        "route_station_hour": {},
        "route_station": {},
        "route_station_seq_hour": {},
        "route_hour": {},
        "route": {},
        "hour": {},
    }

    def add(group_name: str, key: tuple[Any, ...], row: dict[str, Any]) -> None:
        groups[group_name].setdefault(key, []).append(row)

    for row in rows:
        route = row.get("canonical_route_name")
        station_id = row.get("station_id")
        station_seq = row.get("station_seq")
        hour = row.get("hour")
        is_weekend = row.get("is_weekend")
        is_holiday = row.get("is_holiday")
        add("route_station_hour_daytype", (route, station_id, hour, is_weekend, is_holiday), row)
        add("route_station_daytype", (route, station_id, is_weekend, is_holiday), row)
        add("route_station_hour", (route, station_id, hour), row)
        add("route_station", (route, station_id), row)
        add("route_station_seq_hour", (route, station_seq, hour), row)
        add("route_hour", (route, hour), row)
        add("route", (route,), row)
        add("hour", (hour,), row)
    return {"groups": groups, "targetColumn": target_column}


def _schedule_proxy_row_from_index(
    proxy_index: dict[str, Any],
    actual_row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    group_name, history_rows = _schedule_proxy_history_rows(proxy_index, actual_row)
    proxy = _schedule_proxy_stats(group_name, history_rows, proxy_index["targetColumn"])
    capacity = proxy["proxyCapacity"] or _safe_int(actual_row.get("estimated_capacity")) or 45
    remain = proxy["proxyRemainSeatCnt"]
    if remain is None:
        remain = capacity
    crowded = proxy["proxyCrowded"]
    if crowded is None:
        crowded = 0

    proxy_row = dict(actual_row)
    proxy_row.update(
        {
            "plate_no": "scheduled_proxy_backtest",
            "remain_seat_cnt": int(remain),
            "crowded": int(crowded),
            "crowded_label": CROWDED_LABELS.get(int(crowded), "unknown"),
            "estimated_capacity": int(capacity),
            "seat_scarcity_score": calculate_seat_scarcity_score(int(remain), int(capacity)),
            "is_no_seat": 1 if int(remain) == 0 else 0,
            "is_low_seat_2": 1 if 0 <= int(remain) <= 2 else 0,
            "is_low_seat_5": 1 if 0 <= int(remain) <= 5 else 0,
            "prediction_mode": "schedule_prior_backtest",
            **proxy,
        }
    )
    return proxy_row, proxy


def _schedule_proxy_history_rows(
    proxy_index: dict[str, Any],
    row: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    groups = proxy_index["groups"]
    route = row.get("canonical_route_name")
    station_id = row.get("station_id")
    station_seq = row.get("station_seq")
    hour = _safe_int(row.get("hour")) or 0
    is_weekend = row.get("is_weekend")
    is_holiday = row.get("is_holiday")

    attempts: list[tuple[str, list[dict[str, Any]]]] = [
        (
            "route_station_hour_daytype",
            groups["route_station_hour_daytype"].get((route, station_id, hour, is_weekend, is_holiday), []),
        ),
        (
            "route_station_adjacent_hour_daytype",
            _concat_history(
                groups["route_station_hour_daytype"].get((route, station_id, candidate_hour, is_weekend, is_holiday), [])
                for candidate_hour in range(max(0, hour - 1), min(23, hour + 1) + 1)
            ),
        ),
        ("route_station_hour", groups["route_station_hour"].get((route, station_id, hour), [])),
        (
            "route_station_daytype",
            groups["route_station_daytype"].get((route, station_id, is_weekend, is_holiday), []),
        ),
        ("route_station", groups["route_station"].get((route, station_id), [])),
        ("route_station_seq_hour", groups["route_station_seq_hour"].get((route, station_seq, hour), [])),
        ("route_hour", groups["route_hour"].get((route, hour), [])),
        ("route", groups["route"].get((route,), [])),
        ("hour", groups["hour"].get((hour,), [])),
    ]
    for name, rows in attempts:
        if rows:
            return name, rows
    return "none", []


def _concat_history(row_groups: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for group in row_groups:
        for row in group:
            snapshot_id = _safe_int(row.get("snapshot_id"))
            if snapshot_id is None or snapshot_id not in seen:
                rows.append(row)
                if snapshot_id is not None:
                    seen.add(snapshot_id)
    return rows


def _schedule_proxy_stats(group_name: str, rows: list[dict[str, Any]], target_column: str) -> dict[str, Any]:
    if not rows:
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

    remains = [value for value in (_safe_int(row.get("remain_seat_cnt")) for row in rows) if value is not None and value >= 0]
    crowded_values = [value for value in (_safe_int(row.get("crowded")) for row in rows) if value is not None]
    capacities = [value for value in (_safe_int(row.get("estimated_capacity")) for row in rows) if value is not None and value > 0]
    target_values = [value for value in (_safe_int(row.get(target_column)) for row in rows) if value is not None]
    now_values = [1 if _safe_int(row.get("remain_seat_cnt")) == 0 else 0 for row in rows]
    return {
        "historical_proxy_group": group_name,
        "historical_proxy_rows": len(rows),
        "historical_next_station_full_rate": round(sum(target_values) / len(target_values), 6) if target_values else None,
        "historical_now_full_rate": round(sum(now_values) / len(now_values), 6) if now_values else None,
        "proxy_remain_source": "historical_median",
        "proxyRemainSeatCnt": _median_int(remains),
        "proxyCrowded": _mode_int(crowded_values),
        "proxyCapacity": _median_int(capacities),
    }


def _schedule_proxy_confidence(group_name: str, rows: int) -> str:
    if group_name == "route_station_hour_daytype" and rows >= 20:
        return "high"
    if group_name in {"route_station_hour_daytype", "route_station_adjacent_hour_daytype", "route_station_daytype"} and rows >= 10:
        return "medium"
    if group_name not in {"none", "hour"} and rows >= 5:
        return "low"
    return "very_low"


def _schedule_proxy_sample_rows(predictions: list[dict[str, Any]], *, reverse: bool) -> list[dict[str, Any]]:
    rows = sorted(predictions, key=lambda row: float(row["_probability"]), reverse=reverse)[:20]
    return [_public_schedule_proxy_row(row) for row in rows]


def _schedule_proxy_missed_samples(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in predictions
        if int(row["_actual"]) == 1 and int(row["_predicted"]) == 0
    ]
    rows = sorted(rows, key=lambda row: float(row["_probability"]))[:20]
    return [_public_schedule_proxy_row(row) for row in rows]


def _public_schedule_proxy_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "snapshotId",
            "collectedAtKst",
            "routeName",
            "stationId",
            "stationName",
            "stationSeq",
            "hour",
            "actualRemainSeatCnt",
            "proxyRemainSeatCnt",
            "actualCrowded",
            "proxyCrowded",
            "historicalProxyGroup",
            "historicalProxyRows",
            "proxyConfidence",
        )
        if key in row
    } | {
        "actual": int(row["_actual"]),
        "probability": round(float(row["_probability"]), 6),
        "predicted": int(row["_predicted"]),
    }


def _schedule_prior_backtest_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# Schedule Prior Backtest",
        "",
        f"- Target: `{report['targetColumn']}`",
        f"- Strategy: `{report['strategy']}`",
        f"- Model: `{report['modelName']}`",
        f"- Threshold: `{report['threshold']}`",
        f"- Train rows: `{report['trainRows']}`",
        f"- Test rows: `{report['testRows']}`",
        f"- Positive rate: `{report['positiveRate']}`",
        "",
        "## Overall Metrics",
        "",
        f"- Accuracy: `{metrics['accuracy']}`",
        f"- Precision: `{metrics['precision']}`",
        f"- Recall: `{metrics['recall']}`",
        f"- F1: `{metrics['f1']}`",
        f"- ROC-AUC: `{metrics['rocAuc']}`",
        f"- Brier score: `{metrics['brierScore']}`",
        f"- Top decile precision: `{metrics['topDecilePrecision']}`",
        "",
        "## Fallback Group Performance",
        "",
        "| Group | Rows | Positive Rate | Precision | Recall | F1 | ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["fallbackGroupPerformance"]:
        lines.append(
            f"| {item['group']} | {item['rows']} | {item['positiveRate']} | {item['precision']} | {item['recall']} | {item['f1']} | {item['rocAuc']} |"
        )
    lines.extend(
        [
            "",
            "## Confidence Performance",
            "",
            "| Confidence | Rows | Positive Rate | Precision | Recall | F1 | ROC-AUC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["confidencePerformance"]:
        lines.append(
            f"| {item['group']} | {item['rows']} | {item['positiveRate']} | {item['precision']} | {item['recall']} | {item['f1']} | {item['rocAuc']} |"
        )
    lines.extend(
        [
            "",
            "## Calibration",
            "",
            f"- Expected calibration error: `{report['calibration']['expectedCalibrationError']}`",
            "",
            "| Probability Bin | Rows | Avg Predicted | Actual Rate | Abs Error |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["calibration"]["bins"]:
        lines.append(
            f"| {item['bin']} | {item['rows']} | {item['avgPredicted']} | {item['actualRate']} | {item['absError']} |"
        )
    return "\n".join(lines)


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


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


def _camel_case(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _fit_platt_scaler(calibration_predictions: list[dict[str, Any]]) -> dict[str, float]:
    if not calibration_predictions:
        return {"a": 1.0, "b": 0.0}
    a = 1.0
    b = 0.0
    learning_rate = 0.02
    l2 = 0.001
    pairs = [
        (_logit(float(row["_probability"])), int(row["_actual"]))
        for row in calibration_predictions
    ]
    for _ in range(300):
        grad_a = 0.0
        grad_b = 0.0
        for logit_value, actual in pairs:
            predicted = _sigmoid(a * logit_value + b)
            error = predicted - actual
            grad_a += error * logit_value
            grad_b += error
        grad_a = grad_a / len(pairs) + l2 * a
        grad_b = grad_b / len(pairs)
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
    return {"a": a, "b": b}


def _apply_platt_scaler(probability: float, scaler: dict[str, float]) -> float:
    return _sigmoid(scaler["a"] * _logit(probability) + scaler["b"])


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 1 - 1e-6)
    return math.log(clipped / (1 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-min(value, 60))
        return 1 / (1 + exp_neg)
    exp_pos = math.exp(max(value, -60))
    return exp_pos / (1 + exp_pos)


def _group_prediction_metrics(predictions: list[dict[str, Any]], group_column: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        groups.setdefault(str(row.get(group_column) or "unknown"), []).append(row)

    results = []
    for group_value, group_rows in groups.items():
        actuals = [int(row["_actual"]) for row in group_rows]
        predicted = [int(row["_predicted"]) for row in group_rows]
        probabilities = [float(row["_probability"]) for row in group_rows]
        metrics = _classification_metrics(actuals, predicted, probabilities)
        results.append(
            {
                "group": group_value,
                "rows": len(group_rows),
                "positiveRate": round(sum(actuals) / len(actuals), 6) if actuals else 0.0,
                **{_camel_metric_name(key): value for key, value in metrics.items()},
            }
        )
    return sorted(results, key=lambda item: (-item["rows"], item["group"]))


def _calibration_bins(predictions: list[dict[str, Any]], bin_count: int = 10) -> dict[str, Any]:
    bins = []
    total_abs_error = 0.0
    total_rows = 0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        if index == bin_count - 1:
            rows = [row for row in predictions if lower <= float(row["_probability"]) <= upper]
        else:
            rows = [row for row in predictions if lower <= float(row["_probability"]) < upper]
        if not rows:
            bins.append(
                {
                    "bin": f"{lower:.1f}-{upper:.1f}",
                    "rows": 0,
                    "avgPredicted": 0.0,
                    "actualRate": 0.0,
                    "absError": 0.0,
                }
            )
            continue
        avg_predicted = sum(float(row["_probability"]) for row in rows) / len(rows)
        actual_rate = sum(int(row["_actual"]) for row in rows) / len(rows)
        abs_error = abs(avg_predicted - actual_rate)
        total_abs_error += abs_error * len(rows)
        total_rows += len(rows)
        bins.append(
            {
                "bin": f"{lower:.1f}-{upper:.1f}",
                "rows": len(rows),
                "avgPredicted": round(avg_predicted, 6),
                "actualRate": round(actual_rate, 6),
                "absError": round(abs_error, 6),
            }
        )
    return {
        "bins": bins,
        "expectedCalibrationError": round(total_abs_error / total_rows, 6) if total_rows else 0.0,
    }


def _lightgbm_status() -> dict[str, Any]:
    return {
        "available": importlib.util.find_spec("lightgbm") is not None,
        "status": "available" if importlib.util.find_spec("lightgbm") is not None else "not_installed",
        "note": (
            "LightGBM is available and is used by advancedModelComparison when diagnostics run."
            if importlib.util.find_spec("lightgbm") is not None
            else "LightGBM is not installed. Keep it as a later comparison model or add it as an optional dependency."
        ),
    }


def _diagnostics_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 모델 진단 리포트",
        "",
        f"- Target: `{report['targetColumn']}`",
        f"- Exact weather only: `{report['exactWeatherOnly']}`",
        f"- Total rows: `{report['totalRows']}`",
        f"- Test rows: `{report['testRows']}`",
        f"- Positive rate: `{report['positiveRate']}`",
        "",
        "## Ablation",
        "",
        "| Variant | Disabled | Rows | Accuracy | Precision | Recall | F1 | ROC-AUC | Top 10% Precision |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["ablations"]:
        lines.append(
            "| {variant} | {disabled} | {rows} | {accuracy} | {precision} | {recall} | {f1} | {roc} | {top} |".format(
                variant=item["variant"],
                disabled=", ".join(item["disabledFeatureGroups"]) or "-",
                rows=item["totalRows"],
                accuracy=item["accuracy"],
                precision=item["precision"],
                recall=item["recall"],
                f1=item["f1"],
                roc=item["rocAuc"],
                top=item["topDecilePrecision"],
            )
        )

    lines.extend(
        [
            "",
            "## Route Performance",
            "",
            "| Route | Rows | Positive Rate | Precision | Recall | F1 | ROC-AUC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["routePerformance"]:
        lines.append(
            f"| {item['group']} | {item['rows']} | {item['positiveRate']} | {item['precision']} | {item['recall']} | {item['f1']} | {item['rocAuc']} |"
        )

    route_stratified = report.get("routeStratifiedEvaluation", {})
    lines.extend(
        [
            "",
            "## Route-Stratified Performance",
            "",
        ]
    )
    if route_stratified.get("status") == "ok":
        overall = route_stratified["overall"]
        lines.extend(
            [
                f"- Split: `{route_stratified['split']}`",
                f"- Train rows: `{overall['trainRows']}`",
                f"- Test rows: `{overall['testRows']}`",
                f"- F1: `{overall['f1']}`",
                f"- ROC-AUC: `{overall['rocAuc']}`",
                "",
                "| Route | Total Rows | Train Rows | Test Rows | Positive Rate |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in route_stratified["routes"]:
            lines.append(
                f"| {item['route']} | {item['totalRows']} | {item['trainRows']} | {item['testRows']} | {item['positiveRate']} |"
            )
        lines.extend(
            [
                "",
                "| Route | Rows | Precision | Recall | F1 | ROC-AUC |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in route_stratified["routePerformance"]:
            lines.append(
                f"| {item['group']} | {item['rows']} | {item['precision']} | {item['recall']} | {item['f1']} | {item['rocAuc']} |"
            )
    else:
        lines.append(f"- Status: `{route_stratified.get('status', 'unknown')}`")
        lines.append(f"- Reason: {route_stratified.get('reason', '-')}")

    lines.extend(
        [
            "",
            "## Time Period Performance",
            "",
            "| Time Period | Rows | Positive Rate | Precision | Recall | F1 | ROC-AUC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["timePeriodPerformance"]:
        lines.append(
            f"| {item['group']} | {item['rows']} | {item['positiveRate']} | {item['precision']} | {item['recall']} | {item['f1']} | {item['rocAuc']} |"
        )

    lines.extend(
        [
            "",
            "## Station Sequence Segment Performance",
            "",
            "| Segment | Rows | Positive Rate | Precision | Recall | F1 | ROC-AUC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["stationSeqSegmentPerformance"]:
        lines.append(
            f"| {item['group']} | {item['rows']} | {item['positiveRate']} | {item['precision']} | {item['recall']} | {item['f1']} | {item['rocAuc']} |"
        )

    lines.extend(
        [
            "",
            "## Threshold Curve",
            "",
            "| Threshold | Alert Rows | Alert Rate | Precision | Recall | F1 | Accuracy |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["thresholdCurve"]:
        lines.append(
            f"| {item['threshold']} | {item['predictedPositive']} | {item['alertRate']} | {item['precision']} | {item['recall']} | {item['f1']} | {item['accuracy']} |"
        )

    lines.extend(
        [
            "",
            "## Calibration",
            "",
            f"- Expected calibration error: `{report['calibration']['expectedCalibrationError']}`",
            "",
            "| Probability Bin | Rows | Avg Predicted | Actual Rate | Abs Error |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["calibration"]["bins"]:
        lines.append(
            f"| {item['bin']} | {item['rows']} | {item['avgPredicted']} | {item['actualRate']} | {item['absError']} |"
        )

    calibration_model = report.get("calibrationModel", {})
    lines.extend(
        [
            "",
            "## Calibration Model",
            "",
        ]
    )
    if calibration_model.get("status") == "ok":
        lines.extend(
            [
                f"- Method: `{calibration_model['method']}`",
                f"- Split: `{calibration_model['split']}`",
                f"- Train rows: `{calibration_model['trainRows']}`",
                f"- Calibration rows: `{calibration_model['calibrationRows']}`",
                f"- Test rows: `{calibration_model['testRows']}`",
                f"- Raw ECE: `{calibration_model['rawExpectedCalibrationError']}`",
                f"- Calibrated ECE: `{calibration_model['calibratedExpectedCalibrationError']}`",
                "",
                "| Version | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier Score |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        raw = calibration_model["rawMetrics"]
        calibrated = calibration_model["calibratedMetrics"]
        lines.append(
            f"| raw | {raw['accuracy']} | {raw['precision']} | {raw['recall']} | {raw['f1']} | {raw['roc_auc']} | {raw['brier_score']} |"
        )
        lines.append(
            f"| calibrated | {calibrated['accuracy']} | {calibrated['precision']} | {calibrated['recall']} | {calibrated['f1']} | {calibrated['roc_auc']} | {calibrated['brier_score']} |"
        )
    else:
        lines.append(f"- Status: `{calibration_model.get('status', 'unknown')}`")
        lines.append(f"- Reason: {calibration_model.get('reason', '-')}")

    advanced_model = report.get("advancedModelComparison", {})
    lines.extend(
        [
            "",
            "## Advanced Model Comparison",
            "",
        ]
    )
    if advanced_model.get("status") == "ok":
        metrics = advanced_model["metrics"]
        calibration = advanced_model["calibration"]
        lines.extend(
            [
                f"- Model: `{advanced_model['modelName']}`",
                f"- Note: {advanced_model['note']}",
                f"- Train rows: `{advanced_model['trainRows']}`",
                f"- Test rows: `{advanced_model['testRows']}`",
                f"- Feature count: `{advanced_model['featureCount']}`",
                "",
                "| Accuracy | Precision | Recall | F1 | ROC-AUC | Brier Score | Top 10% Precision | ECE |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| {metrics['accuracy']} | {metrics['precision']} | {metrics['recall']} | {metrics['f1']} | {metrics['roc_auc']} | {metrics['brier_score']} | {metrics['top_decile_precision']} | {calibration['expectedCalibrationError']} |",
            ]
        )
        top_features = advanced_model.get("topFeatures") or []
        if top_features:
            lines.extend(
                [
                    "",
                    "| Top Feature | Importance |",
                    "| --- | ---: |",
                ]
            )
            for item in top_features[:15]:
                lines.append(f"| {item['feature']} | {item['importance']} |")
    else:
        lines.append(f"- Status: `{advanced_model.get('status', 'unknown')}`")
        lines.append(f"- Note: {advanced_model.get('note', '-')}")

    lines.extend(
        [
            "",
            "## LightGBM",
            "",
            f"- Status: `{report['lightgbmComparison']['status']}`",
            f"- Note: {report['lightgbmComparison']['note']}",
            "",
        ]
    )
    return "\n".join(lines)


def _camel_metric_name(name: str) -> str:
    mapping = {
        "roc_auc": "rocAuc",
        "brier_score": "brierScore",
        "top_decile_precision": "topDecilePrecision",
    }
    if name in mapping:
        return mapping[name]
    return name


def _feature_rows_for_prediction(
    conn: sqlite3.Connection,
    *,
    limit: int,
    snapshot_id: int | None = None,
    at: str | None = None,
    route_name: str | None = None,
    station_id: int | None = None,
    station_name: str | None = None,
    remain_seat_cnt: int | None = None,
) -> list[dict[str, Any]]:
    conditions = []
    params: dict[str, Any] = {}
    if snapshot_id is not None:
        conditions.append("mf.snapshot_id = :snapshot_id")
        params["snapshot_id"] = snapshot_id
    if at:
        conditions.append("mf.collected_at_kst = :collected_at_kst")
        params["collected_at_kst"] = _normalize_kst_datetime_filter(at)
    if route_name:
        conditions.append("mf.canonical_route_name = :route_name")
        params["route_name"] = route_name
    if station_id is not None:
        conditions.append("mf.station_id = :station_id")
        params["station_id"] = station_id
    if station_name:
        conditions.append("mf.station_name = :station_name")
        params["station_name"] = station_name
    if remain_seat_cnt is not None:
        conditions.append("mf.remain_seat_cnt = :remain_seat_cnt")
        params["remain_seat_cnt"] = remain_seat_cnt
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params["limit"] = max(1, int(limit))
    query = f"""
        SELECT
            mf.snapshot_id,
            mf.collected_at_kst,
            mf.observed_hour_kst,
            mf.service_date,
            mf.day_of_week,
            mf.day_name_ko,
            mf.is_weekend,
            mf.is_holiday,
            mf.holiday_name,
            mf.hour,
            mf.time_bucket_10m,
            mf.time_period,
            mf.route_id,
            mf.route_name,
            mf.canonical_route_name,
            mf.route_type_cd,
            mf.route_type_name,
            mf.veh_id,
            mf.plate_no,
            mf.station_id,
            mf.station_seq,
            mf.station_name,
            mf.mobile_no,
            mf.x,
            mf.y,
            mf.remain_seat_cnt,
            mf.crowded,
            mf.crowded_label,
            mf.estimated_capacity,
            mf.seat_scarcity_score,
            mf.is_no_seat,
            mf.is_low_seat_2,
            mf.is_low_seat_5,
            COALESCE(eb.expected_boardings_at_stop, 0.0) AS expected_boardings_at_stop,
            CASE WHEN eb.expected_boardings_at_stop IS NULL THEN 1 ELSE 0 END AS expected_boardings_missing,
            mf.temperature,
            mf.precipitation,
            mf.humidity,
            mf.wind_speed,
            mf.cloud_amount,
            mf.weather_text,
            mf.weather_imputed,
            mf.pm10,
            mf.pm25,
            mf.o3,
            mf.khai,
            mf.air_quality_grade,
            mf.air_quality_imputed,
            mf.avg_speed,
            mf.traffic_volume,
            mf.delay_time,
            mf.congestion_level,
            mf.traffic_imputed,
            mf.event_count,
            mf.event_nearby_count,
            mf.event_imputed
        FROM model_hourly_features mf
        LEFT JOIN station_expected_boardings_hourly eb
          ON eb.station_id = mf.station_id
         AND eb.route_id = mf.route_id
         AND eb.day_of_week = mf.day_of_week
         AND eb.hour = mf.hour
        {where}
        ORDER BY mf.collected_at_kst DESC, mf.snapshot_id DESC
        LIMIT :limit
    """
    return rows_to_dicts(conn.execute(query, params).fetchall())


def _empty_prediction_context(
    conn: sqlite3.Connection,
    *,
    limit: int,
    snapshot_id: int | None = None,
    at: str | None = None,
    route_name: str | None = None,
    station_id: int | None = None,
    station_name: str | None = None,
    remain_seat_cnt: int | None = None,
) -> dict[str, Any]:
    normalized_at = _normalize_kst_datetime_filter(at) if at else None
    range_row = conn.execute(
        """
        SELECT
            COUNT(*) AS total_rows,
            MIN(collected_at_kst) AS min_collected_at_kst,
            MAX(collected_at_kst) AS max_collected_at_kst
        FROM model_hourly_features
        """
    ).fetchone()
    input_filters = {
        "snapshotId": snapshot_id,
        "at": at,
        "normalizedAtKst": normalized_at,
        "routeName": route_name,
        "stationId": station_id,
        "stationName": station_name,
        "remainSeatCnt": remain_seat_cnt,
    }
    data_range = {
        "totalRows": range_row["total_rows"] if range_row else 0,
        "minCollectedAtKst": range_row["min_collected_at_kst"] if range_row else None,
        "maxCollectedAtKst": range_row["max_collected_at_kst"] if range_row else None,
    }
    context: dict[str, Any] = {
        "message": "조건에 맞는 model_hourly_features row가 없습니다. rows=0은 만차 예측이 아니라 입력 데이터 없음입니다.",
        "inputFilters": input_filters,
        "availableDataRange": data_range,
    }
    if normalized_at:
        date_prefix = normalized_at[:10]
        date_count = conn.execute(
            "SELECT COUNT(*) AS count FROM model_hourly_features WHERE collected_at_kst LIKE ?",
            (f"{date_prefix}%",),
        ).fetchone()
        context["requestedDateRows"] = {
            "date": date_prefix,
            "rows": date_count["count"] if date_count else 0,
        }

    latest_conditions = []
    latest_params: dict[str, Any] = {"limit": max(1, min(int(limit), 10))}
    if route_name:
        latest_conditions.append("canonical_route_name = :route_name")
        latest_params["route_name"] = route_name
    if station_id is not None:
        latest_conditions.append("station_id = :station_id")
        latest_params["station_id"] = station_id
    if station_name:
        latest_conditions.append("station_name = :station_name")
        latest_params["station_name"] = station_name
    latest_where = f"WHERE {' AND '.join(latest_conditions)}" if latest_conditions else ""
    latest_rows = conn.execute(
        f"""
        SELECT
            snapshot_id,
            collected_at_kst,
            canonical_route_name,
            station_id,
            station_name,
            station_seq,
            remain_seat_cnt,
            crowded_label
        FROM model_hourly_features
        {latest_where}
        ORDER BY collected_at_kst DESC, snapshot_id DESC
        LIMIT :limit
        """,
        latest_params,
    ).fetchall()
    context["latestMatchingRowsIgnoringAt"] = rows_to_dicts(latest_rows)
    return context


def _training_dataset_rows(
    conn: sqlite3.Connection,
    target_column: str | None = None,
    *,
    exact_weather_only: bool = False,
) -> list[dict[str, Any]]:
    conditions = []
    if target_column:
        flag_column = ALLOWED_TARGET_COLUMNS[target_column]
        if flag_column:
            conditions.append(f"tl.{flag_column} = 1")
    if exact_weather_only:
        conditions.append("mf.weather_imputed = 0")
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT
            mf.snapshot_id,
            mf.collected_at_kst,
            mf.observed_hour_kst,
            mf.service_date,
            mf.day_of_week,
            mf.day_name_ko,
            mf.is_weekend,
            mf.is_holiday,
            mf.holiday_name,
            mf.hour,
            mf.time_bucket_10m,
            mf.time_period,
            mf.route_id,
            mf.route_name,
            mf.canonical_route_name,
            mf.route_type_cd,
            mf.route_type_name,
            mf.veh_id,
            mf.plate_no,
            mf.station_id,
            mf.station_seq,
            mf.station_name,
            mf.mobile_no,
            mf.x,
            mf.y,
            mf.remain_seat_cnt,
            mf.crowded,
            mf.crowded_label,
            mf.estimated_capacity,
            mf.seat_scarcity_score,
            mf.is_no_seat,
            mf.is_low_seat_2,
            mf.is_low_seat_5,
            COALESCE(eb.expected_boardings_at_stop, 0.0) AS expected_boardings_at_stop,
            CASE WHEN eb.expected_boardings_at_stop IS NULL THEN 1 ELSE 0 END AS expected_boardings_missing,
            mf.temperature,
            mf.precipitation,
            mf.humidity,
            mf.wind_speed,
            mf.cloud_amount,
            mf.weather_text,
            mf.weather_imputed,
            mf.pm10,
            mf.pm25,
            mf.o3,
            mf.khai,
            mf.air_quality_grade,
            mf.air_quality_imputed,
            mf.avg_speed,
            mf.traffic_volume,
            mf.delay_time,
            mf.congestion_level,
            mf.traffic_imputed,
            mf.event_count,
            mf.event_nearby_count,
            mf.event_imputed,
            tl.target_no_seat_now,
            tl.target_low_seat_2_now,
            tl.target_low_seat_5_now,
            tl.target_no_seat_next_5min,
            tl.target_low_seat_2_next_5min,
            tl.target_no_seat_next_10min,
            tl.target_low_seat_2_next_10min,
            tl.target_no_seat_next_station,
            tl.target_low_seat_2_next_station,
            tl.has_future_5min,
            tl.has_future_10min,
            tl.has_next_station
        FROM model_hourly_features mf
        JOIN model_target_labels tl ON tl.snapshot_id = mf.snapshot_id
        LEFT JOIN station_expected_boardings_hourly eb
          ON eb.station_id = mf.station_id
         AND eb.route_id = mf.route_id
         AND eb.day_of_week = mf.day_of_week
         AND eb.hour = mf.hour
        {where}
        ORDER BY mf.collected_at_kst, mf.canonical_route_name, mf.station_seq, mf.snapshot_id
    """
    return rows_to_dicts(conn.execute(query).fetchall())


def _normalize_kst_datetime_filter(value: str) -> str:
    text = value.strip()
    if "T" not in text and " " in text:
        text = text.replace(" ", "T", 1)
    if "+" not in text and not text.endswith("Z"):
        text = f"{text}+09:00"
    return text


def _insert_baseline_metrics(conn: sqlite3.Connection, evaluation: BaselineEvaluation) -> None:
    conn.execute(
        """
        INSERT INTO baseline_model_metrics (
            target_column, total_rows, train_rows, test_rows, positive_rate,
            accuracy, precision, recall, f1, roc_auc, strategy, metrics_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            evaluation.target_column,
            evaluation.total_rows,
            evaluation.train_rows,
            evaluation.test_rows,
            evaluation.positive_rate,
            evaluation.accuracy,
            evaluation.precision,
            evaluation.recall,
            evaluation.f1,
            evaluation.roc_auc,
            evaluation.strategy,
            json.dumps(evaluation.to_dict(), ensure_ascii=False),
            utc_now_iso(),
        ),
    )
    conn.commit()


def _insert_trained_model_metrics(conn: sqlite3.Connection, evaluation: LogisticModelEvaluation) -> None:
    conn.execute(
        """
        INSERT INTO trained_model_metrics (
            model_name, target_column, total_rows, train_rows, test_rows, positive_rate,
            accuracy, precision, recall, f1, roc_auc, strategy, metrics_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            evaluation.model_name,
            evaluation.target_column,
            evaluation.total_rows,
            evaluation.train_rows,
            evaluation.test_rows,
            evaluation.positive_rate,
            evaluation.accuracy,
            evaluation.precision,
            evaluation.recall,
            evaluation.f1,
            evaluation.roc_auc,
            evaluation.strategy,
            json.dumps(evaluation.to_dict(), ensure_ascii=False),
            utc_now_iso(),
        ),
    )
    conn.commit()


def _classification_metrics(actuals: list[int], predictions: list[int], probabilities: list[float]) -> dict[str, float]:
    total = len(actuals)
    tp = sum(1 for actual, predicted in zip(actuals, predictions) if actual == 1 and predicted == 1)
    tn = sum(1 for actual, predicted in zip(actuals, predictions) if actual == 0 and predicted == 0)
    fp = sum(1 for actual, predicted in zip(actuals, predictions) if actual == 0 and predicted == 1)
    fn = sum(1 for actual, predicted in zip(actuals, predictions) if actual == 1 and predicted == 0)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    brier = sum((probability - actual) ** 2 for actual, probability in zip(actuals, probabilities)) / total if total else 0.0
    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "roc_auc": round(_roc_auc(actuals, probabilities), 6),
        "brier_score": round(brier, 6),
        "top_decile_precision": round(_top_decile_precision(actuals, probabilities), 6),
    }


def _roc_auc(actuals: list[int], probabilities: list[float]) -> float:
    positives = sum(actuals)
    negatives = len(actuals) - positives
    if positives == 0 or negatives == 0:
        return 0.5

    pairs = sorted(zip(probabilities, actuals), key=lambda pair: pair[0])
    ranks = [0.0] * len(pairs)
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for rank_index in range(index, end):
            ranks[rank_index] = average_rank
        index = end

    sum_positive_ranks = sum(rank for rank, (_, actual) in zip(ranks, pairs) if actual == 1)
    return (sum_positive_ranks - positives * (positives + 1) / 2.0) / (positives * negatives)


def _top_decile_precision(actuals: list[int], probabilities: list[float]) -> float:
    if not actuals:
        return 0.0
    top_n = max(1, math.ceil(len(actuals) * 0.1))
    top_rows = sorted(zip(probabilities, actuals), key=lambda pair: pair[0], reverse=True)[:top_n]
    return sum(actual for _, actual in top_rows) / len(top_rows)


def _group_rate(rows: list[dict[str, Any]], target_column: str, columns: tuple[str, ...]) -> dict[tuple[Any, ...], float]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[column] for column in columns)
        groups.setdefault(key, []).append(row)
    return {key: _smoothed_rate(group_rows, target_column) for key, group_rows in groups.items()}


def _smoothed_rate(rows: list[dict[str, Any]], target_column: str, alpha: float = 1.0) -> float:
    positives = sum(int(row[target_column]) for row in rows)
    return (positives + alpha) / (len(rows) + 2 * alpha)


def _validate_target_column(target_column: str) -> None:
    if target_column not in ALLOWED_TARGET_COLUMNS:
        allowed = ", ".join(sorted(ALLOWED_TARGET_COLUMNS))
        raise ValueError(f"지원하지 않는 target 컬럼입니다: {target_column}. 허용값: {allowed}")


def _fit_logistic_sgd(
    x_train: list[list[float]],
    y_train: list[int],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> list[float]:
    if not x_train:
        return [0.0]
    feature_count = len(x_train[0])
    weights = [0.0] * (feature_count + 1)
    positives = sum(y_train)
    negatives = len(y_train) - positives
    pos_weight = len(y_train) / (2 * positives) if positives else 1.0
    neg_weight = len(y_train) / (2 * negatives) if negatives else 1.0
    order = list(range(len(x_train)))
    rng = random.Random(20260524)
    for epoch in range(max(1, epochs)):
        rng.shuffle(order)
        lr = learning_rate / math.sqrt(epoch + 1)
        for index in order:
            features = x_train[index]
            actual = y_train[index]
            probability = _predict_logistic(weights, features)
            sample_weight = pos_weight if actual else neg_weight
            error = (probability - actual) * sample_weight
            weights[0] -= lr * error
            for feature_index, value in enumerate(features, start=1):
                penalty = l2 * weights[feature_index]
                weights[feature_index] -= lr * (error * value + penalty)
    return weights


def _predict_logistic(weights: list[float], features: list[float]) -> float:
    z = weights[0] + sum(weight * value for weight, value in zip(weights[1:], features))
    if z >= 0:
        exp_neg = math.exp(-min(z, 60))
        return 1 / (1 + exp_neg)
    exp_pos = math.exp(max(z, -60))
    return exp_pos / (1 + exp_pos)


def _numeric_value(row: dict[str, Any], name: str) -> float:
    if name == "hour_sin":
        return math.sin(2 * math.pi * _to_float(row.get("hour"), 0.0) / 24.0)
    if name == "hour_cos":
        return math.cos(2 * math.pi * _to_float(row.get("hour"), 0.0) / 24.0)
    if name == "day_sin":
        return math.sin(2 * math.pi * _to_float(row.get("day_of_week"), 0.0) / 7.0)
    if name == "day_cos":
        return math.cos(2 * math.pi * _to_float(row.get("day_of_week"), 0.0) / 7.0)
    return _to_float(row.get(name), 0.0)


def _categorical_value(row: dict[str, Any], name: str) -> str:
    value = row.get(name)
    if value is None or value == "":
        return "unknown"
    return str(value)


def _missing_rate(rows: list[dict[str, Any]], flag_column: str) -> float:
    if not rows:
        return 1.0
    return sum(_to_float(row.get(flag_column), 1.0) for row in rows) / len(rows)


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    try:
        file = path.open("w", encoding="utf-8-sig", newline="")
    except PermissionError:
        fallback = path.with_name(f"{path.stem}.new{path.suffix}")
        file = fallback.open("w", encoding="utf-8-sig", newline="")
    with file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
