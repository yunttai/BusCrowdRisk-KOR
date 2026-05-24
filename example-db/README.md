# Example DB

이 폴더는 API 키 없이도 전처리 결과가 어떻게 생기는지 볼 수 있는 샘플 산출물 위치입니다.

생성 명령:

```powershell
python scripts/build_example_db.py
```

생성 파일:

```text
busseat-example.db
target_route.csv
route_station.csv
bus_location_snapshot.csv
station_demand_daily.csv
station_expected_boardings_hourly.csv
external_weather_hourly.csv
external_air_quality_hourly.csv
external_traffic_hourly.csv
external_holiday_daily.csv
external_event_daily.csv
preprocessed_location_features.csv
model_hourly_features.csv
model_target_labels.csv
training_dataset.csv
baseline_metrics.json
baseline_model_metrics.csv
data_quality_report.json
route_station_summary.csv
feature_summary_by_route_time.csv
model_summary_by_route_time.csv
```

## 파일별 의미

| 파일 | 의미 |
| --- | --- |
| `busseat-example.db` | 샘플 SQLite DB |
| `target_route.csv` | 대상 노선 8개의 routeId와 노선명 |
| `route_station.csv` | 각 노선의 전체 경유정류소 샘플 |
| `bus_location_snapshot.csv` | 위치정보 API 원본 형태의 샘플 스냅샷 |
| `station_demand_daily.csv` | 정류소별 일별 승하차 수요 샘플 |
| `station_expected_boardings_hourly.csv` | 정류소/노선/요일/시간별 예상 승차수 |
| `external_weather_hourly.csv` | 시간 단위 기상 샘플 |
| `external_air_quality_hourly.csv` | 시간 단위 대기질 샘플 |
| `external_traffic_hourly.csv` | 시간 단위 교통 샘플 |
| `external_holiday_daily.csv` | 날짜 단위 공휴일 샘플 |
| `external_event_daily.csv` | 날짜 단위 행사 샘플 |
| `preprocessed_location_features.csv` | 요일/시간대/정류소/잔여좌석 feature 전처리 결과 |
| `model_hourly_features.csv` | 버스 feature와 날씨/대기질/교통/행사/공휴일을 결합한 최종 학습용 샘플 |
| `model_target_labels.csv` | 현재/5분/10분/다음정류소 기준 학습 target 라벨 |
| `training_dataset.csv` | feature, expected_boardings, target 라벨을 결합한 학습 CSV |
| `baseline_metrics.json` | 시간순 train/test split 베이스라인 평가 결과 |
| `baseline_model_metrics.csv` | 베이스라인 평가 이력 테이블 |
| `data_quality_report.json` | imputed 비율과 추천/제외 feature 리포트 |
| `route_station_summary.csv` | 노선별 정류소 수 요약 |
| `feature_summary_by_route_time.csv` | 노선/시간대별 잔여좌석, 만차, 저좌석 요약 |
| `model_summary_by_route_time.csv` | 노선/시간대별 잔여좌석과 외부요인 결합 요약 |

현재 샘플 기준 행 수:

```text
target_route.csv                  8행
route_station.csv                 48행
bus_location_snapshot.csv         128행
station_demand_daily.csv          12행
station_expected_boardings_hourly.csv 8064행
preprocessed_location_features.csv 128행
model_hourly_features.csv          128행
model_target_labels.csv            128행
training_dataset.csv               128행
baseline_model_metrics.csv         1행
```

`preprocessed_location_features.csv`에서 먼저 볼 컬럼:

```text
collected_at_kst
day_name_ko
time_period
canonical_route_name
station_seq
station_name
remain_seat_cnt
crowded_label
seat_scarcity_score
is_no_seat
is_low_seat_2
is_low_seat_5
```

외부요인까지 결합된 최종 결과는 `model_hourly_features.csv`에서 확인합니다.

모델 학습 직전 데이터는 `training_dataset.csv`에서 봅니다.
여기에는 `expected_boardings_at_stop`과 `expected_boardings_missing`이 같이 들어가므로,
승하차 원자료가 부족한 구간을 모델 학습에서 제외하거나 별도 처리할 수 있습니다.
