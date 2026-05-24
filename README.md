# BusSeat AI MVP

경기버스 실시간 도착/위치 데이터와 정류소 수요, 날씨, 공휴일, 교통요인을 결합해 **다음 버스 만차 위험도**를 계산하는 공공데이터 공모전용 MVP입니다.

현재 구현된 범위:

```text
1. GBIS v2 API 클라이언트
2. 정류소/노선 검색
3. 도착정보/위치정보 조회
4. SQLite 스냅샷 적재
5. LightGBM artifact 기반 만차확률 예측
6. 기상청 격자 좌표 변환
7. CLI와 FastAPI 엔드포인트
```

## 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
```

`.env`에 운영용 키를 넣습니다.
`data.go.kr` 계열 API는 `PUBLIC_DATA_SERVICE_KEY` 하나로 통일해서 사용합니다.

```text
PUBLIC_DATA_SERVICE_KEY=...
ITS_API_KEY=...
DATA_GG_SERVICE_KEY=...
BUSSEAT_DB_PATH=data/busseat.db
PUBLIC_DATA_TIMEOUT_SECONDS=10
```

## DB 초기화

```powershell
busseat init-db
```

## CLI 사용

```powershell
busseat search-station 명지대입구
busseat search-route 5000
busseat arrivals 228000719
busseat locations 234000016
busseat predict-live-risk 102000070 --route-name 5005 --at "2026-05-25 08:00:00" --output-json data/live_5005_20260525_0800.json
busseat kma-grid 127.0284667 37.49545
```

용인 5000계열 실데이터 파이프라인:

```powershell
busseat prepare-target-routes --region-keyword 용인
busseat validate-api
busseat collect-target-locations
busseat preprocess-location-features --export-csv data/preprocessed_location_features.csv
busseat build-model-hourly-features --export-csv data/model_hourly_features.csv
busseat build-target-labels
busseat export-training-dataset --export-csv data/training_dataset.csv
busseat data-quality
busseat diagnose-model --model lightgbm --output-json data/model_diagnostics.json --output-md data/model_diagnostics.md
busseat train-lightgbm-artifact --output-artifact data/risk_model_lightgbm.pkl --output-json data/risk_model_lightgbm_artifact.json
```

장기 수집을 시작할 때는 반복 수집 명령을 씁니다.

```powershell
busseat collect-loop --interval 60 --preprocess --build-model-features
```

## API 서버

```powershell
uvicorn busseat_ai.api:app --reload --port 8000
```

주요 엔드포인트:

```text
GET /health
GET /stations/search?keyword=명지대입구
GET /routes/search?keyword=5000
GET /arrivals/{station_id}
GET /locations/{route_id}
GET /predict/live-risk?station_id=102000070&route_name=5005&at=2026-05-25%2008:00:00
GET /weather/grid?lon=127.0284667&lat=37.49545
```

## 테스트

```powershell
python -m unittest discover -s tests
```

## 레포 업로드 전 주의사항

공개/공유 레포에는 원본 데이터와 실행 산출물을 올리지 않습니다.

올리지 않는 파일:

```text
.env
sureing_occupancy_backup.sql
data/busseat.db
data/busseat.before_*.db
data/*.pkl
data/*training*.csv
data/*features*.csv
data/live_*.json
data/hourly_risk_*.csv
data/hourly_risk_*.json
```

이 파일들은 `.gitignore`에 등록되어 있습니다. API 키는 `.env.example`만 올리고 실제 키가 들어간 `.env`는 올리지 않습니다.

업로드 전에는 아래를 확인합니다.

```powershell
$env:PYTHONPATH='src'
python -B -m unittest discover -s tests
rg -n "PUBLIC_DATA_SERVICE_KEY=.+|ITS_API_KEY=.+|DATA_GG_SERVICE_KEY=.+|serviceKey=|apiKey=|KEY=" -S . -g "!data/**" -g "!sureing_occupancy_backup.sql" -g "!**/__pycache__/**"
```

보고서 원본과 발표용 문서는 로컬 `보고서/` 디렉터리에 따로 보관하며, 해당 디렉터리는 `.gitignore`로 제외합니다.

## 다음 개발 순서

```text
1. 관심 노선/정류소 3~5개 선정
2. 1분 간격 위치/도착 스냅샷 수집
3. 대기질/교통/행사 데이터 적재
4. 장기 수집 데이터로 모델 재학습
5. LightGBM 비교 진단과 calibration 보강
6. 발표용 데모 화면 제작
```

API 키 없이 전처리 결과를 확인하려면 예시 DB/CSV를 생성합니다.

```powershell
python scripts/build_example_db.py
```

산출물은 [example-db](example-db) 폴더에 생성됩니다.

## 만차확률 계산 원칙

현재 코드는 옛 수식 기반 직접 계산이나 `날씨 +8점` 같은 점수 합산을 쓰지 않습니다.

현재 운영 경로는 아래 하나입니다.

```text
1. GBIS 실시간 도착/위치 row 또는 시간표/과거이력 proxy row 생성
2. model_hourly_features와 같은 feature schema로 변환
3. data/risk_model_lightgbm.pkl artifact 로드
4. LightGBM predict_proba로 target_no_seat_next_station 확률 계산
5. fullSeatProbability, fullSeatRiskScore, riskLevel 출력
```

모델에 들어가는 주요 feature 그룹:

```text
bus_state: remainSeatCnt, crowded, seatScarcityScore 등
route: routeId, routeName, routeTypeCd
station_seq: stationSeq, stationSeqSegment
time: hour, weekday, holiday
weather: 기온/강수/습도 등 수집된 날씨 feature
event: 행사/공휴일 feature
```

`seatScarcityScore`는 잔여좌석을 부드럽게 표현하기 위한 feature/표시값입니다. 최종 만차확률은 이 점수를 직접 확률로 바꾸는 게 아니라 LightGBM 모델이 다른 feature와 함께 판단합니다.

```text
seatScarcityScore
  = 100 * (1 - (ln(remainSeatCnt + 1) / ln(seatCapacity + 1)) ^ 2.15)
```
