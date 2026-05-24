CREATE TABLE IF NOT EXISTS bus_arrival_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT NOT NULL,
    station_id INTEGER,
    route_id INTEGER,
    route_name TEXT,
    route_type_cd INTEGER,
    sta_order INTEGER,
    predict_time1 INTEGER,
    predict_time2 INTEGER,
    predict_time_sec1 INTEGER,
    predict_time_sec2 INTEGER,
    remain_seat_cnt1 INTEGER,
    remain_seat_cnt2 INTEGER,
    crowded1 INTEGER,
    crowded2 INTEGER,
    plate_no1 TEXT,
    plate_no2 TEXT,
    veh_id1 INTEGER,
    veh_id2 INTEGER,
    raw_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_arrival_station_route_time
ON bus_arrival_snapshot (station_id, route_id, collected_at);

CREATE TABLE IF NOT EXISTS bus_location_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT NOT NULL,
    route_id INTEGER,
    veh_id INTEGER,
    plate_no TEXT,
    station_id INTEGER,
    station_seq INTEGER,
    remain_seat_cnt INTEGER,
    crowded INTEGER,
    route_type_cd INTEGER,
    low_plate INTEGER,
    state_cd INTEGER,
    tagless_cd INTEGER,
    raw_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_location_route_vehicle_time
ON bus_location_snapshot (route_id, veh_id, collected_at);

CREATE TABLE IF NOT EXISTS station_demand_daily (
    service_date TEXT NOT NULL,
    station_id INTEGER NOT NULL,
    mobile_no TEXT,
    station_name TEXT,
    city_name TEXT,
    boarding_total INTEGER,
    first_boarding_total INTEGER,
    transfer_total INTEGER,
    alighting_total INTEGER,
    PRIMARY KEY (service_date, station_id)
);

CREATE TABLE IF NOT EXISTS route_cache (
    route_id INTEGER PRIMARY KEY,
    route_name TEXT,
    route_type_cd INTEGER,
    route_type_name TEXT,
    region_name TEXT,
    start_station_name TEXT,
    end_station_name TEXT,
    raw_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS station_cache (
    station_id INTEGER PRIMARY KEY,
    station_name TEXT,
    mobile_no TEXT,
    x REAL,
    y REAL,
    region_name TEXT,
    raw_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS target_route (
    route_id INTEGER PRIMARY KEY,
    canonical_route_name TEXT NOT NULL,
    route_name TEXT,
    route_type_cd INTEGER,
    route_type_name TEXT,
    region_name TEXT,
    start_station_name TEXT,
    end_station_name TEXT,
    raw_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_target_route_canonical
ON target_route (canonical_route_name);

CREATE TABLE IF NOT EXISTS route_station (
    route_id INTEGER NOT NULL,
    station_seq INTEGER NOT NULL,
    station_id INTEGER,
    canonical_route_name TEXT,
    station_name TEXT,
    mobile_no TEXT,
    x REAL,
    y REAL,
    region_name TEXT,
    center_yn TEXT,
    turn_yn TEXT,
    raw_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (route_id, station_seq)
);

CREATE INDEX IF NOT EXISTS idx_route_station_station
ON route_station (station_id);

CREATE TABLE IF NOT EXISTS preprocessed_location_features (
    snapshot_id INTEGER PRIMARY KEY,
    collected_at_utc TEXT NOT NULL,
    collected_at_kst TEXT NOT NULL,
    service_date TEXT NOT NULL,
    day_of_week INTEGER NOT NULL,
    day_name_ko TEXT NOT NULL,
    is_weekend INTEGER NOT NULL,
    hour INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    time_bucket_10m TEXT NOT NULL,
    time_period TEXT NOT NULL,
    route_id INTEGER,
    route_name TEXT,
    canonical_route_name TEXT,
    route_type_cd INTEGER,
    route_type_name TEXT,
    veh_id INTEGER,
    plate_no TEXT,
    station_id INTEGER,
    station_seq INTEGER,
    station_name TEXT,
    mobile_no TEXT,
    x REAL,
    y REAL,
    remain_seat_cnt INTEGER,
    crowded INTEGER,
    crowded_label TEXT,
    state_cd INTEGER,
    low_plate INTEGER,
    tagless_cd INTEGER,
    estimated_capacity INTEGER,
    seat_scarcity_score INTEGER,
    is_no_seat INTEGER,
    is_low_seat_2 INTEGER,
    is_low_seat_5 INTEGER,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_preprocessed_location_route_station_time
ON preprocessed_location_features (route_id, station_seq, collected_at_kst);

CREATE INDEX IF NOT EXISTS idx_preprocessed_location_time
ON preprocessed_location_features (service_date, hour, time_bucket_10m);

CREATE TABLE IF NOT EXISTS external_weather_hourly (
    weather_station_id TEXT NOT NULL,
    weather_station_name TEXT,
    observed_at_kst TEXT NOT NULL,
    service_date TEXT NOT NULL,
    hour INTEGER NOT NULL,
    temperature REAL NOT NULL,
    precipitation REAL NOT NULL,
    humidity REAL NOT NULL,
    wind_speed REAL NOT NULL,
    cloud_amount REAL NOT NULL,
    weather_text TEXT NOT NULL,
    source TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (weather_station_id, observed_at_kst)
);

CREATE INDEX IF NOT EXISTS idx_external_weather_hourly_time
ON external_weather_hourly (observed_at_kst, service_date, hour);

CREATE TABLE IF NOT EXISTS external_air_quality_hourly (
    air_station_name TEXT NOT NULL,
    observed_at_kst TEXT NOT NULL,
    service_date TEXT NOT NULL,
    hour INTEGER NOT NULL,
    pm10 REAL NOT NULL,
    pm25 REAL NOT NULL,
    o3 REAL NOT NULL,
    khai REAL NOT NULL,
    air_quality_grade TEXT NOT NULL,
    source TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (air_station_name, observed_at_kst)
);

CREATE INDEX IF NOT EXISTS idx_external_air_quality_hourly_time
ON external_air_quality_hourly (observed_at_kst, service_date, hour);

CREATE TABLE IF NOT EXISTS external_traffic_hourly (
    traffic_context_key TEXT NOT NULL,
    observed_at_kst TEXT NOT NULL,
    service_date TEXT NOT NULL,
    hour INTEGER NOT NULL,
    avg_speed REAL NOT NULL,
    traffic_volume REAL NOT NULL,
    delay_time REAL NOT NULL,
    congestion_level INTEGER NOT NULL,
    source TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (traffic_context_key, observed_at_kst)
);

CREATE INDEX IF NOT EXISTS idx_external_traffic_hourly_time
ON external_traffic_hourly (observed_at_kst, service_date, hour);

CREATE TABLE IF NOT EXISTS external_holiday_daily (
    service_date TEXT PRIMARY KEY,
    is_holiday INTEGER NOT NULL,
    holiday_name TEXT NOT NULL,
    source TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS external_event_daily (
    area_key TEXT NOT NULL,
    service_date TEXT NOT NULL,
    event_count INTEGER NOT NULL,
    event_nearby_count INTEGER NOT NULL,
    source TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (area_key, service_date)
);

CREATE TABLE IF NOT EXISTS model_hourly_features (
    snapshot_id INTEGER PRIMARY KEY,
    collected_at_kst TEXT NOT NULL,
    observed_hour_kst TEXT NOT NULL,
    service_date TEXT NOT NULL,
    day_of_week INTEGER NOT NULL,
    day_name_ko TEXT NOT NULL,
    is_weekend INTEGER NOT NULL,
    is_holiday INTEGER NOT NULL,
    holiday_name TEXT NOT NULL,
    hour INTEGER NOT NULL,
    time_bucket_10m TEXT NOT NULL,
    time_period TEXT NOT NULL,
    route_id INTEGER NOT NULL,
    route_name TEXT NOT NULL,
    canonical_route_name TEXT NOT NULL,
    route_type_cd INTEGER NOT NULL,
    route_type_name TEXT NOT NULL,
    veh_id INTEGER NOT NULL,
    plate_no TEXT NOT NULL,
    station_id INTEGER NOT NULL,
    station_seq INTEGER NOT NULL,
    station_name TEXT NOT NULL,
    mobile_no TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    remain_seat_cnt INTEGER NOT NULL,
    crowded INTEGER NOT NULL,
    crowded_label TEXT NOT NULL,
    estimated_capacity INTEGER NOT NULL,
    seat_scarcity_score INTEGER NOT NULL,
    is_no_seat INTEGER NOT NULL,
    is_low_seat_2 INTEGER NOT NULL,
    is_low_seat_5 INTEGER NOT NULL,
    temperature REAL NOT NULL,
    precipitation REAL NOT NULL,
    humidity REAL NOT NULL,
    wind_speed REAL NOT NULL,
    cloud_amount REAL NOT NULL,
    weather_text TEXT NOT NULL,
    weather_imputed INTEGER NOT NULL,
    pm10 REAL NOT NULL,
    pm25 REAL NOT NULL,
    o3 REAL NOT NULL,
    khai REAL NOT NULL,
    air_quality_grade TEXT NOT NULL,
    air_quality_imputed INTEGER NOT NULL,
    avg_speed REAL NOT NULL,
    traffic_volume REAL NOT NULL,
    delay_time REAL NOT NULL,
    congestion_level INTEGER NOT NULL,
    traffic_imputed INTEGER NOT NULL,
    event_count INTEGER NOT NULL,
    event_nearby_count INTEGER NOT NULL,
    event_imputed INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_hourly_features_route_time
ON model_hourly_features (canonical_route_name, observed_hour_kst, station_seq);

CREATE INDEX IF NOT EXISTS idx_model_hourly_features_date_hour
ON model_hourly_features (service_date, hour);

CREATE TABLE IF NOT EXISTS model_target_labels (
    snapshot_id INTEGER PRIMARY KEY,
    route_id INTEGER NOT NULL,
    canonical_route_name TEXT NOT NULL,
    veh_id INTEGER NOT NULL,
    plate_no TEXT NOT NULL,
    station_seq INTEGER NOT NULL,
    collected_at_kst TEXT NOT NULL,
    target_no_seat_now INTEGER NOT NULL,
    target_low_seat_2_now INTEGER NOT NULL,
    target_low_seat_5_now INTEGER NOT NULL,
    target_no_seat_next_5min INTEGER NOT NULL,
    target_low_seat_2_next_5min INTEGER NOT NULL,
    target_no_seat_next_10min INTEGER NOT NULL,
    target_low_seat_2_next_10min INTEGER NOT NULL,
    target_no_seat_next_station INTEGER NOT NULL,
    target_low_seat_2_next_station INTEGER NOT NULL,
    has_future_5min INTEGER NOT NULL,
    has_future_10min INTEGER NOT NULL,
    has_next_station INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_target_labels_route_time
ON model_target_labels (canonical_route_name, collected_at_kst, station_seq);

CREATE TABLE IF NOT EXISTS station_expected_boardings_hourly (
    station_id INTEGER NOT NULL,
    route_id INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,
    hour INTEGER NOT NULL,
    station_daily_boarding_avg REAL NOT NULL,
    hour_demand_ratio REAL NOT NULL,
    route_share REAL NOT NULL,
    vehicles_per_hour REAL NOT NULL,
    expected_boardings_at_stop REAL NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (station_id, route_id, day_of_week, hour)
);

CREATE INDEX IF NOT EXISTS idx_expected_boardings_route_hour
ON station_expected_boardings_hourly (route_id, day_of_week, hour);

CREATE TABLE IF NOT EXISTS baseline_model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_column TEXT NOT NULL,
    total_rows INTEGER NOT NULL,
    train_rows INTEGER NOT NULL,
    test_rows INTEGER NOT NULL,
    positive_rate REAL NOT NULL,
    accuracy REAL NOT NULL,
    precision REAL NOT NULL,
    recall REAL NOT NULL,
    f1 REAL NOT NULL,
    roc_auc REAL NOT NULL,
    strategy TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trained_model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    target_column TEXT NOT NULL,
    total_rows INTEGER NOT NULL,
    train_rows INTEGER NOT NULL,
    test_rows INTEGER NOT NULL,
    positive_rate REAL NOT NULL,
    accuracy REAL NOT NULL,
    precision REAL NOT NULL,
    recall REAL NOT NULL,
    f1 REAL NOT NULL,
    roc_auc REAL NOT NULL,
    strategy TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
