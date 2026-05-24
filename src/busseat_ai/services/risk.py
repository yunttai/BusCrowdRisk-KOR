from __future__ import annotations

import math


CROWDED_LABELS = {
    1: "여유",
    2: "보통",
    3: "혼잡",
    4: "매우혼잡",
}

ROUTE_TYPE_LABELS = {
    11: "직행좌석형시내버스",
    12: "좌석형시내버스",
    13: "일반형시내버스",
    14: "광역급행형시내버스",
    15: "따복형시내버스",
    16: "경기순환버스",
    17: "준공영제직행좌석시내버스",
    21: "직행좌석형농어촌버스",
    22: "좌석형농어촌버스",
    23: "일반형농어촌버스",
    30: "마을버스",
    41: "고속형시외버스",
    42: "좌석형시외버스",
    43: "일반형시외버스",
    51: "리무진공항버스",
    52: "좌석형공항버스",
    53: "일반형공항버스",
}

DEFAULT_SEAT_CAPACITY_BY_ROUTE_TYPE = {
    11: 45,
    12: 45,
    14: 45,
    16: 45,
    17: 45,
    21: 45,
    22: 45,
    41: 45,
    42: 45,
    51: 45,
    52: 45,
}

DEFAULT_REFERENCE_SEAT_CAPACITY = 45
SEAT_SCARCITY_GAMMA = 2.15


def estimate_capacity(route_type_cd: int | None, observed_capacity: int | None = None) -> int | None:
    if observed_capacity and observed_capacity > 0:
        return observed_capacity
    if route_type_cd is None:
        return None
    return DEFAULT_SEAT_CAPACITY_BY_ROUTE_TYPE.get(route_type_cd)


def calculate_seat_scarcity_score(
    remain_seat_cnt: int,
    capacity: int | None = None,
    gamma: float = SEAT_SCARCITY_GAMMA,
) -> int:
    """Smooth remaining-seat scarcity score used as a feature/display value.

    This is not the final full-seat probability. The final probability is
    produced by the trained LightGBM artifact in ``busseat_ai.modeling``.
    """
    remain = max(0, int(remain_seat_cnt))
    if remain == 0:
        return 100

    reference_capacity = capacity if capacity and capacity > 0 else DEFAULT_REFERENCE_SEAT_CAPACITY
    reference_capacity = max(reference_capacity, remain)
    if remain >= reference_capacity:
        return 0

    ratio = math.log1p(remain) / math.log1p(reference_capacity)
    score = 100 * (1 - (ratio**gamma))
    return int(round(_bounded(score, 0, 100)))


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
