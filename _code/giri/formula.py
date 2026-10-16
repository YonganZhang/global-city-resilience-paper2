"""
Pipeline v25 · 通用公式与常量（单一来源）

所有 step 脚本共享的 smoothstep / P99 / denominator helper。
不再各自 define。
"""
from __future__ import annotations
import numpy as np


def smoothstep(x):
    """平滑归一到 [0,1]。公式:
       2x²    for x ∈ [0, 0.5]
       1-2(1-x)²  for x ∈ (0.5, 1]
    """
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.5, 2 * x * x, 1 - 2 * (1 - x) ** 2)


# 6 灾害 99 百分位归一参考
P99 = {
    "EQ": 400.0,
    "FL": 600.0,
    "TC": 300.0,
    "DR": 20.0,
    "LS_EQ": 4.0,
    "LS_rain": 4.0,
}
HAZ_KEYS = list(P99.keys())


def hazard_rf(hazard_key: str, weighted_value: float) -> float:
    """把原始 weighted_mean 做 P99 归一 + smoothstep → [0,1]"""
    return float(smoothstep(min(weighted_value / P99[hazard_key], 1.0)))


def compute_composite(zonal_row, haz_keys=None) -> float:
    """从 zonal row 里算 composite = mean(6 灾害 rf)"""
    haz_keys = haz_keys or HAZ_KEYS
    rf = [hazard_rf(h, float(zonal_row.get(f"{h}_weighted_mean", 0) or 0)) for h in haz_keys]
    return float(np.mean(rf))


__all__ = ["smoothstep", "P99", "HAZ_KEYS", "hazard_rf", "compute_composite"]
