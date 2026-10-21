"""r_ph_downscale.py — R_Ph 国家级 → 城市级 HEV 下推 (V28 真实现, 接真源 master_v27_6)

公式: R_Ph_city = R_Ph_national × H_ratio^ALPHA_H × E_ratio^ALPHA_E × V_ratio^ALPHA_V → soft cap
ALPHA: 0.8 / 0.3 / 0.3 (本研究经验设置，非GIRI官方权重；单一真源见 _config.py)

数据接口 — active master 列名 (v28_ucdb 5862 行 / v5_8 6275 城, 经 master_io 选表):
  R_Ph_national : R_Ph__RAW_nat                          (国家级, 同国同值)
  H_ratio       : haz_{EQ,FL,TC,DR,LS_EQ,LS_RAIN,TS,SS}__HAZ / *_nat_avg__HAZ 有效比值算术平均
  E_ratio       : exp_pop_ntl_city__RAW / exp_pop_ntl_nat_avg__RAW
  V_ratio       : 调用方提供的城市吸收缺口 / 国家吸收缺口；未提供时按既有回退规则处理

V28 (2026-05-28): 从薄壳 (反向 import 老 step1b) 重写为真实现.
  老 step1b 按 2026-04 老 master CSV 列名写 (R_Ph_national/hazard_ratio/country_iso3),
  跟真源对不上 → 直接调会 KeyError / 静默退化. 全部废弃.
  R_Ph 下推逻辑 (含原散在 master_io 的 ratio helper) 收拢到此单一真源.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from _config import ALPHA_H, ALPHA_E, ALPHA_V, E_RATIO_BOUNDS, R_PH_SOFT_CAP_THRESHOLD

# 8 灾害 (master haz_*__HAZ 列): 地震/洪水/热带气旋/干旱/地震滑坡/降雨滑坡/海啸/风暴潮
HAZARDS = ["EQ", "FL", "TC", "DR", "LS_EQ", "LS_RAIN", "TS", "SS"]
DOWNSCALE_METHODS = ("analytic", "trapezoidal")


# ══════════════════════════════════════════════════════════════════════
#  三个城市/国家比值 (单城; city_row = master 一行 dict/Series, 接真源列名)
# ══════════════════════════════════════════════════════════════════════

def compute_h_ratio(city_row) -> float:
    """H_ratio = 8 灾害 (haz/nat_avg) 算术平均. 无 hazard 数据 → 1.0.

    2026-06-03 用户拍板: 几何平均 → 算术平均.
      根因: 几何乘积对极小非零值过敏 (Tahoua FL=0.001 把主灾 DR=0.99 抹平到 H=0.18),
      但主灾 ratio 跟国均一样应该让 H≈1。算术平均让主灾自然占主导 (大值不被小值乘性拖死)。
      drop NaN (灾害稀疏的内陆无海啸等), 保留真零 (no exposure 也是真实信号)。
    """
    ratios = []
    for hz in HAZARDS:
        haz = city_row.get(f"haz_{hz}__HAZ")
        nat = city_row.get(f"haz_{hz}_nat_avg__HAZ")
        if haz is None or nat is None or pd.isna(haz) or pd.isna(nat) or nat <= 0:
            continue
        ratios.append(haz / nat)
    if not ratios:
        return 1.0
    return float(np.mean(ratios))


def compute_e_ratio(city_row, bounds: tuple[float, float] = E_RATIO_BOUNDS) -> float:
    """E_ratio = exp_pop_ntl_city / exp_pop_ntl_nat_avg (人口×夜光暴露). 缺 → 1.0.

    下上限保护 clip(bounds), 默认 (0.3, 3.0) (沿用老 step1b 既定口径): 小城 (VIIRS 夜光=0 → ratio=0)
    不应让 R_Ph_city 被压成 0 (物理风险不可能真零, 是数据缺失); 大城上限 3.0 防过放大.
    """
    pop_ntl = city_row.get("exp_pop_ntl_city__RAW")
    nat_avg = city_row.get("exp_pop_ntl_nat_avg__RAW")
    if pop_ntl is None or nat_avg is None or pd.isna(pop_ntl) or pd.isna(nat_avg) or nat_avg <= 0:
        return 1.0
    return float(min(max(pop_ntl / nat_avg, bounds[0]), bounds[1]))


def compute_v_ratio(city_abs_def_avg: Optional[float], nat_abs_def_avg: Optional[float]) -> float:
    """V_ratio = mean(ABS 6 城市 deficit) / mean(ABS 6 国家 deficit). >1 = 比国均更脆弱.

    议会跑完用: 议会 6 ABS indicator deficit (Gate 2 翻完) city level → mean → 比国家级.
    城市级缺 (议会前 / 6225 城无议会) → 1.0 (不调制, extrapolation 退化).

    V28: 签名从 (city_abs_def_avg, nation_iso3) 解耦为传两个数, 去掉每城 load_master_data 的
    低效 IO (旧版每调一次读一遍 parquet). nat_abs_def_avg 由批量层预算好传入.
    """
    if city_abs_def_avg is None or pd.isna(city_abs_def_avg):
        return 1.0
    if nat_abs_def_avg is None or pd.isna(nat_abs_def_avg) or nat_abs_def_avg <= 0:
        return 1.0
    # clip [0.1, 10] (对齐老 step1b / circle_sampling): 防顶尖治理国 (低 nat_def) 的
    # 高脆弱城 V 爆放大 (实测可达 12.7 → V^0.3=2.14× 放大, 超老上限). P5.2 接议会前必生效.
    return float(min(max(city_abs_def_avg / nat_abs_def_avg, 0.1), 10.0))


def apply_r_ph_soft_cap(raw: float) -> float:
    """soft cap: raw>0.95 渐进 1.0 防 megacity (Tokyo/HK/SH/NY) 全饱和失差异度.

    旧硬 clip min(raw,1.0) 让 megacity 全钉 1.0; soft cap 保留 1-3pt 物理风险谱差.
      raw <= 0.95: clip(raw, 0, 1)
      raw >  0.95: 0.95 + 0.05*(1 - exp(-(raw-0.95)/0.05))
    """
    t = R_PH_SOFT_CAP_THRESHOLD
    if raw <= t:
        return max(0.0, min(raw, 1.0))
    return t + (1 - t) * (1 - math.exp(-(raw - t) / (1 - t)))


# ══════════════════════════════════════════════════════════════════════
#  HEV 组装 (单城 + 6275 城批量)
# ══════════════════════════════════════════════════════════════════════

def compute_r_ph_city(
    r_ph_national: float,
    h_ratio: float,
    e_ratio: float,
    v_ratio: float,
    alpha_h: float = ALPHA_H,
    alpha_e: float = ALPHA_E,
    alpha_v: float = ALPHA_V,
    *,
    method: str = "analytic",
) -> float:
    """单城 HEV 组装: R_Ph_city = R_Ph_national × H^aH × E^aE × V^aV → soft cap [0,1].

    三个 ratio >1 = 城市比国家均值暴露/脆弱更高 → 物理风险放大.
    """
    if method not in DOWNSCALE_METHODS:
        raise ValueError(f"Unknown downscaling method: {method}")
    if method == "trapezoidal":
        # Lazy import keeps the common soft-cap helper usable from both engines.
        from trapezoidal_downscale import trapezoid_response
        if r_ph_national is None or pd.isna(r_ph_national):
            return float("nan")
        if not np.isfinite(r_ph_national) or not 0 <= r_ph_national <= 1:
            raise ValueError("national risk must be finite and in [0, 1]")
        factors = [trapezoid_response(q, a) for q, a in zip(
            (h_ratio, e_ratio, v_ratio), (alpha_h, alpha_e, alpha_v))]
        return apply_r_ph_soft_cap(float(r_ph_national * np.prod(factors)))
    if r_ph_national is None or pd.isna(r_ph_national):
        return float("nan")
    raw = r_ph_national * (h_ratio ** alpha_h) * (e_ratio ** alpha_e) * (v_ratio ** alpha_v)
    return apply_r_ph_soft_cap(raw)


def downscale_r_ph_for_master(
    df: pd.DataFrame,
    council_v_ratio: Optional[dict] = None,
    *,
    method: str = "analytic",
) -> pd.DataFrame:
    """6275 城批量下推: master 真源 → 新增 H_ratio/E_ratio/V_ratio/R_Ph_city 列.

    Args:
        df: master_v27_6 (含 R_Ph__RAW_nat / haz_*__HAZ / exp_*__RAW / city / iso3_for_nat).
        council_v_ratio: {city_name_or_ucdb_id: V_ratio} 议会算出的城市级 V_ratio.
                         None / 城市不在 → V_ratio=1.0 (6225 城 extrapolation 退化).
                         ⚠️ P5.2 接议会时: 值必须用 compute_v_ratio(1-F_council, nat_abs_def) 构造
                         (保 deficit 极性 + clip). full-master runs should key by ucdb_id to avoid
                         duplicate city-name ambiguity.

    Returns:
        df 副本 + 4 新列. R_Ph_city = 城市级物理暴露 (0-1, soft cap).
        R_Ph__RAW_nat 为 NaN 的城 (真源 1 城) → R_Ph_city NaN.
    """
    if method not in DOWNSCALE_METHODS:
        raise ValueError(f"Unknown downscaling method: {method}")
    out = df.copy()
    cvr = council_v_ratio or {}
    out["H_ratio"] = out.apply(compute_h_ratio, axis=1)
    out["E_ratio"] = out.apply(compute_e_ratio, axis=1)
    v_by_ucdb = {}
    for k, v in cvr.items():
        try:
            v_by_ucdb[int(k)] = v
        except (TypeError, ValueError):
            continue
    if v_by_ucdb and "ucdb_id" in out.columns:
        out["V_ratio"] = out["ucdb_id"].map(v_by_ucdb).fillna(out["city"].map(cvr)).fillna(1.0)
    else:
        out["V_ratio"] = out["city"].map(cvr).fillna(1.0)
    out["R_Ph_city"] = [
        compute_r_ph_city(rn, h, e, v, method=method)
        for rn, h, e, v in zip(
            out["R_Ph__RAW_nat"], out["H_ratio"], out["E_ratio"], out["V_ratio"]
        )
    ]
    return out
