"""V27.5 真梯形 GIRI 公式 (Cardona 2023 Moncho's Equation + V2 21 点 piecewise + 4 维回归).

V27.5 变更: 不再 thin re-export 到 老 `代码/pipeline/steps/step4_giri_computation.py`
(依赖老 config.py 不存在). 直接在此文件内联真公式 — 代码层独立, 7 行内 self-contained.

V2 精度 (170 国 MAE=1.156 / R²=0.9968, 改善 50% vs V1 MAE=2.34)
Wiki 方法论 source: _wiki-methodology/_wiki/_entities/_梯形GIRI公式.md
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

# V2 系数 (170 国拟合, 来自 代码/pipeline_v25/models/giri_v2_coefficients.json)
# V27.6 DR.13.11: V2 coef 复制到 worktree _data/, 跨节点同步不丢
_REPO = Path(__file__).resolve().parents[2]
_V2_COEF_LOCAL = _REPO / "_data/giri_v2/giri_v2_coefficients.json"
_V2_COEF_FALLBACK = _REPO / "代码/pipeline/models/giri_v2_coefficients.json"
_V2_COEF_PATH = _V2_COEF_LOCAL if _V2_COEF_LOCAL.exists() else _V2_COEF_FALLBACK
_V2 = None
_V2_R_GRID = np.linspace(0.0, 1.0, 21)


def _load_v2_coef():
    global _V2
    if _V2 is None:
        # 主项目 fallback
        path = _V2_COEF_PATH
        if not path.exists():
            # 同位置 worktree
            path = Path(__file__).parents[2] / "代码/pipeline/models/giri_v2_coefficients.json"
        _V2 = json.loads(path.read_text(encoding="utf-8"))
    return _V2


def _ap_ratio(cap_abs: float, cap_resp: float, cap_rest: float) -> float:
    """梯形面积/周长. cap_abs=0 → 0."""
    y = cap_abs
    angle_rad = np.radians(10 + 70 * (cap_rest / 100))
    t_rec = y / np.tan(angle_rad) if np.tan(angle_rad) > 0 else 0.0
    t_resp = cap_resp
    area = 0.5 * (2 * t_resp + t_rec) * y
    side = np.sqrt(y * y + t_rec * t_rec)
    perim = 2 * t_resp + t_rec + y + side
    return area / perim if perim > 0 else 0.0


def compute_trapezoid_giri(F_city: float, cap_resp: float, cap_rest: float, R_Ph: float) -> float:
    """V2 GIRI: 扫 21 点 PC=-A/P → 国家端点 [RR_ymin, 99.8] 归一化 → 真实 R_Ph 处 piecewise-linear 插值.

    POLARITY (2026-06-03 统一, 项目 polarity-and-gates rule):
      F_city = deficit (Cardona 2023 aggravating factor, 高=加剧, 0=完全韧性)
      cap_resp = deficit (高=响应缺口大)
      cap_rest = resilience (高=恢复好) ← 跟前两者方向相反, 由内嵌 angle 公式决定
    调用方必须翻好 polarity 再传 (step_08:110 / step_09:99 已正确翻).
    50 城 baseline 验证: r=0.940 / MAE=8.47 (8 种 polarity 组合中次优 r=0.947, 跟 5/23 rule 一致).

    Args:
        F_city:   议会 ABS 6 mean deficit (0-1)  = 1 - resilience_mean
        cap_resp: 议会 RESP 6 mean × 100 deficit (0-100) = 100 - resilience_mean×100
        cap_rest: 议会 REC  6 mean × 100 resilience (0-100) ← 不翻
        R_Ph: 物理风险 (0-1) deficit
    Returns:
        GCIRI (0-100, 高=韧性好)
    """
    v2 = _load_v2_coef()
    _V2_FEATS = v2["rr_ymin_regression"]["features"]
    _V2_COEFS = [v2["rr_ymin_regression"]["coefficients"][k] for k in _V2_FEATS]
    _V2_INTERCEPT = v2["rr_ymin_regression"]["intercept"]
    _V2_RR_YMAX = v2["rr_ymax_global"]
    _V2_NPOINTS = v2["n_curve_points"]

    raw = np.empty(_V2_NPOINTS)
    for j, r in enumerate(_V2_R_GRID):
        cap_abs_j = r * 100.0 * (1.0 + F_city) / 2.0
        raw[j] = -_ap_ratio(cap_abs_j, cap_resp, cap_rest)

    raw_PC_ymin = raw[-1]
    feat_vals = {"F": F_city, "cap_resp": cap_resp, "cap_rest": cap_rest, "raw_PC_ymin": raw_PC_ymin}
    rr_ymin = _V2_INTERCEPT + sum(_V2_COEFS[i] * feat_vals[k] for i, k in enumerate(_V2_FEATS))

    rmin, rmax = raw.min(), raw.max()
    if rmax - rmin > 1e-12:
        rr_curve = (raw - rmin) / (rmax - rmin) * (_V2_RR_YMAX - rr_ymin) + rr_ymin
    else:
        rr_curve = np.full(_V2_NPOINTS, _V2_RR_YMAX)

    giri = float(np.interp(min(R_Ph, 1.0), _V2_R_GRID, rr_curve))
    return float(np.clip(giri, 0, 100))


def compute_three_capabilities(F_city, cap_resp_mean, cap_rest_mean, r_ph_city):
    """V27.5: F (议会 ABS 真值) + R_Ph_city → cap_abs (Moncho EQ1) + cap_resp/rest (议会真值 × 100)."""
    R_Ph = min(r_ph_city, 1.0)
    cap_abs = R_Ph * 100 * (1 + F_city) / 2
    return {"F_city": F_city, "R_Ph_city": float(R_Ph),
            "cap_abs": float(cap_abs), "cap_resp": cap_resp_mean * 100,
            "cap_rest": cap_rest_mean * 100}


def compute_city_giri(F_city, cap_resp, cap_rest, r_ph_city):
    """城市级 GCIRI 全 chain: F + 三能力 + 梯形."""
    return compute_trapezoid_giri(F_city, cap_resp, cap_rest, r_ph_city)


if __name__ == "__main__":
    # 5 城 v15 真值测试
    cases = [
        ("Hong Kong", 0.6, 71.67, 58.33, 1.0),
        ("Shenzhen",  0.525, 69.17, 73.75, 0.8721),
        ("Guangzhou", 0.4833, 68.33, 67.5, 0.8251),
        ("London",    0.4667, 48.33, 52.5, 0.1547),
        ("Tokyo",     0.9, 82.5, 87.92, 0.8267),
    ]
    print(f"{'City':<12} F      cap_resp cap_rest R_Ph    GCIRI")
    for name, F, resp, rest, rph in cases:
        g = compute_trapezoid_giri(F, resp, rest, rph)
        print(f"{name:<12} {F:<6} {resp:<8} {rest:<8} {rph:<7} {g:.2f}")
