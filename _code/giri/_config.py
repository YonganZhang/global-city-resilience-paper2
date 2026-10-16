"""_config.py — GIRI 算法常数单一真源 (σ² floor + INDICATOR_COLS)

2026-05-28 建: 贝叶斯三源融合 (bayesian_fusion.py) 从老 V25 迁 v4 时, 配套 config 常数
从 _pipelines/00_data_prep/_config.py 收拢到此 (GIRI 算法扣子库的真源).
σ² 数据驱动估计 (compute_sigma2_nat_ab 等) 取代硬编码, floor 只作下限保护.
"""
# ── R_Ph 城市下推 HEV 弹性参数 (单一真源，本研究经验设置) ──────────────
# R_Ph_city = R_Ph_national × H_ratio^ALPHA_H × E_ratio^ALPHA_E × V_ratio^ALPHA_V
# GIRI提供风险构成思路，不提供这三个固定值；来源核对见技术报告9.4及附录D.5。
ALPHA_H = 0.8    # 灾害弹性 (有效灾害比值的算术平均)
ALPHA_E = 0.3    # 暴露弹性 (人口×夜光 city/nat ratio)
ALPHA_V = 0.3    # 脆弱性弹性 (ABS 6 城/国 deficit ratio)
R_PH_SOFT_CAP_THRESHOLD = 0.95   # raw>0.95 渐进 1.0 防 megacity 全饱和失差异

# σ² floor / 兜底常数 (值同 _pipelines/00_data_prep/_config.py)
SIGMA2_NAT_FLOOR        = 0.005   # σ²_nat 下限保护
SIGMA2_UCDB_FLOOR       = 0.005   # σ²_ucdb 下限保护
SIGMA2_UCDB_DEFAULT     = 0.010   # A 类兜底 (数据驱动估计前的 fallback)
SIGMA2_UCDB_B_CLASS     = 0.025   # B 类兜底
SIGMA2_UCDB_LOW_QUALITY = 0.04    # 低质量兜底
SIGMA2_LLM_FLOOR        = 0.020   # LLM σ² 硬下限

# 18 indicator 列名 (real_name, deficit polarity)
INDICATOR_COLS = [
    "ind__cap_abs__Infrastructure_quality",
    "ind__cap_abs__Building_quality_control_index",
    "ind__cap_abs__Ecosystem_vitality",
    "ind__cap_abs__GINI_Index",
    "ind__cap_abs__Housing_deprivation",
    "ind__cap_abs__Global_Peace_Index",
    "ind__cap_resp__2G_3G_and_4G_network_coverage",
    "ind__cap_resp__Control_of_corruption",
    "ind__cap_resp__Gross_National_Savings",
    "ind__cap_resp__Logistics_performance_index",
    "ind__cap_resp__Macroeconomic_stability",
    "ind__cap_resp__Political_stability",
    "ind__cap_rec__Access_to_quality_education",
    "ind__cap_rec__Economic_complexity_index",
    "ind__cap_rec__Government_Effectiveness_Index",
    "ind__cap_rec__Human_Development_Index",
    "ind__cap_rec__Research_Development",
    "ind__cap_rec__Technology_achievement_index",
]
