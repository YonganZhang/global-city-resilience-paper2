"""tier_to_score.py — 议会 V27.3.19 tier 后处理

LLM 输出 tier label (S+/S/A+/.../BOT, 20 档 (跟 Sheet 4 真值一致, 无 A-, BOT=2.5)), 代码后处理为 score (mid-point 0-100) 和 resilience (0-1).

设计 (用户拍板):
- LLM 不算 score, 只给 tier 标签
- 代码自动映射 mid-point / 100 → resilience
- 防 LLM 在连续数值上犯错 (LLM 做离散分类更准)

例:
  tier "S+" → score 97.5 → resilience 0.975
  tier "A"  → score 82.5 → resilience 0.825
  tier "BOT" → score 2.5 → resilience 0.025

V27.3.20 docstring 更新 (paper1 audit L2): 真实行为说明
- 门 1 (Master Excel 构建阶段) 已把 18 indicator real_name 映射到 alias_name (LLM 友好命名).
- alias 全部是 resilience polarity (1=好, S+ = 最强, BOT = 最弱), 包括 cap_abs / cap_resp / cap_rec.
- LLM 给 tier, compute_resilience_from_tier 直接产 resilience score (0-1), 不再做翻转.
- deficit polarity 还原 (alias resilience → real_name deficit) 只在写最终输出 csv 时按需做.
"""
from __future__ import annotations

# 5/23 真修 (回 21 档): LLM 真自由输出 A- (尽管 anchor 20 档), step_08 真 raise ValueError skip
# → cap_resp/cap_rest 真 ≥5/6 不够 → GCIRI NaN. 真**LLM 必输 A-**, 真**代码需 accept**.
# 真 21 档 mid-point: S+/S/A+/A/A-/B+/B/B-/C+/C/C-/D+/D/D-/E+/E/E-/F+/F/F-/BOT.
TIER_TO_MIDPOINT = {
    "S+": 97.5, "S": 92.5, "A+": 87.5, "A": 82.5, "A-": 80.0,
    "B+": 77.5, "B": 72.5, "B-": 67.5, "C+": 62.5, "C": 57.5,
    "C-": 52.5, "D+": 47.5, "D": 42.5, "D-": 37.5, "E+": 32.5,
    "E": 27.5, "E-": 22.5, "F+": 17.5, "F": 12.5, "F-": 7.5,
    "BOT": 2.5,
}

# 反向: midpoint → tier(用于 sanity check)
MIDPOINT_TO_TIER = {v: k for k, v in TIER_TO_MIDPOINT.items()}


def compute_score_from_tier(tier: str) -> float:
    """tier label → score (0-100 mid-point).

    Raises ValueError if tier unknown.
    """
    if tier not in TIER_TO_MIDPOINT:
        raise ValueError(f"Unknown tier: {tier!r}, expected one of {list(TIER_TO_MIDPOINT.keys())}")
    return TIER_TO_MIDPOINT[tier]


def compute_resilience_from_tier(tier: str) -> float:
    """tier label → resilience (0-1)."""
    return compute_score_from_tier(tier) / 100.0


def validate_tier_name(tier: str) -> bool:
    """快速校验 tier 名字是否合规."""
    return tier in TIER_TO_MIDPOINT


def build_tier_lookup_table() -> str:
    """渲染 tier → mid → resilience 完整对照表(给文档用)."""
    lines = ["| Tier | Mid 分 | Resilience |"]
    lines.append("|---|---|---|")
    for tier in TIER_TO_MIDPOINT:
        mid = TIER_TO_MIDPOINT[tier]
        res = mid / 100.0
        lines.append(f"| {tier} | {mid} | {res:.3f} |")
    return "\n".join(lines)
