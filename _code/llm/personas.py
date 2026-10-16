"""
V27.3.11 (2026-05-11): 议会角色分工 + 数据时效 trust weight

每个 LLM 被赋予一个 character (基于厂商背景 + 学术评测), 共享议会协议但聚焦不同评估维度。
数据时效 trust 让议会在评分时考虑各数据源更新年份, 新数据 trust 高、老数据 trust 低。

论文 method 中卖点:
1. Character-aware role assignment (差异化议会)
2. Temporal trust weighting (时效性建模)
"""
from __future__ import annotations
import math
from typing import Dict, List, Any

# ─────────────────────────────────────────────────────────────────
# Character 配置 (基于厂商背景 + 学术评测 MMLU/MATH/HumanEval 等)
# ─────────────────────────────────────────────────────────────────
MODEL_CHARACTERS: Dict[str, Dict[str, str]] = {
    # V27.3.15 (2026-05-12): 添加新模型 character (gpt-5-nano / deepseek-v4-flash / llama-4-scout / grok-4-fast-non-reasoning)
    # 旧 character 保留供 50 城 ablation 比对历史
    "gpt-4.1-nano": {
        "role": "General Analyst",
        "expertise": "balanced reasoning, instruction following, comprehensive multi-domain coverage (OpenAI nano-class, max_tokens 兼容)",
        "evaluation_lens": "produce a balanced baseline assessment across all 18 indicators; you are the calibration anchor.",
    },
    "gpt-5-nano": {
        "role": "General Analyst",
        "expertise": "balanced reasoning, instruction following, comprehensive multi-domain coverage (OpenAI nano-class)",
        "evaluation_lens": "produce a balanced baseline assessment across all 18 indicators; you are the calibration anchor.",
    },
    "gpt-4o-mini": {
        "role": "General Analyst",
        "expertise": "balanced reasoning, instruction following, comprehensive multi-domain coverage",
        "evaluation_lens": "produce a balanced baseline assessment across all 18 indicators; you are the calibration anchor.",
    },
    "deepseek-v4-flash": {
        "role": "Quantitative Specialist",
        "expertise": "mathematics, code, quantitative reasoning, strong Chinese-language understanding (V4 flash, faster)",
        "evaluation_lens": "weight quantitative indicators (HDI, Economic Complexity, Tech Achievement, R&D) higher; verify numerical consistency.",
    },
    "deepseek-chat": {
        "role": "Quantitative Specialist",
        "expertise": "mathematics, code, quantitative reasoning, strong Chinese-language understanding",
        "evaluation_lens": "weight quantitative indicators (HDI, Economic Complexity, Tech Achievement, R&D) higher; verify numerical consistency.",
    },
    "gemini-2.5-flash-lite": {
        "role": "Geospatial Expert",
        "expertise": "multi-language, global geography, Google Search index access",
        "evaluation_lens": "weight hazard/exposure indicators and physical geography context higher; cross-check city location-specific physical risk.",
    },
    "llama-4-scout": {
        "role": "Cross-Cultural Researcher",
        "expertise": "open-source, multi-language, 10M ultra-long-context, international development (scout = maverick 同 Meta 轻量版)",
        "evaluation_lens": "weight cross-national comparison and cultural context indicators (Education access, Peace, Housing) higher; leverage long-context for multi-source synthesis.",
    },
    "llama-4-maverick": {
        "role": "Cross-Cultural Researcher",
        "expertise": "open-source, multi-language, long-context, international development",
        "evaluation_lens": "weight cross-national comparison and cultural context indicators (Education access, Peace, Housing) higher.",
    },
    "doubao-seed-1-6-flash-250828": {
        "role": "Asia Urban Specialist",
        "expertise": "Chinese + East/Southeast Asian context, urban infrastructure of Asian megacities",
        "evaluation_lens": "for Asian cities (CHN/JPN/KOR/SE Asia), apply local domain knowledge; verify Asia-region peer benchmarks.",
    },
    "glm-4-flash": {
        "role": "Governance Researcher",
        "expertise": "Chinese policy/governance literature, institutional analysis",
        "evaluation_lens": "weight governance indicators (Government Effectiveness, Political Stability, Control of Corruption) higher; assess institutional capacity.",
    },
    "grok-4-fast-non-reasoning": {
        "role": "Real-Time Observer",
        "expertise": "real-time X/Twitter data, current events, news cycle awareness (xAI Grok-4 fast 非推理版, 实测 4.5s 最快)",
        "evaluation_lens": "incorporate latest news/political events (≤6 months) into Political Stability, Macroeconomic Stability, Peace indicators; flag recent shocks.",
    },
    "grok-3-mini": {
        "role": "Real-Time Observer",
        "expertise": "real-time X/Twitter data, current events, news cycle awareness",
        "evaluation_lens": "incorporate latest news/political events (≤6 months) into Political Stability, Macroeconomic Stability, Peace indicators; flag recent shocks.",
    },
    "hunyuan-turbo": {
        "role": "Economic Expert",
        "expertise": "Chinese commercial data, logistics, business ecosystem, GDP composition",
        "evaluation_lens": "weight Logistics Performance, Macroeconomic Stability, Infrastructure Quality higher; assess economic complexity from a business-data lens.",
    },
    "moonshot-v1-128k": {
        "role": "Long-Context Evidence Auditor",
        "expertise": "long-context Chinese/English synthesis, cross-source consistency checking, urban policy and public-report evidence",
        "evaluation_lens": "weight source consistency, evidence completeness, and recent policy/report signals higher; challenge unsupported extreme tiers and preserve cross-city comparability.",
    },
    "qwen3.5-flash": {
        "role": "Economic & Logistics Expert (Alibaba)",
        "expertise": "Chinese commercial/logistics ecosystem (Alibaba's domain), supply chain analysis, multilingual reasoning",
        "evaluation_lens": "weight Logistics Performance, Macroeconomic Stability, Infrastructure Quality higher; bring Alibaba's e-commerce/supply-chain perspective on urban economic complexity.",
    },
}


# ─────────────────────────────────────────────────────────────────
# 数据源时效 trust weight (exp decay τ=3y, base year 2025)
# ─────────────────────────────────────────────────────────────────
def _trust_weight(year: int, tau: float = 3.0, base: int = 2025) -> float:
    """指数衰减: trust = exp(-(base - year) / tau)"""
    return round(math.exp(-(base - year) / tau), 2)


DATA_TIMELINESS: Dict[str, Dict[str, Any]] = {
    "LandSCAN_population":    {"year": 2024, "trust": _trust_weight(2024), "desc": "LandSCAN population raster"},
    "GHS_UCDB_R2024A":        {"year": 2024, "trust": _trust_weight(2024), "desc": "GHS-UCDB city boundaries"},
    "VIIRS_NTL":              {"year": 2024, "trust": _trust_weight(2024), "desc": "VIIRS nighttime lights"},
    "UNEP_GRID_hazard":       {"year": 2023, "trust": _trust_weight(2023), "desc": "UNEP-GRID 6-hazard rasters"},
    "Cardona_GIRI_national":  {"year": 2023, "trust": _trust_weight(2023), "desc": "UNDRR/Cardona national GIRI release"},
    "WGI_governance":         {"year": 2022, "trust": _trust_weight(2022), "desc": "World Bank Worldwide Governance Indicators"},
    "UNDP_HDI":               {"year": 2022, "trust": _trust_weight(2022), "desc": "UNDP Human Development Index"},
    "GINI_inequality":        {"year": 2021, "trust": _trust_weight(2021), "desc": "GINI coefficient (World Bank latest)"},
    "Tech_Achievement_Index": {"year": 2020, "trust": _trust_weight(2020), "desc": "Technology Achievement Index (latest available)"},
}


# ─────────────────────────────────────────────────────────────────
# Prompt 注入器
# ─────────────────────────────────────────────────────────────────
def build_character_block(model_name: str) -> str:
    """For R1 / R2 prompt: head injection making the agent identify with its persona."""
    c = MODEL_CHARACTERS.get(model_name)
    if not c:
        return ""
    return (
        f"\n## YOUR ROLE IN THE PARLIAMENT (V27.3.11 character assignment)\n"
        f"YOU ARE: **{c['role']}** ({model_name}).\n"
        f"YOUR EXPERTISE: {c['expertise']}.\n"
        f"YOUR EVALUATION LENS: {c['evaluation_lens']}\n"
        f"You share the parliament protocol (1=most resilient, 0=least), but apply your specialist lens "
        f"when interpreting evidence. Stay within 0-1 range; your specialist weighting affects "
        f"reasoning emphasis, not numeric output range.\n"
    )


def build_trust_block() -> str:
    """For R1 / R2 prompt: head injection on data source temporal trust."""
    rows = []
    for k, v in DATA_TIMELINESS.items():
        rows.append(f"  - {v['desc']}: updated {v['year']}, trust = {v['trust']:.2f}")
    return (
        "\n## DATA SOURCE TIMELINESS (V27.3.11 trust weighting)\n"
        "Different data sources have different update recency. "
        "Newer data = higher trust weight; older data = lower (consider rate of change since update).\n"
        + "\n".join(rows) +
        "\n→ When you cite or rely on a specific indicator, mentally weight it by its source's trust. "
        "If two evidence pieces conflict, prefer the higher-trust source.\n"
    )


def build_persona_injected_msgs(msgs: List[Dict[str, str]], model_name: str) -> List[Dict[str, str]]:
    """Deep-copy + inject character + trust into the system message head.
    Used by R1 _score_one and R2 deliberation_one."""
    import copy
    new_msgs = copy.deepcopy(msgs)
    if new_msgs and new_msgs[0].get("role") == "system":
        head = build_character_block(model_name) + build_trust_block()
        new_msgs[0]["content"] = head + "\n" + new_msgs[0]["content"]
    return new_msgs


__all__ = [
    "MODEL_CHARACTERS",
    "DATA_TIMELINESS",
    "inject_persona",
    "build_character_block",
    "build_trust_block",
]
