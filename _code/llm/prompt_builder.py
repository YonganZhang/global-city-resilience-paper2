"""prompt_builder.py — 议会 V27.3.19 prompt 构建器(从 Master Excel 读, 单一真源)

设计(用户拍板):
- per-indicator × 10 城 batch (input ~50K tokens, 8 评委 128K context 安全)
- anchor 档位匹配格式(多对一 OK, 空档 OK)
- A 段"按平均/客观水平"(不再按最弱子维度)
- 删 country_anchors → 改两种 anchor(国家级 + 360 城市)
- R2 prompt 加坚持己见 / 调整都给理由
- Consul R2 之后终审(不是 R1.5)
- Tavily 读 _data/tavily_cache/<city>.txt 软链接资产

prompt 三个版本:
1. build_r1_prompt() — 议会 R1, 8 评委独立看
2. build_r2_prompt() — 议会 R2, 看其他 7 评委 + reasoning, 鼓励坚持己见
3. build_consul_prompt() — Consul 终审, 看 R1+R2 全部
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from indicator_meta import build_role_domain_block

REPO_ROOT = Path(__file__).resolve().parents[2]

# Master Excel fallback 链 (2026-05-21):
# 1. _data/V27_3_19_Master_Control.xlsx (V4 active 位置)
# 2. 老 pipeline_v25 output (paper 1 SCS 原位, 但当前文件已丢)
# 3. share-public 备份 master_v5_8_xlsx (临时硬备份)
_MASTER_EXCEL_CANDIDATES = [
    REPO_ROOT / "_data" / "V27_3_19_Master_Control.xlsx",
    REPO_ROOT / "代码/pipeline_v25/output/V27_3_19_Master_Control.xlsx",
]


def _resolve_master_excel() -> Path:
    for p in _MASTER_EXCEL_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Master Excel not found in any fallback candidate. "
        f"Please put V27_3_19_Master_Control.xlsx at: {_MASTER_EXCEL_CANDIDATES[0]}\n"
        f"Searched: {[str(p) for p in _MASTER_EXCEL_CANDIDATES]}"
    )


MASTER_EXCEL = _MASTER_EXCEL_CANDIDATES[0]  # 偏好路径 (用于 docstring / display)
TAVILY_CACHE = REPO_ROOT / "_data/tavily_cache_clean"  # V27.3.20 用户拍板: 只用百科+政府+学术段, 去 [News]
CITY_KEY_SUFFIX_RE = re.compile(r"^(?P<city>.+) \[[A-Z]{3}; UCDB \d+\]$")
CITY_ALIAS_SUFFIX_RE = re.compile(r"^(?P<city>.+?)\s+\[[^\]]+\]$")

GOVERNANCE_NEUTRAL_REPLACE = [
    ("corruption-quake mortality", "governance-transparency earthquake mortality"),
    ("corruption", "governance-transparency"),
    ("腐败率", "治理透明度风险"),
    ("腐败", "治理透明度问题"),
    ("被开发商俘获程度", "受单一利益影响程度"),
    ("开发商俘获", "单一利益影响"),
    ("投诉/抗议渠道", "公共服务诉求渠道"),
    ("抗议渠道", "公共服务诉求渠道"),
    ("抗议", "公众活动"),
    ("政变", "体制变更"),
    ("动乱", "公共秩序中断"),
    ("动荡", "体制波动"),
    ("镇压", "管控"),
]

# 20 tier 定义(V27.3.21 档位匹配)
TIER_DEFS = [
    ("S+", 97.5, "Top 5% (全球最优)"),
    ("S",  92.5, "Top 10%"),
    ("A+", 87.5, "Top 15%"),
    ("A",  82.5, "Top 20%"),
    ("A-", 80.0, "Top 22%"),
    ("B+", 77.5, "Top 30%"),
    ("B",  72.5, "Top 40%"),
    ("B-", 67.5, "Top 50% (中位上)"),
    ("C+", 62.5, "Top 60%"),
    ("C",  57.5, "Median"),
    ("C-", 52.5, "Bottom 55%"),
    ("D+", 47.5, "Bottom 60%"),
    ("D",  42.5, "Bottom 50%"),
    ("D-", 37.5, "Bottom 40%"),
    ("E+", 32.5, "Bottom 30%"),
    ("E",  27.5, "Bottom 25%"),
    ("E-", 22.5, "Bottom 22%"),
    ("F+", 17.5, "Bottom 15%"),
    ("F",  12.5, "Bottom 10%"),
    ("F-",  7.5, "Bottom 7%"),
    ("BOT", 2.5, "Bottom 5% (全球最差)"),
]
TIER_ORDER_STR = " > ".join(t[0] for t in TIER_DEFS)


def load_master_excel() -> dict[str, Any]:
    """读 Master Excel Sheet 6(per-indicator prompt 真源)+ 其他 sheets.

    Fallback 链: _data/ → 老 pipeline_v25/output/ → share-public 备份.
    若全部缺失 → FileNotFoundError 含明确指引.
    """
    import openpyxl
    excel_path = _resolve_master_excel()
    wb = openpyxl.load_workbook(excel_path, data_only=True)

    # Sheet 6: 18 indicator 的 prompt 元数据
    sh6 = wb["6_Prompt_Per_Indicator_18"]
    indicators = []
    for row in sh6.iter_rows(min_row=2, values_only=True):
        if not row[0]:
            continue
        indicators.append({
            "real_name": row[0],
            "alias_name": row[1],
            "anchor_tier_map": row[2] or "",        # V27.3.21 档位匹配
            "full_prompt_template": row[3] or "",   # generic prompt
            "city_def": row[4] or "",
            "audit_status": row[5] or "",
        })

    # Sheet 7: 议会模型
    sh7 = wb["7_Parliament_Models"]
    models = []
    for row in sh7.iter_rows(min_row=2, values_only=True):
        if not row[0]:
            continue
        models.append({"vendor": row[0], "model": row[1], "role": row[2] if len(row) > 2 else "evaluator"})

    return {"indicators": indicators, "models": models}


def _city_cache_name(city: str) -> str:
    """Return display city name for cache lookup when city is a full-mode row key."""
    m = CITY_KEY_SUFFIX_RE.match(city)
    return m.group("city") if m else city


def _clean_city_display_for_prompt(city: str) -> str:
    """Return a clean display name while preserving the exact city key elsewhere.

    Row keys can contain alias hints such as ``Zhuhai [Macau] [CHN; UCDB ...]``.
    Those hints are useful for data lineage but can contaminate entity reasoning.
    """
    display = _city_cache_name(city)
    m = CITY_ALIAS_SUFFIX_RE.match(display)
    if m:
        return m.group("city").strip()
    return display


def _render_city_identity_block(cities: list[str]) -> str:
    """Render exact JSON keys plus clean names for entity-stable prompting."""
    lines = []
    for city in cities:
        clean = _clean_city_display_for_prompt(city)
        lines.append(f'- exact_json_key: "{city}"; clean_display_name: "{clean}"')
    return "\n".join(lines)


def _evidence_strength_rules(context: str) -> str:
    """Shared evidence-strength rubric for R1/R2."""
    return f"""**evidence_strength 字段说明 ({context})**:
- **strong**: prompt 中已经出现或 reasoning 明确指出可核验的城市级制度/事件/公开数据；不能只靠国家印象、模型记忆、票数或 anchor 类比
- **medium**: 有城市级一般事实, 但缺来源/年份/量化支撑
- **weak**: 主要靠常识、国家级推断、anchor 类比、模型间一致性、或无法核验的具体说法

📌 票数/多数一致/多个模型都这么说 ≠ strong evidence.
📌 R2 只能在 R1 reasoning 已给出可核验城市级事实时保留/给出 strong；不能因为看了其他模型而把 weak/medium 升为 strong.
📌 不得因为写了具体年份/百分比/政策名就自标 strong；无法核验时必须 medium/weak."""


def _neutralize_indicator_definition(text: str) -> str:
    """Neutralize high-risk governance wording before prompts reach the gateway.

    This is not a semantic rescue. It keeps the same construct but avoids wording
    that repeatedly caused long governance packets to fail or drift.
    """
    if not isinstance(text, str) or not text:
        return text
    cleaned = text
    for src, dst in GOVERNANCE_NEUTRAL_REPLACE:
        cleaned = cleaned.replace(src, dst)
    return cleaned


def load_tavily_for_city(city: str) -> str:
    """读 _data/tavily_cache/<city>.txt 软链接, 返回原始 Tavily 输出.

    返回原始格式 (含 URL + 标题 + Summary), 给 LLM 前可调用 load_summaries_only() 精简.
    """
    display_city = _city_cache_name(city)
    city_safe = display_city.replace(" ", "_")
    f = TAVILY_CACHE / f"{city_safe}.txt"
    if not f.exists():
        return f"[Tavily cache missing for {display_city}]"
    return f.read_text(encoding="utf-8")


def load_summaries_only(tavily_raw: str) -> str:
    """从原始 Tavily 输出抽出只 Summary 字段(去 URL + 标题).

    复用 _code/migration/compress_tavily_cache.py 的逻辑,
    但 inline 这里避免跨目录 import.
    """
    out_lines = []
    in_summary = False
    summary_buf = []

    def flush():
        nonlocal summary_buf
        if summary_buf:
            joined = " ".join(s.strip() for s in summary_buf).strip()
            out_lines.append(f"Summary: {joined}")
            summary_buf = []

    for line in tavily_raw.split("\n"):
        stripped = line.strip()
        if not stripped:
            if in_summary:
                flush()
                in_summary = False
            continue
        if stripped.startswith("--- ["):
            flush()
            in_summary = False
            out_lines.append("")
            out_lines.append(stripped)
            continue
        if stripped.startswith("[") and "#" in stripped:
            flush()
            in_summary = False
            m = re.match(r"^(\[\w+#\d+\])", stripped)
            out_lines.append(m.group(1) if m else stripped)
            continue
        if stripped.startswith("URL:"):
            continue
        if stripped.startswith("Summary:"):
            in_summary = True
            summary_buf = [stripped[8:].strip()]
            continue
        if in_summary:
            summary_buf.append(stripped)
    flush()
    return "\n".join(l for l in out_lines if l).strip()


def build_tier_definitions_block() -> str:
    """渲染 tier 定义表 + 顺序说明 + 严格白名单约束(防 LLM 自创非法档)."""
    n_tiers = len(TIER_DEFS)
    lines = [f"## 档位系统(V27.3.21, 共 {n_tiers} 档)", ""]
    lines.append("| 档 | mid 分 | 百分位 |")
    lines.append("|---|---|---|")
    for tier, mid, pct in TIER_DEFS:
        lines.append(f"| **{tier}** | {mid} | {pct} |")
    lines.append("")
    lines.append(f"**顺序(高 → 低)**: {TIER_ORDER_STR}")
    lines.append("")
    lines.append("⚠️ 关键规则(V27.3.21 档位匹配, **不是城市强排序**):")
    lines.append("- **多个城市可以同一档**(说明该指标该水准上有多个城市)")
    lines.append("- **某些档位可以空**(无城市达到那个水准)")
    lines.append("- 目标:把目标城市落到**最匹配**的档位(允许同档)")
    # 2026-05-31: 防 LLM 自创非法档 (实测 glm-4-flash 输出 G / H)
    legal_tiers = ", ".join(f"`{t[0]}`" for t in TIER_DEFS)
    lines.append(f"- `tier` 必须从上表 {n_tiers} 档**完全一致**选择({legal_tiers}),不许自创或加字符")
    return "\n".join(lines)


def build_r1_prompt(
    indicator: dict[str, Any],
    cities: list[str],
    country_anchors_excel_row: dict | None = None,
) -> tuple[str, str]:
    """议会 R1 prompt builder.

    Args:
        indicator: Sheet 6 一行 (含 alias_name / city_def / anchor_tier_map)
        cities: 10 城 batch
        country_anchors_excel_row: 该 indicator 国家级 anchor 数据(从 Sheet 1)

    Returns:
        (system_prompt, user_prompt) 给 LLM
    """
    real_name = indicator["real_name"]
    alias = indicator["alias_name"]
    city_def = indicator["city_def"]
    city_def = _neutralize_indicator_definition(city_def)
    anchor_tier_map = indicator["anchor_tier_map"]

    # V27.3.20 用户拍板: Tavily 改可选, 默认**不加** (避免长 prompt 让 deepseek/qwen 400).
    # 想加 Tavily 改全局开关 env LLM_INCLUDE_TAVILY=1, 或者调用方传 indicator["_include_tavily"]=True.
    include_tavily = bool(indicator.get("_include_tavily", False)) or os.getenv("LLM_INCLUDE_TAVILY", "").lower() in ("1", "true", "yes")
    tavily_blocks = []
    if include_tavily:
        # 拼 Tavily summaries (只 Summary 字段, 去 URL+标题)
        for city in cities:
            raw = load_tavily_for_city(city)
            summaries = load_summaries_only(raw)
            tavily_blocks.append(f"### {city}\n{summaries}\n")
    else:
        # 默认: 仅列城市名, 不附外部证据；模型必须显式降低证据强度。
        for city in cities:
            clean = _clean_city_display_for_prompt(city)
            tavily_blocks.append(
                f"### JSON key: {city}\n"
                f"clean_display_name: {clean}\n"
                "(未附外部参考；不得编造年份/比例/案件/排名。"
                "可用常识与 anchor 作保守判断，但证据强度不能因措辞具体而自动升为 strong。)\n"
            )

    tier_block = build_tier_definitions_block()
    role_domain = build_role_domain_block(real_name)

    system = f"""{role_domain}

## 任务说明

把 {len(cities)} 个目标城市分到 21 档位之一 (允许同档, 允许空档), 针对**单一 indicator**:
**{alias}**

输入: anchor 城市清单 (仅校准, 不评) + 目标城市 identity. 外部参考资料可能为空.
输出: 每个目标城市的 tier + 30-60 字中文证据. 缺少城市级证据时必须保守标注不确定性.

⚠️ 评分极性: 此 indicator 衡量**机构能力**, 档次越高 = 能力越强 (S+ 最强 / BOT 最弱).
深层 deficit/resilience 翻转由代码后处理 (门 2), 你只判断**能力**, 无需考虑底层公式方向.

## Indicator 子维度定义 (从 Master Excel Sheet 6)

{city_def}

{tier_block}

## 档位匹配 anchor 表

{anchor_tier_map}

## 你的输出格式

**严格 JSON**:
```json
{{
  "indicator": "{alias}",
  "ratings": {{
    "<Exact JSON key>": {{
      "tier": "A+",
      "evidence_strength": "strong",
      "reasoning": "30-60 字中文证据"
    }},
    ...
  }}
}}
```

{_evidence_strength_rules("R1/Consul 加权用")}

**字段说明**:
- `tier`: 21 档之一 (S+ 到 BOT)
- `evidence_strength`: "strong" | "medium" | "weak"
- `reasoning`: 30-60 字中文, 含城市名 + ≥1 具体事实

## 规则 (专业版 7 条, 强约束)

1. **系统视角** — 把 8 子维度作整合系统评估, 不是逐项加权平均
2. **只用城市级证据** — 用目标城市自身事实 (法规 / 记录 / 真实事件); 国家级印象只能作 weak 背景
3. **Anchor 夹击** — 挑 2-3 个能力夹住目标的 anchor 城市作对照, 论证目标落在哪一侧
4. **极性** — 机构能力越强 ↔ 档次越高 (deficit 指标在后处理反转, 你只判断能力)
5. **证据-或-不确定** — 不确定可以给保守 tier, 但 reasoning 必须说明证据薄弱, 不要编
6. **不要国家级平均** — 国家分 ≠ 城市分 (例: Manila ≠ 菲律宾均值)
7. 多城同档 / 某档空 都允许 (后处理不强制每档有城市)
8. 输出**只用 JSON**, 不要 markdown 包裹, 不要前后散文
# 2026-06-01 用户拍板: 删除规则 9 (Reasoning 中性化引导) — 反复引入 self-suicide bug (体制/宗教/抗议 等
# 我自己写入 prompt 触发阿里云审查). 改靠后处理治: step_04 cleanse_if_china 拼 R2 packet 前清洗 +
# step_05d_nan_rescue 兜底救援. R1 LLM 自由写, 后处理治, 不再加 prompt 引导.
"""

    exact_key_lines = _render_city_identity_block(cities)
    evidence_header = "## 目标 {n} 城 + 外部参考资料 (默认不附 Tavily; 缺证据需保守标注)" .format(n=len(cities))
    user = f"""{evidence_header}

{''.join(tavily_blocks)}

## JSON key 强约束
`ratings` 的 key 必须逐字使用下面 `exact_json_key` 的值，不要包含 `exact_json_key:` / `clean_display_name:` 标签，不要删掉 ISO3 / UCDB 后缀，不要翻译，不要改写。
评分时使用 clean_display_name 理解城市实体；括号里的 alias 仅作别名提示, 不能把相邻城市/地区混为一体：
{exact_key_lines}

按 system prompt 的 anchor + city_def, 给这 {len(cities)} 城每个分到 21 档之一. 输出 JSON.
注: 没有外部 Tavily 资料时, 可以基于常识 + anchor 保守评分, 但 evidence_strength 应降为 medium/weak.
"""
    return system, user


def build_r2_prompt(
    indicator: dict[str, Any],
    cities: list[str],
    r1_packet: dict[str, dict[str, Any]],  # {model_name: {city: {tier, reasoning}}}
    country_anchors_excel_row: dict | None = None,
) -> tuple[str, str]:
    """议会 R2 prompt builder.

    R2 LLM 看 R1 全部 8 评委 + 各自 reasoning, 鼓励坚持己见或调整, 都给理由.
    """
    real_name = indicator["real_name"]
    alias = indicator["alias_name"]
    city_def = indicator["city_def"]
    city_def = _neutralize_indicator_definition(city_def)
    anchor_tier_map = indicator["anchor_tier_map"]
    tier_block = build_tier_definitions_block()
    role_domain = build_role_domain_block(real_name)

    # 拼 R1 packet
    packet_lines = ["## 议会 R1 全部 8 评委的打分 + reasoning(已匿名为 Model_A/B/C...)"]
    packet_lines.append("")
    for city in cities:
        packet_lines.append(f"### {city}")
        for i, (model, ratings) in enumerate(sorted(r1_packet.items())):
            alias_id = chr(ord("A") + i)
            r = ratings.get(city, {})
            tier = r.get("tier", "—")
            reason = _neutralize_indicator_definition(r.get("reasoning", "—"))
            packet_lines.append(f"- Model_{alias_id}: {tier} — {reason}")
        packet_lines.append("")
    packet = "\n".join(packet_lines)

    system = f"""{role_domain}

V27.3.19 议会 R2(辩论轮).

## 任务

你看到全部 8 评委 R1 阶段的打分 + reasoning(匿名).
**重新评估**自己的打分:
- ✅ **可以调整**(看了其他评委 reasoning 觉得有新角度)
- ✅ **可以坚持己见**(觉得自己的 R1 是对的)
- ⚠️ **不论调整或坚持, 都必须给理由**(30-60 字中文)

## Indicator(同 R1)

{alias}: {city_def}

{tier_block}

{anchor_tier_map}

## R2 评分原则

1. **坚持己见是值得鼓励的**, 多数票不一定对
2. **不要跟风**, 也不要 anchor to mean
3. **看证据**: 哪个评委给的是可核验城市级事实; 具体数字/年份若无法核验, 不自动算强证据
4. 输出**严格 JSON**

## 输出格式

```json
{{
  "indicator": "{alias}",
  "ratings": {{
    "<Exact JSON key>": {{
      "tier": "A+",
      "decision": "adjusted" | "kept",
      "evidence_strength": "strong" | "medium" | "weak",
      "reasoning": "30-60 字: 为什么调整/坚持"
    }},
    ...
  }}
}}
```

{_evidence_strength_rules("V27.3.20 P3 新增, Consul 加权用")}
- Consul 在 mean kept 评委时按 evidence_strength 加权 (strong=1.5, medium=1.0, weak=0.5)
"""

    exact_key_lines = _render_city_identity_block(cities)
    user = packet + f"""

## JSON key 强约束
`ratings` 的 key 必须逐字使用下面 `exact_json_key` 的值，不要包含 `exact_json_key:` / `clean_display_name:` 标签，不要删掉 ISO3 / UCDB 后缀，不要翻译，不要改写。
评分时使用 clean_display_name 理解城市实体：
{exact_key_lines}

按上面 R1 packet 重新评估, 决定坚持或调整, 都给理由 + 标 evidence_strength. 输出 JSON.
"""
    return system, user


def build_consul_prompt(
    indicator: dict[str, Any],
    cities: list[str],
    r1_packet: dict[str, dict[str, Any]],
    r2_packet: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    """Consul 终审 prompt builder(R2 之后跑, 不是 R1.5).

    Consul model: gemini-2.5-pro(看 R1 + R2 全部, 给最终 tier).
    """
    real_name = indicator["real_name"]
    alias = indicator["alias_name"]
    city_def = indicator["city_def"]
    city_def = _neutralize_indicator_definition(city_def)
    anchor_tier_map = indicator["anchor_tier_map"]
    tier_block = build_tier_definitions_block()
    role_domain = build_role_domain_block(real_name)

    # 拼 R1 + R2 packet
    def fmt_packet(packet: dict, label: str) -> str:
        lines = [f"## 议会 {label} 全部评委的打分"]
        for city in cities:
            lines.append(f"### {city}")
            for i, (model, ratings) in enumerate(sorted(packet.items())):
                alias_id = chr(ord("A") + i)
                r = ratings.get(city, {})
                tier = r.get("tier", "—")
                reason = _neutralize_indicator_definition(r.get("reasoning", "—"))
                decision = r.get("decision", "")
                deco = f" ({decision})" if decision else ""
                lines.append(f"- Model_{alias_id}: {tier}{deco} — {reason}")
            lines.append("")
        return "\n".join(lines)

    system = f"""{role_domain}

你是议会主席 (Consul), 看完 R1 + R2 全部 8 评委的打分 + reasoning. 🔴 **你的唯一权力是去除极值** (用户拍板 2026-05-17):

- **去除最大值评委** (1 个, 那个给最高 tier 的) — 如果它过高
- **去除最小值评委** (1 个, 那个给最低 tier 的) — 如果它过低
- **去除两端**: 同时去除最高和最低 (如果两端都有问题)
- **不去**: 8 评委一致 / 极值合理时

⚠️ **你绝对不能**: 重新打分 / 自己给一个 tier / 强加判断. 你只决定"删谁".

## Indicator

{alias}: {city_def}

{tier_block}

{anchor_tier_map}

## Consul 决策原则 (只判断"去不去极值")

1. **极值是否反常**: 最高/最低评委的 reasoning 是否站不住脚 (无证据 / 极性反 / 与多数差 ≥3 档)
2. **多数共识**: 8 评委 R2 后中间 6 个是否一致 (差 ≤2 档)
3. **看证据质量**, 不看票数. 一致但都没可核验城市级证据 ≠ 强共识
4. **保守原则**: 不确定就 keep_all (不删)
5. **证据强度和删极值分开**: Consul 只删极值, 不要把“多数一致”描述成 strong evidence；多数一致只是 vote consensus, 不是外部证据.

## 输出 (严格 JSON, drop 数组只列删谁, 代码后处理算最终 tier)

🔴 **8 评委编号 1-8** (字典序: Model_A=1, B=2, C=3, D=4, E=5, F=6, G=7, H=8).
Consul 只输出"删谁的位置 + 理由".

```json
{{
  "indicator": "{alias}",
  "decisions": {{
    "<Exact JSON key>": {{
      "drop": [5],
      "reasoning": "Model_5 给 A- 但 reasoning 无具体证据反常. 30-50 字."
    }},
    "<Another City>": {{
      "drop": [],
      "reasoning": "8 评委 R2 共识 B+/A-, 极值合理 keep_all. 30-50 字."
    }},
    "<Another>": {{
      "drop": [1, 5],
      "reasoning": "Model_1 极低 + Model_5 极高都 reasoning 弱, 双侧剔除. 30-50 字."
    }}
  }}
}}
```

**字段约束**:
- `drop`: 整数数组, 元素 1-8 之间. **空数组 [] = keep_all 不删**. **最多 2 个元素** (1 max + 1 min).
  - 长度 ≥3 = 议会失败信号, 必须改 [] 让 polarity_aggregate 后处理.
- `reasoning`: 必填中文 30-50 字, 说明删谁/为什么 OR 为什么 keep_all.

**5 种 drop 典型值** (按 prompt 给的 Model_A/B/C/D/E/F/G/H 字典序对应 1-8):
| drop 值 | 含义 | 何时用 |
|---|---|---|
| `[]` | keep_all 不删 | 8 评委共识 / 极值合理 / 不确定 (默认保守) |
| `[5]` | 删 Model_E (5 号) | Model_E 极高/极低 + reasoning 无证据 |
| `[1]` | 删 Model_A (1 号) | Model_A 极低/极高 + reasoning 弱 |
| `[1, 5]` | 双删 Model_A + Model_E | 双侧反常 (极高极低都无证据) |
| `[1, 2, 3]` | 🔴 **禁止** | ≥3 是议会失败信号, 必须 [] |

**极值反常硬约束 (C.1, 双条件 AND)**:
反常 = (该评委 R2 tier 跟中位差 ≥3 档) **AND** (reasoning 无具体证据).
两条都符才算反常. 只满足一条 (e.g. 差 ≥3 档但有证据) 可能是 hyper-local 真发现 → 保留.

**后处理 (代码做, 不是 LLM 做)**: 用 drop 数组 index 删对应 Model_X, 剩余 kept 评委 R2 tier → resilience → mean → 找最近 tier 作最终输出.
"""

    exact_key_lines = _render_city_identity_block(cities)
    user = fmt_packet(r1_packet, "R1") + "\n\n" + fmt_packet(r2_packet, "R2") + f"""

## JSON key 强约束
`decisions` 的 key 必须逐字使用下面 `exact_json_key` 的值，不要包含 `exact_json_key:` / `clean_display_name:` 标签，不要删掉 ISO3 / UCDB 后缀，不要翻译，不要改写。
判断时使用 clean_display_name 理解城市实体：
{exact_key_lines}

按上面 R1+R2 全部, 给 {alias} 的 {len(cities)} 城最终 tier. 输出 JSON.
"""
    return system, user
