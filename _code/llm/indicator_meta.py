"""indicator_meta.py — 18 indicator 的 v3 prompt 元数据 (Role / Domain / Example).

prompt v3 设计 (用户拍板 2026-05-17):
- 每个 indicator 有专属 Role (领域专家身份) / Domain Knowledge (含行业 ref) / Example Response (HK 示例)
- 共同部分 (Anchor 20 城 / Rules 6 条 / 输出格式) 由 prompt_builder.py 统一拼

prompt_builder.py 在 build_r1/R2/Consul 时 import INDICATOR_META, 用 real_name 作 key 取 dict.
"""
from __future__ import annotations

INDICATOR_META: dict[str, dict[str, str]] = {
    "Building_safety_governance": {
        "role_cn": "有 15+ 年田野经验的资深城市基础设施韧性专家",
        "domain_cn": "建筑安全治理 (BSG) 是城市在建筑全生命周期中确保其能抵御预期灾害的机构能力, 跨 8 子维度作为整合系统协同。行业参考: ISO 22327, UN 仙台框架优先事项 3, IBC/Eurocode 灾害区划。",
        "example_cn": '{"tier": "B", "evidence": "HK 建筑物条例 + 1973 年前老房强制检查 (40 年验楼计划), 但 2010 马头围塌楼暴露非正规改造缺口。强过 Berlin (B-, 老房执法弱), 弱过 Wellington (B+, 抗震优先文化)。"}',
    },
    "Urban_ecosystem_health": {
        "role_cn": "城市生态与基于自然的解决方案专家",
        "domain_cn": "城市生态系统提供调节服务 (洪水 / 热) 减少灾害暴露。参考: IUCN 基于自然的解决方案框架, UNEP 城市生态核算, 海绵城市标准。",
        "example_cn": '{"tier": "B", "evidence": "HK 郊野公园覆盖 40% + 生态尚可, 但核心城区树冠 <25% (vs Singapore S+ 47%). 老区热岛未缓解。比 Wellington (B+) 略弱, 比 Vienna (C+) 强。"}',
    },
    "Urban_income_equity": {
        "role_cn": "城市不平等与社会脆弱性研究员",
        "domain_cn": "收入公平影响韧性能力: 不平等高 = 低收入区脆弱性集中, 灾后恢复慢。参考: 世界银行城市贫困地图集, OECD 美好生活指数不平等模块。",
        "example_cn": '{"tier": "C", "evidence": "HK 基尼 0.539 (2021, 发达亚洲最高之一), 住房可负担比 23 年收入, 30% 人口住㓥房。比 Seoul (C, 基尼 0.345) 差, 接近 London (C-)。"}',
    },
    "Urban_safety_and_social_cohesion": {
        "role_cn": "城市安全与社会资本分析师",
        "domain_cn": "社会凝聚是韧性的'隐性基础设施' — 高信任社区灾后自组织更快。参考: 世界价值观调查, OECD 社会资本指标, Putnam 桥接/纽带资本理论。",
        "example_cn": '{"tier": "B+", "evidence": "HK 凶杀率 0.31/10 万 (极低), 但近年社会信任面临挑战。老区邻里互助强。接近 Wellington (B+) 偏 Berlin (B-) 一侧。"}',
    },
    "Housing_habitability": {
        "role_cn": "住房政策与避难韧性专家",
        "domain_cn": "可居住性决定灾害发生时谁能就地避难的下限。参考: UN-Habitat 适足住房标准, ILO 体面住房, SDG 11.1。",
        "example_cn": '{"tier": "C+", "evidence": "HK 人均居住 16 m² (发达亚洲最低), 22 万人住㓥房。结构合规但拥挤。比 Beijing (E, 24 m² 但法律地位差) 强, 比 Singapore (S, 28 m² + 组屋) 差。"}',
    },
    "Infrastructure_quality": {
        "role_cn": "城市基础设施系统工程师",
        "domain_cn": "基础设施质量是灾害最先暴露的 — 老管爆裂, 老电网瘫。参考: WEF 全球竞争力报告基础设施支柱, ASCE 基础设施报告卡方法学。",
        "example_cn": '{"tier": "B+", "evidence": "HK 地铁准点 99.9%, SAIDI ~2 min/年 (全球顶级), 但供水主干 30%+ 超 40 年。公交近 Tokyo (S+), 水网近 Seoul (C)。综合 B+。"}',
    },
    "Urban_mobile_connectivity": {
        "role_cn": "ICT 韧性与数字基础设施分析师",
        "domain_cn": "现代危机中, 连通性 = 神经系统。参考: ITU 网络韧性指南, GSMA 灾害响应框架。",
        "example_cn": '{"tier": "S", "evidence": "HK 手机普及 280% (多卡), 5G 城区覆盖 95%+, 2018 山竹后强制备电。全球前 3 (与 Tokyo S+ 相近)。"}',
    },
    "Municipal_public_integrity": {
        "role_cn": "反腐与公共部门治理专家",
        "domain_cn": "廉洁直接关系到危机资金能否到达受灾者而非蒸发。参考: 国际透明组织 CPI, 世界银行全球治理指标。",
        "example_cn": '{"tier": "S", "evidence": "HK 廉政公署 (1974-) 全球一流反腐, CPI 全球第 14。强过 Berlin (B-, Wirecard 丑闻)。接近 Singapore (S+)。"}',
    },
    "Local_financial_buffer_capacity": {
        "role_cn": "市政财政与主权风险分析师",
        "domain_cn": "财政缓冲 = 城市无外部救助下持续应急支出的时长。参考: IMF 财政监测, S&P 次主权信用框架。",
        "example_cn": '{"tier": "S", "evidence": "HK 财政储备 8000 亿港币 (~24 个月支出), AA+ 评级, 无外债。全球顶级 (与 Singapore S+, Oslo A 同级)。"}',
    },
    "Logistics_performance_index": {
        "role_cn": "供应链与人道主义物流专家",
        "domain_cn": "物流决定救援物资移动速度。参考: 世界银行 LPI 方法学, WFP 人道主义物流框架。",
        "example_cn": '{"tier": "S+", "evidence": "HK 港口全球第 9, 机场货运全球第 1 (500 万吨/年), 平均通关 4 小时。与 Singapore (S+) 同级, 略胜 Tokyo (S)。"}',
    },
    "Local_economic_stability": {
        "role_cn": "区域经济学家与宏观稳定研究员",
        "domain_cn": "经济稳定让家庭消费抗冲击。参考: IMF WEO (国家级宏观), OECD 区域福祉指标。",
        "example_cn": '{"tier": "B", "evidence": "HK GDP 波动 std 3.2% (2010-2023), 失业率 3-5%, 但行业集中度高 (金融+地产占 50%+ GDP)。强过 London (C-, 脱欧冲击), 弱过 Singapore (S+, 4 支柱多元)。"}',
    },
    "Local_governance_stability": {
        "role_cn": "治理稳定与政治风险分析师",
        "domain_cn": "治理稳定 = 灾后决策可预期。参考: 世界银行 WGI 政治稳定性, V-Dem 制度指标。",
        "example_cn": '{"tier": "B-", "evidence": "HK 公务员连续性强 (政务官 5+ 年), 但近年制度调整影响政策可预测性。强过 Bangkok (E-, 历史治理动荡), 弱过 Tokyo (S+, 几十年稳定)。"}',
    },
    "Access_to_quality_education": {
        "role_cn": "教育系统与人力资本研究员",
        "domain_cn": "教育构建长期恢复能力 (重建+适应)。参考: UNESCO 城市教育监测, PISA 2022 城市子样本。",
        "example_cn": '{"tier": "A", "evidence": "HK PISA 2022 数学 540 (全球前 5), 4 所大学 QS 前 100, 但高教入学率 30% (vs Seoul A, 70%+)。与 Seoul (A) 相当, 接近 Helsinki (A+), 弱过 Singapore (S, 90%+)。"}',
    },
    "Economic_complexity_index": {
        "role_cn": "经济复杂度与创新系统分析师",
        "domain_cn": "ECI 预测长期增长潜力 (Hidalgo & Hausmann)。参考: 哈佛经济复杂度地图集, MIT 经济复杂度观测站。",
        "example_cn": '{"tier": "B", "evidence": "HK 经济在金融/贸易服务复杂, 但缺制造深度, HK ECI ~0.9。ECI 偏商品出口口径, 系统性低估 HK 服务业复杂度。弱过 Seoul (A, ECI 1.7, 制造多元) 与 Singapore (S+, 枢纽多元)。"}',
    },
    "Government_Effectiveness_Index": {
        "role_cn": "公共行政与政府绩效研究员",
        "domain_cn": "政府效能 = 真做事的能力, 不只规划。参考: 世界银行 WGI GE, OECD 一目了然政府。",
        "example_cn": '{"tier": "S", "evidence": "HK WGI GE 百分位 95+ (2010-2022 稳定), 智慧政府 80%+ 使用率, 地铁/机场按时建成。接近 Singapore (S+), 与 Tokyo (S) 同级。"}',
    },
    "Human_Development_Index": {
        "role_cn": "人类发展与福祉经济学家",
        "domain_cn": "HDI 是 UNDP 经典 3 支柱人类福祉综合指数。参考: UNDP HDR 2023/24。",
        "example_cn": '{"tier": "S+", "evidence": "HK HDI 0.956 (2022, 全球第 4), 预期寿命 85.5 (全球最长), 受教育 17 年, 人均 GNI PPP 6.3 万美元。与 Tokyo (S+), Singapore (S+) 同级。"}',
    },
    "Research_Development": {
        "role_cn": "创新政策与 R&D 生态研究员",
        "domain_cn": "R&D 能力 = 应对未知冲击的未来防御。参考: OECD MSTI 数据库, WIPO 全球创新指数城市子排名。",
        "example_cn": '{"tier": "A", "evidence": "HK R&D/GDP 1.07% (2022, 从 2018 的 0.79% 上升), 基础研究强 (港大+中大), 但私营 R&D 薄 (vs Tokyo S+ 3.3%)。接近 Helsinki (A+), 与 Seoul (A) 相当——韩国 4.9% 是国家级 (三星苏原+大田科学城为主), 非首尔市本身。"}',
    },
    "Technology_achievement_index": {
        "role_cn": "智慧城市与数字化转型分析师",
        "domain_cn": "技术成就 = 城市'用'技术 (不只是'有') 的程度。参考: UN 电子政务调查, IMD 智慧城市指数, ITU ICT 发展指数。",
        "example_cn": '{"tier": "A", "evidence": "HK 智慧城市蓝图 2.0 (近年), 转数快全球领先, 5G 覆盖 95%, 但中小企 AI 采用滞后。接近 Helsinki (A+), 弱过 Singapore (S+, 全球顶级)。"}',
    },
}


def load_meta(real_name: str) -> dict[str, str]:
    """取某 indicator 的 v3 元数据 (role/domain/example).

    若 indicator 未注册, 返回通用 fallback (不破坏 prompt 生成).
    """
    return INDICATOR_META.get(real_name, {
        "role_cn": "城市韧性与基础设施专家",
        "domain_cn": "城市韧性是城市在灾害冲击下保持运作 + 快速恢复的综合能力。",
        "example_cn": '{"tier": "B", "evidence": "城市 X 在该指标上的关键证据 + 对照 anchor 城市 Y。"}',
    })


def build_role_domain_block(real_name: str) -> str:
    """渲染 # 角色 + # 领域知识 + # 回复示例 3 个 prompt 段."""
    m = load_meta(real_name)
    return f"""## 角色
你是一位{m['role_cn']}, 有同行评议论文与政策咨询经验。

## 领域知识
{m['domain_cn']}

## 回复示例 (Hong Kong, 单城单 indicator 示意)
```json
{m['example_cn']}
```
"""
