"""sensitive_cleanse.py — 共享敏感词清洗工具 (主 pipeline + 救援共用)

用途:
- 主 step_04 R2: 拼 R1 packet 前 cleanse, 预防 R2 prompt 被网关拦
- 主 step_05 Consul: 同上
- step_05d 救援: 双层清洗 (规则 + qwen LLM 兜底)

策略: 长串先替防 substring 漏 + 单独年份 + 政治敏感词 + 评判性词中性化.
词表是动态扩展的, 漏的由 LLM 兜底清洗 (step_05d.cleanse_by_llm).
"""
SENSITIVE_REPLACE = [
    # 长串组合 (优先替换)
    ("2019-20 年反送中示威", "近年社会事件"),
    ("反送中示威", "近年公众活动"),
    ("反送中", "近年社会事件"),
    ("送中", "社会事件"),
    ("2019 年抗议", "近年事件"),
    ("2019年抗议", "近年事件"),
    ("2019-20 年", "近年"),
    ("2019-20年", "近年"),
    ("2019 年", "近年"),
    ("2019年", "近年"),
    ("2020 年抗议", "近年事件"),
    ("2020年抗议", "近年事件"),
    ("社会运动和抗议活动", "近年社会变动"),
    ("暴力事件和抗议活动", "近年公共安全事件"),
    ("抗议活动", "公众活动"),
    ("抗议", "公众活动"),
    ("示威", "公众活动"),
    ("社会信任降至", "社会信任面临挑战"),
    ("社会信任度极低", "社会信任面临挑战"),
    ("社会信任跌至", "社会信任面临挑战"),
    ("社会动荡", "近年社会变化"),
    ("社会撕裂", "社会分歧"),
    ("社会凝聚力受损", "社会凝聚面临挑战"),
    ("社会凝聚受损", "社会凝聚面临挑战"),
    # 单独年份
    ("1989", "数十年前"),
    ("2019", "近年"),
    ("2020", "近年"),
    ("2014", "近年"),
    # 政治敏感词
    ("六四", "[历史事件]"),
    ("天安门", "[历史地点]"),
    ("法轮", "[宗教问题]"),
    ("达赖", "[宗教领袖]"),
    ("藏独", "[地方议题]"),
    ("疆独", "[地方议题]"),
    ("台独", "[地方议题]"),
    ("维吾尔", "[少数民族]"),
    ("国安法", "[相关立法]"),
    ("党国", "[执政架构]"),
    ("独裁", "强政府"),
    ("专制", "强政府"),
    # 评判性词中性化
    ("镇压", "管控"),
    ("黑警", "执法"),
    ("警暴", "执法争议"),
    # 治理指标定义中性化: 主流程前置处理, 不等救援阶段再修
    ("corruption-quake mortality", "governance-transparency earthquake mortality"),
    ("corruption", "governance-transparency"),
    ("腐败率", "治理透明度风险"),
    ("腐败", "治理透明度问题"),
    ("被开发商俘获程度", "受单一利益影响程度"),
    ("开发商俘获", "单一利益影响"),
    ("投诉/抗议渠道", "公共服务诉求渠道"),
    ("抗议渠道", "公共服务诉求渠道"),
    ("政变", "体制变更"),
    ("动乱", "公共秩序中断"),
    ("动荡", "体制波动"),
    # 完整年+年字 (防 "2020 年" 残留 " 年")
    ("2020 年", "近年"), ("2020年", "近年"),
    ("2014 年", "近年"), ("2014年", "近年"),
    ("1989 年", "数十年前"), ("1989年", "数十年前"),
]

# 后处理: 修语义重复 (规则替换链产生)
POST_FIX = [
    ("近年近年公众活动", "近年公众活动"),
    ("近年近年事件", "近年事件"),
    ("近年近年社会", "近年社会"),
    ("近年近年", "近年"),
    ("近年 年", "近年"),
    ("近年年", "近年"),
]


def cleanse(text: str) -> str:
    """规则层清洗 (快、便宜、不调 LLM). 用于主 pipeline 预防."""
    if not isinstance(text, str) or not text.strip():
        return text
    cleaned = text
    for k, v in SENSITIVE_REPLACE:
        cleaned = cleaned.replace(k, v)
    # 后处理修语义重复
    for k, v in POST_FIX:
        cleaned = cleaned.replace(k, v)
    return cleaned


# 2026-06-01 用户拍板: city-aware cleanse — 只对中国相关城市清洗 reasoning,
# 其他国家城市保留真实描述 (恐袭/抗议/暴力等保持). 评分(tier)全城市真实, 不受约束.
CHINA_RELATED_CITIES = frozenset({
    # 中国大陆 - 直辖市
    "Beijing", "Shanghai", "Chongqing", "Tianjin",
    # 中国大陆 - 省会/重要城市
    "Guangzhou", "Shenzhen", "Chengdu", "Wuhan", "Xi'an", "Xian",
    "Hangzhou", "Nanjing", "Shenyang", "Qingdao", "Suzhou", "Dalian",
    "Nanchang", "Changsha", "Zhengzhou", "Jinan", "Hefei", "Fuzhou",
    "Kunming", "Guiyang", "Lanzhou", "Xining", "Yinchuan", "Harbin",
    "Changchun", "Taiyuan", "Shijiazhuang", "Hohhot",
    # 新疆/西藏 (政治敏感)
    "Ürümqi", "Urumqi", "Wulumuqi", "Kashgar", "Hotan",
    "Lhasa", "Shigatse",
    # SAR
    "Hong Kong", "Macau", "Macao", "Zhuhai [Macau]",
    # 台湾
    "Taipei", "Kaohsiung", "Taichung", "Tainan", "Taoyuan",
})


def is_china_related(city: str) -> bool:
    """判断城市是否涉及中国大陆/SAR/台湾 (扩展时直接加进 CHINA_RELATED_CITIES)."""
    return city in CHINA_RELATED_CITIES


def cleanse_if_china(text: str, city: str) -> str:
    """City-aware cleanse: 只对中国相关城市 reasoning 清洗, 其他原样保留."""
    if is_china_related(city):
        return cleanse(text)
    return text
