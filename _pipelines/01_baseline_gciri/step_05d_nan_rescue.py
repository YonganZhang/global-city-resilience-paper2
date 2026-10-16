"""step_05d_nan_rescue.py — MISSING 格救援: 清洗 R1 evidence 中性化 → 重跑 R2/Consul → 拿真值

设计 (用户 2026-06-01 拍板, 替代静默 fallback):
- share-ai §0.2: 错就让它显式可见. NaN 不用均值掩盖, 改用"清洗 evidence 重新评估"取得真值.
- 清洗策略: 双层 — 规则替换 (敏感词表) + 实际重新调 R2/Consul 看效果
- 清洗后 R2 调用走全 7 议会成员 (R1 reasoning 不含敏感词后, 网关不再拦截)
- 输出 step_05_consul_rescued.csv + _rescue_diff.csv (清洗前后对比, 审稿透明)
"""
from __future__ import annotations
import argparse, json, sys, time, re
from pathlib import Path
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "_code" / "llm"))
sys.path.insert(0, str(REPO / "_pipelines" / "01_baseline_gciri"))

from prompt_builder import build_r2_prompt, build_consul_prompt  # type: ignore
from client import run_llm_chat  # type: ignore
from step_05_consul import _apply_consul_decisions  # type: ignore


def _strip_md_wrap(raw: str) -> str:
    """inline (step_04 同款): 去 markdown 代码块包裹."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json|JSON)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    return raw.strip()

# 敏感词替换表 (双层: 长串先替, 单词后替防 substring 冲突)
SENSITIVE_REPLACE = [
    # 长串组合 (优先替换防 substring 漏)
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
    # 单独年份 (1989/2019/2020/2014 等) → 替成 "近年"
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
    # 评判性词替成中性
    ("镇压", "管控"),
    ("黑警", "执法"),
    ("警暴", "执法争议"),
    # 治理指标定义 / reasoning 中性化, 与 sensitive_cleanse.py 主流程保持一致
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
]


def cleanse(text: str) -> str:
    """规则层清洗 (快、便宜、不调 LLM)."""
    if not isinstance(text, str) or not text.strip():
        return text
    cleaned = text
    for k, v in SENSITIVE_REPLACE:
        cleaned = cleaned.replace(k, v)
    return cleaned


def _clean_cell(value) -> str:
    """CSV/LLM 中间产物里可能有 NaN/float/list; prompt 侧统一成字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, float) and pd.isna(value):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _coerce_rating_record(value) -> dict:
    """兼容异常模型输出: dict 正常, [dict] 展平, 其他坏值按空记录处理。"""
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]
    return {}


def _sanitize_ratings_map(value) -> dict:
    if not isinstance(value, dict):
        return {}
    return {city: _coerce_rating_record(record) for city, record in value.items()}


def cleanse_by_llm(text: str, max_tokens_out: int = 12000) -> str:
    """LLM 兜底清洗 (qwen-flash 中国模型, 网关不拦, 识别规则漏的敏感词)."""
    if not isinstance(text, str) or not text.strip():
        return text
    prompt = (
        "任务: 把以下文本中**任何可能触发中国网络监管敏感词审查**的内容改写成中性政策表达, "
        "完整保留原意和评估结论. 例如: 具体年份事件 → '近年事件'; 政治人物 → '相关方'; "
        "争议性社会事件 → '社会动态变化'. **只输出改写后的完整文本**, 不加任何解释、前缀或后缀.\n\n"
        "原文:\n" + text + "\n\n改写后的文本:"
    )
    try:
        result = run_llm_chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens_out,
            model="qwen-flash",
            trace_tag="rescue_llm_cleanse",
            web_search=False,
        )
        return result.strip() if isinstance(result, str) and result.strip() else text
    except Exception:
        return text  # LLM 清洗失败 → 用规则版本


def rewrite_city_def_by_ai(indicator: dict, retry_round: int = 1) -> dict:
    """用 deepseek-v4-flash 重写 indicator['city_def'] 中性化, 减少 contentFilter 1301 触发.

    2026-06-02 用户拍板 (E 方案: 快路径 + 慢救援):
    主 pipeline 撞智增增 nginx contentFilter (HTTP 400 + code 1301) → 标 NaN 不内嵌 retry.
    跑后 step_05d 用 AI 改 prompt 中性化重跑. 比规则 cleanse 保留更多语义.

    保留 6 子维度结构 + 学术编号 (1)-(6) + 文献引用 (Author Year).
    敏感词替换: '腐败' → '治理透明度', '投诉/抗议' → '公共服务诉求渠道',
    '政变/动荡/镇压' → '体制变更/体制波动/管控'.
    """
    from client import run_llm_chat  # noqa
    orig = indicator.get('city_def', '') or ''
    if not orig.strip():
        return indicator
    prompt = (
        f"任务: 把以下 {indicator.get('alias_name', '?')} 子维度定义**中性化**, "
        "去掉可能触发智增增 nginx 网关 contentFilter (HTTP 400 + code 1301) 的敏感词组合, "
        "如 '腐败' → '治理透明度', '投诉/抗议' → '公共服务诉求渠道', "
        "'政变 / 动荡 / 镇压' → '体制变更 / 体制波动 / 管控'. "
        f"严格保留 6 个子维度结构 + 学术编号 (1)-(6) + 文献引用 (Author Year). "
        f"完整输出新版定义文本, 不加任何解释/前缀/后缀.\n\n"
        f"原文:\n{orig}\n\n新版:"
    )
    try:
        result = run_llm_chat(
            [{"role": "user", "content": prompt}],
            max_tokens=2000, model="deepseek-v4-flash",
            trace_tag=f"rescue_rewrite_r{retry_round}", web_search=False,
        )
        # 校验: 新版长度合理 (不能太短, 不能空)
        if result and result.strip() and len(result) > len(orig) * 0.4:
            new_ind = dict(indicator)
            new_ind['city_def'] = result.strip()
            return new_ind
    except Exception as e:
        print(f"   [ai_rewrite] failed for {indicator.get('alias_name')}: {str(e)[:100]}")
    return indicator


def find_indicator_dict(meta: dict, alias_or_real: str) -> dict:
    for ind in meta['indicators']:
        if (ind['alias_name'] == alias_or_real or
                ind.get('real_name') == alias_or_real):
            return ind
    raise ValueError(f"Indicator not found: {alias_or_real}")


def _parse_r2_response(raw: str) -> dict:
    s = _strip_md_wrap(raw)
    if not s.startswith("{"):
        return {}
    try:
        p = json.loads(s)
        return p.get("ratings", {}) or {}
    except Exception:
        return {}


def rescue_one_batch(meta: dict, indicator_alias: str, batch_cities: list,
                     r1_data: dict, models: list, consul_model: dict,
                     diff_log: list) -> dict:
    """对一个 (indicator × batch) 重跑救援"""
    ind = find_indicator_dict(meta, indicator_alias)
    # 2026-06-02 (E 方案): 救援前 AI 重写 city_def 中性化, 减少 contentFilter 1301 触发
    ind = rewrite_city_def_by_ai(ind)
    print(f"\n>>> 救援: {indicator_alias} | batch {batch_cities[:3]}... ({len(batch_cities)} 城)")

    # 1. 清洗 R1 packet (规则替换)
    r1_for_batch = {}
    n_changed = 0
    for m in r1_data.get(indicator_alias, {}):
        r1_for_batch[m] = {}
        for c in batch_cities:
            orig = _coerce_rating_record(r1_data[indicator_alias][m].get(c, {}))
            if not orig:
                r1_for_batch[m][c] = {}
                continue
            orig_reason = _clean_cell(orig.get("reasoning", ""))
            cleaned_reason = cleanse(orig_reason)
            if cleaned_reason != orig_reason:
                n_changed += 1
                diff_log.append({
                    "indicator": indicator_alias, "city": c, "model": m,
                    "tier_r1": _clean_cell(orig.get("tier", "")),
                    "reasoning_orig": orig_reason[:300],
                    "reasoning_cleaned": cleaned_reason[:300],
                })
            r1_for_batch[m][c] = {"tier": _clean_cell(orig.get("tier", "")),
                                  "reasoning": cleaned_reason}
    print(f"   清洗 R1 reasoning 改动: {n_changed} 条")

    # 2. 重调 R2 (全 7 模型, 清洗后 prompt 应不再触发审查)
    system, user = build_r2_prompt(ind, batch_cities, r1_for_batch)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    r2_results = {}
    with ThreadPoolExecutor(max_workers=len(models)) as ex:
        futures = {}
        for m in models:
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
            futures[ex.submit(run_llm_chat, msgs,
                              max_tokens=16000,
                              model=m["vendor"],
                              trace_tag=f"rescue_r2_{m['vendor']}",
                              web_search=False)] = m
        for fut in as_completed(futures):
            m = futures[fut]
            mn = m["vendor"]
            try:
                raw = fut.result()
                ratings = _parse_r2_response(raw)
                ratings = _sanitize_ratings_map(ratings)
                r2_results[mn] = ratings if ratings else {"_error": "empty"}
            except Exception as e:
                r2_results[mn] = {"_error": str(e)[:100]}

    # 统计 R2 成功数
    n_r2_ok = sum(1 for r in r2_results.values() if isinstance(r, dict) and "_error" not in r)
    print(f"   R2 救援后成功模型: {n_r2_ok}/{len(models)}")

    # 3. Consul 重审 (用救援后的 R2; R2 reasoning 也 cleanse 防 Consul prompt 触发)
    from prompt_builder import build_consul_prompt as _build_consul  # type: ignore
    r2_for_batch = {}
    for m, ratings in r2_results.items():
        if not isinstance(ratings, dict) or "_error" in ratings:
            r2_for_batch[m] = {c: {} for c in batch_cities}
            continue
        r2_for_batch[m] = {}
        for c in batch_cities:
            rc = _coerce_rating_record(ratings.get(c, {}))
            # 清洗 R2 reasoning (LLM 自己写的, 可能又含敏感词)
            if _clean_cell(rc.get("reasoning", "")):
                rc = dict(rc)
                rc["reasoning"] = cleanse(_clean_cell(rc["reasoning"]))
            r2_for_batch[m][c] = rc

    final_per_city = {}
    if n_r2_ok >= 3:
        c_system, c_user = _build_consul(ind, batch_cities, r1_for_batch, r2_for_batch)
        # 双层清洗: 规则替换 → qwen LLM 兜底
        c_system = cleanse(c_system)
        c_user = cleanse(c_user)
        # 2026-06-02 (E 方案 v2): 去掉 qwen LLM cleanse 串行 (已被 AI 重写 city_def 替代, redundant 且 +8s/batch)
        try:
            raw = run_llm_chat(
                [{"role": "system", "content": c_system},
                 {"role": "user", "content": c_user}],
                max_tokens=16000,
                model=consul_model["vendor"],
                trace_tag=f"rescue_consul_{consul_model['vendor']}",
                web_search=False,
            )
            stripped = _strip_md_wrap(raw)
            if stripped.startswith("{"):
                parsed = json.loads(stripped)
                decisions = parsed.get("decisions", parsed.get("final", {}))
                if isinstance(decisions, list):
                    decisions = {d.get("city"): d for d in decisions if isinstance(d, dict)}
                final_per_city = _apply_consul_decisions(r2_for_batch, decisions, batch_cities)
                print(f"   ✓ Consul 救援成功")
        except Exception as e:
            print(f"   🔴 Consul 救援失败: {str(e)[:100]}")

    return final_per_city


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--consul-model', default=None,
                   help='覆盖 Sheet 7 Consul model (用于救援后期切换到 deepseek 绕过 gemini contentFilter)')
    args = p.parse_args()
    out_dir = Path(args.out)

    print(f"=== Phase 1: NaN 救援 (清洗 evidence 重跑 R2/Consul) on {out_dir.name} ===\n")

    # 读 meta + R1 + Consul (找 MISSING)
    meta = yaml.safe_load((out_dir / 'step_01_meta.yml').read_text())
    co = pd.read_csv(out_dir / 'step_05_consul.csv')

    # 找 MISSING 格
    missing = co[(co['tier'].isna()) | (co['tier'] == '')]
    print(f"找到 {len(missing)} 个 MISSING 格需救援")
    if not len(missing):
        print("(无 MISSING, 跳过救援)")
        return

    # 按 (indicator, batch) 聚合 (BATCH_SIZE = 10)
    BATCH_SIZE = 10
    cities = meta['cities']
    city_to_batch = {c: i // BATCH_SIZE for i, c in enumerate(cities)}

    by_ind_batch = {}
    for _, m in missing.iterrows():
        ind = m['indicator']; city = m['city']
        bi = city_to_batch.get(city)
        if bi is None: continue
        key = (ind, bi)
        by_ind_batch.setdefault(key, []).append(city)
    print(f"分 {len(by_ind_batch)} 个 (indicator, batch) 救援")

    # 加载 R1 data
    r1_data = {}
    r1 = pd.read_csv(out_dir / 'step_03_r1.csv')
    for _, row in r1.iterrows():
        r1_data.setdefault(row['indicator'], {}).setdefault(row['model'], {})[row['city']] = {
            "tier": _clean_cell(row['tier']),
            "reasoning": _clean_cell(row.get('reasoning', '')),
            "evidence_strength": _clean_cell(row.get('evidence_strength', 'medium'))}

    # 议会成员 + Consul 配置 (从 meta 拿)
    models = [m for m in meta['models'] if m.get('model') == '议会成员'][:8]
    consul_cands = [m for m in meta['models'] if m.get('model') == 'Consul 裁判']
    consul_model = consul_cands[0]
    # 2026-06-02 (E 方案 v2): 允许覆盖 Consul model — round 3+ 时切到 deepseek 绕过 gemini contentFilter
    if args.consul_model:
        consul_model = dict(consul_model)
        consul_model['vendor'] = args.consul_model
        print(f"  [override] Consul model 覆盖为: {args.consul_model}")
    print(f"议会: {len(models)} 成员, Consul: {consul_model['vendor']}")

    # 跑救援 — 2026-06-02 修: batch 间加 max_workers=48 并发, 跟 step_03/04/05 同款
    # 原 for 串行漂移 (200 城 sample 实测 ~4-5h), 改并发后 ~6 min (48x 加速)
    diff_log = []
    rescued_results = []  # [(indicator, city, final_dict)]
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one_rescue_task(item):
        (ind, bi), missing_cities_in_batch = item
        batch_start = bi * BATCH_SIZE
        full_batch = cities[batch_start: batch_start + BATCH_SIZE]
        try:
            final = rescue_one_batch(meta, ind, full_batch, r1_data, models,
                                     consul_model, diff_log)
            return [(ind, c, final[c]) for c in missing_cities_in_batch if c in final]
        except Exception as e:
            print(f"   🔴 batch {ind}/{bi} 救援异常: {e}")
            return []

    print(f"  [Rescue 并发] {len(by_ind_batch)} 任务, max_workers=48 (跟 step_03/04/05 同款)")
    n_done = 0
    with ThreadPoolExecutor(max_workers=48) as pool:
        futures = {pool.submit(_one_rescue_task, item): item for item in by_ind_batch.items()}
        for fut in as_completed(futures):
            rescued_results.extend(fut.result())
            n_done += 1
            if n_done % 10 == 0:
                print(f"  [Rescue] 完成 {n_done}/{len(by_ind_batch)} 任务")

    # 写救援结果回 consul
    co_rescued = co.copy()
    n_rescued = 0
    for ind, city, fd in rescued_results:
        fd = _coerce_rating_record(fd)
        mask = (co_rescued['indicator'] == ind) & (co_rescued['city'] == city)
        if not mask.any(): continue
        ft = _clean_cell(fd.get('tier', ''))
        if ft:
            co_rescued.loc[mask, 'tier'] = ft
            co_rescued.loc[mask, 'mean_score'] = fd.get('mean_score', '')
            co_rescued.loc[mask, 'reasoning'] = '[RESCUED] ' + _clean_cell(fd.get('reasoning', ''))
            co_rescued.loc[mask, 'kept_models'] = fd.get('kept_models', '')
            co_rescued.loc[mask, 'drop_models'] = fd.get('drop_models', '')
            n_rescued += 1

    co_rescued.to_csv(out_dir / 'step_05_consul_rescued.csv', index=False)
    pd.DataFrame(diff_log).to_csv(out_dir / '_rescue_diff.csv', index=False)

    print(f"\n=== 救援总结 ===")
    print(f"  MISSING 格: {len(missing)}")
    print(f"  救援成功: {n_rescued}")
    print(f"  仍 NaN: {len(missing) - n_rescued} (真留 NaN, 不掩盖)")
    print(f"  清洗 diff 条数: {len(diff_log)}")
    print(f"\n✓ 写 step_05_consul_rescued.csv")
    print(f"✓ 写 _rescue_diff.csv (清洗前后对比, 审稿透明)")


if __name__ == '__main__':
    main()
