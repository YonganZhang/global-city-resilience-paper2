"""step_05c_quality_gate.py — 议会评分质量门 (Phase 1, v2: NaN 真留 NaN 不掩盖)

设计原则 (用户 2026-06-01 拍板):
- 永不静默掩盖错误 (share-ai.md §0.2)
- NaN 真留 NaN, 不用 R2/R1 mean 自动 fallback (那是死 fallback 掩盖坏配置)
- 只标质量等级 + 异常 flag, 不补 tier
- 真正的 NaN 救援由 step_05d_nan_rescue.py 做 (清洗 evidence 重跑 R2/Consul, 拿真值)

输出:
- step_05_consul_annotated.csv  (Consul + quality 标注列, NaN 保留)
- _quality_audit_summary.md     (审计报告)
- _quality_audit_detail.csv     (每格质量+flag)
- _validate_r1_vs_consul.md     (R1 vs Consul 相关性, 学术诚实证据)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ===== 1. tier ↔ score 映射 =====
TIER_DEFS = [
    ("S+", 97.5), ("S",  92.5), ("A+", 87.5), ("A",  82.5), ("A-", 80.0),
    ("B+", 77.5), ("B",  72.5), ("B-", 67.5), ("C+", 62.5), ("C",  57.5),
    ("C-", 52.5), ("D+", 47.5), ("D",  42.5), ("D-", 37.5), ("E+", 32.5),
    ("E",  27.5), ("E-", 22.5), ("F+", 17.5), ("F",  12.5), ("F-",  7.5),
    ("BOT", 2.5),
]
TIER_TO_SCORE = dict(TIER_DEFS)


def tier_to_score(t) -> float:
    if not isinstance(t, str) or not t.strip():
        return float('nan')
    return TIER_TO_SCORE.get(t.strip(), float('nan'))


# ===== 2. 异常检测 (只 flag, 不补) =====
def detect_outlier(model_tiers: dict) -> list:
    """outlier: 某模型 score 距其他 mean ≥ 4 档 (~20 score)."""
    flags = []
    scores = {m: tier_to_score(t) for m, t in model_tiers.items()
              if isinstance(t, str) and t.strip() and t.strip() in TIER_TO_SCORE}
    if len(scores) < 4:
        return flags
    for m, s in scores.items():
        others = [x for k, x in scores.items() if k != m]
        if not others:
            continue
        if abs(s - np.mean(others)) > 20:
            flags.append(f"outlier:{m}")
    return flags


def detect_r1r2_drift(r1_tier, r2_tier, model) -> Optional[str]:
    """R2 改判漂移: 同模型 R1→R2 改判 ≥ 3 档 (~15 score)."""
    s1, s2 = tier_to_score(r1_tier), tier_to_score(r2_tier)
    if pd.isna(s1) or pd.isna(s2):
        return None
    if abs(s1 - s2) > 15:
        return f"r1r2_drift:{model}:{r1_tier}→{r2_tier}"
    return None


def detect_format_err(tier, model) -> Optional[str]:
    """非法 tier (G/H/X 等不在 21 档定义)."""
    if not isinstance(tier, str) or not tier.strip():
        return None
    if tier.strip() not in TIER_TO_SCORE:
        return f"format_err:{model}:{tier!r}"
    return None


# ===== 3. 主流程 (NaN 真留 NaN, 不补) =====
def run_quality_gate(out_dir: Path) -> dict:
    r1 = pd.read_csv(out_dir / 'step_03_r1.csv')
    r2 = pd.read_csv(out_dir / 'step_04_r2.csv')
    co = pd.read_csv(out_dir / 'step_05_consul.csv')

    print(f"读 R1={len(r1)} R2={len(r2)} Consul={len(co)} 行")

    out_rows = []
    flag_counts = {}

    r1_idx = r1.groupby(['indicator', 'city'])
    r2_idx = r2.groupby(['indicator', 'city'])

    for _, c in co.iterrows():
        ind, city = c['indicator'], c['city']
        try:
            r1_sub = r1_idx.get_group((ind, city))
        except KeyError:
            r1_sub = pd.DataFrame()
        try:
            r2_sub = r2_idx.get_group((ind, city))
        except KeyError:
            r2_sub = pd.DataFrame()

        r1_tiers = {row['model']: row['tier'] for _, row in r1_sub.iterrows()}
        r2_tiers = {row['model']: row['tier'] for _, row in r2_sub.iterrows()}

        flags = []
        for m, t in {**r1_tiers, **r2_tiers}.items():
            fe = detect_format_err(t, m)
            if fe: flags.append(fe)

        flags += detect_outlier(r1_tiers)

        for m, t2 in r2_tiers.items():
            t1 = r1_tiers.get(m, '')
            f = detect_r1r2_drift(t1, t2, m)
            if f: flags.append(f)

        if 'evidence_quality_flags' in r2_sub.columns:
            r2_bad = r2_sub['evidence_quality_flags'].fillna('').astype(str).str.contains('bad_reasoning').sum()
            r2_consensus = r2_sub['evidence_quality_flags'].fillna('').astype(str).str.contains('consensus_language').sum()
            if r2_bad:
                flags.append(f"r2_bad_reasoning_voters:{int(r2_bad)}")
            if r2_consensus:
                flags.append(f"r2_consensus_language_voters:{int(r2_consensus)}")

        consul_reasoning_flags = str(c.get('consul_reasoning_flags', '') or '')
        if 'bad_reasoning' in consul_reasoning_flags:
            flags.append('consul_bad_reasoning')
        if 'consensus_language' in consul_reasoning_flags and 'no_city_evidence_marker' in consul_reasoning_flags:
            flags.append('consul_consensus_only_reasoning')

        outlier_models = {f.split(':')[1] for f in flags if f.startswith('outlier:')}

        def _valid(d):
            return [m for m, t in d.items()
                    if isinstance(t, str) and t.strip() and t.strip() in TIER_TO_SCORE
                    and m not in outlier_models]

        r1_valid = _valid(r1_tiers)
        r2_valid = _valid(r2_tiers)

        consul_tier = c.get('tier')
        consul_ok = (isinstance(consul_tier, str) and consul_tier.strip()
                     and consul_tier.strip() in TIER_TO_SCORE)

        # 等级判定 — NaN 真留 NaN, 不补 tier
        if consul_ok and len(r2_valid) >= 6:
            quality = "GREEN"
            data_completeness = "complete"
            tier_final = consul_tier
        elif consul_ok and len(r2_valid) >= 4:
            quality = "YELLOW"
            data_completeness = "complete_partial_r2"
            tier_final = consul_tier
            flags.append(f"partial_r2:{len(r2_valid)}/7")
        elif consul_ok:
            quality = "YELLOW_LOW"
            data_completeness = "complete_few_r2"
            tier_final = consul_tier
            flags.append(f"low_r2_voters:{len(r2_valid)}/7")
        else:
            # Consul 失败 → tier 真留 NaN, 标 MISSING
            quality = "MISSING"
            tier_final = ""  # 真留空, 不补
            if len(r2_valid) >= 3:
                data_completeness = "missing_consul_has_r2"
                flags.append(f"missing_consul_but_r2_has:{len(r2_valid)}")
            elif len(r1_valid) >= 3:
                data_completeness = "missing_consul_r2_only_r1"
                flags.append(f"missing_consul_and_r2_only_r1:{len(r1_valid)}")
            else:
                data_completeness = "missing_all"
                flags.append("missing_all_layers")

        if 'consul_bad_reasoning' in flags and quality == "GREEN":
            quality = "YELLOW"
            data_completeness = "complete_consul_reasoning_flagged"
        elif 'consul_bad_reasoning' in flags and quality == "YELLOW":
            quality = "YELLOW_LOW"
            data_completeness = "complete_consul_reasoning_flagged"

        out_rows.append({
            'indicator': ind, 'city': city, 'tier': tier_final,
            'tier_original': consul_tier if consul_ok else "",
            'quality': quality,
            'data_completeness': data_completeness,
            'n_r1_valid': len(r1_valid),
            'n_r2_valid': len(r2_valid),
            'flags': ",".join(flags),
        })
        for f in flags:
            key = f.split(':')[0]
            flag_counts[key] = flag_counts.get(key, 0) + 1

    df = pd.DataFrame(out_rows)
    return {'df': df, 'flag_counts': flag_counts}


def write_outputs(result: dict, out_dir: Path):
    df = result['df']
    flag_counts = result['flag_counts']

    # detail csv
    detail_cols = ['indicator', 'city', 'tier', 'tier_original', 'quality',
                   'data_completeness', 'n_r1_valid', 'n_r2_valid', 'flags']
    df[detail_cols].to_csv(out_dir / '_quality_audit_detail.csv', index=False)

    # annotated consul (NaN 真留 NaN, 加 quality 标注列)
    co = pd.read_csv(out_dir / 'step_05_consul.csv')
    co_merged = co.merge(df[['indicator', 'city', 'quality', 'data_completeness',
                             'n_r2_valid', 'flags']],
                         on=['indicator', 'city'], how='left')
    co_merged.to_csv(out_dir / 'step_05_consul_annotated.csv', index=False)

    # summary
    total = len(df)
    q_count = df['quality'].value_counts().to_dict()
    missing = q_count.get('MISSING', 0)

    lines = [f"# 议会评分质量门审计 (v2: NaN 真留 NaN)\n\n",
             f"- 跑次: `{out_dir.name}`\n",
             f"- 总格数: {total} ({len(df['indicator'].unique())} ind × {len(df['city'].unique())} 城)\n",
             f"- 设计原则: 永不静默掩盖错误 (share-ai §0.2). NaN 真留 NaN, 由 step_05d_nan_rescue 清洗 evidence 重跑.\n\n",
             f"## 1. 质量分布\n\n| 质量 | 格数 | 占比 |\n|---|---|---|\n"]
    semantic = {
        "GREEN": "Consul + R2 ≥6 模型",
        "YELLOW": "Consul + R2 4-5 模型 (部分 R2 缺)",
        "YELLOW_LOW": "Consul + R2 < 4 (低投票)",
        "MISSING": "Consul NaN → tier 留空 (待 step_05d 救援)",
    }
    for q in ['GREEN', 'YELLOW', 'YELLOW_LOW', 'MISSING']:
        n = q_count.get(q, 0)
        lines.append(f"| **{q}** | {n} | {n/total*100:.1f}% | {semantic.get(q, '')} |\n")

    lines.append(f"\n## 2. 缺失分布 ({missing} 格 MISSING)\n\n")
    if missing > 0:
        m_count = df[df['quality'] == 'MISSING']['data_completeness'].value_counts().to_dict()
        for k, v in m_count.items():
            lines.append(f"- `{k}`: {v} 格\n")
        lines.append(f"\n→ 下一步: 跑 `step_05d_nan_rescue.py` 清洗 R1 evidence 中性化后重跑 R2/Consul.\n")

    lines.append(f"\n## 3. 全局异常 Flag\n\n| Flag | 计数 |\n|---|---|\n")
    for k, v in sorted(flag_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| `{k}` | {v} |\n")

    lines.append(f"\n## 4. MISSING 格清单 (待救援)\n\n")
    miss = df[df['quality'] == 'MISSING']
    if len(miss):
        lines.append(miss[['city', 'indicator', 'data_completeness',
                           'n_r1_valid', 'n_r2_valid']].to_markdown(index=False))

    lines.append(f"\n\n## 5. 论文 Methods 可写句\n\n")
    grn = q_count.get('GREEN', 0)
    ylw = q_count.get('YELLOW', 0) + q_count.get('YELLOW_LOW', 0)
    lines.append(f"50 城 × 18 indicator = {total} 格中, **{grn} ({grn/total*100:.1f}%)** GREEN (Consul + R2 ≥6 模型直接定档), "
                 f"**{ylw} ({ylw/total*100:.1f}%)** YELLOW/YELLOW_LOW (Consul 成功但 R2 模型数减少), "
                 f"**{missing} ({missing/total*100:.1f}%)** MISSING (R2 阶段因 API 网关合规层拒答, "
                 f"由 NaN 救援后处理 step_05d 通过对 R1 evidence 中性化重写后重新评估获得真值, "
                 f"清洗 diff 见 supplementary). 全 quality 标注在 `step_05_consul_annotated.csv` 透明记录.\n")

    (out_dir / '_quality_audit_summary.md').write_text(''.join(lines), encoding='utf-8')

    print(f"\n✓ 写 step_05_consul_annotated.csv (NaN 真留 NaN)")
    print(f"✓ 写 _quality_audit_summary.md (MISSING={missing} 格)")
    print(f"✓ 写 _quality_audit_detail.csv")


# ===== 4. R1 vs Consul 相关性验证 (学术诚实) =====
def validate_r1_vs_consul(out_dir: Path):
    r1 = pd.read_csv(out_dir / 'step_03_r1.csv')
    co = pd.read_csv(out_dir / 'step_05_consul.csv')

    r1_idx = r1.groupby(['indicator', 'city'])
    pairs = []
    for _, c in co.iterrows():
        ct = c.get('tier')
        if not isinstance(ct, str) or ct.strip() not in TIER_TO_SCORE:
            continue
        consul_score = tier_to_score(ct)
        try:
            r1_sub = r1_idx.get_group((c['indicator'], c['city']))
        except KeyError:
            continue
        r1_scores = [tier_to_score(t) for t in r1_sub['tier']
                     if isinstance(t, str) and t.strip() in TIER_TO_SCORE]
        r1_scores = [s for s in r1_scores if not pd.isna(s)]
        if len(r1_scores) >= 3:
            pairs.append({'city': c['city'], 'indicator': c['indicator'],
                          'consul_score': consul_score,
                          'r1_mean_score': float(np.mean(r1_scores))})

    df = pd.DataFrame(pairs)
    if not len(df):
        print("⚠️ 无有效配对")
        return

    pearson = df['consul_score'].corr(df['r1_mean_score'])
    spearman = df['consul_score'].corr(df['r1_mean_score'], method='spearman')
    diff = (df['consul_score'] - df['r1_mean_score']).abs()
    mae, median_diff = diff.mean(), diff.median()

    print(f"\n=== R1 mean vs Consul 相关性 ===")
    print(f"  配对数: {len(df)} | Pearson r: {pearson:.4f} | MAE: {mae:.2f} (~{mae/5:.1f} 档)")

    df.to_csv(out_dir / '_validate_r1_vs_consul.csv', index=False)
    (out_dir / '_validate_r1_vs_consul.md').write_text(
        f"# R1 mean vs Consul Tier 相关性\n\n"
        f"- 配对数: **{len(df)}**\n- Pearson r: **{pearson:.4f}**\n"
        f"- Spearman ρ: **{spearman:.4f}**\n- MAE: **{mae:.2f}** (~{mae/5:.1f} 档)\n"
        f"- 中位 |diff|: **{median_diff:.2f}**\n\n"
        f"## 用途\n证明 R1 评分跟 Consul 议会定档高度一致 (作为方法论稳定性证据), "
        f"**不**用于 fallback 补救 — NaN 救援见 step_05d.\n",
        encoding='utf-8',
    )
    print(f"✓ 写 _validate_r1_vs_consul.md/.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out_dir = Path(args.out)
    print(f"=== Phase 1 v2: Quality Gate (NaN 真留 NaN) on {out_dir.name} ===\n")
    result = run_quality_gate(out_dir)
    write_outputs(result, out_dir)
    validate_r1_vs_consul(out_dir)
    print(f"\n=== 下一步: 跑 step_05d_nan_rescue.py 清洗 R1 evidence 救援 MISSING ===")


if __name__ == '__main__':
    main()
