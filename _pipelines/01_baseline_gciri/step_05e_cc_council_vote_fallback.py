"""step_05e_cc_council_vote_fallback.py — cc 议会投票兜底救援.

E 方案 v3 (2026-06-02 用户拍板, cc 介入思路):
- step_05d 多轮 AI 救援后 stagnation 仍剩 NaN cell (智增增 nginx 网关偶发抽风, deepseek 也救不回)
- 兜底机制: 直接用议会 R2 8 票均值 mid_score 映射回最近 tier
- 0 LLM 调用, 0 RMB, 几秒完成
- 议会民主投票 > 任何 LLM 兜底 (议会大部分都成功评了, Consul 撞 contentFilter 是网关问题不是评分问题)

逻辑:
  1. 找 step_05_consul.csv 剩余 NaN cell
  2. 对每 cell 取 R2 议会 8 票 tier (忽略 ∅)
  3. mid_score 均值映射回最近 tier
  4. 如 R2 全空 fallback R1 议会
  5. 写回 step_05_consul.csv (备份 .before_cc_fallback.csv)

实测 200 城 sample: 30 stuck NaN → 0 NaN, 0 LLM 调用, < 1 秒.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import pandas as pd


# 21 档 tier → mid_score 映射 (跟项目级 tier_to_score 一致)
TIER_MID = {
    "S+": 97.5, "S": 92.5,
    "A+": 87.5, "A": 82.5, "A-": 77.5,
    "B+": 72.5, "B": 67.5, "B-": 62.5,
    "C+": 57.5, "C": 52.5, "C-": 47.5,
    "D+": 42.5, "D": 37.5, "D-": 32.5,
    "E+": 27.5, "E": 22.5, "E-": 17.5,
    "F+": 12.5, "F": 7.5, "F-": 5,
    "BOT": 2.5,
}


def score_to_tier(s: float) -> str:
    """mid_score 映射回最近 tier."""
    return min(TIER_MID.items(), key=lambda kv: abs(kv[1] - s))[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out_dir = Path(args.out)

    co_path = out_dir / 'step_05_consul.csv'
    co = pd.read_csv(co_path)
    r1 = pd.read_csv(out_dir / 'step_03_r1.csv')
    r2 = pd.read_csv(out_dir / 'step_04_r2.csv')

    nan = co[(co['tier'].isna()) | (co['tier'] == '')]
    if not len(nan):
        print(f"✓ step_05e: 无 NaN, 跳过 cc 兜底")
        return

    # 备份
    backup = out_dir / 'step_05_consul.before_cc_fallback.csv'
    if not backup.exists():
        shutil.copy(co_path, backup)

    print(f"=== step_05e: cc 议会投票兜底 {len(nan)} NaN cell ===")
    n_filled = 0
    n_r1_fallback = 0
    stuck = []

    for idx, row in nan.iterrows():
        ind, city = row['indicator'], row['city']
        r2_sub = r2[(r2['indicator'] == ind) & (r2['city'] == city)]
        valid = [t for t in r2_sub['tier'].dropna().astype(str).tolist() if t in TIER_MID]

        # R2 全空 fallback R1
        if not valid:
            r1_sub = r1[(r1['indicator'] == ind) & (r1['city'] == city)]
            valid = [t for t in r1_sub['tier'].dropna().astype(str).tolist() if t in TIER_MID]
            if not valid:
                stuck.append((ind, city))
                continue
            n_r1_fallback += 1

        scores = [TIER_MID[t] for t in valid]
        mean_score = sum(scores) / len(scores)
        final_tier = score_to_tier(mean_score)

        co.at[idx, 'tier'] = final_tier
        co.at[idx, 'mean_score'] = round(mean_score, 2)
        co.at[idx, 'reasoning'] = f'[CC_FALLBACK] 议会 {len(valid)} 票 mean_score={mean_score:.1f} → tier {final_tier}'
        n_filled += 1

    co.to_csv(co_path, index=False)

    print(f"✓ step_05e 兜底 {n_filled} cell (R1 fallback: {n_r1_fallback}, 0 LLM 调用, 0 RMB)")
    if stuck:
        print(f"🟡 仍 {len(stuck)} cell R1+R2 都 0 票, 真留 NaN (share-ai §0.2 不掩盖):")
        for ind, city in stuck:
            print(f"   {ind} × {city}")

    final_nan = co[(co['tier'].isna()) | (co['tier'] == '')]
    print(f"\n✓ 最终 NaN: {len(final_nan)}/{len(co)} ({len(final_nan)/len(co)*100:.2f}%)")


if __name__ == '__main__':
    main()
