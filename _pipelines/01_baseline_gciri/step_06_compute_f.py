"""step_06_compute_f.py — F 因子 = mean(LLM 给的 6 个 ABS resilience)

V27.3.19 设计:
- F 是国家级有, 城市级没有, 所以 LLM 议会算 6 个 ABS indicator
- F_city = mean(6 个 ABS resilience), 0-1 范围
- F 用于 step_08 三能力计算 (Moncho 公式)

ABS 6 indicator (来自 cap_abs):
- Building_safety_governance
- Urban_ecosystem_health
- Urban_income_equity
- Urban_safety_and_social_cohesion
- Housing_habitability
- Infrastructure_quality
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "_code" / "llm"))

from tier_to_score import compute_resilience_from_tier  # noqa: E402

ABS_6 = {
    "Building_safety_governance",
    "Urban_ecosystem_health",
    "Urban_income_equity",
    "Urban_safety_and_social_cohesion",
    "Housing_habitability",
    "Infrastructure_quality",
}


def compute_f_for_cities(consul_csv: Path, out_dir: Path) -> dict[str, float]:
    """读 Consul 最终 tier, 计算每城 F 因子."""
    by_city = {}  # city -> list of (alias, resilience)
    with consul_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ind = row["indicator"]
            if ind not in ABS_6:
                continue
            tier = row["tier"]
            if not tier:
                continue
            try:
                res = compute_resilience_from_tier(tier)
            except ValueError:
                continue
            by_city.setdefault(row["city"], []).append((ind, res))

    f_factors = {}
    for city, vals in by_city.items():
        if len(vals) >= 5:  # 至少 5/6 indicator 有, 算 F
            f_factors[city] = sum(v[1] for v in vals) / len(vals)

    with (out_dir / "step_06_f.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["city", "f_factor", "n_indicators_used"])
        for city, f_val in sorted(f_factors.items()):
            n = len(by_city[city])
            w.writerow([city, round(f_val, 4), n])
    print(f"✓ step_06 F factor: {len(f_factors)} 城")
    return f_factors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/v01")
    args = p.parse_args()
    consul_csv = Path(args.out) / "step_05_consul.csv"
    compute_f_for_cities(consul_csv, Path(args.out))


if __name__ == "__main__":
    main()
