"""step_08_compute_capacities.py — 三能力 cap_abs / cap_resp / cap_rest

V27.3.19 公式 (GIRI 官方梯形 Moncho's Equation):
- cap_abs  = min(R_Ph_city, 1) × 100 × (1 + F_deficit) / 2
- cap_resp = (1 - mean(6 个 RESP resilience)) × 100   # 翻成 deficit polarity
- cap_rest = mean(6 个 REC resilience) × 100          # 保留 resilience polarity

5/23 POLARITY FIX (门 3, 见 .claude/rules/polarity-and-gates.md):
  议会 tier_to_score 输出 resilience polarity (S+=0.975 高, BOT=0.025 低).
  官方 Cardona 2023 梯形要: F deficit + cap_resp deficit + cap_rest resilience.
  本 step 进梯形前翻转 F + cap_resp, cap_rest 不翻.
  csv 额外输出 F_council / cap_resp_council 原值供 audit.

输入:
- step_05_consul.csv (LLM 议会最终 tier → resilience)
- step_06_f.csv (F 因子, resilience polarity)
- step_07_r_ph_city.csv (R_Ph_city)

输出: outputs/v<NN>/step_08_capacities.csv  50 城 × cap_abs/resp/rest (公式入参) + audit 列
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "_code" / "llm"))

from tier_to_score import compute_resilience_from_tier  # noqa: E402

RESP_6 = {
    "Urban_mobile_connectivity",
    "Municipal_public_integrity",
    "Local_financial_buffer_capacity",
    "Logistics_performance_index",
    "Local_economic_stability",
    "Local_governance_stability",
}
REC_6 = {
    "Access_to_quality_education",
    "Economic_complexity_index",
    "Government_Effectiveness_Index",
    "Human_Development_Index",
    "Research_Development",
    "Technology_achievement_index",
}


def compute_capacities_for_all(consul_csv: Path, f_csv: Path, r_ph_csv: Path, out_dir: Path) -> dict:
    """读 3 个 csv, 算 50 城三能力."""
    # 1. 读 consul.csv 抽 resilience by city × category
    resp_by_city = {}
    rec_by_city = {}
    with consul_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ind = row["indicator"]
            tier = row["tier"]
            city = row["city"]
            if not tier:
                continue
            try:
                res = compute_resilience_from_tier(tier)
            except ValueError:
                continue
            if ind in RESP_6:
                resp_by_city.setdefault(city, []).append(res)
            elif ind in REC_6:
                rec_by_city.setdefault(city, []).append(res)

    # 2. 读 F factor
    f_factors = {}
    with f_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                f_factors[row["city"]] = float(row["f_factor"])
            except ValueError:
                pass

    # 3. 读 R_Ph_city
    r_ph = {}
    with r_ph_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row["r_ph_city"]
            if v:
                try:
                    r_ph[row["city"]] = float(v)
                except ValueError:
                    pass

    # 4. 计算三能力 — V27.3.20 修 S5 (paper1 audit): 改调 trapezoid.compute_three_capabilities 单一真源
    # 改 trapezoid 公式 → 自动传播到 step_08 (不再内联)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "_code" / "giri"))
    from trapezoid import compute_three_capabilities  # noqa: E402

    results = {}
    all_cities = set(resp_by_city) | set(rec_by_city) | set(f_factors) | set(r_ph)
    for city in sorted(all_cities):
        F_council = f_factors.get(city)   # resilience polarity 0-1 (议会 6 ABS mean)
        R = r_ph.get(city)
        resp_vals = resp_by_city.get(city, [])
        rec_vals = rec_by_city.get(city, [])
        cap_resp_mean_council = sum(resp_vals) / len(resp_vals) if len(resp_vals) >= 5 else None  # resilience 0-1
        cap_rest_mean = sum(rec_vals) / len(rec_vals) if len(rec_vals) >= 5 else None              # resilience 0-1

        # POLARITY FIX 5/23 (门 3): F + cap_resp 议会 resilience → 官方公式 deficit
        F_deficit = (1 - F_council) if F_council is not None else None
        cap_resp_deficit_mean = (1 - cap_resp_mean_council) if cap_resp_mean_council is not None else None

        if F_deficit is not None and R is not None and cap_resp_deficit_mean is not None and cap_rest_mean is not None:
            caps = compute_three_capabilities(F_deficit, cap_resp_deficit_mean, cap_rest_mean, R)
            results[city] = {
                "F": F_deficit, "R_Ph_city": caps["R_Ph_city"],
                "cap_abs": caps["cap_abs"], "cap_resp": caps["cap_resp"], "cap_rest": caps["cap_rest"],
                "F_council_resilience": F_council,
                "cap_resp_council_resilience": cap_resp_mean_council * 100,
            }
        else:
            results[city] = {
                "F": F_deficit, "R_Ph_city": R,
                "cap_abs": (min(R, 1) * 100 * (1 + F_deficit) / 2) if (F_deficit is not None and R is not None) else None,
                "cap_resp": cap_resp_deficit_mean * 100 if cap_resp_deficit_mean is not None else None,
                "cap_rest": cap_rest_mean * 100 if cap_rest_mean is not None else None,
                "F_council_resilience": F_council,
                "cap_resp_council_resilience": cap_resp_mean_council * 100 if cap_resp_mean_council is not None else None,
            }

    with (out_dir / "step_08_capacities.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["city", "F", "R_Ph_city", "cap_abs", "cap_resp", "cap_rest",
                    "F_council_resilience", "cap_resp_council_resilience"])
        for city, r in results.items():
            w.writerow([city,
                        round(r["F"], 4) if r["F"] is not None else "",
                        round(r["R_Ph_city"], 4) if r["R_Ph_city"] is not None else "",
                        round(r["cap_abs"], 2) if r["cap_abs"] is not None else "",
                        round(r["cap_resp"], 2) if r["cap_resp"] is not None else "",
                        round(r["cap_rest"], 2) if r["cap_rest"] is not None else "",
                        round(r["F_council_resilience"], 4) if r["F_council_resilience"] is not None else "",
                        round(r["cap_resp_council_resilience"], 2) if r["cap_resp_council_resilience"] is not None else ""])
    n_complete = sum(1 for r in results.values() if r["cap_abs"] is not None and r["cap_resp"] is not None and r["cap_rest"] is not None)
    print(f"✓ step_08 capacities: {n_complete}/{len(results)} 城三能力完整")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/v01")
    args = p.parse_args()
    out = Path(args.out)
    compute_capacities_for_all(
        out / "step_05_consul.csv",
        out / "step_06_f.csv",
        out / "step_07_r_ph_city.csv",
        out,
    )


if __name__ == "__main__":
    main()
