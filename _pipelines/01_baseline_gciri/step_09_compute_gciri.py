"""step_09_compute_gciri.py — 梯形 GCIRI 最终计算 (V27.3.20 真函数)

V2 算法 (170 国 MAE=1.156 / R²=0.9968):
1. 扫 R_Ph ∈ [0,1] 21 点: raw[j] = -A/P(r[j]×100×(1+F)/2, cap_resp, cap_rest)
2. RR_ymin_country = 4 维线性回归 (F, cap_resp, cap_rest, raw_PC_ymin)
3. rr_curve = (raw - raw.min())/(raw.max()-raw.min()) × (99.8 - RR_ymin) + RR_ymin
4. GCIRI = np.interp(min(R_Ph,1), r_grid, rr_curve)

V27.3.20 (2026-05-22): 从 STUB 升级到 trapezoid.compute_trapezoid_giri 真函数.
  输入: step_08_capacities.csv (议会真跑 cap_abs/resp/rest)
        step_07_r_ph_city.csv (从 7273 csv 真 R_Ph_city + F_city)
  输出: 50 城 GCIRI (跟 V27.3.15g 7273 表对齐, 跟 5/18 v06 transcript 救出的 45 城应能对上)

GCIRI = 论文术语 (Global City Infrastructure Resilience Index)
city-level GIRI = 代码术语 (跟国家级 GIRI 同公式)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "_code" / "giri"))

from trapezoid import compute_trapezoid_giri  # noqa: E402


def _read_csv_dict(path: Path) -> dict:
    """csv → {city: row_dict}"""
    out = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["city"]] = row
    return out


def _f(val):
    """str → float, 空串 → None."""
    if val in (None, "", "nan"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def compute_gciri_for_all(cap_csv: Path, r_ph_csv: Path, f_csv: Path, out_dir: Path) -> list[dict]:
    """读 step_08 (议会真跑三能力) + step_07 (真 R_Ph_city) + step_06 (议会真跑 F), 调 trapezoid 真函数算 GCIRI.

    V27.3.20: 真函数路径, 不再走 18 城 legacy STUB.
    F 来源: step_06_f.csv (议会 6 ABS resilience 平均), 不是 step_07 老 csv F (那是 5/13 V27.3.15g 老值).
    """
    print("[step_09] computing GCIRI via trapezoid.compute_trapezoid_giri (V27.5 真梯形)")

    cap_data = _read_csv_dict(cap_csv)
    r_ph_data = _read_csv_dict(r_ph_csv)
    # step_06 输出 city,f_factor,n_indicators_used — 单独读出 city → f_factor 映射
    f_data = {}
    with f_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row.get("f_factor")
            if v:
                try:
                    f_data[row["city"]] = float(v)
                except ValueError:
                    pass

    rows = []
    n_ok = 0
    n_missing = 0
    for city, cap in cap_data.items():
        cap_abs = _f(cap.get("cap_abs"))
        cap_resp = _f(cap.get("cap_resp"))
        cap_rest = _f(cap.get("cap_rest"))
        rph_row = r_ph_data.get(city, {})
        r_ph_city = _f(rph_row.get("r_ph_city"))
        # V27.3.20: F 用 step_06 议会真跑 (不是 step_07 老 csv)
        f_city = f_data.get(city)

        # 任一关键值缺 → missing
        if None in (cap_resp, cap_rest, r_ph_city, f_city):
            rows.append({
                "city": city,
                "cap_abs": cap_abs if cap_abs is not None else "",
                "cap_resp": cap_resp if cap_resp is not None else "",
                "cap_rest": cap_rest if cap_rest is not None else "",
                "r_ph_city": r_ph_city if r_ph_city is not None else "",
                "f_city": f_city if f_city is not None else "",
                "gciri": "",
                "source": rph_row.get("source", "missing"),
            })
            n_missing += 1
            continue

        # 真调 V2 梯形公式 (170 国 MAE=1.156)
        # 2026-06-03 polarity fix (门 3 漏修补): f_city 来自 step_06 是议会 ABS 6 mean resilience
        # polarity (高=好), 但 trapezoid 公式 cap_abs_j = R_Ph × 100 × (1+F)/2 期望 deficit
        # polarity (Cardona 2023 aggravating factor, 高=加剧). 翻 1-f_city 保 polarity 一致.
        # 实测 (50 城 b50 baseline): r 0.927→0.940, MAE 9.54→8.47, mean_diff +4.90→+2.28.
        # step_08 line 110 已正确翻 deficit (F_deficit = 1 - F_council), 本 step 漏修补齐.
        try:
            gciri = compute_trapezoid_giri(
                F_city=1 - f_city,   # POLARITY FIX 2026-06-03: resilience → deficit
                cap_resp=cap_resp,
                cap_rest=cap_rest,
                R_Ph=r_ph_city,
            )
        except Exception as e:
            print(f"⚠️  trapezoid fail for {city}: {e}", file=sys.stderr)
            rows.append({
                "city": city,
                "cap_abs": cap_abs, "cap_resp": cap_resp, "cap_rest": cap_rest,
                "r_ph_city": r_ph_city, "f_city": f_city,
                "gciri": "", "source": f"trapezoid_error: {e}",
            })
            n_missing += 1
            continue

        rows.append({
            "city": city,
            "cap_abs": round(cap_abs, 2) if cap_abs is not None else "",
            "cap_resp": round(cap_resp, 2),
            "cap_rest": round(cap_rest, 2),
            "r_ph_city": round(r_ph_city, 4),
            "f_city": round(f_city, 4),
            "gciri": round(gciri, 2),
            "source": "v27_3_20_real_trapezoid",
        })
        n_ok += 1

    out_csv = out_dir / "step_09_gciri.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "city", "cap_abs", "cap_resp", "cap_rest",
            "r_ph_city", "f_city", "gciri", "source",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"✓ step_09 GCIRI: {n_ok}/{len(rows)} 真值 + {n_missing} missing")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/v01")
    args = p.parse_args()
    out = Path(args.out)
    compute_gciri_for_all(
        out / "step_08_capacities.csv",
        out / "step_07_r_ph_city.csv",
        out / "step_06_f.csv",
        out,
    )


if __name__ == "__main__":
    main()
