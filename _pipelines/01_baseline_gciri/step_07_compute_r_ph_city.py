"""step_07_compute_r_ph_city.py — R_Ph 国家级 → 城市级 HEV 下推 (V28 真算, 接真源 master_v27_6)

R_Ph_city = R_Ph_national × H_ratio^aH × E_ratio^aE × V_ratio^aV → soft cap (aH/aE/aV=0.8/0.3/0.3)
真算引擎: _code/giri/r_ph_downscale.downscale_r_ph_for_master (接 master_v27_6 真源).

V28 (2026-05-28): 从"查 5/13 死 CSV (giri_7273_final.csv)"升级到"真源 master_v27_6 真算".
  旧版 read giri_7273_final.csv (7273 城旧表 + 老列名 R_Ph_city/R_Ph_national) 不真算 → 废弃.
  新版 load master_v27_6 (6275 城真源) → 真算 H_ratio (8 灾害几何平均) + E_ratio + 国家级锚.
  V_ratio (议会 6 ABS deficit 脆弱性微调) 已接入: b50/custom 可按 city 名, full 模式按 ucdb_id.

benchmark 缺城 fallback 随 active master 不同 (见 _MISSING_FALLBACK_BY_MASTER):
  v5_8: Macau→HK 代理, Male→0.95.  v28: Macau 有真数据 (Zhuhai [Macau]); Male→0.95;
  Shenzhen 已作为专属补入行进入 v28_ucdb (ucdb_id=999991); Male 仍缺.

输出:
  outputs/v<NN>/step_07_r_ph_city.csv              50 城 benchmark (paper-1 pipeline 契约)
  outputs/v<NN>/step_07_r_ph_city_full_6275.parquet active master 全量真算 (历史文件名保留)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "_code" / "giri"))
sys.path.insert(0, str(REPO_ROOT / "_pipelines" / "01_baseline_gciri"))

import master_io  # noqa: E402
from r_ph_downscale import downscale_r_ph_for_master, compute_v_ratio, DOWNSCALE_METHODS  # noqa: E402
import step_01_load_metadata as meta  # noqa: E402
from step_01_load_metadata import (  # noqa: E402
    MASTER_MISSING, BENCHMARK_50, load_master_name_for_benchmark, extract_city_record,
)
import pandas as pd  # noqa: E402

# 议会 6 ABS indicator alias (resilience polarity, S+=最强 → BOT=最弱)
ABS_ALIAS = [
    'Building_safety_governance', 'Urban_ecosystem_health',
    'Urban_income_equity', 'Urban_safety_and_social_cohesion',
    'Housing_habitability', 'Infrastructure_quality',
]
# 档位 → 分数用议会的 21 档单一真源 (含 A-=80.0)。2026-10-21 前这里自带一张缺 A- 的 20 档表,
# map 后 dropna 静默丢掉了 351 城 538 条 A- 吸收评分, V_ratio 偏高 (v28_full5862_20260702_020421 受影响)。
sys.path.append(str(REPO_ROOT / "_code" / "llm"))
from tier_to_score import TIER_TO_MIDPOINT  # noqa: E402

# display 名 → iso3 (撞名城 London CAN vs GBR 用 (city, iso3) 联合 key 消歧)
_NAME_TO_ISO3 = dict(BENCHMARK_50)


# benchmark 缺城的代理/硬值补全 — **随 active master 表不同** (缺城清单本身就不同):
#   v5_8 : Macau (缺)→HK 代理, Male (缺)→0.95 硬值 (5/23 拍板).
#   v28  : Macau 已有真数据 (Zhuhai [Macau], 不再代理); Male→0.95 硬值;
#          Shenzhen 已用专属补入行进入 v28_ucdb (ucdb_id=999991); 不再 fallback.
_MISSING_FALLBACK_BY_MASTER = {
    "v5_8": {
        "Macau": {"proxy_city": "Hong Kong", "rationale": "邻近 HK + 同 China region + 同台风暴露 (5/23 拍板)"},
        "Male":  {"proxy_city": None, "r_ph_city": 0.95,
                  "rationale": "Maldives 印度洋低地 + 海啸+海平面上升 + 国 GIRI 57.5 真高暴露 (5/23 拍板)"},
    },
    "v28_ucdb": {
        "Male":  {"proxy_city": None, "r_ph_city": 0.95,
                  "rationale": "Maldives 印度洋低地 + 海啸+海平面上升 + 国 GIRI 57.5 真高暴露 (5/23 拍板)"},
        # Shenzhen 已在 master 中, 不再 fallback.
    },
}
MISSING_FALLBACK = _MISSING_FALLBACK_BY_MASTER.get(
    meta._active_master_key(), _MISSING_FALLBACK_BY_MASTER["v28_ucdb"]
)

_FIELDS = [
    "city", "display_city", "ucdb_id", "iso3_for_nat",
    "r_ph_city", "r_ph_national", "h_ratio", "e_ratio", "v_ratio", "source",
]


def _row_from_master(city: str, rec, source: str, city_record: dict | None = None) -> dict:
    """从全量真算结果一行 (Series) 抽 step_07 输出字段."""
    city_record = city_record or extract_city_record(city)
    return {
        "city": city,
        "display_city": city_record.get("city") or rec.get("city", city),
        "ucdb_id": city_record.get("ucdb_id") or rec.get("ucdb_id", ""),
        "iso3_for_nat": city_record.get("iso3_for_nat") or rec.get("iso3_for_nat", ""),
        "r_ph_city": round(float(rec["R_Ph_city"]), 6) if rec["R_Ph_city"] == rec["R_Ph_city"] else "",
        "r_ph_national": round(float(rec["R_Ph__RAW_nat"]), 6) if rec["R_Ph__RAW_nat"] == rec["R_Ph__RAW_nat"] else "",
        "h_ratio": round(float(rec["H_ratio"]), 4),
        "e_ratio": round(float(rec["E_ratio"]), 4),
        "v_ratio": round(float(rec["V_ratio"]), 4),
        "source": source,
    }


def build_council_v_ratio(
    consul_csv: Path,
    master_df: pd.DataFrame,
    city_records: list[dict] | None = None,
) -> dict:
    """从 step_05_consul.csv 算 V_ratio dict per city (议会 6 ABS deficit / nat ABS deficit).

    2026-06-03 接通: step_07 docstring P5.2 待接, 之前一直 None → V=1.0 退化 (议会脆弱性
    信号完全没进 R_Ph). Tahoua/Abéché 偏差大根因之一.

    议会 tier (resilience polarity) → score (0-100) → deficit = (100-score)/100
    → mean over 6 ABS indicators → city_abs_def. nat_abs_def 直接用 master F__DEF_nat 列
    (= 6 ABS deficit nat-aggregate, NER=0.7358 等). compute_v_ratio 内已含 clip[0.1, 10].
    """
    if not consul_csv.exists():
        return {}
    c = pd.read_csv(consul_csv)
    abs_only = c[c['indicator'].isin(ABS_ALIAS)].copy()
    if abs_only.empty:
        return {}
    abs_only = abs_only.dropna(subset=['tier'])
    unknown = set(abs_only['tier']) - set(TIER_TO_MIDPOINT)
    if unknown:
        raise ValueError(f"step_05_consul.csv 含未知档位 {sorted(unknown)}")
    abs_only['score'] = abs_only['tier'].map(TIER_TO_MIDPOINT)
    abs_only['deficit'] = (100 - abs_only['score']) / 100
    city_abs_def = abs_only.groupby('city')['deficit'].mean().to_dict()

    records_by_key = {r.get("city_key"): r for r in (city_records or []) if r.get("city_key")}
    master_by_ucdb = {}
    if "ucdb_id" in master_df.columns:
        master_by_ucdb = {
            int(row["ucdb_id"]): row
            for row in master_df.dropna(subset=["ucdb_id"]).to_dict("records")
        }

    # 城市 → iso3 (benchmark/custom 兼容路径)
    city_iso = master_df.drop_duplicates('city').set_index('city')['iso3_for_nat'].to_dict()
    # iso3 → F__DEF_nat (国家 6 ABS aggregate, 一国所有城同值, first 即可)
    nat_def_by_iso = master_df.groupby('iso3_for_nat')['F__DEF_nat'].first().to_dict()

    v_dict = {}
    for city_key, c_def in city_abs_def.items():
        record = records_by_key.get(city_key) or extract_city_record(city_key)
        ucdb_id = record.get("ucdb_id")
        iso = record.get("iso3_for_nat")
        v_key = city_key
        if ucdb_id is not None and int(ucdb_id) in master_by_ucdb:
            master_row = master_by_ucdb[int(ucdb_id)]
            iso = iso or master_row.get("iso3_for_nat")
            v_key = int(ucdb_id)
        if iso is None:
            iso = city_iso.get(record.get("city") or city_key)
        if iso is None:
            continue
        nat_def = nat_def_by_iso.get(iso)
        v = compute_v_ratio(c_def, nat_def)
        v_dict[v_key] = v
    return v_dict


def compute_r_ph_city_for_all(
    cities: list[str],
    out_dir: Path,
    city_records: list[dict] | None = None,
    *,
    downscale_method: str = "analytic",
) -> dict:
    """主入口: active master 真算全量城市 → 抽 step_01 目标城市.

    V28: 真算路径, 不再查死 CSV. 全量结果存 parquet 供 paper-2 复用.
    2026-06-03: 接通议会 V_ratio (build_council_v_ratio), 不再全部 V=1.0 退化.
    2026-06-06: P5.2 full 模式按 city_records/ucdb_id 抽行, 不再按 city 名去重.
    """
    if downscale_method not in DOWNSCALE_METHODS:
        raise ValueError(f"Unknown downscaling method: {downscale_method}")
    # 1. active master 全量真算 + 议会 V_ratio 接入.
    master = master_io.load_master_data()
    consul_csv = out_dir / 'step_05_consul.csv'
    city_records = city_records or [extract_city_record(c) for c in cities]
    records_by_key = {r.get("city_key"): r for r in city_records if r.get("city_key")}
    v_dict = build_council_v_ratio(consul_csv, master, city_records)
    print(f"[step_07] 议会 V_ratio 接入 {len(v_dict)} 城 (from {consul_csv.name})")
    if v_dict:
        sample_v = list(v_dict.items())[:5]
        print(f"  sample V: {sample_v}")
    full = downscale_r_ph_for_master(master, council_v_ratio=v_dict, method=downscale_method)
    full_path = out_dir / "step_07_r_ph_city_full_6275.parquet"
    full.to_parquet(full_path)
    (out_dir / 'step_07_method.json').write_text(json.dumps({
        'method': downscale_method,
        'engine': '_code/giri/r_ph_downscale.py',
        'output': full_path.name,
        'rows': len(full),
    }, indent=2) + '\n', encoding='utf-8')
    n_valid = int(full["R_Ph_city"].notna().sum())
    print(f"[step_07] 全量真算 {len(full)} 城 ({n_valid} 有效): "
          f"R_Ph_city min={full['R_Ph_city'].min():.4f} "
          f"mean={full['R_Ph_city'].mean():.4f} max={full['R_Ph_city'].max():.4f}")
    print(f"  H_ratio mean={full['H_ratio'].mean():.3f}, E_ratio mean={full['E_ratio'].mean():.3f}")
    print(f"  → 全量存 {full_path.name}")

    # 2. 抽目标城市: b50 走 display→master 名; full mode 走 ucdb_id 行级主键.
    rows_out = []
    n_have = n_missing = 0

    def _pick(master_name, iso3):
        """按 (master 真名, iso3) 联合取行, 消歧撞名城 (London CAN vs GBR)."""
        m = full[full["city"] == master_name]
        if iso3 is not None and len(m) > 1:
            m = m[m["iso3_for_nat"] == iso3]
        return m.iloc[0] if len(m) else None

    def _pick_by_ucdb(ucdb_id):
        m = full[full["ucdb_id"] == int(ucdb_id)]
        return m.iloc[0] if len(m) else None

    def _missing_row(city, src, city_record=None):
        city_record = city_record or extract_city_record(city)
        return {
            "city": city,
            "display_city": city_record.get("city") or city,
            "ucdb_id": city_record.get("ucdb_id") or "",
            "iso3_for_nat": city_record.get("iso3_for_nat") or "",
            "r_ph_city": "", "r_ph_national": "",
            "h_ratio": "", "e_ratio": "", "v_ratio": "", "source": src,
        }

    for city in cities:
        city_record = records_by_key.get(city) or extract_city_record(city)
        ucdb_id = city_record.get("ucdb_id")
        if ucdb_id is not None:
            rec = _pick_by_ucdb(ucdb_id)
            if rec is not None:
                rows_out.append(_row_from_master(city, rec, "master_v28_ucdb_hev", city_record))
                n_have += 1
            else:
                rows_out.append(_missing_row(city, "missing_ucdb_id", city_record))
                n_missing += 1
            continue

        # 5/23 fallback: Macau/Male (proxy 也用 iso3 消歧)
        if city in MISSING_FALLBACK:
            fb = MISSING_FALLBACK[city]
            proxy = fb.get("proxy_city")
            rec = _pick(proxy, _NAME_TO_ISO3.get(proxy)) if proxy else None
            if rec is not None:
                rows_out.append(_row_from_master(city, rec, f"fallback_proxy_{proxy}", city_record))
            else:
                rows_out.append({**_missing_row(city, "fallback_hardcoded", city_record),
                                 "r_ph_city": fb.get("r_ph_city", "")})
            n_have += 1
            continue
        # display → active master 真名 (v28: Delhi→New Delhi 等; 缺城→None 已上面处理)
        master_name = load_master_name_for_benchmark(city)
        rec = _pick(master_name, _NAME_TO_ISO3.get(city)) if master_name else None
        if rec is not None:
            rows_out.append(_row_from_master(city, rec, "master_v27_6_hev", city_record))
            n_have += 1
        else:
            src = "missing_ucdb_6275" if city in MASTER_MISSING else "name_mismatch_master"
            rows_out.append(_missing_row(city, src, city_record))
            n_missing += 1

    out_csv = out_dir / "step_07_r_ph_city.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"✓ step_07 R_Ph_city: {n_have}/{len(cities)} 真值/fallback + {n_missing} 缺")
    return {r["city"]: r["r_ph_city"] for r in rows_out if r["r_ph_city"] != ""}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/v01")
    p.add_argument("--downscale-method", choices=DOWNSCALE_METHODS, default="analytic")
    args = p.parse_args()
    import yaml
    meta = yaml.safe_load((Path(args.out) / "step_01_meta.yml").read_text())
    compute_r_ph_city_for_all(meta["cities"], Path(args.out), meta.get("city_records"),
                             downscale_method=args.downscale_method)


if __name__ == "__main__":
    main()
