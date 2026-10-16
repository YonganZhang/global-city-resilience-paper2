"""step_01_load_metadata.py — 议会跑前加载元数据

读取:
- Master Excel Sheet 6 → 18 indicator 元数据(city_def + anchor 档位匹配)
- Sheet 7 → 8 评委 + Consul 模型清单
- 50 城议会清单(BENCHMARK_50_NAMES, 含 Melbourne)

输出: outputs/v<NN>/step_01_meta.yml
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
import random
import csv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "_code" / "llm"))

from prompt_builder import load_master_excel  # noqa: E402

# 真 50 城议会清单 (从 代码/pipeline_v25/src/run_50_open_parliament.py:BENCHMARK_50)
# ⚠️ 不是 49 城!Melbourne 加在 #34(2026-04-24 per user request)
# 历史踩坑:step07_ablation.py 的 BENCHMARK_50 只有 49 城是 active bug
BENCHMARK_50 = [
    ("Hong Kong", "CHN"), ("Shenzhen", "CHN"), ("Guangzhou", "CHN"), ("Macau", "CHN"),
    ("Beijing", "CHN"), ("Shanghai", "CHN"), ("Chongqing", "CHN"),
    ("Tokyo", "JPN"), ("Osaka", "JPN"), ("Seoul", "KOR"),
    ("Singapore", "SGP"), ("Jakarta", "IDN"), ("Bangkok", "THA"), ("Manila", "PHL"),
    ("Kuala Lumpur", "MYS"), ("Ho Chi Minh City", "VNM"),
    ("Mumbai", "IND"), ("Delhi", "IND"), ("Kolkata", "IND"), ("Dhaka", "BGD"), ("Karachi", "PAK"),
    ("London", "GBR"), ("Paris", "FRA"), ("Berlin", "DEU"), ("Madrid", "ESP"),
    ("Rome", "ITA"), ("Amsterdam", "NLD"), ("Stockholm", "SWE"),
    ("New York", "USA"), ("Los Angeles", "USA"), ("Chicago", "USA"), ("Toronto", "CAN"),
    ("Sydney", "AUS"), ("Melbourne", "AUS"), ("Auckland", "NZL"),
    ("Istanbul", "TUR"), ("Dubai", "ARE"), ("Tehran", "IRN"),
    ("Cairo", "EGY"), ("Lagos", "NGA"), ("Nairobi", "KEN"), ("Johannesburg", "ZAF"),
    ("Kinshasa", "COD"),
    ("Mexico City", "MEX"), ("Sao Paulo", "BRA"), ("Buenos Aires", "ARG"), ("Lima", "PER"),
    ("Male", "MDV"), ("Kabul", "AFG"), ("Niamey", "NER"),
]
assert len(BENCHMARK_50) == 50, f"BENCHMARK_50 必须 50 城, 实际 {len(BENCHMARK_50)}"
BENCHMARK_50_NAMES = [c[0] for c in BENCHMARK_50]
BENCHMARK_50_ISO = dict(BENCHMARK_50)

CITY_KEY_SUFFIX_RE = re.compile(r"^(?P<city>.+) \[(?P<iso3>[A-Z]{3}); UCDB (?P<ucdb_id>\d+)\]$")

# BENCHMARK_50 display 名 → master.city 真名. 命名**随 active master 表不同**, 故按表键控:
#   v5_8     : 老清单, Osaka [Kyoto] 式后缀; Macau 缺 / Shenzhen 在.
#   v28_ucdb : UCDB GC_UCN_MAI 主名 (Osaka / New Delhi / Manila / New York City);
#              Macau→Zhuhai [Macau] 补上; Shenzhen 后续以专属补入行 ucdb_id=999991 保留.
# 议会 / Tavily / prompt 用 display 名; step_07 R_Ph 下推 / fig 查 master 需用真名.
_BENCHMARK_NAME_BY_MASTER = {
    "v5_8": {
        "name_map": {
            "Osaka":     "Osaka [Kyoto]",
            "Delhi":     "Delhi [New Delhi]",
            "Manila":    "Quezon City [Manila]",
            "Sao Paulo": "São Paulo",
        },
        "missing": {"Male", "Macau"},
    },
    "v28_ucdb": {
        "name_map": {
            "Osaka":     "Osaka",            # 显式: v28 主名 (v5_8 是 Osaka [Kyoto])
            "Manila":    "Manila",           # 显式: v28 主名 (v5_8 是 Quezon City [Manila])
            "Delhi":     "New Delhi",        # v28 主名 (v5_8 是 Delhi [New Delhi])
            "New York":  "New York City",    # v28 主名 (v5_8 是 New York)
            "Sao Paulo": "São Paulo",
            "Macau":     "Zhuhai [Macau]",   # v28 补上 (v5_8 缺)
        },
        "missing": {"Male"},     # 2026-05-31: Shenzhen 已用专属 polygon 真算加入 master (用户拍板); Male 太小仍缺
    },
}


def _active_master_key() -> str:
    """当前 active master 的 key (跟随 master_io / env GCIRI_MASTER). 缺省 v28_ucdb."""
    sys.path.insert(0, str(REPO_ROOT / "_code" / "giri"))
    import master_io  # noqa: E402
    return getattr(master_io, "_ACTIVE_MASTER_KEY", "v28_ucdb")


def _benchmark_name_cfg() -> dict:
    key = _active_master_key()
    return _BENCHMARK_NAME_BY_MASTER.get(key, _BENCHMARK_NAME_BY_MASTER["v28_ucdb"])


# 向后兼容别名 (step_07 等 import MASTER_MISSING): 按当前 active master 解析.
MASTER_NAME_MAP = _benchmark_name_cfg()["name_map"]
MASTER_MISSING = _benchmark_name_cfg()["missing"]


def load_master_name_for_benchmark(city: str) -> str | None:
    """BENCHMARK_50 display 名 → active master.city 真名.

    Returns None if city 在当前 master 缺 (v5_8: Male/Macau; v28: Male).
    Otherwise returns master 真名 (按表后缀映射 / 其余原名).
    V27.3.20 L2 lint 改名: benchmark_to_master_name → load_master_name_for_benchmark.
    """
    cfg = _benchmark_name_cfg()
    if city in cfg["missing"]:
        return None
    return cfg["name_map"].get(city, city)


def build_city_key(city: str, iso3: str | None = None, ucdb_id: int | str | None = None) -> str:
    """Build the internal row-level city key used by full-master council runs."""
    if ucdb_id in (None, ""):
        return city
    return f"{city} [{iso3 or 'UNK'}; UCDB {int(ucdb_id)}]"


def extract_city_record(city_key: str) -> dict:
    """Parse a city key into display metadata.

    Raw benchmark/custom names remain valid keys and parse to ucdb_id=None.
    """
    m = CITY_KEY_SUFFIX_RE.match(city_key)
    if not m:
        return {"city_key": city_key, "city": city_key, "iso3_for_nat": None, "ucdb_id": None}
    return {
        "city_key": city_key,
        "city": m.group("city"),
        "iso3_for_nat": m.group("iso3"),
        "ucdb_id": int(m.group("ucdb_id")),
    }


def _record_from_name(city: str, iso3: str | None = None) -> dict:
    return {
        "city_key": city,
        "city": city,
        "iso3_for_nat": iso3,
        "ucdb_id": None,
    }


def _records_from_names(cities: list[str]) -> list[dict]:
    return [_record_from_name(c, BENCHMARK_50_ISO.get(c)) for c in cities]


def load_step_metadata(
    out_dir: Path,
    cities: list | None = None,
    city_records: list[dict] | None = None,
    limit_indicators: int | None = None,
) -> dict:
    """主入口: 读 Master Excel + 写 step_01_meta.yml."""
    import yaml
    excel_data = load_master_excel()
    indicators_full = excel_data["indicators"]
    indicators_used = indicators_full[:limit_indicators] if limit_indicators else indicators_full
    if city_records is None:
        city_records = _records_from_names(cities if cities else BENCHMARK_50_NAMES)
    cities = [r["city_key"] for r in city_records]
    if len(cities) != len(set(cities)):
        raise RuntimeError("step_01 city_key 不唯一: full 模式必须按 UCDB 行级主键生成城市清单")
    meta = {
        "schema": "v1",
        "indicators": indicators_used,
        "models": excel_data["models"],
        "cities": cities,
        "city_records": city_records,
        "city_key_schema": "raw city for b50/custom; '<city> [<ISO3>; UCDB <id>]' for full master rows",
        "n_cities": len(cities),
        "n_unique_city_names": len({r["city"] for r in city_records}),
        "indicators_limit": limit_indicators,
        "n_indicators": len(indicators_used),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "step_01_meta.yml").write_text(
        yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"✓ step_01 load metadata: {meta['n_indicators']} ind × {meta['n_cities']} 城")
    return meta


def build_city_records_for_mode(mode, city, cities_arg):
    """三模式城市清单解析 (全议会, 三模式同代码同 anchor_tier_map, 仅城市数量不同).

    用户 2026-05-28 拍板: 全量跑议会, 无 hybrid / 无国家级 extrapolation.
      single   : --city <名> 单城
      b50      : BENCHMARK_50 (50 城默认, 复现 paper-1)
      full6275 : 历史参数名; 实际读取 active master 全量城市名
    自定义 --cities 优先于 --mode.
    """
    if cities_arg:
        return _records_from_names(cities_arg)
    if mode == "single":
        if not city:
            raise SystemExit("🔴 --mode single 需配 --city <城市名>")
        return _records_from_names([city])
    if mode == "full6275":
        # 走 master_io active master (env GCIRI_MASTER 可切表), 不再硬编码 v5_8 路径.
        # v28_ucdb 5862 行 / v5_8 6275 行 — full6275 名义沿用, 实际数随 active master.
        sys.path.insert(0, str(REPO_ROOT / "_code" / "giri"))
        import master_io  # noqa: E402
        master = master_io.load_master_data()
        required = {"city", "ucdb_id", "iso3_for_nat"}
        missing = required - set(master.columns)
        if missing:
            raise RuntimeError(f"active master 缺少 full-mode 主键列: {sorted(missing)}")
        m = master.dropna(subset=["city", "ucdb_id"]).copy()
        m = m.sort_values(["city", "iso3_for_nat", "ucdb_id"], kind="stable")
        records = []
        for row in m.to_dict("records"):
            city_name = str(row["city"])
            iso3 = row.get("iso3_for_nat")
            ucdb_id = int(row["ucdb_id"])
            rec = {
                "city_key": build_city_key(city_name, iso3, ucdb_id),
                "city": city_name,
                "iso3_for_nat": iso3,
                "ucdb_id": ucdb_id,
            }
            for optional in ("country_name", "lat", "lon", "pop_2025"):
                if optional in row:
                    val = row.get(optional)
                    if val == val:
                        rec[optional] = val.item() if hasattr(val, "item") else val
            records.append(rec)
        keys = [r["city_key"] for r in records]
        if len(keys) != len(set(keys)):
            raise RuntimeError("full6275 city_key 仍不唯一, 请检查 ucdb_id/city/iso3")
        print(f"[mode=full6275] active master {master_io.MASTER_DATA_LATEST.name} 读 {len(records)} 行 "
              f"({len({r['city'] for r in records})} 个唯一城市名), 按 UCDB 行级主键运行")
        return records
    return _records_from_names(BENCHMARK_50_NAMES)  # 默认 b50


def sample_city_records(records: list[dict], sample_size: int | None, seed: int) -> list[dict]:
    """Deterministically sample row-level city records while preserving city_records.

    This is for paid pilot runs: it keeps the full-mode city_key/iso3/ucdb_id/lat/lon
    metadata instead of falling back to raw city names.
    """
    if sample_size is None:
        return records
    if sample_size <= 0:
        raise SystemExit("🔴 --sample-size 必须是正整数")
    if sample_size > len(records):
        raise SystemExit(f"🔴 --sample-size={sample_size} 超过可用城市数 {len(records)}")
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(records)), sample_size))
    return [records[i] for i in idx]


def load_city_records_csv(path: Path) -> list[dict]:
    """Load explicit row-level city records for forced audit panels.

    Required columns: city_key, city, iso3_for_nat, ucdb_id.
    Optional columns such as country_name, lat, lon, pop_2025 are preserved.
    """
    records = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            city_key = (row.get("city_key") or "").strip()
            city = (row.get("city") or "").strip()
            iso3 = (row.get("iso3_for_nat") or row.get("iso3") or "").strip()
            ucdb_raw = (row.get("ucdb_id") or "").strip()
            if not city_key or not city or not iso3 or not ucdb_raw:
                raise SystemExit(f"🔴 city-records CSV 缺必填字段: {path} row={row}")
            rec = {
                "city_key": city_key,
                "city": city,
                "iso3_for_nat": iso3,
                "ucdb_id": int(float(ucdb_raw)),
            }
            for optional in ("country_name", "lat", "lon", "pop_2025", "audit_group", "audit_expected"):
                if optional in row and str(row.get(optional, "")).strip():
                    val = row[optional]
                    if optional in ("lat", "lon", "pop_2025"):
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    rec[optional] = val
            records.append(rec)
    keys = [r["city_key"] for r in records]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"🔴 city-records CSV city_key 不唯一: {path}")
    return records


def build_cities_for_mode(mode, city, cities_arg):
    """Backward-compatible wrapper returning city keys only."""
    return [r["city_key"] for r in build_city_records_for_mode(mode, city, cities_arg)]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/v01")
    p.add_argument("--mode", choices=["single", "b50", "full6275"], default="b50",
                   help="三模式 (全议会同代码): single(配--city) / b50(50城默认) / full6275(master全量)")
    p.add_argument("--city", default=None, help="--mode single 时指定城市名")
    p.add_argument("--cities", default=None, help="逗号分隔自定义城市清单 (优先于 --mode)")
    p.add_argument("--city-records-csv", default=None,
                   help="显式 city_records CSV; 优先于 --cities/--mode, 保留 ISO3/UCDB/lat/lon")
    p.add_argument("--limit-indicators", type=int, default=None, help="只跑前 N 个 indicator (节省 token)")
    p.add_argument("--sample-size", type=int, default=None,
                   help="从解析后的城市记录中确定性抽样 N 行; full6275 pilot 用, 保留 city_records")
    p.add_argument("--sample-seed", type=int, default=20260630,
                   help="--sample-size 的随机种子")
    args = p.parse_args()
    cities_arg = [c.strip() for c in args.cities.split(",") if c.strip()] if args.cities else None
    if args.city_records_csv:
        city_records = load_city_records_csv(Path(args.city_records_csv))
    else:
        city_records = build_city_records_for_mode(args.mode, args.city, cities_arg)
    city_records = sample_city_records(city_records, args.sample_size, args.sample_seed)
    meta = load_step_metadata(Path(args.out), city_records=city_records, limit_indicators=args.limit_indicators)
    if args.sample_size is not None:
        meta["sample_size"] = args.sample_size
        meta["sample_seed"] = args.sample_seed
        meta["sample_source_mode"] = args.mode
        import yaml
        (Path(args.out) / "step_01_meta.yml").write_text(
            yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        print(f"✓ sample: {args.sample_size} rows from mode={args.mode}, seed={args.sample_seed}")


if __name__ == "__main__":
    main()
