"""V27.6 Master I/O helpers — 统一 read/write master_data, master_council, master_prompts.

设计原则:
- master_data.parquet: 1 行 1 城, wide 格式, 含 Phase 0/B/C 全数据
- master_council.parquet: long format, 1 行 1 evaluation (city + indicator + phase + model)
- master_prompts.parquet: 1 行 1 prompt 版本

所有 pipeline step 通过此 module read/write master, 不再直接读 csv / hardcoded path.

═══════════════════════════════════════════════════════════════════════════
极性方向 (硬规则, 不靠记忆): 详见 _top/findings/2026-05-18_polarity_verification.md
─────────────────────────────────────────────────────────────────────────
  cap_abs  6 indicator  → deficit  (raw 高=差) → res = 1 - raw
  cap_resp 6 indicator  → deficit  (raw 高=差) → res = 1 - raw
  cap_rec  6 indicator  → capability (raw 高=好) → res = raw (不翻)

  master 列后缀: __DEF_nat (12 个 deficit) / __CAP_nat (6 个 capability + 3 idx)
═══════════════════════════════════════════════════════════════════════════

命名陷阱 (cap_rest vs cap_rec): _wiki/_entities/_cap_rest.md 命名 "Capacity to Restore",
但 indicator 前缀用 cap_rec ("Recovery"). 历史命名漂移, _code/giri/provenance.py 已注释:
  cap_rest_idx_nat = mean(6 ind__cap_rec__*_nat)  # rest 是 capability 聚合, rec 是 indicator 前缀
grep "cap_rest" 找聚合层, grep "cap_rec" 找 indicator 层.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import pandas as pd

# Master 表路径 (相对项目根, 跨机/跨 worktree 健壮)
# 5/27 修: 旧版硬编码已删的 worktree 路径 (.claude/worktrees/six-in-one-arch/) → import 即崩.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _REPO_ROOT / "_outputs" / "master_v27_6"

MASTER_COUNCIL = _OUT_DIR / "master_council.parquet"  # 议会跑后填
MASTER_COUNCIL_JSONL = _OUT_DIR / "master_council.jsonl"  # append-only, 并发安全
MASTER_PROMPTS = _OUT_DIR / "master_prompts.parquet"  # prompt archive

# 5/29 修 glob 选错表 bug: 旧 _resolve_master_data_latest() glob 'master_v27_6_v*.parquet'
#   按字典序取 last → 选中 v6_circle (CANDIDATE, 从不该 active); 且 v28_ucdb 前缀 master_v28_
#   根本不匹配 glob. "命名约定隐式选表" 在命名换代时静默失效. 改显式注册表 + 单一 active 指针.
# 切表只改 _ACTIVE_MASTER_KEY 一处 (或 env GCIRI_MASTER 临时覆盖, 不改码即可复现 v5_8 paper-1).
_MASTER_FILES = {
    "v28_ucdb":  _OUT_DIR / "master_v28_ucdb.parquet",       # UCDB-native 5862 行 (5861 UCDB 城市中心 + 深圳专属行)
    "v5_8":      _OUT_DIR / "master_v27_6_v5_8.parquet",     # legacy 6275 城 (paper-1 复现 / 对照)
    "v6_circle": _OUT_DIR / "master_v27_6_v6_circle.parquet",  # circle CANDIDATE (历史方案)
}
# 默认 v28 (用户 5/29 "把下游评分匹配到新表"). registry 正式 ACTIVE 升级仍待 FL 口径/海外领地拍板.
_ACTIVE_MASTER_KEY = os.environ.get("GCIRI_MASTER", "v28_ucdb")


def _resolve_active_master() -> Path:
    """显式取当前 active master (env GCIRI_MASTER 可覆盖 key)."""
    if _ACTIVE_MASTER_KEY not in _MASTER_FILES:
        raise ValueError(
            f"GCIRI_MASTER={_ACTIVE_MASTER_KEY!r} 未知; 可选: {list(_MASTER_FILES)}"
        )
    path = _MASTER_FILES[_ACTIVE_MASTER_KEY]
    if not path.exists():
        raise FileNotFoundError(f"active master ({_ACTIVE_MASTER_KEY}) 不存在: {path}")
    return path


MASTER_DATA_LATEST = _resolve_active_master()


def load_master_data() -> pd.DataFrame:
    """读 master_data 全城 × 全列 (Phase 0 + B + C 数据).

    Returns:
        DataFrame N 城 × M 列 (随 active master 变: v28_ucdb 5862×52 / v5_8 6275×142).
        列含 city / iso3_for_nat / lat / lon / R_Ph__RAW_nat / 18 国家级 anchor /
        8 haz_*__HAZ + nat_avg / exp_*__RAW. (H/E/V_ratio / R_Ph_city / cap_* / GCIRI 议会后写回.)
    """
    return pd.read_parquet(MASTER_DATA_LATEST)


def load_city_subset(cities: list[str], iso3s: Optional[list[str]] = None) -> pd.DataFrame:
    """读特定 city 子集 (BENCHMARK_50 / 全量 / 5 城).

    Args:
        cities: city name 列表
        iso3s: 可选 iso3 联合键 (避免 London CAN vs GBR 混)
    """
    df = load_master_data()
    if iso3s and len(iso3s) == len(cities):
        # (city, iso3) 联合 key
        pairs = list(zip(cities, iso3s))
        mask = df.apply(lambda r: (r["city"], r["iso3_for_nat"]) in pairs, axis=1)
        return df[mask].copy()
    return df[df["city"].isin(cities)].copy()


def export_for_review(fmt: str = "xlsx", n_rows: Optional[int] = None,
                       output_path: Optional[Path] = None) -> Path:
    """V27.6 给人类 review 的派生视图 — 真源是 parquet, 这里只是按需转格式.

    设计哲学 (share-arch §2 单一真源 + 派生视图):
      parquet = 单一真源 (机器永远读这个, 改不动)
      xlsx/csv/tsv/html = 派生视图 (按需生成给人类看)

    Args:
        fmt: "xlsx" (推荐, Excel 原生, 中文+引号完美) /
             "tsv"  (tab 分隔无逗号歧义, 适合命令行 grep) /
             "csv"  (全字段引号包裹, 兼容老工具) /
             "html" (浏览器看, 可读性最好但文件大)
        n_rows: 抽样多少行 (None = 全 6271, 50 = 只 50 行) — 用户想速看时用
        output_path: 自定义输出位置 (None = 跟 latest parquet 同目录同名换后缀)

    Returns:
        生成文件的 Path
    """
    src = _resolve_active_master()
    df = pd.read_parquet(src)
    if n_rows is not None:
        df = df.head(n_rows)

    if output_path is None:
        suffix = "_review" if n_rows is None else f"_review_top{n_rows}"
        ext_map = {"xlsx": ".xlsx", "tsv": ".tsv", "csv": ".csv", "html": ".html"}
        output_path = src.with_name(src.stem + suffix + ext_map[fmt])

    if fmt == "xlsx":
        df.to_excel(output_path, index=False, engine="openpyxl")
    elif fmt == "tsv":
        df.to_csv(output_path, sep="\t", index=False)
    elif fmt == "csv":
        import csv as _csv
        df.to_csv(output_path, index=False, quoting=_csv.QUOTE_ALL)  # 全字段引号防错乱
    elif fmt == "html":
        df.to_html(output_path, index=False, table_id="master")
    else:
        raise ValueError(f"Unknown fmt: {fmt}. Use xlsx/tsv/csv/html.")

    return output_path


def get_country_anchor(iso3: str, indicator_real_name: str) -> Optional[float]:
    """读某国某 indicator 的 res anchor (V27.6 prompt inject 用).

    Args:
        iso3: HKG / GBR / JPN / CHN ...
        indicator_real_name: ind__cap_abs__GINI_Index / ind__cap_rec__HDI ...

    Returns:
        res value [0,1] 或 None (国家或 indicator 没数据)
    """
    df = load_master_data()
    # V27.6 master 列名加了 __DEF_nat / __CAP_nat 极性后缀
    # cap_rec 是 capability form → __CAP_nat, 其他 (abs/resp) deficit → __DEF_nat
    suffix = "__CAP_nat" if "__cap_rec__" in indicator_real_name else "__DEF_nat"
    nat_col = f"{indicator_real_name}{suffix}"
    if nat_col not in df.columns:
        # fallback 老列名 (兼容 v5_2 前)
        nat_col_old = f"{indicator_real_name}_nat"
        if nat_col_old in df.columns:
            nat_col = nat_col_old
        else:
            return None
    row = df[df["iso3_for_nat"] == iso3]
    if row.empty:
        return None
    raw_v = row[nat_col].iloc[0]
    if pd.isna(raw_v):
        return None
    # V27.6 修复 (2026-05-18 用户拍板): raw=0.0 不是 fillna 假零, 是 GIRI 真零 (治理顶尖)!
    # USA Building=0.167 / JPN=0.099 / HKG=0 / CHN=0 — HKG/CHN 真零意味"deficit 完美无缺陷"
    # 之前 mask 把真零屏蔽让议员看不到 anchor, HK 韧性顶尖被议员错判 B 档
    # 现在 raw=0 → res=1.0 (S+ 档), 议员看到"HK Building = S+ anchor"
    # ABS/RESP raw 是 deficit form (高=差), 翻 res = 1 - raw
    # REC raw 是 capability form (高=好), 不翻
    if "__cap_rec__" in indicator_real_name:
        return float(raw_v)
    return float(1 - raw_v)


def write_council_row(city: str, iso3: str, indicator_real: str, indicator_alias: str,
                       phase: str, model: str, tier: str, reasoning: str,
                       evidence_strength: str = "", decision: str = "",
                       prompt_id: str = "", run_id: str = "",
                       timestamp: str = "") -> dict:
    """构造 1 行 master_council long entry.

    Args:
        phase: "R1" / "R2" / "Consul"
        model: vendor name (deepseek-v4-flash / gpt-5-nano / ...)
        tier: S+ ~ BOT
        evidence_strength: strong / medium / weak (只 R1)
        decision: kept / adjusted (只 R2)
    """
    return {
        "city": city, "iso3": iso3,
        "indicator_real": indicator_real, "indicator_alias": indicator_alias,
        "phase": phase, "model": model,
        "tier": tier, "reasoning": reasoning,
        "evidence_strength": evidence_strength, "decision": decision,
        "prompt_id": prompt_id, "run_id": run_id, "timestamp": timestamp,
    }


def append_council_jsonl(rows: list[dict]) -> int:
    """V27.6 并发安全: jsonl append + fcntl 锁.

    议会 8 评委 ThreadPoolExecutor 并发调时, parquet read-modify-write 会丢数据.
    jsonl 是 append-only, 每行原子写, 配 fcntl flock 保证多 process 不混行.

    Returns:
        实际 append 的行数.
    """
    import fcntl, json
    if not rows:
        return 0
    MASTER_COUNCIL_JSONL.parent.mkdir(parents=True, exist_ok=True)
    lock_path = MASTER_COUNCIL_JSONL.with_suffix(".jsonl.lock")
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            with open(MASTER_COUNCIL_JSONL, "a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    return len(rows)


# V27.3.21 真 DEPRECATED (codex+gemini 双 LLM 真核共识): parquet append 并发不安全,
# 真**用 append_council_jsonl** 替代 (jsonl 真安全). 留着只为 backward compat, V28+ 真删.
def append_council(rows: list[dict]) -> None:
    """🟡 deprecated: 用 append_council_jsonl 替代 (并发不安全).

    保留作向后兼容. parquet read-modify-write 时序不安全, 议会并发调时会丢数据.
    """
    import warnings
    warnings.warn("append_council (parquet) is concurrency-unsafe; use append_council_jsonl",
                  DeprecationWarning, stacklevel=2)
    df_new = pd.DataFrame(rows)
    if MASTER_COUNCIL.exists():
        df_old = pd.read_parquet(MASTER_COUNCIL)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_parquet(MASTER_COUNCIL, index=False)


def load_council_jsonl() -> pd.DataFrame:
    """读 master_council.jsonl 回 DataFrame (论文 reproduce / audit 用)."""
    import json
    if not MASTER_COUNCIL_JSONL.exists():
        return pd.DataFrame()
    rows = []
    with open(MASTER_COUNCIL_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def write_data_columns(updates: dict[str, dict[str, float]]) -> None:
    """更新 master_data 多列 (议会后处理后写回 cap_* / GCIRI / V_ratio / R_Ph_city 等).

    Args:
        updates: {city_name: {col1: val1, col2: val2, ...}}
                例 {"Hong Kong": {"F_city": 0.4, "cap_abs": 70.0, "GCIRI": 65.29}}
    """
    df = load_master_data()
    for city, cols in updates.items():
        mask = df["city"] == city
        for col, val in cols.items():
            if col not in df.columns:
                df[col] = pd.NA
            df.loc[mask, col] = val
    df.to_parquet(MASTER_DATA_LATEST, index=False)
    df.to_csv(MASTER_DATA_LATEST.with_suffix(".csv"), index=False)


def register_prompt(prompt_id: str, phase: str, indicator: str,
                     template_text: str, version_hash: str, run_id: str) -> None:
    """注册 prompt 模板版本到 master_prompts."""
    new_row = pd.DataFrame([{
        "prompt_id": prompt_id, "phase": phase, "indicator": indicator,
        "template_text": template_text, "version_hash": version_hash, "run_id": run_id,
        "timestamp": pd.Timestamp.now().isoformat(),
    }])
    if MASTER_PROMPTS.exists():
        df_old = pd.read_parquet(MASTER_PROMPTS)
        # 同 hash 不重复
        if version_hash in df_old["version_hash"].values:
            return
        df_all = pd.concat([df_old, new_row], ignore_index=True)
    else:
        df_all = new_row
    df_all.to_parquet(MASTER_PROMPTS, index=False)


if __name__ == "__main__":
    # Self-test
    df = load_master_data()
    print(f"✓ master_data: {df.shape}")
    print(f"  HK anchor (Building): {get_country_anchor('HKG', 'ind__cap_abs__Building_quality_control_index')}")
    print(f"  GBR anchor (HDI): {get_country_anchor('GBR', 'ind__cap_rec__Human_Development_Index')}")
    print(f"  CHN anchor (GINI): {get_country_anchor('CHN', 'ind__cap_abs__GINI_Index')}")
