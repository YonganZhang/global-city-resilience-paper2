"""step_04_run_r2.py — 议会 R2: 看 R1 packet, 坚持己见 / 调整都给理由

V27.3.19 R2(CoT 群体版 / Multi-Agent Debate):
- 输入: R1 阶段 8 评委的全 tier + reasoning(匿名为 Model_A/B/.../H)
- 8 评委重新评估,可调整可坚持,都给理由
- 标准差从 R1 的 0.058 → R2 的 0.025 收敛(50 城实测)

输出: outputs/v<NN>/step_04_r2.csv 同 R1 格式
"""
from __future__ import annotations

import re
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "_code" / "llm"))

from prompt_builder import build_r2_prompt  # noqa: E402
from sensitive_cleanse import cleanse_if_china  # noqa: E402, city-aware: 只清洗中国相关 city reasoning
from evidence_quality import normalize_r2_evidence_strength  # noqa: E402

def _strip_md_wrap(raw: str) -> str:
    """去 LLM 输出的 markdown 代码块包裹 (```json...``` 或 ```...```)."""
    raw = raw.strip()
    if raw.startswith("```"):
        # 去 ```json 或 ``` 头
        raw = re.sub(r"^```(?:json|JSON)?\s*\n?", "", raw)
        # 去尾 ```
        raw = re.sub(r"\n?```\s*$", "", raw)
    return raw.strip()

sys.path.insert(0, str(REPO_ROOT / "_pipelines" / "01_baseline_gciri"))
from step_03_run_r1 import run_llm_call, _chunk, BATCH_SIZE, normalize_ratings_city_keys  # noqa: E402


def _coerce_rating_record(value) -> dict:
    """Return a city rating dict, tolerating the common one-item-list shape."""
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]
    return {}


def _sanitize_ratings_map(ratings) -> dict:
    """Drop parser sentinels and malformed city records before aggregation."""
    if not isinstance(ratings, dict):
        return {}
    clean = {}
    for city, value in ratings.items():
        if city == "_error":
            continue
        rec = _coerce_rating_record(value)
        if rec:
            clean[city] = rec
    return clean


def run_r2_for_indicator_batch(
    indicator: dict, cities_batch: list[str], models: list[dict],
    r1_for_batch: dict, dry_run: bool,
) -> dict:
    """R2 跑单个 indicator + 单 batch.

    r1_for_batch: {model: {city: {tier, reasoning}}} 仅含本 batch 城市
    """
    # 2026-06-01 用户拍板: city-aware cleanse — 只清洗中国相关城市 reasoning,
    # 其他国家城市 reasoning 保留真实描述 (恐袭/抗议/暴力等保持). 评分全城市真实.
    r1_cleaned = {
        m: {c: {**(r1_for_batch.get(m, {}).get(c, {}) or {}),
                "reasoning": cleanse_if_china(
                    (r1_for_batch.get(m, {}).get(c, {}) or {}).get("reasoning", ""), c)}
            for c in cities_batch}
        for m in r1_for_batch
    }
    system, user = build_r2_prompt(indicator, cities_batch, r1_cleaned)
    results = {}
    with ThreadPoolExecutor(max_workers=len(models)) as ex:
        future_to_model = {
            ex.submit(run_llm_call, m, system, user, dry_run, "r2_"): m  # trace_prefix r2_
            for m in models
        }
        for fut in as_completed(future_to_model):
            m_cfg = future_to_model[fut]
            mname = m_cfg["vendor"]
            # V27.3.20 (subagent 最终审 [中]): R2 也加 1 次 retry, 跟 step_03 R1 一致.
            ratings = None
            last_err = None
            for attempt in range(2):
                try:
                    raw = fut.result() if attempt == 0 else run_llm_call(m_cfg, system, user, dry_run)
                    stripped = _strip_md_wrap(raw)
                    if not stripped.startswith("{"):
                        last_err = f"non-JSON output (first 80c): {stripped[:80]!r}"
                        continue
                    parsed = json.loads(stripped)
                    ratings = parsed.get("ratings", {})
                    if ratings:
                        ratings = normalize_ratings_city_keys(ratings, cities_batch)
                        ratings = _sanitize_ratings_map(ratings)
                        break
                    last_err = f"JSON valid but ratings empty (attempt {attempt+1})"
                except Exception as e:
                    last_err = str(e)
            results[mname] = ratings if ratings else {"_error": last_err or "unknown_parse_fail"}
    return results


def run_step_r2(meta: dict, r1_data: dict, out_dir: Path, dry_run: bool = False) -> dict:
    indicators = meta["indicators"]
    cities = meta["cities"]
    models = [m for m in meta["models"] if m.get("model") == "议会成员"][:8]

    evidence_dir = out_dir / "step_04_r2_evidence"
    evidence_dir.mkdir(exist_ok=True, parents=True)

    # 5/23 用户拍板加并发: R2 跨 indicator + 跨 batch 并发 (max_workers=48, 5/23 测真 sweet spot).
    # 跟 step_05_consul 真同设计. 真省 ~10x (5 batch × 18 ind = 90 任务 → ThreadPool(32) → 真 3 波 ~10 min)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_r2 = {ind["alias_name"]: {m["vendor"]: {} for m in models} for ind in indicators}
    tasks = []  # [(ind, alias, batch_idx, batch_cities)]
    for ind in indicators:
        alias = ind["alias_name"]
        for batch_idx, batch in enumerate(_chunk(cities, BATCH_SIZE), 1):
            tasks.append((ind, alias, batch_idx, batch))
    print(f"  [R2 并发] {len(tasks)} 任务 ({len(indicators)} ind × ? batch), max_workers=48")

    def _one_r2_task(t):
        ind, alias, batch_idx, batch = t
        ev_file = evidence_dir / f"{alias}_batch_{batch_idx}.json"
        if ev_file.exists() and not dry_run:
            batch_res = json.loads(ev_file.read_text(encoding="utf-8"))
            return (alias, batch_res)
        r1_for_batch = {
            m: {c: r1_data.get(alias, {}).get(m, {}).get(c, {}) for c in batch}
            for m in (mm["vendor"] for mm in models)
        }
        batch_res = run_r2_for_indicator_batch(ind, batch, models, r1_for_batch, dry_run)
        ev_file.write_text(json.dumps(batch_res, ensure_ascii=False, indent=2), encoding="utf-8")
        return (alias, batch_res)

    n_done = 0
    with ThreadPoolExecutor(max_workers=48) as pool:
        futures = {pool.submit(_one_r2_task, t): t for t in tasks}
        for fut in as_completed(futures):
            alias, batch_res = fut.result()
            for mname, ratings in batch_res.items():
                all_r2.setdefault(alias, {}).setdefault(mname, {}).update(_sanitize_ratings_map(ratings))
            n_done += 1
            if n_done % 10 == 0:
                print(f"  [R2] 完成 {n_done}/{len(tasks)} 任务")

    # csv
    # V27.3.20 (用户拍板): LLM 自己填的 decision 字段经常**语义错** (gpt-5-nano R1=R2=B- 但
    # 自标 adjusted; glm-4-flash R1=B→R2=C- 实际改了但自标 kept). 代码强制按 R1 vs R2
    # 真 tier 比较, 覆盖 LLM 自报, 加 llm_self_decision 列保留原值审计.
    import csv
    rows = []
    for ind_alias, by_model in all_r2.items():
        for mname, by_city in by_model.items():
            for city in cities:
                r = _coerce_rating_record(by_city.get(city, {}))
                r2_tier = r.get("tier", "")
                r1_tier = r1_data.get(ind_alias, {}).get(mname, {}).get(city, {}).get("tier", "")
                llm_decision = r.get("decision", "")
                if r1_tier and r2_tier:
                    true_decision = "kept" if r1_tier == r2_tier else "adjusted"
                else:
                    true_decision = llm_decision  # R1 缺时 fallback LLM 自报
                raw_evidence_strength = r.get("evidence_strength", "")
                evidence_strength, evidence_quality_flags = normalize_r2_evidence_strength(
                    raw_evidence_strength,
                    r.get("reasoning", ""),
                    r1_data.get(ind_alias, {}).get(mname, {}).get(city, {}).get("evidence_strength", ""),
                )
                rows.append({
                    "indicator": ind_alias,
                    "city": city,
                    "model": mname,
                    "tier": r2_tier,
                    "decision": true_decision,                # V27.3.20: 代码按 R1 vs R2 真算
                    "llm_self_decision": llm_decision,        # V27.3.20: LLM 自报 (审计 LLM 语义识别准确度)
                    "r1_tier": r1_tier,                       # V27.3.20: R1 tier 留档对照
                    "reasoning": r.get("reasoning", ""),
                    "evidence_strength": evidence_strength,
                    "raw_evidence_strength": raw_evidence_strength,
                    "evidence_quality_flags": evidence_quality_flags,
                })
    with (out_dir / "step_04_r2.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "indicator", "city", "model", "tier", "decision",
            "llm_self_decision", "r1_tier", "reasoning", "evidence_strength",
            "raw_evidence_strength", "evidence_quality_flags",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"✓ step_04 R2: {len(indicators)} ind × {len(cities)} 城")
    return all_r2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/v01")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    import yaml
    meta = yaml.safe_load((Path(args.out) / "step_01_meta.yml").read_text())
    # load R1 from CSV
    import csv
    r1 = {}
    with (Path(args.out) / "step_03_r1.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r1.setdefault(row["indicator"], {}).setdefault(row["model"], {})[row["city"]] = {
                "tier": row["tier"],
                "reasoning": row["reasoning"],
                "evidence_strength": row.get("evidence_strength", ""),
            }
    run_step_r2(meta, r1, Path(args.out), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
