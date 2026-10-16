"""step_05_consul.py — Consul (gemini-2.5-pro) R2 之后终审

V27.3.19 关键: Consul 在 R2 之后(不是 R1.5 audit).
Consul 看 R1 + R2 全部 8 评委, 给最终 tier + 聚合 reasoning.

输出: outputs/v<NN>/step_05_consul.csv  50 城 × 18 ind × 最终 tier(单一真值)
"""
from __future__ import annotations

import re
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "_code" / "llm"))
sys.path.insert(0, str(REPO_ROOT / "_pipelines" / "01_baseline_gciri"))

from prompt_builder import build_consul_prompt  # noqa: E402
from evidence_quality import (
    detect_evidence_quality_flags,
    normalize_r2_evidence_strength,
    render_flags,
)  # noqa: E402

def _strip_md_wrap(raw: str) -> str:
    """去 LLM 输出的 markdown 代码块包裹 (```json...``` 或 ```...```)."""
    raw = raw.strip()
    if raw.startswith("```"):
        # 去 ```json 或 ``` 头
        raw = re.sub(r"^```(?:json|JSON)?\s*\n?", "", raw)
        # 去尾 ```
        raw = re.sub(r"\n?```\s*$", "", raw)
    return raw.strip()

from step_03_run_r1 import run_llm_call, _chunk, BATCH_SIZE, normalize_ratings_city_keys  # noqa: E402


def _tier_to_mid_score(tier: str) -> float:
    """20 档 → mid-point 分数 (S+=97.5 / S=92.5 / ... / BOT=2.5)."""
    sys.path.insert(0, str(REPO_ROOT / "_code" / "llm"))
    from prompt_builder import TIER_DEFS  # noqa: E402
    for t, mid, _pct in TIER_DEFS:
        if t == tier:
            return mid
    return float("nan")


def _score_to_tier(score: float) -> str:
    """mid-point 分数 → 最近 tier."""
    sys.path.insert(0, str(REPO_ROOT / "_code" / "llm"))
    from prompt_builder import TIER_DEFS  # noqa: E402
    best = min(TIER_DEFS, key=lambda t: abs(t[1] - score))
    return best[0]


def _apply_consul_decisions(r2_for_batch: dict, decisions: dict, cities_batch: list) -> dict:
    """🔴 V28 schema (2026-05-18 用户拍板): drop 数组 + reasoning 2 字段.

    Consul 输出 {city: {drop: [1-8 编号数组], reasoning: 30-50 字}}
    本函数: 用 drop 数组 index 删 Model_X, 在 kept 评委 R2 tier 算 mid-score mean → 找最近 tier.

    评委编号: 1-8 (字典序: Model_A=1, B=2, ..., H=8)
    drop = [] keep_all; [5] 单删; [1,5] 双删; ≥3 个 = 议会失败应 keep_all
    """
    # 8 评委字典序编号 (1-based)
    model_names = sorted(r2_for_batch.keys())  # 字典序
    idx_to_model = {i + 1: name for i, name in enumerate(model_names)}

    final = {}
    for city in cities_batch:
        d = decisions.get(city, {})
        drop_idxs = d.get("drop", [])
        # 防呆: drop 长度 ≥3 → keep_all (议会失败信号)
        if isinstance(drop_idxs, list) and len(drop_idxs) >= 3:
            drop_idxs = []
        drop_models = {idx_to_model.get(i) for i in (drop_idxs or []) if i in idx_to_model}
        kept_models = [m for m in model_names if m not in drop_models]

        # V27.3.20 P3 用户拍板: evidence_strength 加权平均 (strong=1.5 / medium=1.0 / weak=0.5)
        # strong reasoning 评委权重高, weak 评委权重低. 反映"看证据不看票数"的议会精神.
        EVIDENCE_WEIGHT = {"strong": 1.5, "medium": 1.0, "weak": 0.5}
        kept_tiers = []           # 兼容老 (无加权 fallback)
        kept_weighted = []        # [(mid_score, weight)] 加权用
        kept_flag_strs = []
        for m in kept_models:
            r = r2_for_batch.get(m, {}).get(city, {})
            tier = r.get("tier", "")
            if tier:
                mid = _tier_to_mid_score(tier)
                if mid == mid:
                    kept_tiers.append(mid)
                    es, es_flags = normalize_r2_evidence_strength(
                        r.get("evidence_strength", ""),
                        r.get("reasoning", ""),
                        r.get("r1_evidence_strength", ""),
                    )
                    w = EVIDENCE_WEIGHT.get(es, 1.0)
                    kept_weighted.append((mid, w))
                    kept_flag_strs.append(es_flags)
        if not kept_tiers:
            # fallback: 全部 R2 评委均值 (无加权)
            for m in r2_for_batch:
                rr = r2_for_batch.get(m, {}).get(city, {})
                tier = rr.get("tier", "")
                if tier:
                    mid = _tier_to_mid_score(tier)
                    if mid == mid:
                        kept_tiers.append(mid)
                        es, es_flags = normalize_r2_evidence_strength(
                            rr.get("evidence_strength", ""),
                            rr.get("reasoning", ""),
                            rr.get("r1_evidence_strength", ""),
                        )
                        kept_weighted.append((mid, EVIDENCE_WEIGHT.get(es, 1.0)))
                        kept_flag_strs.append(es_flags)
        # 加权平均: sum(mid × w) / sum(w); 等价于无加权当所有 w 一致 (老 evidence_strength 缺失时 fallback)
        if kept_weighted:
            sum_w = sum(w for _, w in kept_weighted)
            mean_score = sum(mid * w for mid, w in kept_weighted) / sum_w if sum_w > 0 else float("nan")
        else:
            mean_score = float("nan")
        final_tier = _score_to_tier(mean_score) if kept_tiers else ""
        # 审计: 记加权 vs 简单平均差异
        mean_simple = sum(kept_tiers) / len(kept_tiers) if kept_tiers else float("nan")
        normalized_es = []
        for m in kept_models:
            rr = r2_for_batch.get(m, {}).get(city, {})
            es, _flags = normalize_r2_evidence_strength(
                rr.get("evidence_strength", ""),
                rr.get("reasoning", ""),
                rr.get("r1_evidence_strength", ""),
            )
            normalized_es.append(es[:1])
        es_weight_str = ",".join(normalized_es)
        final[city] = {
            "tier": final_tier,
            "reasoning": d.get("reasoning", ""),
            "drop": ",".join(str(i) for i in (drop_idxs or [])),
            "drop_models": ",".join(sorted(drop_models)),
            "kept_models": ",".join(kept_models),
            "mean_score": round(mean_score, 2) if kept_tiers else "",     # V27.3.20: 加权 mean
            "mean_simple": round(mean_simple, 2) if kept_tiers else "",   # V27.3.20: 简单 mean (审计对比)
            "evidence_weights": es_weight_str,                            # V27.3.20: kept 评委各自 evidence (s/m/w 简写)
            "evidence_quality_flags": render_flags(
                f for flag_str in kept_flag_strs for f in flag_str.split(",")
            ),
            "consul_reasoning_flags": render_flags(detect_evidence_quality_flags(d.get("reasoning", ""))),
        }
    return final


def run_consul_for_indicator_batch(indicator, cities_batch, consul_model, r1_for_batch, r2_for_batch, dry_run):
    system, user = build_consul_prompt(indicator, cities_batch, r1_for_batch, r2_for_batch)
    # V27.3.20 (subagent 最终审 [中]): Consul 也加 1 次 retry, gemini-2.5-pro 偶发非严格 JSON 该 batch 救一次.
    # 5/23 修: gemini-2.5-pro thinking 真吃光 max_tokens=4000 (88 次 EMPTY_CONTENT). 改 16000.
    decisions = None
    last_err = None
    for attempt in range(2):
        try:
            # 真绕过 run_llm_call (固定 4000), 真直接 run_llm_chat 传 max_tokens=16000
            if dry_run:
                raw = '{"indicator":"<DRY_RUN>","decisions":{}}'
            else:
                from client import run_llm_chat
                msgs = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                raw = run_llm_chat(
                    msgs,
                    max_tokens=16000,  # gemini-2.5-pro 真支持 64K, 4000→16000 修 EMPTY_CONTENT
                    model=consul_model["vendor"],
                    trace_tag="consul_" + consul_model["vendor"],
                    web_search=False,
                )
            stripped = _strip_md_wrap(raw)
            if not stripped.startswith("{"):
                last_err = f"non-JSON output (first 80c): {stripped[:80]!r}"
                continue
            parsed = json.loads(stripped)
            # 🔴 新 schema (2026-05-17 用户拍板): Consul 只输出 'decisions' 不输出 'final'
            decisions = parsed.get("decisions", parsed.get("final", {}))
            # 容错 (2026-05-30): LLM 偶尔输出 [{city: X, ...}, ...] 而非 {city: {...}} → 转 dict 防 _apply 崩
            # 50 城 5 batches 概率比 5 城 1 batch 高 5×, 实测触发过一次 AttributeError 拖垮整个 step
            if isinstance(decisions, list):
                decisions = {
                    (d.get("city") or d.get("name")): {k: v for k, v in d.items() if k not in ("city", "name")}
                    for d in decisions
                    if isinstance(d, dict) and (d.get("city") or d.get("name"))
                }
            if isinstance(decisions, dict):
                decisions = normalize_ratings_city_keys(decisions, cities_batch)
            if decisions:
                break
            last_err = f"JSON valid but decisions empty (attempt {attempt+1})"
        except Exception as e:
            last_err = str(e)
    if not decisions:
        return {"_error": last_err or "consul_parse_fail"}
    # 代码后处理: 在 kept_models 上算 mean tier
    return _apply_consul_decisions(r2_for_batch, decisions, cities_batch)


def run_step_consul(meta, r1_data, r2_data, out_dir, dry_run=False) -> dict:
    indicators = meta["indicators"]
    cities = meta["cities"]
    # V27.3.20 修 S4 Consul 模型选择: prompt_builder.py 把 Master Excel Sheet 7 列重命名:
    #   Excel `model_id`  → 代码 key `vendor` (例 "gemini-2.5-pro")
    #   Excel `role`      → 代码 key `model`  (例 "议会成员" / "Consul 裁判")
    #   Excel `provider`  → 代码 key `role`   (例 "Google")
    # 旧代码用 m.get("role") == "consul" 字面值, 实际 role 字段值是 "Google" → 永远 fallback.
    # 正确做法: 用 m.get("model") == "Consul 裁判" (代码 model 字段就是 Excel role 列).
    consul_candidates = [m for m in meta["models"] if m.get("model") == "Consul 裁判"]
    if not consul_candidates:
        raise RuntimeError(
            "❌ Master Excel Sheet 7 找不到 Consul 裁判 — 检查 Sheet 7 role 列是否有 'Consul 裁判'. "
            "fallback 已禁用 (旧 fallback 会让 vendor='google' 传给 LLM gateway, 不安全)."
        )
    consul_model = consul_candidates[0]

    # 5/23 用户拍板加并发: Consul 真**跨 indicator + 跨 batch 并发** (max_workers=48, 5/23 实测 sweet spot).
    # 真 54 任务 (18 ind × 3 batch) → ThreadPool(32) → ~2 波完成, 真 1-2 min vs 串行 27-54 min (~20x 提速)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_final = {alias: {} for alias in (ind["alias_name"] for ind in indicators)}
    tasks = []  # [(ind, alias, batch_idx, batch_cities)]
    for ind_idx, ind in enumerate(indicators, 1):
        alias = ind["alias_name"]
        for b_idx, batch in enumerate(_chunk(cities, BATCH_SIZE), 1):
            tasks.append((ind, alias, b_idx, batch))
    n_batches = sum(1 for _ in _chunk(cities, BATCH_SIZE))
    print(f"  [Consul 并发] {len(tasks)} 任务 ({len(indicators)} ind × {n_batches} batch), max_workers=48")

    def _one_consul_task(t):
        ind, alias, b_idx, batch = t
        r1_for_batch = {
            m: {c: r1_data.get(alias, {}).get(m, {}).get(c, {}) for c in batch}
            for m in r1_data.get(alias, {}).keys()
        }
        r2_for_batch = {
            m: {c: r2_data.get(alias, {}).get(m, {}).get(c, {}) for c in batch}
            for m in r2_data.get(alias, {}).keys()
        }
        return (alias, run_consul_for_indicator_batch(
            ind, batch, consul_model, r1_for_batch, r2_for_batch, dry_run,
        ))

    n_done = 0
    with ThreadPoolExecutor(max_workers=48) as pool:
        futures = {pool.submit(_one_consul_task, t): t for t in tasks}
        for fut in as_completed(futures):
            alias, batch_final = fut.result()
            all_final.setdefault(alias, {}).update(batch_final)
            n_done += 1
            if n_done % 10 == 0:
                print(f"  [Consul] 完成 {n_done}/{len(tasks)} 任务")

    import csv
    # V27.3.20 修 #9: csv 列对齐 _apply_consul_decisions 真实写出的字段, 保留 Consul 决策审计.
    # V27.3.20 P3 加 mean_simple + evidence_weights 列 (evidence_strength 加权 mean 对比)
    rows = []
    for ind_alias, by_city in all_final.items():
        for city in cities:
            r = by_city.get(city, {})
            rows.append({
                "indicator": ind_alias,
                "city": city,
                "tier": r.get("tier", ""),
                "reasoning": r.get("reasoning", ""),
                "drop": r.get("drop", ""),
                "drop_models": r.get("drop_models", ""),
                "kept_models": r.get("kept_models", ""),
                "mean_score": r.get("mean_score", ""),
                "mean_simple": r.get("mean_simple", ""),
                "evidence_weights": r.get("evidence_weights", ""),
                "evidence_quality_flags": r.get("evidence_quality_flags", ""),
                "consul_reasoning_flags": r.get("consul_reasoning_flags", ""),
            })
    with (out_dir / "step_05_consul.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "indicator", "city", "tier", "reasoning",
            "drop", "drop_models", "kept_models", "mean_score",
            "mean_simple", "evidence_weights", "evidence_quality_flags",
            "consul_reasoning_flags",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"✓ step_05 Consul: {len(indicators)} ind × {len(cities)} 城 最终 tier")
    return all_final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/v01")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    import yaml, csv
    meta = yaml.safe_load((Path(args.out) / "step_01_meta.yml").read_text())

    def load_csv(path):
        out = {}
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out.setdefault(row["indicator"], {}).setdefault(row["model"], {})[row["city"]] = {
                    "tier": row["tier"],
                    "reasoning": row["reasoning"],
                    "decision": row.get("decision", ""),
                    "evidence_strength": row.get("evidence_strength", ""),
                    "r1_evidence_strength": row.get("r1_evidence_strength", ""),
                    "evidence_quality_flags": row.get("evidence_quality_flags", ""),
                }
        return out

    r1 = load_csv(Path(args.out) / "step_03_r1.csv")
    r2 = load_csv(Path(args.out) / "step_04_r2.csv")
    for ind, by_model in r2.items():
        for model, by_city in by_model.items():
            for city, rec in by_city.items():
                rec["r1_evidence_strength"] = (
                    r1.get(ind, {}).get(model, {}).get(city, {}).get("evidence_strength", "")
                )
    run_step_consul(meta, r1, r2, Path(args.out), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
