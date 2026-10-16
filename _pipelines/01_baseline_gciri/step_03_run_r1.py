"""step_03_run_r1.py — 议会 R1: 8 评委 × per-indicator × 10 城 batch

V27.3.19 议会 R1 实现:
- 18 indicator 循环
- 每 indicator 把 50 城分 5 batch(10 城/batch, 最后 batch 9 城)
- 每 batch 调 8 评委(并发)
- 每评委输出: tier label + 30-60 字 reasoning
- LLM 调用走 `_code/llm/client.py:run_llm_chat` (V4 P3.5.k 后 active 路径)

输出:
- outputs/v<NN>/step_03_r1.csv         50 城 × 18 indicator × 8 评委的 tier
- outputs/v<NN>/step_03_r1_evidence/   reasoning 详细(每文件 1 batch × 1 ind)

注意:
- 本 step 真调 LLM($, 时间), --dry-run 模式只生成 prompt 不调
- 8 评委并发用 ThreadPoolExecutor
- JSON 解析失败自动重试 1 次(LLM 偶尔输出不严格 JSON)
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

from prompt_builder import build_r1_prompt  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "_pipelines" / "01_baseline_gciri"))
from evidence_quality import normalize_r1_evidence_strength  # noqa: E402

def _strip_md_wrap(raw: str) -> str:
    """去 LLM 输出的 markdown 代码块包裹 (```json...``` 或 ```...```)."""
    raw = raw.strip()
    if raw.startswith("```"):
        # 去 ```json 或 ``` 头
        raw = re.sub(r"^```(?:json|JSON)?\s*\n?", "", raw)
        # 去尾 ```
        raw = re.sub(r"\n?```\s*$", "", raw)
    return raw.strip()


BATCH_SIZE = 10  # 5/23 用户拍板改回 10: R2 含 R1 packet, BATCH=20 真让 gemini-flash truncate (max_tokens 不够). BATCH=10 真本质修.


def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _normalize_city_label(label: str) -> str:
    """Normalize LLM-emitted city labels for matching, not for display."""
    label = (label or "").strip().strip('"').strip("'")
    label = re.sub(r"\s+", " ", label)
    return label.casefold()


def _display_city_from_key(city: str) -> str:
    """Return the display name from a full-mode key like 'City [ISO3; UCDB 123]'."""
    m = re.match(r"^(?P<city>.+) \[[A-Z]{3}; UCDB \d+\]$", city or "")
    return m.group("city") if m else city


def normalize_ratings_city_keys(ratings: dict, cities_batch: list[str]) -> dict:
    """Map LLM output keys back to exact run city keys.

    Full-mode prompts use row-level keys (`City [ISO3; UCDB id]`) to avoid
    duplicate city names. Some models still return the naked display city.
    This keeps unique naked-name matches and leaves ambiguous/unmatched keys
    untouched so audits can see them.
    """
    alias_to_city: dict[str, str | None] = {}
    for city in cities_batch:
        aliases = {
            _normalize_city_label(city),
            _normalize_city_label(_display_city_from_key(city)),
        }
        for alias in aliases:
            if not alias:
                continue
            if alias in alias_to_city and alias_to_city[alias] != city:
                alias_to_city[alias] = None
            else:
                alias_to_city[alias] = city

    normalized: dict = {}
    for raw_city, rating in (ratings or {}).items():
        target = alias_to_city.get(_normalize_city_label(raw_city))
        normalized[target or raw_city] = rating
    return normalized


def run_llm_call(model_config: dict, system: str, user: str, dry_run: bool = False,
                 trace_prefix: str = "r1_") -> str:
    """Call one LLM (V4 P3.5.k: 调 _code/llm/client.py:run_llm_chat).

    Dry-run: 返回 placeholder JSON.
    2026-06-01: trace_prefix 参数化 - step_04 (R2) 传 "r2_" 区分调用阶段.
    """
    if dry_run:
        return '{"indicator":"<DRY_RUN>","ratings":{}}'
    from client import run_llm_chat  # noqa: E402, V4 _code/llm/client.py (sys.path 已在 28 行加)
    try:
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return run_llm_chat(
            msgs,
            max_tokens=16000,  # 5/23 修: gemini-2.5-flash thinking 真吃 token, 4000 真truncate 890 次. 改 16000.
            model=model_config.get("vendor"),
            trace_tag=trace_prefix + (model_config.get("vendor") or "default"),
            web_search=False,  # V27.3.20: 议会已预拼 Tavily evidence 到 user prompt, 不需 LLM call 时再 web search.
        )
    except Exception as e:
        return f'{{"error":"{e}"}}'


def run_r1_for_indicator_batch(
    indicator: dict, cities_batch: list[str], models: list[dict], dry_run: bool
) -> dict:
    """对单个 indicator + 单个 batch (10 城), 8 评委并发跑.

    Returns: {model_name: {city: {tier, reasoning}}}
    """
    system, user = build_r1_prompt(indicator, cities_batch)
    results = {}
    with ThreadPoolExecutor(max_workers=len(models)) as ex:
        future_to_model = {
            ex.submit(run_llm_call, m, system, user, dry_run): m
            for m in models
        }
        for fut in as_completed(future_to_model):
            m_cfg = future_to_model[fut]
            mname = m_cfg["vendor"]
            # V27.3.20 修 #8: JSON 解析 silent error → 1 次 retry (docstring 承诺过自动重试).
            # V27.3.20 codex 最终审补丁: 也覆盖 "JSON 合法但 ratings 空" corner case.
            parsed = None
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
                    if ratings:  # 非空 dict 才接受
                        ratings = normalize_ratings_city_keys(ratings, cities_batch)
                        break
                    last_err = f"JSON valid but ratings empty (attempt {attempt+1})"
                except Exception as e:
                    last_err = str(e)
            if ratings:
                results[mname] = ratings
            else:
                results[mname] = {"_error": last_err or "unknown_parse_fail"}
    return results


def run_step_r1(meta: dict, out_dir: Path, dry_run: bool = False) -> dict:
    """主入口: 18 indicator × N batch × 8 评委.

    V28 (5/23 用户拍板): R1 跨 indicator+batch 并发 ThreadPool(48) — 跟 R2 同模板.
    实测 R1 50 城串行 ~75min, 改 48 并发后 ~5-8min (10x 加速).

    Returns: dict {indicator_alias: {model: {city: rating}}}
    """
    indicators = meta["indicators"]
    cities = meta["cities"]
    models = [m for m in meta["models"] if m.get("model") == "议会成员"][:8]
    if len(models) < 8:
        print(f"⚠️  只有 {len(models)} 评委 (期望 8), 看 Sheet 7", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = out_dir / "step_03_r1_evidence"
    evidence_dir.mkdir(exist_ok=True)

    # 5/23 用户拍板 (max_workers=48): R1 跨 indicator+batch 并发
    all_r1 = {ind["alias_name"]: {m["vendor"]: {} for m in models} for ind in indicators}
    tasks = []
    for ind in indicators:
        alias = ind["alias_name"]
        for batch_idx, batch in enumerate(_chunk(cities, BATCH_SIZE), 1):
            tasks.append((ind, alias, batch_idx, batch))
    print(f"  [R1 并发] {len(tasks)} 任务 ({len(indicators)} ind × ? batch), max_workers=48")

    def _one_r1_task(t):
        ind, alias, batch_idx, batch = t
        batch_res = run_r1_for_indicator_batch(ind, batch, models, dry_run)
        ev_file = evidence_dir / f"{alias}_batch_{batch_idx}.json"
        ev_file.write_text(json.dumps(batch_res, ensure_ascii=False, indent=2), encoding="utf-8")
        return (alias, batch_res)

    n_done = 0
    with ThreadPoolExecutor(max_workers=48) as pool:
        futures = {pool.submit(_one_r1_task, t): t for t in tasks}
        for fut in as_completed(futures):
            alias, batch_res = fut.result()
            for mname, ratings in batch_res.items():
                all_r1.setdefault(alias, {}).setdefault(mname, {}).update(ratings)
            n_done += 1
            if n_done % 10 == 0:
                print(f"  [R1] 完成 {n_done}/{len(tasks)} 任务")

    # 汇总 csv
    save_step_r1_csv(all_r1, cities, out_dir)
    print(f"✓ step_03 R1: {len(indicators)} ind × {len(cities)} 城 × {len(models)} 评委")
    return all_r1


def save_step_r1_csv(all_r1: dict, cities: list[str], out_dir: Path):
    """长格式 csv: indicator, city, model, tier, reasoning, evidence_strength.

    V27.3.20 P3 修 (Codex+Subagent 审): R1 prompt 要求 LLM 输出 evidence_strength (strong/medium/weak),
    但旧 csv 没落盘. Consul 想程序化识别"无证据极值"时找不到字段 → 修.
    """
    import csv
    rows = []
    for ind_alias, by_model in all_r1.items():
        for mname, by_city in by_model.items():
            for city in cities:
                r = by_city.get(city, {})
                raw_evidence_strength = r.get("evidence_strength", "")
                evidence_strength, evidence_quality_flags = normalize_r1_evidence_strength(
                    raw_evidence_strength,
                    r.get("reasoning", ""),
                )
                rows.append({
                    "indicator": ind_alias,
                    "city": city,
                    "model": mname,
                    "tier": r.get("tier", ""),
                    "reasoning": r.get("reasoning", ""),
                    "evidence_strength": evidence_strength,  # normalized for downstream R2/Consul weighting
                    "raw_evidence_strength": raw_evidence_strength,
                    "evidence_quality_flags": evidence_quality_flags,
                })
    csv_path = out_dir / "step_03_r1.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "indicator", "city", "model", "tier", "reasoning", "evidence_strength",
            "raw_evidence_strength", "evidence_quality_flags",
        ])
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/v01")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    import yaml
    meta = yaml.safe_load((Path(args.out) / "step_01_meta.yml").read_text())
    run_step_r1(meta, Path(args.out), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
