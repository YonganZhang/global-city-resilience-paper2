"""run_all.py — 议会 V27.3.19 pipeline 主调度

跑 9 step 顺序执行,自动建 outputs/v<NN>/ 版本目录.

用法:
  python run_all.py --new-version              # 新建 v<NN+1>/
  python run_all.py --out outputs/v01 --dry-run  # 指定版本, dry-run 不真调 LLM
  python run_all.py --skip-llm                  # 跳过 step_03/04/05(议会),只跑后处理
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
STEP_FILES = [
    "step_01_load_metadata.py",
    "step_02_load_tavily.py",
    "step_03_run_r1.py",
    "step_04_run_r2.py",
    "step_05_consul.py",
    "step_06_compute_f.py",
    "step_07_compute_r_ph_city.py",
    "step_08_compute_capacities.py",
    "step_09_compute_gciri.py",
]
LLM_STEPS = {"step_03_run_r1.py", "step_04_run_r2.py", "step_05_consul.py"}
ALLOW_PAID_ENV = "GCIRI_ALLOW_PAID_LLM"


def _find_next_version(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base_dir / f"v{n:02d}").exists():
        n += 1
    return base_dir / f"v{n:02d}"


def run_step(step_file: str, out_dir: Path, dry_run: bool, cities: str | None = None,
             limit_ind: int | None = None, mode: str | None = None, city: str | None = None,
             downscale_method: str = "analytic") -> int:
    cmd = [sys.executable, str(THIS_DIR / step_file), "--out", str(out_dir)]
    if dry_run and step_file in LLM_STEPS:
        cmd.append("--dry-run")
    if step_file == "step_07_compute_r_ph_city.py":
        cmd.extend(["--downscale-method", downscale_method])
    # step_01 接受 --mode/--city/--cities + --limit-indicators (其他 step 从 step_01_meta.yml 读城)
    if step_file == "step_01_load_metadata.py":
        if cities:
            cmd.extend(["--cities", cities])
        elif mode:
            cmd.extend(["--mode", mode])
            if city:
                cmd.extend(["--city", city])
        if limit_ind:
            cmd.extend(["--limit-indicators", str(limit_ind)])
    print(f"\n=== {step_file} ===")
    return subprocess.call(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new-version", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true", help="LLM step 用 dry-run(不真调)")
    p.add_argument("--skip-llm", action="store_true", help="跳过 step_03/04/05")
    p.add_argument("--allow-paid-llm", action="store_true",
                   help=f"允许真调付费 LLM；也可设置 {ALLOW_PAID_ENV}=1")
    p.add_argument("--why", default="", help="why_this_run(写 _run_meta.yml)")
    p.add_argument("--mode", choices=["single", "b50", "full6275"], default="b50",
                   help="三模式 (全议会同代码): single(配--city) / b50(50城默认) / full6275(master全量)")
    p.add_argument("--city", default=None, help="--mode single 时指定城市名")
    p.add_argument("--cities", default=None, help="逗号分隔自定义城市清单 (优先于 --mode)")
    p.add_argument("--limit-indicators", type=int, default=None, help="只跑前 N 个 indicator")
    p.add_argument("--downscale-method", choices=["analytic", "trapezoidal"], default="analytic")
    args = p.parse_args()
    paid_llm_requested = not args.dry_run and not args.skip_llm
    paid_llm_allowed = args.allow_paid_llm or os.getenv(ALLOW_PAID_ENV) == "1"
    if paid_llm_requested and not paid_llm_allowed:
        raise SystemExit(
            f"🔴 付费 LLM 步骤默认禁用。当前不会调用智增增/API。\n"
            f"   测试请用 --dry-run 或 --skip-llm；真跑前显式加 --allow-paid-llm "
            f"或设置 {ALLOW_PAID_ENV}=1。"
        )
    if args.allow_paid_llm:
        os.environ[ALLOW_PAID_ENV] = "1"

    base = THIS_DIR / "outputs"
    if args.new_version or not args.out:
        out_dir = _find_next_version(base)
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 写 _run_meta.yml(跑前 + 跑后)
    git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    meta = {
        "run_id": out_dir.name,
        "start_time": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": git_hash,
        "why_this_run": args.why or "pipeline run",
        "dry_run": args.dry_run,
        "skip_llm": args.skip_llm,
        "pipeline": "_pipelines/01_baseline_gciri",
        "downscale_method": args.downscale_method,
        "steps": [],
    }

    for step in STEP_FILES:
        if args.skip_llm and step in LLM_STEPS:
            print(f"--- skip {step} (--skip-llm) ---")
            meta["steps"].append({"step": step, "status": "skipped"})
            continue
        rc = run_step(step, out_dir, args.dry_run, cities=args.cities,
                      limit_ind=args.limit_indicators, mode=args.mode, city=args.city,
                      downscale_method=args.downscale_method)
        meta["steps"].append({"step": step, "status": "ok" if rc == 0 else f"failed rc={rc}"})
        if rc != 0:
            print(f"🔴 {step} 失败 rc={rc}, 继续后续步骤")

    meta["end_time"] = datetime.datetime.utcnow().isoformat() + "Z"
    meta["outcome"] = "success" if all(s["status"] in ("ok", "skipped") for s in meta["steps"]) else "partial"

    import yaml
    (out_dir / "_run_meta.yml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"\n✅ 跑完 {out_dir.name}, outcome={meta['outcome']}")
    print(f"   _run_meta.yml 已写")


if __name__ == "__main__":
    main()
