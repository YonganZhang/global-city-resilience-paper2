"""V4 入口: 国家级 GIRI 归一化参数反推 (✅ 已完成的前置工作).

⚠️ 关键事实 (2026-05-17 用户 audit 提醒):
官方 GIRI (Cardona 2023) 没公开 S/Z 归一化的 Xmin/Xmax 参数. 用户在
`国家级GIRI复现/reproduce_giri.py` Phase 3 反推过, 结果存到
`国家级GIRI复现/giri_band_ref.json` (5 档 VeryHigh/High/Medium/Low/VeryLow).

这是一次性已完成的前置工作 — 后续 pipeline 直接读 JSON 用, 不重算.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_REPRO = _REPO / "国家级GIRI复现"
if str(_REPRO) not in sys.path:
    sys.path.insert(0, str(_REPRO))

BAND_REF_JSON = _REPRO / "giri_band_ref.json"


def load_band_ref() -> dict:
    """读已反推的归一化 band_means 参数 (5 档 × 18 indicator).

    输出: {indicator_name: {VeryHigh, High, Medium, Low, VeryLow: float}}
    """
    if not BAND_REF_JSON.exists():
        raise FileNotFoundError(f"归一化参数缺: {BAND_REF_JSON}")
    return json.loads(BAND_REF_JSON.read_text(encoding="utf-8"))


def run_reproduce_giri(*args, **kwargs):
    """V4 入口: 国家级 GIRI 复现 (Phase 1-3, 含归一化参数反推).

    实现: 国家级GIRI复现/reproduce_giri.py::main
    """
    from reproduce_giri import main  # noqa: E402
    return main(*args, **kwargs)
