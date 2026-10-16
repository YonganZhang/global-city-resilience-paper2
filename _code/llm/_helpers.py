"""_code/llm/_helpers.py — V4 P3.5.l: consul 等依赖的 helpers
(2026-05-21 P3.5.k 终结后: 老 lib/agent.py 已归档至 代码/legacy/giri_city_agent/agent.py,
本文件是该 helpers 的 active 落地, 仅 _code/llm/ 内部使用).
"""
from __future__ import annotations
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict

def _strip_diacritics(text: str) -> str:
    """Remove diacritical marks from text (e.g. Quảng Ngãi → Quang Ngai).
    Keeps ASCII-safe characters to avoid API 400 errors from some providers."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty LLM output.")
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)

    start = s.find("{")
    if start < 0:
        raise ValueError("No '{' found in LLM output.")

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                return json.loads(candidate)

    raise ValueError("Unclosed JSON object in LLM output.")
