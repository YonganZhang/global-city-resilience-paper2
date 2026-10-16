"""Evidence/reasoning hygiene helpers for GCIRI council outputs.

These checks are intentionally conservative. They do not try to prove a
reasoning is true; they only prevent low-information or vote-consensus text from
receiving a high evidence weight.
"""
from __future__ import annotations

import re
from typing import Iterable

VALID_EVIDENCE = {"strong", "medium", "weak"}
EVIDENCE_ORDER = {"weak": 0, "medium": 1, "strong": 2}

TIER_ONLY_RE = re.compile(
    r"^(?:[A-F](?:[+-])?)(?:[\s/,;，、]*[A-F](?:[+-])?)*$|^(?:[A-F](?:[+-]?)){1,4}$"
)
CONSENSUS_RE = re.compile(
    r"(多数|评委|模型|共识|投票|票数|votes?|majority|consensus|model[_\s-]?[a-h])",
    re.IGNORECASE,
)
CITY_EVIDENCE_RE = re.compile(
    r"(\d{4}|\d+(?:\.\d+)?%|条例|法规|法案|计划|项目|机构|部门|港|机场|地铁|"
    r"hospital|airport|port|metro|authority|department|program|act|code|"
    r"ordinance|law|policy|index)",
    re.IGNORECASE,
)


def normalize_evidence_label(value: object, default: str = "medium") -> str:
    label = str(value or "").strip().lower()
    return label if label in VALID_EVIDENCE else default


def normalize_evidence_cap(label: str, cap: str) -> str:
    label = normalize_evidence_label(label)
    cap = normalize_evidence_label(cap)
    if EVIDENCE_ORDER[label] <= EVIDENCE_ORDER[cap]:
        return label
    return cap


def is_bad_reasoning(reasoning: object) -> bool:
    if not isinstance(reasoning, str) or not reasoning.strip():
        return True
    text = reasoning.strip()
    if len(text) <= 8 and TIER_ONLY_RE.fullmatch(text):
        return True
    residue = re.sub(
        r"[A-F][+\-]?|R2|Model_[A-H]|[0-9]|[\s:：,，/()（）.\-]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return len(text) <= 18 and len(residue) <= 2


def detect_consensus_language(reasoning: object) -> bool:
    return isinstance(reasoning, str) and bool(CONSENSUS_RE.search(reasoning))


def detect_city_evidence_marker(reasoning: object) -> bool:
    return isinstance(reasoning, str) and bool(CITY_EVIDENCE_RE.search(reasoning))


def detect_evidence_quality_flags(reasoning: object) -> list[str]:
    flags: list[str] = []
    if is_bad_reasoning(reasoning):
        flags.append("bad_reasoning")
    if detect_consensus_language(reasoning):
        flags.append("consensus_language")
    if not detect_city_evidence_marker(reasoning):
        flags.append("no_city_evidence_marker")
    return flags


def render_flags(flags: Iterable[str]) -> str:
    return ",".join(dict.fromkeys(f for f in flags if f))


def normalize_r2_evidence_strength(
    raw_strength: object,
    reasoning: object,
    r1_strength: object = "",
) -> tuple[str, str]:
    """Return normalized evidence strength plus audit flags.

    R2 can preserve strong evidence only when the same model's R1 row was already
    strong. R2 consensus wording is not independent evidence.
    """
    raw = normalize_evidence_label(raw_strength)
    r1 = normalize_evidence_label(r1_strength, default="weak")
    flags = detect_evidence_quality_flags(reasoning)
    normalized = raw

    if raw == "strong" and r1 != "strong":
        normalized = normalize_evidence_cap(normalized, r1)
        flags.append(f"capped_by_r1_{r1}")

    if "bad_reasoning" in flags:
        normalized = "weak"
        flags.append("demoted_bad_reasoning")
    elif raw == "strong" and "consensus_language" in flags and "no_city_evidence_marker" in flags:
        normalized = "medium"
        flags.append("demoted_consensus_only")

    return normalized, render_flags(flags)


def normalize_r1_evidence_strength(
    raw_strength: object,
    reasoning: object,
) -> tuple[str, str]:
    """Return normalized R1 evidence strength plus audit flags."""
    raw = normalize_evidence_label(raw_strength)
    flags = detect_evidence_quality_flags(reasoning)
    normalized = raw

    if "bad_reasoning" in flags:
        normalized = "weak"
        flags.append("demoted_bad_reasoning")
    elif raw == "strong" and "no_city_evidence_marker" in flags:
        normalized = "medium"
        flags.append("demoted_no_city_evidence_marker")

    return normalized, render_flags(flags)
