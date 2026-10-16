from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ._shared import (
    EXPECTED_FULL_ROWS,
    FULL_GCIRI_REL,
    MASTER_UCDB_REL,
    build_project_path,
    load_full_gciri,
    load_master_ucdb,
)


PAPER_REL = Path("_paper/gciri_full5862_spine_20260704")
STAMP = datetime.now().strftime("%Y%m%d")

MASTER_COLUMNS = [
    "ucdb_id",
    "area_km2",
    "pop_2025",
    "exp_pop_density_city__RAW",
    "exp_ntl_city__RAW",
    "exp_pop_ntl_city__RAW",
    "GIRI__RAW_nat",
    "R_Ph__RAW_nat",
    "F__DEF_nat",
    "cap_abs_idx__CAP_nat",
    "cap_resp_idx__CAP_nat",
    "cap_rest_idx__CAP_nat",
]

PROXIES = [
    {
        "column": "pop_2025_master",
        "label": "UCDB/WorldPop city population",
        "independence": "descriptive covariate; not independent validation",
        "allowed": "city-size sanity and stratification check",
    },
    {
        "column": "area_km2",
        "label": "UCDB built-up area",
        "independence": "descriptive UCDB geometry covariate; not outcome validation",
        "allowed": "city-footprint sanity check",
    },
    {
        "column": "exp_pop_density_city__RAW",
        "label": "WorldPop-derived population density exposure",
        "independence": "input-adjacent exposure proxy",
        "allowed": "exposure-chain sanity check only",
    },
    {
        "column": "exp_ntl_city__RAW",
        "label": "VIIRS night-time lights exposure",
        "independence": "input-adjacent remote-sensing proxy",
        "allowed": "urban activity sanity check only",
    },
    {
        "column": "exp_pop_ntl_city__RAW",
        "label": "Population-nightlight exposure composite",
        "independence": "input-adjacent exposure proxy",
        "allowed": "exposure-chain sanity check only",
    },
    {
        "column": "GIRI__RAW_nat",
        "label": "Official national GIRI inherited by master",
        "independence": "national anchor; not independent validation",
        "allowed": "gross national-consistency anchor",
    },
    {
        "column": "R_Ph__RAW_nat",
        "label": "National physical-risk baseline",
        "independence": "formula input lineage",
        "allowed": "formula lineage check",
    },
    {
        "column": "F__DEF_nat",
        "label": "National fragility / capacity factor",
        "independence": "formula input lineage",
        "allowed": "formula lineage check",
    },
    {
        "column": "cap_abs_idx__CAP_nat",
        "label": "National absorptive-capacity input",
        "independence": "capacity input lineage",
        "allowed": "capacity-inheritance boundary check",
    },
    {
        "column": "cap_resp_idx__CAP_nat",
        "label": "National response-capacity input",
        "independence": "capacity input lineage",
        "allowed": "capacity-inheritance boundary check",
    },
    {
        "column": "cap_rest_idx__CAP_nat",
        "label": "National restorative-capacity input",
        "independence": "capacity input lineage",
        "allowed": "capacity-inheritance boundary check",
    },
]


def _corr_pair(df: pd.DataFrame, left: str, right: str, method: str) -> float | None:
    pair = df[[left, right]].dropna()
    if len(pair) < 3:
        return None
    value = pair[left].corr(pair[right], method=method)
    if pd.isna(value):
        return None
    return round(float(value), 4)


def _format_value(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _write_boundary_matrix(out_path: Path) -> None:
    rows = [
        (
            "Official GIRI",
            "_pipelines/.../analysis/country_vs_official_giri.csv and master `GIRI__RAW_nat`",
            "Country-level anchor inherited by GCIRI lineage",
            "Report gross consistency only",
            "Do not call it independent city-level validation",
        ),
        (
            "UCDB identity, coordinates, area, and population",
            "_outputs/master_v27_6/master_v28_ucdb.parquet",
            "Current city frame and descriptive covariates",
            "Use to verify row identity, coverage, city-size patterns, and mapping",
            "Do not treat city size or area as resilience outcomes",
        ),
        (
            "WorldPop and VIIRS exposure proxies",
            "Registered remote-sensing sources inside active master",
            "Input-adjacent exposure information used in the physical-risk chain",
            "Use as sanity checks for exposure-chain behavior",
            "Do not use as external validation of final GCIRI",
        ),
        (
            "National capacity and fragility fields",
            "Current master national GIRI/capacity fields",
            "Formula lineage and inherited country baseline",
            "Use to explain broad national structure and formula boundary",
            "Do not present as independent proof of local capacity",
        ),
        (
            "Case-pair and anomaly audits",
            "Current paper audit ledgers",
            "Manual and code-assisted plausibility checks",
            "Use to flag common-sense passes, caveats, and unresolved rows",
            "Do not generalize case pairs as causal tests",
        ),
        (
            "Independent city-level outcome validation",
            "Not present in current package",
            "Missing evidence class",
            "State as future validation need",
            "Do not imply validation against losses, service continuity, or recovery outcomes",
        ),
    ]
    lines = [
        "# Validation Boundary Matrix: GCIRI full5862",
        "",
        f"Created: {STAMP}",
        "",
        "This matrix separates evidence that can support current paper claims from evidence that only provides a sanity or lineage check.",
        "",
        "| Evidence item | Current source | Relation to GCIRI | Allowed use | Forbidden wording |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "Bottom line: the current package supports a bounded city-scale screening layer and common-sense plausibility checks. It does not yet support a claim of independent city-level outcome validation.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(out_path: Path, stats: pd.DataFrame, merged: pd.DataFrame) -> None:
    official = stats.loc[stats["column"] == "GIRI__RAW_nat"].iloc[0]
    ntl = stats.loc[stats["column"] == "exp_ntl_city__RAW"].iloc[0]
    pop = stats.loc[stats["column"] == "pop_2025_master"].iloc[0]
    exposure = stats.loc[stats["column"] == "exp_pop_ntl_city__RAW"].iloc[0]
    lines = [
        "# City Proxy Sanity Summary: GCIRI full5862",
        "",
        f"Created: {STAMP}",
        "",
        "## Data contract",
        "",
        f"- GCIRI source: `{FULL_GCIRI_REL.as_posix()}`",
        f"- Master source: `{MASTER_UCDB_REL.as_posix()}`",
        f"- Joined rows: {len(merged)} of expected {EXPECTED_FULL_ROWS}",
        "- Join key: `ucdb_id`, validated as one-to-one after renaming duplicate population fields.",
        "",
        "## What this analysis can and cannot prove",
        "",
        "This is a validation-boundary audit, not an independent validation study. The checked proxy fields come from the current UCDB, WorldPop, VIIRS, and national GIRI lineage. They are useful for detecting gross contradictions, but several are inputs or input-adjacent variables in the GCIRI construction chain.",
        "",
        "## Main numeric checks",
        "",
        f"- Official national GIRI vs final GCIRI: Spearman {_format_value(official['spearman_gciri'])}, Pearson {_format_value(official['pearson_gciri'])}. This is a gross national anchor, not independent proof.",
        f"- VIIRS night-time lights exposure vs final GCIRI: Spearman {_format_value(ntl['spearman_gciri'])}, Pearson {_format_value(ntl['pearson_gciri'])}. This is an urban-activity sanity check only.",
        f"- Population-nightlight exposure composite vs final GCIRI: Spearman {_format_value(exposure['spearman_gciri'])}, Pearson {_format_value(exposure['pearson_gciri'])}. This should be interpreted as exposure-chain behavior.",
        f"- City population vs final GCIRI: Spearman {_format_value(pop['spearman_gciri'])}, Pearson {_format_value(pop['pearson_gciri'])}. This indicates whether the final score is strongly dominated by city size.",
        "",
        "## Manuscript consequence",
        "",
        "The manuscript should say that GCIRI passes internal consistency, national-anchor, and common-sense plausibility checks. It should not say that GCIRI has been externally validated against independent city-level disaster-loss, service-continuity, or recovery-outcome observations.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    full = load_full_gciri().copy()
    master = load_master_ucdb(MASTER_COLUMNS).copy()
    if master["ucdb_id"].duplicated().any():
        raise ValueError("master_v28_ucdb has duplicated ucdb_id values")
    if full["ucdb_id"].duplicated().any():
        raise ValueError("full_gciri_with_meta has duplicated ucdb_id values")
    master = master.rename(columns={"pop_2025": "pop_2025_master"})
    merged = full.merge(master, on="ucdb_id", how="left", validate="one_to_one")
    if len(merged) != EXPECTED_FULL_ROWS:
        raise ValueError(f"joined table has {len(merged)} rows; expected {EXPECTED_FULL_ROWS}")
    required_after_join = {proxy["column"] for proxy in PROXIES}
    missing = sorted(required_after_join.difference(merged.columns))
    if missing:
        raise ValueError(f"joined table missing proxy columns: {missing}")

    rows = []
    for proxy in PROXIES:
        col = proxy["column"]
        rows.append(
            {
                "column": col,
                "label": proxy["label"],
                "n_nonmissing": int(merged[col].notna().sum()),
                "pearson_gciri": _corr_pair(merged, col, "gciri", "pearson"),
                "spearman_gciri": _corr_pair(merged, col, "gciri", "spearman"),
                "pearson_r_ph_city": _corr_pair(merged, col, "r_ph_city", "pearson"),
                "spearman_r_ph_city": _corr_pair(merged, col, "r_ph_city", "spearman"),
                "pearson_f_city": _corr_pair(merged, col, "f_city", "pearson"),
                "spearman_f_city": _corr_pair(merged, col, "f_city", "spearman"),
                "independence_class": proxy["independence"],
                "allowed_wording": proxy["allowed"],
            }
        )
    stats = pd.DataFrame(rows)

    out_dir = build_project_path(PAPER_REL)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / f"city_proxy_sanity_stats_{STAMP}.csv"
    matrix_path = out_dir / f"validation_boundary_matrix_{STAMP}.md"
    summary_path = out_dir / f"city_proxy_sanity_summary_{STAMP}.md"
    meta_path = out_dir / f"validation_boundary_sources_{STAMP}.json"
    stats.to_csv(stats_path, index=False)
    _write_boundary_matrix(matrix_path)
    _write_summary(summary_path, stats, merged)
    meta = {
        "created": STAMP,
        "full_gciri_source": FULL_GCIRI_REL.as_posix(),
        "master_source": MASTER_UCDB_REL.as_posix(),
        "rows": len(merged),
        "expected_rows": EXPECTED_FULL_ROWS,
        "join_key": "ucdb_id",
        "outputs": [p.name for p in [stats_path, matrix_path, summary_path]],
        "claim_boundary": "sanity and lineage checks only; not independent city-level outcome validation",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {stats_path}")
    print(f"Wrote {matrix_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
