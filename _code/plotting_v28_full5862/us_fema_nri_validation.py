from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from _shared import PROJECT_ROOT, load_full_gciri


PAPER_DIR = PROJECT_ROOT / "_paper/gciri_full5862_spine_20260704"
ITEM_URL = "https://www.arcgis.com/sharing/rest/content/items/39485e8035d446a5bff03259508ae355"
SERVICE_URL = "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/National_Risk_Index_Counties/FeatureServer/0/query"
OUT_FIELDS = ",".join(
    [
        "STCOFIPS",
        "STATEABBRV",
        "COUNTY",
        "POPULATION",
        "RISK_SCORE",
        "RISK_RATNG",
        "EAL_SCORE",
        "EAL_RATNG",
        "SOVI_SCORE",
        "SOVI_RATNG",
        "RESL_SCORE",
        "RESL_RATNG",
    ]
)


def _urlopen_json(url: str, params: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "gciri-validation/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"ArcGIS API error: {data['error']}")
    return data


def _build_output_dir() -> Path:
    explicit = os.environ.get("GCIRI_US_FEMA_OUTDIR")
    if explicit:
        out = Path(explicit)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = PAPER_DIR / f"external_validation_us_fema_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _fetch_item_metadata() -> dict[str, Any]:
    meta = _urlopen_json(ITEM_URL, {"f": "json"})
    description = meta.get("description") or ""
    match = re.search(r"National Risk Index Data Version:\s*([^<]+)", description)
    return {
        "item_id": meta.get("id"),
        "title": meta.get("title"),
        "owner": meta.get("owner"),
        "modified": meta.get("modified"),
        "url": meta.get("url"),
        "version_text": match.group(1).strip() if match else None,
        "snippet": meta.get("snippet"),
        "source_item_url": f"{ITEM_URL}?f=json",
    }


def _query_county_for_point(lon: float, lat: float) -> dict[str, Any] | None:
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": OUT_FIELDS,
        "returnGeometry": "false",
        "f": "json",
    }
    data = _urlopen_json(SERVICE_URL, params)
    features = data.get("features") or []
    if not features:
        return None
    if len(features) > 1:
        raise RuntimeError(f"Point returned multiple counties for lon={lon}, lat={lat}")
    return features[0].get("attributes") or {}


def _rating_counts(series: pd.Series) -> str:
    counts = series.fillna("Missing").value_counts().sort_index()
    return ", ".join(f"{idx}: {int(val)}" for idx, val in counts.items())


def _corr_table(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    checks = [
        ("gciri", "RESL_SCORE", "expected positive if both encode stronger resilience"),
        ("gciri", "SOVI_SCORE", "expected negative if higher vulnerability lowers resilience"),
        ("gciri", "EAL_SCORE", "expected negative or weak because FEMA EAL is loss/risk"),
        ("gciri", "RISK_SCORE", "expected negative or weak because FEMA risk is hazard-loss-vulnerability"),
        ("r_ph_city", "EAL_SCORE", "expected positive if both track physical or expected-loss risk"),
        ("r_ph_city", "RISK_SCORE", "expected positive if both track risk exposure"),
        ("cap_rest", "RESL_SCORE", "expected positive but definitions differ"),
    ]
    for left, right, expectation in checks:
        subset = joined[[left, right]].dropna()
        rows.append(
            {
                "left": left,
                "right": right,
                "n": len(subset),
                "pearson": subset[left].corr(subset[right], method="pearson") if len(subset) >= 3 else None,
                "spearman": subset[left].corr(subset[right], method="spearman") if len(subset) >= 3 else None,
                "expectation": expectation,
            }
        )
    return pd.DataFrame(rows)


def _write_summary(out_dir: Path, joined: pd.DataFrame, corr: pd.DataFrame, meta: dict[str, Any]) -> Path:
    matched = joined["fema_county_fips"].notna().sum()
    missing = len(joined) - matched
    lines = [
        "# US FEMA/NRI validation slice for GCIRI full5862",
        "",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Source contract",
        "",
        f"- GCIRI input: `_pipelines/01_baseline_gciri/outputs/v28_full5862_20260702_020421/analysis/full_gciri_with_meta.csv`",
        f"- GCIRI rows validated by loader: 5862",
        f"- USA city rows tested: {len(joined)}",
        f"- FEMA/NRI source: `{meta.get('title')}`",
        f"- FEMA/NRI version text: `{meta.get('version_text')}`",
        f"- FEMA/NRI API item: {meta.get('source_item_url')}",
        f"- FEMA/NRI service: {SERVICE_URL}",
        "",
        "## Join result",
        "",
        f"- Matched city points to FEMA/NRI counties: {matched}/{len(joined)}",
        f"- Missing county matches: {missing}",
        f"- FEMA county resilience ratings: {_rating_counts(joined['RESL_RATNG'])}",
        f"- FEMA composite risk ratings: {_rating_counts(joined['RISK_RATNG'])}",
        "",
        "## Correlation checks",
        "",
        "| GCIRI-side variable | FEMA/NRI variable | n | Pearson | Spearman | Expected direction |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in corr.to_dict("records"):
        pearson = "" if pd.isna(row["pearson"]) else f"{row['pearson']:.4f}"
        spearman = "" if pd.isna(row["spearman"]) else f"{row['spearman']:.4f}"
        lines.append(
            f"| `{row['left']}` | `{row['right']}` | {int(row['n'])} | {pearson} | {spearman} | {row['expectation']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is a US-only, county-level, modeled benchmark. It is useful as a",
            "supplementary external reasonableness check, not as global city-level",
            "outcome validation. A strong or weak correlation here should not be",
            "overstated because GCIRI is city-level and global, while FEMA/NRI is",
            "county/tract-level and US-specific.",
        ]
    )
    path = out_dir / "us_fema_nri_validation_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    out_dir = _build_output_dir()
    full = load_full_gciri()
    usa = full.loc[full["iso3"].eq("USA")].copy()
    if len(usa) != 185:
        raise ValueError(f"Expected 185 USA rows in current full5862 table, found {len(usa)}")

    meta = _fetch_item_metadata()
    records: list[dict[str, Any]] = []
    for i, row in enumerate(usa.to_dict("records"), start=1):
        attrs = _query_county_for_point(float(row["lon"]), float(row["lat"]))
        record = {
            "city": row["city"],
            "city_name": row["city_name"],
            "iso3": row["iso3"],
            "ucdb_id": row["ucdb_id"],
            "country_name": row["country_name"],
            "lat": row["lat"],
            "lon": row["lon"],
            "pop_2025": row["pop_2025"],
            "gciri": row["gciri"],
            "r_ph_city": row["r_ph_city"],
            "cap_abs": row["cap_abs"],
            "cap_resp": row["cap_resp"],
            "cap_rest": row["cap_rest"],
        }
        if attrs:
            record.update(
                {
                    "fema_county_fips": attrs.get("STCOFIPS"),
                    "fema_state": attrs.get("STATEABBRV"),
                    "fema_county": attrs.get("COUNTY"),
                    "fema_population_2020": attrs.get("POPULATION"),
                    "RISK_SCORE": attrs.get("RISK_SCORE"),
                    "RISK_RATNG": attrs.get("RISK_RATNG"),
                    "EAL_SCORE": attrs.get("EAL_SCORE"),
                    "EAL_RATNG": attrs.get("EAL_RATNG"),
                    "SOVI_SCORE": attrs.get("SOVI_SCORE"),
                    "SOVI_RATNG": attrs.get("SOVI_RATNG"),
                    "RESL_SCORE": attrs.get("RESL_SCORE"),
                    "RESL_RATNG": attrs.get("RESL_RATNG"),
                }
            )
        else:
            record.update(
                {
                    "fema_county_fips": None,
                    "fema_state": None,
                    "fema_county": None,
                    "fema_population_2020": None,
                    "RISK_SCORE": None,
                    "RISK_RATNG": None,
                    "EAL_SCORE": None,
                    "EAL_RATNG": None,
                    "SOVI_SCORE": None,
                    "SOVI_RATNG": None,
                    "RESL_SCORE": None,
                    "RESL_RATNG": None,
                }
            )
        records.append(record)
        if i % 25 == 0:
            print(f"queried {i}/{len(usa)} USA city points")
        time.sleep(0.04)

    joined = pd.DataFrame(records)
    for col in ["RISK_SCORE", "EAL_SCORE", "SOVI_SCORE", "RESL_SCORE"]:
        joined[col] = pd.to_numeric(joined[col], errors="coerce")
    corr = _corr_table(joined)

    joined_path = out_dir / "us_fema_nri_city_join.csv"
    corr_path = out_dir / "us_fema_nri_correlations.csv"
    source_path = out_dir / "us_fema_nri.source.json"
    joined.to_csv(joined_path, index=False)
    corr.to_csv(corr_path, index=False)
    summary_path = _write_summary(out_dir, joined, corr, meta)
    source_path.write_text(
        json.dumps(
            {
                "gciri_source": "_pipelines/01_baseline_gciri/outputs/v28_full5862_20260702_020421/analysis/full_gciri_with_meta.csv",
                "gciri_expected_rows": 5862,
                "usa_rows": len(usa),
                "fema_nri": meta,
                "fema_nri_service_query": SERVICE_URL,
                "outputs": {
                    "city_join": joined_path.relative_to(PROJECT_ROOT).as_posix(),
                    "correlations": corr_path.relative_to(PROJECT_ROOT).as_posix(),
                    "summary": summary_path.relative_to(PROJECT_ROOT).as_posix(),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"output_dir={out_dir}")
    print(f"matched={joined['fema_county_fips'].notna().sum()}/{len(joined)}")
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
