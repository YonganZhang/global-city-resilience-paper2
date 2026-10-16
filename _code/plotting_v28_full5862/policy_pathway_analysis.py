from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL_GCIRI_REL = Path(
    "_pipelines/01_baseline_gciri/outputs/"
    "v28_full5862_20260702_020421/analysis/full_gciri_with_meta.csv"
)
COUNTRY_META_REL = Path("_data/master_global_cities/master_global_cities_v1.csv")
EXPECTED_ROWS = 5862

REGION_LABELS = {
    1: "East & Southeast Asia",
    2: "South, Central & West Asia",
    3: "Americas",
    4: "Europe",
    5: "Africa",
    6: "Oceania",
}
WEAKNESS_ORDER = ["absorptive", "response", "restorative"]
SOURCE_ORDER = [
    "hazard-dominant",
    "capacity-dominant",
    "compound",
    "benchmark-saturated",
]
SOURCE_CODES = {
    "hazard-dominant": "H",
    "capacity-dominant": "C",
    "compound": "X",
    "benchmark-saturated": "M",
}
WEAKNESS_CODES = {"absorptive": "A", "response": "R", "restorative": "T"}


def _load_formula():
    formula_dir = PROJECT_ROOT / "_code" / "giri"
    sys.path.insert(0, str(formula_dir))
    from trapezoid import compute_trapezoid_giri

    return compute_trapezoid_giri


COMPUTE_GCIRI = _load_formula()


def _mode_or_nan(series: pd.Series):
    values = series.dropna()
    modes = values.mode()
    return modes.iloc[0] if len(modes) else np.nan


def load_policy_input() -> pd.DataFrame:
    score_path = PROJECT_ROOT / FULL_GCIRI_REL
    meta_path = PROJECT_ROOT / COUNTRY_META_REL
    if not score_path.exists():
        raise FileNotFoundError(score_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    frame = pd.read_csv(score_path)
    required = {
        "city",
        "city_name",
        "iso3",
        "ucdb_id",
        "country_name",
        "lat",
        "lon",
        "cap_abs",
        "cap_resp",
        "cap_rest",
        "r_ph_city",
        "f_city",
        "gciri",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{score_path} missing columns: {missing}")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(frame)}")
    if frame["city"].nunique() != EXPECTED_ROWS:
        raise ValueError("Expected one unique row-level city key per row")
    if frame["ucdb_id"].isna().any() or frame["iso3"].isna().any():
        raise ValueError("Policy analysis requires complete ucdb_id and iso3")

    raw_meta = pd.read_csv(meta_path)
    meta_required = {"country_iso3", "income_group", "region_id"}
    meta_missing = sorted(meta_required.difference(raw_meta.columns))
    if meta_missing:
        raise ValueError(f"{meta_path} missing columns: {meta_missing}")
    country_meta = (
        raw_meta.dropna(subset=["country_iso3"])
        .groupby("country_iso3")
        .agg(
            income_group=("income_group", _mode_or_nan),
            region_id=("region_id", _mode_or_nan),
        )
        .reset_index()
        .rename(columns={"country_iso3": "iso3"})
    )
    frame = frame.merge(country_meta, on="iso3", how="left", validate="many_to_one")
    if frame["region_id"].isna().any():
        missing_iso = sorted(frame.loc[frame["region_id"].isna(), "iso3"].unique())
        raise ValueError(f"Missing region metadata for: {missing_iso}")
    frame["region_id"] = frame["region_id"].astype(int)
    frame["region_group"] = frame["region_id"].map(REGION_LABELS)
    if frame["region_group"].isna().any():
        raise ValueError("Unknown region_id in policy analysis input")
    return frame


def _gciri(f_strength: float, response_deficit: float, restorative_strength: float, risk: float) -> float:
    return COMPUTE_GCIRI(
        F_city=1.0 - float(f_strength),
        cap_resp=float(response_deficit),
        cap_rest=float(restorative_strength),
        R_Ph=float(risk),
    )


def _regional_benchmarks(frame: pd.DataFrame, quantile: float) -> pd.DataFrame:
    if not 0.5 < quantile < 1.0:
        raise ValueError("benchmark_quantile must be between 0.5 and 1.0")
    lower = 1.0 - quantile
    return (
        frame.groupby("region_id")
        .agg(
            benchmark_f=("f_city", lambda values: values.quantile(quantile)),
            benchmark_response_deficit=("cap_resp", lambda values: values.quantile(lower)),
            benchmark_restorative=("cap_rest", lambda values: values.quantile(quantile)),
            benchmark_risk=("r_ph_city", lambda values: values.quantile(lower)),
        )
        .reset_index()
    )


def _compute_scenarios(frame: pd.DataFrame, benchmark_quantile: float) -> pd.DataFrame:
    work = frame.merge(
        _regional_benchmarks(frame, benchmark_quantile),
        on="region_id",
        how="left",
        validate="many_to_one",
    ).copy()

    baseline = []
    abs_10 = []
    resp_10 = []
    rest_10 = []
    abs_benchmark = []
    resp_benchmark = []
    rest_benchmark = []
    all_capacity_benchmark = []
    risk_benchmark = []

    for row in work.itertuples():
        base = _gciri(row.f_city, row.cap_resp, row.cap_rest, row.r_ph_city)
        baseline.append(base)
        abs_10.append(_gciri(min(1.0, row.f_city + 0.10), row.cap_resp, row.cap_rest, row.r_ph_city))
        resp_10.append(_gciri(row.f_city, max(0.0, row.cap_resp - 10.0), row.cap_rest, row.r_ph_city))
        rest_10.append(_gciri(row.f_city, row.cap_resp, min(100.0, row.cap_rest + 10.0), row.r_ph_city))

        improved_f = max(row.f_city, row.benchmark_f)
        improved_response = min(row.cap_resp, row.benchmark_response_deficit)
        improved_restorative = max(row.cap_rest, row.benchmark_restorative)
        lower_risk = min(row.r_ph_city, row.benchmark_risk)

        abs_benchmark.append(_gciri(improved_f, row.cap_resp, row.cap_rest, row.r_ph_city))
        resp_benchmark.append(_gciri(row.f_city, improved_response, row.cap_rest, row.r_ph_city))
        rest_benchmark.append(_gciri(row.f_city, row.cap_resp, improved_restorative, row.r_ph_city))
        all_capacity_benchmark.append(
            _gciri(improved_f, improved_response, improved_restorative, row.r_ph_city)
        )
        risk_benchmark.append(_gciri(row.f_city, row.cap_resp, row.cap_rest, lower_risk))

    work["gciri_recalculated"] = baseline
    work["delta_absorptive_10pp_raw"] = np.asarray(abs_10) - work["gciri_recalculated"]
    work["delta_response_10pp_raw"] = np.asarray(resp_10) - work["gciri_recalculated"]
    work["delta_restorative_10pp_raw"] = np.asarray(rest_10) - work["gciri_recalculated"]
    work["delta_absorptive_benchmark_raw"] = np.asarray(abs_benchmark) - work["gciri_recalculated"]
    work["delta_response_benchmark_raw"] = np.asarray(resp_benchmark) - work["gciri_recalculated"]
    work["delta_restorative_benchmark_raw"] = np.asarray(rest_benchmark) - work["gciri_recalculated"]
    work["capacity_benchmark_gciri"] = all_capacity_benchmark
    work["risk_benchmark_gciri"] = risk_benchmark
    work["capacity_headroom"] = np.maximum(
        0.0, work["capacity_benchmark_gciri"] - work["gciri_recalculated"]
    )
    work["hazard_burden"] = np.maximum(
        0.0, work["risk_benchmark_gciri"] - work["gciri_recalculated"]
    )
    work["total_modeled_headroom"] = work["capacity_headroom"] + work["hazard_burden"]
    work["hazard_burden_share"] = np.divide(
        work["hazard_burden"],
        work["total_modeled_headroom"],
        out=np.full(len(work), 0.5),
        where=work["total_modeled_headroom"].to_numpy() > 0,
    )
    return work


def _assign_constraint_source(
    frame: pd.DataFrame,
    dominance_share: float,
    headroom_floor: float,
) -> pd.Series:
    if not 0.5 < dominance_share < 1.0:
        raise ValueError("dominance_share must be between 0.5 and 1.0")
    limited = frame["total_modeled_headroom"] < headroom_floor
    hazard = frame["hazard_burden_share"] >= dominance_share
    capacity = frame["hazard_burden_share"] <= (1.0 - dominance_share)
    result = np.select(
        [limited, hazard, capacity],
        ["benchmark-saturated", "hazard-dominant", "capacity-dominant"],
        default="compound",
    )
    return pd.Series(result, index=frame.index, dtype="object")


def _assign_weakness(frame: pd.DataFrame) -> pd.Series:
    strengths = pd.DataFrame(
        {
            "absorptive": 100.0 - frame["cap_abs"],
            "response": 100.0 - frame["cap_resp"],
            "restorative": frame["cap_rest"],
        },
        index=frame.index,
    )
    return strengths.idxmin(axis=1)


def _policy_pathway(source: str, weakness: str) -> str:
    weakness_action = {
        "absorptive": "risk reduction and protective infrastructure",
        "response": "preparedness, emergency logistics and communication",
        "restorative": "recovery finance, social protection and reconstruction capacity",
    }[weakness]
    if source == "hazard-dominant":
        return f"residual-risk management plus {weakness_action}"
    if source == "capacity-dominant":
        return f"targeted {weakness_action}"
    if source == "compound":
        return f"sequenced multi-sector package led by {weakness_action}"
    return f"maintenance and stress testing, with monitoring of {weakness_action}"


def derive_policy_pathways(
    frame: pd.DataFrame | None = None,
    benchmark_quantile: float = 0.75,
    dominance_share: float = 2.0 / 3.0,
    headroom_floor: float = 2.0,
) -> pd.DataFrame:
    source = load_policy_input() if frame is None else frame.copy()
    work = _compute_scenarios(source, benchmark_quantile)
    reproduction_error = np.abs(work["gciri_recalculated"] - work["gciri"])
    if reproduction_error.max() > 0.02:
        raise ValueError(
            f"GCIRI formula reproduction failed: max error={reproduction_error.max():.6f}"
        )

    work["absorptive_strength"] = 100.0 - work["cap_abs"]
    work["response_strength"] = 100.0 - work["cap_resp"]
    work["restorative_strength"] = work["cap_rest"]
    work["weakest_dimension"] = _assign_weakness(work)
    work["constraint_source"] = _assign_constraint_source(
        work, dominance_share=dominance_share, headroom_floor=headroom_floor
    )
    work["policy_type_id"] = (
        work["constraint_source"].map(SOURCE_CODES)
        + "-"
        + work["weakest_dimension"].map(WEAKNESS_CODES)
    )
    work["policy_type"] = (
        work["constraint_source"].str.replace("-", " ", regex=False).str.title()
        + " / "
        + work["weakest_dimension"].str.title()
        + " weakness"
    )
    work["screening_pathway"] = [
        _policy_pathway(source_name, weakness)
        for source_name, weakness in zip(
            work["constraint_source"], work["weakest_dimension"]
        )
    ]
    work["restorative_nonmonotonic_flag"] = (
        work["delta_restorative_benchmark_raw"] < -0.01
    )
    work["benchmark_quantile"] = benchmark_quantile
    work["dominance_share_threshold"] = dominance_share
    work["headroom_floor"] = headroom_floor
    return work


def summarize_region_weakness(city: pd.DataFrame) -> pd.DataFrame:
    table = (
        pd.crosstab(city["region_group"], city["weakest_dimension"], normalize="index")
        .mul(100.0)
        .reindex(columns=WEAKNESS_ORDER, fill_value=0.0)
    )
    counts = city.groupby("region_group").size().rename("n")
    return table.join(counts).reset_index()


def summarize_city_types(city: pd.DataFrame) -> pd.DataFrame:
    return (
        city.groupby(
            ["constraint_source", "weakest_dimension", "policy_type_id", "policy_type"],
            observed=True,
        )
        .agg(
            n=("city", "size"),
            share_pct=("city", lambda values: len(values) / len(city) * 100.0),
            mean_gciri=("gciri", "mean"),
            median_gciri=("gciri", "median"),
            mean_capacity_headroom=("capacity_headroom", "mean"),
            mean_hazard_burden=("hazard_burden", "mean"),
            mean_total_headroom=("total_modeled_headroom", "mean"),
            restorative_nonmonotonic_share=(
                "restorative_nonmonotonic_flag",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(["constraint_source", "weakest_dimension"])
    )


def summarize_scenarios(city: pd.DataFrame) -> pd.DataFrame:
    scenario_columns = {
        "absorptive +10 pp": "delta_absorptive_10pp_raw",
        "response +10 pp": "delta_response_10pp_raw",
        "restorative +10 pp": "delta_restorative_10pp_raw",
        "absorptive to regional benchmark": "delta_absorptive_benchmark_raw",
        "response to regional benchmark": "delta_response_benchmark_raw",
        "restorative to regional benchmark": "delta_restorative_benchmark_raw",
        "all capacities to regional benchmark": "capacity_headroom",
        "physical risk to regional benchmark": "hazard_burden",
    }
    rows = []
    for label, column in scenario_columns.items():
        values = city[column].to_numpy(dtype=float)
        rows.append(
            {
                "scenario": label,
                "n": len(values),
                "mean_delta_gciri": values.mean(),
                "median_delta_gciri": np.median(values),
                "q25_delta_gciri": np.quantile(values, 0.25),
                "q75_delta_gciri": np.quantile(values, 0.75),
                "negative_n": int(np.sum(values < -0.01)),
            }
        )
    return pd.DataFrame(rows)


def summarize_country_portfolios(city: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iso3, group in city.groupby("iso3"):
        source_share = group["constraint_source"].value_counts(normalize=True)
        weakness_share = group["weakest_dimension"].value_counts(normalize=True)
        top_source = source_share.index[0]
        top_source_share = float(source_share.iloc[0])
        top_weakness = weakness_share.index[0]
        top_weakness_share = float(weakness_share.iloc[0])
        n_cities = len(group)

        if n_cities < 5:
            portfolio = "sparse-evidence"
        elif (
            group["gciri"].median() >= 70
            and source_share.get("benchmark-saturated", 0.0) >= 0.50
        ):
            portfolio = "high-resilience maintenance"
        elif top_source_share >= 0.50:
            portfolio = top_source
        else:
            portfolio = "mixed portfolio"

        weakness_portfolio = (
            top_weakness if top_weakness_share >= 0.50 else "mixed weakness"
        )
        rows.append(
            {
                "iso3": iso3,
                "country_name": group["country_name"].iloc[0],
                "n_cities": n_cities,
                "mean_gciri": group["gciri"].mean(),
                "median_gciri": group["gciri"].median(),
                "country_portfolio": portfolio,
                "country_weakness_portfolio": weakness_portfolio,
                "top_constraint_source": top_source,
                "top_constraint_share": top_source_share * 100.0,
                "top_weakness": top_weakness,
                "top_weakness_share": top_weakness_share * 100.0,
                **{
                    f"share_{source_name.replace('-', '_')}": source_share.get(
                        source_name, 0.0
                    )
                    * 100.0
                    for source_name in SOURCE_ORDER
                },
                **{
                    f"share_weak_{weakness}": weakness_share.get(weakness, 0.0)
                    * 100.0
                    for weakness in WEAKNESS_ORDER
                },
            }
        )
    return pd.DataFrame(rows).sort_values("n_cities", ascending=False)


def sensitivity_analysis(base: pd.DataFrame) -> pd.DataFrame:
    base_labels = base["constraint_source"].to_numpy()
    raw = base[
        [
            "city",
            "city_name",
            "iso3",
            "ucdb_id",
            "country_name",
            "lat",
            "lon",
            "cap_abs",
            "cap_resp",
            "cap_rest",
            "r_ph_city",
            "f_city",
            "gciri",
            "income_group",
            "region_id",
            "region_group",
        ]
    ].copy()
    settings = [
        (0.70, 2.0 / 3.0, 2.0),
        (0.80, 2.0 / 3.0, 2.0),
        (0.75, 0.60, 2.0),
        (0.75, 0.75, 2.0),
        (0.75, 2.0 / 3.0, 1.0),
        (0.75, 2.0 / 3.0, 5.0),
    ]
    rows = []
    for quantile, dominance, floor in settings:
        scenario = _compute_scenarios(raw, quantile)
        labels = _assign_constraint_source(scenario, dominance, floor).to_numpy()
        counts = pd.Series(labels).value_counts()
        rows.append(
            {
                "benchmark_quantile": quantile,
                "dominance_share_threshold": dominance,
                "headroom_floor": floor,
                "adjusted_rand_vs_primary": adjusted_rand_score(base_labels, labels),
                **{
                    f"n_{source_name.replace('-', '_')}": int(
                        counts.get(source_name, 0)
                    )
                    for source_name in SOURCE_ORDER
                },
            }
        )
    return pd.DataFrame(rows)


def policy_dictionary() -> pd.DataFrame:
    rows = []
    for source in SOURCE_ORDER:
        for weakness in WEAKNESS_ORDER:
            rows.append(
                {
                    "policy_type_id": f"{SOURCE_CODES[source]}-{WEAKNESS_CODES[weakness]}",
                    "constraint_source": source,
                    "weakest_dimension": weakness,
                    "screening_pathway": _policy_pathway(source, weakness),
                    "claim_boundary": (
                        "Formula-based screening priority; not a causal intervention "
                        "effect or cost-benefit estimate."
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_analysis(out_dir: Path, city: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    city_out = city[
        [
            "city",
            "city_name",
            "iso3",
            "ucdb_id",
            "country_name",
            "lat",
            "lon",
            "region_group",
            "income_group",
            "gciri",
            "absorptive_strength",
            "response_strength",
            "restorative_strength",
            "weakest_dimension",
            "constraint_source",
            "policy_type_id",
            "policy_type",
            "screening_pathway",
            "capacity_headroom",
            "hazard_burden",
            "total_modeled_headroom",
            "hazard_burden_share",
            "delta_absorptive_10pp_raw",
            "delta_response_10pp_raw",
            "delta_restorative_10pp_raw",
            "delta_absorptive_benchmark_raw",
            "delta_response_benchmark_raw",
            "delta_restorative_benchmark_raw",
            "capacity_benchmark_gciri",
            "risk_benchmark_gciri",
            "restorative_nonmonotonic_flag",
        ]
    ].copy()
    city_out.to_csv(out_dir / "city_policy_typology.csv", index=False)

    global_weakness = (
        city["weakest_dimension"]
        .value_counts()
        .reindex(WEAKNESS_ORDER, fill_value=0)
        .rename_axis("weakest_dimension")
        .reset_index(name="n")
    )
    global_weakness["share_pct"] = global_weakness["n"] / len(city) * 100.0
    global_weakness.to_csv(out_dir / "global_weakness_summary.csv", index=False)

    region_weakness = summarize_region_weakness(city)
    region_weakness.to_csv(out_dir / "region_weakness_composition.csv", index=False)
    city_types = summarize_city_types(city)
    city_types.to_csv(out_dir / "city_policy_type_summary.csv", index=False)
    source_summary = (
        city["constraint_source"]
        .value_counts()
        .reindex(SOURCE_ORDER, fill_value=0)
        .rename_axis("constraint_source")
        .reset_index(name="n")
    )
    source_summary["share_pct"] = source_summary["n"] / len(city) * 100.0
    source_summary.to_csv(out_dir / "constraint_source_summary.csv", index=False)
    summarize_scenarios(city).to_csv(out_dir / "scenario_summary.csv", index=False)
    country = summarize_country_portfolios(city)
    country.to_csv(out_dir / "country_policy_portfolios.csv", index=False)
    (
        country.groupby("country_portfolio")
        .agg(n_countries=("iso3", "size"), n_cities=("n_cities", "sum"))
        .reset_index()
        .sort_values("n_cities", ascending=False)
        .to_csv(out_dir / "country_portfolio_summary.csv", index=False)
    )
    sensitivity_analysis(city).to_csv(out_dir / "sensitivity_summary.csv", index=False)
    policy_dictionary().to_csv(out_dir / "policy_pathway_dictionary.csv", index=False)

    reproduction = np.abs(city["gciri_recalculated"] - city["gciri"])
    restorative_nonmonotonic = int(city["restorative_nonmonotonic_flag"].sum())
    lines = [
        "# GCIRI Weakness-to-Policy Pathway Analysis",
        "",
        f"- Current-data rows: {len(city)}",
        f"- Unique row-level city keys: {city['city'].nunique()}",
        f"- Maximum GCIRI formula reproduction error: {reproduction.max():.6f}",
        f"- Benchmark quantile: {city['benchmark_quantile'].iloc[0]:.2f}",
        f"- Dominance-share threshold: {city['dominance_share_threshold'].iloc[0]:.4f}",
        f"- Limited-headroom threshold: {city['headroom_floor'].iloc[0]:.2f} GCIRI points",
        "",
        "## Interpretation",
        "",
        "The analysis separates the direction-corrected weakest capacity from the "
        "modeled source of headroom. Capacity headroom is the GCIRI change when all "
        "modifiable capacity inputs move to the regional upper-quartile benchmark "
        "while physical risk is held fixed. Hazard burden is the GCIRI change when "
        "physical risk moves to the regional lower-quartile benchmark while all "
        "capacity inputs are held fixed.",
        "",
        "Constraint-source labels use the share of total modeled headroom. They are "
        "screening categories, not causal effects or cost-benefit estimates.",
        "",
        "## Formula Boundary",
        "",
        f"Raising restorative capacity to the regional benchmark produced a negative "
        f"raw GCIRI delta for {restorative_nonmonotonic} cities. These rows remain "
        "visible in the source data and are not interpreted as evidence that recovery "
        "investment is harmful. They expose a non-monotonic sensitivity of the inherited "
        "trapezoid formula and bound policy claims based on score changes.",
        "",
        "## Sources",
        "",
        f"- `{FULL_GCIRI_REL.as_posix()}`",
        f"- `{COUNTRY_META_REL.as_posix()}`",
        "- `_code/giri/trapezoid.py`",
        "",
    ]
    (out_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "analysis": "gciri_weakness_to_policy_pathways",
        "rows": len(city),
        "source": FULL_GCIRI_REL.as_posix(),
        "country_metadata": COUNTRY_META_REL.as_posix(),
        "formula": "_code/giri/trapezoid.py",
        "benchmark_quantile": float(city["benchmark_quantile"].iloc[0]),
        "dominance_share_threshold": float(
            city["dominance_share_threshold"].iloc[0]
        ),
        "headroom_floor": float(city["headroom_floor"].iloc[0]),
        "policy_claim_boundary": (
            "Formula-based screening; no causal effect or cost-benefit claim."
        ),
    }
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--benchmark-quantile", type=float, default=0.75)
    parser.add_argument("--dominance-share", type=float, default=2.0 / 3.0)
    parser.add_argument("--headroom-floor", type=float, default=2.0)
    args = parser.parse_args()
    if args.out:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (
            PROJECT_ROOT
            / "_paper"
            / "gciri_full5862_spine_20260704"
            / f"policy_pathway_analysis_{stamp}"
        )
    city = derive_policy_pathways(
        benchmark_quantile=args.benchmark_quantile,
        dominance_share=args.dominance_share,
        headroom_floor=args.headroom_floor,
    )
    write_analysis(out_dir, city)
    print(out_dir)


if __name__ == "__main__":
    main()
