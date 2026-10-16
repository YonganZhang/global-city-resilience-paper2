from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

try:
    from ._shared import (
        EXPECTED_FULL_ROWS,
        FULL_GCIRI_REL,
        PROJECT_ROOT,
        load_country_giri,
        load_full_gciri,
    )
except ImportError:
    from _shared import (
        EXPECTED_FULL_ROWS,
        FULL_GCIRI_REL,
        PROJECT_ROOT,
        load_country_giri,
        load_full_gciri,
    )


COUNTRY_META_REL = Path("_data/master_global_cities/master_global_cities_v1.csv")
DEFAULT_OUTPUT_PARENT = Path("_paper/gciri_full5862_spine_20260704")

MODE_ORDER = [
    "balanced high-resilience",
    "imbalanced high-resilience",
    "mixed-deficit transition",
    "single-bottleneck transition",
    "compound low-resilience",
    "single-deficit low-resilience",
]
MODE_CODES = {mode: f"M{index + 1}" for index, mode in enumerate(MODE_ORDER)}
MODE_PATHWAYS = {
    "balanced high-resilience": "Maintain strengths and stress-test compound risks",
    "imbalanced high-resilience": "Protect overall gains and repair the weaker capacity",
    "mixed-deficit transition": "Coordinate improvements across multiple capacities",
    "single-bottleneck transition": "Remove the dominant capacity bottleneck first",
    "compound low-resilience": "Build foundational capacity across all three dimensions",
    "single-deficit low-resilience": "Repair the severe deficit while protecting remaining capacity",
}

BASE_PARAMETERS = {
    "low_threshold": 40.0,
    "high_threshold": 70.0,
    "weakness_margin": 8.0,
    "balance_spread": 20.0,
}
SENSITIVITY_VALUES = {
    "low_threshold": (38.0, 40.0, 42.0),
    "high_threshold": (68.0, 70.0, 72.0),
    "weakness_margin": (6.0, 8.0, 10.0),
    "balance_spread": (18.0, 20.0, 22.0),
}
CLUSTER_FEATURES = [
    "gciri",
    "absorptive_strength",
    "response_strength",
    "restorative_strength",
]


def load_country_metadata() -> pd.DataFrame:
    path = PROJECT_ROOT / COUNTRY_META_REL
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    required = {
        "country_iso3",
        "income_group",
        "region_id",
        "region_name",
        "subregion_name",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    rows: list[dict[str, object]] = []
    for iso3, group in raw.dropna(subset=["country_iso3"]).groupby("country_iso3"):
        row: dict[str, object] = {"iso3": iso3}
        for column in [
            "income_group",
            "region_id",
            "region_name",
            "subregion_name",
        ]:
            values = group[column].dropna()
            row[column] = values.mode().iloc[0] if len(values) else np.nan
        rows.append(row)
    metadata = pd.DataFrame(rows)
    metadata["region_id"] = pd.to_numeric(
        metadata["region_id"], errors="coerce"
    ).astype("Int64")
    region_labels = {
        1: "East & Southeast Asia",
        2: "South, Central & West Asia",
        3: "Americas",
        4: "Europe",
        5: "Africa",
        6: "Oceania",
    }
    metadata["region_group"] = metadata["region_id"].map(region_labels)
    return metadata


def compute_development_modes(
    city: pd.DataFrame,
    *,
    low_threshold: float = 40.0,
    high_threshold: float = 70.0,
    weakness_margin: float = 8.0,
    balance_spread: float = 20.0,
) -> pd.DataFrame:
    if low_threshold >= high_threshold:
        raise ValueError("low_threshold must be lower than high_threshold")
    required = {
        "gciri",
        "absorptive_strength",
        "response_strength",
        "restorative_strength",
    }
    missing = sorted(required.difference(city.columns))
    if missing:
        raise ValueError(f"Mode classification missing columns: {missing}")

    result = city.copy()
    strengths = result[
        ["absorptive_strength", "response_strength", "restorative_strength"]
    ].to_numpy(dtype=float)
    deficits = 100.0 - strengths
    ordered = np.sort(deficits, axis=1)
    weakness_gap = ordered[:, -1] - ordered[:, -2]
    mixed = weakness_gap < weakness_margin
    weakness_index = deficits.argmax(axis=1)
    weakness_names = np.array(["absorptive", "response", "restorative"])
    dominant = weakness_names[weakness_index].astype(object)
    dominant[mixed] = "mixed"

    capacity_spread = strengths.max(axis=1) - strengths.min(axis=1)
    score = result["gciri"].to_numpy(dtype=float)
    high = score >= high_threshold
    low = score < low_threshold
    transition = ~(high | low)

    modes = np.empty(len(result), dtype=object)
    modes[high & (capacity_spread < balance_spread)] = MODE_ORDER[0]
    modes[high & (capacity_spread >= balance_spread)] = MODE_ORDER[1]
    modes[transition & mixed] = MODE_ORDER[2]
    modes[transition & ~mixed] = MODE_ORDER[3]
    modes[low & mixed] = MODE_ORDER[4]
    modes[low & ~mixed] = MODE_ORDER[5]

    result["dominant_weakness"] = dominant
    result["weakness_gap"] = weakness_gap
    result["capacity_spread"] = capacity_spread
    result["development_mode"] = modes
    result["mode_code"] = pd.Series(modes, index=result.index).map(MODE_CODES)
    result["screening_pathway"] = pd.Series(modes, index=result.index).map(
        MODE_PATHWAYS
    )
    if result["development_mode"].isna().any():
        raise ValueError("Mode classification produced missing labels")
    return result


def load_mode_city_data() -> pd.DataFrame:
    city = load_full_gciri().copy()
    city = city.merge(
        load_country_metadata(), on="iso3", how="left", validate="many_to_one"
    )
    country = load_country_giri().copy()
    country_columns = ["iso3", "official_giri", "gciri_mean", "n_cities"]
    city = city.merge(
        country[country_columns],
        on="iso3",
        how="left",
        validate="many_to_one",
    )
    city["north_south_proxy"] = pd.Series(
        np.nan, index=city.index, dtype="object"
    )
    city.loc[
        city["income_group"].eq("High income"), "north_south_proxy"
    ] = "North proxy"
    city.loc[
        city["income_group"].notna()
        & ~city["income_group"].eq("High income"),
        "north_south_proxy",
    ] = "South proxy"
    city["absorptive_strength"] = 100.0 - city["cap_abs"]
    city["response_strength"] = 100.0 - city["cap_resp"]
    city["restorative_strength"] = city["cap_rest"]
    strength_columns = [
        "absorptive_strength",
        "response_strength",
        "restorative_strength",
    ]
    city["weakest_dimension"] = city[strength_columns].idxmin(axis=1).map(
        {
            "absorptive_strength": "absorptive",
            "response_strength": "response",
            "restorative_strength": "restorative",
        }
    )
    city = compute_development_modes(city, **BASE_PARAMETERS)

    if len(city) != EXPECTED_FULL_ROWS:
        raise ValueError(
            f"Mode analysis requires {EXPECTED_FULL_ROWS} rows, found {len(city)}"
        )
    if city["city"].nunique() != EXPECTED_FULL_ROWS:
        raise ValueError("Mode analysis requires 5862 unique row-level city keys")
    required_nonmissing = [
        "city",
        "ucdb_id",
        "iso3",
        "lat",
        "lon",
        "gciri",
        *strength_columns,
        "weakest_dimension",
        "development_mode",
    ]
    missing_counts = city[required_nonmissing].isna().sum()
    if int(missing_counts.sum()):
        raise ValueError(
            f"Mode analysis contains missing required values: "
            f"{missing_counts[missing_counts > 0].to_dict()}"
        )
    return city


def compute_mode_summary(city: pd.DataFrame) -> pd.DataFrame:
    summary = (
        city.groupby("development_mode", observed=True)
        .agg(
            n=("city", "size"),
            mean_gciri=("gciri", "mean"),
            median_gciri=("gciri", "median"),
            mean_absorptive=("absorptive_strength", "mean"),
            mean_response=("response_strength", "mean"),
            mean_restorative=("restorative_strength", "mean"),
            mean_capacity_spread=("capacity_spread", "mean"),
            high_income_share=(
                "income_group",
                lambda values: values.eq("High income").mean() * 100.0,
            ),
        )
        .reindex(MODE_ORDER)
        .reset_index()
    )
    summary["mode_code"] = summary["development_mode"].map(MODE_CODES)
    summary["share_pct"] = summary["n"] / len(city) * 100.0
    summary["screening_pathway"] = summary["development_mode"].map(
        MODE_PATHWAYS
    )
    return summary[
        [
            "mode_code",
            "development_mode",
            "n",
            "share_pct",
            "mean_gciri",
            "median_gciri",
            "mean_absorptive",
            "mean_response",
            "mean_restorative",
            "mean_capacity_spread",
            "high_income_share",
            "screening_pathway",
        ]
    ]


def compute_group_mode_composition(
    city: pd.DataFrame,
    group_column: str,
    group_order: list[str] | None = None,
) -> pd.DataFrame:
    if group_column not in city.columns:
        raise ValueError(f"Unknown group column: {group_column}")
    subset = city.dropna(subset=[group_column]).copy()
    counts = pd.crosstab(subset[group_column], subset["development_mode"])
    counts = counts.reindex(columns=MODE_ORDER, fill_value=0)
    if group_order is not None:
        counts = counts.reindex(group_order).fillna(0)
    shares = counts.div(counts.sum(axis=1), axis=0).mul(100.0)
    tidy = (
        shares.rename_axis(index=group_column, columns="development_mode")
        .stack(future_stack=True)
        .rename("share_pct")
        .reset_index()
    )
    count_tidy = (
        counts.rename_axis(index=group_column, columns="development_mode")
        .stack(future_stack=True)
        .rename("n")
        .reset_index()
    )
    tidy = tidy.merge(
        count_tidy,
        on=[group_column, "development_mode"],
        how="left",
        validate="one_to_one",
    )
    tidy["mode_code"] = tidy["development_mode"].map(MODE_CODES)
    return tidy


def compute_country_mode_portfolios(city: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (iso3, country_name), group in city.groupby(
        ["iso3", "country_name"], dropna=False
    ):
        shares = (
            group["development_mode"]
            .value_counts(normalize=True)
            .reindex(MODE_ORDER, fill_value=0.0)
        )
        positive = shares[shares > 0].to_numpy(dtype=float)
        entropy = float(-(positive * np.log(positive)).sum() / np.log(len(MODE_ORDER)))
        dominant_mode = str(shares.idxmax())
        record: dict[str, object] = {
            "iso3": iso3,
            "country_name": country_name,
            "n_cities": len(group),
            "mean_gciri": float(group["gciri"].mean()),
            "median_gciri": float(group["gciri"].median()),
            "official_giri": float(group["official_giri"].iloc[0]),
            "income_group": group["income_group"].dropna().iloc[0]
            if group["income_group"].notna().any()
            else np.nan,
            "region_group": group["region_group"].dropna().iloc[0]
            if group["region_group"].notna().any()
            else np.nan,
            "dominant_mode": dominant_mode,
            "dominant_mode_code": MODE_CODES[dominant_mode],
            "dominant_mode_share_pct": float(shares.max() * 100.0),
            "mode_diversity": int((shares > 0).sum()),
            "normalized_mode_entropy": entropy,
        }
        for mode in MODE_ORDER:
            record[f"share_{MODE_CODES[mode].lower()}"] = float(
                shares[mode] * 100.0
            )
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["n_cities", "mean_gciri"], ascending=[False, False]
    )


def compute_sensitivity_grid(
    city: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = compute_development_modes(city, **BASE_PARAMETERS)
    baseline_labels = baseline["mode_code"].to_numpy()
    configurations = itertools.product(
        SENSITIVITY_VALUES["low_threshold"],
        SENSITIVITY_VALUES["high_threshold"],
        SENSITIVITY_VALUES["weakness_margin"],
        SENSITIVITY_VALUES["balance_spread"],
    )
    grid_records: list[dict[str, object]] = []
    assignments: list[np.ndarray] = []
    for index, (low, high, margin, spread) in enumerate(configurations, start=1):
        classified = compute_development_modes(
            city,
            low_threshold=low,
            high_threshold=high,
            weakness_margin=margin,
            balance_spread=spread,
        )
        labels = classified["mode_code"].to_numpy()
        assignments.append(labels)
        grid_records.append(
            {
                "configuration_id": f"S{index:02d}",
                "low_threshold": low,
                "high_threshold": high,
                "weakness_margin": margin,
                "balance_spread": spread,
                "adjusted_rand_vs_baseline": adjusted_rand_score(
                    baseline_labels, labels
                ),
                "city_retention_vs_baseline": float(
                    np.mean(labels == baseline_labels)
                ),
                "is_baseline": bool(
                    low == BASE_PARAMETERS["low_threshold"]
                    and high == BASE_PARAMETERS["high_threshold"]
                    and margin == BASE_PARAMETERS["weakness_margin"]
                    and spread == BASE_PARAMETERS["balance_spread"]
                ),
            }
        )
    assignment_matrix = np.vstack(assignments)
    retention = np.mean(assignment_matrix == baseline_labels, axis=0)
    city_stability = baseline[
        [
            "city",
            "city_name",
            "iso3",
            "ucdb_id",
            "gciri",
            "mode_code",
            "development_mode",
            "weakness_gap",
            "capacity_spread",
        ]
    ].copy()
    city_stability["mode_retention_rate"] = retention
    city_stability["stable_at_75pct"] = retention >= 0.75
    city_stability["stable_at_50pct"] = retention >= 0.50
    return pd.DataFrame(grid_records), city_stability


def compute_cluster_validation(
    city: pd.DataFrame,
    *,
    random_state: int = 20260718,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = city[CLUSTER_FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    rule_labels = city["mode_code"].to_numpy()
    records: list[dict[str, object]] = []
    selected_model: KMeans | None = None
    for cluster_count in range(4, 9):
        model = KMeans(
            n_clusters=cluster_count,
            n_init=30,
            random_state=random_state,
        ).fit(scaled)
        sample_size = min(3000, len(city))
        records.append(
            {
                "cluster_count": cluster_count,
                "silhouette": silhouette_score(
                    scaled,
                    model.labels_,
                    sample_size=sample_size,
                    random_state=random_state,
                ),
                "adjusted_rand_vs_rule_modes": adjusted_rand_score(
                    rule_labels, model.labels_
                ),
                "normalized_mutual_info_vs_rule_modes": normalized_mutual_info_score(
                    rule_labels, model.labels_
                ),
                "selected_six_cluster_check": cluster_count == 6,
            }
        )
        if cluster_count == 6:
            selected_model = model
    if selected_model is None:
        raise RuntimeError("Six-cluster validation model was not fitted")

    centroids = pd.DataFrame(
        scaler.inverse_transform(selected_model.cluster_centers_),
        columns=CLUSTER_FEATURES,
    ).sort_values("gciri", ascending=False)
    centroids.insert(0, "cluster_rank_by_gciri", np.arange(1, len(centroids) + 1))
    return pd.DataFrame(records), centroids


def compute_bootstrap_cluster_stability(
    city: pd.DataFrame,
    *,
    repeats: int = 60,
    sample_fraction: float = 0.80,
    random_state: int = 20260718,
) -> pd.DataFrame:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    features = city[CLUSTER_FEATURES].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(features)
    rule_labels = city["mode_code"].to_numpy()
    base_model = KMeans(
        n_clusters=6, n_init=50, random_state=random_state
    ).fit(scaled)
    rng = np.random.default_rng(random_state)
    sample_size = int(round(sample_fraction * len(city)))
    records: list[dict[str, object]] = []
    for repeat in range(repeats):
        indices = rng.choice(len(city), size=sample_size, replace=True)
        model = KMeans(
            n_clusters=6,
            n_init=10,
            random_state=random_state + repeat + 1,
        ).fit(scaled[indices])
        labels = model.predict(scaled)
        records.append(
            {
                "repeat": repeat + 1,
                "sample_fraction": sample_fraction,
                "adjusted_rand_vs_full_k6": adjusted_rand_score(
                    base_model.labels_, labels
                ),
                "adjusted_rand_vs_rule_modes": adjusted_rand_score(
                    rule_labels, labels
                ),
                "normalized_mutual_info_vs_rule_modes": normalized_mutual_info_score(
                    rule_labels, labels
                ),
            }
        )
    return pd.DataFrame(records)


def save_mode_analysis(
    output_dir: Path,
    *,
    bootstrap_repeats: int = 60,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    city = load_mode_city_data()
    mode_summary = compute_mode_summary(city)
    north_south = compute_group_mode_composition(
        city,
        "north_south_proxy",
        ["North proxy", "South proxy"],
    )
    region_order = [
        "East & Southeast Asia",
        "South, Central & West Asia",
        "Americas",
        "Europe",
        "Africa",
        "Oceania",
    ]
    region = compute_group_mode_composition(
        city,
        "region_group",
        region_order,
    )
    country = compute_country_mode_portfolios(city)
    sensitivity, city_stability = compute_sensitivity_grid(city)
    cluster_validation, cluster_centroids = compute_cluster_validation(city)
    bootstrap = compute_bootstrap_cluster_stability(
        city, repeats=bootstrap_repeats
    )
    policy = mode_summary[
        ["mode_code", "development_mode", "screening_pathway"]
    ].copy()

    outputs = {
        "city_mode_table": "city_mode_table.csv",
        "mode_summary": "mode_summary.csv",
        "north_south_composition": "north_south_mode_composition.csv",
        "region_composition": "region_mode_composition.csv",
        "country_portfolios": "country_mode_portfolios.csv",
        "sensitivity_grid": "sensitivity_grid.csv",
        "city_rule_stability": "city_rule_stability.csv",
        "cluster_validation": "cluster_validation.csv",
        "cluster_centroids": "cluster_k6_centroids.csv",
        "bootstrap_cluster_stability": "bootstrap_cluster_stability.csv",
        "policy_dictionary": "policy_pathway_dictionary.csv",
    }
    tables = {
        "city_mode_table": city,
        "mode_summary": mode_summary,
        "north_south_composition": north_south,
        "region_composition": region,
        "country_portfolios": country,
        "sensitivity_grid": sensitivity,
        "city_rule_stability": city_stability,
        "cluster_validation": cluster_validation,
        "cluster_centroids": cluster_centroids,
        "bootstrap_cluster_stability": bootstrap,
        "policy_dictionary": policy,
    }
    for key, table in tables.items():
        table.to_csv(output_dir / outputs[key], index=False)

    north = north_south.loc[
        north_south["north_south_proxy"] == "North proxy"
    ].set_index("mode_code")["share_pct"]
    south = north_south.loc[
        north_south["north_south_proxy"] == "South proxy"
    ].set_index("mode_code")["share_pct"]
    sensitivity_stats = {
        "median_adjusted_rand": float(
            sensitivity["adjusted_rand_vs_baseline"].median()
        ),
        "minimum_adjusted_rand": float(
            sensitivity["adjusted_rand_vs_baseline"].min()
        ),
        "median_city_retention": float(
            sensitivity["city_retention_vs_baseline"].median()
        ),
        "minimum_city_retention": float(
            sensitivity["city_retention_vs_baseline"].min()
        ),
        "cities_stable_at_75pct": float(
            city_stability["stable_at_75pct"].mean()
        ),
        "cities_stable_at_50pct": float(
            city_stability["stable_at_50pct"].mean()
        ),
    }
    k6 = cluster_validation.loc[
        cluster_validation["cluster_count"] == 6
    ].iloc[0]
    bootstrap_quantiles = bootstrap["adjusted_rand_vs_full_k6"].quantile(
        [0.05, 0.50, 0.95]
    )
    findings = {
        "global_mode_shares": {
            row["mode_code"]: float(row["share_pct"])
            for _, row in mode_summary.iterrows()
        },
        "north_high_mode_share": float(north["M1"] + north["M2"]),
        "south_high_mode_share": float(south["M1"] + south["M2"]),
        "north_low_mode_share": float(north["M5"] + north["M6"]),
        "south_low_mode_share": float(south["M5"] + south["M6"]),
        "sensitivity": sensitivity_stats,
        "six_cluster_silhouette": float(k6["silhouette"]),
        "six_cluster_adjusted_rand_vs_rule_modes": float(
            k6["adjusted_rand_vs_rule_modes"]
        ),
        "bootstrap_k6_adjusted_rand_quantiles": {
            "p05": float(bootstrap_quantiles.loc[0.05]),
            "median": float(bootstrap_quantiles.loc[0.50]),
            "p95": float(bootstrap_quantiles.loc[0.95]),
        },
    }
    manifest = {
        "schema_version": "gciri-development-mode-analysis/v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "truth_source": FULL_GCIRI_REL.as_posix(),
        "source_rows": len(city),
        "unique_city_keys": int(city["city"].nunique()),
        "classification": {
            "type": "transparent decision-oriented typology",
            "base_parameters": BASE_PARAMETERS,
            "mode_order": MODE_ORDER,
            "mode_codes": MODE_CODES,
        },
        "robustness": {
            "focused_parameter_grid": SENSITIVITY_VALUES,
            "grid_size": len(sensitivity),
            "alternative_cluster_counts": [4, 5, 6, 7, 8],
            "bootstrap_repeats": bootstrap_repeats,
            "bootstrap_sample_fraction": 0.80,
            "random_state": 20260718,
        },
        "findings": findings,
        "claim_boundary": [
            "The six modes are rule-defined planning types, not natural clusters.",
            "Mode-specific pathways are screening priorities, not causal or optimal policies.",
            "Income-defined North and South groups are transparent proxies.",
        ],
        "outputs": outputs,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# Six-mode development typology validation",
        "",
        f"- Current source rows: {len(city):,}.",
        "- Classification: transparent decision-oriented typology.",
        "- The labels are not presented as natural clusters.",
        "",
        "## Global composition",
        "",
    ]
    for _, row in mode_summary.iterrows():
        report_lines.append(
            f"- {row['mode_code']} {row['development_mode']}: "
            f"{int(row['n']):,} cities ({row['share_pct']:.2f}%)."
        )
    report_lines.extend(
        [
            "",
            "## Structural divide",
            "",
            f"- High-resilience modes (M1-M2): {findings['north_high_mode_share']:.1f}% "
            f"in the North proxy and {findings['south_high_mode_share']:.1f}% "
            "in the South proxy.",
            f"- Low-resilience modes (M5-M6): {findings['north_low_mode_share']:.1f}% "
            f"in the North proxy and {findings['south_low_mode_share']:.1f}% "
            "in the South proxy.",
            "",
            "## Robustness",
            "",
            f"- Focused 81-setting sensitivity grid: median adjusted Rand index "
            f"{sensitivity_stats['median_adjusted_rand']:.3f}; minimum "
            f"{sensitivity_stats['minimum_adjusted_rand']:.3f}.",
            f"- Median city-label retention is "
            f"{sensitivity_stats['median_city_retention'] * 100:.1f}%; "
            f"{sensitivity_stats['cities_stable_at_50pct'] * 100:.1f}% of cities "
            "retain their baseline label in at least half of settings.",
            f"- A six-cluster K-means check has silhouette {k6['silhouette']:.3f} "
            f"and adjusted Rand agreement {k6['adjusted_rand_vs_rule_modes']:.3f} "
            "with the rule modes.",
            f"- Resampled six-cluster solutions are internally stable "
            f"(median adjusted Rand {bootstrap_quantiles.loc[0.50]:.3f}) but "
            "do not reproduce the rule typology closely.",
            "",
            "## Interpretation boundary",
            "",
            "The robustness evidence supports using the six labels as transparent "
            "planning types. It does not support describing them as six natural, "
            "data-emergent clusters. Borderline cities should be read as screening "
            "cases, and local policy still requires feasibility, cost and observed "
            "intervention evidence.",
            "",
        ]
    )
    (output_dir / "analysis_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return manifest


def run_mode_analysis(
    output_dir: Path | None = None,
    *,
    bootstrap_repeats: int = 60,
) -> Path:
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            PROJECT_ROOT
            / DEFAULT_OUTPUT_PARENT
            / f"result4_mode_validation_{stamp}"
        )
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    save_mode_analysis(output_dir, bootstrap_repeats=bootstrap_repeats)
    return output_dir


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the current full5862 six-mode planning typology."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=60)
    args = parser.parse_args()
    output = run_mode_analysis(
        args.output_dir,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(output)


if __name__ == "__main__":
    run_cli()

