from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ._shared import (
        EXPECTED_FULL_ROWS,
        FORBIDDEN_PATH_MARKERS,
        COUNTRY_GIRI_REL,
        FULL_GCIRI_REL,
        MODE_DIR_REL,
        PAPER_DIR_REL,
        POLICY_DIR_REL,
        PROJECT_ROOT,
        RESTORATIVE_DIR_REL,
        apply_style,
        load_country_giri,
        load_full_gciri,
        load_plotting_backend,
        panel_label,
    )
except ImportError:
    from _shared import (
        EXPECTED_FULL_ROWS,
        FORBIDDEN_PATH_MARKERS,
        COUNTRY_GIRI_REL,
        FULL_GCIRI_REL,
        MODE_DIR_REL,
        PAPER_DIR_REL,
        POLICY_DIR_REL,
        PROJECT_ROOT,
        RESTORATIVE_DIR_REL,
        apply_style,
        load_country_giri,
        load_full_gciri,
        load_plotting_backend,
        panel_label,
    )


DISCOVERY_DIR_REL = PAPER_DIR_REL / "discovery_20260705_020407"
HISTORY_REL = PAPER_DIR_REL / "_history/markdown_era_2026-07"
FEMA_DIR_REL = (
    HISTORY_REL / "external_validation_us_fema_20260705_034300"
)

PAIR_REL = DISCOVERY_DIR_REL / "case_pair_contrasts.csv"
ANOMALY_REL = DISCOVERY_DIR_REL / "anomaly_ledger.csv"
SENSITIVITY_REL = MODE_DIR_REL / "sensitivity_grid.csv"
CLUSTER_REL = MODE_DIR_REL / "cluster_validation.csv"
BOOTSTRAP_REL = MODE_DIR_REL / "bootstrap_cluster_stability.csv"
RESTORATIVE_SUMMARY_REL = (
    RESTORATIVE_DIR_REL / "scenario_mode_change_summary.csv"
)
SCENARIO_SUMMARY_REL = POLICY_DIR_REL / "scenario_summary.csv"
POLICY_CITY_REL = POLICY_DIR_REL / "city_policy_typology.csv"
IDENTITY_REL = HISTORY_REL / "identity_anchor_scan_20260705.csv"
COVARIATE_REL = (
    HISTORY_REL / "city_level_covariate_validation_stats_20260705.csv"
)
FEMA_CORRELATION_REL = FEMA_DIR_REL / "us_fema_nri_correlations.csv"

FORBIDDEN_SOURCE_MARKERS = FORBIDDEN_PATH_MARKERS

INK = "#252A2E"
MUTED = "#6C747A"
GRID = "#D9DEE2"
BLUE = "#376795"
TEAL = "#528FAD"
GOLD = "#FFD06F"
RED = "#E76254"
PURPLE = "#8896A0"
LIGHT_BLUE = "#AADCE0"
PALE_YELLOW = "#FFE6B7"
AUDIT_COLORS = ["#376795", "#528FAD", "#72BCD5", "#FFD06F", "#EF8A47", "#E76254"]


def load_csv(relative_path: Path, required: set[str]) -> pd.DataFrame:
    text = relative_path.as_posix().lower()
    if any(marker in text for marker in FORBIDDEN_SOURCE_MARKERS):
        raise ValueError(f"Forbidden source path: {relative_path}")
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def _load_model_inputs(truth: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Mode, scenario and formula inputs derived from the current run, checked against it."""
    sensitivity = load_csv(
        SENSITIVITY_REL,
        {
            "configuration_id",
            "low_threshold",
            "high_threshold",
            "weakness_margin",
            "balance_spread",
            "adjusted_rand_vs_baseline",
            "city_retention_vs_baseline",
            "is_baseline",
        },
    )
    cluster = load_csv(
        CLUSTER_REL,
        {
            "cluster_count",
            "silhouette",
            "adjusted_rand_vs_rule_modes",
            "selected_six_cluster_check",
        },
    )
    bootstrap = load_csv(
        BOOTSTRAP_REL,
        {"repeat", "adjusted_rand_vs_full_k6", "adjusted_rand_vs_rule_modes"},
    )
    restorative = load_csv(
        RESTORATIVE_SUMMARY_REL,
        {
            "scenario",
            "mode_changed_n",
            "mode_changed_pct",
            "gciri_tier_changed_n",
            "gciri_tier_changed_pct",
        },
    )
    scenarios = load_csv(
        SCENARIO_SUMMARY_REL,
        {
            "scenario",
            "n",
            "mean_delta_gciri",
            "median_delta_gciri",
            "q25_delta_gciri",
            "q75_delta_gciri",
            "negative_n",
        },
    )
    policy_city = load_csv(
        POLICY_CITY_REL,
        {
            "city",
            "gciri",
            "delta_restorative_10pp_raw",
            "restorative_nonmonotonic_flag",
        },
    )
    policy_city = policy_city.merge(
        truth[["city", "r_ph_city"]],
        on="city",
        how="left",
        validate="one_to_one",
    )
    if policy_city["r_ph_city"].isna().any():
        raise ValueError("Formula diagnostic rows do not all match current truth")
    if len(truth) != EXPECTED_FULL_ROWS or truth["city"].nunique() != EXPECTED_FULL_ROWS:
        raise ValueError("Current truth table does not preserve 5862 unique city keys")
    if len(sensitivity) != 81:
        raise ValueError("Focused sensitivity input must contain 81 settings")
    if cluster["cluster_count"].tolist() != [4, 5, 6, 7, 8]:
        raise ValueError("Cluster validation must cover k=4 through k=8")
    if len(bootstrap) != 60:
        raise ValueError("Bootstrap cluster input must contain 60 resamples")
    if len(policy_city) != EXPECTED_FULL_ROWS:
        raise ValueError("Policy diagnostic table must contain 5862 cities")
    if not np.allclose(
        policy_city.set_index("city").loc[truth["city"], "gciri"].to_numpy(),
        truth["gciri"].to_numpy(),
        atol=0.011,
    ):
        raise ValueError("Formula diagnostic table does not match current GCIRI")
    return {"sensitivity": sensitivity, "cluster": cluster, "bootstrap": bootstrap,
            "restorative": restorative, "scenarios": scenarios, "policy_city": policy_city}


def load_supplementary_inputs(*, include_july_checks: bool = True) -> dict[str, pd.DataFrame]:
    """Inputs of the supplementary figures.

    include_july_checks=False skips the July 2026 reasonableness pairs, anomaly ledger, identity scan
    and covariate tables, which were computed on the July results and feed only the S3, S4, S9 and
    S10 figures that are not part of the paper.
    """
    truth = load_full_gciri()
    country = load_country_giri()
    data = _load_model_inputs(truth)
    if not include_july_checks:
        return {"truth": truth, "country": country, **data}
    pairs = load_csv(
        PAIR_REL,
        {
            "group",
            "left_city",
            "left_key",
            "left_gciri",
            "right_city",
            "right_key",
            "right_gciri",
            "gciri_gap_left_minus_right",
            "distance_km",
            "status",
        },
    )
    anomalies = load_csv(
        ANOMALY_REL,
        {"flag", "city", "iso3", "gciri", "r_ph_city", "reason_to_audit"},
    )
    sensitivity, cluster, bootstrap = data["sensitivity"], data["cluster"], data["bootstrap"]
    restorative, scenarios, policy_city = data["restorative"], data["scenarios"], data["policy_city"]
    identity = load_csv(IDENTITY_REL, {"check", "count", "note"})
    covariates = load_csv(
        COVARIATE_REL,
        {
            "column",
            "label",
            "n_nonmissing",
            "spearman_gciri",
            "independence_class",
        },
    )
    fema = load_csv(
        FEMA_CORRELATION_REL,
        {"left", "right", "n", "spearman", "expectation"},
    )

    if set(pairs["status"]) != {"ok"}:
        raise ValueError("Reasonableness-pair input contains a failed pair")

    truth_scores = truth.set_index("city")["gciri"]
    for key_column, score_column in [
        ("left_key", "left_gciri"),
        ("right_key", "right_gciri"),
    ]:
        expected = pairs[key_column].map(truth_scores)
        if expected.isna().any():
            raise ValueError("A reasonableness pair is absent from the current truth table")
        if not np.allclose(expected, pairs[score_column], atol=0.011):
            raise ValueError("Reasonableness-pair scores do not match current truth")
    if not set(anomalies["city"]).issubset(set(truth["city"])):
        raise ValueError("Anomaly ledger contains a city outside current truth")

    return {
        "truth": truth,
        "country": country,
        "pairs": pairs,
        "anomalies": anomalies,
        "sensitivity": sensitivity,
        "cluster": cluster,
        "bootstrap": bootstrap,
        "restorative": restorative,
        "scenarios": scenarios,
        "policy_city": policy_city,
        "identity": identity,
        "covariates": covariates,
        "fema": fema,
    }


def build_output_dir(output_dir: Path | None) -> Path:
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            PROJECT_ROOT
            / "_figures"
            / f"gciri_full5862_supplementary_{stamp}"
        )
    elif not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "source_data").mkdir()
    return output_dir


def style_axis(ax, *, zero_line: bool = False) -> None:
    ax.tick_params(colors=INK, width=0.5, length=2.5)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.grid(axis="y", color=GRID, linewidth=0.45, alpha=0.8, zorder=0)
    if zero_line:
        ax.axvline(0, color=MUTED, linewidth=0.7, zorder=1)


def save_source_data(
    output_dir: Path,
    filename: str,
    frame: pd.DataFrame,
) -> str:
    relative = Path("source_data") / filename
    frame.to_csv(output_dir / relative, index=False)
    return relative.as_posix()


def save_supplementary_figure(
    fig,
    output_dir: Path,
    stem: str,
    *,
    panels: int,
    input_sources: list[Path],
    source_data: list[str],
) -> None:
    formats = {
        "png": {"dpi": 300},
        "pdf": {},
        "svg": {},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }
    for suffix, options in formats.items():
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **options,
        )
    metadata = {
        "schema_version": "gciri-supplementary-figure/v1",
        "stem": stem,
        "truth_source": FULL_GCIRI_REL.as_posix(),
        "source_rows": EXPECTED_FULL_ROWS,
        "unique_city_keys": EXPECTED_FULL_ROWS,
        "input_sources": [path.as_posix() for path in input_sources],
        "panels": panels,
        "source_data": source_data,
        "formats": list(formats),
        "figure_status": "supplementary submission candidate",
    }
    (output_dir / f"{stem}.source.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def build_s1_country_spread(data, output_dir, plt) -> str:
    truth = data["truth"]
    summary = (
        truth.groupby(["iso3", "country_name"], as_index=False)
        .agg(
            n_cities=("gciri", "size"),
            mean_gciri=("gciri", "mean"),
            q10_gciri=("gciri", lambda values: values.quantile(0.10)),
            q90_gciri=("gciri", lambda values: values.quantile(0.90)),
            min_gciri=("gciri", "min"),
            max_gciri=("gciri", "max"),
        )
    )
    summary["range_gciri"] = summary["max_gciri"] - summary["min_gciri"]
    selected = (
        summary.query("n_cities >= 10")
        .nlargest(20, "range_gciri")
        .sort_values("range_gciri")
        .reset_index(drop=True)
    )
    source = save_source_data(output_dir, "figs1_country_spread.csv", selected)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(selected))
    axes[0].hlines(y, selected["min_gciri"], selected["max_gciri"], color=LIGHT_BLUE, linewidth=3.4)
    axes[0].hlines(y, selected["q10_gciri"], selected["q90_gciri"], color=BLUE, linewidth=5.8)
    axes[0].scatter(selected["mean_gciri"], y, s=20, color=GOLD, edgecolor="white", linewidth=0.4, zorder=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(selected["iso3"])
    axes[0].set_xlabel("City GCIRI")
    axes[0].set_ylabel("Country (ISO3; n ≥ 10 cities)")
    axes[0].set_xlim(-2, 102)
    style_axis(axes[0])
    panel_label(axes[0], "a")

    axes[1].scatter(
        summary.query("n_cities >= 10")["n_cities"],
        summary.query("n_cities >= 10")["range_gciri"],
        s=20,
        color=TEAL,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
    )
    for row in selected.nlargest(7, "range_gciri").itertuples():
        axes[1].annotate(
            row.iso3,
            (row.n_cities, row.range_gciri),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=6.5,
            color=INK,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Number of cities (log scale)")
    axes[1].set_ylabel("Within-country GCIRI range")
    axes[1].set_ylim(0, max(70, selected["range_gciri"].max() + 4))
    style_axis(axes[1])
    panel_label(axes[1], "b")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.13, wspace=0.30)
    stem = "figs1_within_country_gciri_spread"
    save_supplementary_figure(
        fig,
        output_dir,
        stem,
        panels=2,
        input_sources=[FULL_GCIRI_REL],
        source_data=[source],
    )
    plt.close(fig)
    return stem


def build_s2_country_anchor(data, output_dir, plt) -> str:
    from matplotlib.colors import LinearSegmentedColormap

    country = data["country"].dropna(subset=["official_giri", "gciri_mean"]).copy()
    country["residual"] = country["gciri_mean"] - country["official_giri"]
    source = save_source_data(output_dir, "figs2_country_official_anchor.csv", country)

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    sizes = 12 + 3 * np.sqrt(country["n_cities"].clip(lower=1))
    residual_cmap = LinearSegmentedColormap.from_list(
        "ukiyoe_signed_residual",
        [BLUE, PALE_YELLOW, RED],
        N=256,
    )
    points = ax.scatter(
        country["official_giri"],
        country["gciri_mean"],
        s=sizes,
        c=country["residual"],
        cmap=residual_cmap,
        vmin=-20,
        vmax=20,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.35,
    )
    colorbar = fig.colorbar(points, ax=ax, pad=0.02, fraction=0.05)
    colorbar.set_label("Mean city GCIRI minus official GIRI")
    limits = [min(country["official_giri"].min(), country["gciri_mean"].min()) - 3, 103]
    ax.plot(limits, limits, color=MUTED, linestyle="--", linewidth=0.8)
    for row in pd.concat([country.nsmallest(4, "residual"), country.nlargest(4, "residual")]).drop_duplicates("iso3").itertuples():
        ax.annotate(
            row.iso3,
            (row.official_giri, row.gciri_mean),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=6.5,
            color=INK,
        )
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Official national GIRI")
    ax.set_ylabel("Mean city GCIRI")
    ax.grid(color=GRID, linewidth=0.45, alpha=0.75)
    rho = country[["official_giri", "gciri_mean"]].corr(method="spearman").iloc[0, 1]
    ax.text(
        0.03,
        0.97,
        f"Spearman ρ = {rho:.2f}; n = {len(country)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=INK,
    )
    panel_label(ax, "a")
    fig.subplots_adjust(left=0.15, right=0.96, top=0.95, bottom=0.14)
    stem = "figs2_country_mean_gciri_official_anchor"
    save_supplementary_figure(
        fig,
        output_dir,
        stem,
        panels=1,
        input_sources=[FULL_GCIRI_REL, COUNTRY_GIRI_REL],
        source_data=[source],
    )
    plt.close(fig)
    return stem


def build_s3_reasonableness_pairs(data, output_dir, plt) -> str:
    pairs = data["pairs"].copy()
    pairs["pair_label"] = pairs["left_city"] + " / " + pairs["right_city"]
    pairs = pairs.sort_values("gciri_gap_left_minus_right").reset_index(drop=True)
    source = save_source_data(output_dir, "figs3_reasonableness_pairs.csv", pairs)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(pairs))
    axes[0].hlines(y, pairs["left_gciri"], pairs["right_gciri"], color=GRID, linewidth=2.0)
    axes[0].scatter(pairs["left_gciri"], y, color=BLUE, s=23, label="Left city", zorder=3)
    axes[0].scatter(pairs["right_gciri"], y, color=GOLD, s=23, label="Right city", zorder=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(pairs["pair_label"])
    axes[0].set_xlabel("GCIRI")
    axes[0].set_xlim(-2, 102)
    axes[0].legend(frameon=False, loc="lower right")
    style_axis(axes[0])
    panel_label(axes[0], "a")

    colors = np.where(
        pairs["gciri_gap_left_minus_right"] >= 0,
        BLUE,
        RED,
    )
    axes[1].barh(y, pairs["gciri_gap_left_minus_right"], color=colors, height=0.62)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Left minus right GCIRI")
    axes[1].axvline(0, color=MUTED, linewidth=0.7)
    style_axis(axes[1])
    panel_label(axes[1], "b")
    fig.subplots_adjust(left=0.23, right=0.97, top=0.94, bottom=0.13, wspace=0.12)
    stem = "figs3_neighboring_city_reasonableness_pairs"
    save_supplementary_figure(
        fig,
        output_dir,
        stem,
        panels=2,
        input_sources=[FULL_GCIRI_REL, PAIR_REL],
        source_data=[source],
    )
    plt.close(fig)
    return stem


def build_s4_anomaly_classes(data, output_dir, plt) -> str:
    anomalies = data["anomalies"].copy()
    summary = (
        anomalies.groupby("flag", as_index=False)
        .agg(n=("city", "size"), median_gciri=("gciri", "median"), median_risk=("r_ph_city", "median"))
        .sort_values("n")
    )
    source_a = save_source_data(output_dir, "figs4_anomaly_summary.csv", summary)
    source_b = save_source_data(output_dir, "figs4_anomaly_rows.csv", anomalies)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(summary))
    labels = summary["flag"].str.replace("_", " ", regex=False)
    axes[0].barh(y, summary["n"], color=PURPLE, height=0.64)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("Flagged rows")
    style_axis(axes[0])
    panel_label(axes[0], "a")

    for index, (flag, group) in enumerate(anomalies.groupby("flag", sort=True)):
        jitter = np.linspace(-0.12, 0.12, len(group)) if len(group) > 1 else np.array([0.0])
        axes[1].scatter(
            group["gciri"],
            group["r_ph_city"] + jitter * 0.03,
            s=17,
            alpha=0.72,
            color=AUDIT_COLORS[index % len(AUDIT_COLORS)],
            label=flag.replace("_", " "),
        )
    axes[1].set_xlabel("GCIRI")
    axes[1].set_ylabel("Physical risk")
    axes[1].set_xlim(-2, 102)
    axes[1].set_ylim(-0.02, 1.03)
    axes[1].legend(frameon=False, fontsize=6, loc="lower left")
    axes[1].grid(color=GRID, linewidth=0.45, alpha=0.75)
    panel_label(axes[1], "b")
    fig.subplots_adjust(left=0.18, right=0.98, top=0.94, bottom=0.14, wspace=0.32)
    stem = "figs4_anomaly_and_boundary_classes"
    save_supplementary_figure(
        fig,
        output_dir,
        stem,
        panels=2,
        input_sources=[FULL_GCIRI_REL, ANOMALY_REL],
        source_data=[source_a, source_b],
    )
    plt.close(fig)
    return stem


def build_s5_physical_risk(data, output_dir, plt) -> str:
    truth = data["truth"][["city", "iso3", "r_ph_city", "gciri", "pop_2025"]].copy()
    truth["risk_bin"] = pd.cut(truth["r_ph_city"], bins=np.linspace(0, 1, 21), include_lowest=True)
    quantiles = (
        truth.groupby("risk_bin", observed=True)
        .agg(
            risk_mid=("r_ph_city", "mean"),
            gciri_q10=("gciri", lambda values: values.quantile(0.10)),
            gciri_median=("gciri", "median"),
            gciri_q90=("gciri", lambda values: values.quantile(0.90)),
            n=("gciri", "size"),
        )
        .reset_index(drop=True)
    )
    source_a = save_source_data(output_dir, "figs5_city_physical_risk_gciri.csv", truth)
    source_b = save_source_data(output_dir, "figs5_risk_bin_quantiles.csv", quantiles)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    hb = ax.hexbin(
        truth["r_ph_city"],
        truth["gciri"],
        gridsize=38,
        mincnt=1,
        cmap="Blues",
        linewidths=0,
    )
    ax.fill_between(
        quantiles["risk_mid"],
        quantiles["gciri_q10"],
        quantiles["gciri_q90"],
        color=GOLD,
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(quantiles["risk_mid"], quantiles["gciri_median"], color=RED, linewidth=1.5)
    cbar = fig.colorbar(hb, ax=ax, orientation="vertical", pad=0.02, fraction=0.05)
    cbar.set_label("Cities per hexagon")
    ax.set_xlabel("Physical risk")
    ax.set_ylabel("GCIRI")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-2, 102)
    ax.grid(color=GRID, linewidth=0.4, alpha=0.6)
    panel_label(ax, "a")
    fig.subplots_adjust(left=0.12, right=0.90, top=0.95, bottom=0.14)
    stem = "figs5_physical_risk_and_gciri"
    save_supplementary_figure(
        fig,
        output_dir,
        stem,
        panels=1,
        input_sources=[FULL_GCIRI_REL],
        source_data=[source_a, source_b],
    )
    plt.close(fig)
    return stem


def build_s6_threshold_sensitivity(data, output_dir, plt) -> str:
    grid = data["sensitivity"].copy()
    source = save_source_data(output_dir, "figs6_threshold_sensitivity.csv", grid)
    parameters = [
        ("low_threshold", "Low threshold"),
        ("high_threshold", "High threshold"),
        ("weakness_margin", "Weakness margin"),
        ("balance_spread", "Balance spread"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharey=True)
    for letter, ax, (column, label) in zip("abcd", axes.flat, parameters):
        levels = sorted(grid[column].unique())
        positions = np.arange(len(levels))
        for position, level in zip(positions, levels):
            values = grid.loc[grid[column] == level, "city_retention_vs_baseline"] * 100
            jitter = np.linspace(-0.13, 0.13, len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=13,
                color=LIGHT_BLUE,
                alpha=0.65,
                edgecolor="none",
            )
            ax.scatter(
                position,
                values.median(),
                marker="D",
                s=30,
                color=RED,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{value:g}" for value in levels])
        ax.set_xlabel(label)
        ax.set_ylim(82, 101)
        style_axis(ax)
        panel_label(ax, letter)
    axes[0, 0].set_ylabel("City-label retention (%)")
    axes[1, 0].set_ylabel("City-label retention (%)")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.10, hspace=0.34, wspace=0.16)
    stem = "figs6_six_mode_threshold_sensitivity"
    save_supplementary_figure(
        fig,
        output_dir,
        stem,
        panels=4,
        input_sources=[SENSITIVITY_REL],
        source_data=[source],
    )
    plt.close(fig)
    return stem


def build_s7_cluster_boundary(data, output_dir, plt) -> str:
    cluster = data["cluster"].copy()
    bootstrap = data["bootstrap"].copy()
    source_a = save_source_data(output_dir, "figs7_cluster_validation.csv", cluster)
    source_b = save_source_data(output_dir, "figs7_cluster_bootstrap.csv", bootstrap)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    axes[0].plot(cluster["cluster_count"], cluster["silhouette"], marker="o", color=TEAL, label="Silhouette")
    axes[0].plot(
        cluster["cluster_count"],
        cluster["adjusted_rand_vs_rule_modes"],
        marker="s",
        color=PURPLE,
        label="ARI vs rule modes",
    )
    axes[0].axvline(6, color=MUTED, linestyle="--", linewidth=0.7)
    axes[0].set_xticks(cluster["cluster_count"])
    axes[0].set_xlabel("K-means cluster count")
    axes[0].set_ylabel("Agreement or separation score")
    axes[0].set_ylim(0.20, 0.42)
    axes[0].legend(frameon=False)
    style_axis(axes[0])
    panel_label(axes[0], "a")

    axes[1].hist(
        bootstrap["adjusted_rand_vs_full_k6"],
        bins=np.linspace(0.55, 1.0, 13),
        color=BLUE,
        alpha=0.82,
        edgecolor="white",
    )
    median = bootstrap["adjusted_rand_vs_full_k6"].median()
    axes[1].axvline(median, color=RED, linewidth=1.2)
    axes[1].text(
        median - 0.01,
        axes[1].get_ylim()[1] * 0.90,
        f"Median {median:.3f}",
        ha="right",
        va="top",
        fontsize=7,
        color=RED,
    )
    axes[1].set_xlabel("Resampled ARI versus full k=6")
    axes[1].set_ylabel("Resamples")
    style_axis(axes[1])
    panel_label(axes[1], "b")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.94, bottom=0.15, wspace=0.28)
    stem = "figs7_alternative_clustering_boundary"
    save_supplementary_figure(
        fig,
        output_dir,
        stem,
        panels=2,
        input_sources=[CLUSTER_REL, BOOTSTRAP_REL],
        source_data=[source_a, source_b],
    )
    plt.close(fig)
    return stem


def build_s8_formula_diagnostics(data, output_dir, plt) -> str:
    scenarios = data["scenarios"].copy()
    restorative = data["restorative"].copy()
    policy_city = data["policy_city"].copy()
    source_a = save_source_data(output_dir, "figs8_scenario_summary.csv", scenarios)
    source_b = save_source_data(output_dir, "figs8_mode_change_summary.csv", restorative)
    source_c = save_source_data(
        output_dir,
        "figs8_restorative_city_response.csv",
        policy_city[
            [
                "city",
                "gciri",
                "r_ph_city",
                "delta_restorative_10pp_raw",
                "restorative_nonmonotonic_flag",
            ]
        ],
    )

    selected = scenarios[
        scenarios["scenario"].isin(
            ["absorptive +10 pp", "response +10 pp", "restorative +10 pp"]
        )
    ].copy()
    selected["short"] = selected["scenario"].str.replace(" +10 pp", "", regex=False)
    selected = selected.set_index("short").loc[["absorptive", "response", "restorative"]].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.05, 1.1, 1]})
    y = np.arange(len(selected))
    xerr = np.vstack(
        [
            selected["median_delta_gciri"] - selected["q25_delta_gciri"],
            selected["q75_delta_gciri"] - selected["median_delta_gciri"],
        ]
    )
    axes[0].errorbar(
        selected["median_delta_gciri"],
        y,
        xerr=xerr,
        fmt="o",
        color=BLUE,
        ecolor=LIGHT_BLUE,
        capsize=3,
    )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(selected["short"])
    axes[0].set_xlabel("Modeled GCIRI change")
    axes[0].axvline(0, color=MUTED, linewidth=0.7)
    style_axis(axes[0])
    panel_label(axes[0], "a")

    axes[1].hist(
        policy_city["delta_restorative_10pp_raw"],
        bins=np.linspace(-3.5, 2.5, 35),
        color=TEAL,
        alpha=0.82,
        edgecolor="white",
    )
    axes[1].axvline(0, color=RED, linewidth=0.9)
    axes[1].set_xlabel("Restorative +10 modeled change")
    axes[1].set_ylabel("Cities")
    style_axis(axes[1])
    panel_label(axes[1], "b")

    labels = [
        "Score response",
        "Capacity profile",
        "Combined",
    ]
    axes[2].barh(
        np.arange(3),
        restorative["mode_changed_pct"],
        color=[GOLD, PURPLE, BLUE],
        height=0.62,
    )
    axes[2].set_yticks(np.arange(3))
    axes[2].set_yticklabels(labels)
    axes[2].set_xlabel("Mode labels changed (%)")
    axes[2].set_xlim(0, 40)
    style_axis(axes[2])
    panel_label(axes[2], "c")
    fig.subplots_adjust(left=0.10, right=0.99, top=0.94, bottom=0.16, wspace=0.42)
    stem = "figs8_formula_and_restorative_boundary"
    save_supplementary_figure(
        fig,
        output_dir,
        stem,
        panels=3,
        input_sources=[SCENARIO_SUMMARY_REL, RESTORATIVE_SUMMARY_REL, POLICY_CITY_REL],
        source_data=[source_a, source_b, source_c],
    )
    plt.close(fig)
    return stem


def build_s9_identity_checks(data, output_dir, plt) -> str:
    identity = data["identity"].copy()
    critical_order = [
        "duplicate_ucdb_id",
        "missing_ucdb_id",
        "missing_lat_or_lon",
        "result_missing_gciri",
        "result_duplicate_ucdb_id",
        "special_synthetic_ucdb_id_ge_999900",
    ]
    descriptive_order = [
        "duplicate_city_name_any_country",
        "duplicate_city_iso_pair",
        "non_ascii_city_name",
        "bracketed_city_name",
    ]
    indexed = identity.set_index("check")
    critical = indexed.loc[critical_order].reset_index()
    descriptive = indexed.loc[descriptive_order].reset_index()
    source = save_source_data(output_dir, "figs9_identity_checks.csv", identity)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    for ax, frame, letter, color, xlabel in [
        (axes[0], critical, "a", TEAL, "Rows"),
        (axes[1], descriptive, "b", GOLD, "Rows flagged for name review"),
    ]:
        y = np.arange(len(frame))
        ax.barh(y, frame["count"], color=color, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(frame["check"].str.replace("_", " ", regex=False))
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        for yi, value in zip(y, frame["count"]):
            ax.text(value + max(frame["count"].max() * 0.02, 0.15), yi, str(int(value)), va="center", fontsize=7)
        style_axis(ax)
        panel_label(ax, letter)
    axes[0].set_xlim(0, max(4, critical["count"].max() + 2))
    axes[1].set_xlim(0, descriptive["count"].max() * 1.18)
    fig.subplots_adjust(left=0.26, right=0.98, top=0.94, bottom=0.14, wspace=0.68)
    stem = "figs9_completeness_and_identity_checks"
    save_supplementary_figure(
        fig,
        output_dir,
        stem,
        panels=2,
        input_sources=[FULL_GCIRI_REL, IDENTITY_REL],
        source_data=[source],
    )
    plt.close(fig)
    return stem


def build_s10_validation_boundary(data, output_dir, plt) -> str:
    covariates = data["covariates"].copy()
    fema = data["fema"].copy()
    selected_columns = [
        "pop_2025_master",
        "exp_ntl_city__RAW",
        "exp_pop_ntl_city__RAW",
        "GIRI__RAW_nat",
    ]
    selected = covariates.set_index("column").loc[selected_columns].reset_index()
    selected["short"] = ["Population", "Night lights", "Pop. x night lights", "Official GIRI"]
    fema_selected = fema[
        (fema["left"] == "gciri")
        & fema["right"].isin(["RESL_SCORE", "SOVI_SCORE", "EAL_SCORE", "RISK_SCORE"])
    ].copy()
    fema_selected["short"] = fema_selected["right"].map(
        {
            "RESL_SCORE": "FEMA resilience",
            "SOVI_SCORE": "FEMA social vulnerability",
            "EAL_SCORE": "FEMA expected annual loss",
            "RISK_SCORE": "FEMA composite risk",
        }
    )
    source_a = save_source_data(output_dir, "figs10_covariate_correlations.csv", selected)
    source_b = save_source_data(output_dir, "figs10_fema_correlations.csv", fema_selected)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharex=True)
    y = np.arange(len(selected))
    colors = [MUTED, GOLD, LIGHT_BLUE, PURPLE]
    axes[0].barh(y, selected["spearman_gciri"], color=colors, height=0.62)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(selected["short"])
    axes[0].set_xlabel("Spearman correlation with GCIRI")
    axes[0].axvline(0, color=MUTED, linewidth=0.7)
    axes[0].set_xlim(-0.5, 0.95)
    style_axis(axes[0])
    panel_label(axes[0], "a")

    y2 = np.arange(len(fema_selected))
    colors2 = [TEAL if value >= 0 else RED for value in fema_selected["spearman"]]
    axes[1].barh(y2, fema_selected["spearman"], color=colors2, height=0.62)
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels(fema_selected["short"])
    axes[1].set_xlabel("Spearman correlation with GCIRI")
    axes[1].axvline(0, color=MUTED, linewidth=0.7)
    style_axis(axes[1])
    panel_label(axes[1], "b")
    fig.subplots_adjust(left=0.18, right=0.98, top=0.94, bottom=0.15, wspace=0.48)
    stem = "figs10_validation_boundary_correlations"
    save_supplementary_figure(
        fig,
        output_dir,
        stem,
        panels=2,
        input_sources=[COVARIATE_REL, FEMA_CORRELATION_REL],
        source_data=[source_a, source_b],
    )
    plt.close(fig)
    return stem


def write_report(output_dir: Path, stems: list[str]) -> None:
    report = [
        "# GCIRI full5862 supplementary figure report",
        "",
        f"- Truth source: `{FULL_GCIRI_REL.as_posix()}`",
        f"- Source rows: {EXPECTED_FULL_ROWS}",
        f"- Figures generated: {len(stems)}",
        "- Status: supplementary submission candidates",
        "",
        "## Figures",
        "",
    ]
    report.extend(f"- `{stem}`" for stem in stems)
    report.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Official national GIRI is a lineage anchor, not independent validation.",
            "- Pair and anomaly figures are reasonableness audits, not city rankings.",
            "- Rule modes are transparent planning types, not natural clusters.",
            "- Formula scenarios are diagnostics, not observed or cost-optimal interventions.",
            "- FEMA/NRI is a US county-level reasonableness slice, not global validation.",
            "",
        ]
    )
    (output_dir / "_supplementary_figure_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def run_supplementary_figures(output_dir: Path | None = None) -> Path:
    mpl, plt = load_plotting_backend()
    apply_style(mpl)
    data = load_supplementary_inputs()
    output_dir = build_output_dir(output_dir)
    builders = [
        build_s1_country_spread,
        build_s2_country_anchor,
        build_s3_reasonableness_pairs,
        build_s4_anomaly_classes,
        build_s5_physical_risk,
        build_s6_threshold_sensitivity,
        build_s7_cluster_boundary,
        build_s8_formula_diagnostics,
        build_s9_identity_checks,
        build_s10_validation_boundary,
    ]
    stems = [builder(data, output_dir, plt) for builder in builders]
    write_report(output_dir, stems)
    return output_dir


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Render current-data Supplementary Figs. S1-S10"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional repo-relative or absolute output directory",
    )
    args = parser.parse_args()
    output = run_supplementary_figures(args.output_dir)
    try:
        print(output.relative_to(PROJECT_ROOT))
    except ValueError:
        print(output)


if __name__ == "__main__":
    run_cli()
