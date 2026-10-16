from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

try:
    from ._shared import (
        EXPECTED_FULL_ROWS,
        FULL_GCIRI_REL,
        PROJECT_ROOT,
        apply_projected_world_axis,
        apply_style,
        compute_figure_size,
        compute_scatter_size,
        draw_projected_world_basemap,
        load_country_giri,
        load_full_gciri,
        load_plotting_backend,
        normalize_fonts,
        panel_label,
        project_robinson,
    )
    from .policy_pathway_analysis import (
        SOURCE_ORDER,
        derive_policy_pathways,
        summarize_city_types,
        summarize_country_portfolios,
        summarize_region_weakness,
        summarize_scenarios,
    )
except ImportError:
    from _shared import (
        EXPECTED_FULL_ROWS,
        FULL_GCIRI_REL,
        PROJECT_ROOT,
        apply_projected_world_axis,
        apply_style,
        compute_figure_size,
        compute_scatter_size,
        draw_projected_world_basemap,
        load_country_giri,
        load_full_gciri,
        load_plotting_backend,
        normalize_fonts,
        panel_label,
        project_robinson,
    )
    from policy_pathway_analysis import (
        SOURCE_ORDER,
        derive_policy_pathways,
        summarize_city_types,
        summarize_country_portfolios,
        summarize_region_weakness,
        summarize_scenarios,
    )


COUNTRY_META_REL = Path("_data/master_global_cities/master_global_cities_v1.csv")
OUTPUT_PREFIX = "gciri_full5862_formal"

UKIYOE = [
    "#E76254",
    "#EF8A47",
    "#F7AA58",
    "#FFD06F",
    "#FFE6B7",
    "#AADCE0",
    "#72BCD5",
    "#528FAD",
    "#376795",
    "#1E466E",
]
# Continuous score colors are reserved for GCIRI/capacity magnitude. Categorical
# dimensions use a separate color-blind-safe family.
DIM_COLORS = {
    "absorptive": "#C74A3A",
    "response": "#356D9B",
    "restorative": "#708B78",
}
DIM_ORDER = ["absorptive", "response", "restorative"]
DIM_LABELS = {"absorptive": "Absorptive", "response": "Response", "restorative": "Restorative"}
SOURCE_COLORS = {
    "hazard-dominant": "#A33F35",
    "capacity-dominant": "#3978A8",
    "compound": "#7A5B8E",
    "benchmark-saturated": "#8C9691",
}
SOURCE_LABELS = {
    "hazard-dominant": "Hazard-dominant",
    "capacity-dominant": "Capacity-dominant",
    "compound": "Compound",
    "benchmark-saturated": "Benchmark-saturated",
}
TIER_COLORS = {"<40": "#E76254", "40-70": "#FFD06F", ">=70": "#376795"}
TIER_LABELS = {"<40": "< 40", "40-70": "40-70", ">=70": ">= 70"}

REGION_LABELS = {
    1: "East & Southeast Asia",
    2: "South, Central & West Asia",
    3: "Americas",
    4: "Europe",
    5: "Africa",
    6: "Oceania",
}
REGION_SHORT = {
    1: "E & SE Asia",
    2: "S, C & W Asia",
    3: "Americas",
    4: "Europe",
    5: "Africa",
    6: "Oceania",
}
REGION_COLORS = {
    1: "#E76254",
    2: "#EF8A47",
    3: "#72BCD5",
    4: "#376795",
    5: "#FFD06F",
    6: "#1E466E",
}
NAME_TO_SHORT = {REGION_LABELS[i]: REGION_SHORT[i] for i in range(1, 7)}
INCOME_ORDER = ["High income", "Upper middle income", "Lower middle income", "Low income"]
INCOME_SHORT = ["High", "Upper middle", "Lower middle", "Low"]
INCOME_COLORS = ["#1E466E", "#72BCD5", "#FFD06F", "#E76254"]
PROXY_COLORS = {"South proxy": "#E76254", "North proxy": "#376795"}
PROXY_DISPLAY = {"South proxy": "Global South", "North proxy": "Global North"}
PROXY_INCOME = {"South proxy": "low and middle income", "North proxy": "high income"}


def _output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT_ROOT / "_figures" / f"{OUTPUT_PREFIX}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "source_data").mkdir()
    (out / "qa").mkdir()
    return out


def _score_cmap(mpl):
    return mpl.colors.LinearSegmentedColormap.from_list("gciri_ukiyoe", UKIYOE, N=256)


def _country_metadata() -> pd.DataFrame:
    path = PROJECT_ROOT / COUNTRY_META_REL
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    required = {"country_iso3", "income_group", "region_id", "region_name", "subregion_name"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for iso3, group in raw.dropna(subset=["country_iso3"]).groupby("country_iso3"):
        row: dict[str, object] = {"iso3": iso3}
        for col in ["income_group", "region_id", "region_name", "subregion_name"]:
            values = group[col].dropna()
            row[col] = values.mode().iloc[0] if len(values) else np.nan
        rows.append(row)
    meta = pd.DataFrame(rows)
    meta["region_id"] = pd.to_numeric(meta["region_id"], errors="coerce").astype("Int64")
    meta["region_group"] = meta["region_id"].map(REGION_LABELS)
    meta["region_short"] = meta["region_id"].map(REGION_SHORT)
    return meta


def _prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_full_gciri().copy()
    df = df.merge(_country_metadata(), on="iso3", how="left", validate="many_to_one")
    df["north_south_proxy"] = pd.Series(np.nan, index=df.index, dtype="object")
    df.loc[df["income_group"].eq("High income"), "north_south_proxy"] = "North proxy"
    df.loc[df["income_group"].notna() & ~df["income_group"].eq("High income"), "north_south_proxy"] = "South proxy"
    df["absorptive_strength"] = 100.0 - df["cap_abs"]
    df["response_strength"] = 100.0 - df["cap_resp"]
    df["restorative_strength"] = df["cap_rest"]
    strength_cols = ["absorptive_strength", "response_strength", "restorative_strength"]
    df["weakest_dimension"] = df[strength_cols].idxmin(axis=1).map(
        {
            "absorptive_strength": "absorptive",
            "response_strength": "response",
            "restorative_strength": "restorative",
        }
    )
    if len(df) != EXPECTED_FULL_ROWS or df["city"].nunique() != EXPECTED_FULL_ROWS:
        raise ValueError("Formal figure data must contain 5862 unique row-level city keys")
    country = load_country_giri().copy()
    df = df.merge(
        country[["iso3", "official_giri", "gciri_mean", "n_cities"]],
        on="iso3",
        how="left",
        validate="many_to_one",
    )
    if df["official_giri"].isna().any():
        missing_iso = sorted(df.loc[df["official_giri"].isna(), "iso3"].unique())
        raise ValueError(f"Missing official GIRI for ISO3 values: {missing_iso}")
    df["city_minus_official"] = df["gciri"] - df["official_giri"]
    return df, country


def _write_source(df: pd.DataFrame, out: Path, name: str) -> Path:
    path = out / "source_data" / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def _save_figure(fig, out: Path, stem: str, panels: int, source_files: list[str]) -> list[Path]:
    normalize_fonts(fig)
    paths = {
        "png": out / f"{stem}.png",
        "pdf": out / f"{stem}.pdf",
        "svg": out / f"{stem}.svg",
        "tiff": out / f"{stem}.tiff",
    }
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(
        paths["tiff"],
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    metadata = {
        "stem": stem,
        "source_table": FULL_GCIRI_REL.as_posix(),
        "source_rows": EXPECTED_FULL_ROWS,
        "panels": panels,
        "canvas_size_inches": [
            round(float(value), 3) for value in fig.get_size_inches()
        ],
        "source_data": source_files,
        "formats": ["png", "pdf", "svg", "tiff"],
        "figure_status": "formal_candidate",
    }
    (out / f"{stem}.source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return list(paths.values())


def _style_cartopy_world(ax) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="#EAF2F8", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#F7F7F5", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.20, edgecolor="#C4CBD0", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.32, edgecolor="#8F9AA2", zorder=1)
    ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.22,
        color="#C7D0D7", alpha=0.55, linestyle="-", zorder=0,
    )
    ax.spines["geo"].set_linewidth(0.45)
    ax.spines["geo"].set_edgecolor("#7F8B92")


def _draw_city_map(fig, ax, df: pd.DataFrame, value_col: str, cmap, cax=None, label: str | None = None, size_scale: float = 0.42):
    import cartopy.crs as ccrs

    plot_df = df.sort_values(value_col)
    _style_cartopy_world(ax)
    scatter = ax.scatter(
        plot_df["lon"],
        plot_df["lat"],
        c=plot_df[value_col],
        s=compute_scatter_size(plot_df["pop_2025"]) * size_scale,
        cmap=cmap,
        vmin=0,
        vmax=100,
        alpha=0.80,
        edgecolors="white",
        linewidths=0.08,
        rasterized=True,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    if cax is not None:
        cbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
        if label:
            cbar.set_label(label)
    return scatter


def _boxplot(ax, arrays: list[np.ndarray], labels: list[str], colors: list[str], ylabel: str, ylim=None) -> None:
    bp = ax.boxplot(
        arrays,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.62,
        medianprops={"color": "#111111", "linewidth": 0.9},
        whiskerprops={"color": "#555555", "linewidth": 0.7},
        capprops={"color": "#555555", "linewidth": 0.7},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.6)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.35)
    ax.set_axisbelow(True)


def _violin_boxplot(
    ax,
    arrays: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    ylabel: str,
    ylim=None,
) -> None:
    """Distribution-first violin with a compact box-and-median overlay."""
    positions = np.arange(1, len(arrays) + 1)
    violin = ax.violinplot(
        arrays,
        positions=positions,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.25,
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_linewidth(0.65)
        body.set_alpha(0.38)
    box = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.20,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.0},
        whiskerprops={"color": "#555555", "linewidth": 0.65},
        capprops={"color": "#555555", "linewidth": 0.65},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_linewidth(0.9)
        patch.set_alpha(0.92)
    ax.scatter(
        positions,
        [np.mean(values) for values in arrays],
        s=10,
        marker="D",
        c=colors,
        edgecolors="white",
        linewidths=0.35,
        zorder=4,
    )
    ax.set_xticks(positions, labels)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.35)
    ax.set_axisbelow(True)


def _bubble_matrix(
    ax,
    table: pd.DataFrame,
    row_labels: list[str],
    col_labels: list[str],
    colors: list[str],
    counts: list[int] | None = None,
) -> None:
    """Bubble area encodes row-wise percentage; cell text preserves exact values."""
    values = table.to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    y = np.arange(n_rows)[::-1]
    for j in range(n_cols):
        sizes = np.where(
            values[:, j] > 0,
            24.0 + 360.0 * np.clip(values[:, j], 0, 100) / 100.0,
            0.0,
        )
        ax.scatter(
            np.full(n_rows, j),
            y,
            s=sizes,
            color=colors[j],
            alpha=0.82,
            edgecolors="white",
            linewidths=0.7,
            zorder=2,
        )
        for yi, value in zip(y, values[:, j]):
            ax.text(
                j,
                yi,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="#111111",
                zorder=3,
            )
    ax.set_xticks(np.arange(n_cols), col_labels)
    ax.set_yticks(y, row_labels)
    right = n_cols - 0.35
    if counts is not None:
        right += 0.72
        for yi, count in zip(y, counts):
            ax.text(n_cols - 0.04, yi, f"{int(count):,}", ha="left", va="center", fontsize=6.3, color="#666666")
        ax.text(n_cols - 0.04, n_rows - 0.16, "n", ha="left", va="center", fontsize=6.2, color="#666666")
    ax.set_xlim(-0.62, right)
    ax.set_ylim(-0.65, n_rows - 0.35)
    ax.set_axisbelow(True)
    ax.grid(True, color="#E7E9EB", linewidth=0.35)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _tier_share(df: pd.DataFrame, group_col: str, order: list[str]) -> pd.DataFrame:
    tiers = pd.cut(df["gciri"], [-0.001, 40, 70, 100.001], labels=["<40", "40-70", ">=70"], right=False)
    table = pd.crosstab(df[group_col], tiers, normalize="index").mul(100)
    return table.reindex(order).fillna(0)[["<40", "40-70", ">=70"]]


def _hstacked(
    ax,
    table: pd.DataFrame,
    row_labels: list[str],
    colors: list[str],
    seg_labels: list[str],
    counts: list[int] | None = None,
    show_yticklabels: bool = True,
    xlabel: str = "Cities (%)",
) -> None:
    """Horizontal 100% stacked composition bars; first table row drawn at the top."""
    n = len(table)
    y = np.arange(n)[::-1]
    left = np.zeros(n)
    values = table.to_numpy(dtype=float)
    for j in range(values.shape[1]):
        ax.barh(y, values[:, j], left=left, height=0.66, color=colors[j], edgecolor="white", linewidth=0.4, label=seg_labels[j])
        left += values[:, j]
    ax.set_yticks(y)
    ax.set_yticklabels(row_labels if show_yticklabels else [""] * n)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", color="#D9D9D9", linewidth=0.35)
    ax.set_axisbelow(True)
    if counts is not None:
        for yi, c in zip(y, counts):
            ax.text(102.0, yi, f"{int(c):,}", va="center", ha="left", fontsize=6.5, color="#555555")


def _regional_rainclouds(
    ax,
    ax_labels,
    frame: pd.DataFrame,
    region_order: list[str],
    labels: list[str],
    mpl,
) -> pd.DataFrame:
    """Layer regional density, empirical samples and quantile intervals."""
    x_grid = np.linspace(0, 100, 360)
    y = np.arange(len(region_order))[::-1]
    score_cmap = _score_cmap(mpl)
    rng = np.random.default_rng(20260811)
    summary_rows: list[dict[str, float | int | str]] = []

    for baseline, region, label in zip(y, region_order, labels):
        values = frame.loc[frame["region_group"].eq(region), "gciri"].dropna().to_numpy(dtype=float)
        if len(values) < 5:
            raise ValueError(f"Region {region!r} has too few GCIRI values: {len(values)}")
        q05, q10, q25, q50, q75, q90, q95 = np.quantile(
            values,
            [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        )
        density = gaussian_kde(values, bw_method=0.24)(x_grid)
        density = density / max(float(density.max()), 1e-12) * 0.54
        ridge_color = score_cmap(float(np.clip(q50 / 100.0, 0, 1)))
        ax.fill_between(
            x_grid,
            baseline,
            baseline + density,
            color=ridge_color,
            alpha=0.38,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x_grid,
            baseline + density,
            color=ridge_color,
            linewidth=0.85,
            zorder=2,
        )

        sample = values
        sample_y = baseline - rng.uniform(0.10, 0.25, size=len(sample))
        ax.scatter(
            sample,
            sample_y,
            s=1.7,
            c=sample,
            cmap=score_cmap,
            vmin=0,
            vmax=100,
            alpha=0.24,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )

        interval_y = baseline + 0.025
        ax.plot([q05, q95], [interval_y, interval_y], color="#667078", linewidth=0.45, zorder=3)
        ax.plot([q10, q90], [interval_y, interval_y], color="#4E5961", linewidth=0.85, zorder=3)
        ax.plot([q25, q75], [interval_y, interval_y], color="#26333B", linewidth=2.25, solid_capstyle="round", zorder=4)
        ax.scatter(
            [q50],
            [interval_y],
            s=15,
            marker="o",
            facecolor="white",
            edgecolor="#18252C",
            linewidth=0.65,
            zorder=5,
        )

        summary_rows.append(
            {
                "region_group": region,
                "region_short": label,
                "n": int(len(values)),
                "mean_gciri": float(values.mean()),
                "p05": float(q05),
                "p10": float(q10),
                "p25": float(q25),
                "median": float(q50),
                "p75": float(q75),
                "p90": float(q90),
                "p95": float(q95),
            }
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.34, len(y) - 0.36)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([])
    ax.set_xlabel("GCIRI")
    ax.grid(True, axis="x", color="#DDE1E3", linewidth=0.35, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)

    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(ax.get_ylim())
    ax_labels.set_axis_off()
    for baseline, label in zip(y, labels):
        ax_labels.text(
            0.02,
            baseline + 0.14,
            label,
            ha="left",
            va="center",
            fontsize=6.2,
            color="#222222",
        )
    return pd.DataFrame(summary_rows)


def _global_distribution_anatomy(ax_density, ax_quantile, values: np.ndarray, mpl) -> None:
    """Show global density shape with a separate quantile and empirical strip."""
    score_cmap = _score_cmap(mpl)
    bins = np.linspace(0, 100, 45)
    _, bin_edges, patches = ax_density.hist(
        values,
        bins=bins,
        density=True,
        color="#AADCE0",
        alpha=0.84,
        edgecolor="white",
        linewidth=0.22,
    )
    for left, right, patch in zip(bin_edges[:-1], bin_edges[1:], patches):
        patch.set_facecolor(score_cmap(((left + right) * 0.5) / 100.0))
    kde = gaussian_kde(values)
    x_grid = np.linspace(0, 100, 500)
    density = kde(x_grid)
    ax_density.fill_between(x_grid, 0, density, color="#1E466E", alpha=0.075, linewidth=0)
    ax_density.plot(x_grid, density, color="#1E466E", linewidth=1.35)
    ax_density.set_ylabel("Density")
    ax_density.set_xlim(0, 100)
    ax_density.set_xticks([0, 20, 40, 60, 80, 100])
    ax_density.tick_params(axis="x", labelbottom=False)
    ax_density.grid(True, axis="y", color="#D9D9D9", linewidth=0.35)
    ax_density.set_axisbelow(True)

    q05, q10, q25, q50, q75, q90, q95 = np.quantile(
        values,
        [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
    )
    interval_y = 0.72
    ax_quantile.plot([q05, q95], [interval_y, interval_y], color="#667078", linewidth=0.5)
    ax_quantile.plot([q10, q90], [interval_y, interval_y], color="#4E5961", linewidth=0.95)
    ax_quantile.plot([q25, q75], [interval_y, interval_y], color="#26333B", linewidth=3.1, solid_capstyle="round")
    ax_quantile.scatter(
        [q50],
        [interval_y],
        s=19,
        marker="o",
        facecolor="white",
        edgecolor="#18252C",
        linewidth=0.7,
        zorder=3,
    )

    ordered = np.sort(values)
    sample_indices = np.linspace(0, len(ordered) - 1, 420).round().astype(int)
    sample = ordered[sample_indices]
    ax_quantile.vlines(
        sample,
        0.06,
        0.28,
        colors=score_cmap(np.clip(sample / 100.0, 0, 1)),
        linewidth=0.28,
        alpha=0.46,
    )
    ax_quantile.set_xlim(0, 100)
    ax_quantile.set_ylim(0, 1)
    ax_quantile.set_xticks([0, 20, 40, 60, 80, 100])
    ax_quantile.set_yticks([])
    ax_quantile.set_xlabel("GCIRI")
    for spine in ["left", "right", "top"]:
        ax_quantile.spines[spine].set_visible(False)
    ax_density.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for tick_label in ax_density.get_xticklabels():
        tick_label.set_visible(False)


def _binned_spatial_profile(
    df: pd.DataFrame,
    coordinate: str,
    edges: np.ndarray,
    min_count: int = 5,
) -> pd.DataFrame:
    """Aggregate city-level GCIRI into auditable longitude or latitude bands."""
    values = pd.to_numeric(df[coordinate], errors="coerce").to_numpy(dtype=float)
    scores = pd.to_numeric(df["gciri"], errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, float | int | bool]] = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            mask = (values >= left) & (values <= right)
        else:
            mask = (values >= left) & (values < right)
        band_scores = scores[mask & np.isfinite(scores)]
        row: dict[str, float | int | bool] = {
            "bin_left": float(left),
            "bin_right": float(right),
            "bin_center": float((left + right) / 2.0),
            "n": int(len(band_scores)),
            "displayed": bool(len(band_scores) >= min_count),
        }
        if len(band_scores):
            q10, q25, q50, q75, q90 = np.quantile(band_scores, [0.10, 0.25, 0.50, 0.75, 0.90])
            row.update(
                {
                    "mean_gciri": float(band_scores.mean()),
                    "median_gciri": float(q50),
                    "p10": float(q10),
                    "p25": float(q25),
                    "p75": float(q75),
                    "p90": float(q90),
                }
            )
        else:
            row.update({key: np.nan for key in ["mean_gciri", "median_gciri", "p10", "p25", "p75", "p90"]})
        rows.append(row)
    return pd.DataFrame(rows)


def _profile_marker_sizes(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    return 6.0 + 22.0 * np.sqrt(counts / max(float(np.nanmax(counts)), 1.0))


def _draw_longitude_profile(ax, profile: pd.DataFrame, mpl) -> None:
    valid = profile["displayed"].to_numpy(dtype=bool)
    x = profile["bin_center"].to_numpy(dtype=float)
    score_cmap = _score_cmap(mpl)
    ax.fill_between(
        x,
        profile["p10"],
        profile["p90"],
        where=valid,
        color="#72BCD5",
        alpha=0.10,
        linewidth=0,
    )
    ax.fill_between(
        x,
        profile["p25"],
        profile["p75"],
        where=valid,
        color="#528FAD",
        alpha=0.20,
        linewidth=0,
    )
    means = profile["mean_gciri"].where(profile["displayed"]).to_numpy(dtype=float)
    ax.plot(x, means, color="#1E466E", linewidth=0.95, zorder=3)
    shown = profile.loc[profile["displayed"]]
    ax.scatter(
        shown["bin_center"],
        shown["mean_gciri"],
        s=_profile_marker_sizes(shown["n"].to_numpy(dtype=float)),
        c=shown["mean_gciri"],
        cmap=score_cmap,
        vmin=0,
        vmax=100,
        edgecolor="white",
        linewidth=0.35,
        zorder=4,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 100)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Longitude (degree)")
    ax.set_ylabel("GCIRI")
    ax.grid(True, color="#DDE1E3", linewidth=0.32, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _draw_latitude_profile(ax, profile: pd.DataFrame, mpl) -> None:
    valid = profile["displayed"].to_numpy(dtype=bool)
    y = profile["bin_center"].to_numpy(dtype=float)
    score_cmap = _score_cmap(mpl)
    ax.fill_betweenx(
        y,
        profile["p10"],
        profile["p90"],
        where=valid,
        color="#72BCD5",
        alpha=0.10,
        linewidth=0,
    )
    ax.fill_betweenx(
        y,
        profile["p25"],
        profile["p75"],
        where=valid,
        color="#528FAD",
        alpha=0.20,
        linewidth=0,
    )
    means = profile["mean_gciri"].where(profile["displayed"]).to_numpy(dtype=float)
    ax.plot(means, y, color="#1E466E", linewidth=0.95, zorder=3)
    shown = profile.loc[profile["displayed"]]
    ax.scatter(
        shown["mean_gciri"],
        shown["bin_center"],
        s=_profile_marker_sizes(shown["n"].to_numpy(dtype=float)),
        c=shown["mean_gciri"],
        cmap=score_cmap,
        vmin=0,
        vmax=100,
        edgecolor="white",
        linewidth=0.35,
        zorder=4,
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(-65, 90)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([-60, -30, 0, 30, 60, 90])
    ax.set_xlabel("GCIRI")
    ax.set_ylabel("Latitude (degree)")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.grid(True, color="#DDE1E3", linewidth=0.32, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "left"]:
        ax.spines[spine].set_visible(False)


def _raincloud(ax, arrays: list[np.ndarray], labels: list[str], colors: list[str], ylabel: str) -> None:
    """Half violin, embedded box and deterministic empirical sample."""
    positions = np.arange(1, len(arrays) + 1)
    violin = ax.violinplot(arrays, positions=positions, widths=0.82, showextrema=False, bw_method=0.25)
    for position, body, color in zip(positions, violin["bodies"], colors):
        vertices = body.get_paths()[0].vertices
        vertices[:, 0] = np.minimum(vertices[:, 0], position)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.38)
        body.set_linewidth(0.7)
    box = ax.boxplot(
        arrays, positions=positions + 0.05, widths=0.12, patch_artist=True, showfliers=False,
        medianprops={"color": "#111111", "linewidth": 0.9},
        whiskerprops={"color": "#555555", "linewidth": 0.55},
        capprops={"color": "#555555", "linewidth": 0.55},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_linewidth(0.8)
    rng = np.random.default_rng(20260713)
    for position, values, color in zip(positions, arrays, colors):
        sample = rng.choice(values, size=min(80, len(values)), replace=False)
        x = position + 0.16 + rng.uniform(0.0, 0.16, size=len(sample))
        ax.scatter(x, sample, s=4.5, color=color, alpha=0.38, linewidths=0, rasterized=True)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="#E1E4E6", linewidth=0.35)


def _letter_value(ax, arrays: list[np.ndarray], labels: list[str], colors: list[str]) -> None:
    """Nested quantile bands reveal tails without repeating violin geometry."""
    from matplotlib.patches import Rectangle

    positions = np.arange(1, len(arrays) + 1)
    bands = [(0.05, 0.95, 0.58, 0.14), (0.125, 0.875, 0.44, 0.24), (0.25, 0.75, 0.28, 0.42)]
    for position, values, color in zip(positions, arrays, colors):
        for lo_q, hi_q, width, alpha in bands:
            lo, hi = np.quantile(values, [lo_q, hi_q])
            ax.add_patch(
                Rectangle(
                    (position - width / 2, lo), width, hi - lo,
                    facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.7,
                )
            )
        median = float(np.median(values))
        ax.plot([position - 0.17, position + 0.17], [median, median], color="#111111", linewidth=1.1)
        ax.scatter([position], [np.mean(values)], s=13, marker="D", color=color, edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("GCIRI")
    ax.grid(True, axis="y", color="#E1E4E6", linewidth=0.35)


def build_fig2(out: Path, df: pd.DataFrame, mpl, plt) -> list[Path]:
    import cartopy.crs as ccrs

    fig = plt.figure(figsize=(7.2, 7.2), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.55, 1.0],
        left=0.035,
        right=0.985,
        top=0.985,
        bottom=0.080,
        hspace=0.18,
    )
    map_grid = outer[0].subgridspec(
        3,
        2,
        height_ratios=[1.0, 0.065, 0.18],
        width_ratios=[1.0, 0.16],
        hspace=0.08,
        wspace=0.07,
    )
    bottom_grid = outer[1].subgridspec(
        1,
        2,
        width_ratios=[0.82, 1.18],
        wspace=0.34,
    )
    ax_map = fig.add_subplot(map_grid[0, 0], projection=ccrs.Robinson())
    cbar_row = fig.add_subplot(map_grid[1, 0])
    cbar_row.set_axis_off()
    cax = cbar_row.inset_axes([0.33, 0.30, 0.34, 0.42])
    ax_lon = fig.add_subplot(map_grid[2, 0])
    ax_lat = fig.add_subplot(map_grid[:, 1])
    distribution_grid = bottom_grid[0].subgridspec(
        2,
        1,
        height_ratios=[0.79, 0.21],
        hspace=0.02,
    )
    ax_hist = fig.add_subplot(distribution_grid[0])
    ax_hist_quantile = fig.add_subplot(distribution_grid[1], sharex=ax_hist)
    regional_grid = bottom_grid[1].subgridspec(
        1,
        2,
        width_ratios=[0.84, 0.16],
        wspace=0.02,
    )
    ax_reg = fig.add_subplot(regional_grid[0])
    ax_reg_labels = fig.add_subplot(regional_grid[1], sharey=ax_reg)

    # Reuse the archive figure's cartographic grammar, but render current full5862 data.
    _style_cartopy_world(ax_map)
    ax_map.set_anchor("N")
    map_df = df.sort_values("gciri")
    population = map_df["pop_2025"].clip(lower=0).to_numpy(dtype=float)
    population_ref = max(float(np.nanquantile(population, 0.99)), 1.0)
    symbol_sizes = 2.8 + 17.0 * np.sqrt(np.clip(population / population_ref, 0, 1))
    scatter = ax_map.scatter(
        map_df["lon"],
        map_df["lat"],
        c=map_df["gciri"],
        s=symbol_sizes,
        cmap=_score_cmap(mpl),
        vmin=0,
        vmax=100,
        marker="o",
        edgecolors="white",
        linewidths=0.10,
        alpha=0.86,
        transform=ccrs.PlateCarree(),
        rasterized=True,
        zorder=3,
    )
    cbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.ax.text(
        0.5,
        0.5,
        "GCIRI",
        transform=cbar.ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="#222222",
    )
    cax.tick_params(labelsize=6.5, length=2)
    cbar.outline.set_linewidth(0.4)

    legend_values = [100_000, 1_000_000, 10_000_000]
    legend_sizes = 2.8 + 17.0 * np.sqrt(np.clip(np.asarray(legend_values) / population_ref, 0, 1))
    for size, label in zip(legend_sizes, ["0.1M", "1M", "10M"]):
        ax_map.scatter(
            [], [], s=size, facecolor="#AAB4BA", edgecolor="white", linewidth=0.25,
            label=label,
        )
    pop_legend = ax_map.legend(
        loc="center left",
        bbox_to_anchor=(0.018, 0.36),
        ncol=1,
        title="Population",
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#D0D5D8",
        handletextpad=0.45,
        labelspacing=0.48,
        borderpad=0.45,
    )
    pop_legend.get_frame().set_linewidth(0.4)
    ax_map.text(
        0.006,
        1.015,
        "a",
        transform=ax_map.transAxes,
        fontsize=10,
        fontweight="bold",
        color="#111111",
        va="bottom",
        ha="left",
    )

    longitude_profile = _binned_spatial_profile(
        df,
        "lon",
        np.linspace(-180, 180, 37),
    )
    latitude_profile = _binned_spatial_profile(
        df,
        "lat",
        np.linspace(-90, 90, 37),
    )
    _draw_longitude_profile(ax_lon, longitude_profile, mpl)
    _draw_latitude_profile(ax_lat, latitude_profile, mpl)
    panel_label(ax_lon, "b")
    panel_label(ax_lat, "c")

    _global_distribution_anatomy(ax_hist, ax_hist_quantile, df["gciri"].to_numpy(dtype=float), mpl)
    panel_label(ax_hist, "d")

    region_order = [REGION_LABELS[i] for i in range(1, 7)]
    valid = df.dropna(subset=["region_group"])
    shares = _tier_share(valid, "region_group", region_order)
    region_means = valid.groupby("region_group")["gciri"].mean().reindex(region_order)
    sorted_index = region_means.sort_values(ascending=False).index.tolist()
    shares = shares.reindex(sorted_index)
    counts = valid.groupby("region_group").size().reindex(sorted_index)
    region_means = region_means.reindex(sorted_index)
    row_labels = [NAME_TO_SHORT[name] for name in sorted_index]
    region_distribution = _regional_rainclouds(
        ax_reg,
        ax_reg_labels,
        valid,
        sorted_index,
        row_labels,
        mpl,
    )
    panel_label(ax_reg, "e")

    sources = [
        _write_source(df[["city", "city_name", "iso3", "ucdb_id", "lat", "lon", "pop_2025", "gciri", "region_group"]], out, "fig2_city_gciri").name,
        _write_source(
            shares.assign(mean_gciri=region_means, n=counts).reset_index(),
            out,
            "fig2_region_tier_shares",
        ).name,
        _write_source(
            region_distribution,
            out,
            "fig2_region_distribution_summary",
        ).name,
        _write_source(
            longitude_profile,
            out,
            "fig2_longitude_profile",
        ).name,
        _write_source(
            latitude_profile,
            out,
            "fig2_latitude_profile",
        ).name,
    ]
    paths = _save_figure(fig, out, "fig2_global_gciri_distribution_formal", 5, sources)
    plt.close(fig)
    return paths


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled)


def _bootstrap_effects(north: pd.DataFrame, south: pd.DataFrame, metrics: list[tuple[str, str]], n_boot: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(20260712)
    rows: list[dict[str, float | str]] = []
    for col, label in metrics:
        a = north[col].to_numpy(dtype=float)
        b = south[col].to_numpy(dtype=float)
        observed = _cohens_d(a, b)
        boot = np.empty(n_boot)
        for idx in range(n_boot):
            boot[idx] = _cohens_d(rng.choice(a, len(a), replace=True), rng.choice(b, len(b), replace=True))
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append({"metric": label, "cohens_d": observed, "ci_low": float(lo), "ci_high": float(hi)})
    return pd.DataFrame(rows)


def _bootstrap_group_profiles(
    groups: list[tuple[str, pd.DataFrame]],
    metrics: list[tuple[str, str]],
    n_boot: int = 1000,
) -> pd.DataFrame:
    rng = np.random.default_rng(20260818)
    rows: list[dict[str, float | int | str]] = []
    for group, frame in groups:
        for col, label in metrics:
            values = frame[col].to_numpy(dtype=float)
            boot = np.empty(n_boot)
            for idx in range(n_boot):
                boot[idx] = rng.choice(values, len(values), replace=True).mean()
            lo, hi = np.quantile(boot, [0.025, 0.975])
            rows.append(
                {
                    "group": group,
                    "metric": label,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                }
            )
    return pd.DataFrame(rows)


def build_fig3(out: Path, df: pd.DataFrame, mpl, plt) -> list[Path]:
    import cartopy.crs as ccrs

    plot_df = df.dropna(subset=["north_south_proxy"]).copy()
    north = plot_df.loc[plot_df["north_south_proxy"] == "North proxy"]
    south = plot_df.loc[plot_df["north_south_proxy"] == "South proxy"]
    metrics = [
        ("gciri", "GCIRI"),
        ("absorptive_strength", "Absorptive"),
        ("response_strength", "Response"),
        ("restorative_strength", "Restorative"),
    ]
    effects = _bootstrap_effects(north, south, metrics)
    profiles = _bootstrap_group_profiles(
        [("South proxy", south), ("North proxy", north)], metrics
    )

    fig = plt.figure(figsize=(7.2, 7.2), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        12,
        height_ratios=[1.30, 1.0],
        left=0.065,
        right=0.965,
        top=0.985,
        bottom=0.105,
        hspace=0.31,
        wspace=1.55,
    )
    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
    ax_density = fig.add_subplot(gs[1, 0:5])
    ax_income = fig.add_subplot(gs[1, 5:8])
    ax_effect = fig.add_subplot(gs[1, 8:12])

    _style_cartopy_world(ax_map)
    for label in ["South proxy", "North proxy"]:
        color = PROXY_COLORS[label]
        part = plot_df.loc[plot_df["north_south_proxy"] == label]
        ax_map.scatter(
            part["lon"], part["lat"], s=compute_scatter_size(part["pop_2025"]) * 0.34,
            color=color, alpha=0.78, edgecolors="white", linewidths=0.08,
            transform=ccrs.PlateCarree(), rasterized=True,
            zorder=2,
            label=f"{PROXY_DISPLAY[label]} ({PROXY_INCOME[label]})",
        )
    ax_map.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=2,
        frameon=False,
        handlelength=0.9, columnspacing=1.8, markerscale=1.15, fontsize=6.6,
    )
    panel_label(ax_map, "a")

    x_grid = np.linspace(0, 100, 400)
    for group, frame in [("South proxy", south), ("North proxy", north)]:
        color = PROXY_COLORS[group]
        density = gaussian_kde(frame["gciri"].to_numpy())(x_grid)
        ax_density.fill_between(x_grid, density, color=color, alpha=0.30, linewidth=0)
        ax_density.plot(x_grid, density, color=color, linewidth=1.65, label=PROXY_DISPLAY[group])
    ax_density.set_xlim(0, 100)
    ax_density.set_xlabel("GCIRI")
    ax_density.set_ylabel("Density")
    ax_density.legend(
        loc="upper left", ncol=2,
        frameon=False, handlelength=1.1, columnspacing=1.0, fontsize=6.4,
    )
    ax_density.grid(True, axis="y", color="#D9D9D9", linewidth=0.35)
    ax_density.set_axisbelow(True)
    panel_label(ax_density, "b")

    income_arrays = [df.loc[df["income_group"] == income, "gciri"].to_numpy() for income in INCOME_ORDER]
    income_labels = ["High", "Upper-\nmiddle", "Lower-\nmiddle", "Low"]
    _raincloud(ax_income, income_arrays, income_labels, INCOME_COLORS, "GCIRI")
    ax_income.tick_params(axis="x", labelsize=6.5)
    panel_label(ax_income, "c")

    metric_order = [label for _, label in metrics]
    y = np.arange(len(metric_order))[::-1]
    for yi, metric in zip(y, metric_order):
        rows = profiles.loc[profiles["metric"] == metric].set_index("group")
        south_row = rows.loc["South proxy"]
        north_row = rows.loc["North proxy"]
        ax_effect.plot(
            [south_row["mean"], north_row["mean"]], [yi, yi],
            color="#B8C0C5", linewidth=1.2, zorder=1,
        )
        for group, row in [("South proxy", south_row), ("North proxy", north_row)]:
            ax_effect.errorbar(
                row["mean"], yi,
                xerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
                fmt="o", color=PROXY_COLORS[group], ecolor=PROXY_COLORS[group],
                markeredgecolor="white", markeredgewidth=0.35,
                capsize=2, markersize=4.6, linewidth=0.9, zorder=2,
            )
    ax_effect.set_yticks(y)
    ax_effect.set_yticklabels([])
    ax_effect.set_ylim(-0.6, len(metric_order) - 0.4)
    ax_effect.set_xlim(0, 100)
    ax_effect.set_xticks([0, 20, 40, 60, 80, 100])
    for yi, metric in zip(y, metric_order):
        ax_effect.text(2.5, yi, metric, ha="left", va="center", fontsize=6.7, color="#333333")
    ax_effect.set_xlabel("Mean score (0-100)")
    ax_effect.grid(True, axis="x", color="#D9D9D9", linewidth=0.35)
    ax_effect.set_axisbelow(True)
    panel_label(ax_effect, "d")

    summary = plot_df.groupby("north_south_proxy").agg(
        n=("gciri", "size"), mean_gciri=("gciri", "mean"), median_gciri=("gciri", "median"),
        mean_abs=("absorptive_strength", "mean"), mean_resp=("response_strength", "mean"), mean_rest=("restorative_strength", "mean")
    ).reset_index()
    sources = [
        _write_source(summary, out, "fig3_north_south_summary").name,
        _write_source(effects, out, "fig3_north_south_effects").name,
        _write_source(profiles, out, "fig3_north_south_profiles").name,
        _write_source(plot_df[["city", "iso3", "ucdb_id", "lat", "lon", "gciri", "income_group", "north_south_proxy"]], out, "fig3_city_north_south").name,
    ]
    paths = _save_figure(fig, out, "fig3_income_defined_north_south_divide_formal", 4, sources)
    plt.close(fig)
    return paths


def _region_order() -> list[str]:
    return [REGION_LABELS[i] for i in range(1, 7)]


def _region_weakness_composition(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    table = pd.crosstab(df["region_group"], df["weakest_dimension"], normalize="index").mul(100)
    return table.reindex(order).fillna(0)[DIM_ORDER]


def build_fig4(out: Path, df: pd.DataFrame, mpl, plt) -> list[Path]:
    import cartopy.crs as ccrs

    fig = plt.figure(figsize=(7.2, 7.35), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        12,
        height_ratios=[1.72, 0.30, 1.0],
        left=0.065,
        right=0.955,
        top=0.985,
        bottom=0.080,
        hspace=0.12,
        wspace=0.60,
    )
    maps = outer[0, :].subgridspec(2, 2, hspace=0.02, wspace=0.08)
    ax_abs = fig.add_subplot(maps[0, 0], projection=ccrs.Robinson())
    ax_resp = fig.add_subplot(maps[0, 1], projection=ccrs.Robinson())
    ax_rest = fig.add_subplot(maps[1, 0], projection=ccrs.Robinson())
    ax_imbalance = fig.add_subplot(maps[1, 1], projection=ccrs.Robinson())
    key_row = fig.add_subplot(outer[1, :])
    key_row.set_axis_off()
    cax = key_row.inset_axes([0.08, 0.38, 0.50, 0.26])
    imbalance_cax = key_row.inset_axes([0.68, 0.38, 0.24, 0.26])
    ax_heat = fig.add_subplot(outer[2, 0:6])
    ax_dist = fig.add_subplot(outer[2, 6:12])

    cmap = _score_cmap(mpl)
    for ax, col, letter in [
        (ax_abs, "absorptive_strength", "a"),
        (ax_resp, "response_strength", "b"),
        (ax_rest, "restorative_strength", "c"),
    ]:
        _draw_city_map(fig, ax, df, col, cmap, size_scale=0.34)
        panel_label(ax, letter)

    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cbar.set_label("Capacity strength (higher is better)", fontsize=7.5)
    cbar.ax.xaxis.set_label_position("top")
    cax.tick_params(labelsize=6.5)

    strengths = df[["absorptive_strength", "response_strength", "restorative_strength"]]
    imbalance = strengths.max(axis=1) - strengths.min(axis=1)
    imbalance_df = df.assign(capacity_imbalance=imbalance)
    imbalance_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "capacity_imbalance", ["#F3F0E8", "#D7A85B", "#8D3E35"], N=256
    )
    _style_cartopy_world(ax_imbalance)
    plot_df = imbalance_df.sort_values("capacity_imbalance")
    scatter = ax_imbalance.scatter(
        plot_df["lon"],
        plot_df["lat"],
        c=plot_df["capacity_imbalance"],
        s=compute_scatter_size(plot_df["pop_2025"]) * 0.34,
        cmap=imbalance_cmap,
        vmin=0,
        vmax=max(50.0, float(np.quantile(imbalance, 0.99))),
        alpha=0.80,
        edgecolors="white",
        linewidths=0.08,
        rasterized=True,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    imbalance_bar = fig.colorbar(scatter, cax=imbalance_cax, orientation="horizontal")
    imbalance_bar.set_label("Capacity imbalance", fontsize=6.4)
    imbalance_bar.ax.xaxis.set_label_position("top")
    imbalance_cax.tick_params(labelsize=5.8)
    panel_label(ax_imbalance, "d")

    region_order = _region_order()
    short_labels = [NAME_TO_SHORT[name] for name in region_order]
    profile = (
        df.groupby("region_group")[["absorptive_strength", "response_strength", "restorative_strength"]]
        .mean().reindex(region_order)
    )
    ax_heat.imshow(profile.to_numpy(), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax_heat.set_yticks(np.arange(6), short_labels)
    ax_heat.set_xticks(np.arange(3), ["Absorptive", "Response", "Restorative"])
    ax_heat.tick_params(axis="x", labelsize=7)
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            val = profile.iloc[i, j]
            ax_heat.text(j, i, f"{val:.0f}", ha="center", va="center", color="#111111", fontsize=7.5)
    # Shared 0-100 "Capacity strength" colorbar above (panels a-c) applies to this
    # heatmap too; cell values are annotated, so no separate colorbar is drawn here.
    panel_label(ax_heat, "e")

    arrays = [
        df["absorptive_strength"].to_numpy(),
        df["response_strength"].to_numpy(),
        df["restorative_strength"].to_numpy(),
    ]
    _raincloud(
        ax_dist,
        arrays,
        ["Absorptive", "Response", "Restorative"],
        [DIM_COLORS[dim] for dim in DIM_ORDER],
        "Capacity strength",
    )
    ax_dist.set_ylabel("")
    ax_dist.tick_params(axis="x", labelsize=6.7)
    panel_label(ax_dist, "f")

    capacity_source = df[[
        "city", "city_name", "iso3", "ucdb_id", "lat", "lon", "pop_2025", "region_group",
        "absorptive_strength", "response_strength", "restorative_strength", "weakest_dimension",
    ]].assign(capacity_imbalance=imbalance)
    sources = [
        _write_source(capacity_source, out, "fig4_city_capacity_strengths").name,
        _write_source(profile.reset_index(), out, "fig4_region_capacity_profiles").name,
        _write_source(
            pd.DataFrame(
                {
                    "capacity": ["absorptive", "response", "restorative"],
                    "mean": [values.mean() for values in arrays],
                    "median": [np.median(values) for values in arrays],
                    "q25": [np.quantile(values, 0.25) for values in arrays],
                    "q75": [np.quantile(values, 0.75) for values in arrays],
                }
            ),
            out,
            "fig4_global_capacity_distributions",
        ).name,
    ]
    paths = _save_figure(fig, out, "fig4_three_capacity_geography_formal", 6, sources)
    plt.close(fig)
    return paths


def _draw_weakness_map(ax, df: pd.DataFrame, mpl) -> None:
    import cartopy.crs as ccrs

    _style_cartopy_world(ax)
    for dimension in DIM_ORDER:
        part = df.loc[df["weakest_dimension"] == dimension]
        ax.scatter(
            part["lon"], part["lat"], s=compute_scatter_size(part["pop_2025"]) * 0.40,
            color=DIM_COLORS[dimension], alpha=0.80, edgecolors="white", linewidths=0.08,
            transform=ccrs.PlateCarree(), rasterized=True,
            label=f"{DIM_LABELS[dimension]} ({len(part):,})",
        )
    handles = [
        mpl.lines.Line2D(
            [], [], marker="o", linestyle="", markersize=4.5,
            markerfacecolor=DIM_COLORS[dimension], markeredgecolor="white",
            label=f"{DIM_LABELS[dimension]} ({len(df.loc[df['weakest_dimension'] == dimension]):,})",
        )
        for dimension in DIM_ORDER
    ]
    ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=3, frameon=True, framealpha=0.88, facecolor="white",
        edgecolor="none", handlelength=0.8, columnspacing=1.15,
    )


def _draw_constraint_map(ax, df: pd.DataFrame, mpl) -> None:
    import cartopy.crs as ccrs

    _style_cartopy_world(ax)
    for source_name in SOURCE_ORDER:
        part = df.loc[df["constraint_source"] == source_name]
        ax.scatter(
            part["lon"], part["lat"],
            s=compute_scatter_size(part["pop_2025"]) * 0.28,
            color=SOURCE_COLORS[source_name], alpha=0.78,
            edgecolors="white", linewidths=0.07,
            transform=ccrs.PlateCarree(), rasterized=True,
        )
    handles = [
        mpl.lines.Line2D(
            [], [], marker="o", linestyle="", markersize=3.8,
            markerfacecolor=SOURCE_COLORS[source_name], markeredgecolor="white",
            label=SOURCE_LABELS[source_name],
        )
        for source_name in SOURCE_ORDER
    ]
    ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=2, frameon=True, framealpha=0.88, facecolor="white",
        edgecolor="none", handlelength=0.7, columnspacing=0.8,
    )


def _weakness_composition_bars(
    ax,
    city: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_summary = (
        city["weakest_dimension"]
        .value_counts(normalize=True)
        .reindex(DIM_ORDER, fill_value=0.0)
        .mul(100.0)
        .rename_axis("weakest_dimension")
        .reset_index(name="share_pct")
    )
    region_summary = summarize_region_weakness(city)
    rows = [
        {
            "region": "Global",
            **global_summary.set_index("weakest_dimension")["share_pct"].to_dict(),
        }
    ]
    for region_name in _region_order():
        row = region_summary.loc[region_summary["region_group"] == region_name].iloc[0]
        rows.append(
            {
                "region": NAME_TO_SHORT[region_name],
                **{dimension: float(row[dimension]) for dimension in DIM_ORDER},
            }
        )
    composition = pd.DataFrame(rows)
    y = np.arange(len(composition))[::-1]
    left = np.zeros(len(composition))
    for dimension in DIM_ORDER:
        values = composition[dimension].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            height=0.68,
            color=DIM_COLORS[dimension],
            edgecolor="white",
            linewidth=0.35,
            label=DIM_LABELS[dimension],
        )
        for yi, start, value in zip(y, left, values):
            if value >= 8.0:
                ax.text(
                    start + value / 2.0,
                    yi,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    color="#111111",
                )
        left += values
    ax.set_yticks(y, composition["region"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Cities (%)")
    ax.grid(True, axis="x", color="#E1E5E7", linewidth=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        handlelength=0.7,
        columnspacing=0.65,
    )
    panel_label(ax, "b")
    return global_summary, region_summary


def _policy_type_matrix(ax, city_types: pd.DataFrame, mpl, fig) -> None:
    table = city_types.set_index(["constraint_source", "weakest_dimension"])
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    cmap = _score_cmap(mpl)
    max_n = max(1, int(city_types["n"].max()))
    y_positions = np.arange(len(SOURCE_ORDER))[::-1]
    for yi, source_name in zip(y_positions, SOURCE_ORDER):
        for xi, weakness in enumerate(DIM_ORDER):
            if (source_name, weakness) not in table.index:
                continue
            row = table.loc[(source_name, weakness)]
            size = 25.0 + 570.0 * float(row["n"]) / max_n
            ax.scatter(
                xi, yi, s=size, c=[float(row["mean_gciri"])],
                cmap=cmap, norm=norm, edgecolors="white", linewidths=0.75,
            )
            ax.text(
                xi, yi, str(row["policy_type_id"]), ha="center", va="center",
                fontsize=5.3, color="#111111", fontweight="bold",
            )
    ax.set_xticks(np.arange(3), ["Absorptive", "Response", "Restorative"])
    ax.set_yticks(y_positions, ["Hazard", "Capacity", "Compound", "Saturated"])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.55, len(SOURCE_ORDER) - 0.45)
    ax.grid(True, color="#E3E7E9", linewidth=0.45)
    ax.set_axisbelow(True)
    cax = ax.inset_axes([0.51, 1.025, 0.46, 0.035])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    colorbar.set_label("Mean GCIRI", labelpad=1)
    colorbar.ax.xaxis.set_label_position("top")
    cax.tick_params(length=1.8)
    panel_label(ax, "d")


def _scenario_intervals(ax, city: pd.DataFrame) -> pd.DataFrame:
    columns = [
        (
            "A+10",
            "Absorptive capacity +10 percentage points",
            "delta_absorptive_10pp_raw",
            DIM_COLORS["absorptive"],
        ),
        (
            "R+10",
            "Response capacity +10 percentage points",
            "delta_response_10pp_raw",
            DIM_COLORS["response"],
        ),
        (
            "T+10",
            "Restorative capacity +10 percentage points",
            "delta_restorative_10pp_raw",
            DIM_COLORS["restorative"],
        ),
        (
            "CAP-Q3",
            "All modifiable capacities to regional upper quartile",
            "capacity_headroom",
            "#5A6E7F",
        ),
        (
            "RISK-Q1",
            "Physical risk to regional lower quartile",
            "hazard_burden",
            "#A45A4F",
        ),
    ]
    y = np.arange(len(columns))[::-1]
    rows = []
    ax.axvline(0, color="#777777", linewidth=0.65, linestyle="--")
    for yi, (_code, full_label, column, color) in zip(y, columns):
        values = city[column].to_numpy(dtype=float)
        q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
        ax.plot([q05, q95], [yi, yi], color=color, linewidth=0.9, alpha=0.70)
        ax.plot([q25, q75], [yi, yi], color=color, linewidth=4.0, solid_capstyle="round")
        ax.scatter([median], [yi], s=25, color=color, edgecolors="white", linewidths=0.6, zorder=3)
        rows.append(
            {
                "scenario": full_label,
                "q05": q05,
                "q25": q25,
                "median": median,
                "q75": q75,
                "q95": q95,
                "negative_n": int(np.sum(values < -0.01)),
            }
        )
    ax.set_yticks(y, [code for code, _label, _column, _color in columns])
    ax.set_xlim(-3, 40)
    ax.set_xlabel("Modeled change in GCIRI")
    ax.grid(True, axis="x", color="#E1E5E7", linewidth=0.4)
    ax.set_axisbelow(True)
    panel_label(ax, "e")
    return pd.DataFrame(rows)


def _select_country_portfolios(country: pd.DataFrame) -> pd.DataFrame:
    eligible = country.loc[country["country_portfolio"] != "sparse-evidence"].copy()
    selected = pd.concat(
        [
            eligible.nlargest(8, "n_cities"),
            eligible.loc[eligible["iso3"].isin(["EGY", "MEX"])],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["iso3"]).copy()
    selected = selected.sort_values("mean_gciri", ascending=False)
    selected["country_name_plot"] = selected["country_name"].replace(
        {"Democratic Republic of the Congo": "DR Congo", "United States of America": "United States"}
    )
    return selected


def _mix_with_white(mpl, color: str, share_pct: float) -> tuple[float, float, float]:
    base = np.asarray(mpl.colors.to_rgb(color), dtype=float)
    strength = 0.14 + 0.86 * np.clip(float(share_pct) / 100.0, 0.0, 1.0)
    return tuple(1.0 - strength * (1.0 - base))


def _country_portfolio_summary(
    ax_heat,
    country: pd.DataFrame,
    mpl,
) -> pd.DataFrame:
    selected = _select_country_portfolios(country)
    columns = [
        ("share_weak_absorptive", "A", DIM_COLORS["absorptive"]),
        ("share_weak_response", "R", DIM_COLORS["response"]),
        ("share_weak_restorative", "T", DIM_COLORS["restorative"]),
        ("share_hazard_dominant", "H", SOURCE_COLORS["hazard-dominant"]),
        ("share_capacity_dominant", "C", SOURCE_COLORS["capacity-dominant"]),
        ("share_compound", "X", SOURCE_COLORS["compound"]),
        ("share_benchmark_saturated", "M", SOURCE_COLORS["benchmark-saturated"]),
    ]
    y = np.arange(len(selected))
    for yi, (_idx, row) in zip(y, selected.iterrows()):
        for xi, (column, _label, color) in enumerate(columns):
            value = float(row[column])
            ax_heat.add_patch(
                mpl.patches.Rectangle(
                    (xi - 0.5, yi - 0.5),
                    1.0,
                    1.0,
                    facecolor=_mix_with_white(mpl, color, value),
                    edgecolor="white",
                    linewidth=0.65,
                )
            )
            ax_heat.text(xi, yi, f"{value:.0f}", ha="center", va="center", color="#111111")
    ax_heat.axvline(2.5, color="#737373", linewidth=0.7)
    ax_heat.set_xlim(-0.5, len(columns) - 0.5)
    ax_heat.set_ylim(len(selected) - 0.5, -0.5)
    ax_heat.set_xticks(np.arange(len(columns)), [label for _column, label, _color in columns])
    ax_heat.set_yticks(y, selected["country_name_plot"])
    ax_heat.set_xlabel("City composition (%)")
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    panel_label(ax_heat, "f")
    return selected


def build_fig5(out: Path, df: pd.DataFrame, mpl, plt) -> list[Path]:
    import cartopy.crs as ccrs

    city = derive_policy_pathways(df)
    city_types = summarize_city_types(city)
    country = summarize_country_portfolios(city)

    fig = plt.figure(figsize=compute_figure_size("D"), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        12,
        height_ratios=[1.18, 1.18, 0.98],
        left=0.045,
        right=0.985,
        top=0.985,
        bottom=0.075,
        hspace=0.34,
        wspace=1.05,
    )
    ax_map = fig.add_subplot(gs[0, 0:8], projection=ccrs.Robinson())
    ax_composition = fig.add_subplot(gs[0, 8:12])
    ax_constraint = fig.add_subplot(gs[1, 0:8], projection=ccrs.Robinson())
    ax_types = fig.add_subplot(gs[1, 8:12])
    ax_scenarios = fig.add_subplot(gs[2, 0:4])
    ax_country_heat = fig.add_subplot(gs[2, 5:12])

    ax_map.set_anchor("W")
    ax_constraint.set_anchor("W")
    _draw_weakness_map(ax_map, city, mpl)
    panel_label(ax_map, "a")
    global_weakness, region_weakness = _weakness_composition_bars(ax_composition, city)
    _draw_constraint_map(ax_constraint, city, mpl)
    panel_label(ax_constraint, "c")
    _policy_type_matrix(ax_types, city_types, mpl, fig)
    scenario_intervals = _scenario_intervals(ax_scenarios, city)
    selected_countries = _country_portfolio_summary(
        ax_country_heat,
        country,
        mpl,
    )

    city_source_cols = [
        "city", "city_name", "iso3", "ucdb_id", "lat", "lon", "pop_2025",
        "gciri", "absorptive_strength", "response_strength", "restorative_strength",
        "weakest_dimension", "constraint_source", "policy_type_id", "policy_type",
        "capacity_headroom", "hazard_burden", "total_modeled_headroom",
        "hazard_burden_share", "delta_absorptive_10pp_raw",
        "delta_response_10pp_raw", "delta_restorative_10pp_raw",
        "restorative_nonmonotonic_flag",
    ]
    sources = [
        _write_source(city[city_source_cols], out, "fig5_city_weakness_policy_types").name,
        _write_source(global_weakness, out, "fig5_global_weakness_shares").name,
        _write_source(region_weakness, out, "fig5_region_weakness_shares").name,
        _write_source(city_types, out, "fig5_city_policy_type_summary").name,
        _write_source(scenario_intervals, out, "fig5_scenario_intervals").name,
        _write_source(country, out, "fig5_country_policy_portfolios").name,
        _write_source(selected_countries, out, "fig5_selected_country_portfolios").name,
        _write_source(summarize_scenarios(city), out, "fig5_scenario_summary").name,
    ]
    paths = _save_figure(fig, out, "fig5_weakness_policy_pathways_formal", 6, sources)
    plt.close(fig)
    return paths


def _write_report(out: Path, df: pd.DataFrame, generated: list[Path]) -> None:
    report = [
        "# Formal full5862 main-figure report",
        "",
        f"- Primary table: `{FULL_GCIRI_REL.as_posix()}`",
        f"- Rows: {len(df)}",
        f"- Unique city keys: {df['city'].nunique()}",
        f"- Countries: {df['iso3'].nunique()}",
        "- Legacy figures: design references only; no legacy table is read.",
        "- Capacity polarity: absorptive and response fields reversed; restorative retained.",
        "- North-South construct: high-income versus low/middle-income proxy.",
        "- Fig. 4: continuous capacity geography plus capacity-imbalance and distribution panels.",
        "- Fig. 5: lowest-scoring capacity, regional shares, modeled headroom profiles, scenario headroom and country portfolios.",
        "- Policy pathways are formula-based screening priorities, not causal intervention effects or cost-benefit estimates.",
        "- Non-map grammar: diverging tails, rainclouds, heatmaps, regional composition bars, interval plots, a policy-type matrix and paired country composition summaries.",
        "",
        "## Generated",
        "",
    ]
    report.extend(f"- `{path.name}`" for path in generated if path.suffix == ".png")
    report.extend(["", "Fig. 1 is generated separately from the current pipeline schematic contract.", ""])
    (out / "_formal_figure_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    mpl, plt = load_plotting_backend()
    apply_style(mpl)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    df, _country = _prepare_data()
    out = _output_dir()
    generated: list[Path] = []
    generated.extend(build_fig2(out, df, mpl, plt))
    generated.extend(build_fig3(out, df, mpl, plt))
    generated.extend(build_fig4(out, df, mpl, plt))
    generated.extend(build_fig5(out, df, mpl, plt))
    _write_report(out, df, generated)
    print(out)


if __name__ == "__main__":
    main()
