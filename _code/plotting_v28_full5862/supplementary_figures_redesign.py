"""SCI-style supplementary figure redesigns for the current full5862 paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

try:
    from . import supplementary_figures as sf
except ImportError:
    import supplementary_figures as sf


INK = sf.INK
MUTED = sf.MUTED
GRID = sf.GRID
BLUE = sf.BLUE
TEAL = sf.TEAL
GOLD = sf.GOLD
RED = sf.RED
PURPLE = sf.PURPLE
LIGHT_BLUE = sf.LIGHT_BLUE
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "gciri_score", ["#d73027", "#f46d43", "#fdae61", "#fee08b",
                    "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850"], N=256
)
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "gciri_density", ["#F4F8FA", LIGHT_BLUE, TEAL, BLUE], N=256
)


def _save(fig, output, stem, panels, sources, source_data):
    sf.save_supplementary_figure(
        fig, output, stem, panels=panels, input_sources=sources, source_data=source_data
    )


def _s1(data, output, plt):
    truth = data["truth"]
    summary = (
        truth.groupby(["iso3", "country_name"], as_index=False)
        .agg(
            n_cities=("gciri", "size"),
            mean_gciri=("gciri", "mean"),
            q10_gciri=("gciri", lambda v: v.quantile(0.10)),
            q90_gciri=("gciri", lambda v: v.quantile(0.90)),
            min_gciri=("gciri", "min"),
            max_gciri=("gciri", "max"),
        )
    )
    summary["range_gciri"] = summary["max_gciri"] - summary["min_gciri"]
    selected = summary.query("n_cities >= 10").nlargest(20, "range_gciri").sort_values("range_gciri")
    selected = selected.reset_index(drop=True)
    source = sf.save_source_data(output, "figs1_country_spread.csv", selected)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.3), gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(selected))
    norm = Normalize(0, 100)
    for yi, row in zip(y, selected.itertuples()):
        color = SCORE_CMAP(norm(row.mean_gciri))
        axes[0].hlines(yi, row.min_gciri, row.max_gciri, color="#C9D3D8", linewidth=2.0)
        axes[0].hlines(yi, row.q10_gciri, row.q90_gciri, color=color, linewidth=6.0)
        axes[0].scatter(row.mean_gciri, yi, s=22, color=GOLD, edgecolor="white", linewidth=0.5, zorder=3)
    axes[0].set_yticks(y, selected["iso3"])
    axes[0].set_xlabel("City GCIRI")
    axes[0].set_ylabel("Country (ISO3; n >= 10 cities)")
    axes[0].set_xlim(-2, 102)
    sf.style_axis(axes[0])
    sf.panel_label(axes[0], "a")
    eligible = summary.query("n_cities >= 10")
    axes[1].scatter(eligible["n_cities"], eligible["range_gciri"], s=24, c=eligible["mean_gciri"], cmap=SCORE_CMAP, vmin=0, vmax=100, edgecolor="white", linewidth=0.35, alpha=0.82)
    for row in selected.nlargest(6, "range_gciri").itertuples():
        axes[1].annotate(row.iso3, (row.n_cities, row.range_gciri), xytext=(3, 2), textcoords="offset points", fontsize=6.5, color=INK)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Number of cities (log scale)")
    axes[1].set_ylabel("Within-country GCIRI range")
    axes[1].set_ylim(0, max(70, selected["range_gciri"].max() + 4))
    sf.style_axis(axes[1])
    sf.panel_label(axes[1], "b")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.13, wspace=0.30)
    _save(fig, output, "figs1_within_country_gciri_spread", 2, [sf.FULL_GCIRI_REL], [source])
    plt.close(fig)


def _s2(data, output, plt):
    from matplotlib.colors import LinearSegmentedColormap

    country = data["country"].dropna(subset=["official_giri", "gciri_mean"]).copy()
    country["residual"] = country["gciri_mean"] - country["official_giri"]
    source = sf.save_source_data(output, "figs2_country_official_anchor.csv", country)
    fig, ax = plt.subplots(figsize=(5.9, 4.7))
    residual_cmap = LinearSegmentedColormap.from_list("signed_residual", [BLUE, "#D8E0E2", RED], N=256)
    points = ax.scatter(country["official_giri"], country["gciri_mean"], s=10 + 3 * np.sqrt(country["n_cities"].clip(lower=1)), c=country["residual"], cmap=residual_cmap, vmin=-20, vmax=20, edgecolor="white", linewidth=0.3, alpha=0.82)
    cbar = fig.colorbar(points, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("City-mean GCIRI minus official GIRI")
    for count in [5, 50, 500]:
        ax.scatter([], [], s=10 + 3 * np.sqrt(count), facecolor="#B8C2C8", edgecolor="white", linewidth=0.3, label=f"{count} cities")
    ax.legend(title="Assessed cities", frameon=False, fontsize=6.5, title_fontsize=6.5, loc="lower right", handletextpad=0.3, labelspacing=0.5)
    limits = [min(country["official_giri"].min(), country["gciri_mean"].min()) - 3, 103]
    ax.plot(limits, limits, color=MUTED, linestyle="--", linewidth=0.8)
    for row in pd.concat([country.nsmallest(4, "residual"), country.nlargest(4, "residual")]).drop_duplicates("iso3").itertuples():
        ax.annotate(row.iso3, (row.official_giri, row.gciri_mean), xytext=(3, 2), textcoords="offset points", fontsize=6.5, color=INK)
    rho = country[["official_giri", "gciri_mean"]].corr(method="spearman").iloc[0, 1]
    ax.text(0.03, 0.97, f"Spearman rho = {rho:.2f}; n = {len(country)}", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK)
    ax.set_xlim(limits); ax.set_ylim(limits)
    ax.set_xlabel("Official national GIRI"); ax.set_ylabel("Mean city GCIRI")
    ax.grid(color=GRID, linewidth=0.45, alpha=0.75)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.95, bottom=0.14)
    _save(fig, output, "figs1_country_mean_gciri_official_anchor", 1, [sf.FULL_GCIRI_REL, Path("_pipelines/01_baseline_gciri/outputs/v28_full5862_20260702_020421/analysis/country_vs_official_giri.csv")], [source])
    plt.close(fig)


def _s5(data, output, plt):
    truth = data["truth"][["city", "iso3", "r_ph_city", "gciri", "pop_2025"]].copy()
    truth["risk_bin"] = pd.cut(truth["r_ph_city"], bins=np.linspace(0, 1, 21), include_lowest=True)
    quantiles = truth.groupby("risk_bin", observed=True).agg(risk_mid=("r_ph_city", "mean"), gciri_q10=("gciri", lambda v: v.quantile(0.10)), gciri_median=("gciri", "median"), gciri_q90=("gciri", lambda v: v.quantile(0.90)), n=("gciri", "size")).reset_index(drop=True)
    source_a = sf.save_source_data(output, "figs5_city_physical_risk_gciri.csv", truth)
    source_b = sf.save_source_data(output, "figs5_risk_bin_quantiles.csv", quantiles)
    fig, ax = plt.subplots(figsize=(6.3, 4.35))
    hb = ax.hexbin(truth["r_ph_city"], truth["gciri"], gridsize=42, mincnt=1, cmap=DENSITY_CMAP, linewidths=0, alpha=0.84)
    ax.fill_between(quantiles["risk_mid"], quantiles["gciri_q10"], quantiles["gciri_q90"], color=GOLD, alpha=0.20, linewidth=0)
    ax.plot(quantiles["risk_mid"], quantiles["gciri_median"], color=RED, linewidth=1.6)
    cbar = fig.colorbar(hb, ax=ax, orientation="vertical", pad=0.02, fraction=0.05)
    cbar.set_label("Cities per hexagon")
    ax.set_xlabel("Physical risk"); ax.set_ylabel("GCIRI"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-2, 102)
    ax.grid(color=GRID, linewidth=0.4, alpha=0.6)
    fig.subplots_adjust(left=0.12, right=0.90, top=0.95, bottom=0.14)
    _save(fig, output, "figs2_physical_risk_and_gciri", 1, [sf.FULL_GCIRI_REL], [source_a, source_b])
    plt.close(fig)


def _heat(ax, table, xvals, yvals, xlabel, ylabel, mpl, panel):
    image = ax.imshow(table.to_numpy(dtype=float), origin="lower", aspect="auto", cmap=SCORE_CMAP, vmin=0.75, vmax=1.0)
    ax.set_xticks(range(len(xvals)), [str(v) for v in xvals]); ax.set_yticks(range(len(yvals)), [str(v) for v in yvals])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); sf.style_axis(ax); sf.panel_label(ax, panel)
    return image


def _s6(data, output, plt):
    grid = data["sensitivity"].copy()
    source = sf.save_source_data(output, "figs6_threshold_sensitivity.csv", grid)
    pairs = [("low_threshold", "high_threshold", "Low threshold", "High threshold"), ("weakness_margin", "balance_spread", "Weakness margin", "Balance spread")]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.45), gridspec_kw={"width_ratios": [1, 1]})
    image = None
    for ax, (x, y, xlabel, ylabel), letter in zip(axes, pairs, ["a", "b"]):
        table = grid.pivot_table(index=y, columns=x, values="adjusted_rand_vs_baseline", aggfunc="mean").sort_index()
        image = _heat(ax, table, table.columns.tolist(), table.index.tolist(), xlabel, ylabel, None, letter)
    cbar = fig.colorbar(image, ax=axes, pad=0.03, fraction=0.035)
    cbar.set_label("Adjusted Rand index vs baseline")
    fig.subplots_adjust(left=0.09, right=0.91, top=0.94, bottom=0.18, wspace=0.48)
    _save(fig, output, "figs6_six_mode_threshold_sensitivity", 2, [sf.SENSITIVITY_REL], [source])
    plt.close(fig)


def _s7(data, output, plt):
    cluster = data["cluster"].copy(); bootstrap = data["bootstrap"].copy()
    source_a = sf.save_source_data(output, "figs7_cluster_validation.csv", cluster)
    source_b = sf.save_source_data(output, "figs7_bootstrap_cluster_stability.csv", bootstrap)
    fig, ax = plt.subplots(figsize=(6.2, 4.35))
    x = cluster["cluster_count"].to_numpy()
    silhouette = cluster["silhouette"].to_numpy()
    ari = cluster["adjusted_rand_vs_rule_modes"].to_numpy()
    nmi = cluster["normalized_mutual_info_vs_rule_modes"].to_numpy()
    ax.axvspan(5.75, 6.25, color=GOLD, alpha=0.18, zorder=0)
    ax.plot(x, silhouette, color=BLUE, linewidth=2.0, marker="o", markersize=6.5,
            markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=1.0,
            label="Silhouette", zorder=3)
    ax.plot(x, ari, color=PURPLE, linewidth=2.0, marker="s", markersize=6.0,
            markerfacecolor=PURPLE, markeredgecolor="white", markeredgewidth=1.0,
            label="ARI vs rule patterns", zorder=3)
    ax.plot(x, nmi, color=RED, linewidth=2.0, marker="^", markersize=6.5,
            markerfacecolor=RED, markeredgecolor="white", markeredgewidth=1.0,
            label="NMI vs rule patterns", zorder=3)
    ax.axvline(6, color=MUTED, linestyle=(0, (3, 3)), linewidth=0.9, zorder=1)
    ax.set_xlabel("Number of clusters"); ax.set_ylabel("Diagnostic score (0–1)")
    upper = max(0.40, float(np.nanmax([silhouette.max(), ari.max(), nmi.max()]))) + 0.03
    ax.set_ylim(0.20, upper); ax.set_xlim(x.min() - 0.25, x.max() + 0.25)
    ax.set_xticks(x); ax.legend(frameon=False, fontsize=8, loc="lower right", handlelength=2.2)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.75); sf.style_axis(ax)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=0.14)
    _save(fig, output, "figs3_cluster_count_validation", 1, [sf.CLUSTER_REL], [source_a])
    plt.close(fig)

    vals = [bootstrap["adjusted_rand_vs_full_k6"].dropna().to_numpy(), bootstrap["adjusted_rand_vs_rule_modes"].dropna().to_numpy(), bootstrap["normalized_mutual_info_vs_rule_modes"].dropna().to_numpy()]
    fig, ax = plt.subplots(figsize=(6.2, 4.35))
    y = np.array([2.0, 1.0, 0.0])
    violin = ax.violinplot(vals, positions=y, vert=False, widths=0.58, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(violin["bodies"], [TEAL, GOLD, RED]): body.set_facecolor(color); body.set_edgecolor("white"); body.set_linewidth(0.8); body.set_alpha(0.72)
    rng = np.random.default_rng(7)
    for yi, values, color in zip(y, vals, [TEAL, GOLD, RED]):
        jitter = rng.normal(yi, 0.035, size=len(values))
        ax.scatter(values, jitter, s=10, color=color, alpha=0.22, linewidth=0, zorder=2)
        q25, med, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.plot([q25, q75], [yi, yi], color=INK, linewidth=2.0, solid_capstyle="round", zorder=4)
        ax.scatter([med], [yi], s=28, color=INK, edgecolor="white", linewidth=0.8, zorder=5)
    ax.set_yticks(y, ["ARI vs full k=6", "ARI vs rule patterns", "NMI vs rule patterns"]); ax.set_xlabel("Bootstrap agreement index"); ax.set_xlim(0.2, 1.02); ax.set_ylim(-0.48, 2.48); sf.style_axis(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.75)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=0.16)
    _save(fig, output, "figs4_bootstrap_stability", 1, [sf.BOOTSTRAP_REL], [source_b])
    plt.close(fig)


def _s8(data, output, plt):
    policy_city = data["policy_city"].copy()
    capacity_cols = {
        "Absorptive": ("delta_absorptive_10pp_raw", GOLD),
        "Response": ("delta_response_10pp_raw", TEAL),
        "Restorative": ("delta_restorative_10pp_raw", BLUE),
    }
    intervention = policy_city[["city", "iso3", *[v[0] for v in capacity_cols.values()]]].copy()
    source_a = sf.save_source_data(output, "figs5_capacity_intervention_distributions.csv", intervention)
    long = intervention.melt(id_vars=["city", "iso3"], var_name="capacity", value_name="delta_gciri")
    long["capacity"] = long["capacity"].map({v[0]: k for k, v in capacity_cols.items()})
    fig, ax = plt.subplots(figsize=(6.4, 4.55))
    positions = np.arange(3)
    for pos, (label, (_, color)) in enumerate(capacity_cols.items()):
        values = long.loc[long["capacity"] == label, "delta_gciri"].dropna().to_numpy()
        violin = ax.violinplot(values, positions=[pos], vert=False, widths=0.56, showmeans=False, showmedians=False, showextrema=False)
        violin["bodies"][0].set_facecolor(color); violin["bodies"][0].set_edgecolor("white"); violin["bodies"][0].set_linewidth(0.8); violin["bodies"][0].set_alpha(0.72)
        q25, med, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.plot([q25, q75], [pos, pos], color=INK, linewidth=3.0, solid_capstyle="round", zorder=4)
        ax.scatter([med], [pos], color=INK, s=35, edgecolor="white", linewidth=0.8, zorder=5)
        rng = np.random.default_rng(100 + pos); sample = values[rng.choice(len(values), min(240, len(values)), replace=False)]
        ax.scatter(sample, rng.normal(pos, 0.045, len(sample)), color=color, s=7, alpha=0.16, linewidth=0, zorder=2)
    ax.axvline(0, color=MUTED, linewidth=0.9, linestyle=(0, (3, 3))); ax.set_yticks(positions, list(capacity_cols)); ax.set_xlabel("Modeled GCIRI change for a +10 pp capacity scenario"); ax.set_xlim(min(-1.0, float(long["delta_gciri"].min())) - 0.2, float(long["delta_gciri"].max()) + 0.5); sf.style_axis(ax); ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.75)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.96, bottom=0.16)
    _save(fig, output, "figs5_capacity_intervention_distributions", 1, [sf.POLICY_CITY_REL], [source_a])
    plt.close(fig)

    # Outline cities whose displayed ten-point restorative response is negative beyond the -0.01 tolerance.
    response = policy_city[["city", "iso3", "r_ph_city", "delta_restorative_10pp_raw", "weakest_dimension"]].dropna().copy()
    response["nonmonotonic_10pp"] = response["delta_restorative_10pp_raw"] < -0.01
    source_b = sf.save_source_data(output, "figs6_restorative_response_risk_gradient.csv", response)
    response["risk_bin"] = pd.cut(response["r_ph_city"], bins=np.linspace(0, 1, 21), include_lowest=True)
    trend = response.groupby("risk_bin", observed=True).agg(risk=("r_ph_city", "mean"), median=("delta_restorative_10pp_raw", "median"), q25=("delta_restorative_10pp_raw", lambda v: v.quantile(0.25)), q75=("delta_restorative_10pp_raw", lambda v: v.quantile(0.75))).dropna().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.55))
    hb = ax.hexbin(response["r_ph_city"], response["delta_restorative_10pp_raw"], gridsize=38, mincnt=1, cmap=DENSITY_CMAP, linewidths=0, alpha=0.82)
    ax.fill_between(trend["risk"], trend["q25"], trend["q75"], color=GOLD, alpha=0.22, linewidth=0)
    ax.plot(trend["risk"], trend["median"], color=BLUE, linewidth=2.2)
    flagged = response[response["nonmonotonic_10pp"]]
    if not flagged.empty: ax.scatter(flagged["r_ph_city"], flagged["delta_restorative_10pp_raw"], s=14, facecolor="none", edgecolor=RED, linewidth=0.8, alpha=0.65)
    cbar = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.05); cbar.set_label("Cities per hexagon")
    ax.axhline(0, color=MUTED, linewidth=0.9, linestyle=(0, (3, 3))); ax.set_xlabel("Physical risk"); ax.set_ylabel("Restorative +10 pp modeled GCIRI change"); ax.set_xlim(-0.02, 1.02); sf.style_axis(ax); ax.grid(color=GRID, linewidth=0.45, alpha=0.6)
    fig.subplots_adjust(left=0.14, right=0.88, top=0.96, bottom=0.15)
    _save(fig, output, "figs6_restorative_response_risk_gradient", 1, [sf.POLICY_CITY_REL], [source_b])
    plt.close(fig)

    landscape = policy_city[["city", "iso3", "gciri", "capacity_headroom", "hazard_burden", "weakest_dimension", "total_modeled_headroom", "hazard_burden_share"]].dropna().copy()
    source_c = sf.save_source_data(output, "figs7_policy_pathway_landscape.csv", landscape)
    fig, ax = plt.subplots(figsize=(6.4, 4.55))
    colors = {"absorptive": GOLD, "response": TEAL, "restorative": BLUE}
    rng = np.random.default_rng(19)
    for label, color in colors.items():
        subset = landscape[landscape["weakest_dimension"].str.lower() == label]
        if len(subset) > 650: subset = subset.iloc[rng.choice(len(subset), 650, replace=False)]
        ax.scatter(subset["hazard_burden"], subset["capacity_headroom"], s=10 + 28 * np.clip(subset["gciri"] / 100, 0, 1), color=color, alpha=0.58, edgecolor="white", linewidth=0.25, label=label.capitalize(), zorder=3)
    ax.axvline(landscape["hazard_burden"].median(), color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.axhline(landscape["capacity_headroom"].median(), color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xlabel("Hazard burden"); ax.set_ylabel("Capacity headroom"); ax.set_xlim(0, landscape["hazard_burden"].max() * 1.05); ax.set_ylim(0, landscape["capacity_headroom"].max() * 1.05); ax.legend(frameon=False, fontsize=8, loc="upper right"); sf.style_axis(ax); ax.grid(color=GRID, linewidth=0.45, alpha=0.6)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.15)
    _save(fig, output, "figs7_policy_pathway_landscape", 1, [sf.POLICY_CITY_REL], [source_c])
    plt.close(fig)


BUILDERS = [_s2, _s5, _s7, _s8]


def render(output_dir: Path | None = None) -> Path:
    mpl, plt = sf.load_plotting_backend(); sf.apply_style(mpl); data = sf.load_supplementary_inputs(); output = sf.build_output_dir(output_dir)
    for builder in BUILDERS: builder(data, output, plt)
    (output / "selection_manifest.md").write_text(
        "# Refined supplementary figure package\n\n"
        "Included as standalone figures: S1, S2, S3, S4, S5, S6 and S7.\n\n"
        "The earlier within-country spread figure is excluded because it duplicates\n"
        "the main-text country-scale heterogeneity figure.\n\n"
        "S6 is retained as a sensitivity table rather than a formal figure because\n"
        "the three-by-three parameter grid has low scientific payload.\n\n"
        "S10 is excluded from the figure package because the US FEMA resilience\n"
        "correlation is near zero and is better reported as a limitation/table.\n"
        "S3, S4 and S9 remain internal audits.\n\n"
        "All figures use the current full5862 truth table and include source-data sidecars.\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render refined current-data supplementary figures.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(); output = render(args.output_dir); print(output.relative_to(sf.PROJECT_ROOT) if output.is_relative_to(sf.PROJECT_ROOT) else output)


if __name__ == "__main__": main()
