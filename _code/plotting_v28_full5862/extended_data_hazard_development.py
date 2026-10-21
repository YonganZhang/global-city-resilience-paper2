"""Extended Data figure: local hazards, development level and city size (2026-10-17).

Layout from Gemini candidate v3 selected by the author on 2026-10-17 (job 20261017T163011__gemini__1003864, with
Claude QA): a) hazard quartiles by income group, b) hazard deciles, c) GCIRI by city size and income group (open
symbols: fewer than ten cities), d) weakest capacity by city size.

Supports Results 1 (the GCIRI cost of local hazards grows as development declines) and
Discussion (restorative weakness concentrates in smaller cities, absorptive weakness in
megacities). Reads the same prepared city table as the formal main figures.

Usage (project root): python3 _code/plotting_v28_full5862/extended_data_hazard_development.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import run_formal_main_figures as formal
from _shared import HEV_REL, PROJECT_ROOT, apply_style, load_plotting_backend, normalize_fonts, panel_label

INCOME_ORDER = formal.INCOME_ORDER
INCOME_SHORT = {"High income": "High", "Upper middle income": "Upper-middle",
                "Lower middle income": "Lower-middle", "Low income": "Low"}
INCOME_COLORS = dict(zip(INCOME_ORDER, formal.INCOME_COLORS))
SIZE_BINS = [0, 3e5, 1e6, 5e6, 1e7, np.inf]
SIZE_LABELS = ["0.1–0.3", "0.3–1", "1–5", "5–10", "≥10"]
STRENGTHS = {"absorptive": "absorptive_strength", "response": "response_strength",
             "restorative": "restorative_strength"}


def load_city_table() -> pd.DataFrame:
    df, _ = formal._prepare_data()
    hev = pd.read_csv(PROJECT_ROOT / HEV_REL)[["city", "h_ratio"]]
    df = df.merge(hev, on="city", how="left", validate="one_to_one")
    if df["h_ratio"].isna().any() or len(df) != 5862:
        raise ValueError("Every city needs a local hazard ratio")
    df["size_class"] = pd.cut(df["pop_2025"], SIZE_BINS, right=False, labels=SIZE_LABELS)
    return df


def compute_hazard_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for income in INCOME_ORDER:
        group = df[df["income_group"] == income]
        quartile = pd.qcut(group["h_ratio"].rank(method="first"), 4, labels=False)
        low, high = group.loc[quartile == 0, "gciri"].mean(), group.loc[quartile == 3, "gciri"].mean()
        rows.append({"income_group": income, "n": len(group), "gciri_lowest_hazard_quartile": low,
                     "gciri_highest_hazard_quartile": high, "difference": low - high})
    return pd.DataFrame(rows)


def compute_hazard_deciles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for income in INCOME_ORDER:
        group = df[df["income_group"] == income]
        decile = pd.qcut(group["h_ratio"].rank(method="first"), 10, labels=False)
        for d, cell in group.groupby(decile):
            rows.append({"income_group": income, "decile": int(d) + 1,
                         "median_local_hazard_ratio": cell["h_ratio"].median(),
                         "median_gciri": cell["gciri"].median(),
                         "q25_gciri": cell["gciri"].quantile(0.25), "q75_gciri": cell["gciri"].quantile(0.75),
                         "n": len(cell)})
    return pd.DataFrame(rows)


def compute_gciri_by_income_and_size(df: pd.DataFrame) -> pd.DataFrame:
    """Mean GCIRI by income group and city population class (source data for the Results 1 statistic)."""
    table = df.groupby(["income_group", "size_class"], observed=True)["gciri"].agg(["count", "mean"]).reset_index()
    table["income_group"] = pd.Categorical(table["income_group"], INCOME_ORDER, ordered=True)
    return table.sort_values(["income_group", "size_class"]).rename(columns={"count": "n", "mean": "mean_gciri"})


def compute_weakest_by_size(df: pd.DataFrame) -> pd.DataFrame:
    lowest = df[list(STRENGTHS.values())].idxmin(axis=1).map({v: k for k, v in STRENGTHS.items()})
    table = pd.crosstab(df["size_class"], lowest, normalize="index").reindex(columns=list(STRENGTHS)) * 100
    table["n"] = df["size_class"].value_counts().reindex(table.index).to_numpy()
    return table.reset_index()


def main() -> None:
    mpl, plt = load_plotting_backend()
    apply_style(mpl)
    mpl.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    df = load_city_table()
    quartiles, deciles, sizes = compute_hazard_quartiles(df), compute_hazard_deciles(df), compute_weakest_by_size(df)
    by_size = compute_gciri_by_income_and_size(df)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT_ROOT / "_figures" / f"gciri_full5862_extended_hazard_development_{stamp}"
    (out / "source_data").mkdir(parents=True)
    sources = [formal._write_source(quartiles, out, "edfig_hazard_quartiles_by_income").name,
               formal._write_source(deciles, out, "edfig_hazard_deciles_by_income").name,
               formal._write_source(sizes, out, "edfig_weakest_capacity_by_city_size").name,
               formal._write_source(by_size, out, "edfig_gciri_by_income_and_city_size").name]

    fig = plt.figure(figsize=(7.6, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.38, wspace=0.34)

    # =========================================================================
    # Panel a: Local hazard quartiles (Mean GCIRI in weakest vs strongest)
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    for y, row in enumerate(quartiles.itertuples()):
        color = INCOME_COLORS[row.income_group]
        # Connecting line
        ax_a.plot([row.gciri_highest_hazard_quartile, row.gciri_lowest_hazard_quartile],
                  [y, y], color="#9AA5AD", lw=1.5, zorder=1)
        # Weakest hazard quartile (open circle)
        ax_a.scatter(row.gciri_lowest_hazard_quartile, y, s=36, facecolor="white",
                     edgecolor=color, lw=1.5, zorder=3)
        # Strongest hazard quartile (filled circle)
        ax_a.scatter(row.gciri_highest_hazard_quartile, y, s=36, color=color,
                     edgecolor=color, lw=0.5, zorder=4)
        # Numerical gap annotation centered above connector
        x_mid = (row.gciri_highest_hazard_quartile + row.gciri_lowest_hazard_quartile) / 2
        ax_a.text(x_mid, y - 0.22, f"Δ = {row.difference:.1f}", ha="center", va="bottom",
                  fontsize=6.5, color="#2C343A")

    ax_a.set_yticks(range(len(quartiles)))
    ax_a.set_yticklabels([INCOME_SHORT[i] for i in quartiles.income_group], fontsize=7)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Mean GCIRI")
    ax_a.set_ylabel("Income group")
    ax_a.grid(True, axis="x", color="#E1E5E7", linewidth=0.4, alpha=0.8, zorder=0)

    # Clean legend in open upper-left quadrant
    ax_a.scatter([], [], s=32, facecolor="white", edgecolor="#55626B", lw=1.4,
                 label="Weakest hazard quartile (Q1)")
    ax_a.scatter([], [], s=32, color="#55626B", edgecolor="#55626B", lw=0.5,
                 label="Strongest hazard quartile (Q4)")
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=True,
                facecolor="white", edgecolor="none", framealpha=0.92,
                fontsize=6.3, handletextpad=0.3, borderpad=0.2)
    panel_label(ax_a, "a")

    # =========================================================================
    # Panel b: Local hazard deciles by income group
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    markers = {"High income": "o", "Upper middle income": "s",
               "Lower middle income": "^", "Low income": "v"}
    for income in INCOME_ORDER:
        cell = deciles[deciles.income_group == income]
        color = INCOME_COLORS[income]
        ax_b.fill_between(cell.median_local_hazard_ratio, cell.q25_gciri, cell.q75_gciri,
                          color=color, alpha=0.14, lw=0, zorder=1)
        ax_b.plot(cell.median_local_hazard_ratio, cell.median_gciri, color=color,
                  lw=1.4, marker=markers[income], ms=3.2, zorder=2, label=INCOME_SHORT[income])

    ax_b.set_xscale("log")
    ax_b.set_xticks([0.6, 0.8, 1.0, 1.5, 2.0])
    ax_b.set_xticklabels(["0.6", "0.8", "1.0", "1.5", "2.0"])
    ax_b.set_xlim(0.62, 2.05)
    ax_b.minorticks_off()
    ax_b.set_ylim(0, 100)
    ax_b.set_xlabel("Local hazard ratio $H$ (city to national mean)")
    ax_b.set_ylabel("Median GCIRI")
    ax_b.grid(True, color="#E1E5E7", linewidth=0.4, alpha=0.8, zorder=0)
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, title="Income group (b, c)",
                title_fontsize=6.3, fontsize=6.3, ncol=4, columnspacing=0.8, handlelength=1.6, handletextpad=0.3)
    panel_label(ax_b, "b")

    # =========================================================================
    # Panel c: City size, development and capacity bottlenecks
    # Left: Mean GCIRI by city-size class and income group
    # Right: Weakest capacity composition by city-size class
    # =========================================================================
    gs_c = gs[1, :].subgridspec(1, 2, wspace=0.30)

    # c1: GCIRI by city size & income group
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    for income in INCOME_ORDER:
        sub = by_size[by_size.income_group == income].reset_index(drop=True)
        color = INCOME_COLORS[income]
        ax_c1.plot(range(len(sub)), sub["mean_gciri"], color=color, lw=1.4, zorder=2)
        for x, row in sub.iterrows():  # open symbols: fewer than ten cities
            few = row["n"] < 10
            ax_c1.scatter(x, row["mean_gciri"], marker=markers[income], s=18, zorder=3, linewidths=1.0,
                          facecolor="white" if few else color, edgecolor=color)
            if few:
                ax_c1.text(x, row["mean_gciri"] + 3.0, f"n={int(row['n'])}", fontsize=5.6, color="#55626B", va="bottom", ha="center")

    ax_c1.set_xticks(range(len(SIZE_LABELS)))
    ax_c1.set_xticklabels(SIZE_LABELS, fontsize=7)
    ax_c1.set_ylim(0, 100)
    ax_c1.set_xlabel("City population (millions)")
    ax_c1.set_ylabel("Mean GCIRI")
    ax_c1.grid(True, color="#E1E5E7", linewidth=0.4, alpha=0.8, zorder=0)
    panel_label(ax_c1, "c")

    # c2: Weakest capacity by city size (vertical stacked bars)
    ax_c2 = fig.add_subplot(gs_c[0, 1])
    x_pos = np.arange(len(sizes))
    bar_width = 0.54
    bottom = np.zeros(len(sizes))
    
    for dim in STRENGTHS:
        values = sizes[dim].to_numpy()
        color = formal.DIM_COLORS[dim]
        label = formal.DIM_LABELS[dim]
        ax_c2.bar(x_pos, values, bottom=bottom, width=bar_width, color=color,
                  label=label, edgecolor="white", lw=0.5, zorder=2)
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v >= 11:
                ax_c2.text(x_pos[i], b + v / 2, f"{v:.0f}", ha="center", va="center",
                           fontsize=6.2, color="white", fontweight="bold", zorder=3)
        bottom += values

    # Sample sizes in x-tick labels
    n_labels = [f"{lbl}\n(n={n:,})" for lbl, n in zip(sizes["size_class"], sizes["n"])]
    ax_c2.set_xticks(x_pos)
    ax_c2.set_xticklabels(n_labels, fontsize=6.8)
    ax_c2.set_ylim(0, 100)
    ax_c2.set_yticks([0, 20, 40, 60, 80, 100])
    ax_c2.set_xlabel("City population (millions)")
    ax_c2.set_ylabel("Cities by weakest capacity (%)")
    ax_c2.grid(True, axis="y", color="#E1E5E7", linewidth=0.4, alpha=0.8, zorder=0)
    ax_c2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False,
                 fontsize=6.3, handlelength=1.0, columnspacing=0.8, handletextpad=0.3)
    panel_label(ax_c2, "d")

    normalize_fonts(fig)
    paths = formal._save_figure(fig, out, "edfig_hazard_development_city_size", 4, sources)
    print(out)
    print(quartiles.round(1).to_string(index=False))
    print(sizes.round(1).to_string(index=False))
    for path in paths:
        if path.suffix == ".png":
            print(path)


if __name__ == "__main__":
    main()
