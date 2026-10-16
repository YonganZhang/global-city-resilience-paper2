"""Extended Data figure: local hazards, development level and city size (2026-10-17).

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
from _shared import PROJECT_ROOT, apply_style, load_plotting_backend, panel_label

HEV_REL = Path("_pipelines/01_baseline_gciri/outputs/v28_full5862_20260702_020421/step_07_r_ph_city.csv")
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
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT_ROOT / "_figures" / f"gciri_full5862_extended_hazard_development_{stamp}"
    (out / "source_data").mkdir(parents=True)
    sources = [formal._write_source(quartiles, out, "edfig_hazard_quartiles_by_income").name,
               formal._write_source(deciles, out, "edfig_hazard_deciles_by_income").name,
               formal._write_source(sizes, out, "edfig_weakest_capacity_by_city_size").name,
               formal._write_source(compute_gciri_by_income_and_size(df), out, "edfig_gciri_by_income_and_city_size").name]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.0, 1.15, 1.05], "wspace": 0.55})
    ax = axes[0]
    for y, row in enumerate(quartiles.itertuples()):
        color = INCOME_COLORS[row.income_group]
        ax.plot([row.gciri_highest_hazard_quartile, row.gciri_lowest_hazard_quartile], [y, y], color="#9AA5AD", lw=1.2, zorder=1)
        ax.scatter(row.gciri_lowest_hazard_quartile, y, s=34, facecolor="white", edgecolor=color, lw=1.3, zorder=2)
        ax.scatter(row.gciri_highest_hazard_quartile, y, s=34, color=color, zorder=3)
        ax.text(row.gciri_lowest_hazard_quartile + 3, y, f"−{row.difference:.1f}", va="center", fontsize=6.5, color="#3A4650")
    ax.set_yticks(range(len(quartiles)), [INCOME_SHORT[i] for i in quartiles.income_group])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mean GCIRI")
    ax.scatter([], [], s=28, facecolor="white", edgecolor="#55626B", label="Weakest local hazards")
    ax.scatter([], [], s=28, color="#55626B", label="Strongest local hazards")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), frameon=False, fontsize=6.3, handletextpad=0.2, borderaxespad=0.1)
    panel_label(ax, "a")

    ax = axes[1]
    for income in INCOME_ORDER:
        cell = deciles[deciles.income_group == income]
        color = INCOME_COLORS[income]
        ax.fill_between(cell.median_local_hazard_ratio, cell.q25_gciri, cell.q75_gciri, color=color, alpha=0.12, lw=0)
        ax.plot(cell.median_local_hazard_ratio, cell.median_gciri, color=color, lw=1.4, marker="o", ms=2.4,
                label=INCOME_SHORT[income])
    ax.set_xscale("log")
    ax.set_xticks([0.6, 0.8, 1.0, 1.5, 2.0], ["0.6", "0.8", "1", "1.5", "2"])
    ax.set_xlim(0.62, 2.0)
    ax.minorticks_off()
    ax.set_ylim(0, 100)
    ax.set_xlabel("Local hazard ratio $H$")
    ax.set_ylabel("Median GCIRI")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), frameon=False, fontsize=6.3, ncol=2,
              columnspacing=0.8, handlelength=1.4, borderaxespad=0.1)
    panel_label(ax, "b")

    ax = axes[2]
    left = np.zeros(len(sizes))
    for dim in STRENGTHS:
        ax.barh(range(len(sizes)), sizes[dim], left=left, color=formal.DIM_COLORS[dim], height=0.62,
                label=formal.DIM_LABELS[dim], edgecolor="white", lw=0.4)
        for y, (value, start) in enumerate(zip(sizes[dim], left)):
            if value >= 9:
                ax.text(start + value / 2, y, f"{value:.0f}", ha="center", va="center", fontsize=6.3, color="white")
        left += sizes[dim].to_numpy()
    ax.set_yticks(range(len(sizes)), sizes["size_class"].astype(str))
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Cities by weakest capacity (%)")
    ax.set_ylabel("City population (millions)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False, fontsize=6.3,
              handlelength=0.9, columnspacing=0.6, handletextpad=0.3, borderaxespad=0.1)
    panel_label(ax, "c")

    paths = formal._save_figure(fig, out, "edfig_hazard_development_city_size", 3, sources)
    print(out)
    print(quartiles.round(1).to_string(index=False))
    print(sizes.round(1).to_string(index=False))
    for path in paths:
        if path.suffix == ".png":
            print(path)


if __name__ == "__main__":
    main()
