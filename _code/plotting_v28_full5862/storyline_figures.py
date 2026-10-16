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
        FULL_GCIRI_REL,
        PROJECT_ROOT,
        apply_style,
        compute_scatter_size,
        load_plotting_backend,
        normalize_fonts,
        panel_label,
    )
    from .development_mode_analysis import MODE_ORDER
    from .run_formal_main_figures import (
        DIM_COLORS,
        DIM_LABELS,
        DIM_ORDER,
        NAME_TO_SHORT,
        _draw_city_map,
        _prepare_data,
        _region_order,
        _score_cmap,
        _style_cartopy_world,
        build_fig2,
        build_fig3,
    )
except ImportError:
    from _shared import (
        EXPECTED_FULL_ROWS,
        FULL_GCIRI_REL,
        PROJECT_ROOT,
        apply_style,
        compute_scatter_size,
        load_plotting_backend,
        normalize_fonts,
        panel_label,
    )
    from development_mode_analysis import MODE_ORDER
    from run_formal_main_figures import (
        DIM_COLORS,
        DIM_LABELS,
        DIM_ORDER,
        NAME_TO_SHORT,
        _draw_city_map,
        _prepare_data,
        _region_order,
        _score_cmap,
        _style_cartopy_world,
        build_fig2,
        build_fig3,
    )


DEFAULT_ANALYSIS_REL = Path(
    "_paper/gciri_full5862_spine_20260704/"
    "result4_mode_validation_20260718_194933"
)
OUTPUT_PREFIX = "gciri_full5862_storyline"

MODE_CODES = [f"M{index}" for index in range(1, 7)]
# Data keys stay M1-M6; the manuscript calls the six types resilience weakness patterns P1-P6.
PATTERN_LABELS = {code: f"P{code[1:]}" for code in MODE_CODES}
MODE_COLORS = {
    "M1": "#1E5A78",
    "M2": "#5A9D8F",
    "M3": "#D6A84B",
    "M4": "#D9793A",
    "M5": "#A63D40",
    "M6": "#7B5B8E",
}
MODE_SHORT_LABELS = {
    "M1": "Balanced high",
    "M2": "Imbalanced high",
    "M3": "Mixed transition",
    "M4": "Bottleneck transition",
    "M5": "Compound low",
    "M6": "Single-deficit low",
}
MODE_SHORT_PATHWAYS = {
    "M1": "Maintain gains; stress-test risks",
    "M2": "Protect gains; repair weaker capacity",
    "M3": "Coordinate multiple capacities",
    "M4": "Remove the dominant bottleneck",
    "M5": "Build foundational capacities",
    "M6": "Repair the severe deficit first",
}
MODE_CENTER_MARKERS = {
    "M1": "o",
    "M2": "^",
    "M3": "v",
    "M4": "s",
    "M5": "D",
    "M6": "P",
}
COUNTRY_MAP_COLORS = {
    **MODE_COLORS,
    "Mixed or limited coverage": "#9A7D61",
    "No data": "#E8CFA8",
}
COUNTRY_MAP_HATCHES = {
    "Mixed or limited coverage": "xxx",
    "No data": "///",
}
WEAKNESS_SHORT_LABELS = {
    "absorptive": "Abs.",
    "response": "Resp.",
    "restorative": "Rest.",
}
GROUP_ICON_COLORS = {
    "North proxy": "#E0B84F",
    "South proxy": "#3F4548",
    "East & Southeast Asia": "#B56C8B",
    "South, Central & West Asia": "#CE8740",
    "Americas": "#4C9990",
    "Europe": "#4A70A1",
    "Africa": "#B15D50",
    "Oceania": "#6F83AA",
}
COUNTRY_POLICY_MIN_CITIES = 3
COUNTRY_POLICY_MIN_LEAD_PP = 10.0
TERNARY_DISPLAY_REFERENCES = (0.95, 0.45, 0.55)
GROUP_ICON_WIDTH = 0.1845
GROUP_ICON_HEIGHT = 0.0984
GROUP_ICON_RIGHT_GAP = 0.018
SELECTED_COUNTRIES = [
    "CHN",
    "IND",
    "USA",
    "BRA",
    "MEX",
    "NGA",
    "IDN",
    "EGY",
    "RUS",
    "JPN",
]


def _resolve_repo_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must stay inside the project: {resolved}") from exc
    return resolved


def _build_output_dir(output_dir: Path | None) -> Path:
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = PROJECT_ROOT / "_figures" / f"{OUTPUT_PREFIX}_{stamp}"
    else:
        output = _resolve_repo_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_data").mkdir()
    (output / "qa").mkdir()
    return output


def _load_validated_mode_data(
    analysis_dir: Path,
    full_city: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    analysis_dir = _resolve_repo_path(analysis_dir)
    manifest_path = analysis_dir / "analysis_manifest.json"
    city_path = analysis_dir / "city_mode_table.csv"
    if not manifest_path.is_file() or not city_path.is_file():
        raise FileNotFoundError(
            f"Mode analysis package is incomplete: {analysis_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_rows") != EXPECTED_FULL_ROWS:
        raise ValueError("Mode analysis did not validate 5862 source rows")
    if manifest.get("truth_source") != FULL_GCIRI_REL.as_posix():
        raise ValueError("Mode analysis points to an unexpected truth source")
    if (
        manifest.get("classification", {}).get("type")
        != "transparent decision-oriented typology"
    ):
        raise ValueError("Mode analysis classification boundary is missing")

    modes = pd.read_csv(city_path)
    required = {
        "city",
        "mode_code",
        "development_mode",
        "screening_pathway",
        "weakest_dimension",
        "dominant_weakness",
        "capacity_spread",
        "weakness_gap",
        "gciri",
    }
    missing = sorted(required.difference(modes.columns))
    if missing:
        raise ValueError(f"{city_path} missing columns: {missing}")
    if len(modes) != EXPECTED_FULL_ROWS or modes["city"].nunique() != EXPECTED_FULL_ROWS:
        raise ValueError("Mode table must contain 5862 unique row-level city keys")
    if set(modes["mode_code"].unique()) != set(MODE_CODES):
        raise ValueError("Mode table must contain exactly M1-M6")

    full_keys = full_city[["city", "gciri"]].sort_values("city").reset_index(drop=True)
    mode_keys = modes[["city", "gciri"]].sort_values("city").reset_index(drop=True)
    if not full_keys["city"].equals(mode_keys["city"]):
        raise ValueError("Mode table city keys do not match the current full5862 table")
    if not np.allclose(full_keys["gciri"], mode_keys["gciri"], atol=1e-10):
        raise ValueError("Mode table GCIRI values do not match current full5862 data")

    mode_columns = [
        "city",
        "mode_code",
        "development_mode",
        "screening_pathway",
        "dominant_weakness",
        "capacity_spread",
        "weakness_gap",
    ]
    merged = full_city.merge(
        modes[mode_columns],
        on="city",
        how="left",
        validate="one_to_one",
    )
    if merged["mode_code"].isna().any():
        raise ValueError("Mode merge produced missing labels")
    return merged, manifest


def _write_source(
    data: pd.DataFrame,
    output_dir: Path,
    name: str,
) -> str:
    path = output_dir / "source_data" / f"{name}.csv"
    data.to_csv(path, index=False)
    return path.name


def _save_storyline_figure(
    fig,
    output_dir: Path,
    stem: str,
    panel_count: int,
    source_files: list[str],
    analysis_dir: Path,
) -> list[Path]:
    normalize_fonts(fig)
    paths = {
        "png": output_dir / f"{stem}.png",
        "pdf": output_dir / f"{stem}.pdf",
        "svg": output_dir / f"{stem}.svg",
        "tiff": output_dir / f"{stem}.tiff",
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
        "schema_version": "gciri-storyline-figure/v1",
        "stem": stem,
        "truth_source": FULL_GCIRI_REL.as_posix(),
        "source_rows": EXPECTED_FULL_ROWS,
        "unique_city_keys": EXPECTED_FULL_ROWS,
        "mode_analysis_dir": analysis_dir.relative_to(PROJECT_ROOT).as_posix(),
        "panels": panel_count,
        "canvas_size_inches": [
            round(float(value), 3) for value in fig.get_size_inches()
        ],
        "source_data": source_files,
        "formats": list(paths),
        "figure_status": "storyline-aligned formal candidate",
    }
    (output_dir / f"{stem}.source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return list(paths.values())


def _region_weakness_composition(
    city: pd.DataFrame,
    region_order: list[str],
) -> pd.DataFrame:
    table = pd.crosstab(
        city["region_group"],
        city["weakest_dimension"],
        normalize="index",
    ).mul(100.0)
    return table.reindex(region_order).reindex(columns=DIM_ORDER, fill_value=0.0)


def _draw_weakness_map(
    ax,
    city: pd.DataFrame,
    mpl,
    *,
    show_legend: bool = True,
) -> list:
    import cartopy.crs as ccrs

    _style_cartopy_world(ax)
    for dimension in DIM_ORDER:
        part = city.loc[city["weakest_dimension"].eq(dimension)]
        ax.scatter(
            part["lon"],
            part["lat"],
            s=compute_scatter_size(part["pop_2025"]) * 0.38,
            color=DIM_COLORS[dimension],
            alpha=0.80,
            edgecolors="white",
            linewidths=0.08,
            transform=ccrs.PlateCarree(),
            rasterized=True,
            zorder=2,
        )
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=4.6,
            markerfacecolor=DIM_COLORS[dimension],
            markeredgecolor="white",
            label=(
                f"{DIM_LABELS[dimension]} "
                f"({city['weakest_dimension'].eq(dimension).mean() * 100:.1f}%)"
            ),
        )
        for dimension in DIM_ORDER
    ]
    if show_legend:
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=3,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.90,
            handlelength=0.8,
            columnspacing=1.2,
        )
    return handles


def _draw_region_weakness_bars(
    ax,
    composition: pd.DataFrame,
) -> None:
    labels = [NAME_TO_SHORT[name] for name in composition.index]
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    for dimension in DIM_ORDER:
        values = composition[dimension].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            color=DIM_COLORS[dimension],
            height=0.67,
            edgecolor="white",
            linewidth=0.4,
            label=DIM_LABELS[dimension],
        )
        for index, (start, value) in enumerate(zip(left, values)):
            if value >= 9.0:
                ax.text(
                    start + value / 2,
                    index,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=6.2,
                )
        left += values
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Cities by weakest capacity (%)")
    ax.grid(True, axis="x", color="#E1E4E6", linewidth=0.35)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        frameon=False,
        handlelength=0.9,
        columnspacing=0.8,
    )


def _draw_region_capacity_profile(
    ax,
    profile: pd.DataFrame,
    *,
    show_region_labels: bool = True,
) -> None:
    labels = [NAME_TO_SHORT[name] for name in profile.index]
    y = np.arange(len(labels))
    values = profile[
        ["absorptive_strength", "response_strength", "restorative_strength"]
    ].to_numpy(dtype=float)
    for index, row in enumerate(values):
        ax.plot(
            [row.min(), row.max()],
            [index, index],
            color="#C7CDD1",
            linewidth=1.1,
            zorder=1,
        )
    for column, dimension in zip(range(3), DIM_ORDER):
        ax.scatter(
            values[:, column],
            y,
            s=24,
            color=DIM_COLORS[dimension],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
            label=DIM_LABELS[dimension],
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels if show_region_labels else [])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mean capacity strength")
    ax.grid(True, axis="x", color="#E1E4E6", linewidth=0.35)


def build_fig4_storyline(
    output_dir: Path,
    city: pd.DataFrame,
    analysis_dir: Path,
    mpl,
    plt,
) -> list[Path]:
    import cartopy.crs as ccrs

    fig = plt.figure(figsize=(7.2, 7.2), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        12,
        height_ratios=[2.22, 0.17, 1.02],
        left=0.045,
        right=0.985,
        top=0.985,
        bottom=0.075,
        hspace=0.12,
        wspace=0.10,
    )
    maps = grid[0, :].subgridspec(2, 2, hspace=0.02, wspace=0.02)
    cmap = _score_cmap(mpl)
    for position, column, letter in [
        ((0, 0), "absorptive_strength", "a"),
        ((0, 1), "response_strength", "b"),
        ((1, 0), "restorative_strength", "c"),
    ]:
        ax = fig.add_subplot(maps[position], projection=ccrs.Robinson())
        _draw_city_map(fig, ax, city, column, cmap, size_scale=0.29)
        panel_label(ax, letter)

    ax_weakness = fig.add_subplot(maps[1, 1], projection=ccrs.Robinson())
    weakness_handles = _draw_weakness_map(
        ax_weakness,
        city,
        mpl,
        show_legend=False,
    )
    panel_label(ax_weakness, "d")

    key_row = grid[1, :].subgridspec(
        1,
        2,
        width_ratios=[1.06, 0.94],
        wspace=0.18,
    )
    colorbar_slot = fig.add_subplot(key_row[0, 0])
    colorbar_slot.set_axis_off()
    cax = colorbar_slot.inset_axes([0.00, 0.39, 1.00, 0.22])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=0, vmax=100),
            cmap=cmap,
        ),
        cax=cax,
        orientation="horizontal",
    )
    colorbar.set_label("Capacity strength (higher is better)")
    cax.xaxis.set_label_position("top")
    cax.tick_params(labelsize=6.5)

    weakness_key = fig.add_subplot(key_row[0, 1])
    weakness_key.set_axis_off()
    weakness_key.legend(
        handles=weakness_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        handlelength=0.7,
        handletextpad=0.35,
        columnspacing=0.65,
        borderaxespad=0,
    )

    region_order = _region_order()
    composition = _region_weakness_composition(city, region_order)
    profile = (
        city.groupby("region_group")[
            ["absorptive_strength", "response_strength", "restorative_strength"]
        ]
        .mean()
        .reindex(region_order)
    )
    bottom = grid[2, :].subgridspec(
        1,
        2,
        width_ratios=[1.04, 0.96],
        wspace=0.42,
    )
    ax_composition = fig.add_subplot(bottom[0, 0])
    _draw_region_weakness_bars(ax_composition, composition)
    panel_label(ax_composition, "e")
    ax_profile = fig.add_subplot(bottom[0, 1])
    _draw_region_capacity_profile(ax_profile, profile, show_region_labels=True)
    panel_label(ax_profile, "f")

    city_source = city[
        [
            "city",
            "city_name",
            "iso3",
            "ucdb_id",
            "lat",
            "lon",
            "pop_2025",
            "region_group",
            "absorptive_strength",
            "response_strength",
            "restorative_strength",
            "weakest_dimension",
        ]
    ].copy()
    composition_source = (
        composition.rename_axis("region_group")
        .reset_index()
        .rename(
            columns={
                "absorptive": "absorptive_weakness_share_pct",
                "response": "response_weakness_share_pct",
                "restorative": "restorative_weakness_share_pct",
            }
        )
    )
    profile_source = profile.rename_axis("region_group").reset_index()
    sources = [
        _write_source(city_source, output_dir, "fig4_city_capacity_and_weakness"),
        _write_source(
            composition_source,
            output_dir,
            "fig4_region_weakness_composition",
        ),
        _write_source(profile_source, output_dir, "fig4_region_capacity_profiles"),
    ]
    paths = _save_storyline_figure(
        fig,
        output_dir,
        "fig4_capacity_weakness_geography_storyline",
        6,
        sources,
        analysis_dir,
    )
    plt.close(fig)
    return paths


def _draw_mode_map(
    ax,
    city: pd.DataFrame,
    mpl,
    mode_summary: pd.DataFrame | None = None,
    *,
    show_legend: bool = True,
) -> list:
    import cartopy.crs as ccrs

    _style_cartopy_world(ax)
    for mode_code in MODE_CODES:
        part = city.loc[city["mode_code"].eq(mode_code)]
        ax.scatter(
            part["lon"],
            part["lat"],
            s=compute_scatter_size(part["pop_2025"]) * 0.38,
            color=MODE_COLORS[mode_code],
            alpha=0.79,
            edgecolors="white",
            linewidths=0.06,
            transform=ccrs.PlateCarree(),
            rasterized=True,
            zorder=2,
        )
    shares = {}
    if mode_summary is not None:
        shares = mode_summary.set_index("mode_code")["share_pct"].to_dict()
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=4.2,
            markerfacecolor=MODE_COLORS[mode_code],
            markeredgecolor="white",
            label=(
                f"{PATTERN_LABELS[mode_code]} {shares[mode_code]:.1f}%"
                if mode_code in shares
                else PATTERN_LABELS[mode_code]
            ),
        )
        for mode_code in MODE_CODES
    ]
    if show_legend:
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=6,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.90,
            handlelength=0.6,
            columnspacing=0.6,
        )
    return handles


def _derive_country_dominant_modes(country: pd.DataFrame) -> pd.DataFrame:
    frame = country.copy()
    share_columns = [f"share_{mode.lower()}" for mode in MODE_CODES]
    missing = sorted(set(share_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Country mode portfolios lack columns: {missing}")
    mode_shares = frame[share_columns].astype(float)
    ordered = np.sort(mode_shares.to_numpy(dtype=float), axis=1)
    frame["leading_mode_code"] = (
        mode_shares.idxmax(axis=1).str.removeprefix("share_").str.upper()
    )
    frame["leading_mode_share_pct"] = mode_shares.max(axis=1)
    frame["mode_lead_pp"] = ordered[:, -1] - ordered[:, -2]
    reliable = (
        frame["n_cities"].ge(COUNTRY_POLICY_MIN_CITIES)
        & frame["mode_lead_pp"].ge(COUNTRY_POLICY_MIN_LEAD_PP)
    )
    frame["mapped_mode_code"] = frame["leading_mode_code"].where(
        reliable,
        "Mixed or limited coverage",
    )
    return frame


def _draw_country_mode_map(ax, country_modes: pd.DataFrame, mpl) -> list:
    import cartopy.crs as ccrs
    from cartopy.io import shapereader

    _style_cartopy_world(ax)
    by_iso = country_modes.set_index("iso3")["mapped_mode_code"].to_dict()
    shape_path = shapereader.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries",
    )
    for record in shapereader.Reader(shape_path).records():
        iso3 = record.attributes.get("ISO_A3")
        if not iso3 or iso3 == "-99":
            iso3 = record.attributes.get("ADM0_A3")
        mapped_mode = by_iso.get(iso3, "No data")
        ax.add_geometries(
            [record.geometry],
            crs=ccrs.PlateCarree(),
            facecolor=COUNTRY_MAP_COLORS[mapped_mode],
            edgecolor="#FFFFFF" if mapped_mode in MODE_CODES else "#5E5147",
            linewidth=0.22 if mapped_mode in MODE_CODES else 0.30,
            hatch=COUNTRY_MAP_HATCHES.get(mapped_mode),
            zorder=0.7,
        )
    handles = [
        mpl.patches.Patch(
            facecolor=COUNTRY_MAP_COLORS[category],
            edgecolor=("none" if category in MODE_CODES else "#5E5147"),
            linewidth=0.4,
            hatch=COUNTRY_MAP_HATCHES.get(category),
            label=PATTERN_LABELS.get(category, category),
        )
        for category in [
            *MODE_CODES,
            "Mixed or limited coverage",
            "No data",
        ]
    ]
    return handles


def _draw_mode_share_bars(
    ax,
    mode_summary: pd.DataFrame,
) -> None:
    frame = mode_summary.set_index("mode_code").reindex(MODE_CODES)
    y = np.arange(len(frame))
    values = frame["share_pct"].to_numpy(dtype=float)
    ax.barh(
        y,
        values,
        color=[MODE_COLORS[mode] for mode in MODE_CODES],
        height=0.68,
    )
    ax.set_yticks(
        y,
        MODE_CODES,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.23)
    ax.set_xlabel("Global city share (%)")
    ax.grid(True, axis="x", color="#E1E4E6", linewidth=0.35)
    for index, value in enumerate(values):
        ax.text(value + 0.55, index, f"{value:.1f}", va="center", fontsize=6.6)


def _draw_mode_ternary(
    ax,
    city: pd.DataFrame,
    mpl,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root3 = np.sqrt(3.0)
    border = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, root3 / 2], [0.0, 0.0]]
    )
    ax.plot(border[:, 0], border[:, 1], color="#333333", linewidth=0.85)
    for fraction in (0.20, 0.40, 0.60, 0.80):
        ax.plot(
            [fraction / 2, 1 - fraction / 2],
            [fraction * root3 / 2] * 2,
            color="#E0E4E6",
            linewidth=0.34,
        )
        ax.plot(
            [fraction, 0.5 + fraction / 2],
            [0, (1 - fraction) * root3 / 2],
            color="#E0E4E6",
            linewidth=0.34,
        )
        ax.plot(
            [1 - fraction, 0.5 - fraction / 2],
            [0, (1 - fraction) * root3 / 2],
            color="#E0E4E6",
            linewidth=0.34,
        )

    strength_columns = [
        "absorptive_strength",
        "response_strength",
        "restorative_strength",
    ]
    values = city[strength_columns].to_numpy(dtype=float)
    raw_composition = values / values.sum(axis=1, keepdims=True)
    composition_max = raw_composition.max(axis=0)
    display_reference = np.asarray(TERNARY_DISPLAY_REFERENCES, dtype=float)
    # Preserve raw compositions in the export; this column-wise transform only
    # spreads the plotted cloud when the three observed ranges differ sharply.
    reference_scaled = raw_composition / display_reference
    display_composition = reference_scaled / reference_scaled.sum(
        axis=1,
        keepdims=True,
    )
    raw_x = raw_composition[:, 1] + 0.5 * raw_composition[:, 2]
    raw_y = raw_composition[:, 2] * root3 / 2
    x = display_composition[:, 1] + 0.5 * display_composition[:, 2]
    y = display_composition[:, 2] * root3 / 2
    points = city[["city", "mode_code", "weakest_dimension"]].copy()
    points["absorptive_share"] = raw_composition[:, 0]
    points["response_share"] = raw_composition[:, 1]
    points["restorative_share"] = raw_composition[:, 2]
    points["raw_ternary_x"] = raw_x
    points["raw_ternary_y"] = raw_y
    points["display_absorptive_share"] = display_composition[:, 0]
    points["display_response_share"] = display_composition[:, 1]
    points["display_restorative_share"] = display_composition[:, 2]
    for index, capacity in enumerate(DIM_ORDER):
        points[f"{capacity}_above_display_reference"] = (
            raw_composition[:, index] > display_reference[index]
        )
    points["ternary_x"] = x
    points["ternary_y"] = y
    draw_order = np.random.default_rng(20260819).permutation(len(city))
    point_colors = city["weakest_dimension"].map(DIM_COLORS)
    if point_colors.isna().any():
        unknown = sorted(city.loc[point_colors.isna(), "weakest_dimension"].unique())
        raise ValueError(f"Unexpected weakest-capacity labels: {unknown}")
    ax.scatter(
        x[draw_order],
        y[draw_order],
        s=5.2,
        color=point_colors.iloc[draw_order].tolist(),
        alpha=0.30,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    centroids = (
        points.groupby("mode_code")
        .agg(
            ternary_x=("ternary_x", "mean"),
            ternary_y=("ternary_y", "mean"),
            city_count=("city", "size"),
        )
        .reindex(MODE_CODES)
        .reset_index()
    )
    centroids["center_definition"] = (
        "arithmetic mean of transformed ternary display coordinates"
    )
    label_positions = {
        "M1": (0.48, 0.39),
        "M2": (0.31, 0.31),
        "M3": (0.29, 0.22),
        "M4": (0.52, 0.15),
        "M5": (0.34, 0.13),
        "M6": (0.67, 0.35),
    }
    for _, row in centroids.iterrows():
        mode_code = row["mode_code"]
        ax.scatter(
            row["ternary_x"],
            row["ternary_y"],
            s=48,
            marker=MODE_CENTER_MARKERS[mode_code],
            color=MODE_COLORS[mode_code],
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )
        label_x, label_y = label_positions[mode_code]
        ax.annotate(
            PATTERN_LABELS[mode_code],
            xy=(row["ternary_x"], row["ternary_y"]),
            xytext=(label_x, label_y),
            ha="center",
            va="center",
            color=MODE_COLORS[mode_code],
            fontsize=7.0,
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-|>",
                "color": MODE_COLORS[mode_code],
                "linewidth": 0.72,
                "mutation_scale": 6.5,
                "shrinkA": 3.5,
                "shrinkB": 3.5,
            },
            zorder=5,
        )
    ax.text(
        0.005,
        -0.030,
        "Absorptive",
        ha="left",
        va="top",
        fontsize=7,
    )
    ax.text(
        0.995,
        -0.030,
        "Response",
        ha="center",
        va="top",
        fontsize=7,
    )
    ax.text(
        0.5,
        root3 / 2 + 0.010,
        "Restorative",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    vertex_values = [
        (0.080, 0.060, display_reference[0]),
        (0.920, 0.060, display_reference[1]),
        (0.500, root3 / 2 - 0.070, display_reference[2]),
    ]
    for value_x, value_y, value in vertex_values:
        ax.text(
            value_x,
            value_y,
            f"{value * 100:.0f}%",
            ha="center",
            va="center",
            fontsize=7.1,
            fontweight="bold",
            color="#32383B",
            bbox={
                "boxstyle": "square,pad=0.08",
                "facecolor": (1.0, 1.0, 1.0, 0.88),
                "edgecolor": "none",
            },
            zorder=6,
        )
    ax.set_xlim(-0.005, 1.065)
    ax.set_ylim(-0.060, root3 / 2 + 0.038)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_axis_off()
    weakness_handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=4.7,
            markerfacecolor=DIM_COLORS[dimension],
            markeredgecolor="white",
            label=(
                f"{WEAKNESS_SHORT_LABELS[dimension]} "
                f"{city['weakest_dimension'].eq(dimension).mean() * 100:.1f}%"
            ),
        )
        for dimension in DIM_ORDER
    ]
    ax.legend(
        handles=weakness_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        handlelength=0.7,
        handletextpad=0.3,
        columnspacing=0.75,
        borderaxespad=0,
    )
    display_limits = pd.DataFrame(
        {
            "capacity": DIM_ORDER,
            "observed_composition_max_pct": composition_max * 100.0,
            "display_reference_pct": np.round(display_reference * 100.0, 2),
            "observations_above_reference": (
                raw_composition > display_reference
            ).sum(axis=0),
            "geometry": (
                "raw share divided by rounded capacity-specific reference, "
                "then reclosed to sum to one for display coordinates"
            ),
        }
    )
    return centroids, points, display_limits


def _draw_mode_profile_and_pathways(
    ax,
    mode_summary: pd.DataFrame,
    cmap,
) -> None:
    frame = mode_summary.set_index("mode_code").reindex(MODE_CODES)
    columns = [
        "mean_gciri",
        "mean_absorptive",
        "mean_response",
        "mean_restorative",
    ]
    values = frame[columns].to_numpy(dtype=float)
    ax.imshow(values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xlim(-0.5, 9.10)
    ax.set_yticks(np.arange(6), MODE_CODES)
    ax.set_xticks(
        np.arange(4),
        ["GCIRI", "Abs.", "Resp.", "Rest."],
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{values[row, column]:.0f}",
                ha="center",
                va="center",
                color="#111111",
                fontsize=6.5,
            )
        ax.text(
            4.45,
            row,
            f"{frame.loc[MODE_CODES[row], 'share_pct']:.1f}%",
            ha="center",
            va="center",
            fontsize=6.4,
            color="#222222",
        )
        ax.text(
            5.25,
            row,
            MODE_SHORT_PATHWAYS[MODE_CODES[row]],
            ha="left",
            va="center",
            fontsize=6.4,
            color="#222222",
        )
    ax.text(
        4.45,
        -0.80,
        "Global share",
        ha="center",
        va="center",
        fontsize=7,
    )
    ax.text(
        5.25,
        -0.80,
        "Screening priority",
        ha="left",
        va="center",
        fontsize=7,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def _iter_geometry_exteriors(geometry):
    if geometry.geom_type == "Polygon":
        yield np.asarray(geometry.exterior.coords)
        return
    for part in getattr(geometry, "geoms", []):
        yield from _iter_geometry_exteriors(part)


def _record_iso3(record) -> str | None:
    iso3 = record.attributes.get("ISO_A3")
    if not iso3 or iso3 == "-99":
        iso3 = record.attributes.get("ADM0_A3")
    return iso3


def _draw_group_world_card(
    icon_ax,
    shape_records,
    iso_codes: set[str],
    mpl,
    *,
    highlight_color: str,
) -> None:
    icon_ax.set_facecolor("#FAFAF8")
    for record in shape_records:
        selected = _record_iso3(record) in iso_codes
        for vertices in _iter_geometry_exteriors(record.geometry):
            if len(vertices) < 3:
                continue
            icon_ax.add_patch(
                mpl.patches.Polygon(
                    vertices,
                    closed=True,
                    facecolor=(highlight_color if selected else "#E3E6E4"),
                    edgecolor=("#FFFFFF" if selected else "#C9CECB"),
                    linewidth=0.08,
                )
            )
    icon_ax.set_xlim(-180, 180)
    icon_ax.set_ylim(-90, 90)
    icon_ax.set_box_aspect(GROUP_ICON_HEIGHT / GROUP_ICON_WIDTH)
    icon_ax.set_aspect("auto")
    icon_ax.set_xticks([])
    icon_ax.set_yticks([])
    for spine in icon_ax.spines.values():
        spine.set_visible(False)
    icon_ax.add_patch(
        mpl.patches.Rectangle(
            (0, 0),
            1,
            1,
            transform=icon_ax.transAxes,
            facecolor="none",
            edgecolor="#8F9894",
            linewidth=0.55,
            clip_on=False,
            zorder=20,
        )
    )


def _add_group_silhouettes(
    ax,
    city: pd.DataFrame,
    order: list[str],
    mpl,
) -> pd.DataFrame:
    from cartopy.io import shapereader

    shape_path = shapereader.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries",
    )
    shape_records = list(shapereader.Reader(shape_path).records())
    membership_rows: list[dict[str, str]] = []
    for row_index, group in enumerate(order):
        if group in {"North proxy", "South proxy"}:
            selector = city["north_south_proxy"].eq(group)
        else:
            selector = city["region_group"].eq(group)
        iso_codes = set(city.loc[selector, "iso3"].dropna().astype(str))
        membership_rows.extend(
            {
                "group": group,
                "iso3": iso3,
                "highlight_color": GROUP_ICON_COLORS[group],
            }
            for iso3 in sorted(iso_codes)
        )
        y_center = 1.0 - (row_index + 0.5) / len(order)
        icon_ax = ax.inset_axes(
            [
                -(GROUP_ICON_WIDTH + GROUP_ICON_RIGHT_GAP),
                y_center - GROUP_ICON_HEIGHT / 2,
                GROUP_ICON_WIDTH,
                GROUP_ICON_HEIGHT,
            ],
            transform=ax.transAxes,
        )
        _draw_group_world_card(
            icon_ax,
            shape_records,
            iso_codes,
            mpl,
            highlight_color=GROUP_ICON_COLORS[group],
        )
    return pd.DataFrame(membership_rows)


def _draw_group_mode_composition(
    ax,
    north_south: pd.DataFrame,
    region: pd.DataFrame,
    city: pd.DataFrame,
    mpl,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [
            north_south.rename(columns={"north_south_proxy": "group"}).assign(
                section="Income proxy"
            ),
            region.rename(columns={"region_group": "group"}).assign(
                section="Region"
            ),
        ],
        ignore_index=True,
    )
    order = [
        "North proxy",
        "South proxy",
        "East & Southeast Asia",
        "South, Central & West Asia",
        "Americas",
        "Europe",
        "Africa",
        "Oceania",
    ]
    table = (
        combined.pivot(index="group", columns="mode_code", values="share_pct")
        .reindex(index=order, columns=MODE_CODES)
        .fillna(0.0)
    )
    labels = [
        "Global North",
        "Global South",
        "E & SE Asia",
        "S, C & W Asia",
        "Americas",
        "Europe",
        "Africa",
        "Oceania",
    ]
    y = np.arange(len(table))
    left = np.zeros(len(table))
    for mode_code in MODE_CODES:
        values = table[mode_code].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            color=MODE_COLORS[mode_code],
            height=0.67,
            edgecolor="white",
            linewidth=0.35,
        )
        left += values
    ax.axhline(1.5, color="#8F969A", linewidth=0.55)
    ax.set_yticks(y, labels)
    ax.yaxis.tick_right()
    ax.tick_params(
        axis="y",
        labelleft=False,
        labelright=True,
        left=True,
        right=True,
        pad=4,
    )
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("#33383B")
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_color("#33383B")
    ax.spines["right"].set_linewidth(0.65)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Cities by weakness pattern (%)")
    ax.grid(True, axis="x", color="#E1E4E6", linewidth=0.35)
    icon_membership = _add_group_silhouettes(ax, city, order, mpl)
    reference_tick = ax.get_yticklabels()[0]
    ax.text(
        -(GROUP_ICON_RIGHT_GAP + GROUP_ICON_WIDTH / 2),
        -0.035,
        "Location",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=reference_tick.get_fontsize(),
        fontfamily=reference_tick.get_fontfamily(),
        fontweight=reference_tick.get_fontweight(),
        fontstyle=reference_tick.get_fontstyle(),
        color=reference_tick.get_color(),
        clip_on=False,
    )
    return table.rename_axis("group").reset_index(), icon_membership


def _draw_country_mode_portfolios(
    ax,
    country: pd.DataFrame,
    mpl,
) -> pd.DataFrame:
    frame = (
        country.loc[country["iso3"].isin(SELECTED_COUNTRIES)]
        .set_index("iso3")
        .reindex(SELECTED_COUNTRIES)
        .reset_index()
    )
    if frame["country_name"].isna().any():
        missing = frame.loc[frame["country_name"].isna(), "iso3"].tolist()
        raise ValueError(f"Selected country portfolios are missing: {missing}")

    share_columns = [f"share_{mode.lower()}" for mode in MODE_CODES]
    values = frame[share_columns].to_numpy(dtype=float)
    ax.set_xlim(-0.5, 7.0)
    ax.set_ylim(len(frame) - 0.5, -0.5)
    for row in range(len(frame)):
        for column, mode_code in enumerate(MODE_CODES):
            alpha = 0.10 + 0.90 * values[row, column] / 100.0
            color = mpl.colors.to_rgba(MODE_COLORS[mode_code], alpha=alpha)
            rect = mpl.patches.Rectangle(
                (column - 0.48, row - 0.46),
                0.96,
                0.92,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.add_patch(rect)
            if values[row, column] >= 8.0:
                ax.text(
                    column,
                    row,
                    f"{values[row, column]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="#171717",
                )
        ax.text(
            6.25,
            row,
            f"{frame.loc[row, 'mean_gciri']:.0f}",
            ha="center",
            va="center",
            fontsize=6.5,
        )
    ax.set_xticks(np.arange(6), MODE_CODES)
    ax.xaxis.tick_top()
    ax.set_yticks(
        np.arange(len(frame)),
        frame["iso3"].tolist(),
    )
    ax.tick_params(length=0)
    ax.text(
        6.25,
        -0.83,
        "Mean\nGCIRI",
        ha="center",
        va="bottom",
        fontsize=6.4,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return frame


def build_fig5_storyline(
    output_dir: Path,
    city: pd.DataFrame,
    analysis_dir: Path,
    mpl,
    plt,
) -> list[Path]:
    import cartopy.crs as ccrs

    mode_summary = pd.read_csv(analysis_dir / "mode_summary.csv")
    north_south = pd.read_csv(
        analysis_dir / "north_south_mode_composition.csv"
    )
    region = pd.read_csv(analysis_dir / "region_mode_composition.csv")
    country = pd.read_csv(analysis_dir / "country_mode_portfolios.csv")
    country_modes = _derive_country_dominant_modes(country)
    fig = plt.figure(figsize=(7.2, 7.2), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        12,
        height_ratios=[0.72, 0.09, 1.35],
        left=0.060,
        right=0.975,
        top=0.985,
        bottom=0.065,
        hspace=0.00,
        wspace=0.10,
    )
    maps = grid[0, :].subgridspec(1, 2, wspace=0.025)
    ax_city_map = fig.add_subplot(maps[0, 0], projection=ccrs.Robinson())
    mode_handles = _draw_mode_map(
        ax_city_map,
        city,
        mpl,
        mode_summary,
        show_legend=False,
    )
    panel_label(ax_city_map, "a")

    ax_country_map = fig.add_subplot(maps[0, 1], projection=ccrs.Robinson())
    country_handles = _draw_country_mode_map(ax_country_map, country_modes, mpl)
    panel_label(ax_country_map, "b")

    keys = grid[1, :].subgridspec(1, 2, wspace=0.08)
    mode_key = fig.add_subplot(keys[0, 0])
    mode_key.set_axis_off()
    mode_legend = mode_key.legend(
        handles=mode_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.55),
        ncol=3,
        frameon=False,
        handlelength=0.6,
        handletextpad=0.35,
        columnspacing=0.65,
        borderaxespad=0,
    )
    country_key = fig.add_subplot(keys[0, 1])
    country_key.set_axis_off()
    country_legend = country_key.legend(
        handles=country_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.55),
        ncol=4,
        frameon=False,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.65,
        borderaxespad=0,
    )
    for legend in (mode_legend, country_legend):
        for text_object in legend.get_texts():
            label = text_object.get_text().split()[0]
            mode_code = f"M{label[1:]}" if label in PATTERN_LABELS.values() else label
            if mode_code in MODE_COLORS:
                text_object.set_color(MODE_COLORS[mode_code])
                text_object.set_fontweight("bold")

    bottom = grid[2, :].subgridspec(
        1,
        2,
        width_ratios=[1.00, 1.00],
        wspace=0.26,
    )
    ax_ternary = fig.add_subplot(bottom[0, 0])
    centroids, ternary_points, ternary_limits = _draw_mode_ternary(
        ax_ternary,
        city,
        mpl,
    )
    panel_label(ax_ternary, "c")

    ax_groups = fig.add_subplot(bottom[0, 1])
    group_table, group_icon_membership = _draw_group_mode_composition(
        ax_groups,
        north_south,
        region,
        city,
        mpl,
    )
    ax_groups.set_box_aspect(0.905)
    ax_groups.set_anchor("N")
    panel_label(ax_groups, "d")

    sources = [
        _write_source(
            city[
                [
                    "city",
                    "city_name",
                    "iso3",
                    "ucdb_id",
                    "lat",
                    "lon",
                    "pop_2025",
                    "gciri",
                    "absorptive_strength",
                    "response_strength",
                    "restorative_strength",
                    "mode_code",
                    "weakest_dimension",
                    "development_mode",
                    "screening_pathway",
                ]
            ],
            output_dir,
            "fig5_city_development_modes",
        ),
        _write_source(mode_summary, output_dir, "fig5_mode_summary"),
        _write_source(
            country_modes,
            output_dir,
            "fig5_country_dominant_modes",
        ),
        _write_source(centroids, output_dir, "fig5_mode_ternary_centroids"),
        _write_source(ternary_points, output_dir, "fig5_mode_ternary_cities"),
        _write_source(ternary_limits, output_dir, "fig5_ternary_display_limits"),
        _write_source(group_table, output_dir, "fig5_group_mode_composition"),
        _write_source(
            group_icon_membership,
            output_dir,
            "fig5_group_icon_country_membership",
        ),
    ]
    paths = _save_storyline_figure(
        fig,
        output_dir,
        "fig5_six_development_modes_and_pathways_storyline",
        4,
        sources,
        analysis_dir,
    )
    plt.close(fig)
    return paths


def build_fig5_supplementary(
    output_dir: Path,
    city: pd.DataFrame,
    analysis_dir: Path,
    mpl,
    plt,
) -> list[Path]:
    mode_summary = pd.read_csv(analysis_dir / "mode_summary.csv")
    country = pd.read_csv(analysis_dir / "country_mode_portfolios.csv")

    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        12,
        left=0.060,
        right=0.975,
        top=0.970,
        bottom=0.075,
        height_ratios=[0.78, 1.22],
        hspace=0.32,
    )
    ax_pathways = fig.add_subplot(grid[0, :])
    _draw_mode_profile_and_pathways(
        ax_pathways,
        mode_summary,
        _score_cmap(mpl),
    )
    panel_label(ax_pathways, "a")

    ax_country = fig.add_subplot(grid[1, :])
    selected_country = _draw_country_mode_portfolios(ax_country, country, mpl)
    panel_label(ax_country, "b")

    sources = [
        _write_source(mode_summary, output_dir, "figS11_mode_profiles"),
        _write_source(
            selected_country,
            output_dir,
            "figS11_selected_country_mode_portfolios",
        ),
    ]
    paths = _save_storyline_figure(
        fig,
        output_dir,
        "figS11_mode_profiles_and_country_examples",
        2,
        sources,
        analysis_dir,
    )
    plt.close(fig)
    return paths


def _write_report(
    output_dir: Path,
    analysis_dir: Path,
    generated: list[Path],
) -> None:
    report = [
        "# GCIRI four-storyline main-figure package",
        "",
        f"- Truth source: `{FULL_GCIRI_REL.as_posix()}`.",
        f"- Source rows and unique city keys: {EXPECTED_FULL_ROWS:,}.",
        (
            "- Mode analysis: "
            f"`{analysis_dir.relative_to(PROJECT_ROOT).as_posix()}`."
        ),
        "- Legacy figure files were not read as data.",
        "- Figures 2 and 3 were regenerated from current full5862 data.",
        "- Figure 4 links capacity geography to weakest-capacity composition.",
        (
            "- Figure 5 presents six transparent planning types through their "
            "city geography, reliable country-level dominant-mode summaries, "
            "weakest-capacity ternary geometry and regional composition."
        ),
        (
            "- Figure S11 retains mode fingerprints, screening priorities and "
            "selected-country portfolios as supporting evidence."
        ),
        "- The six types are not described as natural clusters.",
        "",
        "## Generated files",
        "",
    ]
    for path in sorted(generated):
        report.append(f"- `{path.name}`")
    report.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "Mode pathways are screening priorities. They do not establish "
                "causal effects, intervention feasibility, costs or optimal policy."
            ),
            "",
        ]
    )
    (output_dir / "_storyline_figure_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def run_storyline_figures(
    analysis_dir: Path = DEFAULT_ANALYSIS_REL,
    output_dir: Path | None = None,
) -> Path:
    analysis_dir = _resolve_repo_path(analysis_dir)
    output = _build_output_dir(output_dir)
    mpl, plt = load_plotting_backend()
    apply_style(mpl)
    full_city, _ = _prepare_data()
    city, _ = _load_validated_mode_data(analysis_dir, full_city)

    generated: list[Path] = []
    generated.extend(build_fig2(output, city, mpl, plt))
    generated.extend(build_fig3(output, city, mpl, plt))
    generated.extend(build_fig4_storyline(output, city, analysis_dir, mpl, plt))
    generated.extend(build_fig5_storyline(output, city, analysis_dir, mpl, plt))
    generated.extend(build_fig5_supplementary(output, city, analysis_dir, mpl, plt))
    _write_report(output, analysis_dir, generated)
    return output


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Render GCIRI Figures 2-5 for the confirmed four-storyline."
    )
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_REL)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(run_storyline_figures(args.analysis_dir, args.output_dir))


if __name__ == "__main__":
    run_cli()
