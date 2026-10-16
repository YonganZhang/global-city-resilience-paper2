from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

try:
    from . import run_formal_main_figures as formal
    from . import storyline_figures as storyline
    from ._shared import PROJECT_ROOT, apply_style, load_plotting_backend, panel_label
except ImportError:
    import run_formal_main_figures as formal
    import storyline_figures as storyline
    from _shared import PROJECT_ROOT, apply_style, load_plotting_backend, panel_label


REGION_ICON_COLOR = "#528FAD"
REGION_ICON_WIDTH = 0.265
REGION_ICON_HEIGHT = 0.112
REGION_ICON_GAP = 0.020


def _output_dir(output_dir: Path | None) -> Path:
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "_figures" / f"gciri_fig2_redesign_{stamp}"
    elif not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "source_data").mkdir()
    (output_dir / "qa").mkdir()
    return output_dir


def _region_order(city: pd.DataFrame) -> list[str]:
    return (
        city.dropna(subset=["region_group"])
        .groupby("region_group")["gciri"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )


def _region_short(region: str) -> str:
    reverse = {value: formal.REGION_SHORT[key] for key, value in formal.REGION_LABELS.items()}
    return reverse[region]


def _region_icon_color(region: str) -> str:
    """Use the same region identity colors as the approved Fig. 5 cards."""
    return storyline.GROUP_ICON_COLORS.get(region, REGION_ICON_COLOR)


def _draw_city_map(fig, ax, cax, city: pd.DataFrame, mpl) -> None:
    import cartopy.crs as ccrs

    formal._style_cartopy_world(ax)
    ax.set_anchor("N")
    frame = city.sort_values("gciri")
    population = frame["pop_2025"].clip(lower=0).to_numpy(dtype=float)
    population_ref = max(float(np.nanquantile(population, 0.99)), 1.0)
    sizes = 2.8 + 17.0 * np.sqrt(np.clip(population / population_ref, 0, 1))
    scatter = ax.scatter(
        frame["lon"],
        frame["lat"],
        c=frame["gciri"],
        s=sizes,
        cmap=formal._score_cmap(mpl),
        vmin=0,
        vmax=100,
        edgecolors="white",
        linewidths=0.10,
        alpha=0.86,
        transform=ccrs.PlateCarree(),
        rasterized=True,
        zorder=3,
    )
    colorbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    colorbar.set_ticks([0, 20, 40, 60, 80, 100])
    colorbar.ax.text(
        0.5,
        0.5,
        "GCIRI",
        transform=colorbar.ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="#222222",
        clip_on=False,
    )
    cax.tick_params(labelsize=6.5, length=2)
    colorbar.outline.set_linewidth(0.4)

    legend_values = np.asarray([100_000, 1_000_000, 10_000_000], dtype=float)
    legend_sizes = 2.8 + 17.0 * np.sqrt(np.clip(legend_values / population_ref, 0, 1))
    for size, label in zip(legend_sizes, ["0.1M", "1M", "10M"]):
        ax.scatter(
            [],
            [],
            s=size,
            facecolor="#AAB4BA",
            edgecolor="white",
            linewidth=0.25,
            label=label,
        )
    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(0.018, 0.36),
        title="Population",
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#D0D5D8",
        handletextpad=0.45,
        labelspacing=0.48,
        borderpad=0.45,
    )
    legend.get_frame().set_linewidth(0.4)
    panel_label(ax, "a")


def _add_region_icons(
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
    membership: list[dict[str, str]] = []
    for row_index, region in enumerate(order):
        iso_codes = set(
            city.loc[city["region_group"].eq(region), "iso3"].dropna().astype(str)
        )
        icon_color = _region_icon_color(region)
        membership.extend(
            {"region_group": region, "iso3": iso3, "highlight_color": icon_color}
            for iso3 in sorted(iso_codes)
        )
        y_center = 1.0 - (row_index + 0.5) / len(order)
        icon_ax = ax.inset_axes(
            [
                -(REGION_ICON_WIDTH + REGION_ICON_GAP),
                y_center - REGION_ICON_HEIGHT / 2,
                REGION_ICON_WIDTH,
                REGION_ICON_HEIGHT,
            ],
            transform=ax.transAxes,
        )
        storyline._draw_group_world_card(
            icon_ax,
            shape_records,
            iso_codes,
            mpl,
            highlight_color=icon_color,
        )
    reference_tick = ax.get_yticklabels()[0]
    ax.text(
        -(REGION_ICON_GAP + REGION_ICON_WIDTH / 2),
        -0.055,
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
    return pd.DataFrame(membership)


def _style_region_axis(ax, order: list[str]) -> None:
    y = np.arange(len(order))[::-1]
    ax.set_yticks(y, [_region_short(region) for region in order])
    ax.yaxis.tick_right()
    ax.tick_params(
        axis="y",
        labelleft=False,
        labelright=True,
        left=True,
        right=True,
        pad=4,
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.42, len(order) - 0.38)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("GCIRI")
    ax.grid(True, axis="x", color="#DDE1E3", linewidth=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    for side in ["left", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#33383B")
        ax.spines[side].set_linewidth(0.65)


def _draw_region_raincloud(
    ax,
    city: pd.DataFrame,
    order: list[str],
    mpl,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_grid = np.linspace(0, 100, 360)
    y = np.arange(len(order))[::-1]
    score_cmap = formal._score_cmap(mpl)
    rng = np.random.default_rng(20260819)
    rows: list[dict[str, float | int | str]] = []
    for baseline, region in zip(y, order):
        values = city.loc[city["region_group"].eq(region), "gciri"].dropna().to_numpy()
        q05, q10, q25, q50, q75, q90, q95 = np.quantile(
            values, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        )
        density = gaussian_kde(values, bw_method=0.24)(x_grid)
        density = density / max(float(density.max()), 1e-12) * 0.52
        color = score_cmap(float(np.clip(q50 / 100.0, 0, 1)))
        ax.fill_between(
            x_grid,
            baseline,
            baseline + density,
            color=color,
            alpha=0.38,
            linewidth=0,
            zorder=1,
        )
        ax.plot(x_grid, baseline + density, color=color, linewidth=0.85, zorder=2)
        sample_y = baseline - rng.uniform(0.10, 0.24, size=len(values))
        ax.scatter(
            values,
            sample_y,
            s=1.6,
            c=values,
            cmap=score_cmap,
            vmin=0,
            vmax=100,
            alpha=0.22,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        interval_y = baseline + 0.025
        ax.plot([q05, q95], [interval_y] * 2, color="#667078", linewidth=0.45)
        ax.plot([q10, q90], [interval_y] * 2, color="#4E5961", linewidth=0.85)
        ax.plot(
            [q25, q75],
            [interval_y] * 2,
            color="#26333B",
            linewidth=2.25,
            solid_capstyle="round",
        )
        ax.scatter(
            [q50],
            [interval_y],
            s=15,
            facecolor="white",
            edgecolor="#18252C",
            linewidth=0.65,
            zorder=4,
        )
        rows.append(
            {
                "region_group": region,
                "n": len(values),
                "mean_gciri": float(values.mean()),
                "p05": q05,
                "p10": q10,
                "p25": q25,
                "median": q50,
                "p75": q75,
                "p90": q90,
                "p95": q95,
            }
        )
    _style_region_axis(ax, order)
    membership = _add_region_icons(ax, city, order, mpl)
    return pd.DataFrame(rows), membership


def _draw_region_intervals(
    ax,
    city: pd.DataFrame,
    order: list[str],
    mpl,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = np.arange(len(order))[::-1]
    score_cmap = formal._score_cmap(mpl)
    rows: list[dict[str, float | int | str]] = []
    for baseline, region in zip(y, order):
        values = city.loc[city["region_group"].eq(region), "gciri"].dropna().to_numpy()
        q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
        color = score_cmap(float(np.clip(q50 / 100.0, 0, 1)))
        ax.plot([q05, q95], [baseline] * 2, color="#AEB8BE", linewidth=0.75)
        ax.plot(
            [q25, q75],
            [baseline] * 2,
            color=color,
            linewidth=4.8,
            solid_capstyle="round",
        )
        ax.scatter(
            [q50],
            [baseline],
            s=27,
            facecolor="white",
            edgecolor="#26333B",
            linewidth=0.75,
            zorder=3,
        )
        rows.append(
            {
                "region_group": region,
                "n": len(values),
                "mean_gciri": float(values.mean()),
                "p05": q05,
                "p25": q25,
                "median": q50,
                "p75": q75,
                "p95": q95,
            }
        )
    _style_region_axis(ax, order)
    membership = _add_region_icons(ax, city, order, mpl)
    return pd.DataFrame(rows), membership


def _draw_global_distribution(ax, city: pd.DataFrame, mpl) -> None:
    """Draw only the distribution; omit the old quantile line and white marker."""
    values = city["gciri"].to_numpy(dtype=float)
    score_cmap = formal._score_cmap(mpl)
    bins = np.linspace(0, 100, 45)
    _, edges, patches = ax.hist(
        values,
        bins=bins,
        density=True,
        color="#AADCE0",
        alpha=0.84,
        edgecolor="white",
        linewidth=0.22,
    )
    for left, right, patch in zip(edges[:-1], edges[1:], patches):
        patch.set_facecolor(score_cmap(((left + right) * 0.5) / 100.0))
    x_grid = np.linspace(0, 100, 500)
    density = gaussian_kde(values)(x_grid)
    ax.fill_between(x_grid, 0, density, color="#1E466E", alpha=0.075, linewidth=0)
    ax.plot(x_grid, density, color="#1E466E", linewidth=1.35)
    ax.set_ylabel("Density")
    ax.set_xlabel("GCIRI")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.35)
    ax.set_axisbelow(True)


def _north_south_profile(
    city: pd.DataFrame,
    coordinate: str,
    centers: np.ndarray,
    bandwidth: float,
    min_effective_n: float = 15.0,
) -> pd.DataFrame:
    """Create a high-resolution one-degree kernel profile."""
    rows: list[dict[str, float | int | str | bool]] = []
    for group in ["North proxy", "South proxy"]:
        group_df = city.loc[city["north_south_proxy"].eq(group)]
        values = pd.to_numeric(group_df[coordinate], errors="coerce").to_numpy(dtype=float)
        scores = pd.to_numeric(group_df["gciri"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values) & np.isfinite(scores)
        values = values[finite]
        scores = scores[finite]
        for center in centers:
            distance = np.abs(values - center)
            local = distance <= 3.0 * bandwidth
            local_scores = scores[local]
            local_distance = distance[local]
            weights = np.exp(-0.5 * (local_distance / bandwidth) ** 2)
            weight_sum = float(weights.sum())
            effective_n = (weight_sum**2 / float(np.square(weights).sum())) if weight_sum else 0.0
            row: dict[str, float | int | str | bool] = {
                "group": group,
                "coordinate": coordinate,
                "bin_left": float(center - 0.5),
                "bin_right": float(center + 0.5),
                "bin_center": float(center),
                "n": int(len(local_scores)),
                "effective_n": effective_n,
                "displayed": bool(effective_n >= min_effective_n),
            }
            if len(local_scores) and weight_sum:
                mean = float(np.average(local_scores, weights=weights))
                variance = float(np.average(np.square(local_scores - mean), weights=weights))
                se = float(np.sqrt(variance / effective_n)) if effective_n > 1 else np.nan
                row.update(
                    {
                        "mean_gciri": mean,
                        "ci_low": max(0.0, mean - 1.96 * se) if np.isfinite(se) else np.nan,
                        "ci_high": min(100.0, mean + 1.96 * se) if np.isfinite(se) else np.nan,
                    }
                )
            else:
                row.update({key: np.nan for key in ["mean_gciri", "ci_low", "ci_high"]})
            rows.append(row)
    return pd.DataFrame(rows)


def _draw_north_south_profile(
    ax,
    profile: pd.DataFrame,
    coordinate: str,
    show_legend: bool = False,
) -> None:
    colors = {"North proxy": "#376795", "South proxy": "#E76254"}
    display = {"North proxy": "High income", "South proxy": "Low and middle income"}
    wide = profile.pivot(index="bin_center", columns="group")
    centers = wide.index.to_numpy(dtype=float)
    north = wide.get(("mean_gciri", "North proxy"), pd.Series(index=wide.index, dtype=float)).to_numpy(dtype=float)
    south = wide.get(("mean_gciri", "South proxy"), pd.Series(index=wide.index, dtype=float)).to_numpy(dtype=float)
    north_lo = wide.get(("ci_low", "North proxy"), pd.Series(index=wide.index, dtype=float)).to_numpy(dtype=float)
    north_hi = wide.get(("ci_high", "North proxy"), pd.Series(index=wide.index, dtype=float)).to_numpy(dtype=float)
    south_lo = wide.get(("ci_low", "South proxy"), pd.Series(index=wide.index, dtype=float)).to_numpy(dtype=float)
    south_hi = wide.get(("ci_high", "South proxy"), pd.Series(index=wide.index, dtype=float)).to_numpy(dtype=float)
    displayed_north = wide.get(("displayed", "North proxy"), pd.Series(index=wide.index, dtype=bool)).to_numpy(dtype=bool)
    displayed_south = wide.get(("displayed", "South proxy"), pd.Series(index=wide.index, dtype=bool)).to_numpy(dtype=bool)
    north_ok = np.isfinite(north) & displayed_north
    south_ok = np.isfinite(south) & displayed_south
    valid = north_ok & south_ok

    def _segments(x: np.ndarray, y: np.ndarray, ok: np.ndarray):
        segments = []
        start = None
        for i, flag in enumerate(ok):
            if flag and start is None:
                start = i
            if start is not None and (not flag or i == len(ok) - 1):
                end = i if flag and i == len(ok) - 1 else i - 1
                if end >= start:
                    segments.append((x[start : end + 1], y[start : end + 1]))
                start = None
        return segments

    if coordinate == "lon":
        ax.fill_between(centers, north, south, where=valid, color="#B7C3C9", alpha=0.24, linewidth=0, zorder=1)
        labeled: set[str] = set()
        for group, mean, low, high in [
            ("North proxy", north, north_lo, north_hi),
            ("South proxy", south, south_lo, south_hi),
        ]:
            valid_group = north_ok if group == "North proxy" else south_ok
            ax.fill_between(centers, low, high, where=valid_group, color=colors[group], alpha=0.08, linewidth=0)
            for x_seg, y_seg in _segments(centers, mean, valid_group):
                label = display[group] if group not in labeled else "_nolegend_"
                ax.plot(x_seg, y_seg, color=colors[group], linewidth=0.72, label=label, zorder=3)
                labeled.add(group)
            ax.scatter(centers[valid_group], mean[valid_group], s=2.2, color=colors[group], edgecolor="none", zorder=4)
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel("Longitude (degree)")
        ax.set_ylabel("Mean GCIRI")
    else:
        ax.fill_betweenx(centers, north, south, where=valid, color="#B7C3C9", alpha=0.24, linewidth=0, zorder=1)
        labeled = set()
        for group, mean, low, high in [
            ("North proxy", north, north_lo, north_hi),
            ("South proxy", south, south_lo, south_hi),
        ]:
            valid_group = north_ok if group == "North proxy" else south_ok
            ax.fill_betweenx(centers, low, high, where=valid_group, color=colors[group], alpha=0.08, linewidth=0)
            for x_seg, y_seg in _segments(mean, centers, valid_group):
                label = display[group] if group not in labeled else "_nolegend_"
                ax.plot(x_seg, y_seg, color=colors[group], linewidth=0.72, label=label, zorder=3)
                labeled.add(group)
            ax.scatter(mean[valid_group], centers[valid_group], s=2.2, color=colors[group], edgecolor="none", zorder=4)
        ax.set_ylim(-60, 90)
        ax.set_yticks([-60, -30, 0, 30, 60, 90])
        ax.set_xlabel("Mean GCIRI")
        ax.set_ylabel("Latitude (degree)")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    ax.set_ylim(0, 100) if coordinate == "lon" else ax.set_xlim(0, 100)
    ax.grid(True, color="#DDE1E3", linewidth=0.32, zorder=0)
    ax.set_axisbelow(True)
    for spine in (["top", "right"] if coordinate == "lon" else ["top", "left"]):
        ax.spines[spine].set_visible(False)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.38), ncol=2, frameon=False, handlelength=1.4, columnspacing=1.2)


def _draw_region_tiers(
    ax,
    city: pd.DataFrame,
    order: list[str],
) -> pd.DataFrame:
    valid = city.dropna(subset=["region_group"])
    table = formal._tier_share(valid, "region_group", order)
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for tier in ["<40", "40-70", ">=70"]:
        values = table[tier].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            height=0.67,
            color=formal.TIER_COLORS[tier],
            edgecolor="white",
            linewidth=0.35,
            label=formal.TIER_LABELS[tier],
        )
        left += values
    ax.set_yticks(y, [_region_short(region) for region in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Cities (%)")
    ax.grid(True, axis="x", color="#DDE1E3", linewidth=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        frameon=False,
        handlelength=0.9,
        columnspacing=1.0,
    )
    return table.rename_axis("region_group").reset_index()


def _region_latitude_bands(city: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    edges = np.arange(-60, 91, 15, dtype=float)
    rows: list[dict[str, float | int | str]] = []
    for region in order:
        part = city.loc[city["region_group"].eq(region)]
        for left, right in zip(edges[:-1], edges[1:]):
            mask = part["lat"].ge(left) & part["lat"].lt(right)
            values = part.loc[mask, "gciri"]
            rows.append(
                {
                    "region_group": region,
                    "lat_left": left,
                    "lat_right": right,
                    "lat_center": (left + right) / 2,
                    "n": len(values),
                    "mean_gciri": values.mean() if len(values) >= 5 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _draw_region_latitude_heatmap(
    ax,
    bands: pd.DataFrame,
    order: list[str],
    mpl,
) -> None:
    centers = sorted(bands["lat_center"].unique())
    matrix = (
        bands.pivot(index="region_group", columns="lat_center", values="mean_gciri")
        .reindex(index=order, columns=centers)
        .to_numpy(dtype=float)
    )
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(
        masked,
        aspect="auto",
        interpolation="none",
        cmap=formal._score_cmap(mpl),
        vmin=0,
        vmax=100,
        extent=[centers[0] - 7.5, centers[-1] + 7.5, len(order) - 0.5, -0.5],
    )
    ax.set_yticks(np.arange(len(order)), [_region_short(region) for region in order])
    ax.set_xticks([-45, -15, 15, 45, 75])
    ax.set_xlabel("Latitude (degree)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color("#8F9894")
        spine.set_linewidth(0.55)
    cax = ax.inset_axes([0.22, 1.035, 0.56, 0.035])
    colorbar = ax.figure.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.set_ticks([0, 50, 100])
    colorbar.ax.tick_params(labelsize=6, length=1.5)
    colorbar.outline.set_linewidth(0.4)


def _build_candidate(
    out: Path,
    city: pd.DataFrame,
    mpl,
    plt,
    *,
    stem: str,
    regional_style: str,
) -> list[Path]:
    import cartopy.crs as ccrs

    order = _region_order(city)
    fig = plt.figure(figsize=(8.8, 8.7 if regional_style != "latitude" else 7.6), constrained_layout=False)
    if regional_style != "latitude":
        # Jointplot-style block: D is directly below A, E is directly to its
        # right, and the otherwise empty corner holds the compact colorbar.
        outer = fig.add_gridspec(
            2, 1,
            height_ratios=[2.95, 1.25],
            left=0.070, right=0.985, top=0.975, bottom=0.075,
            hspace=0.16,
        )
        # The approved oval Robinson map is retained; the adjacent profiles
        # are descriptive coordinate summaries rather than pixel-aligned axes.
        joint = outer[0].subgridspec(
            2, 2,
            width_ratios=[4.0, 0.315],
            height_ratios=[2.0, 0.21],
            wspace=0.05, hspace=0.06,
        )
        ax_map = fig.add_subplot(joint[0, 0], projection=ccrs.Robinson())
        ax_lat = fig.add_subplot(joint[0, 1])
        ax_lon = fig.add_subplot(joint[1, 0])
        ax_corner = fig.add_subplot(joint[1, 1])
        ax_corner.axis("off")
        # Keep the GCIRI legend directly underneath map A, as in the original
        # figure.  The fourth joint cell remains intentionally empty.
        cax = ax_map.inset_axes([0.34, -0.125, 0.32, 0.045])
        bottom = outer[1].subgridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.32)
        ax_region = fig.add_subplot(bottom[0, 0])
        ax_distribution = fig.add_subplot(bottom[0, 1])
        ax_fourth = None
    else:
        # The latitude diagnostic retains its exploratory four-panel layout.
        outer = fig.add_gridspec(
            2,
            12,
            height_ratios=[1.45, 1.0],
            left=0.080,
            right=0.970,
            top=0.985,
            bottom=0.080,
            hspace=0.24,
            wspace=0.62,
        )
        ax_map = fig.add_subplot(outer[0, 0:8], projection=ccrs.Robinson())
        cax = ax_map.inset_axes([0.30, -0.115, 0.42, 0.045])
        ax_region = fig.add_subplot(outer[0, 9:12])
        ax_distribution = fig.add_subplot(outer[1, 0:6])
        ax_fourth = fig.add_subplot(outer[1, 6:12])

    _draw_city_map(fig, ax_map, cax, city, mpl)
    if regional_style == "raincloud":
        regional, membership = _draw_region_raincloud(ax_region, city, order, mpl)
    elif regional_style == "interval":
        regional, membership = _draw_region_intervals(ax_region, city, order, mpl)
    elif regional_style == "latitude":
        regional, membership = _draw_region_raincloud(ax_region, city, order, mpl)
    else:
        raise ValueError(f"Unknown regional_style: {regional_style}")
    panel_label(ax_region, "d")

    _draw_global_distribution(ax_distribution, city, mpl)
    panel_label(ax_distribution, "e")

    profile_sources: list[str] = []
    if regional_style != "latitude":
        longitude_profile = _north_south_profile(city, "lon", np.arange(-180, 181, 1), bandwidth=6.0)
        latitude_profile = _north_south_profile(city, "lat", np.arange(-60, 91, 1), bandwidth=4.0)
        _draw_north_south_profile(ax_lon, longitude_profile, "lon", show_legend=True)
        _draw_north_south_profile(ax_lat, latitude_profile, "lat", show_legend=False)
        panel_label(ax_lon, "b")
        panel_label(ax_lat, "c")
        profile_sources.extend(
            [
                formal._write_source(longitude_profile, out, f"{stem}_north_south_longitude").name,
                formal._write_source(latitude_profile, out, f"{stem}_north_south_latitude").name,
            ]
        )

    if regional_style == "latitude":
        bands = _region_latitude_bands(city, order)
        _draw_region_latitude_heatmap(ax_fourth, bands, order, mpl)
        fourth_name = "fig2_candidate_region_latitude_bands"
        fourth_source = formal._write_source(bands, out, fourth_name).name
        panel_label(ax_fourth, "d")

    source_files = [
        formal._write_source(
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
                    "region_group",
                ]
            ],
            out,
            "fig2_candidate_city_gciri",
        ).name,
        formal._write_source(regional, out, f"{stem}_region_summary").name,
        formal._write_source(membership, out, "fig2_candidate_region_icon_membership").name,
    ]
    source_files.extend(profile_sources)
    if regional_style == "latitude":
        source_files.append(fourth_source)
    paths = formal._save_figure(
        fig,
        out,
        stem,
        4 if regional_style == "latitude" else 5,
        source_files,
    )
    plt.close(fig)
    return paths


def run_candidates(output_dir: Path | None = None) -> Path:
    out = _output_dir(output_dir)
    mpl, plt = load_plotting_backend()
    apply_style(mpl)
    city, _ = formal._prepare_data()
    if len(city) != formal.EXPECTED_FULL_ROWS or city["city"].nunique() != formal.EXPECTED_FULL_ROWS:
        raise ValueError("Figure 2 candidates require 5862 unique current city keys")

    _build_candidate(
        out,
        city,
        mpl,
        plt,
        stem="fig2_candidate_a_regional_raincloud",
        regional_style="raincloud",
    )
    _build_candidate(
        out,
        city,
        mpl,
        plt,
        stem="fig2_candidate_b_regional_intervals",
        regional_style="interval",
    )
    _build_candidate(
        out,
        city,
        mpl,
        plt,
        stem="fig2_candidate_c_region_stratified_latitude",
        regional_style="latitude",
    )
    return out


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Render non-destructive Figure 2 redesign candidates.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(run_candidates(args.output_dir))


if __name__ == "__main__":
    run_cli()
