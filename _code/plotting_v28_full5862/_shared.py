from __future__ import annotations

import json
import math
import os
import struct
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FULL_ROWS = 5862
FULL_RUN_REL = Path("_pipelines/01_baseline_gciri/outputs/v28_full5862_20260702_020421")
FULL_GCIRI_REL = FULL_RUN_REL / "analysis/full_gciri_with_meta.csv"
COUNTRY_SPREAD_REL = FULL_RUN_REL / "analysis/within_country_spread.csv"
COUNTRY_GIRI_REL = FULL_RUN_REL / "analysis/country_vs_official_giri.csv"
MASTER_UCDB_REL = Path("_outputs/master_v27_6/master_v28_ucdb.parquet")
COASTLINE_SHP_REL = Path("代码/compound_hazard_pipeline/data/ne_10m_coastline.shp")

REQUIRED_FULL_COLUMNS = {
    "city",
    "city_name",
    "iso3",
    "ucdb_id",
    "country_name",
    "lat",
    "lon",
    "pop_2025",
    "cap_abs",
    "cap_resp",
    "cap_rest",
    "r_ph_city",
    "f_city",
    "gciri",
}

FORBIDDEN_PATH_MARKERS = (
    "_legacy",
    "master_50city",
    "v28_softcap_coordfix_50city",
    "global_resilience_master",
    "giri_global_final",
    "giri_global_v17",
    "giri_global_v18",
    "giri_global_v19",
    "7273",
    "7300",
)

FIGURE_MODES = {
    "A": (3.5, 3.5),
    "B": (7.2, 3.5),
    "C": (3.5, 5.5),
    "D": (7.2, 7.2),
    "E": (7.2, 2.5),
}


def build_project_path(rel: Path | str) -> Path:
    rel_path = Path(rel)
    if rel_path.is_absolute():
        raise ValueError(f"Only repo-relative paths are allowed: {rel_path}")
    text = rel_path.as_posix()
    for marker in FORBIDDEN_PATH_MARKERS:
        if marker in text:
            raise ValueError(f"Forbidden legacy path marker in input path: {marker}")
    return PROJECT_ROOT / rel_path


def _validate_full_gciri(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    missing = sorted(REQUIRED_FULL_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    if len(df) != EXPECTED_FULL_ROWS:
        raise ValueError(f"{source} has {len(df)} rows; expected {EXPECTED_FULL_ROWS}")
    if df["city"].nunique() != EXPECTED_FULL_ROWS:
        raise ValueError("Full GCIRI table must have 5862 unique row-level city keys")
    if df["ucdb_id"].isna().any() or df["iso3"].isna().any():
        raise ValueError("Full GCIRI table contains missing ucdb_id or iso3")
    numeric_cols = ["lat", "lon", "pop_2025", "cap_abs", "cap_resp", "cap_rest", "r_ph_city", "f_city", "gciri"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not df["lat"].between(-90, 90).all():
        raise ValueError("Latitude outside [-90, 90]")
    if not df["lon"].between(-180, 180).all():
        raise ValueError("Longitude outside [-180, 180]")
    for col in ["cap_abs", "cap_resp", "cap_rest", "gciri"]:
        if not df[col].between(0, 100).all():
            raise ValueError(f"{col} outside [0, 100]")
    return df


def load_full_gciri() -> pd.DataFrame:
    path = build_project_path(FULL_GCIRI_REL)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return _validate_full_gciri(df, path)


def load_country_spread() -> pd.DataFrame:
    path = build_project_path(COUNTRY_SPREAD_REL)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"iso3", "country_name", "n", "mean", "std", "min", "max", "spread"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def load_country_giri() -> pd.DataFrame:
    path = build_project_path(COUNTRY_GIRI_REL)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"iso3", "gciri_mean", "n_cities", "country_name", "official_giri"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def load_master_ucdb(columns: Iterable[str] | None = None) -> pd.DataFrame:
    path = build_project_path(MASTER_UCDB_REL)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path, columns=list(columns) if columns else None)
    if len(df) != EXPECTED_FULL_ROWS:
        raise ValueError(f"{path} has {len(df)} rows; expected {EXPECTED_FULL_ROWS}")
    return df


def build_output_dir() -> Path:
    explicit = os.environ.get("GCIRI_FIGURE_OUTDIR")
    if explicit:
        out = Path(explicit)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = PROJECT_ROOT / "_figures" / f"gciri_full5862_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def load_plotting_backend():
    try:
        import matplotlib as mpl
        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to render figures. Install it in a project/user "
            "environment, then rerun this plotting package."
        ) from exc
    return mpl, plt


def apply_style(mpl) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    )


def compute_figure_size(mode: str) -> tuple[float, float]:
    if mode not in FIGURE_MODES:
        raise ValueError(f"Unknown figure mode {mode}; choose one of {sorted(FIGURE_MODES)}")
    return FIGURE_MODES[mode]


def panel_label(ax, letter: str) -> None:
    ax.text(
        -0.10,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color="#111111",
        va="bottom",
        ha="left",
    )


def normalize_fonts(fig) -> None:
    for ax in fig.get_axes():
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(7)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(7)
        for text in ax.texts:
            if text.get_text() in list("abcdefghijklmnopqrstuvwxyz"):
                text.set_fontsize(10)
                text.set_fontweight("bold")
            elif text.get_fontsize() < 7:
                text.set_fontsize(7)


def save_figure(fig, out_dir: Path, stem: str, source_rows: int) -> tuple[Path, Path]:
    normalize_fonts(fig)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    meta = {
        "stem": stem,
        "source_table": FULL_GCIRI_REL.as_posix(),
        "source_rows": source_rows,
        "expected_rows": EXPECTED_FULL_ROWS,
    }
    (out_dir / f"{stem}.source.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return png, pdf


def compute_scatter_size(pop: pd.Series) -> np.ndarray:
    values = pd.to_numeric(pop, errors="coerce").fillna(0).clip(lower=0)
    return 2.0 + np.sqrt(values / max(values.max(), 1.0)) * 18.0


def _read_polyline_shapefile(path: Path) -> list[np.ndarray]:
    """Read PolyLine shapefile records without requiring GIS dependencies."""
    segments: list[np.ndarray] = []
    with path.open("rb") as fh:
        header = fh.read(100)
        if len(header) != 100:
            raise ValueError(f"Invalid shapefile header: {path}")
        file_code = struct.unpack(">i", header[:4])[0]
        if file_code != 9994:
            raise ValueError(f"Unsupported shapefile file code {file_code}: {path}")
        while True:
            record_header = fh.read(8)
            if not record_header:
                break
            if len(record_header) != 8:
                raise ValueError(f"Truncated shapefile record header: {path}")
            _record_number, content_words = struct.unpack(">2i", record_header)
            content = fh.read(content_words * 2)
            if len(content) != content_words * 2:
                raise ValueError(f"Truncated shapefile record content: {path}")
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                continue
            if shape_type not in {3, 13, 23}:
                continue
            offset = 4 + 32
            num_parts, num_points = struct.unpack("<2i", content[offset : offset + 8])
            offset += 8
            parts = list(struct.unpack(f"<{num_parts}i", content[offset : offset + 4 * num_parts]))
            offset += 4 * num_parts
            points_raw = np.frombuffer(content[offset : offset + 16 * num_points], dtype="<f8")
            points = points_raw.reshape(num_points, 2)
            stops = parts[1:] + [num_points]
            for start, stop in zip(parts, stops):
                segment = points[start:stop]
                if len(segment) >= 2:
                    segments.append(segment.copy())
    if not segments:
        raise ValueError(f"No polyline segments found in {path}")
    return segments


_COASTLINE_CACHE: list[np.ndarray] | None = None


def load_coastline_segments() -> list[np.ndarray]:
    global _COASTLINE_CACHE
    if _COASTLINE_CACHE is None:
        path = build_project_path(COASTLINE_SHP_REL)
        if not path.exists():
            raise FileNotFoundError(path)
        _COASTLINE_CACHE = _read_polyline_shapefile(path)
    return _COASTLINE_CACHE


def draw_world_basemap(ax) -> None:
    from matplotlib.collections import LineCollection

    segments = load_coastline_segments()
    collection = LineCollection(
        segments,
        colors="#9EA7AE",
        linewidths=0.24,
        alpha=0.85,
        zorder=0,
        rasterized=True,
    )
    ax.add_collection(collection)


_ROBINSON_LAT = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90], dtype=float)
_ROBINSON_X = np.array(
    [1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600, 0.9427, 0.9216, 0.8962, 0.8679, 0.8350, 0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722, 0.5322],
    dtype=float,
)
_ROBINSON_Y = np.array(
    [0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720, 0.4340, 0.4958, 0.5571, 0.6176, 0.6769, 0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761, 1.0000],
    dtype=float,
)


def project_robinson(lon: np.ndarray | pd.Series | float, lat: np.ndarray | pd.Series | float) -> tuple[np.ndarray, np.ndarray]:
    """Project longitude/latitude to unit Robinson coordinates.

    The output is for plotting only. It keeps the source data in geographic
    coordinates and avoids adding a cartographic dependency to the paper figure
    pipeline.
    """
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    lat_abs = np.clip(np.abs(lat_arr), 0, 90)
    x_coeff = np.interp(lat_abs, _ROBINSON_LAT, _ROBINSON_X)
    y_coeff = np.interp(lat_abs, _ROBINSON_LAT, _ROBINSON_Y)
    x = 0.8487 * x_coeff * np.deg2rad(np.clip(lon_arr, -180, 180))
    y = 1.3523 * y_coeff * np.sign(lat_arr)
    return x, y


def _split_dateline_segments(segments: list[np.ndarray]) -> list[np.ndarray]:
    split: list[np.ndarray] = []
    for segment in segments:
        if len(segment) < 2:
            continue
        breaks = np.where(np.abs(np.diff(segment[:, 0])) > 180)[0] + 1
        starts = np.r_[0, breaks]
        stops = np.r_[breaks, len(segment)]
        for start, stop in zip(starts, stops):
            part = segment[start:stop]
            if len(part) >= 2:
                split.append(part)
    return split


def draw_projected_world_basemap(ax) -> None:
    from matplotlib.collections import LineCollection

    projected_segments = []
    for segment in _split_dateline_segments(load_coastline_segments()):
        x, y = project_robinson(segment[:, 0], segment[:, 1])
        projected_segments.append(np.column_stack([x, y]))
    collection = LineCollection(
        projected_segments,
        colors="#9EA7AE",
        linewidths=0.24,
        alpha=0.85,
        zorder=0,
        rasterized=True,
    )
    ax.add_collection(collection)


def apply_projected_world_axis(ax) -> None:
    lon_edge = np.linspace(-180, 180, 361)
    x_top, y_top = project_robinson(lon_edge, np.full_like(lon_edge, 85.0))
    x_bottom, y_bottom = project_robinson(lon_edge, np.full_like(lon_edge, -60.0))
    x_left, y_left = project_robinson(np.full(146, -180.0), np.linspace(-60, 85, 146))
    x_right, y_right = project_robinson(np.full(146, 180.0), np.linspace(-60, 85, 146))
    xs = np.concatenate([x_top, x_bottom, x_left, x_right])
    ys = np.concatenate([y_top, y_bottom, y_left, y_right])
    ax.set_xlim(float(xs.min()) - 0.08, float(xs.max()) + 0.08)
    ax.set_ylim(float(ys.min()) - 0.05, float(ys.max()) + 0.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#F7FAFC")
    for lon in [-120, -60, 0, 60, 120]:
        lat_line = np.linspace(-60, 85, 146)
        x, y = project_robinson(np.full_like(lat_line, lon, dtype=float), lat_line)
        ax.plot(x, y, color="#D8D8D8", linewidth=0.28, alpha=0.65, zorder=0)
    for lat in [-40, 0, 40, 80]:
        lon_line = np.linspace(-180, 180, 361)
        x, y = project_robinson(lon_line, np.full_like(lon_line, lat, dtype=float))
        ax.plot(x, y, color="#D8D8D8", linewidth=0.28, alpha=0.65, zorder=0)


def apply_world_axis(ax) -> None:
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
    ax.set_yticks([-40, 0, 40, 80])
    ax.grid(True, color="#D8D8D8", linewidth=0.35, alpha=0.7)
    ax.set_facecolor("#F7FAFC")


def compute_quantile_bins(series: pd.Series, n: int = 40) -> np.ndarray:
    clean = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if clean.size == 0:
        raise ValueError("Cannot bin an empty series")
    lo, hi = float(np.nanmin(clean)), float(np.nanmax(clean))
    if math.isclose(lo, hi):
        hi = lo + 1.0
    return np.linspace(lo, hi, n)
