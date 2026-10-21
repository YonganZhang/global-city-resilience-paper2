from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from textwrap import fill

import pandas as pd

try:
    from ._shared import (
        EXPECTED_FULL_ROWS,
        FULL_GCIRI_REL,
        FULL_RUN_REL,
        MASTER_UCDB_REL,
        MODE_DIR_REL,
        PAPER_DIR_REL,
        POLICY_DIR_REL,
        PROJECT_ROOT,
        RESTORATIVE_DIR_REL,
        apply_style,
        load_full_gciri,
        load_master_ucdb,
        load_plotting_backend,
    )
except ImportError:
    from _shared import (
        EXPECTED_FULL_ROWS,
        FULL_GCIRI_REL,
        FULL_RUN_REL,
        MASTER_UCDB_REL,
        MODE_DIR_REL,
        PAPER_DIR_REL,
        POLICY_DIR_REL,
        PROJECT_ROOT,
        RESTORATIVE_DIR_REL,
        apply_style,
        load_full_gciri,
        load_master_ucdb,
        load_plotting_backend,
    )


MODE_SUMMARY_REL = MODE_DIR_REL / "mode_summary.csv"
MODE_CHANGE_REL = RESTORATIVE_DIR_REL / "scenario_mode_change_summary.csv"
POLICY_CITY_REL = POLICY_DIR_REL / "city_policy_typology.csv"
# County FEMA National Risk Index scores matched to the 184 US cities on 2026-07-05.
FEMA_JOIN_REL = (
    PAPER_DIR_REL
    / "_history/markdown_era_2026-07/external_validation_us_fema_20260705_034300/us_fema_nri_city_join.csv"
)
REFERENCE_VARIABLES = [  # (label, master column or FEMA score, relation to GCIRI)
    ("Official national GIRI", "GIRI__RAW_nat", "National input to GCIRI"),
    ("City population", "pop_2025", "City characteristic"),
    ("VIIRS night-time lights", "exp_ntl_city__RAW", "Component of the exposure input"),
    ("Population \u00d7 night-time lights", "exp_pop_ntl_city__RAW", "Exposure input"),
    ("FEMA NRI community resilience", "RESL_SCORE", "US county index"),
    ("FEMA NRI social vulnerability", "SOVI_SCORE", "US county index"),
    ("FEMA NRI expected annual loss", "EAL_SCORE", "US county index"),
    ("FEMA NRI risk index", "RISK_SCORE", "US county index"),
]

INK = "#252A2E"
MUTED = "#697278"
BLUE = "#2E6F9E"
HEADER = "#E8F0F5"
ZEBRA = "#F7F9FA"
GRID = "#C9D1D6"


def load_csv(relative_path: Path, required: set[str]) -> pd.DataFrame:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def build_output_dir(output_dir: Path | None) -> Path:
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            PROJECT_ROOT
            / "_figures"
            / f"gciri_full5862_supplementary_tables_{stamp}"
        )
    elif not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "source_data").mkdir()
    return output_dir


def reference_correlation_table(truth: pd.DataFrame) -> pd.DataFrame:
    """Extended Data Table 2: Spearman correlation of city GCIRI with reference variables."""
    master = load_master_ucdb(columns=["ucdb_id", "pop_2025", "exp_ntl_city__RAW", "exp_pop_ntl_city__RAW", "GIRI__RAW_nat"])
    cities = truth[["city", "ucdb_id", "gciri"]].merge(master, on="ucdb_id", how="left", validate="one_to_one")
    fema = load_csv(FEMA_JOIN_REL, {"city", "RESL_SCORE", "SOVI_SCORE", "EAL_SCORE", "RISK_SCORE"})
    cities = cities.merge(fema.drop(columns=[c for c in fema.columns if c not in
                                             {"city", "RESL_SCORE", "SOVI_SCORE", "EAL_SCORE", "RISK_SCORE"}]),
                          on="city", how="left", validate="one_to_one")
    rows = []
    for label, column, relation in REFERENCE_VARIABLES:
        pair = cities[["gciri", column]].dropna()
        rows.append({"Reference variable": label, "n": f"{len(pair):,}",
                     "Spearman\u2019s \u03c1": f"{pair['gciri'].corr(pair[column], method='spearman'):.2f}",
                     "Relation to GCIRI": relation})
    return pd.DataFrame(rows)


def rating_and_score_table(truth: pd.DataFrame) -> pd.DataFrame:
    """Extended Data Table 3: rating aggregation and properties of the GCIRI score."""
    import sys
    sys.path.append(str(PROJECT_ROOT / "_code" / "llm"))
    from tier_to_score import TIER_TO_MIDPOINT

    run = PROJECT_ROOT / FULL_RUN_REL
    first = pd.read_csv(run / "step_03_r1.csv", usecols=["indicator", "city", "tier"])
    council = pd.read_csv(run / "step_05_consul.csv", usecols=["indicator", "city", "tier"])
    mean8 = first.assign(score=first["tier"].map(TIER_TO_MIDPOINT)).groupby(["indicator", "city"])["score"].mean()
    pair = council.assign(council=council["tier"].map(TIER_TO_MIDPOINT)).set_index(["indicator", "city"])[["council"]]
    pair = pair.join(mean8.rename("mean8"), how="inner").dropna()
    scenario = load_csv(POLICY_CITY_REL, {"city", "delta_restorative_10pp_raw"})
    falls = scenario["delta_restorative_10pp_raw"].lt(-0.01)
    at_floor = truth["gciri"].eq(0)
    return pd.DataFrame([
        {"Analysis": "Unweighted mean of the eight first-round ratings compared with council ratings",
         "n": f"{len(pair):,} city indicators",
         "Result": f"Pearson\u2019s r = {pair['mean8'].corr(pair['council']):.3f}, mean absolute difference "
                   f"{(pair['mean8'] - pair['council']).abs().mean():.2f} points"},
        {"Analysis": "Cities at the lower bound of the GCIRI scale", "n": f"{len(truth):,} cities",
         "Result": f"{int(at_floor.sum())} cities with GCIRI of 0"},
        {"Analysis": "Ten-point increase in restorative strength", "n": f"{len(scenario):,} cities",
         "Result": f"GCIRI decreases by more than 0.01 points in {int(falls.sum()):,} cities ({falls.mean() * 100:.1f}%)"},
    ])


def make_table_frames() -> dict[str, tuple[pd.DataFrame, list[Path]]]:
    truth = load_full_gciri()
    if len(truth) != EXPECTED_FULL_ROWS or truth["city"].nunique() != EXPECTED_FULL_ROWS:
        raise ValueError("Current truth must contain 5862 unique city keys")

    s1 = reference_correlation_table(truth)
    s2 = rating_and_score_table(truth)

    mode = load_csv(
        MODE_SUMMARY_REL,
        {"mode_code", "n", "share_pct", "mean_gciri", "mean_capacity_spread"},
    )
    s3 = mode[
        ["mode_code", "n", "share_pct", "mean_gciri", "mean_capacity_spread"]
    ].copy()
    s3.columns = ["Mode", "n", "Share (%)", "Mean GCIRI", "Mean capacity spread"]
    for column in ["Share (%)", "Mean GCIRI", "Mean capacity spread"]:
        s3[column] = s3[column].map(lambda value: f"{value:.2f}")

    mode_change = load_csv(
        MODE_CHANGE_REL,
        {
            "scenario",
            "mode_changed_n",
            "mode_changed_pct",
            "gciri_tier_changed_n",
            "gciri_tier_changed_pct",
        },
    )
    scenario_labels = {
        "formula score response only": "GCIRI change only",
        "capacity profile change only": "Capacity profile change only",
        "combined score and capacity change": "Both changes",
    }
    s4 = mode_change.copy()
    s4["scenario"] = s4["scenario"].map(scenario_labels)
    s4 = s4[
        [
            "scenario",
            "mode_changed_n",
            "mode_changed_pct",
            "gciri_tier_changed_n",
            "gciri_tier_changed_pct",
        ]
    ]
    s4.columns = [
        "Change",
        "Modes changed",
        "Modes changed (%)",
        "Resilience levels changed",
        "Resilience levels changed (%)",
    ]
    for column in ["Modes changed (%)", "Resilience levels changed (%)"]:
        s4[column] = s4[column].map(lambda value: f"{value:.2f}")

    return {
        "table_s1_validation_boundary": (
            s1,
            [FULL_GCIRI_REL, MASTER_UCDB_REL, FEMA_JOIN_REL],
        ),
        "table_s2_rating_and_score_properties": (
            s2,
            [FULL_GCIRI_REL, FULL_RUN_REL / "step_03_r1.csv", FULL_RUN_REL / "step_05_consul.csv", POLICY_CITY_REL],
        ),
        "table_s3_development_modes": (
            s3,
            [FULL_GCIRI_REL, MODE_SUMMARY_REL],
        ),
        "table_s4_restorative_boundary": (
            s4,
            [FULL_GCIRI_REL, MODE_CHANGE_REL],
        ),
    }


def render_table(
    frame: pd.DataFrame,
    output_dir: Path,
    stem: str,
    sources: list[Path],
    plt,
) -> None:
    if stem == "table_s2_rating_and_score_properties":
        row_height, font_size = 0.64, 6.8
        col_widths = [0.40, 0.20, 0.40]
        wrap_by_column = {"Analysis": 40, "Result": 40}
    elif stem == "table_s1_validation_boundary":
        row_height, font_size = 0.40, 7.2
        col_widths = [0.35, 0.10, 0.16, 0.39]
        wrap_by_column = {"Reference variable": 31, "Relation to GCIRI": 31}
    elif stem == "table_s3_development_modes":
        row_height, font_size = 0.38, 7.5
        col_widths = [0.19, 0.16, 0.22, 0.22, 0.21]
        wrap_by_column = {}
    else:
        row_height, font_size = 0.48, 7.2
        col_widths = [0.38, 0.16, 0.17, 0.14, 0.15]
        wrap_by_column = {"Change": 31}

    display = frame.copy()
    for column, width in wrap_by_column.items():
        if display[column].dtype == object:
            display[column] = display[column].map(
                lambda value: fill(str(value), width=width)
            )
    header_labels = {
        "Spearman\u2019s \u03c1": "Spearman\u2019s\n\u03c1",
        "Share (%)": "Share\n(%)",
        "Mean GCIRI": "Mean\nGCIRI",
        "Mean capacity spread": "Mean capacity\nspread",
        "Modes changed": "Modes\nchanged",
        "Modes changed (%)": "Modes changed\n(%)",
        "Resilience levels changed": "Resilience levels\nchanged",
        "Resilience levels changed (%)": "Resilience levels\nchanged (%)",
    }
    columns = [header_labels.get(column, column) for column in display.columns]
    figure_height = 0.55 + row_height * (len(display) + 1)
    fig, ax = plt.subplots(figsize=(7.2, figure_height))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=columns,
        colLoc="left",
        cellLoc="left",
        colWidths=col_widths,
        bbox=[0.005, 0.02, 0.99, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    numeric_columns = {
        index
        for index, column in enumerate(display.columns)
        if column not in {"Reference variable", "Relation to GCIRI", "Analysis", "Result", "Mode", "Change"}
    }
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.45)
        cell.get_text().set_color(INK)
        cell.get_text().set_va("center")
        cell.get_text().set_ha("right" if column in numeric_columns else "left")
        if row == 0:
            cell.set_facecolor(HEADER)
            cell.get_text().set_weight("bold")
            cell.get_text().set_color(BLUE)
        elif row % 2 == 0:
            cell.set_facecolor(ZEBRA)
        else:
            cell.set_facecolor("white")

    csv_path = output_dir / "source_data" / f"{stem}.csv"
    frame.to_csv(csv_path, index=False)
    for suffix, options in {
        "png": {"dpi": 300},
        "pdf": {},
        "svg": {},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.03,
            **options,
        )
    metadata = {
        "schema_version": "gciri-supplementary-table/v1",
        "stem": stem,
        "truth_source": FULL_GCIRI_REL.as_posix(),
        "source_rows": EXPECTED_FULL_ROWS,
        "unique_city_keys": EXPECTED_FULL_ROWS,
        "table_rows": len(frame),
        "input_sources": [path.as_posix() for path in sources],
        "source_data": f"source_data/{stem}.csv",
        "formats": ["png", "pdf", "svg", "tiff"],
        "status": "supplementary submission candidate",
    }
    (output_dir / f"{stem}.source.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def run_supplementary_tables(output_dir: Path | None = None) -> Path:
    output_dir = build_output_dir(output_dir)
    mpl, plt = load_plotting_backend()
    apply_style(mpl)
    frames = make_table_frames()
    for stem, (frame, sources) in frames.items():
        render_table(frame, output_dir, stem, sources, plt)
    report = {
        "schema_version": "gciri-supplementary-table-package/v1",
        "truth_source": FULL_GCIRI_REL.as_posix(),
        "source_rows": EXPECTED_FULL_ROWS,
        "tables": list(frames),
    }
    (output_dir / "table_package_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Render current full5862 Supplementary tables"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = run_supplementary_tables(args.output_dir)
    print(output.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    run_cli()
