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
        PROJECT_ROOT,
        apply_style,
        load_full_gciri,
        load_plotting_backend,
    )
except ImportError:
    from _shared import (
        EXPECTED_FULL_ROWS,
        FULL_GCIRI_REL,
        PROJECT_ROOT,
        apply_style,
        load_full_gciri,
        load_plotting_backend,
    )


PAPER_DIR_REL = Path("_paper/gciri_full5862_spine_20260704")
SUPPLEMENTARY_SOURCE_REL = Path(
    "_figures/gciri_full5862_supplementary_20260718_211003/source_data"
)
MODE_SUMMARY_REL = (
    PAPER_DIR_REL
    / "result4_mode_validation_20260718_194933"
    / "mode_summary.csv"
)
MODE_CHANGE_REL = (
    PAPER_DIR_REL
    / "result4_restorative_boundary_20260718_201645"
    / "scenario_mode_change_summary.csv"
)
COVARIATE_REL = SUPPLEMENTARY_SOURCE_REL / "figs10_covariate_correlations.csv"
FEMA_REL = SUPPLEMENTARY_SOURCE_REL / "figs10_fema_correlations.csv"

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


def make_table_frames() -> dict[str, tuple[pd.DataFrame, list[Path]]]:
    truth = load_full_gciri()
    if len(truth) != EXPECTED_FULL_ROWS or truth["city"].nunique() != EXPECTED_FULL_ROWS:
        raise ValueError("Current truth must contain 5862 unique city keys")

    covariates = load_csv(
        COVARIATE_REL,
        {"short", "n_nonmissing", "spearman_gciri", "independence_class"},
    )
    fema = load_csv(
        FEMA_REL,
        {"short", "n", "spearman"},
    )
    covariate_labels = {
        "Population": "City population",
        "Night lights": "VIIRS night-time lights",
        "Pop. x night lights": "Population-nightlight exposure",
        "Official GIRI": "Official national GIRI",
    }
    evidence_labels = {
        "descriptive city covariate": "Descriptive city covariate",
        "input-adjacent urban activity proxy": "Input-adjacent activity proxy",
        "input-adjacent exposure proxy": "Input-adjacent exposure proxy",
        "national-lineage variable": "Inherited national anchor",
    }
    s1_covariates = pd.DataFrame(
        {
            "Check": covariates["short"].map(covariate_labels),
            "n": covariates["n_nonmissing"].astype(int),
            "Spearman rho": covariates["spearman_gciri"],
            "Evidence class": covariates["independence_class"].map(evidence_labels),
        }
    )
    s1_fema = pd.DataFrame(
        {
            "Check": fema["short"],
            "n": fema["n"].astype(int),
            "Spearman rho": fema["spearman"],
            "Evidence class": "US county modeled benchmark",
        }
    )
    s1 = pd.concat([s1_covariates, s1_fema], ignore_index=True)
    desired_order = [
        "Official national GIRI",
        "City population",
        "VIIRS night-time lights",
        "Population-nightlight exposure",
        "FEMA resilience",
        "FEMA social vulnerability",
        "FEMA expected annual loss",
        "FEMA composite risk",
    ]
    s1["order"] = s1["Check"].map({value: index for index, value in enumerate(desired_order)})
    s1 = s1.sort_values("order").drop(columns="order").reset_index(drop=True)
    s1["Spearman rho"] = s1["Spearman rho"].map(lambda value: f"{value:.4f}")

    s2 = pd.DataFrame(
        [
            [
                "Technical completeness",
                "5862/5862 final scores; no missing GCIRI",
                "Complete for current analyses",
                "Not construct validity",
            ],
            [
                "Council quality gate",
                "R1-Consul r=0.9568; MAE=5.74",
                "Stable for bounded screening",
                "Country-prior and identity risk remain",
            ],
            [
                "50-city continuity",
                "50/50 matched; r=0.7718; MAE=11.30",
                "Updated v28 run",
                "Not an exact numerical extension",
            ],
            [
                "Common-sense audit",
                "No catastrophic global failure found",
                "Conservative comparative use",
                "Avoid precise league tables",
            ],
            [
                "Reverse attribution",
                "Extremes follow risk and capacity terms",
                "Component-explainable",
                "Floor clipping limits order",
            ],
            [
                "Formula boundary",
                "Restorative response is non-monotonic for 1397 cities",
                "Keep scenarios supplementary",
                "No optimization claim",
            ],
        ],
        columns=["Audit", "Key result", "Use", "Boundary"],
    )

    mode = load_csv(
        MODE_SUMMARY_REL,
        {"mode_code", "n", "share_pct", "mean_gciri", "mean_capacity_spread"},
    )
    s3 = mode[
        ["mode_code", "n", "share_pct", "mean_gciri", "mean_capacity_spread"]
    ].copy()
    s3.columns = ["Mode", "n", "Share (%)", "Mean GCIRI", "Mean spread"]
    for column in ["Share (%)", "Mean GCIRI", "Mean spread"]:
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
        "formula score response only": "Formula score response only",
        "capacity profile change only": "Capacity profile change only",
        "combined score and capacity change": "Combined score and capacity change",
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
        "Scenario",
        "Modes changed",
        "Modes changed (%)",
        "Tiers changed",
        "Tiers changed (%)",
    ]
    for column in ["Modes changed (%)", "Tiers changed (%)"]:
        s4[column] = s4[column].map(lambda value: f"{value:.2f}")

    return {
        "table_s1_validation_boundary": (
            s1,
            [FULL_GCIRI_REL, COVARIATE_REL, FEMA_REL],
        ),
        "table_s2_quality_audits": (
            s2,
            [FULL_GCIRI_REL, PAPER_DIR_REL / "scan_memo_20260705.md"],
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
    if stem == "table_s2_quality_audits":
        row_height, font_size = 0.64, 6.8
        col_widths = [0.20, 0.31, 0.23, 0.26]
        wrap_by_column = {
            "Audit": 22,
            "Key result": 31,
            "Use": 22,
            "Boundary": 25,
        }
    elif stem == "table_s1_validation_boundary":
        row_height, font_size = 0.40, 7.2
        col_widths = [0.35, 0.10, 0.16, 0.39]
        wrap_by_column = {"Check": 31, "Evidence class": 31}
    elif stem == "table_s3_development_modes":
        row_height, font_size = 0.38, 7.5
        col_widths = [0.19, 0.16, 0.22, 0.22, 0.21]
        wrap_by_column = {}
    else:
        row_height, font_size = 0.48, 7.2
        col_widths = [0.38, 0.16, 0.17, 0.14, 0.15]
        wrap_by_column = {"Scenario": 31}

    display = frame.copy()
    for column, width in wrap_by_column.items():
        if display[column].dtype == object:
            display[column] = display[column].map(
                lambda value: fill(str(value), width=width)
            )
    header_labels = {
        "Spearman rho": "Spearman\nrho",
        "Share (%)": "Share\n(%)",
        "Mean GCIRI": "Mean\nGCIRI",
        "Mean spread": "Mean\nspread",
        "Modes changed": "Modes\nchanged",
        "Modes changed (%)": "Modes changed\n(%)",
        "Tiers changed": "Tiers\nchanged",
        "Tiers changed (%)": "Tiers changed\n(%)",
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
        if column not in {"Check", "Evidence class", "Audit", "Key result", "Use", "Boundary", "Mode", "Scenario"}
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
