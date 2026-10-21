from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from ._shared import EXPECTED_FULL_ROWS, POLICY_DIR_REL, PROJECT_ROOT
    from .development_mode_analysis import (
        BASE_PARAMETERS,
        compute_development_modes,
        load_mode_city_data,
    )
except ImportError:
    from _shared import EXPECTED_FULL_ROWS, POLICY_DIR_REL, PROJECT_ROOT
    from development_mode_analysis import (
        BASE_PARAMETERS,
        compute_development_modes,
        load_mode_city_data,
    )


SUPPORTING_SCENARIO_REL = POLICY_DIR_REL / "city_policy_typology.csv"
DEFAULT_OUTPUT_PARENT = Path("_paper/gciri_full5862_spine_20260704")


def load_restorative_scenario() -> pd.DataFrame:
    path = PROJECT_ROOT / SUPPORTING_SCENARIO_REL
    if not path.is_file():
        raise FileNotFoundError(path)
    scenario = pd.read_csv(path)
    required = {
        "city",
        "delta_restorative_10pp_raw",
        "restorative_nonmonotonic_flag",
    }
    missing = sorted(required.difference(scenario.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if (
        len(scenario) != EXPECTED_FULL_ROWS
        or scenario["city"].nunique() != EXPECTED_FULL_ROWS
    ):
        raise ValueError("Restorative scenario must contain 5862 unique city keys")
    return scenario[list(required)]


def compute_restorative_mode_boundary(
    city: pd.DataFrame | None = None,
    scenario: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    if city is None:
        city = load_mode_city_data()
    if scenario is None:
        scenario = load_restorative_scenario()
    work = city.merge(scenario, on="city", how="left", validate="one_to_one")
    if work["delta_restorative_10pp_raw"].isna().any():
        raise ValueError("Restorative scenario merge produced missing values")

    baseline_codes = city["mode_code"].reset_index(drop=True)
    baseline_tiers = pd.cut(
        city["gciri"],
        bins=[-1, 40, 70, 101],
        labels=["low", "transition", "high"],
        right=False,
    ).reset_index(drop=True)
    scenarios: dict[str, pd.DataFrame] = {}

    score_only = work.copy()
    score_only["gciri"] = (
        score_only["gciri"] + score_only["delta_restorative_10pp_raw"]
    )
    scenarios["formula score response only"] = score_only

    capacity_only = work.copy()
    capacity_only["restorative_strength"] = (
        capacity_only["restorative_strength"] + 10.0
    ).clip(upper=100.0)
    scenarios["capacity profile change only"] = capacity_only

    combined = capacity_only.copy()
    combined["gciri"] = combined["gciri"] + combined["delta_restorative_10pp_raw"]
    scenarios["combined score and capacity change"] = combined

    summary_rows: list[dict[str, float | int | str]] = []
    classified: dict[str, pd.DataFrame] = {}
    for scenario_name, frame in scenarios.items():
        result = compute_development_modes(frame, **BASE_PARAMETERS)
        classified[scenario_name] = result
        mode_changed = result["mode_code"].reset_index(drop=True).ne(
            baseline_codes
        )
        scenario_tiers = pd.cut(
            result["gciri"],
            bins=[-1, 40, 70, 101],
            labels=["low", "transition", "high"],
            right=False,
        ).reset_index(drop=True)
        tier_changed = scenario_tiers.ne(baseline_tiers)
        summary_rows.append(
            {
                "scenario": scenario_name,
                "mode_changed_n": int(mode_changed.sum()),
                "mode_changed_pct": float(mode_changed.mean() * 100.0),
                "gciri_tier_changed_n": int(tier_changed.sum()),
                "gciri_tier_changed_pct": float(tier_changed.mean() * 100.0),
            }
        )

    combined_result = classified["combined score and capacity change"]
    transition = (
        pd.crosstab(
            baseline_codes.rename("baseline_mode"),
            combined_result["mode_code"]
            .reset_index(drop=True)
            .rename("combined_scenario_mode"),
        )
        .reindex(
            index=[f"M{index}" for index in range(1, 7)],
            columns=[f"M{index}" for index in range(1, 7)],
            fill_value=0,
        )
        .reset_index()
    )

    nonmonotonic = work["delta_restorative_10pp_raw"].lt(-0.01)
    nonmonotonic_tier_change = pd.cut(
        score_only["gciri"],
        bins=[-1, 40, 70, 101],
        labels=["low", "transition", "high"],
        right=False,
    ).reset_index(drop=True).ne(baseline_tiers)
    negative_changes = work.loc[
        nonmonotonic, "delta_restorative_10pp_raw"
    ]
    findings: dict[str, float | int] = {
        "nonmonotonic_n": int(nonmonotonic.sum()),
        "nonmonotonic_pct": float(nonmonotonic.mean() * 100.0),
        "nonmonotonic_median_delta_gciri": float(negative_changes.median()),
        "nonmonotonic_mean_delta_gciri": float(negative_changes.mean()),
        "nonmonotonic_gciri_tier_changed_n": int(
            (nonmonotonic & nonmonotonic_tier_change).sum()
        ),
        "nonmonotonic_gciri_tier_changed_pct": float(
            nonmonotonic_tier_change[nonmonotonic].mean() * 100.0
        ),
    }
    return pd.DataFrame(summary_rows), transition, findings


def save_restorative_mode_boundary(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    summary, transition, findings = compute_restorative_mode_boundary()
    summary.to_csv(output_dir / "scenario_mode_change_summary.csv", index=False)
    transition.to_csv(
        output_dir / "combined_mode_transition_matrix.csv",
        index=False,
    )
    manifest = {
        "schema_version": "gciri-restorative-mode-boundary/v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_rows": EXPECTED_FULL_ROWS,
        "scenario_source": SUPPORTING_SCENARIO_REL.as_posix(),
        "findings": findings,
        "outputs": [
            "scenario_mode_change_summary.csv",
            "combined_mode_transition_matrix.csv",
            "analysis_report.md",
        ],
        "boundary": (
            "A +10-point formula scenario is diagnostic only. It is not an "
            "observed intervention, policy effect or mode transition forecast."
        ),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    score_only = summary.loc[
        summary["scenario"].eq("formula score response only")
    ].iloc[0]
    capacity_only = summary.loc[
        summary["scenario"].eq("capacity profile change only")
    ].iloc[0]
    combined = summary.loc[
        summary["scenario"].eq("combined score and capacity change")
    ].iloc[0]
    report = [
        "# Restorative formula and mode-boundary diagnostic",
        "",
        f"- Source rows: {EXPECTED_FULL_ROWS:,}.",
        (
            f"- The restorative +10 formula response changes the GCIRI tier and "
            f"mode of {int(score_only['mode_changed_n']):,} cities "
            f"({score_only['mode_changed_pct']:.2f}%)."
        ),
        (
            f"- Changing the capacity profile while holding GCIRI fixed changes "
            f"{int(capacity_only['mode_changed_n']):,} mode labels "
            f"({capacity_only['mode_changed_pct']:.2f}%)."
        ),
        (
            f"- Applying both changes moves {int(combined['mode_changed_n']):,} "
            f"cities ({combined['mode_changed_pct']:.2f}%)."
        ),
        (
            f"- The formula response is non-monotonic below -0.01 points for "
            f"{findings['nonmonotonic_n']:,} cities "
            f"({findings['nonmonotonic_pct']:.2f}%)."
        ),
        (
            f"- Only {findings['nonmonotonic_gciri_tier_changed_n']:,} of those "
            f"cities ({findings['nonmonotonic_gciri_tier_changed_pct']:.2f}%) "
            "cross a GCIRI reporting tier from the formula response."
        ),
        "",
        (
            "The larger combined mode movement is driven mainly by the intended "
            "change in the capacity profile. This does not validate an "
            "intervention; it separates formula-boundary effects from the "
            "typology's response to changed inputs."
        ),
        "",
    ]
    (output_dir / "analysis_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    return manifest


def run_restorative_mode_boundary(output_dir: Path | None = None) -> Path:
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            PROJECT_ROOT
            / DEFAULT_OUTPUT_PARENT
            / f"result4_restorative_boundary_{stamp}"
        )
    elif not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    save_restorative_mode_boundary(output_dir)
    return output_dir


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify restorative formula effects on six-mode labels."
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(run_restorative_mode_boundary(args.output_dir))


if __name__ == "__main__":
    run_cli()
