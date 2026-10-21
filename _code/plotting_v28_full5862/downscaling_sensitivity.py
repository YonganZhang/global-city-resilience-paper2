"""Sensitivity of the main results to the downscaling settings (Supplementary Note 3, Supplementary Table 4).

The response parameters alpha_H, alpha_E and alpha_V of the trapezoid-equivalent downscaling (Methods; 0.8, 0.3
and 0.3) and the bounds of the exposure ratio E (0.3 to 3) are settings of this assessment, not GIRI values
(_code/giri/_config.py). Each setting recomputes city physical risk with the pipeline functions
(_code/giri/r_ph_downscale.py) from the hazard, exposure and vulnerability ratios of the paper run, then the
absorptive loss and GCIRI with the unchanged council capacities (_code/giri/trapezoid.py), at the precision the
pipeline files carry. No model is called. The baseline setting must reproduce steps 07-09 of the paper run exactly.

Settings: the baseline, one change at a time (alpha_H 0.6 and 1.0, alpha_E and alpha_V 0.2 and 0.4, exposure
bounds 0.5 to 2, 0.2 to 5 and 0.1 to 10), and the 27 combinations of these parameter values.

Usage (project root): python3 _code/plotting_v28_full5862/downscaling_sensitivity.py
Writes _paper/gciri_full5862_spine_20260704/downscaling_sensitivity_<stamp>/; point
_shared.DOWNSCALING_SENSITIVITY_REL at it.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "_code/giri"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import ALPHA_E, ALPHA_H, ALPHA_V, E_RATIO_BOUNDS  # noqa: E402
from r_ph_downscale import compute_e_ratio, compute_r_ph_city  # noqa: E402
from trapezoid import compute_three_capabilities, compute_trapezoid_giri  # noqa: E402
import development_mode_analysis as D  # noqa: E402
from _shared import EXPECTED_FULL_ROWS, FULL_RUN_REL, PAPER_DIR_REL  # noqa: E402

RUN = ROOT / FULL_RUN_REL
BASELINE = {"alpha_h": ALPHA_H, "alpha_e": ALPHA_E, "alpha_v": ALPHA_V, "e_bounds": E_RATIO_BOUNDS}
ONE_AT_A_TIME = [("alpha_h", 0.6), ("alpha_h", 1.0), ("alpha_e", 0.2), ("alpha_e", 0.4), ("alpha_v", 0.2),
                 ("alpha_v", 0.4), ("e_bounds", (0.5, 2.0)), ("e_bounds", (0.2, 5.0)), ("e_bounds", (0.1, 10.0))]
GRID = {"alpha_h": (0.6, 0.8, 1.0), "alpha_e": (0.2, 0.3, 0.4), "alpha_v": (0.2, 0.3, 0.4)}
LABELS = {"alpha_h": "Hazard parameter", "alpha_e": "Exposure parameter", "alpha_v": "Vulnerability parameter"}
LEVELS = ["low", "transition", "high"]  # GCIRI tiers of Fig. 2: below 40, 40 to 70, 70 or above
INPUT_FILES = ["step_06_f.csv", "step_07_r_ph_city.csv", "step_07_r_ph_city_full_6275.parquet",
               "step_08_capacities.csv", "step_09_gciri.csv"]


def load_inputs() -> pd.DataFrame:
    """One row per city: national risk, H, V and the exposure fields, f and the response and restorative inputs."""
    keys = pd.read_csv(RUN / "step_07_r_ph_city.csv", usecols=["city", "ucdb_id", "r_ph_city"])
    full = pd.read_parquet(RUN / "step_07_r_ph_city_full_6275.parquet")[
        ["ucdb_id", "R_Ph__RAW_nat", "H_ratio", "E_ratio", "V_ratio", "exp_pop_ntl_city__RAW", "exp_pop_ntl_nat_avg__RAW"]]
    f = pd.read_csv(RUN / "step_06_f.csv", usecols=["city", "f_factor"])
    caps = pd.read_csv(RUN / "step_08_capacities.csv", usecols=["city", "cap_abs", "cap_resp", "cap_rest"])
    gciri = pd.read_csv(RUN / "step_09_gciri.csv", usecols=["city", "gciri"])
    inputs = (keys.merge(full, on="ucdb_id", how="left", validate="one_to_one")
              .merge(f, on="city", validate="one_to_one").merge(caps, on="city", validate="one_to_one")
              .merge(gciri, on="city", validate="one_to_one"))
    if len(inputs) != EXPECTED_FULL_ROWS or inputs.drop(columns="exp_pop_ntl_city__RAW").isna().any().any():
        raise ValueError(f"expected {EXPECTED_FULL_ROWS} complete city rows, found {len(inputs)}")
    return inputs


def compute_city_scores(inputs: pd.DataFrame, alpha_h: float, alpha_e: float, alpha_v: float,
                        e_bounds: tuple[float, float]) -> pd.DataFrame:
    """City risk, absorptive loss and GCIRI for one setting, rounded as in step_07, step_08 and step_09."""
    exposure = inputs.apply(compute_e_ratio, axis=1, bounds=e_bounds)
    risk = [round(compute_r_ph_city(rn, h, e, v, alpha_h, alpha_e, alpha_v), 6)
            for rn, h, e, v in zip(inputs.R_Ph__RAW_nat, inputs.H_ratio, exposure, inputs.V_ratio)]
    cap_abs = [round(compute_three_capabilities(1 - f, resp / 100, rest / 100, r)["cap_abs"], 2)
               for f, resp, rest, r in zip(inputs.f_factor, inputs.cap_resp, inputs.cap_rest, risk)]
    gciri = [round(compute_trapezoid_giri(F_city=1 - f, cap_resp=resp, cap_rest=rest, R_Ph=r), 2)
             for f, resp, rest, r in zip(inputs.f_factor, inputs.cap_resp, inputs.cap_rest, risk)]
    return pd.DataFrame({"city": inputs.city, "e_ratio": exposure, "r_ph_city": risk, "cap_abs": cap_abs,
                         "gciri": gciri})


def check_baseline(inputs: pd.DataFrame, scores: pd.DataFrame) -> dict:
    """The baseline setting must reproduce the stored exposure ratios, city risk, absorptive loss and GCIRI."""
    checks = {"e_ratio": (scores.e_ratio - inputs.E_ratio).abs().max(),
              "r_ph_city": (scores.r_ph_city - inputs.r_ph_city).abs().max(),
              "cap_abs": (scores.cap_abs - inputs.cap_abs).abs().max(),
              "gciri": (scores.gciri - inputs.gciri).abs().max()}
    if any(value != 0 for value in checks.values()):
        raise ValueError(f"baseline does not reproduce the paper run: {checks}")
    return {name: float(value) for name, value in checks.items()}


def summarize_setting(base: pd.DataFrame, scores: pd.DataFrame) -> dict:
    """Main results of one setting, compared with the baseline city table (indexed by city)."""
    city = base.copy()
    new = scores.set_index("city").reindex(city.index)
    city["gciri"], city["cap_abs"] = new.gciri, new.cap_abs
    city = D.classify_capacities(city)
    north, south = city.north_south_proxy.eq("North proxy"), city.north_south_proxy.eq("South proxy")
    level = pd.cut(city.gciri, [-1, 40, 70, 101], right=False, labels=LEVELS)
    high_pattern = city.mode_code.isin(["M1", "M2"])
    change = (city.gciri - base.gciri).abs()
    return {
        "spearman_vs_baseline": float(spearmanr(city.gciri, base.gciri).correlation),
        "mean_abs_change": float(change.mean()),
        "median_abs_change": float(change.median()),
        "p95_abs_change": float(change.quantile(0.95)),
        "south_north_ratio_pct": float(city.gciri[south].mean() / city.gciri[north].mean() * 100),
        "restorative_weakest_pct": float(city.weakest_dimension.eq("restorative").mean() * 100),
        **{f"most_common_weakest_{name}": city.weakest_dimension[level == name].value_counts().idxmax()
           for name in LEVELS},
        "low_resilience_pct": float((city.gciri < 40).mean() * 100),
        "high_resilience_pct": float((city.gciri >= 70).mean() * 100),
        "high_patterns_north_pct": float(high_pattern[north].mean() * 100),
        "high_patterns_south_pct": float(high_pattern[south].mean() * 100),
        "patterns_retained_pct": float(city.mode_code.eq(base.mode_code).mean() * 100),
        "mean_city_risk": float(scores.r_ph_city.mean()),
    }


def describe_setting(name: str, value) -> str:
    if name == "e_bounds":
        return f"Exposure bounds {value[0]:g} to {value[1]:g}"
    return f"{LABELS[name]} {value:.1f}"


def build_settings() -> list[tuple[str, str, dict]]:
    """(group, label, parameters) for the baseline, the one-at-a-time changes and the parameter grid."""
    settings = [("baseline", "None (baseline)", dict(BASELINE))]
    settings += [("one_at_a_time", describe_setting(name, value), {**BASELINE, name: value})
                 for name, value in ONE_AT_A_TIME]
    for values in itertools.product(*GRID.values()):
        parameters = {**BASELINE, **dict(zip(GRID, values))}
        settings.append(("grid", ", ".join(describe_setting(k, v) for k, v in zip(GRID, values)), parameters))
    return settings


def format_table(results: pd.DataFrame) -> pd.DataFrame:
    rows = results[results.group.isin(["baseline", "one_at_a_time"])]
    return pd.DataFrame({
        "Changed setting": rows.label,
        "Spearman’s ρ": rows.spearman_vs_baseline.map("{:.3f}".format),
        "Mean change": rows.mean_abs_change.map("{:.1f}".format),
        "South as % of North": rows.south_north_ratio_pct.map("{:.1f}".format),
        "Restorative weakest (%)": rows.restorative_weakest_pct.map("{:.1f}".format),
        "Low resilience (%)": rows.low_resilience_pct.map("{:.1f}".format),
        "High-resilience patterns, North / South (%)": [
            f"{n:.1f} / {s:.1f}" for n, s in zip(rows.high_patterns_north_pct, rows.high_patterns_south_pct)],
        "Patterns retained (%)": rows.patterns_retained_pct.map("{:.1f}".format),
    })


def summarize_grid(results: pd.DataFrame) -> dict:
    grid = results[results.group.eq("grid")]
    numeric = ["spearman_vs_baseline", "mean_abs_change", "south_north_ratio_pct", "restorative_weakest_pct",
               "low_resilience_pct", "high_patterns_north_pct", "high_patterns_south_pct", "patterns_retained_pct"]
    weakest = grid[[f"most_common_weakest_{name}" for name in LEVELS]]
    return {"settings": int(len(grid)),
            "ranges": {col: [float(grid[col].min()), float(grid[col].max())] for col in numeric},
            "restorative_most_common_at_every_level": int(weakest.eq("restorative").all(axis=1).sum())}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    inputs = load_inputs()
    base = D.load_mode_city_data().set_index("city")
    rows, baseline_check = [], None
    for group, label, parameters in build_settings():
        scores = compute_city_scores(inputs, **parameters)
        if group == "baseline":
            baseline_check = check_baseline(inputs, scores)
        rows.append({"group": group, "label": label, **{k: str(v) if k == "e_bounds" else v
                                                         for k, v in parameters.items()},
                     **summarize_setting(base, scores),
                     "exposure_at_lower_bound_pct": float(scores.e_ratio.eq(parameters["e_bounds"][0]).mean() * 100)})
        print(f"{label}: {rows[-1]['south_north_ratio_pct']:.1f} {rows[-1]['low_resilience_pct']:.1f}", flush=True)
    results = pd.DataFrame(rows)
    out = ROOT / PAPER_DIR_REL / f"downscaling_sensitivity_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(exist_ok=False)
    results.to_csv(out / "downscaling_sensitivity_settings.csv", index=False)
    format_table(results).to_csv(out / "supplementary_table_downscaling_sensitivity.csv", index=False)
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).relative_to(ROOT)),
        "run": str(FULL_RUN_REL),
        "inputs_sha256": {name: digest(RUN / name) for name in INPUT_FILES},
        "baseline": {k: list(v) if k == "e_bounds" else v for k, v in BASELINE.items()},
        "baseline_reproduction_max_abs_difference": baseline_check,
        "grid": summarize_grid(results),
        "model_calls": 0,
        "outputs": ["downscaling_sensitivity_settings.csv", "supplementary_table_downscaling_sensitivity.csv"],
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
