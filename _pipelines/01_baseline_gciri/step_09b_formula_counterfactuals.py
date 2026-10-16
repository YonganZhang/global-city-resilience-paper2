"""step_09b_formula_counterfactuals.py — post-run R_Ph dominance diagnostics.

This does not change GCIRI. It creates a small counterfactual audit for paired
or grouped pilot panels so formula-driven anomalies are visible before scaling.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "_code" / "giri"))

from trapezoid import compute_trapezoid_giri  # noqa: E402


def _find_panel_csv(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("pairwise_common_sense_*.csv"))
    return candidates[0] if candidates else None


def _city_label(city_key: str) -> str:
    return city_key.split(" [", 1)[0]


def _calc_gciri(row, r_ph: float) -> float:
    # step_09 stores f_city as resilience; trapezoid expects F deficit.
    return compute_trapezoid_giri(
        F_city=1 - float(row["f_city"]),
        cap_resp=float(row["cap_resp"]),
        cap_rest=float(row["cap_rest"]),
        R_Ph=float(r_ph),
    )


def _pair_rows(df: pd.DataFrame, panel: pd.DataFrame | None) -> list[tuple[str, str, str]]:
    if panel is None or "audit_group" not in panel.columns:
        return []
    available = set(df["city"])
    pairs: list[tuple[str, str, str]] = []
    for group, g in panel.groupby("audit_group", sort=False):
        keys = [c for c in g["city_key"].tolist() if c in available]
        if len(keys) > 4:
            # Large audit groups are usually ordered as deliberate pairs
            # (e.g. San Diego/Tijuana, McAllen/Reynosa). All-pairs is noisy.
            for i in range(0, len(keys) - 1, 2):
                pairs.append((str(group), keys[i], keys[i + 1]))
        else:
            for a, b in itertools.combinations(keys, 2):
                pairs.append((str(group), a, b))
    return pairs


def build_counterfactuals(out_dir: Path, panel_csv: Path | None = None) -> pd.DataFrame:
    step09 = out_dir / "step_09_gciri.csv"
    if not step09.exists():
        raise FileNotFoundError(f"missing {step09}")
    df = pd.read_csv(step09)
    required = {"city", "gciri", "r_ph_city", "f_city", "cap_resp", "cap_rest"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"step_09_gciri.csv missing columns: {sorted(missing)}")

    panel = None
    if panel_csv is None:
        panel_csv = _find_panel_csv(out_dir)
    if panel_csv and panel_csv.exists():
        panel = pd.read_csv(panel_csv)

    pairs = _pair_rows(df, panel)
    if not pairs:
        # Fallback: inspect nearest high-gap rows by current GCIRI rank neighbors.
        top = df.sort_values("gciri", ascending=False)["city"].tolist()
        pairs = [("rank_neighbor", a, b) for a, b in zip(top, top[1:])]

    by_city = df.set_index("city")
    rows = []
    for group, a, b in pairs:
        if a not in by_city.index or b not in by_city.index:
            continue
        ra = by_city.loc[a]
        rb = by_city.loc[b]
        avg_r = (float(ra["r_ph_city"]) + float(rb["r_ph_city"])) / 2
        for city, peer, row, prow in ((a, b, ra, rb), (b, a, rb, ra)):
            current = float(row["gciri"])
            avg_gciri = _calc_gciri(row, avg_r)
            swap_gciri = _calc_gciri(row, float(prow["r_ph_city"]))
            cap08_gciri = _calc_gciri(row, min(float(row["r_ph_city"]), 0.8))
            cap06_gciri = _calc_gciri(row, min(float(row["r_ph_city"]), 0.6))
            max_delta = max(
                abs(avg_gciri - current),
                abs(swap_gciri - current),
                abs(cap08_gciri - current),
                abs(cap06_gciri - current),
            )
            rows.append({
                "audit_group": group,
                "city": city,
                "peer": peer,
                "city_short": _city_label(city),
                "peer_short": _city_label(peer),
                "current_gciri": round(current, 2),
                "current_r_ph": round(float(row["r_ph_city"]), 4),
                "peer_r_ph": round(float(prow["r_ph_city"]), 4),
                "gciri_if_r_ph_pair_avg": round(avg_gciri, 2),
                "gciri_if_r_ph_swapped_with_peer": round(swap_gciri, 2),
                "gciri_if_r_ph_capped_0_8": round(cap08_gciri, 2),
                "gciri_if_r_ph_capped_0_6": round(cap06_gciri, 2),
                "max_counterfactual_delta": round(max_delta, 2),
                "formula_dominance_flag": max_delta >= 10,
            })
    return pd.DataFrame(rows)


def write_report(out_dir: Path, cf: pd.DataFrame) -> None:
    analysis = out_dir / "analysis" / "formula_counterfactuals"
    analysis.mkdir(parents=True, exist_ok=True)
    csv_path = analysis / "formula_counterfactuals.csv"
    md_path = analysis / "formula_counterfactuals.md"
    cf.to_csv(csv_path, index=False)

    flagged = cf[cf["formula_dominance_flag"]].copy()
    lines = [
        "# Formula Counterfactual Diagnostics",
        "",
        f"- Rows inspected: {len(cf)}",
        f"- Formula-dominance flags (`max_delta >= 10`): {len(flagged)}",
        "",
        "This audit changes only `R_Ph` counterfactually; it does not change the official GCIRI output.",
        "",
    ]
    if len(flagged):
        lines.extend([
            "## Flagged Rows",
            "",
            "| group | city | peer | current | r_ph | peer_r_ph | max_delta | avg_r | swapped | cap_0.8 | cap_0.6 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for _, r in flagged.sort_values("max_counterfactual_delta", ascending=False).head(80).iterrows():
            lines.append(
                f"| {r.audit_group} | {r.city_short} | {r.peer_short} | "
                f"{r.current_gciri:.2f} | {r.current_r_ph:.4f} | {r.peer_r_ph:.4f} | "
                f"{r.max_counterfactual_delta:.2f} | {r.gciri_if_r_ph_pair_avg:.2f} | "
                f"{r.gciri_if_r_ph_swapped_with_peer:.2f} | {r.gciri_if_r_ph_capped_0_8:.2f} | "
                f"{r.gciri_if_r_ph_capped_0_6:.2f} |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ formula counterfactuals: {csv_path}")
    print(f"✓ formula counterfactual report: {md_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Run output dir")
    p.add_argument("--panel-csv", default=None, help="Optional pairwise panel CSV")
    args = p.parse_args()
    out_dir = Path(args.out)
    panel_csv = Path(args.panel_csv) if args.panel_csv else None
    cf = build_counterfactuals(out_dir, panel_csv)
    write_report(out_dir, cf)


if __name__ == "__main__":
    main()
