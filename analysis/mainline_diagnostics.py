"""Statistics reported in Results and Discussion, from the same prepared city table as the formal figures.

Prints the two-peak distribution, same-band North/South comparison, risk-capacity coupling, North-South
capacity ratios, restorative weakness by group and region, pattern shares by group and income, population
by resilience level and hazard quartile, and the weakest capacity of megacities. Written on 2026-10-17 for
the Results rewrite (promoted from _sandbox/2026-10-17-results-mainline on 2026-10-22); the public release
ships it as analysis/mainline_diagnostics.py.

Usage (repository root): python3 analysis/mainline_diagnostics.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_code/plotting_v28_full5862"))
import development_mode_analysis as D  # noqa: E402
import run_formal_main_figures as R  # noqa: E402
from _shared import HEV_REL  # noqa: E402

S = ["absorptive_strength", "response_strength", "restorative_strength"]
df, _ = R._prepare_data()
g = df["gciri"]

x = np.linspace(0, 100, 1001)
k = gaussian_kde(g)(x)
peaks = [round(x[i], 1) for i in range(1, 1000) if k[i] > k[i - 1] and k[i] > k[i + 1]]
trough = [round(x[i], 1) for i in range(1, 1000) if k[i] < k[i - 1] and k[i] < k[i + 1]]
print("R1 KDE peaks", peaks, "trough", trough)
print((df.assign(below=g < trough[0]).groupby("income_group").below.mean() * 100).round(1).to_string())

for col, width in [("lon", 10), ("lat", 5)]:
    gaps = []
    for _, d in df.groupby((df[col] // width) * width):
        north, south = d[d.north_south_proxy == "North proxy"], d[d.north_south_proxy == "South proxy"]
        if len(north) >= 5 and len(south) >= 5:
            gaps.append(north.gciri.mean() - south.gciri.mean())
    gaps = np.array(gaps)
    print(f"R1 {col} {width}-degree bands with both groups: {len(gaps)}, North higher in {(gaps > 0).sum()}, mean gap {gaps.mean():.1f}")

for c in ["response_strength", "restorative_strength"]:
    print("R1 physical risk vs", c, "Spearman", round(spearmanr(df.r_ph_city, df[c]).correlation, 3))
print(df.groupby(pd.qcut(df.r_ph_city, 4, labels=["Q1", "Q2", "Q3", "Q4"]), observed=True)[["gciri"] + S[1:]].mean().round(1).to_string())

ns = df.groupby("north_south_proxy")[["gciri"] + S + ["r_ph_city"]].mean()
print("R2 South as % of North\n", (ns.loc["South proxy"] / ns.loc["North proxy"] * 100).round(1).to_string())
north = np.sort(df.loc[df.north_south_proxy == "North proxy", "gciri"].to_numpy())
south = df.loc[df.north_south_proxy == "South proxy", "gciri"].to_numpy()
print("R2 P(North city > South city)", round((len(north) - np.searchsorted(north, south, side="right")).sum() / (len(north) * len(south)), 3))

lowest = df[S].idxmin(axis=1)
print("R3 lowest capacity by group (%)\n", (pd.crosstab(df.north_south_proxy, lowest, normalize="index") * 100).round(1).to_string())
print("R3 lowest capacity by region (%)\n", (pd.crosstab(df.region_short, lowest, normalize="index") * 100).round(1).to_string())
print("R3 regional mean strengths\n", df.groupby("region_short")[S].mean().round(1).to_string())

c = D.load_mode_city_data()
print("R4 pattern shares by group (%)\n", (pd.crosstab(c.north_south_proxy, c.mode_code, normalize="index") * 100).round(1).to_string())
print("R4 pattern shares by income (%)\n", (pd.crosstab(c.income_group, c.mode_code, normalize="index") * 100).round(1).to_string())

# ---- Round 2 (author asked to give Result 1 more scientific value): population, hazard by development level ----
hev = pd.read_csv(ROOT / HEV_REL)
df = df.merge(hev[["city", "h_ratio", "e_ratio"]], on="city", validate="one_to_one")
total = df.pop_2025.sum()
for lo, hi in [(0, 40), (40, 70), (70, 101)]:
    d = df[(df.gciri >= lo) & (df.gciri < hi)]
    print(f"R1 GCIRI {lo}-{hi}: {len(d) / len(df) * 100:.1f}% of cities, {d.pop_2025.sum() / 1e9:.2f} bn people ({d.pop_2025.sum() / total * 100:.1f}%)")
print(f"R1 assessed population {total / 1e9:.2f} bn; cities in 50-65 band {((g >= 50) & (g < 65)).mean() * 100:.1f}%")
for name, d in df.groupby("income_group"):
    q = pd.qcut(d.h_ratio.rank(method="first"), 4, labels=False)
    print(f"R1 {name}: GCIRI sd {d.gciri.std():.1f}, low-hazard quartile {d.gciri[q == 0].mean():.1f}, "
          f"high-hazard quartile {d.gciri[q == 3].mean():.1f}, Spearman(GCIRI, local hazard) {spearmanr(d.gciri, d.h_ratio).correlation:.2f}")
low_income = df[df.income_group == "Low income"]
print(f"R1 low-income cities >=1M: mean {low_income[low_income.pop_2025 >= 1e6].gciri.mean():.1f} (n={(low_income.pop_2025 >= 1e6).sum()}), "
      f"<0.3M: mean {low_income[low_income.pop_2025 < 3e5].gciri.mean():.1f}")
for name, d in df.groupby("north_south_proxy"):
    print(f"R2 {name}: {d.pop_2025.sum() / 1e9:.2f} bn people, {d[d.gciri < 40].pop_2025.sum() / 1e9:.3f} bn in low-resilience cities "
          f"({d[d.gciri < 40].pop_2025.sum() / d.pop_2025.sum() * 100:.1f}%)")
print("R2 North minus South (points)\n", (ns.loc["North proxy"] - ns.loc["South proxy"]).round(1).to_string())

# ---- Round 3 (panel-order rewrite): capacity low-value spread and country dominance (Fig. 5b rule) ----
print("R3 share of cities with strength < 40 (%)\n", (df.groupby("region_short")[S].apply(lambda d: (d < 40).mean() * 100)).round(1).to_string())
print("R3 global share < 40 (%)", ((df[S] < 40).mean() * 100).round(1).to_dict())
print("R4 pattern shares by region (%)\n", (pd.crosstab(c.region_group, c.mode_code, normalize="index") * 100).round(1).to_string())
portfolio = D.compute_country_mode_portfolios(c)
eligible = portfolio[portfolio.n_cities >= 3].copy()
shares = eligible[[f"share_m{i}" for i in range(1, 7)]]
lead = eligible.dominant_mode_share_pct - shares.apply(lambda r: sorted(r, reverse=True)[1], axis=1)
dominant = eligible[lead >= 10]
print(f"R4 countries with >=3 cities: {len(eligible)}, with a dominant pattern: {len(dominant)}, "
      f"their share of cities {dominant.n_cities.sum() / eligible.n_cities.sum() * 100:.1f}%")

# ---- Discussion v2: weakest capacity of megacities (>= 10 million residents) ----
mega = c[c.pop_2025 >= 1e7]
mega_low = mega[S].idxmin(axis=1)
print(f"D2 megacities: {len(mega)}, weakest capacity shares {(mega_low.value_counts(normalize=True) * 100).round(1).to_dict()}")
print("D2 examples", mega.assign(low=mega_low).loc[lambda d: d.city_name.isin(["Mumbai", "Jakarta", "Manila", "Lagos", "Dhaka"]), ["city_name", "country_name", "low"]].to_string(index=False))
