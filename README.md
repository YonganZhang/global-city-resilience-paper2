# Global City Infrastructure Resilience Index (GCIRI) for 5,862 cities

This repository contains the code and derived data behind the manuscript
*Global urban resilience divides, capacity weaknesses and development priorities*
(Zhang, Jiang, Guo, Chen and Yan; submitted). The study assesses urban
infrastructure resilience for 5,862 cities with at least 100,000 residents in
161 countries. A multi-agent council of eight large language models rates 18
city indicators, and the Global City Infrastructure Resilience Index (GCIRI)
extends the GIRI framework to the city scale by downscaling national physical
risk and combining it with absorptive, response and restorative capacities.

The model code that introduced the council and the downscaling for a 50-city
benchmark is released separately at
<https://github.com/YonganZhang/global-city-resilience-paper1-scs>.

## Contents

| Path | Contents |
|---|---|
| `_pipelines/01_baseline_gciri/` | Assessment pipeline: metadata (step 1), council rounds R1, R2 and Consul (steps 3-5), absorptive factor (6), physical-risk downscaling (7), capacities (8) and GCIRI (9) |
| `_pipelines/.../outputs/v28_full5862_20260702_020421/` | Results of the full run: council indicator ratings (`step_05_consul.csv`), intermediate tables and the city results (`analysis/full_gciri_with_meta.csv`) |
| `_code/llm/` | Prompt builders, indicator definitions, rating scale and the gateway client used by the council |
| `_code/giri/` | Physical-risk downscaling and the GIRI performance geometry |
| `_code/plotting_v28_full5862/` | Analyses and figure code for the main, Extended Data and Supplementary figures |
| `_paper/.../` | Derived analysis tables used by the figures (weakness patterns, sensitivity, validation) |
| `_outputs/master_v27_6/master_v28_ucdb.parquet` | City input table: urban-centre identifiers, hazard, exposure and national reference values |
| `_data/` | Country metadata, the indicator prompt and council roster workbook, and GIRI calibration coefficients |
| `analysis/mainline_diagnostics.py` | Reproduces the statistics reported in Results and Discussion |
| `source_data/` | Source data for every main and Extended Data figure (Excel, one workbook per figure) |

## Reproducing the results

```bash
pip install -r requirements.txt

# GCIRI from the released council ratings (exactly reproduces analysis/full_gciri_with_meta.csv)
cd _pipelines/01_baseline_gciri
mkdir -p outputs/rerun
cp outputs/v28_full5862_20260702_020421/step_01_meta.yml outputs/v28_full5862_20260702_020421/step_05_consul.csv outputs/rerun/
for step in step_06_compute_f step_07_compute_r_ph_city step_08_compute_capacities step_09_compute_gciri; do
  python3 $step.py --out outputs/rerun
done
cd ../..

# Statistics reported in the manuscript
python3 analysis/mainline_diagnostics.py

# Figures (written to _figures/)
cd _code/plotting_v28_full5862
python3 fig2_redesign_candidates.py              # Fig. 2
python3 storyline_figures.py                     # Figs. 3-5
python3 extended_data_hazard_development.py      # Extended Data Fig. 1
python3 supplementary_figures_redesign.py        # Extended Data Figs. 2-7 and Supplementary Fig. 1
```

Re-running the council (steps 1 and 3-5) requires an OpenAI-compatible gateway
that serves the eight registered models. Set `LLM_BASE_URL`, `LLM_API_KEY` (or
`LLM_API_KEY_POOL`, comma-separated) and `GCIRI_ALLOW_PAID_LLM=1`. No endpoint
or credential is stored in this repository. Model outputs vary over time, so a
new run will not reproduce the released ratings exactly.

Web retrieval of city sources (Stage 1 in Fig. 1) is optional and off by
default. Set `LLM_INCLUDE_TAVILY=1` and `TAVILY_API_KEY` (or `TAVILY_API_KEYS`,
comma-separated) to add encyclopaedia, government and international-organization
and academic summaries to the first-round prompts. Cities without cached sources
are searched through the Tavily API and cached under `_data/tavily_cache_clean/`.
The released ratings were produced without retrieval.

## Data

- `analysis/full_gciri_with_meta.csv`: one row per city with UCDB identifier,
  country, coordinates, 2025 population, absorptive loss (`cap_abs`), response
  deficit (`cap_resp`), restorative capacity (`cap_rest`), city physical risk
  (`r_ph_city`), absorptive factor (`f_city`) and GCIRI. Strengths used in the
  paper are `100 - cap_abs`, `100 - cap_resp` and `cap_rest`.
- `step_05_consul.csv`: final rating tier and score for each city and indicator,
  with retained and excluded council models and evidence weights. The free-text
  reasoning of the models is not released.
- `step_07_r_ph_city.csv`: local hazard (`h_ratio`), exposure (`e_ratio`) and
  vulnerability (`v_ratio`) ratios and the downscaled physical risk.

Not included: raw hazard rasters, population and night-time-light grids, urban
centre polygons, model logs and API records. These are available from their
providers: GHSL Urban Centre Database R2024A, GIRI (UNEP-GRID and CDRI),
WorldPop and NASA Black Marble VIIRS.

## Licences

Code is released under the MIT License (`LICENSE`). Derived data are released
under CC BY 4.0 (`DATA_LICENSE.md`); upstream datasets remain under their own
terms. Please cite the article (see `CITATION.cff`) when using this material.
