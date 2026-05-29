# Bay Wheels OD Demand Model

Poisson origin–destination demand model for Bay Wheels (Lyft) bike-share,
fitted on ~4.7 million trips across the San Francisco Bay Area.

## Model

Hourly OD counts N_ijt follow a Poisson distribution:

```
log μ_ijt = α_i + β_j + η_t + γ_dist·d_ij + γ_elev·Δh_ij + γ_wx·w_t
```

where α_i, β_j are station fixed effects, η_t captures hour-of-day,
day-of-week, month, and holiday structure, and the γ covariates add
distance decay, elevation gain, and area-wide weather effects.

All models are fitted via L-BFGS-B with analytic gradients exploiting
the **M·S factorisation**: the expected-count sum over I²·T cells reduces
to O(I²) + O(T) per gradient step, making large station networks tractable.

## Results summary

| Model | Test Poisson Deviance | Train NLL |
|-------|-----------------------|-----------|
| Null (distance + FEs only) | 27,879,897 | 11,081,357 |
| Full (+ elevation + weather) | 27,721,769 | 11,065,707 |

Key fitted coefficients (full model):

| Covariate | γ | Interpretation |
|-----------|---|----------------|
| Distance | −0.592 | Demand halves every ~1.2 km |
| Holiday | −0.233 | 21% fewer trips on holidays |
| Temperature | +0.012 | +1.2% per °C |
| Precipitation | −0.521 | **40% fewer trips per mm/hr rain** |
| Wind speed | −0.004 | Negligible |
| Elevation gain | −0.00001 | Negligible at Bay Wheels scale |

## Repository layout

```
src/baywheels/
  data/
    loader.py        Load and normalise raw trip CSVs (handles two legacy formats)
    aggregator.py    Aggregate to hourly OD counts; haversine distances; elevation matrix
    calendar.py      Build training-period calendar with temporal features + weather
  model/
    params.py        Parameter vector layout (α, β, η, γ)
    poisson.py       Poisson log-likelihood and analytic gradient (M·S factorisation)
    fit.py           L-BFGS-B wrapper; FitResult dataclass
  eval/
    metrics.py       MAE, RMSE, Pearson R, Poisson deviance
    importance.py    Permutation feature importance
    diagnostics.py   Convergence curves, marginal-balance check, residual plots

scripts/
  prepare_data.py    Aggregate raw CSVs → processed parquets, distance/elevation matrices
  train_compare.py   Train null and full models; save .pkl bundles
  train_baseline.py  Single-model training entry point
  evaluate.py        Per-model accuracy metrics and coefficient summary
  report.py          All report figures (7 plots + metrics table)
  map_viz.py         Static + interactive maps (contextily + folium)

tests/
  test_model.py      Gradient check vs finite differences; convergence invariant
  test_data.py       Loader schema; haversine accuracy; aggregator counts

figures/             Pre-generated plots and maps
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Aggregate raw trip CSVs into processed parquets
python scripts/prepare_data.py \
    --data-dir ~/baywheels-tripdata-augmented \
    --out-dir  data/processed

# 2. Train null, full, and ZIP models
python scripts/train_compare.py \
    --data-dir data/processed \
    --out-dir  models \
    --ridge    1e-3 \
    --maxiter  10000

# 3. Generate all report figures
python scripts/report.py \
    --model-dir models \
    --data-dir  data/processed \
    --out-dir   figures

# 4. Generate maps
python scripts/map_viz.py \
    --model-dir models \
    --data-dir  data/processed \
    --out-dir   figures
```

## Data pipeline

Raw Bay Wheels trip CSVs are augmented with:
- **Weather**: ERA5-Land hourly reanalysis (temperature, precipitation, wind, humidity)
  via a local SQLite weather cache, averaged across valid land cells.
- **Elevation**: USGS EPQS API, queried per station (named) or per 111m GPS grid
  cell (dockless trips). Elevation gain matrix Δh_ij = h_j − h_i.

These augmentation scripts live in the parent directory (`augment_weather.py`,
`augment_elevation.py`) and are not part of this package.

## Design notes

**M·S factorisation**: The Poisson partition function factorises as
M·S where M = Σ_ij exp(α_i + log K_ij + β_j) and S = Σ_t exp(η_t).
This holds when weather enters η_t (temporal-only), maintaining O(I²)+O(T)
cost per gradient step instead of O(I²T).

**Ridge on γ only**: The L₂ penalty applies to covariate coefficients
(γ_dist, γ_holiday, γ_elev, γ_wx, δ_dist) but not to station or temporal
fixed effects, which are identified by the data.

