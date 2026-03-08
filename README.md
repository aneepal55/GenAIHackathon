# The Montgomery Guardian

Step 1 implementation (ingestion + cleaning) for the Guardian pipeline.

## What this currently does
- Pulls records from ArcGIS FeatureServer/MapServer endpoints.
- Normalizes timestamps to UTC ISO strings.
- Cleans addresses into a standard upper-case format.
- Standardizes point geometry into `latitude`/`longitude`.
- Writes per-dataset CSV outputs under `data/processed/`.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_ingestion.py --config config/datasets.yaml
python scripts/bootstrap_sample_data.py --output-dir data/processed
python scripts/run_step2.py --input-dir data/processed --output-dir data/processed --h3-resolution 9
python scripts/run_step4.py --input-dir data/processed --output-dir data/processed --h3-resolution 9 --lag-days 30 --window-days 30 --permutations 499
python scripts/run_step5.py --data-dir data/processed --output data/processed/pitch_summary.md
python scripts/run_all_steps.py --use-sample-data --data-dir data/processed
python scripts/prepare_served_data.py --source-dir data/processed --served-dir data/served
streamlit run app/dashboard.py
```

## Notes
- Update endpoint URLs in `config/datasets.yaml` for your target Montgomery datasets.
- ArcGIS pagination is handled with `resultOffset` + `resultRecordCount`.
- Non-point geometry (for example `City Limit`) is retained in a `geometry_json` column.
- Current config is aligned to your available datasets: `911_calls`, `code_violations`, `paving_project`, `fire_and_police_station`, `city_limit`, and `received_311_service_requests`.

## Step 2 outputs
- `data/processed/grid_features.csv`: engineered per-cell features.
- `data/processed/predictions.csv`: predicted calls per H3 cell.
- `data/processed/top_predictors.csv`: model feature importance ranking.
- `data/processed/model_metrics.json`: baseline model metrics (`MAE`, `R2`).
- Step 2 now includes POI features when `point_of_interest.csv` exists:
  - `poi_count`
  - `distance_to_nearest_poi_km`
- Step 2 also uses `most_visited_locations.csv` when available:
  - `most_visited_count`
  - `most_visited_total_visits`
- Dashboard location labels can use landmark datasets when present:
  - `most_visited_locations.csv`, `point_of_interest.csv`, `community_centers.csv`,
    `parks_and_trail.csv`, `education_facility.csv`, `pharmacy_locator.csv`,
    `business_license.csv`, `received_311_service_requests.csv`
- Technical View in dashboard now includes a dataset-usage status table for all configured YAML datasets.

Step 2 currently uses a proxy target (`recent 30-day calls`) when future-labeled outcomes are not yet explicitly available.
When 911 data is non-spatial (aggregated), Step 2 automatically uses spatial 311 requests as the response target.

## Step 3 dashboard
- Launch with `streamlit run app/dashboard.py`.
- Includes:
  - 3D extruded coarser H3 display zones (`H3HexagonLayer`)
  - 30-day timeline slider
  - Safety ROI panel for selected vulnerability pocket
  - Community View chance labels (`Low/Moderate/High chance today`) and non-decimal counts
  - Technical View with raw model diagnostics and dataset usage table

## Step 4 correlation + calibration
- Launch with `python scripts/run_step4.py --input-dir data/processed --output-dir data/processed`.
- Outputs:
  - `data/processed/bivariate_moran_summary.json`
  - `data/processed/bivariate_moran_cells.csv`
  - `data/processed/calibration_by_decile.csv` (created during Step 2 model training)

## Step 5 judging polish
- Launch with `python scripts/run_step5.py --data-dir data/processed --output data/processed/pitch_summary.md`.
- Outputs:
  - `data/processed/pitch_summary.md`
- Dashboard additions:
  - KPI cards (predicted calls, high-risk pockets, Moran's I, model R2)
  - One-click pitch narrative generation + markdown download

## Step 6 demo hardening
- `python scripts/bootstrap_sample_data.py` creates synthetic demo-safe CSVs.
- `python scripts/run_all_steps.py` runs Step 1 -> 2 -> 4 -> 5 end-to-end.
- Use `--use-sample-data` on `run_all_steps.py` to skip API ingestion and run fully offline.

## Deployment (judge-ready link)
- Build artifacts once:
  - `python scripts/run_all_steps.py --config config/datasets.yaml --data-dir data/processed --publish-served --served-dir data/served`
- The dashboard reads from `data/served` first (fallback: `data/processed`).
- Deploy only the Streamlit app + `data/served` snapshot so judges do not need to run training.
