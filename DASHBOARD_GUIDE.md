# Montgomery Guardian Dashboard Guide

This guide explains what each dashboard element means and how to present it.

## 1) Title + Subtitle
- **The Montgomery Guardian**: project identity.
- **Digital twin view...**: this is a live spatial forecast, not a static report.

## 2) Timeline Slider (Next 30 Days)
- Controls which forecast day you are viewing (`Day 0` to `Day 30`).
- When you move it:
  - Map hex heights/colors change.
  - `Forecast Day Demand` changes.
  - `Top Pockets (Today)` reorders.
- Interpretation: how risk pockets shift over time.

## 3) KPI Row
- **Predicted Service Demand (30d)**:
  - Sum of predicted service events across all H3 cells for the next 30 days.
- **High-Risk Pockets**:
  - Count of cells in the top 10% risk for the currently selected day.
- **Bivariate Moran’s I**:
  - Spatial correlation signal between lagged code violations and later response demand.
  - Positive = clustering in same direction; near zero = weak spatial structure.
- **Forecast Day Demand**:
  - Total predicted demand on the selected timeline day.
- **Top predictor** (caption):
  - Most influential feature in the model (`top_predictors.csv`).
- **Target source** (caption):
  - Which dataset was predicted (`911_calls` if spatial, otherwise `received_311_service_requests`).

## 4) 3D H3 Map
- Each hexagon is an H3 grid cell.
- **Height** = predicted demand for the selected day.
- **Color**:
  - Blue = lower relative demand.
  - Orange = higher relative demand.
- Tooltip shows:
  - H3 id
  - Forecast day demand for that cell
  - Code violations
  - 311 requests

## 5) Safety ROI Panel (Right Side)
- **Vulnerability Pocket (H3)** dropdown:
  - Selects one high-risk cell for inspection.
- Card values:
  - `Predicted Demand (30d)`: 30-day model forecast for that cell.
  - `Forecast Demand (Selected Day)`: day-specific value from timeline.
  - `Code Violations`, `311 Requests`, `POIs in Cell`
  - Distance to nearest station and POI.
- ROI text:
  - Heuristic intervention estimate (cleanup/enforcement -> projected demand reduction).
  - Use as decision-support narrative, not a guaranteed causal estimate.

## 6) Strategy Lens
- Pre-written policy framing:
  - Resource optimization
  - Equity lens
  - Proactive governance
- Purpose: quickly communicate value to judges/city staff.

## 7) Top Pockets (Today) Table
- Top 10 cells for the currently selected day.
- Columns:
  - `Forecast_Day`
  - `Forecast_30d`
  - `Code`
  - `Req311`
- Use this to identify where operations should focus first.

## 8) Pitch Brief Generator
- Button creates a markdown summary from current pipeline outputs.
- Download button exports it as `pitch_summary.md`.

## How To Explain In 30 Seconds
1. "This map forecasts **where service demand will concentrate** over the next month."
2. "The slider shows **how hotspots shift day by day**."
3. "When I click a pocket, we get an **actionable ROI estimate** for preventive enforcement."
4. "This lets the city move from reactive dispatch to proactive intervention."

## Important Caveats
- If 911 data is aggregated and non-spatial, model target switches to spatial 311 requests.
- ROI text is heuristic and should be treated as prioritization guidance.
- Correlation (Moran’s I) is not proof of causation by itself.
