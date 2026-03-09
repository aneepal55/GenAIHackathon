# The Montgomery Guardian

The Montgomery Guardian is a proactive city-operations dashboard that predicts where service demand is likely to rise over the next 30 days.

This README is a **dashboard guide**: what each section means, how to read it, and how to use it during a demo.

## What This Dashboard Does
- Forecasts expected neighborhood service demand by H3 map zone.
- Converts model output into plain-language operational guidance.
- Lets users simulate interventions and compare scenario outcomes.
- Builds budget-constrained action portfolios with ROI context.
- Surfaces early warning and equity-focused watchlists.
- Generates a decision-ready pitch narrative for judges/city leaders.

## Run The Project

### End-to-end pipeline (data + model + served snapshot)
```bash
python scripts/run_all_steps.py --config config/datasets.yaml --data-dir data/processed --publish-served --served-dir data/served
```

### Next.js dashboard (primary UI)
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000`

### Streamlit dashboard (backup UI)
```bash
streamlit run app/dashboard.py
```

## Global Dashboard Elements

### Header
- **The Montgomery Guardian**: app title.
- **Subtitle**: plain-language project purpose.
- **Data last updated**: date from `data/served/metadata.json`.

### Left sidebar tabs
- `Overview`
- `Innovation`
- `Budget`
- `Operations`
- `Alerts`
- `Pitch`

### Top controls (tab-specific)
- **Choose an area to inspect**: appears where area-specific detail is shown.
- **Timeline slider**: appears where day-by-day demand changes are visualized.
- **Choose budget scenario**: appears in budget planning view.

---

## Overview Tab

The Overview tab is the city command view.

### KPI cards
- **Expected Requests This Month**
  - Projected total service requests across displayed zones over next 30 days.
- **Areas Needing Attention Today**
  - Number of zones currently in higher-risk range for selected day.
- **Expected Requests On Selected Day**
  - Citywide expected requests for the day selected by timeline slider.
- **Date**
  - Active forecast date.

### Timeline slider
- Moves from day 0 to day 30.
- Changes zone risk for map and day-dependent metrics.
- Use this to show how pressure shifts across time.

### 3D risk map
- **Hexagons** represent coarser display zones (aggregated H3 cells).
- **Color**: lower need (blue) to higher need (orange).
- **Height**: higher hex height means higher forecasted pressure.
- Hover tooltip explains selected zone in plain terms.

### Area Snapshot panel
For the selected area:
- **Expected requests this month**: projected requests in next 30 days.
- **Forecast range this month**: low/high estimate band.
- **Chance of requests today**: Low/Moderate/High chance label.
- **What that means**: plain-language interpretation.
- **Uncertainty score (30d)**: spread between low/high forecast.
- **Code issues reported**: open code complaints signal.
- **311 requests (history)**: historical service request pressure.
- **Places/POIs in this area**: nearby activity indicator.
- **Distance to nearest POI**: nearest known place proximity.
- **Distance to station**: shown only when valid station data exists.

### ROI message box
- Converts risk signals into an estimated intervention impact statement.
- Explains potential request reduction if nuisance/code issues are addressed.

---

## Innovation Tab

The Innovation tab is scenario simulation.

### Intervention sliders
- **Code cleanup effort**
- **Rapid response coverage**
- **Hotspot patrol intensity**
- **Special events pressure**

These controls modify the scenario multiplier and update scenario metrics.

### KPI cards
- **Scenario Expected Requests (30d)**
- **Estimated Requests Prevented (30d)**
- **Scenario Pressure vs Baseline**

### Scenario line chart
- **Blue line**: Baseline (current operations).
- **Orange line**: Scenario (with selected actions).
- **X-axis**: Date (next 30 days).
- **Y-axis**: Expected service requests per day.

### Download button
- **Download Area Action Plan (.md)**
- Exports a concise operational plan for selected area.

---

## Budget Tab

The Budget tab converts strategy into constrained planning.

### Budget selector
- Choose a scenario budget (for example: $120k, $300k, $600k).

### KPI cards
- **Planned Actions**: number of selected interventions.
- **Cells Covered**: number of unique zones touched.
- **Prevented Requests (30d)**: total expected prevention.
- **Equity Spend Share**: percentage of spend in underserved zones.

### Budget note
- Shows budget used, utilization %, and average ROI.

### Action mix chart
- Bar chart of estimated prevented requests by intervention type.
- Helps explain which action class contributes most impact.

### Portfolio table
- Location-level intervention recommendations with:
  - action,
  - cost,
  - expected prevented requests,
  - ROI,
  - underserved flag.

---

## Operations Tab

The Operations tab is execution-focused.

### What This Means Today
- Three plain-language prompts to guide immediate action.

### Top Areas Today
- Ranked area cards with:
  - recommended action,
  - chance category,
  - expected requests,
  - code complaints,
  - 311 pressure,
  - prioritization reason.

### Intervention Leaderboard
- Ranks areas by preventable demand under current scenario.
- Includes baseline vs scenario comparison and blind-spot flag.

### AI Review Queue (High Uncertainty Zones)
- Prioritizes areas where forecast spread is high.
- Supports field validation and data refresh workflow.

---

## Alerts Tab

The Alerts tab provides monitoring views.

### Early Warning Radar
- Bar chart: number of zones by alert level (`watch`, `elevated`, `critical`).
- Alert cards include:
  - reason for flag,
  - expected demand,
  - high-end estimate,
  - stress score.

### Equity Watchlist
- Focuses on high-stress zones with equity considerations.
- Shows risk, stress score, and primary concern.
- Station distance is hidden when station-location data is unavailable.

---

## Pitch Tab

The Pitch tab is communication output.

### Pitch Brief Generator
- **Generate Pitch Narrative**: renders structured brief sections.
- **Download Brief (.md)**: exports `pitch_summary.md`.
- Includes a readable sectioned view plus optional raw markdown view.

Use this tab for final judging narration and executive summaries.

---

## Data Files Used By Dashboard

The Next.js app reads from `data/served` (fallback: `data/processed`):
- `predictions.csv`
- `model_metrics.json`
- `model_governance.json`
- `top_predictors.csv`
- `bivariate_moran_summary.json`
- `operations_summary.json`
- `intervention_portfolio.csv`
- `early_warning_alerts.csv`
- `equity_watchlist.csv`
- `pitch_summary.md`
- context label datasets (for friendly location naming)

## Known Behavior Notes
- If station dataset is missing/unavailable, station distance is treated as unavailable instead of fake zero.
- Alert chart now reflects full alert output and is not hard-capped to 120 rows.
- Location labels are generated from nearest known landmarks when available.

## Hackathon Demo Flow (Suggested)
1. Start in `Overview`: map + timeline + area snapshot.
2. Move to `Innovation`: simulate interventions with sliders.
3. Open `Budget`: show constrained action planning + ROI.
4. Show `Operations` and `Alerts`: prioritization + monitoring.
5. End on `Pitch`: generate/download narrative.
