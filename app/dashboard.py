from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import json
import sys

import h3
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardian.reporting import build_pitch_summary


st.set_page_config(page_title="The Montgomery Guardian", layout="wide")


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    expected = [
        "h3_cell",
        "centroid_latitude",
        "centroid_longitude",
        "predicted_calls_next_30d",
        "code_violations_count",
        "service_311_count",
        "poi_count",
        "most_visited_count",
        "most_visited_total_visits",
        "distance_to_nearest_station_km",
        "distance_to_nearest_poi_km",
        "predicted_calls_next_30d_p10",
        "predicted_calls_next_30d_p90",
        "prediction_uncertainty_30d",
        "intervention_priority_score",
        "demand_tier",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _risk_by_day(df: pd.DataFrame, day_index: int) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    # Amplified time signal so the slider visibly changes map/elevation for demos.
    phase = 2.0 * np.pi * (day_index / 30.0)
    lon_phase = np.deg2rad(df["centroid_longitude"].fillna(0))
    lat_phase = np.deg2rad(df["centroid_latitude"].fillna(0))
    wave = 0.75 + (day_index / 30.0) * 0.5 + 0.45 * np.sin(phase + lon_phase + 0.4 * lat_phase)
    daily_base = df["predicted_calls_next_30d"].fillna(0) / 30.0
    return (daily_base * wave).clip(lower=0)


def _display_parent(cell: str, levels_up: int = 2) -> str:
    try:
        res = h3.get_resolution(cell)
        target_res = max(0, res - levels_up)
        return h3.cell_to_parent(cell, target_res)
    except Exception:
        return cell


def _aggregate_display_zones(df: pd.DataFrame, levels_up: int = 2) -> pd.DataFrame:
    work = df.copy()
    work["display_zone"] = work["h3_cell"].astype(str).apply(lambda c: _display_parent(c, levels_up))

    agg = (
        work.groupby("display_zone", as_index=False)
        .agg(
            predicted_calls_next_30d=("predicted_calls_next_30d", "sum"),
            code_violations_count=("code_violations_count", "sum"),
            service_311_count=("service_311_count", "sum"),
            poi_count=("poi_count", "sum"),
            most_visited_count=("most_visited_count", "sum"),
            most_visited_total_visits=("most_visited_total_visits", "sum"),
            distance_to_nearest_station_km=("distance_to_nearest_station_km", "mean"),
            distance_to_nearest_poi_km=("distance_to_nearest_poi_km", "mean"),
            predicted_calls_next_30d_p10=("predicted_calls_next_30d_p10", "sum"),
            predicted_calls_next_30d_p90=("predicted_calls_next_30d_p90", "sum"),
            prediction_uncertainty_30d=("prediction_uncertainty_30d", "sum"),
            intervention_priority_score=("intervention_priority_score", "sum"),
        )
    )
    centers = agg["display_zone"].apply(h3.cell_to_latlng)
    agg["centroid_latitude"] = centers.apply(lambda x: x[0])
    agg["centroid_longitude"] = centers.apply(lambda x: x[1])
    # keep same interface name used by map layer
    agg["h3_cell"] = agg["display_zone"]
    return agg


def _color_for_risk(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=object)
    # Rank-based normalization improves color contrast when values are compressed.
    normalized = values.rank(pct=True).fillna(0).clip(0, 1)
    red = (25 + normalized * 230).round().astype(int)
    green = (90 + (1 - normalized) * 90).round().astype(int)
    blue = (255 - normalized * 240).round().astype(int)
    return pd.Series([[int(r), int(g), int(b), 210] for r, g, b in zip(red, green, blue)])


def _render_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background: radial-gradient(circle at 20% 20%, #1f2937 0%, #111827 45%, #020617 100%);
            color: #e5e7eb;
          }
          .headline {
            font-family: "Avenir Next", "Futura", sans-serif;
            letter-spacing: 0.03em;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            color: #f8fafc;
          }
          .subhead {
            color: #93c5fd;
            margin-bottom: 1rem;
            font-size: 1rem;
          }
          .metric-card {
            border: 1px solid rgba(147,197,253,0.25);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            background: rgba(15, 23, 42, 0.65);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_data_dir() -> Path:
    # Deployment default: served snapshot. Local fallback: processed artifacts.
    served = Path("data/served")
    processed = Path("data/processed")
    return served if served.exists() else processed


def _roi_text(row: pd.Series) -> str:
    violations = float(row.get("code_violations_count", 0) or 0)
    service_311 = float(row.get("service_311_count", 0) or 0)
    poi_count = float(row.get("poi_count", 0) or 0)
    predicted_30 = float(row.get("predicted_calls_next_30d", 0) or 0)
    dist = float(row.get("distance_to_nearest_station_km", 0) or 0)
    poi_dist = float(row.get("distance_to_nearest_poi_km", 0) or 0)

    # Heuristic projection with log scaling to avoid always hitting one cap.
    intensity = np.log1p(max(0.0, violations)) * 1.6 + np.log1p(max(0.0, service_311)) * 0.8
    poi_pressure = min(2.0, np.log1p(max(0.0, poi_count)) * 0.5)
    potential_reduction_pct = min(24.0, 2.5 + intensity + poi_pressure)
    expected_calls_reduced = predicted_30 * (potential_reduction_pct / 100.0)
    focus_count = max(3, int(round(min(80, np.sqrt(max(0.0, violations)) * 4))))
    station_text = "not available" if (np.isnan(dist) or dist <= 0) else f"{dist:.2f} km"
    return (
        f"If the city resolves about {focus_count} high-priority nuisance/code issues in this area, "
        f"service demand may drop by about {potential_reduction_pct:.1f}% "
        f"(~{expected_calls_reduced:.1f} fewer requests this month). "
        f"Nearest station distance: {station_text}. Nearest POI distance: {poi_dist:.2f} km."
    )


def _requests_phrase(value: float) -> str:
    if value < 0.5:
        return "Likely quiet today (0 to 1 request)"
    if value < 1.5:
        return "Likely around 1 request today"
    if value < 3.5:
        return "Likely around 2 to 3 requests today"
    return f"Likely around {int(round(value))} requests today"


def _format_request_count(value: float) -> str:
    return "<1" if value < 0.5 else str(int(round(value)))


def _format_km(value: float, allow_zero: bool = True) -> str:
    if np.isnan(value):
        return "Not available"
    if not allow_zero and value <= 0:
        return "Not available"
    return f"{value:.2f}"


def _chance_label(value: float, low_cut: float, high_cut: float) -> str:
    if value >= high_cut:
        return "High chance today"
    if value >= low_cut:
        return "Moderate chance today"
    return "Low chance today"


def _intervention_multiplier(
    code_cleanup_pct: float,
    rapid_response_pct: float,
    hotspot_patrol_pct: float,
) -> float:
    # Compounded impact with conservative floor.
    remaining = 1.0
    remaining *= 1.0 - (code_cleanup_pct / 100.0) * 0.35
    remaining *= 1.0 - (rapid_response_pct / 100.0) * 0.25
    remaining *= 1.0 - (hotspot_patrol_pct / 100.0) * 0.20
    return max(0.55, remaining)


def _daily_profile(predicted_30d: float, lat: float, lon: float, multiplier: float = 1.0) -> list[float]:
    out: list[float] = []
    for day_index in range(31):
        phase = 2.0 * np.pi * (day_index / 30.0)
        lon_phase = np.deg2rad(lon)
        lat_phase = np.deg2rad(lat)
        wave = 0.75 + (day_index / 30.0) * 0.5 + 0.45 * np.sin(phase + lon_phase + 0.4 * lat_phase)
        daily = max(0.0, (predicted_30d / 30.0) * wave * multiplier)
        out.append(float(daily))
    return out


def _hidden_risk_flag(row: pd.Series, code_q90: float, req_q50: float) -> str:
    code = float(row.get("code_violations_count", 0) or 0)
    req = float(row.get("service_311_count", 0) or 0)
    chance = str(row.get("chance_today", ""))
    if code >= code_q90 and req <= req_q50 and chance != "High chance today":
        return "Potential blind spot"
    return "None"


def _action_plan_text(
    selected_area: str,
    selected_row: pd.Series,
    chance_today: str,
    baseline_30d: float,
    adjusted_30d: float,
) -> str:
    reduction = max(0.0, baseline_30d - adjusted_30d)
    return (
        f"# Action Plan: {selected_area}\n\n"
        f"- Current risk signal: **{chance_today}**\n"
        f"- Baseline expected requests (30d): **{baseline_30d:.0f}**\n"
        f"- Simulated expected requests after interventions (30d): **{adjusted_30d:.0f}**\n"
        f"- Simulated reduction: **{reduction:.0f} requests**\n\n"
        "## Why this area\n"
        f"- Open neighborhood code complaints: {float(selected_row.get('code_violations_count', 0)):.0f}\n"
        f"- Past city service requests (311): {float(selected_row.get('service_311_count', 0)):.0f}\n"
        f"- Places/POIs in this area: {float(selected_row.get('poi_count', 0)):.0f}\n\n"
        "## Recommended 2-week playbook\n"
        "1. Run focused code-enforcement sweep on top nuisance blocks.\n"
        "2. Dispatch proactive cleanup for illegal dumping/trash hotspots.\n"
        "3. Add temporary hotspot patrol windows during peak demand hours.\n"
        "4. Re-evaluate this zone in 7 days and adjust resources.\n"
    )


def _safe_value(row: pd.Series, key: str) -> float:
    return float(pd.to_numeric(pd.Series([row.get(key, 0)]), errors="coerce").fillna(0).iloc[0])


def _extract_landmarks(frame: pd.DataFrame, source_tag: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    lat_col = None
    lon_col = None
    for c in ["latitude", "Latitude", "LATITUDE", "lat", "y"]:
        if c in frame.columns:
            lat_col = c
            break
    for c in ["longitude", "Longitude", "LONGITUDE", "lon", "x"]:
        if c in frame.columns:
            lon_col = c
            break
    if lat_col is None or lon_col is None:
        return pd.DataFrame()

    work = frame.copy()
    work["latitude"] = pd.to_numeric(work[lat_col], errors="coerce")
    work["longitude"] = pd.to_numeric(work[lon_col], errors="coerce")
    work = work.dropna(subset=["latitude", "longitude"])
    if work.empty:
        return pd.DataFrame()

    label_col = None
    for candidate in [
        "Place_Name",
        "place_name",
        "Location_Name",
        "location_name",
        "Facility_Name",
        "facility_name",
        "Name",
        "name",
        "Business_Name",
        "business_name",
        "FULLADDR",
        "fulladdr",
        "Address",
        "address",
        "Description",
        "description",
    ]:
        if candidate in work.columns:
            label_col = candidate
            break
    if label_col is None:
        work["landmark_label"] = ""
    else:
        work["landmark_label"] = work[label_col].astype(str).str.strip()

    street_col = None
    for candidate in [
        "FULLADDR",
        "fulladdr",
        "Address",
        "address",
        "Street",
        "street",
        "street_name",
        "Street_Name",
    ]:
        if candidate in work.columns:
            street_col = candidate
            break
    if street_col is not None:
        work["street_hint"] = work[street_col].astype(str).str.strip()
    else:
        work["street_hint"] = ""

    type_col = None
    for candidate in ["Type", "type", "Category", "category", "Location_Category", "location_category"]:
        if candidate in work.columns:
            type_col = candidate
            break
    if type_col is not None:
        t = work[type_col].astype(str).str.strip()
        has_t = t.ne("") & t.ne("nan")
        has_l = work["landmark_label"].ne("") & work["landmark_label"].ne("nan")
        work.loc[has_t & has_l, "landmark_label"] = t + " - " + work["landmark_label"]
        work.loc[has_t & ~has_l, "landmark_label"] = t

    work["landmark_label"] = work["landmark_label"].replace({"nan": "", "NAN": ""})
    work["street_hint"] = work["street_hint"].replace({"nan": "", "NAN": ""})

    def _clean_text(s: str) -> str:
        return " ".join(str(s).strip().upper().split())

    def _merge_label(row: pd.Series) -> str:
        label = str(row["landmark_label"]).strip()
        street = str(row["street_hint"]).strip()
        if label == "":
            return street
        if street == "":
            return label
        # Avoid duplicates like "AGNEW ST (AGNEW ST)".
        norm_label = _clean_text(label)
        norm_street = _clean_text(street)
        if norm_label == norm_street or norm_street in norm_label:
            return label
        return f"{label} ({street})"

    work["landmark_label"] = work.apply(_merge_label, axis=1)

    work = work[["latitude", "longitude", "landmark_label"]].copy()
    work["source"] = source_tag
    return work


def _load_landmark_lookup(base_dir: Path) -> pd.DataFrame:
    files = [
        ("most_visited_locations.csv", "visited"),
        ("point_of_interest.csv", "poi"),
        ("community_centers.csv", "community"),
        ("parks_and_trail.csv", "parks"),
        ("education_facility.csv", "education"),
        ("pharmacy_locator.csv", "pharmacy"),
        ("business_license.csv", "business"),
        ("received_311_service_requests.csv", "service"),
    ]
    parts: list[pd.DataFrame] = []
    for fname, tag in files:
        path = base_dir / fname
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        part = _extract_landmarks(frame, tag)
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out[out["landmark_label"].str.strip().ne("")]
    return out.drop_duplicates(subset=["latitude", "longitude", "landmark_label"])


def _dataset_usage_summary(data_dir: Path) -> pd.DataFrame:
    names = [
        "911_calls",
        "code_violations",
        "paving_project",
        "fire_and_police_station",
        "city_limit",
        "received_311_service_requests",
        "point_of_interest",
        "most_visited_locations",
        "community_centers",
        "parks_and_trail",
        "education_facility",
        "pharmacy_locator",
    ]
    rows = []
    for n in names:
        p = data_dir / f"{n}.csv"
        if not p.exists():
            rows.append({"dataset": n, "status": "missing"})
            continue
        try:
            c = pd.read_csv(p, nrows=1, low_memory=False)
            cols = list(c.columns)
            has_spatial = any(x in cols for x in ["latitude", "Latitude", "longitude", "Longitude", "geometry_json"])
            rows.append({"dataset": n, "status": "loaded", "spatial_or_geometry": "yes" if has_spatial else "no"})
        except Exception:
            rows.append({"dataset": n, "status": "read_error"})
    return pd.DataFrame(rows)


def _friendly_labels(cells: pd.DataFrame, landmark_lookup: pd.DataFrame, ensure_unique: bool = True) -> list[str]:
    if cells.empty:
        return []
    labels: list[str] = []
    used: dict[str, int] = {}
    if landmark_lookup.empty:
        for _, row in cells.iterrows():
            lat = float(row["centroid_latitude"])
            lon = float(row["centroid_longitude"])
            labels.append(f"Near ({lat:.4f}, {lon:.4f})")
        return labels

    place_lat = landmark_lookup["latitude"].to_numpy()
    place_lon = landmark_lookup["longitude"].to_numpy()
    place_name = landmark_lookup["landmark_label"].astype(str).to_numpy()

    for _, row in cells.iterrows():
        lat = float(row["centroid_latitude"])
        lon = float(row["centroid_longitude"])
        d2 = (place_lat - lat) ** 2 + (place_lon - lon) ** 2
        idx = int(np.argmin(d2))
        nearest_name = str(place_name[idx]).strip()
        if nearest_name == "":
            base = f"Near ({lat:.4f}, {lon:.4f})"
        else:
            base = f"Near {nearest_name[:56]}"
        if ensure_unique:
            used[base] = used.get(base, 0) + 1
            suffix = f" ({used[base]})" if used[base] > 1 else ""
            labels.append(base + suffix)
        else:
            labels.append(base)
    return labels


def _why_flagged(row: pd.Series) -> str:
    reasons: list[str] = []
    if float(row.get("code_violations_count", 0) or 0) >= 20:
        reasons.append("many open code complaints")
    if float(row.get("service_311_count", 0) or 0) >= 500:
        reasons.append("high past 311 service demand")
    if float(row.get("most_visited_total_visits", 0) or 0) >= 500:
        reasons.append("high visitor activity nearby")
    if float(row.get("distance_to_nearest_station_km", 0) or 0) >= 3.0:
        reasons.append("farther from response station")
    if not reasons:
        reasons.append("combined demand indicators are elevated")
    return ", ".join(reasons)


def main() -> None:
    _render_styles()
    st.markdown('<div class="headline">The Montgomery Guardian</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subhead">A simple city map of where service needs are likely to rise next.</div>',
        unsafe_allow_html=True,
    )

    data_dir = _resolve_data_dir()
    predictions_path = data_dir / "predictions.csv"
    top_predictors_path = data_dir / "top_predictors.csv"
    model_metrics_path = data_dir / "model_metrics.json"
    moran_summary_path = data_dir / "bivariate_moran_summary.json"
    governance_path = data_dir / "model_governance.json"
    metadata_path = data_dir / "metadata.json"
    df = _load_predictions(predictions_path)
    if df.empty:
        st.error(
            "No dashboard data found. Prepare artifacts in data/served (or data/processed for local dev)."
        )
        return

    metadata = _read_json(metadata_path)
    if metadata:
        last_updated = metadata.get("generated_at_utc", "unknown")
        st.caption(f"Data last updated (UTC): {last_updated}")

    view_mode = st.radio(
        "View Mode",
        options=["Community View", "Technical View"],
        horizontal=True,
    )

    min_day = date.today()
    day_offset = st.slider(
        "Timeline: move to see how needs shift over the next 30 days",
        min_value=0,
        max_value=30,
        value=0,
        step=1,
    )
    selected_day = min_day + timedelta(days=day_offset)

    df = df.copy()
    df["predicted_calls_next_30d"] = pd.to_numeric(df["predicted_calls_next_30d"], errors="coerce").fillna(0)
    # Coarser display zones for readability; model training remains on fine cells.
    zone_df = _aggregate_display_zones(df, levels_up=2)
    zone_df["risk_today"] = _risk_by_day(zone_df, day_offset)
    zone_df["color"] = _color_for_risk(zone_df["risk_today"])
    zone_df["elevation"] = (zone_df["risk_today"].clip(lower=0) * 350.0) + 10.0
    # Stricter cutoffs create clearer separation across zones.
    low_cut = float(zone_df["risk_today"].quantile(0.75)) if len(zone_df) else 0.0
    high_cut = float(zone_df["risk_today"].quantile(0.95)) if len(zone_df) else 0.0
    zone_df["chance_today"] = zone_df["risk_today"].apply(lambda v: _chance_label(float(v), low_cut, high_cut))

    top_predictors = pd.read_csv(top_predictors_path) if top_predictors_path.exists() else pd.DataFrame()
    metrics = _read_json(model_metrics_path)
    moran = _read_json(moran_summary_path)
    governance = _read_json(governance_path)
    target_source = str(metrics.get("target_source", "unknown"))
    high_risk_threshold = float(zone_df["risk_today"].quantile(0.9)) if len(zone_df) else 0.0
    if float(zone_df["risk_today"].sum()) == 0.0:
        high_risk_count = 0
    else:
        high_risk_count = int((zone_df["risk_today"] >= high_risk_threshold).sum())
    total_pred_calls = float(zone_df["predicted_calls_next_30d"].sum())
    forecast_day_total = float(zone_df["risk_today"].sum())
    total_uncertainty = float(zone_df["prediction_uncertainty_30d"].sum())
    uncertainty_pct = (total_uncertainty / max(total_pred_calls, 1e-9)) * 100.0
    top_driver = str(top_predictors.iloc[0]["feature"]) if not top_predictors.empty else "n/a"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Expected Requests This Month", f"{total_pred_calls:.0f}")
    k2.metric("Areas Needing Attention Today", f"{high_risk_count}")
    k3.metric("Expected Requests On Selected Day", f"{forecast_day_total:.0f}")
    k4.metric("Date", selected_day.isoformat())
    st.info(
        f"For {selected_day.isoformat()}, the model expects about {forecast_day_total:.0f} citywide requests. "
        "Taller and warmer-colored areas on the map are places to prioritize."
    )
    st.caption(
        "This is an estimate, not an exact count. 'Expected requests in next 30 days' means projected total service "
        "requests from that location over the upcoming month."
    )
    st.caption(
        f"Forecast confidence band (citywide): +/- {total_uncertainty / 2.0:.0f} requests "
        f"({uncertainty_pct:.1f}% uncertainty width)."
    )

    if view_mode == "Technical View":
        with st.expander("Technical Details"):
            st.write(f"Target source: `{target_source}`")
            st.write(f"Top predictor: `{top_driver}`")
            st.write(f"Bivariate Moran's I: `{moran.get('bivariate_moran_i', 'n/a')}`")
            st.write(f"Model R2: `{metrics.get('r2', 'n/a')}`")
            if governance:
                holdout = governance.get("evaluation", {}).get("holdout", {})
                st.write(f"Model RMSE: `{holdout.get('rmse', metrics.get('rmse', 'n/a'))}`")
                cv_ens = governance.get("evaluation", {}).get("cross_validation", {}).get("ensemble", {})
                if cv_ens:
                    st.write(
                        "Cross-validation (ensemble): "
                        f"MAE `{cv_ens.get('mae_mean', 'n/a')}`, R2 `{cv_ens.get('r2_mean', 'n/a')}`"
                    )
            st.markdown("**Dataset Usage Check**")
            st.dataframe(_dataset_usage_summary(data_dir), use_container_width=True, hide_index=True)

    landmark_lookup = _load_landmark_lookup(data_dir)
    # Label all cells so tooltips never fall back to "Other Area".
    zone_df["area_label"] = _friendly_labels(zone_df, landmark_lookup, ensure_unique=True)
    zone_df["forecast_today_count"] = zone_df["risk_today"].apply(lambda v: _format_request_count(float(v)))
    zone_df["forecast_today_text"] = zone_df["risk_today"].apply(lambda v: _requests_phrase(float(v)))
    code_q90 = float(zone_df["code_violations_count"].quantile(0.9)) if len(zone_df) else 0.0
    req_q50 = float(zone_df["service_311_count"].quantile(0.5)) if len(zone_df) else 0.0
    zone_df["hidden_risk_flag"] = zone_df.apply(lambda r: _hidden_risk_flag(r, code_q90, req_q50), axis=1)

    top_n = min(20, len(zone_df))
    top_cells = zone_df.nlargest(top_n, "risk_today").copy().reset_index(drop=True)
    top_cells["area_label_display"] = [
        f"{row.area_label} (Top {i + 1})" for i, row in top_cells.iterrows()
    ]
    options = top_cells["area_label_display"].tolist()
    default_idx = 0 if options else None

    left, right = st.columns([3, 1.2], gap="large")
    with left:
        view_state = pdk.ViewState(
            latitude=float(zone_df["centroid_latitude"].median()),
            longitude=float(zone_df["centroid_longitude"].median()),
            zoom=10.8,
            pitch=48,
            bearing=15,
        )
        layer = pdk.Layer(
            "H3HexagonLayer",
            data=zone_df,
            pickable=True,
            stroked=False,
            filled=True,
            extruded=True,
            get_hexagon="h3_cell",
            get_fill_color="color",
            get_elevation="elevation",
            elevation_scale=1,
            opacity=0.92,
        )
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style=pdk.map_styles.CARTO_DARK,
            height=620,
            tooltip={
                "html": "<b>{area_label}</b><br/><b>Expected requests today:</b> {forecast_today_count}<br/><b>Meaning:</b> {forecast_today_text}<br/><b>Code issues:</b> {code_violations_count}<br/><b>311 requests:</b> {service_311_count}",
                "style": {"backgroundColor": "#0f172a", "color": "white"},
            },
        )
        st.pydeck_chart(deck, use_container_width=True)
        st.caption("Color guide: blue = lower need, orange = higher need. Height also increases with need.")
        st.caption("Location names are based on the nearest known place in the city dataset.")

    with right:
        st.markdown("### Area Snapshot")
        selected_area = st.selectbox("Choose an area to inspect", options=options, index=default_idx)
        selected_row = (
            top_cells[top_cells["area_label_display"] == selected_area].iloc[0] if selected_area else zone_df.iloc[0]
        )

        st.markdown(
            f"""
            <div class="metric-card">
              <b>Expected requests this month:</b> {_format_request_count(float(selected_row['predicted_calls_next_30d']))}<br/>
              <b>Forecast range this month:</b> {_format_request_count(float(selected_row.get('predicted_calls_next_30d_p10', 0)))} to {_format_request_count(float(selected_row.get('predicted_calls_next_30d_p90', 0)))}<br/>
              <b>Chance of requests today:</b> {selected_row['chance_today']}<br/>
              <b>What that means:</b> {_requests_phrase(float(selected_row['risk_today']))}<br/>
              <b>Uncertainty score (30d):</b> {_format_request_count(float(selected_row.get('prediction_uncertainty_30d', 0)))}<br/>
              <b>Code issues reported:</b> {selected_row.get('code_violations_count', 0):.0f}<br/>
              <b>311 requests (history):</b> {selected_row.get('service_311_count', 0):.0f}<br/>
              <b>Places/POIs in this area:</b> {selected_row.get('poi_count', 0):.0f}<br/>
              <b>Distance to Station (km):</b> {_format_km(float(selected_row.get('distance_to_nearest_station_km', np.nan)), allow_zero=False)}<br/>
              <b>Distance to Nearest POI (km):</b> {_format_km(float(selected_row.get('distance_to_nearest_poi_km', np.nan)))}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.info(_roi_text(selected_row))

    st.markdown("### Innovation Lab")
    s1, s2, s3, s4 = st.columns(4)
    code_cleanup_pct = s1.slider("Code cleanup effort", min_value=0, max_value=100, value=35, step=5)
    rapid_response_pct = s2.slider("Rapid response coverage", min_value=0, max_value=100, value=30, step=5)
    hotspot_patrol_pct = s3.slider("Hotspot patrol intensity", min_value=0, max_value=100, value=25, step=5)
    event_surge_pct = s4.slider("Special events pressure", min_value=0, max_value=60, value=10, step=5)

    intervention_mult = _intervention_multiplier(code_cleanup_pct, rapid_response_pct, hotspot_patrol_pct)
    event_mult = 1.0 + (event_surge_pct / 100.0) * 0.45
    zone_df["simulated_30d"] = zone_df["predicted_calls_next_30d"] * intervention_mult * event_mult
    zone_df["simulated_today"] = zone_df["risk_today"] * intervention_mult * event_mult
    zone_df["preventable_30d"] = (zone_df["predicted_calls_next_30d"] - zone_df["simulated_30d"]).clip(lower=0)

    base_city_30d = float(zone_df["predicted_calls_next_30d"].sum())
    sim_city_30d = float(zone_df["simulated_30d"].sum())
    prevented_city_30d = max(0.0, base_city_30d - sim_city_30d)
    il1, il2, il3 = st.columns(3)
    il1.metric("Scenario Expected Requests (30d)", f"{sim_city_30d:.0f}")
    il2.metric("Estimated Requests Prevented (30d)", f"{prevented_city_30d:.0f}")
    il3.metric("Scenario Pressure vs Baseline", f"{(sim_city_30d / max(base_city_30d, 1e-9) - 1.0) * 100:.1f}%")

    selected_baseline_30d = _safe_value(selected_row, "predicted_calls_next_30d")
    selected_adjusted_30d = selected_baseline_30d * intervention_mult * event_mult
    selected_chance = _chance_label(
        float(selected_row["risk_today"]) * intervention_mult * event_mult,
        low_cut * max(intervention_mult * event_mult, 1e-9),
        high_cut * max(intervention_mult * event_mult, 1e-9),
    )
    plan_text = _action_plan_text(
        selected_area.replace(" (Top 1)", ""),
        selected_row,
        selected_chance,
        selected_baseline_30d,
        selected_adjusted_30d,
    )
    st.download_button(
        "Download Area Action Plan (.md)",
        data=plan_text,
        file_name="area_action_plan.md",
        mime="text/markdown",
        use_container_width=True,
    )

    curve_baseline = _daily_profile(
        selected_baseline_30d,
        float(selected_row["centroid_latitude"]),
        float(selected_row["centroid_longitude"]),
        multiplier=1.0,
    )
    curve_sim = _daily_profile(
        selected_baseline_30d,
        float(selected_row["centroid_latitude"]),
        float(selected_row["centroid_longitude"]),
        multiplier=intervention_mult * event_mult,
    )
    days = pd.date_range(selected_day, periods=31, freq="D")
    curve_df = pd.DataFrame(
        {
            "date": days,
            "Baseline": curve_baseline,
            "Scenario": curve_sim,
        }
    ).set_index("date")
    st.line_chart(curve_df, use_container_width=True)
    st.caption("Scenario Lab chart: compare baseline demand vs your intervention scenario for this area.")

    st.markdown("### What This Means Today")
    a1, a2, a3 = st.columns(3)
    a1.info("Start with the top 3 areas below for cleanup and enforcement.")
    a2.info("Prioritize places with high code issues and longer distance to a station.")
    a3.info("Check this view weekly and adjust the priority list.")

    st.markdown("### Top Areas Today")
    top_table_raw = zone_df.nlargest(10, "risk_today")[
        [
            "area_label",
            "risk_today",
            "predicted_calls_next_30d",
            "code_violations_count",
            "service_311_count",
            "hidden_risk_flag",
        ]
    ].copy()
    action_labels = []
    for i in range(len(top_table_raw)):
        if i < 3:
            action_labels.append("Act now")
        elif i < 7:
            action_labels.append("Plan this week")
        else:
            action_labels.append("Monitor")
    top_table = pd.DataFrame(
        {
            "Area": top_table_raw["area_label"],
            "Recommended action": action_labels,
            "Chance of requests today": top_table_raw["risk_today"].apply(
                lambda v: _chance_label(float(v), low_cut, high_cut)
            ),
            "Expected requests in next 30 days": top_table_raw["predicted_calls_next_30d"].apply(
                lambda v: _format_request_count(float(v))
            ),
            "Open neighborhood code complaints": top_table_raw["code_violations_count"].apply(
                lambda v: f"{float(v):.0f}"
            ),
            "Past city service requests (311)": top_table_raw["service_311_count"].apply(lambda v: f"{float(v):.0f}"),
            "Blind-spot signal": top_table_raw["hidden_risk_flag"],
            "Why this area is prioritized": top_table_raw.apply(_why_flagged, axis=1),
        }
    )
    st.dataframe(top_table, use_container_width=True, hide_index=True)
    st.caption(
        "Column meanings: Open neighborhood code complaints = unresolved property/neighborhood issues. "
        "Past city service requests (311) = historical requests from residents in that location."
    )

    st.markdown("### Intervention Leaderboard")
    leader = zone_df.nlargest(10, "preventable_30d")[
        ["area_label", "predicted_calls_next_30d", "simulated_30d", "preventable_30d", "hidden_risk_flag"]
    ].copy()
    leader = leader.rename(
        columns={
            "area_label": "Location",
            "predicted_calls_next_30d": "Baseline expected requests (30d)",
            "simulated_30d": "Scenario expected requests (30d)",
            "preventable_30d": "Estimated preventable requests (30d)",
            "hidden_risk_flag": "Blind-spot signal",
        }
    )
    for c in [
        "Baseline expected requests (30d)",
        "Scenario expected requests (30d)",
        "Estimated preventable requests (30d)",
    ]:
        leader[c] = leader[c].map(lambda v: f"{float(v):.0f}")
    st.dataframe(leader, use_container_width=True, hide_index=True)
    st.caption("Use this leaderboard to prioritize where combined cleanup + response actions can create the largest impact.")

    st.markdown("### AI Review Queue (High Uncertainty Zones)")
    review_queue = zone_df.nlargest(8, "prediction_uncertainty_30d")[
        [
            "area_label",
            "prediction_uncertainty_30d",
            "predicted_calls_next_30d_p10",
            "predicted_calls_next_30d_p90",
            "code_violations_count",
            "service_311_count",
        ]
    ].copy()
    review_queue = review_queue.rename(
        columns={
            "area_label": "Location",
            "prediction_uncertainty_30d": "Forecast spread (30d)",
            "predicted_calls_next_30d_p10": "Lower estimate (30d)",
            "predicted_calls_next_30d_p90": "Upper estimate (30d)",
            "code_violations_count": "Open code complaints",
            "service_311_count": "Past 311 requests",
        }
    )
    for c in ["Forecast spread (30d)", "Lower estimate (30d)", "Upper estimate (30d)"]:
        review_queue[c] = review_queue[c].map(lambda v: f"{float(v):.0f}")
    review_queue["Action"] = "Send field verification + refresh data"
    st.dataframe(review_queue, use_container_width=True, hide_index=True)

    st.markdown("### Pitch Brief Generator")
    if st.button("Generate Pitch Narrative", use_container_width=True):
        narrative = build_pitch_summary(data_dir)
        st.session_state["pitch_brief"] = narrative
    if "pitch_brief" in st.session_state:
        st.markdown(st.session_state["pitch_brief"])
        st.download_button(
            "Download Brief (.md)",
            data=st.session_state["pitch_brief"],
            file_name="pitch_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
