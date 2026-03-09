from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import h3
import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


@dataclass
class Step2Config:
    input_dir: Path
    output_dir: Path
    h3_resolution: int = 9


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _pick_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    lookup = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _count_by_cell(df: pd.DataFrame, output_name: str) -> pd.Series:
    if df.empty or "h3_cell" not in df.columns:
        s = pd.Series(dtype=float, name=output_name)
        s.index.name = "h3_cell"
        return s
    return df.groupby("h3_cell").size().rename(output_name)


def _sum_by_cell(df: pd.DataFrame, value_col: str, output_name: str) -> pd.Series:
    if df.empty or "h3_cell" not in df.columns or value_col not in df.columns:
        s = pd.Series(dtype=float, name=output_name)
        s.index.name = "h3_cell"
        return s
    vals = pd.to_numeric(df[value_col], errors="coerce")
    tmp = df.copy()
    tmp["_metric"] = vals
    return tmp.groupby("h3_cell")["_metric"].sum(min_count=1).fillna(0).rename(output_name)


def _parse_city_limit_bbox(city_limit_df: pd.DataFrame) -> tuple[float, float, float, float] | None:
    if city_limit_df.empty:
        return None
    if "geometry_json" not in city_limit_df.columns:
        return None

    min_lon = float("inf")
    min_lat = float("inf")
    max_lon = float("-inf")
    max_lat = float("-inf")

    for raw in city_limit_df["geometry_json"].dropna().astype(str):
        try:
            geom = json.loads(raw)
        except Exception:
            continue
        rings = geom.get("rings", [])
        for ring in rings:
            for pair in ring:
                if not isinstance(pair, list) or len(pair) < 2:
                    continue
                lon, lat = pair[0], pair[1]
                try:
                    lon_f = float(lon)
                    lat_f = float(lat)
                except Exception:
                    continue
                min_lon = min(min_lon, lon_f)
                min_lat = min(min_lat, lat_f)
                max_lon = max(max_lon, lon_f)
                max_lat = max(max_lat, lat_f)

    if min_lon == float("inf"):
        return None
    return (min_lon, min_lat, max_lon, max_lat)


def _assign_h3(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "latitude" not in work.columns or "longitude" not in work.columns:
        return pd.DataFrame()

    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work = work.dropna(subset=["latitude", "longitude"])
    if work.empty:
        return work

    work["h3_cell"] = work.apply(
        lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], resolution), axis=1
    )
    return work


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _nearest_station_distance_km(cells: pd.Series, station_df: pd.DataFrame) -> pd.Series:
    if station_df.empty:
        return pd.Series(np.nan, index=cells.index)

    stations = station_df.copy()
    lat_col = _pick_column(stations.columns, ["latitude", "lat", "y"])
    lon_col = _pick_column(stations.columns, ["longitude", "lon", "lng", "x"])
    if lat_col is None or lon_col is None:
        return pd.Series(np.nan, index=cells.index)

    stations["latitude"] = pd.to_numeric(stations[lat_col], errors="coerce")
    stations["longitude"] = pd.to_numeric(stations[lon_col], errors="coerce")
    stations = stations.dropna(subset=["latitude", "longitude"])
    if stations.empty:
        return pd.Series(np.nan, index=cells.index)

    st_lat = stations["latitude"].to_numpy()
    st_lon = stations["longitude"].to_numpy()

    out: list[float] = []
    for cell in cells:
        lat, lon = h3.cell_to_latlng(cell)
        lat_arr = np.full(st_lat.shape, lat, dtype=float)
        lon_arr = np.full(st_lon.shape, lon, dtype=float)
        dist = _haversine_km(lat_arr, lon_arr, st_lat, st_lon)
        out.append(float(np.min(dist)))
    return pd.Series(out, index=cells.index)


def _nearest_point_distance_km(cells: pd.Series, points_df: pd.DataFrame) -> pd.Series:
    if points_df.empty or "latitude" not in points_df.columns or "longitude" not in points_df.columns:
        return pd.Series(np.nan, index=cells.index)

    points = points_df.copy()
    points["latitude"] = pd.to_numeric(points["latitude"], errors="coerce")
    points["longitude"] = pd.to_numeric(points["longitude"], errors="coerce")
    points = points.dropna(subset=["latitude", "longitude"])
    if points.empty:
        return pd.Series(np.nan, index=cells.index)

    p_lat = points["latitude"].to_numpy()
    p_lon = points["longitude"].to_numpy()

    out: list[float] = []
    for cell in cells:
        lat, lon = h3.cell_to_latlng(cell)
        lat_arr = np.full(p_lat.shape, lat, dtype=float)
        lon_arr = np.full(p_lon.shape, lon, dtype=float)
        dist = _haversine_km(lat_arr, lon_arr, p_lat, p_lon)
        out.append(float(np.min(dist)))
    return pd.Series(out, index=cells.index)


def _parse_first_date(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    col = _pick_column(df.columns, candidates)
    if col is None:
        return None
    return pd.to_datetime(df[col], errors="coerce", utc=True)


def build_grid_features(config: Step2Config) -> pd.DataFrame:
    input_dir = config.input_dir
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    calls = _assign_h3(_read_csv_if_exists(input_dir / "911_calls.csv"), config.h3_resolution)
    code = _assign_h3(_read_csv_if_exists(input_dir / "code_violations.csv"), config.h3_resolution)
    paving = _assign_h3(_read_csv_if_exists(input_dir / "paving_project.csv"), config.h3_resolution)
    stations = _read_csv_if_exists(input_dir / "fire_and_police_station.csv")
    city_limit = _read_csv_if_exists(input_dir / "city_limit.csv")
    poi = _assign_h3(_read_csv_if_exists(input_dir / "point_of_interest.csv"), config.h3_resolution)
    most_visited = _assign_h3(_read_csv_if_exists(input_dir / "most_visited_locations.csv"), config.h3_resolution)
    community_centers = _assign_h3(_read_csv_if_exists(input_dir / "community_centers.csv"), config.h3_resolution)
    parks_and_trail = _assign_h3(_read_csv_if_exists(input_dir / "parks_and_trail.csv"), config.h3_resolution)
    education_facility = _assign_h3(_read_csv_if_exists(input_dir / "education_facility.csv"), config.h3_resolution)
    pharmacy_locator = _assign_h3(_read_csv_if_exists(input_dir / "pharmacy_locator.csv"), config.h3_resolution)
    service_311 = _assign_h3(
        _read_csv_if_exists(input_dir / "received_311_service_requests.csv"), config.h3_resolution
    )
    response_source = "911_calls"

    cells = pd.Index([])
    for df in (
        calls,
        code,
        paving,
        service_311,
        poi,
        most_visited,
        community_centers,
        parks_and_trail,
        education_facility,
        pharmacy_locator,
    ):
        if not df.empty and "h3_cell" in df.columns:
            cells = cells.union(pd.Index(df["h3_cell"].unique()))

    if len(cells) == 0:
        return pd.DataFrame()

    grid = pd.DataFrame({"h3_cell": cells})

    code_count = _count_by_cell(code, "code_violations_count")
    paving_count = _count_by_cell(paving, "paving_project_count")
    service_311_count = _count_by_cell(service_311, "service_311_count")
    poi_count = _count_by_cell(poi, "poi_count")
    most_visited_count = _count_by_cell(most_visited, "most_visited_count")
    community_centers_count = _count_by_cell(community_centers, "community_centers_count")
    parks_and_trail_count = _count_by_cell(parks_and_trail, "parks_and_trail_count")
    education_facility_count = _count_by_cell(education_facility, "education_facility_count")
    pharmacy_locator_count = _count_by_cell(pharmacy_locator, "pharmacy_locator_count")
    calls_count = _count_by_cell(calls, "historical_911_volume")

    visits_col = _pick_column(
        most_visited.columns,
        [
            "number_of_visits",
            "visit_count",
            "visits",
            "monthly_visits",
            "count",
        ],
    )
    most_visited_total = _sum_by_cell(
        most_visited,
        visits_col if visits_col is not None else "__missing__",
        "most_visited_total_visits",
    )

    grid = grid.merge(code_count, on="h3_cell", how="left")
    grid = grid.merge(paving_count, on="h3_cell", how="left")
    grid = grid.merge(service_311_count, on="h3_cell", how="left")
    grid = grid.merge(poi_count, on="h3_cell", how="left")
    grid = grid.merge(most_visited_count, on="h3_cell", how="left")
    grid = grid.merge(community_centers_count, on="h3_cell", how="left")
    grid = grid.merge(parks_and_trail_count, on="h3_cell", how="left")
    grid = grid.merge(education_facility_count, on="h3_cell", how="left")
    grid = grid.merge(pharmacy_locator_count, on="h3_cell", how="left")
    grid = grid.merge(most_visited_total, on="h3_cell", how="left")
    grid = grid.merge(calls_count, on="h3_cell", how="left")
    grid = grid.fillna(
        {
            "code_violations_count": 0,
            "paving_project_count": 0,
            "service_311_count": 0,
            "poi_count": 0,
            "most_visited_count": 0,
            "most_visited_total_visits": 0,
            "community_centers_count": 0,
            "parks_and_trail_count": 0,
            "education_facility_count": 0,
            "pharmacy_locator_count": 0,
            "historical_911_volume": 0,
        }
    )

    grid["distance_to_nearest_station_km"] = _nearest_station_distance_km(grid["h3_cell"], stations)
    grid["distance_to_nearest_poi_km"] = _nearest_point_distance_km(grid["h3_cell"], poi)

    paving_dt = _parse_first_date(paving, ["project_end", "project_start", "updated_date", "created_date"])
    calls_dt = _parse_first_date(calls, ["call_time", "dispatch_time", "created_date"])
    service_dt = _parse_first_date(service_311, ["request_date", "created_date", "open_date", "create_date"])

    # Choose a spatial response source. 911 can be non-spatial in some jurisdictions.
    if not calls.empty and calls_dt is not None and calls_dt.notna().any():
        response_df = calls
        response_dt = calls_dt
        response_source = "911_calls"
    elif not service_311.empty and service_dt is not None and service_dt.notna().any():
        response_df = service_311
        response_dt = service_dt
        response_source = "received_311_service_requests"
    else:
        response_df = pd.DataFrame()
        response_dt = None
        response_source = "none"

    if response_dt is not None and response_dt.notna().any():
        as_of = response_dt.max()
    else:
        as_of = pd.Timestamp.now(tz="UTC")

    if paving_dt is None or paving.empty:
        grid["days_since_last_paving"] = np.nan
    else:
        paving_with_dt = paving.copy()
        paving_with_dt["event_dt"] = paving_dt
        last_paving = paving_with_dt.groupby("h3_cell")["event_dt"].max()
        grid = grid.merge(last_paving.rename("last_paving_dt"), on="h3_cell", how="left")
        grid["days_since_last_paving"] = (as_of - grid["last_paving_dt"]).dt.days
        grid = grid.drop(columns=["last_paving_dt"])

    # Target for baseline model: recent 30-day spatial response demand per cell.
    if response_dt is None or response_df.empty or response_dt.notna().sum() == 0:
        grid["target_calls_next_30d"] = 0.0
        grid["historical_response_volume"] = 0.0
    else:
        response_with_dt = response_df.copy()
        response_with_dt["event_dt"] = response_dt
        window_start = as_of - pd.Timedelta(days=30)
        recent = response_with_dt[response_with_dt["event_dt"].between(window_start, as_of, inclusive="both")]
        target = recent.groupby("h3_cell").size().rename("target_calls_next_30d")
        grid = grid.merge(target, on="h3_cell", how="left")
        grid["target_calls_next_30d"] = grid["target_calls_next_30d"].fillna(0)
        response_hist = response_df.groupby("h3_cell").size().rename("historical_response_volume")
        grid = grid.merge(response_hist, on="h3_cell", how="left")
        grid["historical_response_volume"] = grid["historical_response_volume"].fillna(0)

    # Keep this for backward compatibility in downstream visuals.
    if "historical_response_volume" in grid.columns:
        grid["historical_911_volume"] = grid["historical_response_volume"]
    grid["target_source"] = response_source

    centroids = grid["h3_cell"].apply(h3.cell_to_latlng)
    grid["centroid_latitude"] = centroids.apply(lambda x: x[0])
    grid["centroid_longitude"] = centroids.apply(lambda x: x[1])

    city_bbox = _parse_city_limit_bbox(city_limit)
    if city_bbox is not None:
        min_lon, min_lat, max_lon, max_lat = city_bbox
        in_city = (
            (grid["centroid_longitude"] >= min_lon)
            & (grid["centroid_longitude"] <= max_lon)
            & (grid["centroid_latitude"] >= min_lat)
            & (grid["centroid_latitude"] <= max_lat)
        )
        grid = grid[in_city].reset_index(drop=True)

    numeric_cols = [
        "code_violations_count",
        "paving_project_count",
        "service_311_count",
        "poi_count",
        "most_visited_count",
        "most_visited_total_visits",
        "community_centers_count",
        "parks_and_trail_count",
        "education_facility_count",
        "pharmacy_locator_count",
        "historical_911_volume",
        "historical_response_volume",
        "distance_to_nearest_station_km",
        "distance_to_nearest_poi_km",
        "days_since_last_paving",
        "target_calls_next_30d",
    ]
    for col in numeric_cols:
        if col in grid.columns:
            grid[col] = pd.to_numeric(grid[col], errors="coerce")

    grid.to_csv(output_dir / "grid_features.csv", index=False)
    return grid
