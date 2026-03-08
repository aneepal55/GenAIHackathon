from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _random_points(rng: np.random.Generator, n: int, lat0: float, lon0: float, spread: float) -> tuple[np.ndarray, np.ndarray]:
    lat = lat0 + rng.normal(0, spread, n)
    lon = lon0 + rng.normal(0, spread, n)
    return lat, lon


def write_sample_data(output_dir: str | Path, seed: int = 42) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    now = pd.Timestamp.now(tz="UTC")
    base_lat, base_lon = 32.3668, -86.3000  # Montgomery, AL

    n_calls = 900
    call_lat, call_lon = _random_points(rng, n_calls, base_lat, base_lon, 0.035)
    calls = pd.DataFrame(
        {
            "call_id": [f"C{i:05d}" for i in range(n_calls)],
            "call_time": (now - pd.to_timedelta(rng.integers(0, 90, n_calls), unit="D")).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dispatch_time": (now - pd.to_timedelta(rng.integers(0, 90, n_calls), unit="D")).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "address": [f"{100 + (i % 900)} SAMPLE ST" for i in range(n_calls)],
            "latitude": call_lat,
            "longitude": call_lon,
        }
    )

    n_code = 650
    code_lat, code_lon = _random_points(rng, n_code, base_lat + 0.004, base_lon - 0.004, 0.03)
    code = pd.DataFrame(
        {
            "violation_id": [f"V{i:05d}" for i in range(n_code)],
            "open_date": (now - pd.to_timedelta(rng.integers(15, 120, n_code), unit="D")).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "created_date": (now - pd.to_timedelta(rng.integers(15, 120, n_code), unit="D")).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "address": [f"{200 + (i % 700)} MOCK AVE" for i in range(n_code)],
            "latitude": code_lat,
            "longitude": code_lon,
        }
    )

    n_311 = 500
    req_lat, req_lon = _random_points(rng, n_311, base_lat - 0.003, base_lon + 0.003, 0.032)
    service = pd.DataFrame(
        {
            "request_id": [f"R{i:05d}" for i in range(n_311)],
            "request_date": (now - pd.to_timedelta(rng.integers(0, 90, n_311), unit="D")).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "created_date": (now - pd.to_timedelta(rng.integers(0, 90, n_311), unit="D")).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "address": [f"{300 + (i % 600)} DEMO BLVD" for i in range(n_311)],
            "latitude": req_lat,
            "longitude": req_lon,
        }
    )

    n_paving = 180
    pave_lat, pave_lon = _random_points(rng, n_paving, base_lat + 0.001, base_lon + 0.001, 0.03)
    paving = pd.DataFrame(
        {
            "project_id": [f"P{i:05d}" for i in range(n_paving)],
            "project_start": (now - pd.to_timedelta(rng.integers(30, 220, n_paving), unit="D")).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "project_end": (now - pd.to_timedelta(rng.integers(0, 120, n_paving), unit="D")).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "address": [f"{400 + (i % 500)} TEST RD" for i in range(n_paving)],
            "latitude": pave_lat,
            "longitude": pave_lon,
        }
    )

    stations = pd.DataFrame(
        {
            "station_name": ["Central Police", "North Fire", "South Fire", "East Police", "West Fire"],
            "address": ["100 CORE ST", "250 NORTH ST", "350 SOUTH ST", "450 EAST ST", "550 WEST ST"],
            "latitude": [32.365, 32.392, 32.338, 32.364, 32.371],
            "longitude": [-86.300, -86.292, -86.304, -86.261, -86.341],
        }
    )

    n_poi = 320
    poi_lat, poi_lon = _random_points(rng, n_poi, base_lat + 0.002, base_lon - 0.002, 0.028)
    poi = pd.DataFrame(
        {
            "poi_id": [f"POI{i:05d}" for i in range(n_poi)],
            "name": [f"POI {i:03d}" for i in range(n_poi)],
            "address": [f"{500 + (i % 500)} PLACE LN" for i in range(n_poi)],
            "latitude": poi_lat,
            "longitude": poi_lon,
        }
    )

    n_visited = 160
    vis_lat, vis_lon = _random_points(rng, n_visited, base_lat + 0.0015, base_lon - 0.0015, 0.022)
    visited = pd.DataFrame(
        {
            "location_name": [f"HOTSPOT {i:03d}" for i in range(n_visited)],
            "address": [f"{600 + (i % 300)} VISITOR WAY" for i in range(n_visited)],
            "location_category": rng.choice(
                ["Retail", "Dining", "Park", "Civic", "School"], size=n_visited, replace=True
            ),
            "number_of_visits": rng.integers(50, 2500, size=n_visited),
            "latitude": vis_lat,
            "longitude": vis_lon,
        }
    )

    city_limit = pd.DataFrame(
        {
            "name": ["Montgomery City Limit"],
            "geometry_json": ['{"rings":[[[-86.42,32.29],[-86.18,32.29],[-86.18,32.46],[-86.42,32.46],[-86.42,32.29]]]}'],
        }
    )

    outputs = {
        "911_calls.csv": calls,
        "code_violations.csv": code,
        "received_311_service_requests.csv": service,
        "paving_project.csv": paving,
        "fire_and_police_station.csv": stations,
        "point_of_interest.csv": poi,
        "most_visited_locations.csv": visited,
        "city_limit.csv": city_limit,
    }

    written: dict[str, Path] = {}
    for filename, frame in outputs.items():
        target = out / filename
        frame.to_csv(target, index=False)
        written[filename] = target
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create demo-safe synthetic data for Montgomery Guardian.")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for CSV files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_sample_data(args.output_dir, seed=args.seed)
    print("Sample data generated:")
    for name, path in outputs.items():
        print(f"- {name}: {path.resolve()}")


if __name__ == "__main__":
    main()
