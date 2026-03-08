from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml


@dataclass
class DatasetConfig:
    name: str
    url: str
    enabled: bool = True
    where: str = "1=1"
    out_fields: str = "*"
    date_fields: list[str] | None = None
    address_fields: list[str] | None = None


def load_config(config_path: str | Path) -> list[DatasetConfig]:
    with open(config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    raw_items = payload.get("datasets", [])
    return [DatasetConfig(**item) for item in raw_items]


def fetch_arcgis_features(
    url: str,
    where: str = "1=1",
    out_fields: str = "*",
    chunk_size: int = 2000,
    timeout_seconds: int = 60,
) -> list[dict[str, Any]]:
    offset = 0
    all_features: list[dict[str, Any]] = []

    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "outSR": "4326",
            "f": "json",
            "returnGeometry": "true",
            "resultOffset": offset,
            "resultRecordCount": chunk_size,
        }
        response = requests.get(url, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise RuntimeError(f"ArcGIS error for {url}: {data['error']}")

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        if len(features) < chunk_size:
            break
        offset += chunk_size

    return all_features


def _extract_feature_row(feature: dict[str, Any]) -> dict[str, Any]:
    attrs = feature.get("attributes", {}) or {}
    geom = feature.get("geometry", {}) or {}

    row = dict(attrs)
    # ArcGIS point geometry generally uses x/y.
    if "x" in geom and "y" in geom:
        row["longitude"] = geom["x"]
        row["latitude"] = geom["y"]
    elif "longitude" in geom and "latitude" in geom:
        row["longitude"] = geom["longitude"]
        row["latitude"] = geom["latitude"]
    elif geom:
        # Preserve non-point geometries (polygon/polyline) for downstream spatial use.
        row["geometry_json"] = json.dumps(geom, separators=(",", ":"))

    return row


def _normalize_date_series(series: pd.Series) -> pd.Series:
    # ArcGIS frequently returns epoch milliseconds for date fields.
    if pd.api.types.is_numeric_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce", utc=True, unit="ms")
        return parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    numeric_candidate = pd.to_numeric(series, errors="coerce")
    numeric_ratio = numeric_candidate.notna().mean() if len(series) else 0.0
    if numeric_ratio > 0.8:
        parsed = pd.to_datetime(numeric_candidate, errors="coerce", utc=True, unit="ms")
        return parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_dataframe(
    df: pd.DataFrame,
    date_fields: list[str] | None = None,
    address_fields: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()

    for col in date_fields or []:
        if col in out.columns:
            out[col] = _normalize_date_series(out[col])

    for col in address_fields or []:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype(str)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .str.upper()
            )

    # Coerce standard coordinate columns if they already exist.
    for lat_candidate in ("latitude", "lat", "y"):
        if lat_candidate in out.columns and "latitude" not in out.columns:
            out["latitude"] = out[lat_candidate]
    for lon_candidate in ("longitude", "lon", "lng", "x"):
        if lon_candidate in out.columns and "longitude" not in out.columns:
            out["longitude"] = out[lon_candidate]

    if "latitude" in out.columns:
        out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    if "longitude" in out.columns:
        out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")

    return out


def run_ingestion(config_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    cfg = load_config(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    for dataset in cfg:
        if not dataset.enabled:
            continue

        features = fetch_arcgis_features(
            url=dataset.url,
            where=dataset.where,
            out_fields=dataset.out_fields,
        )
        rows = [_extract_feature_row(f) for f in features]
        df = pd.DataFrame(rows)
        df = normalize_dataframe(
            df,
            date_fields=dataset.date_fields,
            address_fields=dataset.address_fields,
        )

        target = out_dir / f"{dataset.name}.csv"
        df.to_csv(target, index=False)
        outputs[dataset.name] = target

    return outputs
