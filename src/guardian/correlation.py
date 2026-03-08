from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import h3
import numpy as np
import pandas as pd


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


def _assign_h3(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
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


def _row_standardized_weights(cells: list[str]) -> tuple[dict[str, list[str]], float]:
    cell_set = set(cells)
    neighbors: dict[str, list[str]] = {}
    s0 = 0.0
    for cell in cells:
        ngh = [n for n in h3.grid_disk(cell, 1) if n != cell and n in cell_set]
        neighbors[cell] = ngh
        if ngh:
            s0 += 1.0  # row-standardized row sums to 1
    return neighbors, s0


def _spatial_lag(values: pd.Series, neighbors: dict[str, list[str]]) -> pd.Series:
    out: list[float] = []
    for cell, value in values.items():
        ngh = neighbors.get(cell, [])
        if not ngh:
            out.append(0.0)
            continue
        out.append(float(values.loc[ngh].mean()))
    return pd.Series(out, index=values.index)


def _bivariate_morans_i(zx: np.ndarray, wzy: np.ndarray, s0: float) -> float:
    if len(zx) == 0:
        return float("nan")
    denom = float(np.dot(zx, zx))
    if denom <= 0 or s0 <= 0:
        return float("nan")
    return (len(zx) / s0) * float(np.dot(zx, wzy)) / denom


def run_bivariate_moran(
    input_dir: str | Path,
    output_dir: str | Path,
    h3_resolution: int = 9,
    lag_days: int = 30,
    window_days: int = 30,
    permutations: int = 499,
    random_seed: int = 42,
) -> dict:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    code_raw = _assign_h3(_read_csv_if_exists(input_path / "code_violations.csv"), h3_resolution)
    calls_raw = _assign_h3(_read_csv_if_exists(input_path / "911_calls.csv"), h3_resolution)
    response_dataset_used = "911_calls"
    if calls_raw.empty:
        # Fallback for jurisdictions where 911 is published as aggregated non-spatial counts.
        calls_raw = _assign_h3(_read_csv_if_exists(input_path / "received_311_service_requests.csv"), h3_resolution)
        response_dataset_used = "received_311_service_requests"
    if code_raw.empty or calls_raw.empty:
        raise ValueError(
            "Need non-empty code_violations.csv and a spatial response dataset "
            "(911_calls.csv or received_311_service_requests.csv) with latitude/longitude."
        )

    code_dt_col = _pick_column(code_raw.columns, ["open_date", "created_date", "request_date", "date"])
    calls_dt_col = _pick_column(
        calls_raw.columns,
        ["call_time", "dispatch_time", "created_date", "incident_date", "request_date", "open_date"],
    )
    if code_dt_col is None or calls_dt_col is None:
        raise ValueError("Date columns not found for code violations or 911 calls.")

    code_raw["event_dt"] = pd.to_datetime(code_raw[code_dt_col], errors="coerce", utc=True)
    calls_raw["event_dt"] = pd.to_datetime(calls_raw[calls_dt_col], errors="coerce", utc=True)
    code_raw = code_raw.dropna(subset=["event_dt"])
    calls_raw = calls_raw.dropna(subset=["event_dt"])
    if code_raw.empty or calls_raw.empty:
        raise ValueError("No valid timestamps after parsing.")

    as_of = calls_raw["event_dt"].max()
    response_start = as_of - pd.Timedelta(days=window_days)
    predictor_end = response_start
    predictor_start = predictor_end - pd.Timedelta(days=lag_days)

    code_win = code_raw[code_raw["event_dt"].between(predictor_start, predictor_end, inclusive="left")]
    calls_win = calls_raw[calls_raw["event_dt"].between(response_start, as_of, inclusive="both")]

    x = code_win.groupby("h3_cell").size().rename("code_count_lagged")
    y = calls_win.groupby("h3_cell").size().rename("calls_count_response")

    cells = sorted(set(x.index).union(set(y.index)))
    if not cells:
        raise ValueError("No H3 cells in selected time windows.")

    frame = pd.DataFrame({"h3_cell": cells})
    frame = frame.merge(x, on="h3_cell", how="left")
    frame = frame.merge(y, on="h3_cell", how="left")
    frame = frame.fillna({"code_count_lagged": 0, "calls_count_response": 0})

    neighbors, s0 = _row_standardized_weights(frame["h3_cell"].tolist())
    frame = frame.set_index("h3_cell")

    x_vals = frame["code_count_lagged"].astype(float)
    y_vals = frame["calls_count_response"].astype(float)
    zx = (x_vals - x_vals.mean()) / (x_vals.std(ddof=0) or 1.0)
    zy = (y_vals - y_vals.mean()) / (y_vals.std(ddof=0) or 1.0)
    wzy = _spatial_lag(zy, neighbors)

    observed_i = _bivariate_morans_i(zx.to_numpy(), wzy.to_numpy(), s0)

    rng = np.random.default_rng(random_seed)
    null_vals = []
    zy_np = zy.to_numpy().copy()
    for _ in range(permutations):
        rng.shuffle(zy_np)
        shuffled = pd.Series(zy_np, index=zy.index)
        wzy_perm = _spatial_lag(shuffled, neighbors).to_numpy()
        null_vals.append(_bivariate_morans_i(zx.to_numpy(), wzy_perm, s0))
    null_arr = np.array(null_vals, dtype=float)
    valid_null = null_arr[~np.isnan(null_arr)]
    if len(valid_null) == 0 or np.isnan(observed_i):
        p_value = float("nan")
    else:
        extreme = np.sum(np.abs(valid_null) >= abs(observed_i))
        p_value = float((extreme + 1) / (len(valid_null) + 1))

    local = zx * wzy
    frame["z_code"] = zx
    frame["wz_calls"] = wzy
    frame["local_bv_moran"] = local
    frame["cluster_type"] = np.select(
        [
            (zx >= 0) & (wzy >= 0),
            (zx < 0) & (wzy < 0),
            (zx >= 0) & (wzy < 0),
            (zx < 0) & (wzy >= 0),
        ],
        ["HH", "LL", "HL", "LH"],
        default="NA",
    )

    out_cells = frame.reset_index()
    out_cells.to_csv(output_path / "bivariate_moran_cells.csv", index=False)

    result = {
        "bivariate_moran_i": None if np.isnan(observed_i) else float(observed_i),
        "p_value_two_sided": None if np.isnan(p_value) else float(p_value),
        "response_dataset_used": response_dataset_used,
        "permutations": int(permutations),
        "h3_resolution": int(h3_resolution),
        "lag_days": int(lag_days),
        "window_days": int(window_days),
        "predictor_window_start": predictor_start.isoformat(),
        "predictor_window_end": predictor_end.isoformat(),
        "response_window_start": response_start.isoformat(),
        "response_window_end": as_of.isoformat(),
        "n_cells": int(len(frame)),
    }
    with open(output_path / "bivariate_moran_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
