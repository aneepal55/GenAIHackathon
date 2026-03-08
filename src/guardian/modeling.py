from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


MODEL_FEATURES = [
    "code_violations_count",
    "distance_to_nearest_station_km",
    "distance_to_nearest_poi_km",
    "days_since_last_paving",
    "historical_response_volume",
    "service_311_count",
    "poi_count",
    "most_visited_count",
    "most_visited_total_visits",
    "community_centers_count",
    "parks_and_trail_count",
    "education_facility_count",
    "pharmacy_locator_count",
    "paving_project_count",
]


def _write_calibration_report(work: pd.DataFrame, output_dir: Path) -> None:
    frame = work[["target_calls_next_30d", "predicted_calls_next_30d"]].copy()
    frame = frame.rename(columns={"target_calls_next_30d": "actual", "predicted_calls_next_30d": "pred"})
    if frame.empty:
        return
    if frame["pred"].nunique() == 1:
        frame["pred_decile"] = 0
    else:
        frame["pred_decile"] = pd.qcut(frame["pred"], q=10, labels=False, duplicates="drop")

    cal = (
        frame.groupby("pred_decile", dropna=False)
        .agg(
            bucket_size=("pred", "size"),
            avg_predicted=("pred", "mean"),
            avg_actual=("actual", "mean"),
        )
        .reset_index()
        .sort_values("pred_decile")
    )
    cal["calibration_gap"] = cal["avg_predicted"] - cal["avg_actual"]
    cal.to_csv(output_dir / "calibration_by_decile.csv", index=False)


def train_risk_model(grid: pd.DataFrame, output_dir: str | Path) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if grid.empty:
        raise ValueError("grid is empty; run feature engineering first.")

    work = grid.copy()
    target_source = "unknown"
    if "target_source" in work.columns and work["target_source"].notna().any():
        target_source = str(work["target_source"].dropna().iloc[0])
    for col in MODEL_FEATURES + ["target_calls_next_30d"]:
        if col not in work.columns:
            work[col] = np.nan

    # Keep raw values for downstream display; only impute features for model fitting.
    X = work[MODEL_FEATURES].copy().fillna(0)
    y = work["target_calls_next_30d"].copy().fillna(0)

    if len(work) < 5:
        # Small-data fallback.
        mean_pred = float(y.mean()) if len(y) else 0.0
        work["predicted_calls_next_30d"] = mean_pred
        work.to_csv(out_dir / "predictions.csv", index=False)
        _write_calibration_report(work, out_dir)
        metrics = {
            "model": "fallback_mean",
            "rows": int(len(work)),
            "mae": None,
            "r2": None,
            "target_source": target_source,
            "target_name": "target_calls_next_30d",
        }
        with open(out_dir / "model_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    metrics = {
        "model": "random_forest_regressor",
        "rows": int(len(work)),
        "mae": float(mean_absolute_error(y_test, y_pred_test)),
        "r2": float(r2_score(y_test, y_pred_test)),
        "target_source": target_source,
        "target_name": "target_calls_next_30d",
    }

    work["predicted_calls_next_30d"] = model.predict(X)
    work.to_csv(out_dir / "predictions.csv", index=False)
    _write_calibration_report(work, out_dir)

    importance = pd.DataFrame({"feature": MODEL_FEATURES, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(out_dir / "top_predictors.csv", index=False)

    with open(out_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
