from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
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


def _run_cv(X: pd.DataFrame, y: pd.Series, folds: int = 5) -> dict:
    n = int(len(X))
    if n < 10:
        return {}

    cv = KFold(n_splits=min(folds, max(2, n // 3)), shuffle=True, random_state=42)
    rf_mae: list[float] = []
    rf_r2: list[float] = []
    gbr_mae: list[float] = []
    gbr_r2: list[float] = []
    ens_mae: list[float] = []
    ens_r2: list[float] = []

    for tr_idx, te_idx in cv.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        gbr = GradientBoostingRegressor(
            random_state=42,
            n_estimators=260,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            loss="squared_error",
        )

        rf.fit(X_tr, y_tr)
        gbr.fit(X_tr, y_tr)

        p_rf = rf.predict(X_te)
        p_gbr = gbr.predict(X_te)
        p_ens = 0.6 * p_rf + 0.4 * p_gbr

        rf_mae.append(float(mean_absolute_error(y_te, p_rf)))
        gbr_mae.append(float(mean_absolute_error(y_te, p_gbr)))
        ens_mae.append(float(mean_absolute_error(y_te, p_ens)))
        rf_r2.append(float(r2_score(y_te, p_rf)))
        gbr_r2.append(float(r2_score(y_te, p_gbr)))
        ens_r2.append(float(r2_score(y_te, p_ens)))

    return {
        "folds": int(len(rf_mae)),
        "rf": {"mae_mean": float(np.mean(rf_mae)), "r2_mean": float(np.mean(rf_r2))},
        "gbr": {"mae_mean": float(np.mean(gbr_mae)), "r2_mean": float(np.mean(gbr_r2))},
        "ensemble": {"mae_mean": float(np.mean(ens_mae)), "r2_mean": float(np.mean(ens_r2))},
    }


def _demand_tier(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=object)
    q1 = float(values.quantile(0.5))
    q2 = float(values.quantile(0.85))
    out = []
    for v in values.fillna(0):
        if v >= q2:
            out.append("critical")
        elif v >= q1:
            out.append("elevated")
        else:
            out.append("baseline")
    return pd.Series(out, index=values.index)


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
    rf_model = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    gbr_model = GradientBoostingRegressor(
        random_state=42,
        n_estimators=320,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        loss="squared_error",
    )
    p10_model = GradientBoostingRegressor(
        random_state=42,
        n_estimators=240,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        loss="quantile",
        alpha=0.10,
    )
    p90_model = GradientBoostingRegressor(
        random_state=42,
        n_estimators=240,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        loss="quantile",
        alpha=0.90,
    )

    rf_model.fit(X_train, y_train)
    gbr_model.fit(X_train, y_train)
    p10_model.fit(X_train, y_train)
    p90_model.fit(X_train, y_train)

    y_pred_rf_test = rf_model.predict(X_test)
    y_pred_gbr_test = gbr_model.predict(X_test)
    y_pred_test = 0.6 * y_pred_rf_test + 0.4 * y_pred_gbr_test
    cv_summary = _run_cv(X, y)
    metrics = {
        "model": "ensemble_rf_gbr_quantile",
        "rows": int(len(work)),
        "mae": float(mean_absolute_error(y_test, y_pred_test)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "r2": float(r2_score(y_test, y_pred_test)),
        "rf_mae": float(mean_absolute_error(y_test, y_pred_rf_test)),
        "rf_r2": float(r2_score(y_test, y_pred_rf_test)),
        "gbr_mae": float(mean_absolute_error(y_test, y_pred_gbr_test)),
        "gbr_r2": float(r2_score(y_test, y_pred_gbr_test)),
        "ensemble_weights": {"random_forest": 0.6, "gradient_boosting": 0.4},
        "target_source": target_source,
        "target_name": "target_calls_next_30d",
        "cv_summary": cv_summary,
    }

    rf_full = rf_model.predict(X)
    gbr_full = gbr_model.predict(X)
    p10_full = p10_model.predict(X)
    p90_full = p90_model.predict(X)

    work["predicted_calls_next_30d"] = 0.6 * rf_full + 0.4 * gbr_full
    work["predicted_calls_next_30d_p10"] = np.minimum(p10_full, work["predicted_calls_next_30d"])
    work["predicted_calls_next_30d_p90"] = np.maximum(p90_full, work["predicted_calls_next_30d"])
    work["prediction_uncertainty_30d"] = (
        work["predicted_calls_next_30d_p90"] - work["predicted_calls_next_30d_p10"]
    ).clip(lower=0)
    work["demand_tier"] = _demand_tier(work["predicted_calls_next_30d"])
    work["intervention_priority_score"] = (
        0.45 * work["code_violations_count"].fillna(0)
        + 0.25 * work["service_311_count"].fillna(0)
        + 0.15 * work["predicted_calls_next_30d"].fillna(0)
        + 0.15 * work["prediction_uncertainty_30d"].fillna(0)
    )
    work.to_csv(out_dir / "predictions.csv", index=False)
    _write_calibration_report(work, out_dir)

    rf_imp = np.asarray(rf_model.feature_importances_, dtype=float)
    gbr_imp = np.asarray(gbr_model.feature_importances_, dtype=float)
    combined_imp = 0.6 * rf_imp + 0.4 * gbr_imp
    importance = pd.DataFrame({"feature": MODEL_FEATURES, "importance": combined_imp})
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(out_dir / "top_predictors.csv", index=False)

    governance = {
        "model_family": "stacked_ensemble",
        "point_models": ["random_forest_regressor", "gradient_boosting_regressor"],
        "uncertainty_model": "gradient_boosting_quantile",
        "evaluation": {
            "holdout": {
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            },
            "cross_validation": cv_summary,
        },
        "feature_count": int(len(MODEL_FEATURES)),
        "top_features": importance.head(5).to_dict(orient="records"),
    }
    with open(out_dir / "model_governance.json", "w", encoding="utf-8") as f:
        json.dump(governance, f, indent=2)

    with open(out_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
