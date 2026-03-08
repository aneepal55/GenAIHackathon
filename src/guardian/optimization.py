from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").fillna(0)
    std = float(vals.std(ddof=0))
    if std == 0:
        return pd.Series(np.zeros(len(vals)), index=vals.index)
    return (vals - float(vals.mean())) / std


def _build_candidates(pred: pd.DataFrame) -> pd.DataFrame:
    work = pred.copy()
    for col in [
        "predicted_calls_next_30d",
        "code_violations_count",
        "service_311_count",
        "distance_to_nearest_station_km",
        "prediction_uncertainty_30d",
        "centroid_latitude",
        "centroid_longitude",
    ]:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    if "h3_cell" not in work.columns:
        work["h3_cell"] = [f"cell_{i}" for i in range(len(work))]

    station_q75 = float(work["distance_to_nearest_station_km"].quantile(0.75)) if len(work) else 0.0
    work["underserved_flag"] = (
        (work["distance_to_nearest_station_km"] >= station_q75) | (work["prediction_uncertainty_30d"] > 0)
    )

    rows: list[dict] = []
    for _, row in work.iterrows():
        base = float(row["predicted_calls_next_30d"])
        if base <= 0:
            continue

        code = float(row["code_violations_count"])
        service_311 = float(row["service_311_count"])
        station = float(row["distance_to_nearest_station_km"])
        uncertainty = float(row["prediction_uncertainty_30d"])

        code_intensity = min(1.0, np.log1p(max(0.0, code)) / 4.0)
        service_intensity = min(1.0, np.log1p(max(0.0, service_311)) / 6.0)
        station_gap = min(1.0, max(0.0, station) / 5.0)
        uncertainty_intensity = min(1.0, np.log1p(max(0.0, uncertainty)) / 4.0)

        actions = [
            {
                "action": "Code Enforcement Blitz",
                "cost_usd": 1800.0 + 110.0 * code + 30.0 * service_intensity * 100.0,
                "reduction_pct": min(0.42, 0.09 + 0.25 * code_intensity + 0.05 * service_intensity),
            },
            {
                "action": "Rapid Response Coverage",
                "cost_usd": 2200.0 + 900.0 * station_gap + 400.0 * service_intensity,
                "reduction_pct": min(0.34, 0.06 + 0.16 * station_gap + 0.08 * service_intensity),
            },
            {
                "action": "Hotspot Patrol Windows",
                "cost_usd": 1900.0 + 700.0 * service_intensity + 300.0 * station_gap,
                "reduction_pct": min(0.29, 0.05 + 0.13 * service_intensity + 0.06 * station_gap),
            },
            {
                "action": "Lighting + Cleanup Surge",
                "cost_usd": 2600.0 + 80.0 * code + 500.0 * uncertainty_intensity,
                "reduction_pct": min(0.30, 0.04 + 0.11 * code_intensity + 0.10 * uncertainty_intensity),
            },
        ]

        for act in actions:
            prevented = base * float(act["reduction_pct"])
            if prevented <= 0:
                continue
            cost = max(1.0, float(act["cost_usd"]))
            roi = prevented / (cost / 1000.0)
            rows.append(
                {
                    "h3_cell": row["h3_cell"],
                    "centroid_latitude": float(row["centroid_latitude"]),
                    "centroid_longitude": float(row["centroid_longitude"]),
                    "action": act["action"],
                    "cost_usd": cost,
                    "reduction_pct": float(act["reduction_pct"]) * 100.0,
                    "baseline_calls_30d": base,
                    "prevented_calls_30d": prevented,
                    "roi_calls_prevented_per_1k": roi,
                    "code_violations_count": code,
                    "service_311_count": service_311,
                    "distance_to_nearest_station_km": station,
                    "prediction_uncertainty_30d": uncertainty,
                    "underserved_flag": bool(row["underserved_flag"]),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Normalized score to support ranking alongside ROI.
    out["priority_score"] = (
        0.45 * _zscore(out["prevented_calls_30d"])
        + 0.25 * _zscore(out["roi_calls_prevented_per_1k"])
        + 0.15 * _zscore(out["code_violations_count"])
        + 0.15 * _zscore(out["prediction_uncertainty_30d"])
    )
    return out.sort_values("priority_score", ascending=False).reset_index(drop=True)


def _select_portfolio(
    candidates: pd.DataFrame,
    budget_usd: float,
    min_equity_share: float = 0.35,
    max_actions_per_cell: int = 2,
) -> pd.DataFrame:
    if candidates.empty or budget_usd <= 0:
        return pd.DataFrame(columns=candidates.columns)

    work = candidates.sort_values("roi_calls_prevented_per_1k", ascending=False).copy()
    selected_idx: list[int] = []
    spent = 0.0
    spent_equity = 0.0
    per_cell_count: dict[str, int] = {}

    equity_target = budget_usd * min_equity_share
    underserved = work[work["underserved_flag"] == True]

    def try_take(frame: pd.DataFrame, only_underserved: bool) -> None:
        nonlocal spent, spent_equity
        for idx, row in frame.iterrows():
            if idx in selected_idx:
                continue
            cell = str(row["h3_cell"])
            if per_cell_count.get(cell, 0) >= max_actions_per_cell:
                continue
            cost = float(row["cost_usd"])
            if spent + cost > budget_usd:
                continue
            if only_underserved and not bool(row["underserved_flag"]):
                continue
            selected_idx.append(idx)
            spent += cost
            if bool(row["underserved_flag"]):
                spent_equity += cost
            per_cell_count[cell] = per_cell_count.get(cell, 0) + 1

    if equity_target > 0:
        try_take(underserved, only_underserved=True)
        # If equity target still unmet, continue with remaining underserved first.
        if spent_equity < equity_target:
            try_take(underserved, only_underserved=True)

    try_take(work, only_underserved=False)

    picked = work.loc[selected_idx].copy() if selected_idx else pd.DataFrame(columns=work.columns)
    if picked.empty:
        return picked
    picked["budget_usd"] = float(budget_usd)
    picked["equity_spend_share_pct"] = (spent_equity / max(spent, 1e-9)) * 100.0
    return picked


def _early_warning_alerts(pred: pd.DataFrame) -> pd.DataFrame:
    work = pred.copy()
    for col in [
        "predicted_calls_next_30d",
        "predicted_calls_next_30d_p90",
        "code_violations_count",
        "service_311_count",
        "prediction_uncertainty_30d",
        "distance_to_nearest_station_km",
        "centroid_latitude",
        "centroid_longitude",
    ]:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    stress = (
        0.35 * _zscore(work["predicted_calls_next_30d"])
        + 0.20 * _zscore(work["code_violations_count"])
        + 0.20 * _zscore(work["service_311_count"])
        + 0.15 * _zscore(work["prediction_uncertainty_30d"])
        + 0.10 * _zscore(work["distance_to_nearest_station_km"])
    )
    work["stress_index"] = stress

    q90 = float(stress.quantile(0.90)) if len(stress) else 0.0
    q75 = float(stress.quantile(0.75)) if len(stress) else 0.0
    work["alert_level"] = np.select(
        [work["stress_index"] >= q90, work["stress_index"] >= q75],
        ["critical", "elevated"],
        default="watch",
    )

    reasons: list[str] = []
    for _, row in work.iterrows():
        triggers: list[str] = []
        if row["predicted_calls_next_30d"] >= float(work["predicted_calls_next_30d"].quantile(0.85)):
            triggers.append("high predicted demand")
        if row["code_violations_count"] >= float(work["code_violations_count"].quantile(0.85)):
            triggers.append("concentrated code issues")
        if row["prediction_uncertainty_30d"] >= float(work["prediction_uncertainty_30d"].quantile(0.85)):
            triggers.append("high forecast uncertainty")
        if row["distance_to_nearest_station_km"] >= float(work["distance_to_nearest_station_km"].quantile(0.75)):
            triggers.append("longer response distance")
        if not triggers:
            triggers.append("stacked moderate stress signals")
        reasons.append(", ".join(triggers))
    work["alert_reason"] = reasons

    cols = [
        "h3_cell",
        "centroid_latitude",
        "centroid_longitude",
        "alert_level",
        "stress_index",
        "alert_reason",
        "predicted_calls_next_30d",
        "predicted_calls_next_30d_p90",
        "code_violations_count",
        "service_311_count",
        "prediction_uncertainty_30d",
        "distance_to_nearest_station_km",
    ]
    return work[cols].sort_values("stress_index", ascending=False).head(120).reset_index(drop=True)


def generate_operations_plan(
    input_dir: str | Path,
    output_dir: str | Path,
    budget_levels_usd: tuple[int, ...] = (120000, 300000, 600000),
    min_equity_share: float = 0.35,
) -> dict:
    src = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pred = _read_csv(src / "predictions.csv")
    if pred.empty:
        raise ValueError("Missing predictions.csv; run Step 2 before operations optimization.")

    candidates = _build_candidates(pred)
    if candidates.empty:
        raise ValueError("No valid optimization candidates produced from predictions.")
    candidates.to_csv(out / "intervention_candidates.csv", index=False)

    portfolio_parts: list[pd.DataFrame] = []
    portfolio_summary: list[dict] = []
    for budget in budget_levels_usd:
        chosen = _select_portfolio(
            candidates,
            budget_usd=float(budget),
            min_equity_share=min_equity_share,
            max_actions_per_cell=2,
        )
        if chosen.empty:
            continue
        portfolio_parts.append(chosen)
        spent = float(chosen["cost_usd"].sum())
        prevented = float(chosen["prevented_calls_30d"].sum())
        equity_spend = float(chosen.loc[chosen["underserved_flag"] == True, "cost_usd"].sum())
        portfolio_summary.append(
            {
                "budget_usd": int(budget),
                "actions_selected": int(len(chosen)),
                "cells_touched": int(chosen["h3_cell"].nunique()),
                "spent_usd": spent,
                "budget_utilization_pct": (spent / max(float(budget), 1e-9)) * 100.0,
                "prevented_calls_30d": prevented,
                "avg_roi_calls_prevented_per_1k": float(chosen["roi_calls_prevented_per_1k"].mean()),
                "equity_spend_share_pct": (equity_spend / max(spent, 1e-9)) * 100.0,
            }
        )

    if portfolio_parts:
        pd.concat(portfolio_parts, ignore_index=True).to_csv(out / "intervention_portfolio.csv", index=False)
    else:
        pd.DataFrame().to_csv(out / "intervention_portfolio.csv", index=False)

    alerts = _early_warning_alerts(pred)
    alerts.to_csv(out / "early_warning_alerts.csv", index=False)

    alert_watch = alerts[alerts["alert_level"].isin(["critical", "elevated"])].copy()
    alert_watch["equity_need_score"] = (
        0.55 * _zscore(alert_watch["distance_to_nearest_station_km"])
        + 0.45 * _zscore(alert_watch["stress_index"])
    )
    equity_watch = alert_watch.sort_values("equity_need_score", ascending=False).head(60)
    equity_watch.to_csv(out / "equity_watchlist.csv", index=False)

    summary = {
        "optimizer_version": "v1_budgeted_multiaction",
        "budget_levels_usd": list(budget_levels_usd),
        "min_equity_share": float(min_equity_share),
        "portfolio_summary": portfolio_summary,
        "alerts_generated": int(len(alerts)),
        "critical_alerts": int((alerts["alert_level"] == "critical").sum()),
        "elevated_alerts": int((alerts["alert_level"] == "elevated").sum()),
    }
    with open(out / "operations_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
