from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_pitch_summary(data_dir: str | Path) -> str:
    data_path = Path(data_dir)
    pred = _read_csv(data_path / "predictions.csv")
    top_pred = _read_csv(data_path / "top_predictors.csv")
    moran = _read_json(data_path / "bivariate_moran_summary.json")
    metrics = _read_json(data_path / "model_metrics.json")
    ops = _read_json(data_path / "operations_summary.json")

    total_pred_30d = float(pred["predicted_calls_next_30d"].sum()) if "predicted_calls_next_30d" in pred.columns else 0.0
    if "predicted_calls_next_30d" in pred.columns and len(pred) > 0:
        p90 = float(pred["predicted_calls_next_30d"].quantile(0.9))
        high_risk_cells = int((pred["predicted_calls_next_30d"] >= p90).sum())
    else:
        high_risk_cells = 0

    top_feature = "n/a"
    if not top_pred.empty and "feature" in top_pred.columns:
        top_feature = str(top_pred.iloc[0]["feature"])

    moran_i = moran.get("bivariate_moran_i")
    moran_p = moran.get("p_value_two_sided")
    r2 = metrics.get("r2")
    mae = metrics.get("mae")
    target_source = metrics.get("target_source", "unknown")
    portfolio = ops.get("portfolio_summary", []) if isinstance(ops, dict) else []
    ops_line = "n/a"
    if portfolio:
        best = sorted(portfolio, key=lambda x: float(x.get("prevented_calls_30d", 0)), reverse=True)[0]
        ops_line = (
            f"Budget ${int(best.get('budget_usd', 0)):,}: prevents about "
            f"{float(best.get('prevented_calls_30d', 0)):.1f} requests/30d across "
            f"{int(best.get('cells_touched', 0))} cells"
        )

    lines = [
        "# The Montgomery Guardian - Pitch Brief",
        "",
        "## Executive Signal",
        f"- Predicted service demand next 30 days: **{total_pred_30d:.1f} events** across monitored H3 cells.",
        f"- High-risk pockets (top decile): **{high_risk_cells} cells**.",
        f"- Strongest model driver: **{top_feature}**.",
        f"- Predictive target source: **{target_source}**.",
        "",
        "## Broken Windows Evidence",
        f"- Bivariate Moran's I (lagged): **{moran_i if moran_i is not None else 'n/a'}**.",
        f"- Permutation p-value: **{moran_p if moran_p is not None else 'n/a'}**.",
        "",
        "## Predictive Reliability",
        f"- Model R2: **{r2 if r2 is not None else 'n/a'}**.",
        f"- Model MAE: **{mae if mae is not None else 'n/a'}** events per H3 cell.",
        "",
        "## Policy Action",
        "- Resource Optimization: Prioritize code enforcement sweeps in top-decile vulnerability pockets before dispatch demand peaks.",
        "- Equity Lens: Pair risk scores with response-distance to target underserved zones first.",
        "- Proactive Governance: Use weekly updates from this pipeline to shift from reactive dispatch to prevention-led operations.",
        f"- Optimizer Scenario: **{ops_line}**.",
    ]
    return "\n".join(lines)


def write_pitch_summary(data_dir: str | Path, output_path: str | Path) -> Path:
    content = build_pitch_summary(data_dir)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    return target
