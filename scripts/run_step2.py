from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardian.features import Step2Config, build_grid_features
from guardian.modeling import train_risk_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: H3 feature engineering + Random Forest risk model."
    )
    parser.add_argument(
        "--input-dir",
        default="data/processed",
        help="Directory containing Step 1 normalized CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for Step 2 artifacts.",
    )
    parser.add_argument(
        "--h3-resolution",
        type=int,
        default=9,
        help="H3 resolution (default: 9, near neighborhood-scale).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        cfg = Step2Config(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            h3_resolution=args.h3_resolution,
        )
        grid = build_grid_features(cfg)
        if grid.empty:
            print("No feature grid created. Check that Step 1 outputs exist with latitude/longitude.")
            raise SystemExit(1)

        metrics = train_risk_model(grid, args.output_dir)
        print("Step 2 complete.")
        print(f"- Grid features: {(Path(args.output_dir) / 'grid_features.csv').resolve()}")
        print(f"- Predictions: {(Path(args.output_dir) / 'predictions.csv').resolve()}")
        print(f"- Top predictors: {(Path(args.output_dir) / 'top_predictors.csv').resolve()}")
        print(f"- Metrics: {(Path(args.output_dir) / 'model_metrics.json').resolve()}")
        print(f"- Model summary: {metrics}")
    except Exception as exc:
        print(f"Step 2 failed: {exc}")
        print("Tip: generate demo-safe inputs with `python scripts/bootstrap_sample_data.py` and rerun Step 2.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
