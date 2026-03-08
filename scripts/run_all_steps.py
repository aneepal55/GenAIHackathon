from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from guardian.correlation import run_bivariate_moran
from guardian.features import Step2Config, build_grid_features
from guardian.ingest import run_ingestion
from guardian.modeling import train_risk_model
from guardian.reporting import write_pitch_summary
from bootstrap_sample_data import write_sample_data
from prepare_served_data import prepare_served_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all Montgomery Guardian steps end-to-end.")
    parser.add_argument("--config", default="config/datasets.yaml", help="Ingestion config file.")
    parser.add_argument("--data-dir", default="data/processed", help="Working output directory.")
    parser.add_argument("--h3-resolution", type=int, default=9, help="H3 resolution.")
    parser.add_argument("--lag-days", type=int, default=30, help="Lag for bivariate Moran.")
    parser.add_argument("--window-days", type=int, default=30, help="Response window for Moran.")
    parser.add_argument("--permutations", type=int, default=499, help="Permutation count.")
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Skip API ingestion and generate synthetic sample data for demos.",
    )
    parser.add_argument(
        "--publish-served",
        action="store_true",
        help="Copy deploy-ready snapshot files to data/served after pipeline run.",
    )
    parser.add_argument(
        "--served-dir",
        default="data/served",
        help="Output directory for served snapshot (used with --publish-served).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.use_sample_data:
            print("Step 1: Generating synthetic sample data...")
            write_sample_data(data_dir)
        else:
            print("Step 1: Running ingestion...")
            run_ingestion(args.config, data_dir)

        print("Step 2: Building features + training model...")
        grid = build_grid_features(
            Step2Config(input_dir=data_dir, output_dir=data_dir, h3_resolution=args.h3_resolution)
        )
        if grid.empty:
            raise RuntimeError("Feature grid is empty. Ensure data CSVs include latitude/longitude.")
        train_risk_model(grid, data_dir)

        print("Step 4: Running bivariate Moran correlation...")
        run_bivariate_moran(
            input_dir=data_dir,
            output_dir=data_dir,
            h3_resolution=args.h3_resolution,
            lag_days=args.lag_days,
            window_days=args.window_days,
            permutations=args.permutations,
        )

        print("Step 5: Generating pitch brief...")
        write_pitch_summary(data_dir, data_dir / "pitch_summary.md")

        if args.publish_served:
            print("Step 6: Preparing served snapshot...")
            prepare_served_data(data_dir, args.served_dir)

        print("All steps complete.")
        print(f"- Predictions: {(data_dir / 'predictions.csv').resolve()}")
        print(f"- Moran summary: {(data_dir / 'bivariate_moran_summary.json').resolve()}")
        print(f"- Pitch brief: {(data_dir / 'pitch_summary.md').resolve()}")
        if args.publish_served:
            print(f"- Served snapshot: {(Path(args.served_dir)).resolve()}")
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        print("Tip: rerun with --use-sample-data for a demo-safe local run.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
