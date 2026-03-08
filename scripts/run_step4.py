from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardian.correlation import run_bivariate_moran


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4: Bivariate Moran's I engine with lagged windows."
    )
    parser.add_argument(
        "--input-dir",
        default="data/processed",
        help="Directory containing Step 1 normalized CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for Step 4 output artifacts.",
    )
    parser.add_argument(
        "--h3-resolution",
        type=int,
        default=9,
        help="H3 resolution used for spatial aggregation.",
    )
    parser.add_argument(
        "--lag-days",
        type=int,
        default=30,
        help="Lag between predictor window and response window.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Window size for predictor/response event counts.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=499,
        help="Permutation count for p-value estimation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_bivariate_moran(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            h3_resolution=args.h3_resolution,
            lag_days=args.lag_days,
            window_days=args.window_days,
            permutations=args.permutations,
        )
        print("Step 4 complete.")
        print(f"- Summary: {(Path(args.output_dir) / 'bivariate_moran_summary.json').resolve()}")
        print(f"- Per-cell clusters: {(Path(args.output_dir) / 'bivariate_moran_cells.csv').resolve()}")
        print(f"- Response dataset used: {result.get('response_dataset_used', 'unknown')}")
        print(f"- Result: {result}")
    except Exception as exc:
        print(f"Step 4 failed: {exc}")
        print("Tip: make sure Step 1/2 outputs include both 911_calls.csv and code_violations.csv with date columns.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
