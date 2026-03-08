from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardian.optimization import generate_operations_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 6: Generate operations optimizer outputs (portfolio + alerts + equity watchlist)."
    )
    parser.add_argument(
        "--input-dir",
        default="data/processed",
        help="Directory containing predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for optimization artifacts.",
    )
    parser.add_argument(
        "--min-equity-share",
        type=float,
        default=0.35,
        help="Minimum share of planned spend in underserved zones (default: 0.35).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = generate_operations_plan(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            min_equity_share=float(args.min_equity_share),
        )
        print("Step 6 complete.")
        print(f"- Operations summary: {(Path(args.output_dir) / 'operations_summary.json').resolve()}")
        print(f"- Portfolio: {(Path(args.output_dir) / 'intervention_portfolio.csv').resolve()}")
        print(f"- Alerts: {(Path(args.output_dir) / 'early_warning_alerts.csv').resolve()}")
        print(f"- Equity watchlist: {(Path(args.output_dir) / 'equity_watchlist.csv').resolve()}")
        print(f"- Summary: {summary}")
    except Exception as exc:
        print(f"Step 6 failed: {exc}")
        print("Tip: run Step 2 first to create predictions.csv.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
