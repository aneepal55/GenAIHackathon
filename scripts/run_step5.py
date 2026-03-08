from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardian.reporting import write_pitch_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 5: Generate pitch-ready narrative brief from artifacts."
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing predictions/model/correlation outputs.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/pitch_summary.md",
        help="Markdown file path for generated narrative brief.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        out = write_pitch_summary(args.data_dir, args.output)
        print("Step 5 complete.")
        print(f"- Pitch brief: {Path(out).resolve()}")
    except Exception as exc:
        print(f"Step 5 failed: {exc}")
        print("Tip: run Step 2 and Step 4 first so the brief has full metrics.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
