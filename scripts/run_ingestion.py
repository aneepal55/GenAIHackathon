from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardian.ingest import run_ingestion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Montgomery Guardian ingestion + normalization pipeline."
    )
    parser.add_argument(
        "--config",
        default="config/datasets.yaml",
        help="Path to dataset config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for normalized CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        outputs = run_ingestion(args.config, args.output_dir)
        if not outputs:
            print("No datasets were processed.")
            return
        print("Ingestion complete. Output files:")
        for name, path in outputs.items():
            print(f"- {name}: {Path(path).resolve()}")
    except Exception as exc:
        print(f"Step 1 failed: {exc}")
        print("Check dataset URLs in config/datasets.yaml or use sample data via scripts/bootstrap_sample_data.py.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
