from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


DEFAULT_FILES = [
    "predictions.csv",
    "model_metrics.json",
    "model_governance.json",
    "operations_summary.json",
    "intervention_candidates.csv",
    "intervention_portfolio.csv",
    "early_warning_alerts.csv",
    "equity_watchlist.csv",
    "top_predictors.csv",
    "bivariate_moran_summary.json",
    "pitch_summary.md",
    # Label/context datasets used by dashboard
    "point_of_interest.csv",
    "most_visited_locations.csv",
    "community_centers.csv",
    "parks_and_trail.csv",
    "education_facility.csv",
    "pharmacy_locator.csv",
    "business_license.csv",
    "received_311_service_requests.csv",
]


def prepare_served_data(source_dir: str | Path, served_dir: str | Path) -> dict:
    source = Path(source_dir)
    target = Path(served_dir)
    target.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []

    for filename in DEFAULT_FILES:
        src = source / filename
        dst = target / filename
        if not src.exists():
            missing.append(filename)
            continue
        shutil.copy2(src, dst)
        copied.append(filename)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source.resolve()),
        "served_dir": str(target.resolve()),
        "copied_files": copied,
        "missing_files": missing,
    }
    with open(target / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deploy-ready served snapshot for dashboard.")
    parser.add_argument("--source-dir", default="data/processed", help="Source artifacts directory.")
    parser.add_argument("--served-dir", default="data/served", help="Destination served snapshot directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_served_data(args.source_dir, args.served_dir)
    print("Served snapshot prepared.")
    print(f"- Source: {result['source_dir']}")
    print(f"- Target: {result['served_dir']}")
    print(f"- Copied: {len(result['copied_files'])}")
    if result["missing_files"]:
        print(f"- Missing (not copied): {', '.join(result['missing_files'])}")
    print(f"- Metadata: {(Path(args.served_dir) / 'metadata.json').resolve()}")


if __name__ == "__main__":
    main()
