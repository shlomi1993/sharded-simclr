"""
Compare the recorded throughput of a baseline run and a follow-up run.

Typical usage after two training runs:
    python scripts/compare_runs.py output/baseline output/followup
"""

import argparse
import json

from pathlib import Path

CONFIG_FILENAME = "run_config.json"


def load_images_per_sec(run_dir: Path) -> float:
    """
    Read the recorded global source-image throughput of one run.

    Args:
        run_dir (Path): Run directory containing run_config.json.

    Returns:
        float: Source images per second.
    """
    with open(run_dir / CONFIG_FILENAME, encoding="utf-8") as f:
        return json.load(f)["images_per_sec"]


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for run comparison.

    Returns:
        argparse.ArgumentParser: Configured CLI parser.
    """
    parser = argparse.ArgumentParser(description="Print baseline and follow-up throughput and their ratio.")
    parser.add_argument("baseline", type=Path, help="Baseline run directory.")
    parser.add_argument("followup", type=Path, help="Follow-up run directory.")
    return parser


def main() -> None:
    """
    Entry point: load both runs and print their throughput and ratio.
    """
    args = build_parser().parse_args()
    baseline = load_images_per_sec(args.baseline)
    followup = load_images_per_sec(args.followup)

    print(f"Baseline: {baseline:.2f} source images/s")
    print(f"Follow-up: {followup:.2f} source images/s")
    if baseline > 0:
        print(f"Follow-up / baseline: {followup / baseline:.3f}x (single-run observation)")


if __name__ == "__main__":
    main()
