#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full SUMO demo workflow.")
    parser.add_argument("--num-vehicles", type=int, default=10)
    parser.add_argument("--num-persons", type=int, default=5)
    args = parser.parse_args()

    if args.num_vehicles < 0:
        parser.error("--num-vehicles must be >= 0")
    if args.num_persons < 0:
        parser.error("--num-persons must be >= 0")

    return args


def main() -> None:
    args = parse_args()

    subprocess.run(
        [
            sys.executable,
            "generate_sumo_scenario.py",
            "--num-vehicles",
            str(args.num_vehicles),
            "--num-persons",
            str(args.num_persons),
        ],
        check=True,
    )

    subprocess.run([sys.executable, "test_sumo_consistency.py"], check=True)
    subprocess.run([sys.executable, "plot_sumo_consistency.py"], check=True)

    consistency_csv = Path("scenario/mobility/sumo_consistency_results.csv")
    plots_dir = Path("scenario/mobility/plots")

    print("\nSUMO Demo Summary")
    print(f"- scenario generated: scenario/mobility/")
    print(f"- consistency CSV path: {consistency_csv}")
    print(f"- plots directory path: {plots_dir}")
    print("- reminder: max/mean error are printed by the consistency test")


if __name__ == "__main__":
    main()
