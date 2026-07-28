#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from global_ppo_3gnb_env import GlobalPPO3GNBEnv, SLICE_TYPES


NEIGHBORS = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify that upper expert bias columns are loaded correctly from the scenario CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/upper_heuristic_3gnb_baseline/upper_heuristic_3gnb_scenario_summary.csv"),
    )
    return parser.parse_args()


def expected_bias_columns():
    for source_gnb in range(3):
        for target_gnb in NEIGHBORS[source_gnb]:
            for slice_type in SLICE_TYPES:
                yield source_gnb, target_gnb, slice_type, (
                    f"first_upper_bias_g{source_gnb}_to_g{target_gnb}_{slice_type}"
                )


def main():
    args = parse_args()
    frame = pd.read_csv(args.csv)
    if frame.empty:
        raise SystemExit(f"No rows found in {args.csv}")

    json_mismatches = []
    for _, row in frame.iterrows():
        scenario_name = str(row["scenario_name"])
        bias_json = json.loads(row["first_upper_bias_json"])
        for source_gnb, target_gnb, slice_type, col in expected_bias_columns():
            key = f"{source_gnb}|{target_gnb}|{slice_type}"
            col_value = float(row[col])
            json_value = float(bias_json[key])
            if not np.isclose(col_value, json_value, atol=1e-12):
                json_mismatches.append((scenario_name, col, col_value, key, json_value))

    env = GlobalPPO3GNBEnv(
        training_scenarios="all",
        scenario_selection="cycle",
        expert_bias_csv=args.csv,
        expert_bias_reward_weight=0.05,
        local_steps_per_global=2,
        radio_substeps=1,
        post_handover_settle_steps=1,
    )
    env_mismatches = []
    try:
        for _, row in frame.iterrows():
            scenario_name = str(row["scenario_name"])
            tensor = env._expert_bias_by_scenario.get(scenario_name)
            if tensor is None:
                env_mismatches.append((scenario_name, "missing_env_tensor"))
                continue
            for source_gnb, target_gnb, slice_type, col in expected_bias_columns():
                slot = NEIGHBORS[source_gnb].index(target_gnb)
                slice_idx = SLICE_TYPES.index(slice_type)
                col_value = float(row[col])
                tensor_value = float(tensor[source_gnb, slot, slice_idx])
                if not np.isclose(col_value, tensor_value, atol=1e-7):
                    env_mismatches.append((scenario_name, col, col_value, tensor_value))
    finally:
        env.close()

    print(f"csv={args.csv}")
    print(f"rows={len(frame)}")
    print(f"column_vs_json_mismatches={len(json_mismatches)}")
    print(f"env_vs_csv_mismatches={len(env_mismatches)}")
    print("scenarios=" + ",".join(frame["scenario_name"].astype(str).tolist()))

    if json_mismatches or env_mismatches:
        for item in json_mismatches[:10]:
            print("JSON_MISMATCH", item)
        for item in env_mismatches[:10]:
            print("ENV_MISMATCH", item)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
