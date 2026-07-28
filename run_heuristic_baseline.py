#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Upper-heuristic + lower-heuristic baseline over all controllable scenarios.

Every step, the upper bias comes from compute_upper_expert_directional_bias
(a rule-based load/SLA/radio-margin-pressure controller -- the same "upper
expert" used elsewhere in this repo to generate lower-agent imitation
targets), and that bias is executed exactly as it would be for any other
policy: through strong_directional_heuristic_local_executor inside
GlobalPPO3GNBEnv.step(). So this is "upper heuristic + lower heuristic",
end to end -- no learned model involved. Saves the same per-step QoS /
demand-before-after trace format as run_zero_action_baseline.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_upper_ppo_3gnb import evaluate_upper_policy, make_env, GNB_IDS, NEIGHBORS
from global_ppo_3gnb_env import SLICE_TYPES
from local_a3_training_env import compute_upper_expert_directional_bias
from run_zero_action_baseline import CONTROLLABLE_SCENARIOS, build_args


class HeuristicUpperPolicy:
    """Stand-in for an SB3 model: bias comes from the rule-based upper expert."""

    def __init__(self, monitor_env):
        self.raw_env = monitor_env.unwrapped
        self.gnb_ids = GNB_IDS
        self.neighbors = NEIGHBORS
        self.slice_types = SLICE_TYPES
        self.action_dim = int(np.prod(monitor_env.action_space.shape))

    def predict(self, obs, deterministic: bool = True):
        bias = compute_upper_expert_directional_bias(
            self.raw_env.base_env,
            gnb_ids=self.gnb_ids,
            neighbor_graph=self.neighbors,
            slice_types=self.slice_types,
        )
        action = np.zeros(self.action_dim, dtype=np.float32)
        idx = 0
        for source_id in self.gnb_ids:
            for target_id in self.neighbors[source_id]:
                for slice_type in self.slice_types:
                    action[idx] = bias.get((source_id, target_id, slice_type), 0.0)
                    idx += 1
        return np.clip(action, -1.0, 1.0), None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/upper_lower_heuristic_baseline"))
    parser.add_argument("--episodes-per-scenario", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for scenario_name in CONTROLLABLE_SCENARIOS:
        env_args = build_args(scenario_name, args.seed)
        env = make_env(env_args)
        policy = HeuristicUpperPolicy(env)

        validation_csv = args.out_dir / f"{scenario_name}_heuristic_trace.csv"
        stats = evaluate_upper_policy(
            policy,
            env,
            n_eval_episodes=args.episodes_per_scenario,
            validation_csv=validation_csv,
        )
        env.close()

        stats["scenario_name"] = scenario_name
        summary_rows.append(stats)
        print(f"{scenario_name}: {json.dumps(stats, indent=2)}")

    summary_path = args.out_dir / "upper_lower_heuristic_baseline_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"\nSaved per-scenario traces + summary to {args.out_dir}")


if __name__ == "__main__":
    main()
