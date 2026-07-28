#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a trained upper-PPO checkpoint across all controllable scenarios.

Loads a saved SB3 PPO model (deterministic actions) and runs it through every
scenario in run_zero_action_baseline.CONTROLLABLE_SCENARIOS, saving the same
per-step QoS / demand-before-after trace format as the zero-action and
upper+lower heuristic baselines so the three are directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from train_upper_ppo_3gnb import evaluate_upper_policy, make_env
from run_zero_action_baseline import CONTROLLABLE_SCENARIOS, build_args


TRAINING_EXPERT_BIAS_CSV = Path(
    "results/upper_heuristic_3gnb_baseline/upper_heuristic_3gnb_scenario_summary.csv"
)


def apply_reward_overrides(env_args, args: argparse.Namespace):
    if args.training_reward_config:
        env_args.paper_handover_penalty_weight = 0.0
        env_args.paper_pingpong_penalty_weight = 0.0
        env_args.expert_bias_reward_weight = 0.5
        env_args.expert_bias_csv = TRAINING_EXPERT_BIAS_CSV
        env_args.expert_bias_closeness_threshold = 0.95

    override_fields = (
        "paper_handover_penalty_weight",
        "paper_pingpong_penalty_weight",
        "expert_bias_reward_weight",
        "expert_bias_closeness_threshold",
        "contradictory_bias_penalty_weight",
        "idle_slice_bias_penalty_weight",
        "sinr_penalty_weight",
        "sinr_floor_db",
    )
    for field in override_fields:
        value = getattr(args, field)
        if value is not None:
            setattr(env_args, field, value)

    if args.expert_bias_csv is not None:
        env_args.expert_bias_csv = args.expert_bias_csv
    if args.no_expert_bias_csv:
        env_args.expert_bias_csv = None
    return env_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-scenario", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--training-reward-config",
        action="store_true",
        help=(
            "Use the same evaluation reward settings as the continuation notebook: "
            "no handover/ping-pong paper penalties, expert-bias CSV reward enabled."
        ),
    )
    parser.add_argument("--paper-handover-penalty-weight", type=float, default=None)
    parser.add_argument("--paper-pingpong-penalty-weight", type=float, default=None)
    parser.add_argument("--expert-bias-reward-weight", type=float, default=None)
    parser.add_argument("--expert-bias-csv", type=Path, default=None)
    parser.add_argument("--no-expert-bias-csv", action="store_true")
    parser.add_argument("--expert-bias-closeness-threshold", type=float, default=None)
    parser.add_argument("--contradictory-bias-penalty-weight", type=float, default=None)
    parser.add_argument("--idle-slice-bias-penalty-weight", type=float, default=None)
    parser.add_argument("--sinr-penalty-weight", type=float, default=None)
    parser.add_argument("--sinr-floor-db", type=float, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(args.model_path), device="cpu")
    summary_rows = []

    for scenario_name in CONTROLLABLE_SCENARIOS:
        env_args = build_args(scenario_name, args.seed)
        env_args = apply_reward_overrides(env_args, args)
        env = make_env(env_args)

        validation_csv = args.out_dir / f"{scenario_name}_trained_policy_trace.csv"
        stats = evaluate_upper_policy(
            model,
            env,
            n_eval_episodes=args.episodes_per_scenario,
            validation_csv=validation_csv,
        )
        env.close()

        stats["scenario_name"] = scenario_name
        summary_rows.append(stats)
        print(f"{scenario_name}: {json.dumps(stats, indent=2)}")

    summary_path = args.out_dir / "trained_policy_eval_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"\nSaved per-scenario traces + summary to {args.out_dir}")


if __name__ == "__main__":
    main()
