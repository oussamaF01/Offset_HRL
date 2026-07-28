#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero-action (default A3, no rebalancing) baseline over all controllable scenarios.

Runs the upper-agent environment with the bias action held at 0 for every
step -- i.e. the A3 offset never gets pushed away from its unmodified
default, so handovers only happen if the raw signal geometry crosses the
default A3 condition on its own. This is the "no intelligent layer" reference
point: per-scenario QoS and PRB demand before/after, with no policy applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from train_upper_ppo_3gnb import evaluate_upper_policy, make_env

CONTROLLABLE_SCENARIOS = (
    "jain_balance_controllable",
    "jain_control_urllc",
    "jain_control_mmtc",
    "jain_control_embb_urllc",
    "jain_control_embb_mmtc",
    "jain_control_urllc_mmtc",
    "jain_control_outer_congested",
    "jain_control_mixed",
)


class ZeroActionPolicy:
    """Stand-in for an SB3 model: always predicts the zero (neutral) action."""

    def __init__(self, action_dim: int):
        self.action_dim = int(action_dim)

    def predict(self, obs, deterministic: bool = True):
        return np.zeros(self.action_dim, dtype=np.float32), None


def build_args(scenario_name: str, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        seed=seed,
        n_gnbs=3,
        include_ue_counts=True,
        include_service_metrics=False,
        use_sumo_mobility=False,
        radio_substeps=100,
        radio_tick_seconds=0.001,
        pf_averaging_window_s=0.25,
        local_steps_per_global=10,
        global_steps_per_episode=12,
        scenario_mode="curriculum",
        snapshot_scenario="mixed",
        dense_window_reward=True,
        use_progress_reward=False,
        max_handovers_per_local_step=3,
        max_handovers_per_ue_episode=2,
        max_handovers_per_episode=20,
        handover_pingpong_guard_s=30.0,
        action_direction_reward_weight=0.0,
        snapshot_block_episodes=1,
        light_load_ues=1,
        medium_load_ues=2,
        high_load_ues=3,
        debug=False,
        slice_prb_budgets=None,
        max_prbs_per_ue=None,
        upper_reward_mode="paper_cost",
        load_balance_reward_weight=2.0,
        saturation_reward_weight=1.0,
        bias_smoothing_weight=0.01,
        negative_bias_penalty_weight=0.01,
        gnb_load_target=0.65,
        excess_load_reward_weight=1.0,
        served_share_reward_weight=1.0,
        served_active_floor_reward_weight=1.0,
        served_active_floor=0.20,
        jain_fairness_weight=1.0,
        paper_handover_penalty_weight=1.0,
        paper_pingpong_penalty_weight=5.0,
        paper_excess_load_penalty_weight=3.0,
        contradictory_bias_penalty_weight=1.0,
        idle_slice_bias_penalty_weight=1.0,
        expert_bias_csv=None,
        expert_bias_reward_weight=0.0,
        expert_bias_closeness_threshold=0.95,
        sla_deadband=0.05,
        upper_window_seconds=1.0,
        training_scenarios=scenario_name,
        scenario_selection="cycle",
        curriculum_block_episodes=1,
        fixed_stage_episodes=500,
        slow_stage_episodes=1000,
        global_neutral_bias_weight=0.1,
        neutral_bias_eps=0.05,
        wrong_bias_penalty_weight=0.05,
        global_bad_direction_eta=0.025,
        global_unsafe_target_rho=0.05,
        sla_severity_level_weight=0.1,
        load_balance_level_weight=1.0,
        a3_handover_cooldown_s=2.0,
        a3_min_residence_s=2.0,
        a3_history_window_s=20.0,
        a3_pingpong_threshold_s=5.0,
        safe_admission=True,
        warmup_steps=2,
        post_handover_settle_steps=4,
        demand_calibration_alpha=0.5,
        dynamic_upper_window=True,
        max_dynamic_local_steps_per_global=40,
        center_gap_topology="medium_270m",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/zero_action_baseline"))
    parser.add_argument("--episodes-per-scenario", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for scenario_name in CONTROLLABLE_SCENARIOS:
        env_args = build_args(scenario_name, args.seed)
        env = make_env(env_args)
        action_dim = int(np.prod(env.action_space.shape))
        policy = ZeroActionPolicy(action_dim)

        validation_csv = args.out_dir / f"{scenario_name}_zero_action_trace.csv"
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

    summary_path = args.out_dir / "zero_action_baseline_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"\nSaved per-scenario traces + summary to {args.out_dir}")


if __name__ == "__main__":
    main()
