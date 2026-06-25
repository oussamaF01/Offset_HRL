#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from global_ppo_3gnb_env import GlobalPPO3GNBEnv, SLICE_TYPES
from upper_agent_training_scenarios import (
    CENTER_GAP_GNB_CONFIGS,
    get_upper_training_scenarios,
)


GNB_IDS = (0, 1, 2)
MAX_NEIGHBORS = 2
NEIGHBORS = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
ACTION_FIELDS = [
    f"action_g{source}_to_g{target}_{slice_type}"
    for source in GNB_IDS
    for target in NEIGHBORS[source]
    for slice_type in SLICE_TYPES
]
OBSERVATION_FIELDS = [
    f"obs_{idx}"
    for idx in range(
        len(GNB_IDS) * len(SLICE_TYPES) * 3 + len(ACTION_FIELDS)
    )
]
DIRECTIONAL_BIAS_FIELDS = [
    f"bias_g{source}_to_g{target}_{slice_type}"
    for source in GNB_IDS
    for target in NEIGHBORS[source]
    for slice_type in SLICE_TYPES
]
MATRIX_FIELDS = [
    f"{prefix}_g{gnb_id}_{slice_type}"
    for prefix in (
        "used_prb_start", "used_prb_end",
        "sla", "ue_count",
    )
    for gnb_id in GNB_IDS
    for slice_type in SLICE_TYPES
]
USED_PRB_TOTAL_FIELDS = (
    [f"network_used_prb_{phase}" for phase in ("start", "end")]
    + [f"mean_gnb_used_prb_{phase}" for phase in ("start", "end")]
    + [f"max_gnb_used_prb_{phase}" for phase in ("start", "end")]
    + [
        f"gnb_used_prb_{phase}_g{gnb_id}"
        for phase in ("start", "end")
        for gnb_id in GNB_IDS
    ]
    + [
        f"slice_used_prb_{phase}_{slice_type}"
        for phase in ("start", "end")
        for slice_type in SLICE_TYPES
    ]
)
SERVED_FLOOR_REFERENCE_FIELDS = [
    f"served_active_floor_reference_g{gnb_id}" for gnb_id in GNB_IDS
]

QOS_SCALAR_FIELDS = [
    "network_throughput_mbps",
    "network_offered_mbps",
    "network_delivery_ratio",
    "network_completed_delay_ms",
    "network_mean_hol_delay_ms",
    "network_max_hol_delay_ms",
    "network_queue_kbits",
    "network_drop_ratio",
    "network_packet_failure_ratio",
]

QOS_MATRIX_KEYS = [
    "throughput_mbps_matrix",
    "offered_mbps_matrix",
    "delivery_ratio_matrix",
    "completed_delay_ms_matrix",
    "mean_hol_delay_ms_matrix",
    "max_hol_delay_ms_matrix",
    "queue_kbits_matrix",
    "drop_ratio_matrix",
    "packet_failure_ratio_matrix",
]

TRAINING_FIELDS = [
    "step",
    "ppo_update_index",
    "rollout_step_in_update",
    "policy_has_updated",
    "episode",
    "episode_step",
    "reward",
    "episode_return",
    "done",
    "scenario_name",
    "episode_time_s",
    "episode_duration_s",
    "upper_window_seconds",
    "local_step_seconds",
    "radio_service_seconds_per_upper_window",
    "post_handover_settle_steps",
    "radio_measurement_steps",
    "prb_measurement_mode",
    *USED_PRB_TOTAL_FIELDS,
    "overload_ratio",
    "sla_count",
    "sla_severity",
    "handover_count",
    "used_prb_balance_cost_start",
    "used_prb_balance_cost_end",
    "global_cost_start",
    "global_cost_end",
    "global_cost_improvement",
    "global_action_penalty",
    "global_negative_bias_penalty",
    "reward_used_prb_balance_improvement",
    "reward_used_prb_balance_improvement_raw",
    "reward_active_slice_count",
    "reward_saturation_improvement",
    "reward_excess_load_improvement",
    "reward_excess_load_improvement_raw",
    "reward_served_share_improvement",
    "reward_served_share_improvement_raw",
    "served_share_cost_start",
    "served_share_cost_end",
    "reward_served_active_floor",
    "reward_served_active_floor_raw",
    "served_active_floor_cost_start",
    "served_active_floor_cost_end",
    "served_active_floor",
    "reward_jain_fairness",
    "jain_fairness_raw",
    "jain_fairness_normalized",
    "gnb_excess_load_cost_start",
    "gnb_excess_load_cost_end",
    "gnb_load_target_requested",
    "gnb_load_target_effective",
    "gnb_load_target_feasible",
    "persistent_demand_utilization",
    "reward_sla_improvement",
    "saturation_count",
    "overloaded_negative_fraction",
    "light_nonnegative_fraction",
    "directional_offset_tensor",
    "safe_admission_capacities",
    "safe_admission_accepted",
    "safe_admission_remaining",
    "safe_admission_source_capacities",
    "safe_admission_source_accepted",
    "safe_admission_stats",
    *QOS_SCALAR_FIELDS,
    *SERVED_FLOOR_REFERENCE_FIELDS,
] + DIRECTIONAL_BIAS_FIELDS + MATRIX_FIELDS

VALIDATION_FIELDS = [
    field for field in TRAINING_FIELDS
    if field not in {"episode_return", "done"}
]


def _json_array(value) -> str:
    arr = np.asarray(value)
    return json.dumps(arr.tolist())


def _flat_first(value, expected_size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.full(expected_size, np.nan, dtype=float)
    arr = arr.reshape(-1)
    if arr.size < expected_size:
        padded = np.full(expected_size, np.nan, dtype=float)
        padded[:arr.size] = arr
        return padded
    return arr[:expected_size]


def _matrix_or_nan(value) -> np.ndarray:
    expected_shape = (len(GNB_IDS), len(SLICE_TYPES))
    arr = np.asarray(value, dtype=float)
    if arr.shape != expected_shape:
        return np.full(expected_shape, np.nan, dtype=float)
    return arr


def _bias_quality_scores(bias_matrix, load_matrix) -> tuple[float, float]:
    bias = np.asarray(bias_matrix, dtype=float)
    if bias.shape == (len(GNB_IDS), len(NEIGHBORS[0]), len(SLICE_TYPES)):
        bias = np.min(bias, axis=1)
    else:
        bias = _matrix_or_nan(bias)
    loads = _matrix_or_nan(load_matrix)
    if np.isnan(bias).any() or np.isnan(loads).any():
        return 0.0, 0.0
    overloaded = loads > 0.85
    light = loads < 0.45
    overloaded_negative = float(np.mean(bias[overloaded] < 0.0)) if overloaded.any() else 1.0
    light_nonnegative = float(np.mean(bias[light] >= -0.1)) if light.any() else 1.0
    return overloaded_negative, light_nonnegative


def _add_flat_matrix_fields(row: Dict[str, Any], prefix: str, value) -> None:
    matrix = _matrix_or_nan(value)
    for g_idx, gnb_id in enumerate(GNB_IDS):
        for s_idx, slice_type in enumerate(SLICE_TYPES):
            row[f"{prefix}_g{gnb_id}_{slice_type}"] = float(matrix[g_idx, s_idx])


def _add_directional_bias_fields(row: Dict[str, Any], value) -> None:
    tensor = np.asarray(value, dtype=float)
    expected = (len(GNB_IDS), len(NEIGHBORS[0]), len(SLICE_TYPES))
    if tensor.shape != expected:
        tensor = np.full(expected, np.nan, dtype=float)
    for source in GNB_IDS:
        for slot, target in enumerate(NEIGHBORS[source]):
            for s_idx, slice_type in enumerate(SLICE_TYPES):
                row[f"bias_g{source}_to_g{target}_{slice_type}"] = float(
                    tensor[source, slot, s_idx]
                )


def _add_qos_fields(row: Dict[str, Any], info: Dict[str, Any]) -> None:
    qos = dict(info.get("qos", {}))
    for field in QOS_SCALAR_FIELDS:
        row[field] = float(qos.get(field, 0.0))


def _matrix_sum(value) -> float:
    matrix = _matrix_or_nan(value)
    return float(np.nansum(matrix))


def _row_sum_mean(value) -> float:
    matrix = _matrix_or_nan(value)
    row_sums = np.nansum(matrix, axis=1)
    return float(np.nanmean(row_sums))


def _row_sum_max(value) -> float:
    matrix = _matrix_or_nan(value)
    row_sums = np.nansum(matrix, axis=1)
    return float(np.nanmax(row_sums))


def _add_matrix_total_fields(row: Dict[str, Any], prefix: str, value) -> None:
    matrix = _matrix_or_nan(value)
    for g_idx, gnb_id in enumerate(GNB_IDS):
        row[f"gnb_{prefix}_g{gnb_id}"] = float(np.nansum(matrix[g_idx, :]))
    for s_idx, slice_type in enumerate(SLICE_TYPES):
        row[f"slice_{prefix}_{slice_type}"] = float(np.nansum(matrix[:, s_idx]))


def _add_served_floor_reference_fields(row: Dict[str, Any], value) -> None:
    values = _flat_first(value, len(GNB_IDS))
    for idx, gnb_id in enumerate(GNB_IDS):
        row[f"served_active_floor_reference_g{gnb_id}"] = float(values[idx])


class UpperTrainingCsvCallback(BaseCallback):
    def __init__(
        self,
        log_path: Path,
        best_path: Path | None = None,
        log_every: int = 1,
        flush_every: int = 100,
    ):
        super().__init__()
        self.log_path = Path(log_path)
        self.best_path = Path(best_path) if best_path is not None else None
        self.log_every = max(int(log_every), 1)
        self.flush_every = max(int(flush_every), 1)
        self.file = None
        self.writer = None
        self.rows_since_flush = 0
        self.episode = 0
        self.episode_step = 0
        self.episode_return = 0.0
        self.best_episode_return = -np.inf

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.log_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=TRAINING_FIELDS,
            extrasaction="ignore",
        )
        self.writer.writeheader()

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", [0.0])).reshape(-1)
        dones = np.asarray(self.locals.get("dones", [False])).reshape(-1)
        reward = float(rewards[0])
        done = bool(dones[0])

        self.episode_step += 1
        self.episode_return += reward
        should_log = done or self.num_timesteps == 1 or self.num_timesteps % self.log_every == 0
        if not should_log:
            return True
        infos = self.locals.get("infos", [{}])
        info = dict(infos[0])
        ppo_n_steps = max(int(getattr(self.model, "n_steps", 1)), 1)
        ppo_update_index = max((int(self.num_timesteps) - 1) // ppo_n_steps, 0)
        rollout_step_in_update = ((int(self.num_timesteps) - 1) % ppo_n_steps) + 1
        used_prb_start = info.get("used_prb_matrix_start", [])
        used_prb_end = info.get("used_prb_matrix_end", [])
        overloaded_negative, light_nonnegative = _bias_quality_scores(
            info.get("directional_bias_tensor", []),
            info.get("load_matrix", []),
        )
        row = {
            "step": int(self.num_timesteps),
            "ppo_update_index": int(ppo_update_index),
            "rollout_step_in_update": int(rollout_step_in_update),
            "policy_has_updated": bool(ppo_update_index > 0),
            "episode": int(self.episode),
            "episode_step": int(self.episode_step),
            "reward": reward,
            "episode_return": float(self.episode_return),
            "done": done,
            "scenario_name": str(info.get("scenario_name", "")),
            "episode_time_s": float(info.get("episode_time_s", 0.0)),
            "episode_duration_s": float(info.get("episode_duration_s", 0.0)),
            "upper_window_seconds": float(info.get("upper_window_seconds", 0.0)),
            "local_step_seconds": float(info.get("local_step_seconds", 0.0)),
            "radio_service_seconds_per_upper_window": float(
                info.get("radio_service_seconds_per_upper_window", 0.0)
            ),
            "post_handover_settle_steps": int(
                info.get("post_handover_settle_steps", 0)
            ),
            "radio_measurement_steps": int(
                info.get("radio_measurement_steps", 0)
            ),
            "prb_measurement_mode": str(
                info.get("load_measurement_mode", "")
            ),
            "network_used_prb_start": _matrix_sum(used_prb_start),
            "network_used_prb_end": _matrix_sum(used_prb_end),
            "mean_gnb_used_prb_start": _row_sum_mean(used_prb_start),
            "mean_gnb_used_prb_end": _row_sum_mean(used_prb_end),
            "max_gnb_used_prb_start": _row_sum_max(used_prb_start),
            "max_gnb_used_prb_end": _row_sum_max(used_prb_end),
            "overload_ratio": float(info.get("overload_ratio", 0.0)),
            "sla_count": float(info.get("sla_count", 0.0)),
            "sla_severity": float(info.get("sla_severity", 0.0)),
            "handover_count": int(info.get("handover_count", 0)),
            "used_prb_balance_cost_start": float(
                info.get("used_prb_balance_cost_start", 0.0)
            ),
            "used_prb_balance_cost_end": float(
                info.get("used_prb_balance_cost_end", 0.0)
            ),
            "global_cost_start": float(info.get("global_cost_start", 0.0)),
            "global_cost_end": float(info.get("global_cost_end", 0.0)),
            "global_cost_improvement": float(info.get("global_cost_improvement", 0.0)),
            "global_action_penalty": float(info.get("global_action_penalty", 0.0)),
            "global_negative_bias_penalty": float(
                info.get("global_negative_bias_penalty", 0.0)
            ),
            "reward_used_prb_balance_improvement": float(
                info.get("reward_used_prb_balance_improvement", 0.0)
            ),
            "reward_used_prb_balance_improvement_raw": float(
                info.get("reward_used_prb_balance_improvement_raw", 0.0)
            ),
            "reward_active_slice_count": int(
                info.get("reward_active_slice_count", 0)
            ),
            "reward_saturation_improvement": float(
                info.get("reward_saturation_improvement", 0.0)
            ),
            "reward_excess_load_improvement": float(
                info.get("reward_excess_load_improvement", 0.0)
            ),
            "reward_excess_load_improvement_raw": float(
                info.get("reward_excess_load_improvement_raw", 0.0)
            ),
            "reward_served_share_improvement": float(
                info.get("reward_served_share_improvement", 0.0)
            ),
            "reward_served_share_improvement_raw": float(
                info.get("reward_served_share_improvement_raw", 0.0)
            ),
            "served_share_cost_start": float(
                info.get("served_share_cost_start", 0.0)
            ),
            "served_share_cost_end": float(
                info.get("served_share_cost_end", 0.0)
            ),
            "reward_served_active_floor": float(
                info.get("reward_served_active_floor", 0.0)
            ),
            "reward_served_active_floor_raw": float(
                info.get("reward_served_active_floor_raw", 0.0)
            ),
            "served_active_floor_cost_start": float(
                info.get("served_active_floor_cost_start", 0.0)
            ),
            "served_active_floor_cost_end": float(
                info.get("served_active_floor_cost_end", 0.0)
            ),
            "served_active_floor": float(
                info.get("served_active_floor", 0.0)
            ),
            "reward_jain_fairness": float(
                info.get("reward_jain_fairness", 0.0)
            ),
            "jain_fairness_raw": float(
                info.get("jain_fairness_raw", 0.0)
            ),
            "jain_fairness_normalized": float(
                info.get("jain_fairness_normalized", 0.0)
            ),
            "gnb_excess_load_cost_start": float(
                info.get("gnb_excess_load_cost_start", 0.0)
            ),
            "gnb_excess_load_cost_end": float(
                info.get("gnb_excess_load_cost_end", 0.0)
            ),
            "gnb_load_target_requested": float(
                info.get("gnb_load_target_requested", 0.65)
            ),
            "gnb_load_target_effective": float(
                info.get("gnb_load_target_effective", 0.65)
            ),
            "gnb_load_target_feasible": bool(
                info.get("gnb_load_target_feasible", True)
            ),
            "persistent_demand_utilization": float(
                info.get("persistent_demand_utilization", 0.0)
            ),
            "reward_sla_improvement": float(info.get("reward_sla_improvement", 0.0)),
            "saturation_count": int(info.get("saturation_count", 0)),
            "overloaded_negative_fraction": overloaded_negative,
            "light_nonnegative_fraction": light_nonnegative,
            "directional_offset_tensor": _json_array(info.get("directional_offset_tensor", [])),
            "safe_admission_capacities": json.dumps(
                {
                    ":".join(map(str, key)): value
                    for key, value in info.get("safe_admission", {})
                    .get("capacities", {})
                    .items()
                }
            ),
            "safe_admission_accepted": json.dumps(
                {
                    ":".join(map(str, key)): value
                    for key, value in info.get("safe_admission", {})
                    .get("accepted", {})
                    .items()
                }
            ),
            "safe_admission_remaining": json.dumps(
                {
                    ":".join(map(str, key)): value
                    for key, value in info.get("safe_admission", {})
                    .get("remaining", {})
                    .items()
                }
            ),
            "safe_admission_source_capacities": json.dumps(
                {
                    ":".join(map(str, key)): value
                    for key, value in info.get("safe_admission", {})
                    .get("source_capacities", {})
                    .items()
                }
            ),
            "safe_admission_source_accepted": json.dumps(
                {
                    ":".join(map(str, key)): value
                    for key, value in info.get("safe_admission", {})
                    .get("source_accepted", {})
                    .items()
                }
            ),
            "safe_admission_stats": json.dumps(
                info.get("safe_admission", {}).get("stats", {})
            ),
        }
        _add_qos_fields(row, info)
        _add_matrix_total_fields(row, "used_prb_start", used_prb_start)
        _add_matrix_total_fields(row, "used_prb_end", used_prb_end)
        _add_served_floor_reference_fields(
            row,
            info.get("served_active_floor_reference_gnb_loads", []),
        )
        _add_directional_bias_fields(
            row, info.get("directional_bias_tensor", [])
        )
        for prefix, key in (
            ("target_load", "target_load_matrix"),
            ("balance_target", "balance_target_matrix"),
            ("used_prb_start", "used_prb_matrix_start"),
            ("used_prb_end", "used_prb_matrix_end"),
            ("sla", "sla_matrix"),
            ("ue_count", "ue_count_matrix"),
        ):
            _add_flat_matrix_fields(row, prefix, info.get(key, []))
        self.writer.writerow(row)
        self.rows_since_flush += 1
        if done or self.rows_since_flush >= self.flush_every:
            self.file.flush()
            self.rows_since_flush = 0

        if done:
            if self.best_path is not None and self.episode_return > self.best_episode_return:
                self.best_episode_return = float(self.episode_return)
                self.model.save(self.best_path)
            self.episode += 1
            self.episode_step = 0
            self.episode_return = 0.0
        return True

    def _on_training_end(self) -> None:
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None


def resolve_upper_training_curriculum_args(args):
    """Keep upper PPO training on one coherent scenario unless requested.

    The upper observation includes previous directional bias and the reward has
    an action-smoothness penalty versus that previous bias. Mixing unrelated
    scenarios by default makes the rollout distribution noisy before the policy
    has learned one stable control problem, so the default is a single retained
    scenario. Use --curriculum-training to intentionally train over a pool.
    """
    if bool(getattr(args, "block_curriculum_training", False)):
        default_pool = (
            "high_load_inner_embb,"
            "high_load_inner_mixed,"
            "high_load_inner_asymmetric"
        )
        if str(getattr(args, "training_scenarios", "")).strip() == str(
            getattr(args, "single_training_scenario", "")
        ).strip():
            args.training_scenarios = default_pool
        args.curriculum_training = True
        args.scenario_selection = "block"
        args.curriculum_block_episodes = resolve_curriculum_block_episodes(args)
        return args

    if bool(getattr(args, "curriculum_training", False)):
        if not str(getattr(args, "training_scenarios", "")).strip():
            args.training_scenarios = (
                "high_load_inner_embb,"
                "high_load_inner_mixed,"
                "high_load_inner_asymmetric"
            )
        return args

    args.training_scenarios = str(
        getattr(args, "single_training_scenario", "high_load_inner_asymmetric")
    )
    args.scenario_selection = "cycle"
    return args


def resolve_curriculum_block_episodes(args) -> int:
    requested = int(getattr(args, "curriculum_block_episodes", 0))
    if requested > 0:
        return requested

    scenarios = get_upper_training_scenarios(getattr(args, "training_scenarios", None))
    upper_window_seconds = max(float(getattr(args, "upper_window_seconds", 1.0)), 1e-6)
    episode_steps = max(
        1,
        int(np.ceil(max(float(s.duration_s) for s in scenarios) / upper_window_seconds)),
    )
    ppo_updates = max(int(getattr(args, "ppo_updates_per_scenario", 3)), 1)
    ppo_n_steps = max(int(getattr(args, "ppo_n_steps", 2048)), 1)
    episodes_per_update = int(np.ceil(ppo_n_steps / episode_steps))
    return max(1, ppo_updates * episodes_per_update)


def make_env(args) -> Monitor:
    topology_name = getattr(args, "center_gap_topology", "medium_270m")
    env = GlobalPPO3GNBEnv(
        seed=args.seed,
        n_gnbs=args.n_gnbs,
        slice_types=SLICE_TYPES,
        include_ue_counts=args.include_ue_counts,
        include_service_metrics=args.include_service_metrics,
        use_sumo_mobility=args.use_sumo_mobility,
        radio_substeps=args.radio_substeps,
        radio_tick_seconds=getattr(args, "radio_tick_seconds", None),
        pf_averaging_window_s=getattr(args, "pf_averaging_window_s", 0.25),
        gnb_configs=CENTER_GAP_GNB_CONFIGS[topology_name],
        local_steps_per_global=args.local_steps_per_global,
        global_steps_per_episode=args.global_steps_per_episode,
        scenario_mode=args.scenario_mode,
        snapshot_scenario=args.snapshot_scenario,
        terminal_reward_only=not args.dense_window_reward,
        use_progress_reward=args.use_progress_reward,
        max_handovers_per_local_step=args.max_handovers_per_local_step,
        max_handovers_per_ue_episode=getattr(args, "max_handovers_per_ue_episode", 2),
        max_handovers_per_episode=getattr(args, "max_handovers_per_episode", 20),
        handover_pingpong_guard_s=getattr(args, "handover_pingpong_guard_s", 30.0),
        action_direction_reward_weight=args.action_direction_reward_weight,
        snapshot_block_episodes=args.snapshot_block_episodes,
        light_load_ues=args.light_load_ues,
        medium_load_ues=args.medium_load_ues,
        high_load_ues=args.high_load_ues,
        print_scenarios=args.debug,
        slice_prb_budgets=args.slice_prb_budgets,
        max_prbs_per_ue=args.max_prbs_per_ue,
        directional_global_action=True,
        global_reward_mu=getattr(args, "load_balance_reward_weight", 2.0),
        global_reward_zeta=getattr(args, "saturation_reward_weight", 1.0),
        global_reward_beta=0.0,
        global_action_kappa=getattr(args, "bias_smoothing_weight", 0.01),
        global_action_lambda=getattr(args, "negative_bias_penalty_weight", 0.01),
        gnb_load_target=getattr(args, "gnb_load_target", 0.65),
        excess_load_reward_weight=getattr(
            args, "excess_load_reward_weight", 1.0
        ),
        served_share_reward_weight=getattr(
            args, "served_share_reward_weight", 1.0
        ),
        served_active_floor_reward_weight=getattr(
            args, "served_active_floor_reward_weight", 1.0
        ),
        served_active_floor=getattr(args, "served_active_floor", 0.20),
        jain_fairness_weight=getattr(args, "jain_fairness_weight", 1.0),
        sla_deadband=args.sla_deadband,
        upper_window_seconds=args.upper_window_seconds,
        training_scenarios=args.training_scenarios,
        scenario_selection=args.scenario_selection,
        curriculum_block_episodes=getattr(args, "curriculum_block_episodes", 1),
        fixed_stage_episodes=getattr(args, "fixed_stage_episodes", 500),
        slow_stage_episodes=getattr(args, "slow_stage_episodes", 1000),
        global_neutral_bias_weight=getattr(args, "global_neutral_bias_weight", 0.1),
        neutral_bias_eps=getattr(args, "neutral_bias_eps", 0.05),
        wrong_bias_penalty_weight=getattr(args, "wrong_bias_penalty_weight", 0.05),
        global_bad_direction_eta=getattr(args, "global_bad_direction_eta", 0.025),
        global_unsafe_target_rho=getattr(args, "global_unsafe_target_rho", 0.05),
        sla_severity_level_weight=getattr(args, "sla_severity_level_weight", 0.1),
        load_balance_level_weight=getattr(args, "load_balance_level_weight", 1.0),
        a3_handover_cooldown_s=getattr(args, "a3_handover_cooldown_s", 2.0),
        a3_min_residence_s=getattr(args, "a3_min_residence_s", 2.0),
        a3_history_window_s=getattr(args, "a3_history_window_s", 20.0),
        a3_pingpong_threshold_s=getattr(args, "a3_pingpong_threshold_s", 5.0),
        safe_admission_enabled=getattr(args, "safe_admission", False),
        warmup_steps=getattr(args, "warmup_steps", 2),
        post_handover_settle_steps=getattr(args, "post_handover_settle_steps", 4),
        demand_calibration_alpha=getattr(
            args, "demand_calibration_alpha", 0.5
        ),
    )
    return Monitor(env)


def evaluate_upper_policy(
    model: PPO,
    env: Monitor,
    n_eval_episodes: int = 10,
    validation_csv: Path | None = None,
) -> Dict[str, Any]:
    rows = []
    episode_returns = []
    target_error_deltas = []
    handovers = []
    sla_counts = []
    overloaded_negative_scores = []
    light_nonnegative_scores = []

    for episode in range(int(n_eval_episodes)):
        obs, info = env.reset()
        done = False
        ep_return = 0.0
        step = 0
        first_imbalance = None
        last_imbalance = None

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            step += 1
            ep_return += float(reward)
            if first_imbalance is None:
                first_imbalance = float(
                    info.get(
                        "used_prb_balance_cost_start",
                        info.get("load_imbalance_start", 0.0),
                    )
                )
            last_imbalance = float(
                info.get(
                    "used_prb_balance_cost_end",
                    info.get("load_imbalance_end", 0.0),
                )
            )
            handovers.append(int(info.get("handover_count", 0)))
            sla_counts.append(float(info.get("sla_count", 0.0)))
            directional_bias = np.asarray(
                info.get("directional_bias_tensor", []), dtype=float
            )
            load_matrix = np.asarray(info.get("load_matrix", []), dtype=float)
            overloaded_negative, light_nonnegative = _bias_quality_scores(
                directional_bias, load_matrix
            )
            overloaded_negative_scores.append(overloaded_negative)
            light_nonnegative_scores.append(light_nonnegative)
            used_prb_start = info.get("used_prb_matrix_start", [])
            used_prb_end = info.get("used_prb_matrix_end", [])

            rows.append({
                "episode": int(episode),
                "step": int(step),
                "scenario_name": str(info.get("scenario_name", "")),
                "episode_time_s": float(info.get("episode_time_s", 0.0)),
                "episode_duration_s": float(info.get("episode_duration_s", 0.0)),
                "upper_window_seconds": float(info.get("upper_window_seconds", 0.0)),
                "local_step_seconds": float(info.get("local_step_seconds", 0.0)),
                "radio_service_seconds_per_upper_window": float(
                    info.get("radio_service_seconds_per_upper_window", 0.0)
                ),
                "post_handover_settle_steps": int(
                    info.get("post_handover_settle_steps", 0)
                ),
                "radio_measurement_steps": int(
                    info.get("radio_measurement_steps", 0)
                ),
                "prb_measurement_mode": str(
                    info.get("load_measurement_mode", "")
                ),
                "network_used_prb_start": _matrix_sum(used_prb_start),
                "network_used_prb_end": _matrix_sum(used_prb_end),
                "mean_gnb_used_prb_start": _row_sum_mean(used_prb_start),
                "mean_gnb_used_prb_end": _row_sum_mean(used_prb_end),
                "max_gnb_used_prb_start": _row_sum_max(used_prb_start),
                "max_gnb_used_prb_end": _row_sum_max(used_prb_end),
                "reward": float(reward),
                "load_variance": float(info.get("load_variance", 0.0)),
                "target_load_error": float(info.get("target_load_error", 0.0)),
                "used_prb_balance_cost_start": float(
                    info.get("used_prb_balance_cost_start", 0.0)
                ),
                "used_prb_balance_cost_end": float(
                    info.get("used_prb_balance_cost_end", 0.0)
                ),
                "target_load_error_start": float(info.get("target_load_error_start", 0.0)),
                "target_load_error_end": float(info.get("target_load_error_end", 0.0)),
                "overload_ratio": float(info.get("overload_ratio", 0.0)),
                "sla_count": float(info.get("sla_count", 0.0)),
                "sla_severity": float(info.get("sla_severity", 0.0)),
                "sla_deadband": float(info.get("sla_deadband", 0.0)),
                "handover_count": int(info.get("handover_count", 0)),
                "global_network_cost": float(info.get("global_network_cost", 0.0)),
                "global_cost_start": float(info.get("global_cost_start", 0.0)),
                "global_cost_end": float(info.get("global_cost_end", 0.0)),
                "global_cost_improvement": float(info.get("global_cost_improvement", 0.0)),
                "global_action_penalty": float(info.get("global_action_penalty", 0.0)),
                "global_negative_bias_penalty": float(
                    info.get("global_negative_bias_penalty", 0.0)
                ),
                "reward_used_prb_balance_improvement": float(
                    info.get("reward_used_prb_balance_improvement", 0.0)
                ),
                "reward_used_prb_balance_improvement_raw": float(
                    info.get("reward_used_prb_balance_improvement_raw", 0.0)
                ),
                "reward_active_slice_count": int(
                    info.get("reward_active_slice_count", 0)
                ),
                "reward_saturation_improvement": float(
                    info.get("reward_saturation_improvement", 0.0)
                ),
                "reward_excess_load_improvement": float(
                    info.get("reward_excess_load_improvement", 0.0)
                ),
                "reward_excess_load_improvement_raw": float(
                    info.get("reward_excess_load_improvement_raw", 0.0)
                ),
                "reward_served_share_improvement": float(
                    info.get("reward_served_share_improvement", 0.0)
                ),
                "reward_served_share_improvement_raw": float(
                    info.get("reward_served_share_improvement_raw", 0.0)
                ),
                "served_share_cost_start": float(
                    info.get("served_share_cost_start", 0.0)
                ),
                "served_share_cost_end": float(
                    info.get("served_share_cost_end", 0.0)
                ),
                "reward_served_active_floor": float(
                    info.get("reward_served_active_floor", 0.0)
                ),
                "reward_served_active_floor_raw": float(
                    info.get("reward_served_active_floor_raw", 0.0)
                ),
                "served_active_floor_cost_start": float(
                    info.get("served_active_floor_cost_start", 0.0)
                ),
                "served_active_floor_cost_end": float(
                    info.get("served_active_floor_cost_end", 0.0)
                ),
                "served_active_floor": float(
                    info.get("served_active_floor", 0.0)
                ),
                "reward_jain_fairness": float(
                    info.get("reward_jain_fairness", 0.0)
                ),
                "jain_fairness_raw": float(
                    info.get("jain_fairness_raw", 0.0)
                ),
                "jain_fairness_normalized": float(
                    info.get("jain_fairness_normalized", 0.0)
                ),
                "gnb_excess_load_cost_start": float(
                    info.get("gnb_excess_load_cost_start", 0.0)
                ),
                "gnb_excess_load_cost_end": float(
                    info.get("gnb_excess_load_cost_end", 0.0)
                ),
                "gnb_load_target_requested": float(
                    info.get("gnb_load_target_requested", 0.65)
                ),
                "gnb_load_target_effective": float(
                    info.get("gnb_load_target_effective", 0.65)
                ),
                "gnb_load_target_feasible": bool(
                    info.get("gnb_load_target_feasible", True)
                ),
                "persistent_demand_utilization": float(
                    info.get("persistent_demand_utilization", 0.0)
                ),
                "reward_sla_improvement": float(info.get("reward_sla_improvement", 0.0)),
                "saturation_count": int(info.get("saturation_count", 0)),
                "action_direction_reward": float(info.get("action_direction_reward", 0.0)),
                "overloaded_negative_fraction": overloaded_negative,
                "light_nonnegative_fraction": light_nonnegative,
                "bias_matrix": _json_array(info.get("bias_matrix", [])),
                "directional_bias_tensor": _json_array(info.get("directional_bias_tensor", [])),
                "directional_offset_tensor": _json_array(info.get("directional_offset_tensor", [])),
                "safe_admission_capacities": json.dumps(
                    {
                        ":".join(map(str, key)): value
                        for key, value in info.get("safe_admission", {}).get("capacities", {}).items()
                    }
                ),
                "safe_admission_accepted": json.dumps(
                    {
                        ":".join(map(str, key)): value
                        for key, value in info.get("safe_admission", {}).get("accepted", {}).items()
                    }
                ),
                "safe_admission_source_capacities": json.dumps(
                    {
                        ":".join(map(str, key)): value
                        for key, value in info.get("safe_admission", {})
                        .get("source_capacities", {})
                        .items()
                    }
                ),
                "safe_admission_source_accepted": json.dumps(
                    {
                        ":".join(map(str, key)): value
                        for key, value in info.get("safe_admission", {})
                        .get("source_accepted", {})
                        .items()
                    }
                ),
                "safe_admission_stats": json.dumps(
                    info.get("safe_admission", {}).get("stats", {})
                ),
                "target_load_matrix": _json_array(info.get("target_load_matrix", [])),
                "balance_target_matrix": _json_array(info.get("balance_target_matrix", [])),
                "load_matrix": _json_array(info.get("load_matrix", [])),
                "sla_matrix": _json_array(info.get("sla_matrix", [])),
                "sla_violation_matrix": _json_array(info.get("sla_violation_matrix", [])),
                "sla_severity_matrix": _json_array(info.get("sla_severity_matrix", [])),
                "sla_window_metrics": json.dumps(
                    {
                        ":".join(map(str, key)): value
                        for key, value in info.get("sla_window_metrics", {}).items()
                    }
                ),
                "ue_count_matrix": _json_array(info.get("ue_count_matrix", [])),
            })
            _add_qos_fields(rows[-1], info)
            _add_matrix_total_fields(rows[-1], "used_prb_start", used_prb_start)
            _add_matrix_total_fields(rows[-1], "used_prb_end", used_prb_end)
            _add_served_floor_reference_fields(
                rows[-1],
                info.get("served_active_floor_reference_gnb_loads", []),
            )
            _add_directional_bias_fields(
                rows[-1], info.get("directional_bias_tensor", [])
            )
            for prefix, key in (
                ("target_load", "target_load_matrix"),
                ("balance_target", "balance_target_matrix"),
                ("used_prb_start", "used_prb_matrix_start"),
                ("used_prb_end", "used_prb_matrix_end"),
                ("sla", "sla_matrix"),
                ("ue_count", "ue_count_matrix"),
            ):
                _add_flat_matrix_fields(rows[-1], prefix, info.get(key, []))

        episode_returns.append(ep_return)
        if first_imbalance is not None and last_imbalance is not None:
            target_error_deltas.append(first_imbalance - last_imbalance)

    if validation_csv is not None:
        validation_csv = Path(validation_csv)
        validation_csv.parent.mkdir(parents=True, exist_ok=True)
        with validation_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=VALIDATION_FIELDS,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    field: row.get(field, "")
                    for field in VALIDATION_FIELDS
                })

    return {
        "n_eval_episodes": int(n_eval_episodes),
        "mean_eval_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "mean_target_load_error_delta": (
            float(np.mean(target_error_deltas)) if target_error_deltas else 0.0
        ),
        "mean_handover_count_per_step": float(np.mean(handovers)) if handovers else 0.0,
        "mean_sla_count": float(np.mean(sla_counts)) if sla_counts else 0.0,
        "mean_overloaded_negative_fraction": (
            float(np.mean(overloaded_negative_scores)) if overloaded_negative_scores else 0.0
        ),
        "mean_light_nonnegative_fraction": (
            float(np.mean(light_nonnegative_scores)) if light_nonnegative_scores else 0.0
        ),
        "validation_csv": None if validation_csv is None else str(validation_csv),
    }


def save_learning_curve(
    training_csv: Path,
    output_path: Path,
    rolling_window: int = 200,
) -> Path | None:
    """Create a compact directional-learning dashboard from the training CSV."""
    training_csv = Path(training_csv)
    if not training_csv.exists():
        return None
    with training_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    def series(field: str) -> np.ndarray:
        values = []
        for row in rows:
            try:
                values.append(float(row.get(field, 0.0)))
            except (TypeError, ValueError):
                values.append(0.0)
        return np.asarray(values, dtype=float)

    def rolling(values: np.ndarray) -> np.ndarray:
        window = max(min(int(rolling_window), values.size), 1)
        cumulative = np.cumsum(np.insert(values, 0, 0.0))
        result = np.empty_like(values)
        for idx in range(values.size):
            start = max(0, idx + 1 - window)
            result[idx] = (
                cumulative[idx + 1] - cumulative[start]
            ) / float(idx + 1 - start)
        return result

    steps = series("step")
    reward = series("reward")
    imbalance = series("used_prb_balance_cost_end")
    handovers = series("handover_count")
    left_bias = series("bias_g1_to_g0_eMBB")
    right_bias = series("bias_g1_to_g2_eMBB")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes[0, 0].plot(steps, reward, color="#8ecae6", alpha=0.18, lw=0.7)
    axes[0, 0].plot(
        steps, rolling(reward), color="#023047", lw=2.0,
        label=f"rolling mean ({min(rolling_window, len(rows))})",
    )
    axes[0, 0].axhline(0.0, color="black", lw=0.8)
    axes[0, 0].set(title="Scaled causal reward", ylabel="reward")
    axes[0, 0].legend()

    axes[0, 1].plot(
        steps, rolling(imbalance), color="#d62828", lw=2.0,
        label="final PRB balance cost",
    )
    axes[0, 1].set(title="Useful PRB balancing", ylabel="PRB balance cost")
    axes[0, 1].legend()

    axes[1, 0].plot(
        steps, rolling(left_bias), color="#e76f51", lw=2.0,
        label="center to left",
    )
    axes[1, 0].plot(
        steps, rolling(right_bias), color="#2a9d8f", lw=2.0,
        label="center to right",
    )
    axes[1, 0].axhline(0.0, color="black", lw=0.8)
    axes[1, 0].set(
        title="Learned directional biases",
        xlabel="training step",
        ylabel="bias",
    )
    axes[1, 0].legend()

    axes[1, 1].plot(
        steps, rolling(handovers), color="#6a4c93", lw=2.0,
        label="handovers",
    )
    axes[1, 1].set(
        title="Executed migration volume",
        xlabel="training step",
        ylabel="UEs per episode",
    )
    axes[1, 1].legend()

    for axis in axes.reshape(-1):
        axis.grid(alpha=0.25)
    fig.suptitle("Directional upper-PPO learning curve", fontsize=14)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train Phase-2 upper/global PPO for 3-gNB HRL.")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-dir", type=Path, default=Path("models/upper_ppo_3gnb"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-sumo-mobility", action="store_true")
    parser.add_argument("--include-ue-counts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-service-metrics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--directional-global-action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated in v15: upper PPO always emits one bias per gNB and slice.",
    )
    parser.add_argument(
        "--slice-prb-budgets",
        type=json.loads,
        default=None,
        help='Optional JSON dict, for example \'{"eMBB": 50, "URLLC": 50, "mMTC": 50}\'.',
    )
    parser.add_argument("--max-prbs-per-ue", type=int, default=None)
    parser.add_argument(
        "--sla-deadband",
        type=float,
        default=0.05,
        help="Ignore SLA violation magnitudes at or below this value in the upper reward.",
    )
    parser.add_argument("--n-gnbs", type=int, default=3)
    parser.add_argument("--local-steps-per-global", type=int, default=10)
    parser.add_argument(
        "--radio-substeps",
        type=int,
        default=100,
        help="Number of radio-service ticks simulated inside each local mobility step.",
    )
    parser.add_argument(
        "--pf-averaging-window-s",
        type=float,
        default=0.25,
        help="Physical duration of the proportional-fair throughput averaging window.",
    )
    parser.add_argument(
        "--radio-tick-seconds",
        type=float,
        default=None,
        help=(
            "Duration of one radio tick. By default it is derived so radio, "
            "mobility, and upper-window clocks are exactly synchronized."
        ),
    )
    parser.add_argument("--global-steps-per-episode", type=int, default=12)
    parser.add_argument(
        "--upper-window-seconds",
        type=float,
        default=1.0,
        help="Physical scenario time represented by one upper PPO action.",
    )
    parser.add_argument(
        "--training-scenarios",
        type=str,
        default="high_load_inner_asymmetric",
        help=(
            "Comma-separated slice-aware scenario names, or 'all'. In default "
            "single-scenario mode this is overwritten by --single-training-scenario. "
            "Use --curriculum-training when you intentionally want a scenario pool."
        ),
    )
    parser.add_argument(
        "--single-training-scenario",
        type=str,
        default="high_load_inner_asymmetric",
        help=(
            "Scenario used by default upper PPO training. Keeping one scenario "
            "makes the previous-bias state and bias-smoothness reward coherent "
            "while the policy learns."
        ),
    )
    parser.add_argument(
        "--curriculum-training",
        action="store_true",
        help=(
            "Train over the comma-separated --training-scenarios pool. Leave off "
            "for the default fixed single-scenario training."
        ),
    )
    parser.add_argument(
        "--block-curriculum-training",
        action="store_true",
        help=(
            "Train over a scenario pool in long blocks: repeat one scenario for "
            "--curriculum-block-episodes, then switch to the next scenario."
        ),
    )
    parser.add_argument(
        "--curriculum-block-episodes",
        type=int,
        default=0,
        help=(
            "Episodes to keep each scenario before switching in block curriculum. "
            "Use 0 to compute enough episodes for --ppo-updates-per-scenario PPO updates."
        ),
    )
    parser.add_argument(
        "--ppo-updates-per-scenario",
        type=int,
        default=3,
        help=(
            "When --curriculum-block-episodes is 0, compute a block long enough "
            "for this many PPO rollout updates on one scenario before switching."
        ),
    )
    parser.add_argument(
        "--center-gap-topology",
        choices=tuple(CENTER_GAP_GNB_CONFIGS),
        default="medium_270m",
        help=(
            "Left-center-right gNB gap topology. UE placement and traffic "
            "remain identical across tight_220m, medium_270m, and wide_320m."
        ),
    )
    parser.add_argument(
        "--scenario-selection",
        choices=("cycle", "random", "staged", "block"),
        default="cycle",
        help=(
            "Select retained scenarios by deterministic cycle, random, or staged mode. "
            "Ignored unless --curriculum-training or --block-curriculum-training is set."
        ),
    )
    parser.add_argument(
        "--fixed-stage-episodes",
        type=int,
        default=500,
        help="In staged mode, train retained fixed scenarios for this many episodes.",
    )
    parser.add_argument(
        "--slow-stage-episodes",
        type=int,
        default=1000,
        help="Compatibility option; all retained scenarios currently use the fixed tier.",
    )
    parser.add_argument(
        "--max-handovers-per-local-step",
        type=int,
        default=1,
        help="Limit heuristic handovers during each lower/local simulator step.",
    )
    parser.add_argument(
        "--max-handovers-per-ue-episode",
        type=int,
        default=2,
        help="Reject further handovers after one UE reaches this episode total.",
    )
    parser.add_argument(
        "--max-handovers-per-episode",
        type=int,
        default=20,
        help="Hard safety budget for all successful handovers in one episode.",
    )
    parser.add_argument(
        "--handover-pingpong-guard-s",
        type=float,
        default=30.0,
        help="Block a direct return to the previous gNB for this simulated time.",
    )
    parser.add_argument(
        "--action-direction-reward-weight",
        type=float,
        default=0.0,
        help=(
            "Deprecated v12 diagnostic weight. It is logged only and is not "
            "part of the v15 load-balance reward."
        ),
    )
    parser.add_argument(
        "--global-neutral-bias-weight",
        type=float,
        default=0.1,
        help="Penalty weight for non-zero bias when an active slice is already balanced.",
    )
    parser.add_argument(
        "--neutral-bias-eps",
        type=float,
        default=0.05,
        help="Max per-gNB load deviation from balance target before the neutral-bias penalty fires.",
    )
    parser.add_argument(
        "--wrong-bias-penalty-weight",
        type=float,
        default=0.05,
        help=(
            "Penalty weight for retaining above-average cells or releasing "
            "below-average cells."
        ),
    )
    parser.add_argument(
        "--global-bad-direction-eta",
        type=float,
        default=0.025,
        help="Legacy bad-direction diagnostic weight; excluded from the PDF v15 PPO reward.",
    )
    parser.add_argument(
        "--global-unsafe-target-rho",
        type=float,
        default=0.05,
        help="Legacy unsafe-target diagnostic weight; excluded from the PDF v15 PPO reward.",
    )
    parser.add_argument(
        "--sla-severity-level-weight",
        type=float,
        default=0.1,
        help="Legacy SLA-level diagnostic weight; excluded from the PDF v15 PPO reward.",
    )
    parser.add_argument(
        "--load-balance-level-weight",
        type=float,
        default=1.0,
        help=(
            "Legacy persistent-balance diagnostic weight; excluded from the "
            "PDF v15 PPO reward."
        ),
    )
    parser.add_argument(
        "--a3-history-window-s",
        type=float,
        default=20.0,
        help="Time horizon in seconds for handover failure/ping-pong ratios.",
    )
    parser.add_argument(
        "--a3-pingpong-threshold-s",
        type=float,
        default=5.0,
        help="Maximum elapsed seconds for classifying a return as a ping-pong.",
    )
    parser.add_argument(
        "--a3-handover-cooldown-s",
        type=float,
        default=2.0,
        help="Seconds a UE must wait after a handover before it can trigger another one (2s = 2 global steps).",
    )
    parser.add_argument(
        "--a3-min-residence-s",
        type=float,
        default=2.0,
        help="Minimum seconds a UE must reside on the new cell before re-evaluating (clamped to >= cooldown).",
    )
    parser.add_argument(
        "--load-balance-reward-weight",
        type=float,
        default=2.0,
        help="Legacy diagnostic weight; the PDF v15 PPO reward uses raw load-dispersion improvement.",
    )
    parser.add_argument(
        "--saturation-reward-weight",
        type=float,
        default=1.0,
        help=(
            "Weight for normalized saturation-count improvement. This prevents "
            "allocated-utilization balancing from rewarding all cells at 100%%."
        ),
    )
    parser.add_argument(
        "--gnb-load-target",
        type=float,
        default=0.65,
        help=(
            "Preferred maximum total physical utilization per gNB. It is "
            "raised to mean persistent demand when 0.65 is infeasible."
        ),
    )
    parser.add_argument(
        "--excess-load-reward-weight",
        type=float,
        default=1.0,
        help="Weight for improvement in squared utilization above the feasible target.",
    )
    parser.add_argument(
        "--served-share-reward-weight",
        "--demand-share-reward-weight",
        dest="served_share_reward_weight",
        type=float,
        default=1.0,
        help=(
            "Weight for served-useful-PRB sharing improvement across gNBs. "
            "This discourages emptying one cell while useful PRBs look balanced."
        ),
    )
    parser.add_argument(
        "--served-active-floor-reward-weight",
        type=float,
        default=1.0,
        help="Weight for penalizing initially served gNBs that become nearly idle.",
    )
    parser.add_argument(
        "--served-active-floor",
        type=float,
        default=0.20,
        help="Minimum useful-load row total for a gNB that was active at window start.",
    )
    parser.add_argument(
        "--sla-reward-weight",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility option. SLA is logged and may guard safe "
            "admission, but it is excluded from the upper routing reward."
        ),
    )
    parser.add_argument(
        "--bias-smoothing-weight",
        type=float,
        default=0.01,
        help="PDF v15 lambda_delta for squared upper-bias changes.",
    )
    parser.add_argument(
        "--negative-bias-penalty-weight",
        type=float,
        default=0.01,
        help="Persistent penalty weight on squared negative upper-bias magnitude.",
    )
    parser.add_argument(
        "--scenario-mode",
        choices=("curriculum", "snapshot", "random"),
        default="curriculum",
        help="Use one explicit time-aware curriculum scenario per episode by default.",
    )
    parser.add_argument(
        "--snapshot-scenario",
        type=str,
        default="mixed",
        help=(
            "Snapshot name to train on, or 'mixed' to cycle snapshot scenarios in episode blocks. "
            "Examples: multi_slice_multi_gnb_congestion, embb_g0_offload, urllc_g1_offload, "
            "mmtc_g2_offload, all_neutral."
        ),
    )
    parser.add_argument(
        "--snapshot-block-episodes",
        type=int,
        default=1,
        help="Legacy snapshot mode only. Curriculum mode always changes scenario per episode.",
    )
    parser.add_argument(
        "--light-load-ues",
        type=int,
        default=1,
        help="UEs created for target loads below 0.5.",
    )
    parser.add_argument(
        "--medium-load-ues",
        type=int,
        default=2,
        help="UEs created for target loads from 0.5 to below 0.8.",
    )
    parser.add_argument(
        "--high-load-ues",
        type=int,
        default=3,
        help="UEs created for target loads from 0.8 upward.",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-n-steps", type=int, default=2048)
    parser.add_argument("--ppo-batch-size", type=int, default=256)
    parser.add_argument("--ppo-n-epochs", type=int, default=10)
    parser.add_argument(
        "--dense-window-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return the PDF reward after every upper window (default: enabled). PPO with GAE requires dense per-step rewards for correct advantage estimation.",
    )
    parser.add_argument(
        "--safe-admission",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the SafeAdmissionController quota gate (default: disabled). "
            "When disabled the agent's A3 offsets are the sole driver of handovers, "
            "giving PPO a clean bias→HO gradient without load-based fallback interference."
        ),
    )
    parser.add_argument(
        "--use-progress-reward",
        action="store_true",
        help="Deprecated compatibility flag; target-error shaping is excluded from the PDF v15 reward.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Write one detailed training CSV row every N PPO steps (episode ends are always logged).",
    )
    parser.add_argument(
        "--log-flush-every",
        type=int,
        default=100,
        help="Flush the sampled training CSV after this many written rows.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2,
        help=(
            "Number of upper-window steps to run with zero bias at the start of each episode "
            "before the agent's first action. The SLA window and load state are populated "
            "during warmup so the first real observation is not cold. Warmup interactions "
            "are invisible to PPO (episode counters reset afterwards). A value of 2-4 is "
            "recommended when using 1-step episodes."
        ),
    )
    parser.add_argument(
        "--post-handover-settle-steps",
        type=int,
        default=4,
        help=(
            "Number of local steps to run after applying the action but BEFORE opening the "
            "radio measurement window. During these steps the A3 handover fires and PRBs "
            "recalculate at the new gNB, so the transient coverage-gap peak is excluded from "
            "the reward signal. Must be < local_steps_per_global. A value of handover_ttt+1 "
            "is recommended (e.g. 4 when handover_ttt=3 and local_steps_per_global=10)."
        ),
    )
    parser.add_argument(
        "--demand-calibration-alpha",
        type=float,
        default=0.5,
        help="Smoothing gain for requested-versus-achieved PRB demand calibration.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args = resolve_upper_training_curriculum_args(args)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(args.model_dir) / f"run_{run_timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    training_csv = model_dir / "training_log.csv"
    validation_csv = model_dir / "validation_log.csv"
    learning_curve_path = model_dir / "learning_curve.png"
    final_model_path = model_dir / "upper_ppo_final.zip"
    best_model_path = model_dir / "upper_ppo_best.zip"
    config_path = model_dir / "config.json"

    effective_radio_tick_seconds = (
        float(args.radio_tick_seconds)
        if args.radio_tick_seconds is not None
        else float(args.upper_window_seconds)
        / max(int(args.local_steps_per_global), 1)
        / max(int(args.radio_substeps), 1)
    )
    config = {
        "run_timestamp": run_timestamp,
        "total_timesteps": int(args.total_timesteps),
        "seed": int(args.seed),
        "model_dir": str(model_dir),
        "device": str(args.device),
        "use_sumo_mobility": bool(args.use_sumo_mobility),
        "include_ue_counts": bool(args.include_ue_counts),
        "include_service_metrics": bool(args.include_service_metrics),
        "directional_global_action": True,
        "slice_prb_budgets": args.slice_prb_budgets,
        "max_prbs_per_ue": None if args.max_prbs_per_ue is None else int(args.max_prbs_per_ue),
        "sla_deadband": float(args.sla_deadband),
        "n_gnbs": int(args.n_gnbs),
        "slice_types": list(SLICE_TYPES),
        "local_steps_per_global": int(args.local_steps_per_global),
        "radio_substeps": int(args.radio_substeps),
        "radio_tick_seconds": effective_radio_tick_seconds,
        "pf_averaging_window_s": float(args.pf_averaging_window_s),
        "demand_calibration_alpha": float(args.demand_calibration_alpha),
        "radio_clock_derived": bool(args.radio_tick_seconds is None),
        "global_steps_per_episode": int(args.global_steps_per_episode),
        "upper_window_seconds": float(args.upper_window_seconds),
        "training_scenarios": str(args.training_scenarios),
        "single_training_scenario": str(args.single_training_scenario),
        "curriculum_training": bool(args.curriculum_training),
        "block_curriculum_training": bool(args.block_curriculum_training),
        "curriculum_block_episodes": int(args.curriculum_block_episodes),
        "ppo_updates_per_scenario": int(args.ppo_updates_per_scenario),
        "upper_training_regime": (
            "block_curriculum" if bool(args.block_curriculum_training)
            else "curriculum_pool" if bool(args.curriculum_training)
            else "single_coherent_scenario"
        ),
        "center_gap_topology": str(
            getattr(args, "center_gap_topology", "medium_270m")
        ),
        "scenario_selection": str(args.scenario_selection),
        "fixed_stage_episodes": int(args.fixed_stage_episodes),
        "slow_stage_episodes": int(args.slow_stage_episodes),
        "max_handovers_per_local_step": int(args.max_handovers_per_local_step),
        "max_handovers_per_ue_episode": int(args.max_handovers_per_ue_episode),
        "max_handovers_per_episode": int(args.max_handovers_per_episode),
        "handover_pingpong_guard_s": float(args.handover_pingpong_guard_s),
        "action_direction_reward_weight": float(args.action_direction_reward_weight),
        "load_balance_reward_weight": float(args.load_balance_reward_weight),
        "saturation_reward_weight": float(args.saturation_reward_weight),
        "gnb_load_target": float(args.gnb_load_target),
        "excess_load_reward_weight": float(args.excess_load_reward_weight),
        "served_share_reward_weight": float(args.served_share_reward_weight),
        "served_active_floor_reward_weight": float(
            args.served_active_floor_reward_weight
        ),
        "served_active_floor": float(args.served_active_floor),
        "sla_reward_weight": float(args.sla_reward_weight),
        "load_reward_scaling": "fraction_of_starting_imbalance_clipped_to_minus1_plus1",
        "global_neutral_bias_weight": float(args.global_neutral_bias_weight),
        "neutral_bias_eps": float(args.neutral_bias_eps),
        "wrong_bias_penalty_weight": float(args.wrong_bias_penalty_weight),
        "negative_bias_penalty_weight": float(args.negative_bias_penalty_weight),
        "global_bad_direction_eta": float(args.global_bad_direction_eta),
        "global_unsafe_target_rho": float(args.global_unsafe_target_rho),
        "sla_severity_level_weight": float(args.sla_severity_level_weight),
        "load_balance_level_weight": float(args.load_balance_level_weight),
        "a3_history_window_s": float(args.a3_history_window_s),
        "a3_pingpong_threshold_s": float(args.a3_pingpong_threshold_s),
        "a3_handover_cooldown_s": float(args.a3_handover_cooldown_s),
        "a3_min_residence_s": float(args.a3_min_residence_s),
        "scenario_mode": str(args.scenario_mode),
        "snapshot_scenario": str(args.snapshot_scenario),
        "snapshot_block_episodes": int(args.snapshot_block_episodes),
        "light_load_ues": int(args.light_load_ues),
        "medium_load_ues": int(args.medium_load_ues),
        "high_load_ues": int(args.high_load_ues),
        "terminal_reward_only": bool(not args.dense_window_reward),
        "use_progress_reward": bool(args.use_progress_reward),
        "safe_admission": bool(args.safe_admission),
        "warmup_steps": int(args.warmup_steps),
        "post_handover_settle_steps": int(args.post_handover_settle_steps),
        "log_every": int(args.log_every),
        "log_flush_every": int(args.log_flush_every),
        "learning_rate": float(args.learning_rate),
        "ppo_n_steps": int(args.ppo_n_steps),
        "ppo_batch_size": int(args.ppo_batch_size),
        "ppo_n_epochs": int(args.ppo_n_epochs),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    env = make_env(args)
    try:
        try:
            import tensorboard  # noqa: F401
            tensorboard_log = str(model_dir / "tb")
        except Exception:
            tensorboard_log = None
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=float(args.learning_rate),
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_steps=int(args.ppo_n_steps),
            batch_size=int(args.ppo_batch_size),
            n_epochs=int(args.ppo_n_epochs),
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=args.device,
            seed=args.seed,
        )
        callback = UpperTrainingCsvCallback(
            training_csv,
            best_model_path,
            log_every=args.log_every,
            flush_every=args.log_flush_every,
        )
        model.learn(total_timesteps=int(args.total_timesteps), callback=callback, progress_bar=False)
        model.save(final_model_path)
    finally:
        env.close()

    saved_learning_curve = save_learning_curve(
        training_csv,
        learning_curve_path,
    )

    eval_env = make_env(args)
    try:
        validation = evaluate_upper_policy(
            model,
            eval_env,
            n_eval_episodes=args.eval_episodes,
            validation_csv=validation_csv,
        )
    finally:
        eval_env.close()

    payload = {
        **config,
        "saved_final_model": str(final_model_path),
        "saved_best_model": str(best_model_path) if best_model_path.exists() else None,
        "training_csv": str(training_csv),
        "learning_curve": (
            None if saved_learning_curve is None
            else str(saved_learning_curve)
        ),
        "validation": validation,
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
