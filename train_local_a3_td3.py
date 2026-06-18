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
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from local_a3_training_env import LocalA3RuleBiasTrainingEnv
from local_a3_agent_wrapper import LocalA3OffsetEnv
from local_a3_training_scenarios import (
    EpisodeTrainingScenario,
    local_a3_training_scenario_set,
)


TRACE_SLICE_TYPES = ("eMBB", "URLLC", "mMTC")
TRACE_GNB_IDS = (0, 1)
TRACE_NEIGHBOR_ID = 1
TRACE_LOAD_FIELDS = [
    f"load_{gnb_id}_{slice_type}"
    for gnb_id in TRACE_GNB_IDS
    for slice_type in TRACE_SLICE_TYPES
]
TRACE_ACTION_FIELDS = [
    field
    for slice_type in TRACE_SLICE_TYPES
    for field in (
        f"action_{TRACE_NEIGHBOR_ID}_{slice_type}",
        f"raw_action_{TRACE_NEIGHBOR_ID}_{slice_type}",
        f"target_offset_{TRACE_NEIGHBOR_ID}_{slice_type}",
        f"applied_offset_{TRACE_NEIGHBOR_ID}_{slice_type}",
        f"previous_offset_{TRACE_NEIGHBOR_ID}_{slice_type}",
        f"offset_delta_{TRACE_NEIGHBOR_ID}_{slice_type}",
        f"offset_changed_{TRACE_NEIGHBOR_ID}_{slice_type}",
    )
]
TRACE_RULE_BIAS_FIELDS = [
    field
    for gnb_id in TRACE_GNB_IDS
    for slice_type in TRACE_SLICE_TYPES
    for field in (
        f"rule_bias_{gnb_id}_{slice_type}",
        f"held_rule_bias_{gnb_id}_{slice_type}",
        f"bias_changed_{gnb_id}_{slice_type}",
    )
]
TRACE_COUNT_FIELDS = [
    f"count_{gnb_id}_{slice_type}"
    for gnb_id in TRACE_GNB_IDS
    for slice_type in TRACE_SLICE_TYPES
]
TRACE_TRACKING_FIELDS = [
    field
    for slice_type in TRACE_SLICE_TYPES
    for field in (
        f"desired_offset_{TRACE_NEIGHBOR_ID}_{slice_type}",
        f"tracking_error_{TRACE_NEIGHBOR_ID}_{slice_type}",
    )
]
TRACE_PAIRWISE_FIELDS = [
    field
    for slice_type in TRACE_SLICE_TYPES
    for field in (
        f"serving_bias_0_to_{TRACE_NEIGHBOR_ID}_{slice_type}",
        f"target_bias_0_to_{TRACE_NEIGHBOR_ID}_{slice_type}",
        f"tau_0_to_{TRACE_NEIGHBOR_ID}_{slice_type}",
    )
]
EVAL_SLICE_OFFSET_FIELDS = [
    field
    for slice_type in TRACE_SLICE_TYPES
    for field in (
        f"mean_applied_offset_{slice_type}_bias_neg1",
        f"mean_applied_offset_{slice_type}_bias_0",
        f"mean_applied_offset_{slice_type}_bias_pos1",
        f"average_tracking_error_{slice_type}",
    )
]

TRACE_FIELDS = [
    "phase",
    "run_timestamp",
    "global_step",
    "episode",
    "episode_step",
    "bias_case",
    "bias_case_matched",
    "case_sampling_attempts",
    *TRACE_ACTION_FIELDS,
    "action_hold_counter",
    "action_hold_steps",
    "reward",
    "episode_return",
    "done",
    *TRACE_RULE_BIAS_FIELDS,
    "bias_hold_counter",
    "bias_hold_steps",
    *TRACE_LOAD_FIELDS,
    *TRACE_COUNT_FIELDS,
    *TRACE_PAIRWISE_FIELDS,
    *TRACE_TRACKING_FIELDS,
    "total_ues",
    "connected_ues",
    "ue_tracking",
    "avg_sinr_db",
    "min_sinr_db",
    "avg_rx_power_dbm",
    "avg_throughput_bps",
    "avg_queue_bits",
    "handover_attempts",
    "handover_successes",
    "handover_failures",
    "handover_ping_pongs",
    "tracking_penalty",
    "sla_penalty",
    "mobility_penalty",
    "smoothness_penalty",
    "reward_total",
    "latest_handover",
    # radio summary fields returned by _radio_summary()
    "connected_ues",
    "avg_sinr_db",
    "min_sinr_db",
    "avg_rx_power_dbm",
    "avg_throughput_bps",
    "avg_queue_bits",
]

EVAL_FIELDS = [
    "episode",
    "episode_return",
    "episode_length",
    "mean_applied_offset_bias_neg1",
    "mean_applied_offset_bias_0",
    "mean_applied_offset_bias_pos1",
    *EVAL_SLICE_OFFSET_FIELDS,
    "handover_attempts",
    "handover_successes",
    "handover_failures",
    "handover_ping_pongs",
    "handover_failure_rate",
    "ping_pong_rate",
    "average_tracking_error",
    "offset_changes",
    "mean_offset_hold_duration",
    "mean_abs_offset_delta",
    "max_abs_offset_delta",
]


def _stringify_keys(data: Dict[Any, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in dict(data or {}).items():
        if isinstance(key, tuple):
            name = ":".join(str(part) for part in key)
        else:
            name = str(key)
        result[name] = value
    return result


def _unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def _lookup_tuple_dict(data: Dict[Any, Any], key, default=0.0):
    if key in data:
        return data[key]
    string_key = ":".join(str(part) for part in key)
    if string_key in data:
        return data[string_key]
    return default


def _desired_offset_from_info(env, info: Dict[str, Any], neighbor_id: int = 1, slice_type: str = "eMBB") -> float:
    """Read pairwise desired offset from info, or compute fallback.

    New formulation uses B=[b_i,s], target bias b_j,s and
    tau_i,j,s = 0.5*(b_i,s - b_j,s).
    """
    key = (neighbor_id, slice_type)
    desired_offsets = dict(info.get("desired_offsets", {}))
    if desired_offsets:
        return float(_lookup_tuple_dict(desired_offsets, key, 0.0))

    train_env = _unwrap_env(env)
    local_env = getattr(train_env, "local_env", None)
    if local_env is not None and hasattr(local_env, "_desired_offset"):
        return float(local_env._desired_offset(neighbor_id, slice_type))

    rule_bias = dict(info.get("rule_bias", {}))
    spawn_counts = dict(info.get("spawn_counts", {}))
    b_i = float(_lookup_tuple_dict(rule_bias, (0, slice_type), 0.0))
    b_j = float(_lookup_tuple_dict(rule_bias, (neighbor_id, slice_type), 0.0))
    tau = float(np.clip(0.5 * (b_i - b_j), -1.0, 1.0))
    neighbor_count = int(_lookup_tuple_dict(spawn_counts, (neighbor_id, slice_type), 0))
    alpha_k = float(getattr(local_env, "alpha_k", 2.0))
    k_ref = float(getattr(local_env, "k_ref", {}).get(slice_type, 20.0))
    k_target = float(getattr(local_env, "k_target", {}).get(slice_type, 0.5))
    desired = 6.0 * tau + alpha_k * (neighbor_count / max(k_ref, 1e-9) - k_target)
    return float(np.clip(desired, -6.0, 6.0))


def _pairwise_trace_values(info: Dict[str, Any]) -> Dict[str, float]:
    serving_bias = dict(info.get("serving_bias", {}))
    target_bias = dict(info.get("target_bias", {}))
    pairwise_tau = dict(info.get("pairwise_tau", {}))
    rule_bias = dict(info.get("rule_bias", {}))

    values = {}
    for slice_type in TRACE_SLICE_TYPES:
        key = (TRACE_NEIGHBOR_ID, slice_type)
        b_i = float(_lookup_tuple_dict(
            serving_bias, key, _lookup_tuple_dict(rule_bias, (0, slice_type), 0.0)
        ))
        b_j = float(_lookup_tuple_dict(
            target_bias, key, _lookup_tuple_dict(rule_bias, (TRACE_NEIGHBOR_ID, slice_type), 0.0)
        ))
        tau = float(_lookup_tuple_dict(pairwise_tau, key, 0.5 * (b_i - b_j)))
        values.update({
            f"serving_bias_0_to_{TRACE_NEIGHBOR_ID}_{slice_type}": b_i,
            f"target_bias_0_to_{TRACE_NEIGHBOR_ID}_{slice_type}": b_j,
            f"tau_0_to_{TRACE_NEIGHBOR_ID}_{slice_type}": tau,
        })
    return values


def _slice_load_trace_values(slice_loads: Dict[Any, Any]) -> Dict[str, float]:
    return {
        f"load_{gnb_id}_{slice_type}": float(
            _lookup_tuple_dict(slice_loads, (gnb_id, slice_type), 0.0)
        )
        for gnb_id in TRACE_GNB_IDS
        for slice_type in TRACE_SLICE_TYPES
    }


def _action_trace_values(
    action_arr: np.ndarray,
    action_temporal: Dict[str, Any],
    applied_offsets: Dict[Any, Any],
) -> Dict[str, Any]:
    raw_actions = dict(action_temporal.get("raw_actions", {}))
    target_offsets = dict(action_temporal.get("target_offsets", {}))
    previous_offsets = dict(action_temporal.get("previous_offsets", {}))
    offset_deltas = dict(action_temporal.get("offset_deltas", {}))
    changed_by_key = dict(action_temporal.get("offset_changed_by_key", {}))

    values = {}
    for idx, slice_type in enumerate(TRACE_SLICE_TYPES):
        key = (TRACE_NEIGHBOR_ID, slice_type)
        action_value = float(action_arr[idx]) if idx < action_arr.size else 0.0
        applied_offset = float(_lookup_tuple_dict(applied_offsets, key, 0.0))
        values.update({
            f"action_{TRACE_NEIGHBOR_ID}_{slice_type}": action_value,
            f"raw_action_{TRACE_NEIGHBOR_ID}_{slice_type}": float(
                _lookup_tuple_dict(raw_actions, key, action_value)
            ),
            f"target_offset_{TRACE_NEIGHBOR_ID}_{slice_type}": float(
                _lookup_tuple_dict(target_offsets, key, action_value)
            ),
            f"applied_offset_{TRACE_NEIGHBOR_ID}_{slice_type}": applied_offset,
            f"previous_offset_{TRACE_NEIGHBOR_ID}_{slice_type}": float(
                _lookup_tuple_dict(previous_offsets, key, applied_offset)
            ),
            f"offset_delta_{TRACE_NEIGHBOR_ID}_{slice_type}": float(
                _lookup_tuple_dict(offset_deltas, key, 0.0)
            ),
            f"offset_changed_{TRACE_NEIGHBOR_ID}_{slice_type}": bool(
                _lookup_tuple_dict(changed_by_key, key, False)
            ),
        })
    return values


def _rule_bias_trace_values(
    rule_bias: Dict[Any, Any],
    held_rule_bias: Dict[Any, Any],
    bias_changed: bool,
) -> Dict[str, Any]:
    values = {}
    for gnb_id in TRACE_GNB_IDS:
        for slice_type in TRACE_SLICE_TYPES:
            key = (gnb_id, slice_type)
            values.update({
                f"rule_bias_{gnb_id}_{slice_type}": float(_lookup_tuple_dict(rule_bias, key, 0.0)),
                f"held_rule_bias_{gnb_id}_{slice_type}": float(_lookup_tuple_dict(held_rule_bias, key, 0.0)),
                f"bias_changed_{gnb_id}_{slice_type}": bool(bias_changed),
            })
    return values


def _count_trace_values(spawn_counts: Dict[Any, Any]) -> Dict[str, int]:
    return {
        f"count_{gnb_id}_{slice_type}": int(
            _lookup_tuple_dict(spawn_counts, (gnb_id, slice_type), 0)
        )
        for gnb_id in TRACE_GNB_IDS
        for slice_type in TRACE_SLICE_TYPES
    }


def _tracking_trace_values(env, info: Dict[str, Any], applied_offsets: Dict[Any, Any]) -> Dict[str, float]:
    values = {}
    for slice_type in TRACE_SLICE_TYPES:
        key = (TRACE_NEIGHBOR_ID, slice_type)
        applied_offset = float(_lookup_tuple_dict(applied_offsets, key, 0.0))
        desired_offset = _desired_offset_from_info(
            env,
            info,
            neighbor_id=TRACE_NEIGHBOR_ID,
            slice_type=slice_type,
        )
        values.update({
            f"desired_offset_{TRACE_NEIGHBOR_ID}_{slice_type}": desired_offset,
            f"tracking_error_{TRACE_NEIGHBOR_ID}_{slice_type}": float(applied_offset - desired_offset),
        })
    return values


def _radio_summary(env) -> Dict[str, float]:
    train_env = _unwrap_env(env)
    base_env = getattr(train_env, "base_env", None)
    if base_env is None:
        return {
            "connected_ues": 0,
            "avg_sinr_db": 0.0,
            "min_sinr_db": 0.0,
            "avg_rx_power_dbm": 0.0,
            "avg_throughput_bps": 0.0,
            "avg_queue_bits": 0.0,
        }

    metrics = []
    for ue in base_env.get_all_ues():
        try:
            metrics.append(base_env.get_ue_radio_metrics(int(ue.id)))
        except Exception:
            continue

    connected = [m for m in metrics if bool(m.get("connected", False))]
    if not connected:
        return {
            "connected_ues": 0,
            "avg_sinr_db": 0.0,
            "min_sinr_db": 0.0,
            "avg_rx_power_dbm": 0.0,
            "avg_throughput_bps": 0.0,
            "avg_queue_bits": 0.0,
        }

    sinrs = [float(m.get("sinr_db", 0.0)) for m in connected]
    return {
        "connected_ues": int(len(connected)),
        "avg_sinr_db": float(np.mean(sinrs)),
        "min_sinr_db": float(np.min(sinrs)),
        "avg_rx_power_dbm": float(np.mean([float(m.get("rx_power_dbm", 0.0)) for m in connected])),
        "avg_throughput_bps": float(np.mean([float(m.get("throughput", 0.0)) for m in connected])),
        "avg_queue_bits": float(np.mean([float(m.get("queue", 0.0)) for m in connected])),
    }


def _ue_tracking_summary(env) -> str:
    train_env = _unwrap_env(env)
    base_env = getattr(train_env, "base_env", None)
    if base_env is None:
        return "[]"

    rows = []
    for ue in sorted(base_env.get_all_ues(), key=lambda item: int(item.id)):
        rows.append({
            "ue_id": int(ue.id),
            "slice": str(getattr(ue, "slice_type", "eMBB")),
            "serving_gnb": None if ue.serving_gnb is None else int(ue.serving_gnb),
            "connected": bool(getattr(ue, "connected", False)),
            "prbs": int(getattr(ue, "prbs", 0)),
            "x": round(float(getattr(ue, "x", 0.0)), 2),
            "y": round(float(getattr(ue, "y", 0.0)), 2),
        })
    return json.dumps(rows, sort_keys=True)


class StepCsvLogger:
    def __init__(self, path: Path, run_timestamp: str = ""):
        self.path = Path(path)
        self.run_timestamp = str(run_timestamp)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=TRACE_FIELDS)
        self.writer.writeheader()
        self.episode_by_phase = {"train": 0, "eval": 0}
        self.episode_step_by_phase = {"train": 0, "eval": 0}
        self.episode_return_by_phase = {"train": 0.0, "eval": 0.0}
        self.last_handover_idx_by_phase = {"train": 0, "eval": 0}

    def close(self):
        self.file.close()

    def log_step(self, env, phase: str, global_step: int, action, reward: float, done: bool, info: Dict[str, Any]):
        action_arr = np.asarray(action, dtype=float).reshape(-1)

        self.episode_step_by_phase[phase] += 1
        self.episode_return_by_phase[phase] += float(reward)

        applied_offsets = dict(info.get("applied_offsets", {}))
        rule_bias = dict(info.get("rule_bias", {}))
        action_temporal = dict(info.get("action_temporal", {}))
        bias_temporal = dict(info.get("bias_temporal", {}))
        held_rule_bias = dict(bias_temporal.get("held_rule_bias", {}))
        slice_loads = dict(info.get("post_action_slice_loads", info.get("slice_loads", {})))
        spawn_counts = dict(info.get("post_action_spawn_counts", info.get("spawn_counts", {})))
        handover_stats = dict(info.get("handover_stats", {}))
        reward_breakdown = dict(info.get("reward_breakdown", {}))
        radio = _radio_summary(env)

        latest_handover = ""
        train_env = _unwrap_env(env)
        base_env = getattr(train_env, "base_env", None)
        total_ues = int(len(base_env.get_all_ues())) if base_env is not None else 0
        if base_env is not None:
            events = list(getattr(base_env, "handover_events", []))
            last_idx = self.last_handover_idx_by_phase[phase]
            if len(events) > last_idx:
                latest_handover = json.dumps(events[-1], sort_keys=True)
            self.last_handover_idx_by_phase[phase] = len(events)

        row = {
            "phase": phase,
            "run_timestamp": self.run_timestamp,
            "global_step": int(global_step),
            "episode": int(self.episode_by_phase[phase]),
            "episode_step": int(self.episode_step_by_phase[phase]),
            "bias_case": str(info.get("bias_case", "")),
            "bias_case_matched": bool(info.get("bias_case_matched", False)),
            "case_sampling_attempts": int(info.get("case_sampling_attempts", 0)),
            **_action_trace_values(action_arr, action_temporal, applied_offsets),
            "action_hold_counter": int(action_temporal.get("action_hold_counter", 0)),
            "action_hold_steps": int(action_temporal.get("action_hold_steps", 1)),
            "reward": float(reward),
            "episode_return": float(self.episode_return_by_phase[phase]),
            "done": bool(done),
            **_rule_bias_trace_values(
                rule_bias,
                held_rule_bias,
                bool(bias_temporal.get("bias_changed", False)),
            ),
            "bias_hold_counter": int(bias_temporal.get("bias_hold_counter", 0)),
            "bias_hold_steps": int(bias_temporal.get("bias_hold_steps", 1)),
            **_slice_load_trace_values(slice_loads),
            **_count_trace_values(spawn_counts),
            **_pairwise_trace_values(info),
            **_tracking_trace_values(env, info, applied_offsets),
            "total_ues": total_ues,
            "ue_tracking": _ue_tracking_summary(env),
            "handover_attempts": int(handover_stats.get("attempts", 0)),
            "handover_successes": int(handover_stats.get("successes", 0)),
            "handover_failures": int(handover_stats.get("failures", 0)),
            "handover_ping_pongs": int(handover_stats.get("ping_pongs", 0)),
            "tracking_penalty": float(reward_breakdown.get("tracking_penalty", 0.0)),
            "sla_penalty": float(reward_breakdown.get("sla_penalty", 0.0)),
            "mobility_penalty": float(reward_breakdown.get("mobility_penalty", 0.0)),
            "smoothness_penalty": float(reward_breakdown.get("smoothness_penalty", 0.0)),
            "reward_total": float(reward_breakdown.get("total", reward)),
            "latest_handover": latest_handover,
            **radio,
        }
        self.writer.writerow(row)
        self.file.flush()

        if done:
            self.episode_by_phase[phase] += 1
            self.episode_step_by_phase[phase] = 0
            self.episode_return_by_phase[phase] = 0.0


class TraceCallback(BaseCallback):
    def __init__(self, logger: StepCsvLogger):
        super().__init__()
        self.csv_logger = logger

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        actions = self.locals.get("actions", [[0.0]])
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])
        self.csv_logger.log_step(
            env=env,
            phase="train",
            global_step=self.num_timesteps,
            action=np.asarray(actions)[0],
            reward=float(np.asarray(rewards).reshape(-1)[0]),
            done=bool(np.asarray(dones).reshape(-1)[0]),
            info=dict(infos[0]),
        )
        return True


def make_env(
    seed: int,
    episode_steps: int,
    action_hold_steps: int = 5,
    bias_hold_steps: int = 20,
    max_offset_change_db: float = 2.0,
    slice_types=TRACE_SLICE_TYPES,
    training_scenarios: Sequence[EpisodeTrainingScenario] | None = None,
) -> Monitor:
    env = LocalA3RuleBiasTrainingEnv(
        seed=seed,
        gnb_id=0,
        neighbor_ids=(1,),
        slice_types=tuple(slice_types),
        episode_steps=episode_steps,
        local_ues_range=(2, 3),
        neighbor_ues_range=(2, 3),
        steps_per_action=1,
        radio_substeps=10,
        balance_bias_cases=True,
        training_scenarios=training_scenarios,
        action_hold_steps=action_hold_steps,
        bias_hold_steps=bias_hold_steps,
        max_offset_change_db=max_offset_change_db,
    )
    return Monitor(env)


def _mean_or_zero(values) -> float:
    return float(np.mean(values)) if values else 0.0


def _write_eval_csv(path: Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate(
    model: TD3,
    seed: int,
    episodes: int,
    episode_steps: int,
    action_hold_steps: int = 5,
    bias_hold_steps: int = 20,
    max_offset_change_db: float = 2.0,
    slice_types=TRACE_SLICE_TYPES,
    training_scenarios: Sequence[EpisodeTrainingScenario] | None = None,
    logger: StepCsvLogger | None = None,
    eval_csv: Path | None = None,
) -> Dict[str, Any]:
    env = make_env(
        seed=seed,
        episode_steps=episode_steps,
        action_hold_steps=action_hold_steps,
        bias_hold_steps=bias_hold_steps,
        max_offset_change_db=max_offset_change_db,
        slice_types=slice_types,
        training_scenarios=training_scenarios,
    )
    returns = []
    lengths = []
    successes = []
    failures = []
    ping_pongs = []
    tracking_errors = []
    tracking_errors_by_slice = {slice_type: [] for slice_type in TRACE_SLICE_TYPES}
    offset_changes = []
    offset_hold_durations = []
    abs_offset_deltas = []
    applied_by_bias = {-1.0: [], 0.0: [], 1.0: []}
    applied_by_slice_bias = {
        slice_type: {-1.0: [], 0.0: [], 1.0: []}
        for slice_type in TRACE_SLICE_TYPES
    }
    eval_rows = []
    final_info: Dict[str, Any] = {}

    try:
        for episode in range(int(episodes)):
            obs, info = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0
            length = 0
            ep_successes = 0
            ep_failures = 0
            ep_ping_pongs = 0
            ep_attempts = 0
            ep_tracking_errors = []
            ep_tracking_errors_by_slice = {slice_type: [] for slice_type in TRACE_SLICE_TYPES}
            ep_offset_changes = 0
            ep_hold_durations = []
            current_hold_duration = 0
            ep_abs_offset_deltas = []
            ep_applied_by_bias = {-1.0: [], 0.0: [], 1.0: []}
            ep_applied_by_slice_bias = {
                slice_type: {-1.0: [], 0.0: [], 1.0: []}
                for slice_type in TRACE_SLICE_TYPES
            }

            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                total_reward += float(reward)
                length += 1
                if logger is not None:
                    logger.log_step(
                        env=env,
                        phase="eval",
                        global_step=episode * episode_steps + length,
                        action=action,
                        reward=float(reward),
                        done=done,
                        info=dict(info),
                    )

                handover_stats = dict(info.get("handover_stats", {}))
                ep_attempts += int(handover_stats.get("attempts", 0))
                ep_successes += int(handover_stats.get("successes", 0))
                ep_failures += int(handover_stats.get("failures", 0))
                ep_ping_pongs += int(handover_stats.get("ping_pongs", 0))

                applied_offsets = dict(info.get("applied_offsets", {}))
                rule_bias = dict(info.get("rule_bias", {}))
                for slice_type in TRACE_SLICE_TYPES:
                    applied_offset = float(
                        _lookup_tuple_dict(applied_offsets, (TRACE_NEIGHBOR_ID, slice_type), 0.0)
                    )
                    bias = float(_lookup_tuple_dict(rule_bias, (0, slice_type), 0.0))
                    desired_offset = _desired_offset_from_info(
                        env,
                        info,
                        neighbor_id=TRACE_NEIGHBOR_ID,
                        slice_type=slice_type,
                    )
                    tracking_error = abs(float(applied_offset - desired_offset))
                    ep_tracking_errors.append(tracking_error)
                    tracking_errors.append(tracking_error)
                    ep_tracking_errors_by_slice[slice_type].append(tracking_error)
                    tracking_errors_by_slice[slice_type].append(tracking_error)
                    if bias in ep_applied_by_bias:
                        ep_applied_by_bias[bias].append(applied_offset)
                        applied_by_bias[bias].append(applied_offset)
                        ep_applied_by_slice_bias[slice_type][bias].append(applied_offset)
                        applied_by_slice_bias[slice_type][bias].append(applied_offset)

                action_temporal = dict(info.get("action_temporal", {}))
                action_decision = bool(action_temporal.get("offset_changed", False))
                offset_deltas = dict(action_temporal.get("offset_deltas", {}))
                if offset_deltas:
                    step_abs_deltas = [
                        abs(float(value))
                        for value in offset_deltas.values()
                    ]
                else:
                    step_abs_deltas = [abs(float(action_temporal.get("offset_delta", 0.0)))]
                ep_abs_offset_deltas.extend(step_abs_deltas)
                abs_offset_deltas.extend(step_abs_deltas)
                if action_decision:
                    ep_offset_changes += 1
                    if current_hold_duration > 0:
                        ep_hold_durations.append(current_hold_duration)
                    current_hold_duration = 1
                else:
                    current_hold_duration += 1

            returns.append(total_reward)
            lengths.append(length)
            successes.append(ep_successes)
            failures.append(ep_failures)
            ping_pongs.append(ep_ping_pongs)
            if current_hold_duration > 0:
                ep_hold_durations.append(current_hold_duration)
                offset_hold_durations.extend(ep_hold_durations)
            offset_changes.append(ep_offset_changes)
            attempts = max(ep_attempts, 1)
            success_denominator = max(ep_successes, 1)
            eval_rows.append({
                "episode": int(episode),
                "episode_return": float(total_reward),
                "episode_length": int(length),
                "mean_applied_offset_bias_neg1": _mean_or_zero(ep_applied_by_bias[-1.0]),
                "mean_applied_offset_bias_0": _mean_or_zero(ep_applied_by_bias[0.0]),
                "mean_applied_offset_bias_pos1": _mean_or_zero(ep_applied_by_bias[1.0]),
                **{
                    f"mean_applied_offset_{slice_type}_bias_neg1": _mean_or_zero(
                        ep_applied_by_slice_bias[slice_type][-1.0]
                    )
                    for slice_type in TRACE_SLICE_TYPES
                },
                **{
                    f"mean_applied_offset_{slice_type}_bias_0": _mean_or_zero(
                        ep_applied_by_slice_bias[slice_type][0.0]
                    )
                    for slice_type in TRACE_SLICE_TYPES
                },
                **{
                    f"mean_applied_offset_{slice_type}_bias_pos1": _mean_or_zero(
                        ep_applied_by_slice_bias[slice_type][1.0]
                    )
                    for slice_type in TRACE_SLICE_TYPES
                },
                **{
                    f"average_tracking_error_{slice_type}": _mean_or_zero(
                        ep_tracking_errors_by_slice[slice_type]
                    )
                    for slice_type in TRACE_SLICE_TYPES
                },
                "handover_attempts": int(ep_attempts),
                "handover_successes": int(ep_successes),
                "handover_failures": int(ep_failures),
                "handover_ping_pongs": int(ep_ping_pongs),
                "handover_failure_rate": float(ep_failures / attempts),
                "ping_pong_rate": float(ep_ping_pongs / success_denominator),
                "average_tracking_error": _mean_or_zero(ep_tracking_errors),
                "offset_changes": int(ep_offset_changes),
                "mean_offset_hold_duration": _mean_or_zero(ep_hold_durations),
                "mean_abs_offset_delta": _mean_or_zero(ep_abs_offset_deltas),
                "max_abs_offset_delta": float(np.max(ep_abs_offset_deltas)) if ep_abs_offset_deltas else 0.0,
            })
            final_info = info
    finally:
        env.close()

    total_attempts = sum(int(row["handover_attempts"]) for row in eval_rows)
    total_successes = sum(int(row["handover_successes"]) for row in eval_rows)
    total_failures = sum(int(row["handover_failures"]) for row in eval_rows)
    total_ping_pongs = sum(int(row["handover_ping_pongs"]) for row in eval_rows)
    summary_row = {
        "episode": "mean",
        "episode_return": float(np.mean(returns)) if returns else 0.0,
        "episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_applied_offset_bias_neg1": _mean_or_zero(applied_by_bias[-1.0]),
        "mean_applied_offset_bias_0": _mean_or_zero(applied_by_bias[0.0]),
        "mean_applied_offset_bias_pos1": _mean_or_zero(applied_by_bias[1.0]),
        **{
            f"mean_applied_offset_{slice_type}_bias_neg1": _mean_or_zero(
                applied_by_slice_bias[slice_type][-1.0]
            )
            for slice_type in TRACE_SLICE_TYPES
        },
        **{
            f"mean_applied_offset_{slice_type}_bias_0": _mean_or_zero(
                applied_by_slice_bias[slice_type][0.0]
            )
            for slice_type in TRACE_SLICE_TYPES
        },
        **{
            f"mean_applied_offset_{slice_type}_bias_pos1": _mean_or_zero(
                applied_by_slice_bias[slice_type][1.0]
            )
            for slice_type in TRACE_SLICE_TYPES
        },
        **{
            f"average_tracking_error_{slice_type}": _mean_or_zero(
                tracking_errors_by_slice[slice_type]
            )
            for slice_type in TRACE_SLICE_TYPES
        },
        "handover_attempts": int(total_attempts),
        "handover_successes": int(total_successes),
        "handover_failures": int(total_failures),
        "handover_ping_pongs": int(total_ping_pongs),
        "handover_failure_rate": float(total_failures / max(total_attempts, 1)),
        "ping_pong_rate": float(total_ping_pongs / max(total_successes, 1)),
        "average_tracking_error": _mean_or_zero(tracking_errors),
        "offset_changes": float(np.mean(offset_changes)) if offset_changes else 0.0,
        "mean_offset_hold_duration": _mean_or_zero(offset_hold_durations),
        "mean_abs_offset_delta": _mean_or_zero(abs_offset_deltas),
        "max_abs_offset_delta": float(np.max(abs_offset_deltas)) if abs_offset_deltas else 0.0,
    }
    if eval_csv is not None:
        _write_eval_csv(Path(eval_csv), [*eval_rows, summary_row])

    return {
        "episodes": int(episodes),
        "mean_reward": float(np.mean(returns)) if returns else 0.0,
        "std_reward": float(np.std(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_handover_successes": float(np.mean(successes)) if successes else 0.0,
        "mean_handover_failures": float(np.mean(failures)) if failures else 0.0,
        "mean_ping_pongs": float(np.mean(ping_pongs)) if ping_pongs else 0.0,
        "mean_offset_changes_per_episode": float(np.mean(offset_changes)) if offset_changes else 0.0,
        "mean_offset_hold_duration": _mean_or_zero(offset_hold_durations),
        "mean_abs_offset_delta": _mean_or_zero(abs_offset_deltas),
        "max_abs_offset_delta": float(np.max(abs_offset_deltas)) if abs_offset_deltas else 0.0,
        "mean_applied_offset_when_bias_neg1": _mean_or_zero(applied_by_bias[-1.0]),
        "mean_applied_offset_when_bias_0": _mean_or_zero(applied_by_bias[0.0]),
        "mean_applied_offset_when_bias_pos1": _mean_or_zero(applied_by_bias[1.0]),
        "mean_applied_offset_by_slice_bias": {
            slice_type: {
                str(bias): _mean_or_zero(values)
                for bias, values in bias_map.items()
            }
            for slice_type, bias_map in applied_by_slice_bias.items()
        },
        "average_tracking_error_by_slice": {
            slice_type: _mean_or_zero(values)
            for slice_type, values in tracking_errors_by_slice.items()
        },
        "handover_failure_rate": float(total_failures / max(total_attempts, 1)),
        "ping_pong_rate": float(total_ping_pongs / max(total_successes, 1)),
        "average_tracking_error": _mean_or_zero(tracking_errors),
        "eval_csv": str(eval_csv) if eval_csv is not None else None,
        "final_applied_offsets": _stringify_keys(final_info.get("applied_offsets", {})),
        "final_rule_bias": _stringify_keys(final_info.get("rule_bias", {})),
        "final_spawn_counts": _stringify_keys(final_info.get("spawn_counts", {})),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train one local A3-offset agent with a simple rule-based fake global bias."
    )
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episode-steps", type=int, default=40)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--save-threshold", type=float, default=-250.0)
    parser.add_argument("--model-dir", type=Path, default=Path("models/local_a3_fake_bias"))
    parser.add_argument("--trace-csv", type=Path, default=None)
    parser.add_argument("--eval-csv", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--action-noise-sigma", type=float, default=0.25)
    parser.add_argument("--action-hold-steps", type=int, default=5)
    parser.add_argument("--bias-hold-steps", type=int, default=20)
    parser.add_argument("--max-offset-change-db", type=float, default=2.0)
    parser.add_argument("--slice-types", type=str, default=",".join(TRACE_SLICE_TYPES))
    parser.add_argument(
        "--scenario-set",
        choices=("default", "feasible_mixed"),
        default="default",
        help=(
            "Training scenario set. 'feasible_mixed' keeps two gNBs and all "
            "three slices, with controlled UE placement/load targets that make "
            "the intended per-slice offload/retain/neutral behavior clear."
        ),
    )
    parser.add_argument("--no-timestamp-run-dir", action="store_true")
    args = parser.parse_args()

    slice_types = tuple(
        item.strip()
        for item in str(args.slice_types).split(",")
        if item.strip()
    ) or TRACE_SLICE_TYPES
    training_scenarios = local_a3_training_scenario_set(args.scenario_set)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = (
        args.model_dir
        if args.no_timestamp_run_dir
        else args.model_dir / f"run_{run_timestamp}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    trace_csv = args.trace_csv or (model_dir / "training_trace.csv")
    eval_csv = args.eval_csv or (model_dir / "local_td3_stage1_eval.csv")
    step_logger = StepCsvLogger(trace_csv, run_timestamp=run_timestamp)

    # Fix #8: always close the CSV logger, even if training or eval raises.
    try:
        env = make_env(
            seed=args.seed,
            episode_steps=args.episode_steps,
            action_hold_steps=args.action_hold_steps,
            bias_hold_steps=args.bias_hold_steps,
            max_offset_change_db=args.max_offset_change_db,
            slice_types=slice_types,
            training_scenarios=training_scenarios,
        )
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=float(args.action_noise_sigma) * np.ones(n_actions),
        )

        model = TD3(
            "MlpPolicy",
            env,
            seed=args.seed,
            action_noise=action_noise,
            learning_rate=1e-3,
            buffer_size=100_000,
            learning_starts=min(500, max(10, args.timesteps // 10)),
            batch_size=128,
            gamma=0.95,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_kwargs={"net_arch": [64, 64]},
            verbose=1,
            device=args.device,
        )

        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=TraceCallback(step_logger),
                progress_bar=False,
            )
        finally:
            env.close()

        final_path = model_dir / "local_a3_td3_final.zip"
        model.save(final_path)

        metrics = evaluate(
            model=model,
            seed=args.seed + 10_000,
            episodes=args.eval_episodes,
            episode_steps=args.episode_steps,
            action_hold_steps=args.action_hold_steps,
            bias_hold_steps=args.bias_hold_steps,
            max_offset_change_db=args.max_offset_change_db,
            slice_types=slice_types,
            training_scenarios=training_scenarios,
            logger=step_logger,
            eval_csv=eval_csv,
        )
        metrics["run_timestamp"] = run_timestamp
        metrics["run_dir"] = str(model_dir)
        metrics["base_model_dir"] = str(args.model_dir)
        metrics["timesteps"] = int(args.timesteps)
        metrics["save_threshold"] = float(args.save_threshold)
        metrics["action_hold_steps"] = int(args.action_hold_steps)
        metrics["bias_hold_steps"] = int(args.bias_hold_steps)
        metrics["max_offset_change_db"] = float(args.max_offset_change_db)
        metrics["slice_types"] = list(slice_types)
        metrics["scenario_set"] = str(args.scenario_set)
        metrics["training_scenarios"] = [
            scenario.name for scenario in training_scenarios
        ]
        metrics["saved_final_model"] = str(final_path)
        metrics["trace_csv"] = str(trace_csv)

        expected_max_decisions = int(np.ceil(args.episode_steps / max(args.action_hold_steps, 1)))
        stability_checks = {
            "reward_threshold": bool(metrics["mean_reward"] >= args.save_threshold),
            "no_handover_failures": bool(metrics["mean_handover_failures"] <= 0.0),
            "no_ping_pongs": bool(metrics["mean_ping_pongs"] <= 0.0),
            "bias_neg1_offset_lt_minus3": bool(metrics["mean_applied_offset_when_bias_neg1"] < -3.0),
            "bias_0_offset_near_zero": bool(-1.0 <= metrics["mean_applied_offset_when_bias_0"] <= 1.0),
            "bias_pos1_offset_gt_3": bool(metrics["mean_applied_offset_when_bias_pos1"] > 3.0),
            "offset_delta_limited": bool(
                metrics["max_abs_offset_delta"] <= float(args.max_offset_change_db) + 1e-9
            ),
            "offset_changes_reasonable": bool(
                metrics["mean_offset_changes_per_episode"] <= float(expected_max_decisions)
            ),
        }
        metrics["stability_checks"] = stability_checks

        if all(stability_checks.values()):
            best_path = model_dir / "local_a3_td3_good_start.zip"
            model.save(best_path)
            metrics["saved_good_start_model"] = str(best_path)
            metrics["passed_threshold"] = True
        else:
            metrics["saved_good_start_model"] = None
            metrics["passed_threshold"] = False

        metrics_path = model_dir / "eval_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print(json.dumps(metrics, indent=2))

    finally:
        step_logger.close()


if __name__ == "__main__":
    main()
