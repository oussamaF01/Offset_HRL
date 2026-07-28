#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from local_a3_agent_wrapper import normalize_slice_type, quantize_a3_offset
from lower_rl_training_utils import DEFAULT_SCENARIOS, make_lower_env, save_json


def _json_safe(obj):
    if isinstance(obj, dict):
        return {
            "|".join(map(str, key)) if isinstance(key, tuple) else str(key): _json_safe(value)
            for key, value in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


class CsvUpperBiasPPOEnv(gym.Env):
    """Simulator-in-the-loop lower PPO env driven by the notebook CSV expert table.

    The CSV supplies the upper directional bias for all gNBs and the expert
    offsets for non-controlled gNBs. PPO controls one gNB's local A3 offsets.
    Each step applies all offsets, opens the safe-admission window, runs the
    real MultiGNBWrapper A3/safe-admission/radio step, then rewards the
    controlled gNB for reducing its own demanded load and SLA severity.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        csv_path: str | Path = "results/upper_heuristic_3gnb_baseline/upper_heuristic_3gnb_scenario_summary.csv",
        controlled_gnb_id: int = 1,
        seed: int = 7,
        episode_steps: int = 40,
        target_prefix: str = "first",
        load_reward_weight: float = 10.0,
        sla_reward_weight: float = 5.0,
        handover_reward_weight: float = 0.05,
        offset_penalty_weight: float = 0.01,
        admission_drain_steps: int = 40,
        stop_drain_on_idle: bool = False,
        print_scenarios: bool = False,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.rows = pd.read_csv(self.csv_path).reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No rows found in {self.csv_path}")
        self.controlled_gnb_id = int(controlled_gnb_id)
        self.seed_value = int(seed)
        self.episode_steps = max(1, int(episode_steps))
        self.target_prefix = str(target_prefix)
        self.load_reward_weight = float(load_reward_weight)
        self.sla_reward_weight = float(sla_reward_weight)
        self.handover_reward_weight = float(handover_reward_weight)
        self.offset_penalty_weight = float(offset_penalty_weight)
        self.admission_drain_steps = max(1, int(admission_drain_steps))
        self.stop_drain_on_idle = bool(stop_drain_on_idle)

        scenario_names = ",".join(str(name) for name in self.rows["scenario_name"].tolist())
        self.lower_env = make_lower_env(
            seed=self.seed_value,
            controlled_gnb_id=self.controlled_gnb_id,
            training_scenarios=scenario_names or DEFAULT_SCENARIOS,
            scenario_selection="cycle",
            episode_steps=self.episode_steps,
            action_hold_steps=1,
            bias_hold_steps=1,
            max_offset_change_db=12.0,
            max_handovers_per_local_step=3,
            print_scenarios=print_scenarios,
            scenario_mode="curriculum",
        )
        self.base_env = self.lower_env.base_env
        self.upper_env = self.lower_env.upper_env
        self.gnb_ids = tuple(int(gid) for gid in self.lower_env.gnb_ids)
        self.slice_types = tuple(normalize_slice_type(st) for st in self.lower_env.slice_types)
        self.neighbors = {
            int(gid): tuple(int(nb) for nb in self.lower_env.neighbors[int(gid)])
            for gid in self.gnb_ids
        }
        self.ctrl_env = self.lower_env.local_envs[self.controlled_gnb_id]
        self.action_space = self.ctrl_env.action_space
        self.observation_space = self.ctrl_env.observation_space
        self._row_index = -1
        self._row = None
        self._elapsed_steps = 0

    def reset(self, *, seed=None, options=None):
        self._row_index = (self._row_index + 1) % len(self.rows)
        self._row = self.rows.iloc[self._row_index]
        obs, info = self.lower_env.reset(seed=seed, options=options)
        self.base_env = self.lower_env.base_env
        self.upper_env = self.lower_env.upper_env
        self.ctrl_env = self.lower_env.local_envs[self.controlled_gnb_id]
        self._elapsed_steps = 0
        bias = self._bias_from_row(self._row)
        for local_env in self.lower_env.local_envs.values():
            local_env.set_global_bias(bias)
        info = dict(info or {})
        info.update({
            "csv_row_index": int(self._row_index),
            "csv_scenario_name": str(self._row["scenario_name"]),
            "scenario_name": str(self._row["scenario_name"]),
        })
        return self.ctrl_env._build_observation(), info

    def step(self, action):
        if self._row is None:
            raise RuntimeError("Call reset() before step().")

        before_load = self._demand_matrix()
        before_sla = self._sla_matrix()
        before_own_load = float(np.sum(before_load[self.controlled_gnb_id]))
        before_own_sla = float(np.sum(before_sla[self.controlled_gnb_id]))

        bias = self._bias_from_row(self._row)
        offsets = self._csv_offsets_from_row(self._row, prefix=self.target_prefix)
        ctrl_action = self._quantized_action(action)
        for value, (target_gnb, slice_type) in zip(ctrl_action, self.ctrl_env._iter_keys()):
            offsets[(self.controlled_gnb_id, int(target_gnb), normalize_slice_type(slice_type))] = float(value)

        demand_lookup = {
            (int(gid), normalize_slice_type(st)): float(before_load[gi, si])
            for gi, gid in enumerate(self.gnb_ids)
            for si, st in enumerate(self.slice_types)
        }
        previous_provider = getattr(self.base_env, "safe_admission_load_provider", None)
        self.base_env.safe_admission_load_provider = (
            lambda gid, st, _lookup=demand_lookup: float(
                _lookup.get((int(gid), normalize_slice_type(st)), 0.0)
            )
        )
        try:
            self._apply_bias_and_offsets(bias, offsets)
            start_events = len(getattr(self.base_env, "handover_events", []))
            last_events = start_events
            base_info = {}
            terminated = False
            truncated = False
            base_reward_total = 0.0
            drain_step_handovers = []
            drain_stop_reason = "max_drain_steps"

            for drain_step in range(self.admission_drain_steps):
                _obs, base_reward, terminated, truncated, base_info = self.base_env.step(0)
                base_reward_total += float(base_reward)
                current_events = len(getattr(self.base_env, "handover_events", []))
                step_handovers = int(current_events - last_events)
                drain_step_handovers.append(step_handovers)
                last_events = current_events

                if bool(terminated):
                    drain_stop_reason = "terminated"
                    break
                if bool(truncated):
                    drain_stop_reason = "base_truncated"
                    break
                if self._safe_admission_direction_remaining_total() <= 0:
                    drain_stop_reason = "safe_budget_exhausted"
                    break
                if self.stop_drain_on_idle and step_handovers <= 0:
                    drain_stop_reason = "idle"
                    break

            end_events = len(getattr(self.base_env, "handover_events", []))
        finally:
            self.base_env.safe_admission_load_provider = previous_provider

        after_load = self._demand_matrix()
        after_sla = self._sla_matrix()
        after_own_load = float(np.sum(after_load[self.controlled_gnb_id]))
        after_own_sla = float(np.sum(after_sla[self.controlled_gnb_id]))
        handovers = int(end_events - start_events)

        load_reduction = before_own_load - after_own_load
        sla_reduction = before_own_sla - after_own_sla
        offset_penalty = float(np.mean(np.abs(ctrl_action)) / 6.0)
        reward = (
            self.load_reward_weight * load_reduction
            + self.sla_reward_weight * sla_reduction
            + self.handover_reward_weight * handovers
            - self.offset_penalty_weight * offset_penalty
        )

        self._elapsed_steps += 1
        truncated = bool(truncated or self._elapsed_steps >= self.episode_steps)
        obs = self.ctrl_env._build_observation()
        info = dict(base_info or {})
        info.update({
            "scenario_name": str(self._row["scenario_name"]),
            "csv_row_index": int(self._row_index),
            "controlled_gnb_id": int(self.controlled_gnb_id),
            "reward_load_reduction": float(self.load_reward_weight * load_reduction),
            "reward_sla_reduction": float(self.sla_reward_weight * sla_reduction),
            "reward_handover": float(self.handover_reward_weight * handovers),
            "reward_offset_penalty": float(-self.offset_penalty_weight * offset_penalty),
            "own_load_before": before_own_load,
            "own_load_after": after_own_load,
            "own_sla_before": before_own_sla,
            "own_sla_after": after_own_sla,
            "handover_count": handovers,
            "admission_drain_steps": int(len(drain_step_handovers)),
            "admission_drain_max_steps": int(self.admission_drain_steps),
            "admission_drain_stop_reason": str(drain_stop_reason),
            "admission_drain_handovers": int(sum(drain_step_handovers)),
            "admission_drain_step_handovers_json": json.dumps(drain_step_handovers),
            "base_reward_total": float(base_reward_total),
            "applied_offsets": _json_safe(offsets),
        })
        if hasattr(self.base_env, "get_safe_admission_state"):
            safe_totals = self._safe_admission_totals()
            info["safe_admission"] = safe_totals["state"]
            info["safe_admission_enabled"] = safe_totals["enabled"]
            info["safe_admission_accepted_total"] = safe_totals["accepted_total"]
            info["safe_admission_remaining_total"] = safe_totals["remaining_total"]
            info["safe_admission_capacity_total"] = safe_totals["capacity_total"]
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.lower_env.close()

    def _bias_from_row(self, row) -> Dict[Tuple[int, int, str], float]:
        bias = {}
        for source_gnb in self.gnb_ids:
            for target_gnb in self.neighbors[int(source_gnb)]:
                for slice_type in self.slice_types:
                    col = f"first_upper_bias_g{source_gnb}_to_g{target_gnb}_{slice_type}"
                    bias[(int(source_gnb), int(target_gnb), slice_type)] = float(row.get(col, 0.0))
        return bias

    def _csv_offsets_from_row(self, row, *, prefix: str) -> Dict[Tuple[int, int, str], float]:
        offsets = {}
        for source_gnb in self.gnb_ids:
            for target_gnb in self.neighbors[int(source_gnb)]:
                for slice_type in self.slice_types:
                    col = f"{prefix}_offset_db_g{source_gnb}_to_g{target_gnb}_{slice_type}"
                    offsets[(int(source_gnb), int(target_gnb), slice_type)] = float(row.get(col, 0.0))
        return offsets

    def _quantized_action(self, action) -> np.ndarray:
        arr = np.asarray(action, dtype=float).reshape(-1)
        expected = int(np.prod(self.action_space.shape))
        if arr.size != expected:
            raise ValueError(f"Expected action size {expected}, got {arr.size}")
        return np.asarray([quantize_a3_offset(float(value)) for value in arr], dtype=np.float32)

    def _demand_matrix(self) -> np.ndarray:
        for method_name in ("_load_matrix", "_persistent_demand_load_matrix"):
            if hasattr(self.upper_env, method_name):
                arr = np.asarray(getattr(self.upper_env, method_name)(), dtype=float)
                if arr.shape == (len(self.gnb_ids), len(self.slice_types)):
                    return np.nan_to_num(arr, nan=0.0)
        profile = (
            self.base_env.get_demand_prb_loads()
            if hasattr(self.base_env, "get_demand_prb_loads")
            else getattr(self.base_env, "_last_demand_profile", {})
        )
        matrix = np.zeros((len(self.gnb_ids), len(self.slice_types)), dtype=float)
        for gi, gid in enumerate(self.gnb_ids):
            for si, st in enumerate(self.slice_types):
                item = profile.get((int(gid), normalize_slice_type(st)), {})
                if isinstance(item, dict):
                    matrix[gi, si] = float(item.get("target_load", 0.0))
        return matrix

    def _sla_matrix(self) -> np.ndarray:
        values = (
            self.base_env.get_slice_sla_severity()
            if hasattr(self.base_env, "get_slice_sla_severity")
            else {}
        )
        return np.asarray([
            [float(values.get((int(gid), st), 0.0)) for st in self.slice_types]
            for gid in self.gnb_ids
        ], dtype=float)

    def _bias_tensor(self, bias: Dict[Tuple[int, int, str], float]) -> np.ndarray:
        max_neighbors = max(len(v) for v in self.neighbors.values())
        tensor = np.zeros((len(self.gnb_ids), max_neighbors, len(self.slice_types)), dtype=float)
        for src_idx, source_gnb in enumerate(self.gnb_ids):
            for nb_slot, target_gnb in enumerate(self.neighbors[int(source_gnb)]):
                for s_idx, slice_type in enumerate(self.slice_types):
                    tensor[src_idx, nb_slot, s_idx] = float(
                        bias.get((int(source_gnb), int(target_gnb), slice_type), 0.0)
                    )
        return tensor

    @staticmethod
    def _sum_positive_ints(value) -> int:
        if not isinstance(value, dict):
            return 0
        return int(sum(max(int(v), 0) for v in value.values()))

    def _safe_admission_totals(self) -> Dict:
        state = self.base_env.get_safe_admission_state() if hasattr(self.base_env, "get_safe_admission_state") else {}
        accepted = state.get("accepted", {}) if isinstance(state, dict) else {}
        remaining = state.get("remaining", {}) if isinstance(state, dict) else {}
        capacities = state.get("capacities", {}) if isinstance(state, dict) else {}
        return {
            "enabled": bool(state.get("enabled", False)) if isinstance(state, dict) else False,
            "accepted_total": self._sum_positive_ints(accepted),
            "remaining_total": self._sum_positive_ints(remaining),
            "capacity_total": self._sum_positive_ints(capacities),
            "state": state,
        }

    def _safe_admission_direction_remaining_total(self) -> int:
        return int(self._safe_admission_totals()["remaining_total"])

    def _apply_bias_and_offsets(self, bias, offsets) -> None:
        for local_env in self.lower_env.local_envs.values():
            local_env.set_global_bias(bias)
        self.base_env.begin_safe_admission_window(self._bias_tensor(bias), self.slice_types)
        self.base_env.begin_radio_measurement_window()
        if hasattr(self.base_env, "begin_sla_window"):
            self.base_env.begin_sla_window()
        for source_gnb in self.gnb_ids:
            local_env = self.lower_env.local_envs[int(source_gnb)]
            vec = []
            for target_gnb, slice_type in local_env._iter_keys():
                value = offsets.get((int(source_gnb), int(target_gnb), normalize_slice_type(slice_type)), 0.0)
                vec.append(float(value))
                self.base_env.set_a3_offset(int(source_gnb), int(target_gnb), slice_type, float(value))
            local_env._apply_proto_offsets(np.asarray(vec, dtype=float))


def make_env(args) -> Monitor:
    env = CsvUpperBiasPPOEnv(
        csv_path=args.csv,
        controlled_gnb_id=args.controlled_gnb_id,
        seed=args.seed,
        episode_steps=args.episode_steps,
        target_prefix=args.target_prefix,
        load_reward_weight=args.load_reward_weight,
        sla_reward_weight=args.sla_reward_weight,
        handover_reward_weight=args.handover_reward_weight,
        offset_penalty_weight=args.offset_penalty_weight,
        admission_drain_steps=args.admission_drain_steps,
        stop_drain_on_idle=args.stop_drain_on_idle,
        print_scenarios=args.print_scenarios,
    )
    return Monitor(env)


class RewardCsvLoggerCallback(BaseCallback):
    """Collect per-step reward components from the env info dict."""

    COLUMNS = [
        "timestep",
        "env_index",
        "done",
        "scenario_name",
        "csv_row_index",
        "controlled_gnb_id",
        "reward",
        "reward_load_reduction",
        "reward_sla_reduction",
        "reward_handover",
        "reward_offset_penalty",
        "own_load_before",
        "own_load_after",
        "own_sla_before",
        "own_sla_after",
        "handover_count",
        "admission_drain_steps",
        "admission_drain_stop_reason",
        "admission_drain_handovers",
        "safe_admission_enabled",
        "safe_admission_accepted_total",
        "safe_admission_remaining_total",
        "safe_admission_capacity_total",
        "base_reward_total",
        "admission_drain_step_handovers_json",
    ]

    def __init__(self, out_path: str | Path | None, *, save_freq: int = 32):
        super().__init__()
        self.out_path = Path(out_path) if out_path else None
        self.save_freq = max(1, int(save_freq))
        self.rows = []

    def _flush(self) -> None:
        if self.out_path is None:
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows, columns=self.COLUMNS).to_csv(self.out_path, index=False)

    def _on_training_start(self) -> None:
        self._flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        for env_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            reward = float(rewards[env_idx]) if env_idx < len(rewards) else 0.0
            done = bool(dones[env_idx]) if env_idx < len(dones) else False
            self.rows.append({
                "timestep": int(self.num_timesteps),
                "env_index": int(env_idx),
                "done": done,
                "scenario_name": str(info.get("scenario_name", "unknown")),
                "csv_row_index": int(info.get("csv_row_index", -1)),
                "controlled_gnb_id": int(info.get("controlled_gnb_id", -1)),
                "reward": reward,
                "reward_load_reduction": float(info.get("reward_load_reduction", 0.0)),
                "reward_sla_reduction": float(info.get("reward_sla_reduction", 0.0)),
                "reward_handover": float(info.get("reward_handover", 0.0)),
                "reward_offset_penalty": float(info.get("reward_offset_penalty", 0.0)),
                "own_load_before": float(info.get("own_load_before", 0.0)),
                "own_load_after": float(info.get("own_load_after", 0.0)),
                "own_sla_before": float(info.get("own_sla_before", 0.0)),
                "own_sla_after": float(info.get("own_sla_after", 0.0)),
                "handover_count": int(info.get("handover_count", 0)),
                "admission_drain_steps": int(info.get("admission_drain_steps", 0)),
                "admission_drain_stop_reason": str(info.get("admission_drain_stop_reason", "unknown")),
                "admission_drain_handovers": int(info.get("admission_drain_handovers", 0)),
                "safe_admission_enabled": bool(info.get("safe_admission_enabled", False)),
                "safe_admission_accepted_total": int(info.get("safe_admission_accepted_total", 0)),
                "safe_admission_remaining_total": int(info.get("safe_admission_remaining_total", 0)),
                "safe_admission_capacity_total": int(info.get("safe_admission_capacity_total", 0)),
                "base_reward_total": float(info.get("base_reward_total", 0.0)),
                "admission_drain_step_handovers_json": str(
                    info.get("admission_drain_step_handovers_json", "[]")
                ),
            })
        if self.out_path is not None and self.n_calls % self.save_freq == 0:
            self._flush()
        return True

    def _on_rollout_end(self) -> None:
        self._flush()

    def _on_training_end(self) -> None:
        self._flush()


def evaluate(model: PPO, args, episodes: int) -> Dict:
    env = make_env(args)
    rows = []
    try:
        for episode in range(int(episodes)):
            obs, info = env.reset(seed=args.seed + 10_000 + episode)
            done = False
            episode_return = 0.0
            final_info = dict(info)
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                done = bool(terminated or truncated)
                final_info = dict(info)
            rows.append({
                "episode": int(episode),
                "scenario_name": str(final_info.get("scenario_name", "unknown")),
                "episode_return": float(episode_return),
                "own_load_after": float(final_info.get("own_load_after", 0.0)),
                "own_sla_after": float(final_info.get("own_sla_after", 0.0)),
                "handover_count_last_step": int(final_info.get("handover_count", 0)),
                "admission_drain_steps_last_step": int(final_info.get("admission_drain_steps", 0)),
                "admission_drain_stop_reason_last_step": str(
                    final_info.get("admission_drain_stop_reason", "unknown")
                ),
                "safe_admission_accepted_total_last_step": int(
                    final_info.get("safe_admission_accepted_total", 0)
                ),
                "safe_admission_remaining_total_last_step": int(
                    final_info.get("safe_admission_remaining_total", 0)
                ),
            })
    finally:
        env.close()
    return {
        "episodes": rows,
        "mean_return": float(np.mean([row["episode_return"] for row in rows])) if rows else 0.0,
        "mean_final_own_load": float(np.mean([row["own_load_after"] for row in rows])) if rows else 0.0,
        "mean_final_own_sla": float(np.mean([row["own_sla_after"] for row in rows])) if rows else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO lower A3 offsets in the real simulator using CSV upper bias/expert neighbor offsets."
    )
    parser.add_argument("--csv", default="results/upper_heuristic_3gnb_baseline/upper_heuristic_3gnb_scenario_summary.csv")
    parser.add_argument("--controlled-gnb-id", type=int, default=1)
    parser.add_argument("--total-timesteps", "--timesteps", dest="total_timesteps", type=int, default=20_000)
    parser.add_argument("--episode-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-prefix", choices=("first", "final"), default="first")
    parser.add_argument("--model-dir", type=Path, default=Path("models/csv_offset_ppo"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--reward-log", type=Path, default=None)
    parser.add_argument("--reward-log-save-freq", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--load-reward-weight", type=float, default=10.0)
    parser.add_argument("--sla-reward-weight", type=float, default=5.0)
    parser.add_argument("--handover-reward-weight", type=float, default=0.05)
    parser.add_argument("--offset-penalty-weight", type=float, default=0.01)
    parser.add_argument("--admission-drain-steps", type=int, default=40)
    parser.add_argument(
        "--stop-drain-on-idle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop a PPO decision's inner simulator drain when a tick admits no handovers.",
    )
    parser.add_argument("--print-scenarios", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = args.model_dir / f"run_{timestamp}"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir = args.run_dir
    if args.reward_log is None:
        args.reward_log = args.run_dir / "csv_offset_ppo_reward_log.csv"

    env = make_env(args)
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        gamma=0.95,
        gae_lambda=0.90,
        clip_range=0.20,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
        device=args.device,
    )
    try:
        reward_logger = RewardCsvLoggerCallback(
            args.reward_log,
            save_freq=args.reward_log_save_freq,
        )
        model.learn(
            total_timesteps=int(args.total_timesteps),
            progress_bar=False,
            callback=reward_logger,
        )
    finally:
        env.close()

    model_path = args.model_dir / f"csv_offset_ppo_g{args.controlled_gnb_id}.zip"
    model.save(model_path)
    metrics = evaluate(model, args, args.eval_episodes)
    metrics.update({
        "model_path": str(model_path),
        "run_dir": str(args.run_dir),
        "csv_path": str(args.csv),
        "controlled_gnb_id": int(args.controlled_gnb_id),
        "total_timesteps": int(args.total_timesteps),
        "reward_log": str(args.reward_log),
        "reward_log_save_freq": int(args.reward_log_save_freq),
        "admission_drain_steps": int(args.admission_drain_steps),
        "stop_drain_on_idle": bool(args.stop_drain_on_idle),
        "reward": (
            "load_weight*(own_load_before-own_load_after) + "
            "sla_weight*(own_sla_before-own_sla_after) + "
            "handover_weight*handovers - offset_penalty*mean_abs_offset/6"
        ),
    })
    metrics_path = args.model_dir / f"csv_offset_ppo_g{args.controlled_gnb_id}_metrics.json"
    save_json(metrics_path, metrics)
    print(f"saved PPO model to {model_path}")
    print(f"saved metrics to {metrics_path}")
    print(f"saved reward log to {args.reward_log}")
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
