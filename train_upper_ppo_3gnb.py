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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from global_ppo_3gnb_env import GlobalPPO3GNBEnv, SLICE_TYPES


GNB_IDS = (0, 1, 2)
MAX_NEIGHBORS = 2
ACTION_FIELDS = [f"action_{idx}" for idx in range(len(GNB_IDS) * MAX_NEIGHBORS * len(SLICE_TYPES))]
OBSERVATION_FIELDS = [f"obs_{idx}" for idx in range(len(GNB_IDS) * len(SLICE_TYPES) * 6)]
MATRIX_FIELDS = [
    f"{prefix}_g{gnb_id}_{slice_type}"
    for prefix in ("bias", "target_load", "balance_target", "load", "sla", "ue_count")
    for gnb_id in GNB_IDS
    for slice_type in SLICE_TYPES
]

TRAINING_FIELDS = [
    "step",
    "episode",
    "episode_step",
    "reward",
    "episode_return",
    "done",
    "scenario_name",
    "load_variance",
    "target_load_error",
    "overload_ratio",
    "sla_count",
    "sla_severity",
    "sla_deadband",
    "handover_count",
    "load_imbalance_start",
    "load_imbalance_end",
    "target_load_error_start",
    "target_load_error_end",
    "instant_reward_mean",
    "dense_window_reward",
    "episode_terminal_reward",
    "global_network_cost",
    "global_cost_start",
    "global_cost_end",
    "global_cost_improvement",
    "global_action_penalty",
    "global_bad_direction_penalty",
    "saturation_count",
    "action_direction_reward",
    "terminal_reward_only",
    "use_progress_reward",
    "overloaded_negative_fraction",
    "light_nonnegative_fraction",
    "bias_matrix",
    "directional_bias_tensor",
    "target_load_matrix",
    "balance_target_matrix",
    "load_matrix",
    "sla_matrix",
    "ue_count_matrix",
] + ACTION_FIELDS + OBSERVATION_FIELDS + MATRIX_FIELDS

VALIDATION_FIELDS = [
    "episode",
    "step",
    "scenario_name",
    "reward",
    "load_variance",
    "target_load_error",
    "load_imbalance_start",
    "load_imbalance_end",
    "target_load_error_start",
    "target_load_error_end",
    "overload_ratio",
    "sla_count",
    "sla_severity",
    "sla_deadband",
    "handover_count",
    "global_network_cost",
    "global_cost_start",
    "global_cost_end",
    "global_cost_improvement",
    "global_action_penalty",
    "global_bad_direction_penalty",
    "saturation_count",
    "action_direction_reward",
    "overloaded_negative_fraction",
    "light_nonnegative_fraction",
    "bias_matrix",
    "directional_bias_tensor",
    "target_load_matrix",
    "balance_target_matrix",
    "load_matrix",
    "sla_matrix",
    "ue_count_matrix",
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
    bias = _matrix_or_nan(bias_matrix)
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


class UpperTrainingCsvCallback(BaseCallback):
    def __init__(self, log_path: Path, best_path: Path | None = None):
        super().__init__()
        self.log_path = Path(log_path)
        self.best_path = Path(best_path) if best_path is not None else None
        self.file = None
        self.writer = None
        self.episode = 0
        self.episode_step = 0
        self.episode_return = 0.0
        self.best_episode_return = -np.inf

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.log_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=TRAINING_FIELDS)
        self.writer.writeheader()

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", [0.0])).reshape(-1)
        dones = np.asarray(self.locals.get("dones", [False])).reshape(-1)
        action_values = _flat_first(self.locals.get("actions", []), len(ACTION_FIELDS))
        obs_values = _flat_first(self.locals.get("new_obs", []), len(OBSERVATION_FIELDS))
        infos = self.locals.get("infos", [{}])
        info = dict(infos[0])
        reward = float(rewards[0])
        done = bool(dones[0])
        overloaded_negative, light_nonnegative = _bias_quality_scores(
            info.get("bias_matrix", []),
            info.get("load_matrix", []),
        )

        self.episode_step += 1
        self.episode_return += reward
        row = {
            "step": int(self.num_timesteps),
            "episode": int(self.episode),
            "episode_step": int(self.episode_step),
            "reward": reward,
            "episode_return": float(self.episode_return),
            "done": done,
            "scenario_name": str(info.get("scenario_name", "")),
            "load_variance": float(info.get("load_variance", 0.0)),
            "target_load_error": float(info.get("target_load_error", 0.0)),
            "overload_ratio": float(info.get("overload_ratio", 0.0)),
            "sla_count": float(info.get("sla_count", 0.0)),
            "sla_severity": float(info.get("sla_severity", 0.0)),
            "sla_deadband": float(info.get("sla_deadband", 0.0)),
            "handover_count": int(info.get("handover_count", 0)),
            "load_imbalance_start": float(info.get("load_imbalance_start", 0.0)),
            "load_imbalance_end": float(info.get("load_imbalance_end", 0.0)),
            "target_load_error_start": float(info.get("target_load_error_start", 0.0)),
            "target_load_error_end": float(info.get("target_load_error_end", 0.0)),
            "instant_reward_mean": float(info.get("instant_reward_mean", 0.0)),
            "dense_window_reward": float(info.get("dense_window_reward", 0.0)),
            "episode_terminal_reward": float(info.get("episode_terminal_reward", 0.0)),
            "global_network_cost": float(info.get("global_network_cost", 0.0)),
            "global_cost_start": float(info.get("global_cost_start", 0.0)),
            "global_cost_end": float(info.get("global_cost_end", 0.0)),
            "global_cost_improvement": float(info.get("global_cost_improvement", 0.0)),
            "global_action_penalty": float(info.get("global_action_penalty", 0.0)),
            "global_bad_direction_penalty": float(info.get("global_bad_direction_penalty", 0.0)),
            "saturation_count": int(info.get("saturation_count", 0)),
            "action_direction_reward": float(info.get("action_direction_reward", 0.0)),
            "terminal_reward_only": bool(info.get("terminal_reward_only", False)),
            "use_progress_reward": bool(info.get("use_progress_reward", False)),
            "overloaded_negative_fraction": overloaded_negative,
            "light_nonnegative_fraction": light_nonnegative,
            "bias_matrix": _json_array(info.get("bias_matrix", [])),
            "directional_bias_tensor": _json_array(info.get("directional_bias_tensor", [])),
            "target_load_matrix": _json_array(info.get("target_load_matrix", [])),
            "balance_target_matrix": _json_array(info.get("balance_target_matrix", [])),
            "load_matrix": _json_array(info.get("load_matrix", [])),
            "sla_matrix": _json_array(info.get("sla_matrix", [])),
            "ue_count_matrix": _json_array(info.get("ue_count_matrix", [])),
        }
        for idx, value in enumerate(action_values):
            row[f"action_{idx}"] = float(value)
        for idx, value in enumerate(obs_values):
            row[f"obs_{idx}"] = float(value)
        for prefix, key in (
            ("bias", "bias_matrix"),
            ("target_load", "target_load_matrix"),
            ("balance_target", "balance_target_matrix"),
            ("load", "load_matrix"),
            ("sla", "sla_matrix"),
            ("ue_count", "ue_count_matrix"),
        ):
            _add_flat_matrix_fields(row, prefix, info.get(key, []))
        self.writer.writerow(row)
        self.file.flush()

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
            self.file.close()
            self.file = None


def make_env(args) -> Monitor:
    env = GlobalPPO3GNBEnv(
        seed=args.seed,
        n_gnbs=args.n_gnbs,
        slice_types=SLICE_TYPES,
        include_ue_counts=args.include_ue_counts,
        include_service_metrics=args.include_service_metrics,
        use_sumo_mobility=args.use_sumo_mobility,
        local_steps_per_global=args.local_steps_per_global,
        global_steps_per_episode=args.global_steps_per_episode,
        scenario_mode=args.scenario_mode,
        snapshot_scenario=args.snapshot_scenario,
        terminal_reward_only=not args.dense_window_reward,
        use_progress_reward=args.use_progress_reward,
        max_handovers_per_local_step=args.max_handovers_per_local_step,
        action_direction_reward_weight=args.action_direction_reward_weight,
        snapshot_block_episodes=args.snapshot_block_episodes,
        light_load_ues=args.light_load_ues,
        medium_load_ues=args.medium_load_ues,
        high_load_ues=args.high_load_ues,
        print_scenarios=args.debug,
        slice_prb_budgets=args.slice_prb_budgets,
        max_prbs_per_ue=args.max_prbs_per_ue,
        directional_global_action=args.directional_global_action,
        sla_deadband=args.sla_deadband,
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
                first_imbalance = float(info.get("load_imbalance_start", 0.0))
            last_imbalance = float(info.get("load_imbalance_end", 0.0))
            handovers.append(int(info.get("handover_count", 0)))
            sla_counts.append(float(info.get("sla_count", 0.0)))
            bias_matrix = np.asarray(info.get("bias_matrix", []), dtype=float)
            load_matrix = np.asarray(info.get("load_matrix", []), dtype=float)
            overloaded = load_matrix > 0.85
            light = load_matrix < 0.45
            overloaded_negative = (
                float(np.mean(bias_matrix[overloaded] < 0.0))
                if overloaded.any() and bias_matrix.shape == load_matrix.shape
                else 1.0
            )
            light_nonnegative = (
                float(np.mean(bias_matrix[light] >= -0.1))
                if light.any() and bias_matrix.shape == load_matrix.shape
                else 1.0
            )
            overloaded_negative_scores.append(overloaded_negative)
            light_nonnegative_scores.append(light_nonnegative)

            rows.append({
                "episode": int(episode),
                "step": int(step),
                "scenario_name": str(info.get("scenario_name", "")),
                "reward": float(reward),
                "load_variance": float(info.get("load_variance", 0.0)),
                "target_load_error": float(info.get("target_load_error", 0.0)),
                "load_imbalance_start": float(info.get("load_imbalance_start", 0.0)),
                "load_imbalance_end": float(info.get("load_imbalance_end", 0.0)),
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
                "global_bad_direction_penalty": float(info.get("global_bad_direction_penalty", 0.0)),
                "saturation_count": int(info.get("saturation_count", 0)),
                "action_direction_reward": float(info.get("action_direction_reward", 0.0)),
                "overloaded_negative_fraction": overloaded_negative,
                "light_nonnegative_fraction": light_nonnegative,
                "bias_matrix": _json_array(info.get("bias_matrix", [])),
                "directional_bias_tensor": _json_array(info.get("directional_bias_tensor", [])),
                "target_load_matrix": _json_array(info.get("target_load_matrix", [])),
                "balance_target_matrix": _json_array(info.get("balance_target_matrix", [])),
                "load_matrix": _json_array(info.get("load_matrix", [])),
                "sla_matrix": _json_array(info.get("sla_matrix", [])),
                "ue_count_matrix": _json_array(info.get("ue_count_matrix", [])),
            })

        episode_returns.append(ep_return)
        if first_imbalance is not None and last_imbalance is not None:
            target_error_deltas.append(first_imbalance - last_imbalance)

    if validation_csv is not None:
        validation_csv = Path(validation_csv)
        validation_csv.parent.mkdir(parents=True, exist_ok=True)
        with validation_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=VALIDATION_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

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


def main():
    parser = argparse.ArgumentParser(description="Train Phase-2 upper/global PPO for 3-gNB HRL.")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-dir", type=Path, default=Path("models/upper_ppo_3gnb"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-sumo-mobility", action="store_true")
    parser.add_argument("--include-ue-counts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-service-metrics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--directional-global-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--slice-prb-budgets",
        type=json.loads,
        default=None,
        help='Optional JSON dict, for example \'{"eMBB": 50, "URLLC": 50, "mMTC": 50}\'.',
    )
    parser.add_argument("--max-prbs-per-ue", type=int, default=20)
    parser.add_argument(
        "--sla-deadband",
        type=float,
        default=0.05,
        help="Ignore SLA violation magnitudes at or below this value in the upper reward.",
    )
    parser.add_argument("--n-gnbs", type=int, default=3)
    parser.add_argument("--local-steps-per-global", type=int, default=10)
    parser.add_argument("--global-steps-per-episode", type=int, default=12)
    parser.add_argument(
        "--max-handovers-per-local-step",
        type=int,
        default=1,
        help="Limit heuristic handovers during each lower/local simulator step.",
    )
    parser.add_argument(
        "--action-direction-reward-weight",
        type=float,
        default=2.0,
        help=(
            "Reward weight for matching upper action signs to snapshot targets: "
            "loaded slices prefer negative bias, light slices prefer positive bias, neutral slices prefer near-zero bias."
        ),
    )
    parser.add_argument(
        "--scenario-mode",
        choices=("snapshot", "random"),
        default="snapshot",
        help="Use fixed snapshot load scenarios or random mixed load scenarios.",
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
        default=10,
        help="When snapshot_scenario=mixed, keep each snapshot for this many episodes before cycling.",
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
    parser.add_argument(
        "--dense-window-reward",
        action="store_true",
        help="Return a reward after every upper window instead of only at episode end.",
    )
    parser.add_argument(
        "--use-progress-reward",
        action="store_true",
        help="Add target-load-error progress shaping. Disabled by default for snapshot training.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(args.model_dir) / f"run_{run_timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    training_csv = model_dir / "training_log.csv"
    validation_csv = model_dir / "validation_log.csv"
    final_model_path = model_dir / "upper_ppo_final.zip"
    best_model_path = model_dir / "upper_ppo_best.zip"
    config_path = model_dir / "config.json"

    config = {
        "run_timestamp": run_timestamp,
        "total_timesteps": int(args.total_timesteps),
        "seed": int(args.seed),
        "model_dir": str(model_dir),
        "device": str(args.device),
        "use_sumo_mobility": bool(args.use_sumo_mobility),
        "include_ue_counts": bool(args.include_ue_counts),
        "include_service_metrics": bool(args.include_service_metrics),
        "directional_global_action": bool(args.directional_global_action),
        "slice_prb_budgets": args.slice_prb_budgets,
        "max_prbs_per_ue": None if args.max_prbs_per_ue is None else int(args.max_prbs_per_ue),
        "sla_deadband": float(args.sla_deadband),
        "n_gnbs": int(args.n_gnbs),
        "slice_types": list(SLICE_TYPES),
        "local_steps_per_global": int(args.local_steps_per_global),
        "global_steps_per_episode": int(args.global_steps_per_episode),
        "max_handovers_per_local_step": int(args.max_handovers_per_local_step),
        "action_direction_reward_weight": float(args.action_direction_reward_weight),
        "scenario_mode": str(args.scenario_mode),
        "snapshot_scenario": str(args.snapshot_scenario),
        "snapshot_block_episodes": int(args.snapshot_block_episodes),
        "light_load_ues": int(args.light_load_ues),
        "medium_load_ues": int(args.medium_load_ues),
        "high_load_ues": int(args.high_load_ues),
        "terminal_reward_only": bool(not args.dense_window_reward),
        "use_progress_reward": bool(args.use_progress_reward),
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
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=args.device,
            seed=args.seed,
        )
        callback = UpperTrainingCsvCallback(training_csv, best_model_path)
        model.learn(total_timesteps=int(args.total_timesteps), callback=callback, progress_bar=False)
        model.save(final_model_path)
    finally:
        env.close()

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
        "validation": validation,
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
