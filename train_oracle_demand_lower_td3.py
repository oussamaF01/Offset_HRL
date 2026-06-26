#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from global_ppo_3gnb_env import SLICE_TYPES
from oracle_demand_lower_a3_env import OracleDemandLowerA3Env


SINGLE_SLICE_PHASE_SCENARIOS = (
    "jain_balance_controllable",
    "jain_control_urllc",
    "jain_control_mmtc",
)
MIXED_PHASE_SCENARIO = "jain_control_mixed"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _json_cell(value) -> str:
    return json.dumps(_jsonable(value), separators=(",", ":"))


class LowerTrainingCSVCallback(BaseCallback):
    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = Path(log_path)
        self.phase_name = ""
        self.phase_scenario = ""
        self.episode_index = 0
        self.episode_return = 0.0
        self._fieldnames = [
            "timestep",
            "phase",
            "phase_scenario",
            "episode",
            "done",
            "scenario_name",
            "controlled_gnb",
            "control_interval_index",
            "reward",
            "direction_reward",
            "demand_std_reward",
            "smoothness_penalty",
            "start_demand_std",
            "end_demand_std",
            "demand_std_improvement",
            "start_fixed_demand_loads",
            "end_fixed_demand_loads",
            "qos_slice_sla_metrics",
            "qos_slice_sla_flags",
            "qos_slice_sla_severity",
            "controlled_handover_attempts",
            "controlled_handover_successes",
            "controlled_handover_failures",
            "controlled_handover_ping_pongs",
            "controlled_offset_labels",
            "agent_raw_action",
            "agent_proto_offsets",
            "agent_applied_offsets",
            "controlled_oracle_bias",
            "controlled_perfect_offsets",
        ]

    def _init_callback(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        if not infos:
            return True
        info = dict(infos[0] or {})
        reward = float(rewards[0]) if len(rewards) else float(info.get("reward", 0.0))
        done = bool(dones[0]) if len(dones) else False
        self.episode_return += reward
        parts = dict(info.get("reward_parts", {}))
        qos = dict(info.get("qos", {}))
        controlled_gnb = int(info.get("controlled_gnb", -1))
        handover_stats = dict(info.get("handover_stats", {}))
        controlled_stats = dict(handover_stats.get(controlled_gnb, {}))
        row = {
            "timestep": int(self.num_timesteps),
            "phase": self.phase_name,
            "phase_scenario": self.phase_scenario,
            "episode": int(self.episode_index),
            "done": int(done),
            "scenario_name": str(info.get("scenario_name", self.phase_scenario)),
            "controlled_gnb": controlled_gnb,
            "control_interval_index": int(info.get("control_interval_index", 0)),
            "reward": reward,
            "direction_reward": float(parts.get("direction_reward", 0.0)),
            "demand_std_reward": float(parts.get("demand_std_reward", 0.0)),
            "smoothness_penalty": float(parts.get("smoothness_penalty", 0.0)),
            "start_demand_std": float(parts.get("start_demand_std", 0.0)),
            "end_demand_std": float(parts.get("end_demand_std", 0.0)),
            "demand_std_improvement": float(parts.get("demand_std_improvement", 0.0)),
            "start_fixed_demand_loads": _json_cell(info.get("demand_load_matrix_start", [])),
            "end_fixed_demand_loads": _json_cell(info.get("demand_load_matrix_end", [])),
            "qos_slice_sla_metrics": _json_cell(qos.get("slice_sla_metrics", {})),
            "qos_slice_sla_flags": _json_cell(qos.get("slice_sla_flags", {})),
            "qos_slice_sla_severity": _json_cell(qos.get("slice_sla_severity", {})),
            "controlled_handover_attempts": int(controlled_stats.get("attempts", 0)),
            "controlled_handover_successes": int(controlled_stats.get("successes", 0)),
            "controlled_handover_failures": int(controlled_stats.get("failures", 0)),
            "controlled_handover_ping_pongs": int(controlled_stats.get("ping_pongs", 0)),
            "controlled_offset_labels": _json_cell(info.get("controlled_offset_labels", [])),
            "agent_raw_action": _json_cell(info.get("controlled_raw_action", [])),
            "agent_proto_offsets": _json_cell(info.get("controlled_proto_offsets", [])),
            "agent_applied_offsets": _json_cell(info.get("controlled_applied_offsets", [])),
            "controlled_oracle_bias": _json_cell(info.get("controlled_oracle_bias", [])),
            "controlled_perfect_offsets": _json_cell(info.get("controlled_perfect_offsets", [])),
        }
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writerow(row)
        if done:
            self.episode_index += 1
            self.episode_return = 0.0
        return True


def make_env(args, seed: int, scenario: str | None = None) -> Monitor:
    env = OracleDemandLowerA3Env(
        seed=seed,
        controlled_gnb_id=args.controlled_gnb,
        slice_types=tuple(args.slice_types),
        training_scenarios=args.scenario if scenario is None else scenario,
        scenario_selection=args.scenario_selection,
        topology=args.topology,
        upper_window_seconds=args.upper_window_seconds,
        local_steps_per_global=args.local_steps_per_global,
        radio_substeps=args.radio_substeps,
        control_interval_steps=args.control_interval_steps,
        episode_control_intervals=(
            None if args.episode_control_intervals <= 0 else args.episode_control_intervals
        ),
        warmup_steps=args.warmup_steps,
        safe_admission_enabled=not args.disable_safe_admission,
        load_reward_weight=args.load_reward_weight,
        offset_imitation_weight=args.offset_imitation_weight,
        bias_alignment_weight=args.bias_alignment_weight,
        smoothness_weight=args.smoothness_weight,
        handover_failure_weight=args.handover_failure_weight,
        pingpong_weight=args.pingpong_weight,
    )
    return Monitor(env)


def evaluate(model: TD3, args, seed: int, scenario: str | None = None) -> Dict[str, Any]:
    env = make_env(args, seed=seed, scenario=scenario)
    returns = []
    start_stds = []
    end_stds = []
    std_improvements = []
    final_info: Dict[str, Any] = {}
    try:
        for episode in range(int(args.eval_episodes)):
            obs, _info = env.reset(seed=seed + episode)
            done = False
            episode_return = 0.0
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                episode_return += float(reward)
                parts = dict(info.get("reward_parts", {}))
                start_stds.append(float(parts.get("start_demand_std", 0.0)))
                end_stds.append(float(parts.get("end_demand_std", 0.0)))
                std_improvements.append(float(parts.get("demand_std_improvement", 0.0)))
                final_info = dict(info)
            returns.append(float(episode_return))
    finally:
        env.close()

    return {
        "episodes": int(args.eval_episodes),
        "mean_reward": float(np.mean(returns)) if returns else 0.0,
        "std_reward": float(np.std(returns)) if returns else 0.0,
        "mean_start_demand_std": float(np.mean(start_stds)) if start_stds else 0.0,
        "mean_end_demand_std": float(np.mean(end_stds)) if end_stds else 0.0,
        "mean_demand_std_improvement": float(np.mean(std_improvements)) if std_improvements else 0.0,
        "load_measurement_mode": str(final_info.get("load_measurement_mode", "")),
        "eval_scenario": str(args.scenario if scenario is None else scenario),
    }


def train_phases(
    model: TD3,
    initial_env: Monitor,
    args,
    callback: LowerTrainingCSVCallback,
) -> list[Dict[str, Any]]:
    phases = [
        *[
            {
                "name": scenario,
                "scenario": scenario,
                "timesteps": int(args.single_slice_phase_timesteps),
            }
            for scenario in SINGLE_SLICE_PHASE_SCENARIOS
        ],
        {
            "name": "mixed",
            "scenario": str(args.mixed_phase_scenario),
            "timesteps": int(args.mixed_phase_timesteps),
        },
    ]
    completed = []
    current_env = initial_env
    total_so_far = 0
    try:
        for phase_index, phase in enumerate(phases):
            if phase["timesteps"] <= 0:
                continue
            if phase_index > 0:
                current_env.close()
                current_env = make_env(
                    args,
                    seed=args.seed + phase_index,
                    scenario=phase["scenario"],
                )
                model.set_env(current_env)
            callback.phase_name = str(phase["name"])
            callback.phase_scenario = str(phase["scenario"])
            model.learn(
                total_timesteps=int(phase["timesteps"]),
                reset_num_timesteps=(total_so_far == 0),
                progress_bar=False,
                log_interval=int(args.log_interval),
                callback=callback,
            )
            total_so_far += int(phase["timesteps"])
            completed.append({
                **phase,
                "cumulative_timesteps": int(total_so_far),
            })
    finally:
        current_env.close()
    return completed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train lower TD3 with oracle upper bias and fixed per-UE PRB-demand balancing."
    )
    parser.add_argument("--total-timesteps", "--timesteps", dest="total_timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--model-dir", type=Path, default=Path("models/oracle_demand_lower_td3"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--action-noise-sigma", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=250,
        help="SB3 rollout log interval in episodes. Higher values print less often.",
    )
    parser.add_argument("--scenario", type=str, default="jain_balance_controllable")
    parser.add_argument("--scenario-selection", type=str, default="cycle")
    parser.add_argument("--controlled-gnb", type=int, default=1)
    parser.add_argument("--slice-types", type=str, default=",".join(SLICE_TYPES))
    parser.add_argument("--topology", type=str, default="medium_270m")
    parser.add_argument("--upper-window-seconds", type=float, default=1.0)
    parser.add_argument("--local-steps-per-global", type=int, default=10)
    parser.add_argument("--radio-substeps", type=int, default=20)
    parser.add_argument(
        "--control-interval-steps",
        type=int,
        default=3,
        help="Simulator ticks per lower action. Default 3 matches the real A3 handover_ttt.",
    )
    parser.add_argument("--episode-control-intervals", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument(
        "--disable-safe-admission",
        action="store_true",
        help="Disable safe admission. By default the env uses safe admission for A3 candidates.",
    )
    parser.add_argument("--load-reward-weight", type=float, default=1.0)
    parser.add_argument("--offset-imitation-weight", type=float, default=0.25)
    parser.add_argument("--bias-alignment-weight", type=float, default=0.10)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
    parser.add_argument("--handover-failure-weight", type=float, default=0.25)
    parser.add_argument("--pingpong-weight", type=float, default=0.25)
    parser.add_argument(
        "--phased-scenario-training",
        action="store_true",
        help=(
            "Train sequentially on one-slice controlled scenarios, then mixed. "
            "Phases are eMBB, URLLC, mMTC, then mixed."
        ),
    )
    parser.add_argument(
        "--single-slice-phase-timesteps",
        type=int,
        default=10_000,
        help="Timesteps for each one-slice phase when --phased-scenario-training is used.",
    )
    parser.add_argument(
        "--mixed-phase-timesteps",
        type=int,
        default=40_000,
        help="Timesteps for the mixed phase when --phased-scenario-training is used.",
    )
    parser.add_argument(
        "--mixed-phase-scenario",
        type=str,
        default=MIXED_PHASE_SCENARIO,
        help="Scenario used for the final mixed phase.",
    )
    parser.add_argument("--no-timestamp-run-dir", action="store_true")
    args = parser.parse_args()
    args.slice_types = tuple(
        item.strip()
        for item in str(args.slice_types).split(",")
        if item.strip()
    ) or SLICE_TYPES
    return args


def main():
    args = parse_args()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = (
        args.model_dir
        if args.no_timestamp_run_dir
        else args.model_dir / f"run_{run_timestamp}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.phased_scenario_training:
        args.total_timesteps = (
            3 * int(args.single_slice_phase_timesteps)
            + int(args.mixed_phase_timesteps)
        )

    env = make_env(
        args,
        seed=args.seed,
        scenario=(
            SINGLE_SLICE_PHASE_SCENARIOS[0]
            if args.phased_scenario_training
            else None
        ),
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
        learning_rate=float(args.learning_rate),
        buffer_size=int(args.buffer_size),
        learning_starts=(
            min(1_000, max(10, args.total_timesteps // 10))
            if args.learning_starts is None
            else int(args.learning_starts)
        ),
        batch_size=int(args.batch_size),
        gamma=0.95,
        train_freq=(int(args.train_freq), "step"),
        gradient_steps=int(args.gradient_steps),
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
        device=args.device,
    )
    csv_log_path = model_dir / "training_log.csv"
    csv_callback = LowerTrainingCSVCallback(csv_log_path)
    if args.phased_scenario_training:
        training_phases = train_phases(model, env, args, csv_callback)
    else:
        try:
            csv_callback.phase_name = "single"
            csv_callback.phase_scenario = str(args.scenario)
            model.learn(
                total_timesteps=args.total_timesteps,
                progress_bar=False,
                log_interval=int(args.log_interval),
                callback=csv_callback,
            )
        finally:
            env.close()
        training_phases = [{
            "name": "single",
            "scenario": str(args.scenario),
            "timesteps": int(args.total_timesteps),
            "cumulative_timesteps": int(args.total_timesteps),
        }]

    final_model_path = model_dir / "oracle_demand_lower_td3_final.zip"
    model.save(final_model_path)
    eval_scenario = (
        str(args.mixed_phase_scenario)
        if args.phased_scenario_training
        else str(args.scenario)
    )
    metrics = evaluate(model, args, seed=args.seed + 10_000, scenario=eval_scenario)
    metrics.update({
        "run_timestamp": run_timestamp,
        "run_dir": str(model_dir),
        "training_log_csv": str(csv_log_path),
        "saved_final_model": str(final_model_path),
        "total_timesteps": int(args.total_timesteps),
        "scenario": str(args.scenario),
        "phased_scenario_training": bool(args.phased_scenario_training),
        "training_phases": training_phases,
        "single_slice_phase_timesteps": int(args.single_slice_phase_timesteps),
        "mixed_phase_timesteps": int(args.mixed_phase_timesteps),
        "mixed_phase_scenario": str(args.mixed_phase_scenario),
        "learning_rate": float(args.learning_rate),
        "learning_starts": (
            min(1_000, max(10, args.total_timesteps // 10))
            if args.learning_starts is None
            else int(args.learning_starts)
        ),
        "train_freq": int(args.train_freq),
        "gradient_steps": int(args.gradient_steps),
        "batch_size": int(args.batch_size),
        "buffer_size": int(args.buffer_size),
        "log_interval": int(args.log_interval),
        "controlled_gnb": int(args.controlled_gnb),
        "slice_types": list(args.slice_types),
        "safe_admission_enabled": bool(not args.disable_safe_admission),
    })
    metrics_path = model_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
