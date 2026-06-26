#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from global_ppo_3gnb_env import SLICE_TYPES
from shared_local_a3_td3_env import SharedLocalA3TD3Env


def make_env(
    seed: int = 7,
    slice_types=SLICE_TYPES,
    training_scenarios: str = "jain_control_mixed",
    upper_window_seconds: float = 2.0,
    local_steps_per_global: int = 10,
    radio_substeps: int = 20,
    control_interval_steps: int = 1,
    episode_control_intervals: int | None = None,
    bias_update_intervals: int = 1,
    warmup_steps: int = 1,
) -> Monitor:
    env = SharedLocalA3TD3Env(
        seed=seed,
        slice_types=tuple(slice_types),
        training_scenarios=training_scenarios,
        upper_window_seconds=upper_window_seconds,
        local_steps_per_global=local_steps_per_global,
        radio_substeps=radio_substeps,
        control_interval_steps=control_interval_steps,
        episode_control_intervals=episode_control_intervals,
        bias_update_intervals=bias_update_intervals,
        warmup_steps=warmup_steps,
    )
    return Monitor(env)


def evaluate(
    model: TD3,
    seed: int,
    episodes: int,
    slice_types=SLICE_TYPES,
    training_scenarios: str = "jain_control_mixed",
    upper_window_seconds: float = 2.0,
    local_steps_per_global: int = 10,
    radio_substeps: int = 20,
    control_interval_steps: int = 1,
    episode_control_intervals: int | None = None,
    bias_update_intervals: int = 1,
    warmup_steps: int = 1,
) -> Dict[str, Any]:
    env = make_env(
        seed=seed,
        slice_types=slice_types,
        training_scenarios=training_scenarios,
        upper_window_seconds=upper_window_seconds,
        local_steps_per_global=local_steps_per_global,
        radio_substeps=radio_substeps,
        control_interval_steps=control_interval_steps,
        episode_control_intervals=episode_control_intervals,
        bias_update_intervals=bias_update_intervals,
        warmup_steps=warmup_steps,
    )
    returns = []
    decision_lengths = []
    advanced_steps = []
    handovers = []
    load_improvements = []
    final_info: Dict[str, Any] = {}

    try:
        for episode in range(int(episodes)):
            obs, _info = env.reset(seed=seed + episode)
            done = False
            episode_return = 0.0
            decision_count = 0
            advanced_count = 0
            handover_count = 0
            load_reward_values = []

            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                episode_return += float(reward)
                decision_count += 1
                if bool(info.get("time_advanced", False)):
                    advanced_count += 1
                    handover_count += int(info.get("handover_count", 0))
                    load_reward_values.append(float(info.get("reward_load_improvement", 0.0)))
                final_info = dict(info)

            returns.append(float(episode_return))
            decision_lengths.append(int(decision_count))
            advanced_steps.append(int(advanced_count))
            handovers.append(int(handover_count))
            load_improvements.append(float(np.mean(load_reward_values)) if load_reward_values else 0.0)
    finally:
        env.close()

    return {
        "episodes": int(episodes),
        "mean_reward": float(np.mean(returns)) if returns else 0.0,
        "std_reward": float(np.std(returns)) if returns else 0.0,
        "mean_td3_decisions_per_episode": float(np.mean(decision_lengths)) if decision_lengths else 0.0,
        "mean_physical_control_intervals_per_episode": float(np.mean(advanced_steps)) if advanced_steps else 0.0,
        "mean_handovers_per_episode": float(np.mean(handovers)) if handovers else 0.0,
        "mean_load_improvement_reward": float(np.mean(load_improvements)) if load_improvements else 0.0,
        "time_coherent_turns_per_interval": int(len(final_info.get("agent_order", [])) or 0),
        "control_interval_steps": int(control_interval_steps),
        "episode_control_intervals": int(env.env.episode_control_intervals),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train one shared TD3 lower policy reused by all gNBs."
    )
    parser.add_argument("--total-timesteps", "--timesteps", dest="total_timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--model-dir", type=Path, default=Path("models/shared_local_a3_td3"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--action-noise-sigma", type=float, default=0.25)
    parser.add_argument("--slice-types", type=str, default=",".join(SLICE_TYPES))
    parser.add_argument("--training-scenarios", type=str, default="jain_control_mixed")
    parser.add_argument("--upper-window-seconds", type=float, default=2.0)
    parser.add_argument("--local-steps-per-global", type=int, default=10)
    parser.add_argument("--radio-substeps", type=int, default=20)
    parser.add_argument("--control-interval-steps", type=int, default=1)
    parser.add_argument(
        "--episode-control-intervals",
        type=int,
        default=0,
        help="Physical lower control intervals per episode. Use 0 to derive from scenario duration.",
    )
    parser.add_argument("--bias-update-intervals", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--no-timestamp-run-dir", action="store_true")
    args = parser.parse_args()

    slice_types = tuple(
        item.strip()
        for item in str(args.slice_types).split(",")
        if item.strip()
    ) or SLICE_TYPES

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = (
        args.model_dir
        if args.no_timestamp_run_dir
        else args.model_dir / f"run_{run_timestamp}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        seed=args.seed,
        slice_types=slice_types,
        training_scenarios=args.training_scenarios,
        upper_window_seconds=args.upper_window_seconds,
        local_steps_per_global=args.local_steps_per_global,
        radio_substeps=args.radio_substeps,
        control_interval_steps=args.control_interval_steps,
        episode_control_intervals=(
            None if args.episode_control_intervals <= 0 else args.episode_control_intervals
        ),
        bias_update_intervals=args.bias_update_intervals,
        warmup_steps=args.warmup_steps,
    )
    actual_episode_control_intervals = int(env.env.episode_control_intervals)
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
        buffer_size=200_000,
        learning_starts=min(1_000, max(10, args.total_timesteps // 10)),
        batch_size=128,
        gamma=0.95,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
        device=args.device,
    )

    try:
        model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    finally:
        env.close()

    final_model_path = model_dir / "shared_local_a3_td3_final.zip"
    model.save(final_model_path)

    metrics = evaluate(
        model=model,
        seed=args.seed + 10_000,
        episodes=args.eval_episodes,
        slice_types=slice_types,
        training_scenarios=args.training_scenarios,
        upper_window_seconds=args.upper_window_seconds,
        local_steps_per_global=args.local_steps_per_global,
        radio_substeps=args.radio_substeps,
        control_interval_steps=args.control_interval_steps,
        episode_control_intervals=(
            None if args.episode_control_intervals <= 0 else args.episode_control_intervals
        ),
        bias_update_intervals=args.bias_update_intervals,
        warmup_steps=args.warmup_steps,
    )
    metrics.update({
        "run_timestamp": run_timestamp,
        "run_dir": str(model_dir),
        "saved_final_model": str(final_model_path),
        "total_timesteps": int(args.total_timesteps),
        "device": str(args.device),
        "slice_types": list(slice_types),
        "training_scenarios": str(args.training_scenarios),
        "upper_window_seconds": float(args.upper_window_seconds),
        "episode_control_intervals": int(actual_episode_control_intervals),
        "shared_policy": True,
        "time_coherent": True,
    })

    metrics_path = model_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved shared lower TD3 model to {final_model_path}")
    print(f"Saved evaluation metrics to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
