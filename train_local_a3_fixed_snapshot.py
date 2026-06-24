#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from local_a3_training_env import LocalA3RuleBiasTrainingEnv
from local_a3_training_scenarios import (
    EpisodeTrainingScenario,
    fixed_embb_snapshot_scenarios,
    local_a3_training_scenario_set,
)
from train_local_a3_td3 import _desired_offset_from_info, _lookup_tuple_dict


TRACE_SLICE_TYPES = ("eMBB", "URLLC", "mMTC")
SNAPSHOT_EXPECTED_SIGN = {
    "fixed_embb_offload_snapshot": "negative",
    "fixed_embb_retain_snapshot": "positive",
    "fixed_embb_neutral_snapshot": "near zero",
}


def _expected_sign_for_case(case_name: str) -> str:
    if case_name == "offload":
        return "negative"
    if case_name == "retain":
        return "positive"
    if case_name == "neutral":
        return "near zero"
    if case_name == "risky_offload":
        return "negative"
    return "unknown"


def _offset_matches_case(offset: float, case_name: str) -> bool:
    if case_name in {"offload", "risky_offload"}:
        return float(offset) < -0.5
    if case_name == "retain":
        return float(offset) > 0.5
    if case_name == "neutral":
        return abs(float(offset)) <= 1.0
    return False


EPISODE_SLICE_FIELDS = [
    field
    for slice_type in TRACE_SLICE_TYPES
    for field in (
        f"expected_case_{slice_type}",
        f"local_load_{slice_type}",
        f"neighbor_load_{slice_type}",
        f"scenario_local_load_{slice_type}",
        f"scenario_neighbor_load_{slice_type}",
        f"local_ue_count_{slice_type}",
        f"neighbor_ue_count_{slice_type}",
        f"local_bias_{slice_type}",
        f"neighbor_bias_{slice_type}",
        f"mean_raw_action_{slice_type}",
        f"mean_applied_offset_{slice_type}",
        f"mean_desired_offset_{slice_type}",
        f"mean_tracking_error_{slice_type}",
    )
]
EPISODE_FIELDS = [
    "phase",
    "episode",
    "global_step",
    "snapshot_name",
    "expected_case",
    "local_load",
    "neighbor_load",
    "scenario_local_load",
    "scenario_neighbor_load",
    "local_ue_count",
    "neighbor_ue_count",
    "local_bias",
    "neighbor_bias",
    "mean_raw_action",
    "mean_applied_offset",
    "mean_desired_offset",
    "mean_tracking_error",
    "handover_attempts",
    "handover_successes",
    "handover_failures",
    "handover_ping_pongs",
    "episode_return",
    *EPISODE_SLICE_FIELDS,
]


def _mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _model_zip(path: Path) -> Path:
    path = Path(path)
    if path.suffix == ".zip":
        return path
    final_path = path / "local_a3_td3_final.zip"
    if final_path.exists():
        return final_path
    good_start_path = path / "local_a3_td3_good_start.zip"
    if good_start_path.exists():
        return good_start_path
    return final_path


def _save_locations(save_path: Path) -> tuple[Path, Path]:
    save_path = Path(save_path)
    if save_path.suffix == ".zip":
        return save_path.parent, save_path
    return save_path, save_path / "local_a3_td3_final.zip"


def make_env(
    *,
    seed: int,
    episode_steps: int,
    control_interval_steps: int,
    scenarios: tuple[EpisodeTrainingScenario, ...],
    slice_types: Sequence[str] = ("eMBB",),
    scenario_hold_episodes: int = 1,
    debug: bool = False,
) -> Monitor:
    control_interval_steps = max(1, int(control_interval_steps))
    env = LocalA3RuleBiasTrainingEnv(
        seed=seed,
        gnb_id=0,
        neighbor_ids=(1,),
        slice_types=tuple(slice_types),
        episode_steps=episode_steps,
        steps_per_action=control_interval_steps,
        radio_substeps=10,
        balance_bias_cases=True,
        training_scenarios=scenarios,
        scenario_hold_episodes=scenario_hold_episodes,
        print_scenarios=debug,
        action_hold_steps=1,
        bias_hold_steps=20,
        max_offset_change_db=2.0,
    )
    return Monitor(env)


@dataclass
class EpisodeAccumulator:
    phase: str
    slice_types: tuple[str, ...] = ("eMBB",)
    episode: int = 0
    episode_return: float = 0.0
    snapshot_name: str = ""
    expected_case: str = ""
    local_loads: list[float] | None = None
    neighbor_loads: list[float] | None = None
    scenario_local_loads: list[float] | None = None
    scenario_neighbor_loads: list[float] | None = None
    local_ue_counts: list[float] | None = None
    neighbor_ue_counts: list[float] | None = None
    local_biases: list[float] | None = None
    neighbor_biases: list[float] | None = None
    raw_actions: list[float] | None = None
    applied_offsets: list[float] | None = None
    desired_offsets: list[float] | None = None
    tracking_errors: list[float] | None = None
    per_slice: dict[str, dict[str, list[float] | str]] | None = None
    handover_attempts: int = 0
    handover_successes: int = 0
    handover_failures: int = 0
    handover_ping_pongs: int = 0

    def __post_init__(self):
        self.reset_episode()

    def reset_episode(self):
        self.episode_return = 0.0
        self.snapshot_name = ""
        self.expected_case = ""
        self.local_loads = []
        self.neighbor_loads = []
        self.scenario_local_loads = []
        self.scenario_neighbor_loads = []
        self.local_ue_counts = []
        self.neighbor_ue_counts = []
        self.local_biases = []
        self.neighbor_biases = []
        self.raw_actions = []
        self.applied_offsets = []
        self.desired_offsets = []
        self.tracking_errors = []
        self.per_slice = {
            slice_type: {
                "expected_case": "",
                "local_loads": [],
                "neighbor_loads": [],
                "scenario_local_loads": [],
                "scenario_neighbor_loads": [],
                "local_ue_counts": [],
                "neighbor_ue_counts": [],
                "local_biases": [],
                "neighbor_biases": [],
                "raw_actions": [],
                "applied_offsets": [],
                "desired_offsets": [],
                "tracking_errors": [],
            }
            for slice_type in TRACE_SLICE_TYPES
        }
        self.handover_attempts = 0
        self.handover_successes = 0
        self.handover_failures = 0
        self.handover_ping_pongs = 0

    def step(self, env, reward: float, info: Dict[str, Any]):
        self.episode_return += float(reward)
        self.snapshot_name = str(info.get("scenario_name", self.snapshot_name))
        cases = dict(info.get("slice_bias_cases", {}))
        self.expected_case = str(cases.get("eMBB", self.expected_case))

        loads = dict(info.get("post_action_slice_loads", info.get("slice_loads", {})))
        scenario_loads = dict(info.get("scenario_slice_loads", info.get("slice_loads", {})))
        ue_counts = dict(info.get("post_action_spawn_counts", info.get("spawn_counts", {})))
        rule_bias = dict(info.get("rule_bias", {}))
        action_temporal = dict(info.get("action_temporal", {}))
        raw_actions = dict(action_temporal.get("raw_actions", {}))
        applied_offsets = dict(info.get("applied_offsets", {}))
        handover_stats = dict(info.get("handover_stats", {}))

        for slice_type in TRACE_SLICE_TYPES:
            values = self.per_slice[slice_type]
            local_load = float(_lookup_tuple_dict(loads, (0, slice_type), 0.0))
            neighbor_load = float(_lookup_tuple_dict(loads, (1, slice_type), 0.0))
            scenario_local_load = float(_lookup_tuple_dict(scenario_loads, (0, slice_type), 0.0))
            scenario_neighbor_load = float(_lookup_tuple_dict(scenario_loads, (1, slice_type), 0.0))
            local_ue_count = float(_lookup_tuple_dict(ue_counts, (0, slice_type), 0.0))
            neighbor_ue_count = float(_lookup_tuple_dict(ue_counts, (1, slice_type), 0.0))
            local_bias = float(_lookup_tuple_dict(rule_bias, (0, 1, slice_type), 0.0))
            neighbor_bias = float(_lookup_tuple_dict(rule_bias, (1, 0, slice_type), 0.0))
            raw_action = float(_lookup_tuple_dict(raw_actions, (1, slice_type), 0.0))
            applied_offset = float(_lookup_tuple_dict(applied_offsets, (1, slice_type), 0.0))
            desired_offset = float(_desired_offset_from_info(env, info, neighbor_id=1, slice_type=slice_type))

            values["expected_case"] = str(cases.get(slice_type, "inactive"))
            values["local_loads"].append(local_load)
            values["neighbor_loads"].append(neighbor_load)
            values["scenario_local_loads"].append(scenario_local_load)
            values["scenario_neighbor_loads"].append(scenario_neighbor_load)
            values["local_ue_counts"].append(local_ue_count)
            values["neighbor_ue_counts"].append(neighbor_ue_count)
            values["local_biases"].append(local_bias)
            values["neighbor_biases"].append(neighbor_bias)
            values["raw_actions"].append(raw_action)
            values["applied_offsets"].append(applied_offset)
            values["desired_offsets"].append(desired_offset)
            values["tracking_errors"].append(applied_offset - desired_offset)

            if slice_type == "eMBB":
                self.local_loads.append(local_load)
                self.neighbor_loads.append(neighbor_load)
                self.scenario_local_loads.append(scenario_local_load)
                self.scenario_neighbor_loads.append(scenario_neighbor_load)
                self.local_ue_counts.append(local_ue_count)
                self.neighbor_ue_counts.append(neighbor_ue_count)
                self.local_biases.append(local_bias)
                self.neighbor_biases.append(neighbor_bias)
                self.raw_actions.append(raw_action)
                self.applied_offsets.append(applied_offset)
                self.desired_offsets.append(desired_offset)
                self.tracking_errors.append(applied_offset - desired_offset)
        self.handover_attempts += int(handover_stats.get("attempts", 0))
        self.handover_successes += int(handover_stats.get("successes", 0))
        self.handover_failures += int(handover_stats.get("failures", 0))
        self.handover_ping_pongs += int(handover_stats.get("ping_pongs", 0))

    def row(self, global_step: int) -> Dict[str, Any]:
        row = {
            "phase": self.phase,
            "episode": int(self.episode),
            "global_step": int(global_step),
            "snapshot_name": self.snapshot_name,
            "expected_case": self.expected_case,
            "local_load": _mean_or_zero(self.local_loads),
            "neighbor_load": _mean_or_zero(self.neighbor_loads),
            "scenario_local_load": _mean_or_zero(self.scenario_local_loads),
            "scenario_neighbor_load": _mean_or_zero(self.scenario_neighbor_loads),
            "local_ue_count": _mean_or_zero(self.local_ue_counts),
            "neighbor_ue_count": _mean_or_zero(self.neighbor_ue_counts),
            "local_bias": _mean_or_zero(self.local_biases),
            "neighbor_bias": _mean_or_zero(self.neighbor_biases),
            "mean_raw_action": _mean_or_zero(self.raw_actions),
            "mean_applied_offset": _mean_or_zero(self.applied_offsets),
            "mean_desired_offset": _mean_or_zero(self.desired_offsets),
            "mean_tracking_error": _mean_or_zero(self.tracking_errors),
            "handover_attempts": int(self.handover_attempts),
            "handover_successes": int(self.handover_successes),
            "handover_failures": int(self.handover_failures),
            "handover_ping_pongs": int(self.handover_ping_pongs),
            "episode_return": float(self.episode_return),
        }
        for slice_type in TRACE_SLICE_TYPES:
            values = self.per_slice[slice_type]
            row.update({
                f"expected_case_{slice_type}": str(values["expected_case"]),
                f"local_load_{slice_type}": _mean_or_zero(values["local_loads"]),
                f"neighbor_load_{slice_type}": _mean_or_zero(values["neighbor_loads"]),
                f"scenario_local_load_{slice_type}": _mean_or_zero(values["scenario_local_loads"]),
                f"scenario_neighbor_load_{slice_type}": _mean_or_zero(values["scenario_neighbor_loads"]),
                f"local_ue_count_{slice_type}": _mean_or_zero(values["local_ue_counts"]),
                f"neighbor_ue_count_{slice_type}": _mean_or_zero(values["neighbor_ue_counts"]),
                f"local_bias_{slice_type}": _mean_or_zero(values["local_biases"]),
                f"neighbor_bias_{slice_type}": _mean_or_zero(values["neighbor_biases"]),
                f"mean_raw_action_{slice_type}": _mean_or_zero(values["raw_actions"]),
                f"mean_applied_offset_{slice_type}": _mean_or_zero(values["applied_offsets"]),
                f"mean_desired_offset_{slice_type}": _mean_or_zero(values["desired_offsets"]),
                f"mean_tracking_error_{slice_type}": _mean_or_zero(values["tracking_errors"]),
            })
        return row

    def finish(self):
        self.episode += 1
        self.reset_episode()


class EpisodeCsvLogger:
    def __init__(self, path: Path, slice_types: Sequence[str] = ("eMBB",)):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=EPISODE_FIELDS)
        self.writer.writeheader()
        self.accumulators = {
            "train": EpisodeAccumulator("train", tuple(slice_types)),
            "eval": EpisodeAccumulator("eval", tuple(slice_types)),
        }

    def close(self):
        self.file.close()

    def log_step(self, env, phase: str, global_step: int, reward: float, done: bool, info: Dict[str, Any]):
        acc = self.accumulators[phase]
        acc.step(env, reward, info)
        if done:
            self.writer.writerow(acc.row(global_step))
            self.file.flush()
            acc.finish()


class EpisodeLogCallback(BaseCallback):
    def __init__(self, logger: EpisodeCsvLogger):
        super().__init__()
        self.episode_logger = logger

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        rewards = np.asarray(self.locals.get("rewards", [0.0])).reshape(-1)
        dones = np.asarray(self.locals.get("dones", [False])).reshape(-1)
        infos = self.locals.get("infos", [{}])
        self.episode_logger.log_step(
            env=env,
            phase="train",
            global_step=self.num_timesteps,
            reward=float(rewards[0]),
            done=bool(dones[0]),
            info=dict(infos[0]),
        )
        return True


def evaluate_snapshot(
    model: TD3,
    *,
    scenario: EpisodeTrainingScenario,
    seed: int,
    episodes: int,
    episode_steps: int,
    control_interval_steps: int,
    slice_types: Sequence[str] = ("eMBB",),
    scenario_hold_episodes: int = 1,
    logger: EpisodeCsvLogger | None = None,
) -> Dict[str, Any]:
    env = make_env(
        seed=seed,
        episode_steps=episode_steps,
        control_interval_steps=control_interval_steps,
        scenarios=(scenario,),
        slice_types=slice_types,
        scenario_hold_episodes=scenario_hold_episodes,
        debug=False,
    )
    applied_offsets_by_slice = {slice_type: [] for slice_type in TRACE_SLICE_TYPES}
    try:
        for episode in range(int(episodes)):
            obs, _info = env.reset(seed=seed + episode)
            done = False
            step = 0
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                step += 1
                applied_offsets = dict(info.get("applied_offsets", {}))
                for slice_type in TRACE_SLICE_TYPES:
                    applied_offsets_by_slice[slice_type].append(
                        float(_lookup_tuple_dict(applied_offsets, (1, slice_type), 0.0))
                    )
                if logger is not None:
                    logger.log_step(
                        env=env,
                        phase="eval",
                        global_step=episode * episode_steps + step,
                        reward=float(reward),
                        done=done,
                        info=dict(info),
                    )
    finally:
        env.close()

    mean_applied_by_slice = {
        slice_type: _mean_or_zero(values)
        for slice_type, values in applied_offsets_by_slice.items()
    }
    mean_applied = float(mean_applied_by_slice.get("eMBB", 0.0))
    expected_by_slice = {
        slice_type: _expected_sign_for_case(scenario.for_slice(slice_type).case)
        for slice_type in TRACE_SLICE_TYPES
    }
    passed_by_slice = {
        slice_type: _offset_matches_case(
            mean_applied_by_slice.get(slice_type, 0.0),
            scenario.for_slice(slice_type).case,
        )
        for slice_type in TRACE_SLICE_TYPES
    }
    expected = SNAPSHOT_EXPECTED_SIGN.get(
        scenario.name,
        expected_by_slice.get("eMBB", "unknown"),
    )
    if scenario.name.endswith("offload_snapshot"):
        passed = mean_applied < -0.5
    elif scenario.name.endswith("retain_snapshot"):
        passed = mean_applied > 0.5
    elif scenario.name.endswith("neutral_snapshot"):
        passed = abs(mean_applied) <= 1.0
    else:
        active_slices = [slice_type for slice_type in slice_types if slice_type in TRACE_SLICE_TYPES]
        passed = all(bool(passed_by_slice[slice_type]) for slice_type in active_slices)
    return {
        "snapshot_name": scenario.name,
        "expected_offset_sign": expected,
        "expected_offset_sign_by_slice": expected_by_slice,
        "mean_applied_offset": float(mean_applied),
        "mean_applied_offset_by_slice": mean_applied_by_slice,
        "passed_by_slice": passed_by_slice,
        "passed": bool(passed),
    }


def evaluate_all_snapshots(
    model: TD3,
    *,
    seed: int,
    episodes: int,
    episode_steps: int,
    control_interval_steps: int,
    scenarios: tuple[EpisodeTrainingScenario, ...] | None = None,
    slice_types: Sequence[str] = ("eMBB",),
    scenario_hold_episodes: int = 1,
    logger: EpisodeCsvLogger | None = None,
) -> list[Dict[str, Any]]:
    results = []
    eval_scenarios = scenarios or tuple(
        fixed_embb_snapshot_scenarios(snapshot)[0]
        for snapshot in ("offload", "retain", "neutral")
    )
    for idx, scenario in enumerate(eval_scenarios):
        results.append(
            evaluate_snapshot(
                model,
                scenario=scenario,
                seed=seed + 1000 * idx,
                episodes=episodes,
                episode_steps=episode_steps,
                control_interval_steps=control_interval_steps,
                slice_types=slice_types,
                scenario_hold_episodes=scenario_hold_episodes,
                logger=logger,
            )
        )
    return results


def print_eval_results(results: list[Dict[str, Any]]):
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"Evaluation on {result['snapshot_name']}:")
        print(f"    expected offset sign: {result['expected_offset_sign']}")
        print(f"    mean applied offset: {result['mean_applied_offset']:.3f}")
        if "mean_applied_offset_by_slice" in result:
            by_slice = result["mean_applied_offset_by_slice"]
            print(
                "    mean applied by slice: "
                + ", ".join(f"{key}={float(value):.3f}" for key, value in by_slice.items())
            )
        if "passed_by_slice" in result:
            by_slice = result["passed_by_slice"]
            print(
                "    per-slice pass: "
                + ", ".join(f"{key}={bool(value)}" for key, value in by_slice.items())
            )
        print(f"    pass/fail: {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Train local TD3 on fixed/feasible A3-offset snapshots."
    )
    parser.add_argument(
        "--snapshot",
        choices=("offload", "retain", "neutral", "mixed", "feasible_mixed"),
        required=True,
    )
    parser.add_argument("--total-timesteps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-path", type=Path, default=Path("models/local_td3_fixed_snapshot"))
    parser.add_argument("--load-path", type=Path, default=None)
    parser.add_argument("--episode-steps", type=int, default=40)
    parser.add_argument(
        "--control-interval-steps",
        type=int,
        default=5,
        help=(
            "Number of base env steps to run after each TD3 offset decision before "
            "returning the reward."
        ),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--scenario-hold-episodes",
        type=int,
        default=1,
        help=(
            "Keep the same sampled training scenario for this many episodes "
            "before sampling a new one. Useful for feasible_mixed to avoid "
            "changing the per-slice objective every episode."
        ),
    )
    args = parser.parse_args()

    if args.snapshot == "feasible_mixed":
        scenarios = local_a3_training_scenario_set("feasible_mixed")
        slice_types = TRACE_SLICE_TYPES
        eval_scenarios = scenarios
    else:
        scenarios = fixed_embb_snapshot_scenarios(args.snapshot)
        slice_types = ("eMBB",)
        eval_scenarios = tuple(
            fixed_embb_snapshot_scenarios(snapshot)[0]
            for snapshot in ("offload", "retain", "neutral")
        )
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir, final_model_path = _save_locations(Path(args.save_path))
    save_dir.mkdir(parents=True, exist_ok=True)
    episode_csv = save_dir / "episode_trace.csv"
    eval_json = save_dir / "fixed_snapshot_eval.json"
    logger = EpisodeCsvLogger(episode_csv, slice_types=slice_types)

    try:
        env = make_env(
            seed=args.seed,
            episode_steps=args.episode_steps,
            control_interval_steps=args.control_interval_steps,
            scenarios=scenarios,
            slice_types=slice_types,
            scenario_hold_episodes=args.scenario_hold_episodes,
            debug=args.debug,
        )

        if args.load_path is not None:
            load_model_path = _model_zip(Path(args.load_path))
            model = TD3.load(load_model_path, env=env, seed=args.seed, device=args.device)
        else:
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.25 * np.ones(n_actions),
            )
            model = TD3(
                "MlpPolicy",
                env,
                seed=args.seed,
                action_noise=action_noise,
                learning_rate=1e-3,
                buffer_size=100_000,
                learning_starts=min(500, max(10, args.total_timesteps // 10)),
                batch_size=128,
                gamma=0.95,
                train_freq=(1, "step"),
                gradient_steps=1,
                policy_kwargs={"net_arch": [64, 64]},
                verbose=1 if args.debug else 0,
                device=args.device,
            )

        try:
            model.learn(
                total_timesteps=int(args.total_timesteps),
                callback=EpisodeLogCallback(logger),
                progress_bar=False,
            )
        finally:
            env.close()

        model.save(final_model_path)
        results = evaluate_all_snapshots(
            model,
            seed=args.seed + 10_000,
            episodes=args.eval_episodes,
            episode_steps=args.episode_steps,
            control_interval_steps=args.control_interval_steps,
            scenarios=eval_scenarios,
            slice_types=slice_types,
            scenario_hold_episodes=1,
            logger=logger,
        )
        payload = {
            "snapshot_mode": args.snapshot,
            "slice_types": list(slice_types),
            "training_scenarios": [scenario.name for scenario in scenarios],
            "scenario_hold_episodes": int(args.scenario_hold_episodes),
            "run_timestamp": run_timestamp,
            "total_timesteps": int(args.total_timesteps),
            "episode_steps": int(args.episode_steps),
            "seed": int(args.seed),
            "save_path": str(save_dir),
            "saved_model": str(final_model_path),
            "load_path": None if args.load_path is None else str(args.load_path),
            "episode_csv": str(episode_csv),
            "evaluation": results,
            "passed_all": bool(all(result["passed"] for result in results)),
        }
        eval_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print_eval_results(results)
        print(json.dumps(payload, indent=2))
    finally:
        logger.close()


if __name__ == "__main__":
    main()
