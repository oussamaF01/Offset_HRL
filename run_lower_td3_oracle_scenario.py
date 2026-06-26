#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from stable_baselines3 import TD3

from global_ppo_3gnb_env import GlobalPPO3GNBEnv
from local_a3_agent_wrapper import LocalA3OffsetEnv, quantize_a3_offset
from upper_agent_training_scenarios import CENTER_GAP_GNB_CONFIGS


def oracle_directional_bias(env: GlobalPPO3GNBEnv, loads: np.ndarray) -> np.ndarray:
    """Load-balancing oracle bias used for controlled scenario probing.

    Negative bias means source -> target release. Positive bias means retain at
    source. This mirrors the existing feasibility-test oracle, with an added
    positive retain signal for source/target pairs where the target is heavier.
    """
    loads = np.asarray(loads, dtype=float)
    means = loads.mean(axis=0)
    bias = np.zeros((env.n_gnbs, env.max_neighbors, len(env.slice_types)), dtype=np.float32)

    for source in range(env.n_gnbs):
        for slice_idx, slice_type in enumerate(env.slice_types):
            source_excess = loads[source, slice_idx] - means[slice_idx]
            deficits = [
                max(means[slice_idx] - loads[target, slice_idx], 0.0)
                for target in env.neighbors[source]
            ]
            total_deficit = sum(deficits)

            for slot, target in enumerate(env.neighbors[source]):
                target_excess = loads[target, slice_idx] - means[slice_idx]
                if source_excess > 1e-9 and total_deficit > 0.0 and deficits[slot] > 0.0:
                    if _has_radio_feasible_ue(env, source, target, slice_type):
                        bias[source, slot, slice_idx] = -float(deficits[slot] / total_deficit)
                elif target_excess > 1e-9:
                    bias[source, slot, slice_idx] = float(
                        np.clip(target_excess / max(means[slice_idx], 0.05), 0.0, 1.0)
                    )

    return bias


def _has_radio_feasible_ue(
    env: GlobalPPO3GNBEnv,
    source: int,
    target: int,
    slice_type: str,
    min_margin_db: float = -5.0,
) -> bool:
    target_gnb = env.base_env._get_gnb_by_id(int(target))
    source_gnb = env.base_env._get_gnb_by_id(int(source))
    if target_gnb is None or source_gnb is None:
        return False
    for ue in env.base_env.get_all_ues():
        if (
            not getattr(ue, "connected", False)
            or ue.serving_gnb is None
            or int(ue.serving_gnb) != int(source)
            or str(getattr(ue, "slice_type", "")) != str(slice_type)
        ):
            continue
        if not env.base_env._is_in_coverage(target_gnb, ue):
            continue
        margin = (
            env.base_env._measure_rsrp(target_gnb, ue)
            - env.base_env._measure_rsrp(source_gnb, ue)
        )
        if margin > float(min_margin_db):
            return True
    return False


def _bias_dict(env: GlobalPPO3GNBEnv, bias: np.ndarray) -> Dict[Tuple[int, int, str], float]:
    result = {}
    for source in range(env.n_gnbs):
        for slot, target in enumerate(env.neighbors[source]):
            for s_idx, slice_type in enumerate(env.slice_types):
                result[(int(source), int(target), slice_type)] = float(bias[source, slot, s_idx])
    return result


def _local_envs(env: GlobalPPO3GNBEnv) -> Dict[int, LocalA3OffsetEnv]:
    return {
        int(gnb_id): LocalA3OffsetEnv(
            env.base_env,
            gnb_id=int(gnb_id),
            neighbor_ids=tuple(int(n) for n in env.neighbors[int(gnb_id)]),
            slice_types=env.slice_types,
            steps_per_action=1,
            ttt=1,
        )
        for gnb_id in range(env.n_gnbs)
    }


def _reset_local_view(local_env: LocalA3OffsetEnv, env: GlobalPPO3GNBEnv) -> None:
    local_env.base_env = env.base_env
    for key in local_env._offsets:
        local_env._offsets[key] = 0.0
        local_env._prev_proto_offsets[key] = 0.0
        local_env._mobility_counters[key] = {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "ping_pongs": 0,
        }
    local_env._ttt_counters.clear()
    local_env._last_serving = {
        int(ue.id): ue.serving_gnb
        for ue in env.base_env.get_all_ues()
    }
    local_env._prev_serving = {ue_id: None for ue_id in local_env._last_serving}
    local_env._last_reward_breakdown = {}


def _apply_offsets_to_base(local_env: LocalA3OffsetEnv) -> None:
    for (neighbor_id, slice_type), offset in local_env.get_applied_offsets().items():
        local_env.base_env.set_a3_offset(
            int(local_env.gnb_id),
            int(neighbor_id),
            slice_type,
            float(offset),
        )


def _iter_local_rows(
    *,
    step: int,
    controlled_gnb: int,
    local_env: LocalA3OffsetEnv,
    td3_raw: np.ndarray,
    td3_applied: Dict[Tuple[int, str], float],
    perfect_offsets: np.ndarray,
    bias: np.ndarray,
    loads_before: np.ndarray,
    loads_after: np.ndarray,
    counts_before: Dict[Tuple[int, str], int],
    sla_before: Dict[Tuple[int, str], float],
    handover_stats: Dict[int, Dict[str, int]],
) -> Iterable[Dict[str, Any]]:
    idx = 0
    for slot, neighbor_id in enumerate(local_env.neighbor_ids):
        for s_idx, slice_type in enumerate(local_env.slice_types):
            key = (int(neighbor_id), slice_type)
            raw = float(td3_raw[idx])
            applied = float(td3_applied.get(key, 0.0))
            perfect_raw = float(perfect_offsets[int(controlled_gnb), slot, s_idx])
            perfect_applied = quantize_a3_offset(perfect_raw)
            b = float(bias[int(controlled_gnb), slot, s_idx])
            yield {
                "step": int(step),
                "controlled_gnb": int(controlled_gnb),
                "neighbor_id": int(neighbor_id),
                "slice_type": slice_type,
                "upper_oracle_bias": b,
                "local_count_before": int(counts_before.get((controlled_gnb, slice_type), 0)),
                "neighbor_count_before": int(counts_before.get((int(neighbor_id), slice_type), 0)),
                "local_load_before": float(loads_before[int(controlled_gnb), s_idx]),
                "neighbor_load_before": float(loads_before[int(neighbor_id), s_idx]),
                "local_load_after": float(loads_after[int(controlled_gnb), s_idx]),
                "neighbor_load_after": float(loads_after[int(neighbor_id), s_idx]),
                "local_sla_before": float(sla_before.get((controlled_gnb, slice_type), 0.0)),
                "neighbor_sla_before": float(sla_before.get((int(neighbor_id), slice_type), 0.0)),
                "td3_raw_offset_db": raw,
                "td3_applied_offset_db": applied,
                "perfect_raw_offset_db": perfect_raw,
                "perfect_applied_offset_db": perfect_applied,
                "offset_error_db": float(applied - perfect_applied),
                "bias_aligned": _bias_aligned(b, applied),
                "handover_attempts": int(handover_stats[controlled_gnb].get("attempts", 0)),
                "handover_successes": int(handover_stats[controlled_gnb].get("successes", 0)),
                "handover_failures": int(handover_stats[controlled_gnb].get("failures", 0)),
                "handover_ping_pongs": int(handover_stats[controlled_gnb].get("ping_pongs", 0)),
            }
            idx += 1


def _bias_aligned(bias: float, offset: float) -> bool:
    bias = float(bias)
    offset = float(offset)
    if abs(bias) <= 0.1:
        return abs(offset) <= 2.0
    return bias * (offset / 6.0) > 0.0


def _write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    env = GlobalPPO3GNBEnv(
        seed=args.seed,
        scenario_mode="curriculum",
        training_scenarios=args.scenario,
        scenario_selection="cycle",
        gnb_configs=CENTER_GAP_GNB_CONFIGS[args.topology],
        upper_window_seconds=args.upper_window_seconds,
        local_steps_per_global=args.local_steps_per_global,
        radio_substeps=args.radio_substeps,
        terminal_reward_only=False,
        safe_admission_enabled=False,
        warmup_steps=args.warmup_steps,
        print_scenarios=False,
    )
    model = TD3.load(str(args.model), device=args.device)
    rows: list[Dict[str, Any]] = []

    try:
        _obs, reset_info = env.reset(seed=args.seed)
        local_envs = _local_envs(env)
        for local_env in local_envs.values():
            _reset_local_view(local_env, env)

        controlled_env = local_envs[int(args.controlled_gnb)]
        if tuple(model.observation_space.shape) != tuple(controlled_env.observation_space.shape):
            raise ValueError(
                f"Model observation shape {model.observation_space.shape} does not match "
                f"controlled lower observation {controlled_env.observation_space.shape}."
            )
        if tuple(model.action_space.shape) != tuple(controlled_env.action_space.shape):
            raise ValueError(
                f"Model action shape {model.action_space.shape} does not match "
                f"controlled lower action {controlled_env.action_space.shape}."
            )

        scenario_name = str(reset_info.get("scenario_name", args.scenario))
        for step in range(int(args.steps)):
            loads_before = np.asarray(env._load_matrix(), dtype=float)
            bias = oracle_directional_bias(env, loads_before)
            bias_map = _bias_dict(env, bias)
            for local_env in local_envs.values():
                local_env.set_global_bias(bias_map)

            perfect_offsets, _debug = env._compute_strong_local_offsets(bias)
            counts_before = controlled_env._slice_counts()
            sla_before = controlled_env._slice_sla_flags_by_gnb()

            handover_stats: Dict[int, Dict[str, int]] = {}
            for gnb_id, local_env in local_envs.items():
                if int(gnb_id) == int(args.controlled_gnb):
                    obs = local_env._build_observation()
                    action, _state = model.predict(obs, deterministic=not args.stochastic)
                    td3_raw = np.asarray(action, dtype=float).reshape(-1)
                    applied_vector = np.asarray(
                        [quantize_a3_offset(value) for value in td3_raw],
                        dtype=np.float32,
                    )
                    local_env._apply_proto_offsets(applied_vector)
                else:
                    offset_values = []
                    for slot, _neighbor_id in enumerate(local_env.neighbor_ids):
                        for s_idx, _slice_type in enumerate(local_env.slice_types):
                            offset_values.append(float(perfect_offsets[int(gnb_id), slot, s_idx]))
                    local_env._apply_proto_offsets(np.asarray(offset_values, dtype=np.float32))
                _apply_offsets_to_base(local_env)
                handover_stats[int(gnb_id)] = local_env._execute_a3_handovers()

            env.base_env.begin_radio_measurement_window()
            terminated = truncated = False
            for _ in range(max(1, int(args.control_interval_steps))):
                _base_obs, _base_reward, terminated, truncated, _base_info = env.base_env.step(0)
                if terminated or truncated:
                    break

            loads_after = np.asarray(env._load_matrix(), dtype=float)
            td3_offsets = controlled_env.get_applied_offsets()
            rows.extend(_iter_local_rows(
                step=step,
                controlled_gnb=int(args.controlled_gnb),
                local_env=controlled_env,
                td3_raw=td3_raw,
                td3_applied=td3_offsets,
                perfect_offsets=perfect_offsets,
                bias=bias,
                loads_before=loads_before,
                loads_after=loads_after,
                counts_before=counts_before,
                sla_before=sla_before,
                handover_stats=handover_stats,
            ))
            if terminated or truncated:
                break

        _write_csv(args.output_csv, rows)
        summary = {
            "scenario": scenario_name,
            "model": str(args.model),
            "controlled_gnb": int(args.controlled_gnb),
            "steps_requested": int(args.steps),
            "rows": len(rows),
            "output_csv": str(args.output_csv),
            "mean_abs_offset_error_db": (
                float(np.mean([abs(float(row["offset_error_db"])) for row in rows]))
                if rows else 0.0
            ),
            "bias_alignment_rate": (
                float(np.mean([bool(row["bias_aligned"]) for row in rows]))
                if rows else 0.0
            ),
            "total_controlled_handover_successes": int(sum(
                int(step_rows[0]["handover_successes"])
                for _step, step_rows in _rows_by_step(rows).items()
                if step_rows
            )),
        }
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        env.close()


def _rows_by_step(rows: list[Dict[str, Any]]) -> Dict[int, list[Dict[str, Any]]]:
    grouped: Dict[int, list[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["step"]), []).append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled upper scenario with oracle upper bias, TD3 lower "
            "offsets for one controlled gNB, and perfect heuristic offsets for "
            "the other gNBs."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scenario", type=str, default="jain_balance_controllable")
    parser.add_argument("--controlled-gnb", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--topology", choices=tuple(CENTER_GAP_GNB_CONFIGS), default="medium_270m")
    parser.add_argument("--upper-window-seconds", type=float, default=1.0)
    parser.add_argument("--local-steps-per-global", type=int, default=10)
    parser.add_argument("--radio-substeps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--control-interval-steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=Path("results/lower_td3_oracle_scenario_trace.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("results/lower_td3_oracle_scenario_summary.json"))
    args = parser.parse_args()

    summary = run_probe(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
