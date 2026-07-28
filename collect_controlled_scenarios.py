#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect (state, action, PRB demand distribution) for every controllable
scenario type using both upper heuristic and lower heuristic for all 3 gNBs.

Dataset layout
--------------
datasets/controlled_scenarios/
    <case_type>.npz   — one file per case type
    metadata.json     — collection config + per-case summary

Each NPZ contains arrays indexed by sample:
    observations      (N, obs_dim)    float32
    actions           (N, action_dim) float32   quantized dB A3 offsets
    demand_matrix     (N, 3, 3)       float32   gNBs × slices PRB demand load
    demand_totals     (N, 3)          float32   per-gNB total PRB demand
    sample_gnb_id     (N,)            int32
    episode           (N,)            int32
    step              (N,)            int32
    reward            (N,)            float32
    handover_count    (N,)            int32
    done              (N,)            bool
    upper_bias_json   (N,)            object    serialised bias dict
    admission_json    (N,)            object    serialised admission state
    metadata_json     scalar          object    JSON string
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── project imports ────────────────────────────────────────────────────────────
from local_a3_training_scenarios import (
    DEMAND_CASE_TYPES,
    generate_random_demand_scenario,
)
from lower_rl_training_utils import (
    collect_step_metrics,
    compute_expert_action,
    extract_demand_load_matrix,
    json_safe,
    make_lower_env,
    set_current_upper_bias,
)

# ── controllable case types (no mixed_random) ──────────────────────────────────
CONTROLLABLE_CASE_TYPES: tuple[str, ...] = tuple(
    ct for ct in DEMAND_CASE_TYPES if ct != "mixed_random"
)

GNB_IDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect lower-heuristic demonstrations for controlled scenarios."
    )
    p.add_argument("--out-dir", default="datasets/controlled_scenarios")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demand-seed", type=int, default=99)
    p.add_argument("--scenarios-per-case", type=int, default=20,
                   help="Number of distinct demand configs per case type.")
    p.add_argument("--episodes-per-scenario", type=int, default=3,
                   help="How many episodes to run per scenario config.")
    p.add_argument("--episode-steps", type=int, default=40)
    p.add_argument("--action-hold-steps", type=int, default=5)
    p.add_argument("--bias-hold-steps", type=int, default=20)
    p.add_argument("--max-offset-change-db", type=float, default=2.0)
    p.add_argument("--min-demand-load", type=float, default=0.05)
    p.add_argument("--max-demand-load", type=float, default=0.95)
    p.add_argument("--safe-total-gnb-load", type=float, default=0.65)
    p.add_argument("--min-total-gnb-load", type=float, default=0.25)
    p.add_argument("--placement-jitter-m", type=float, default=18.0)
    p.add_argument("--no-randomize-traffic", action="store_true")
    p.add_argument("--case-types", default=None,
                   help="Comma-separated subset of case types (default: all controllable).")
    return p.parse_args()


def handover_count_from_info(info: dict) -> int:
    for key in ("handover_count", "handover_successes", "handover_attempts"):
        if key in info:
            try:
                return int(info[key] or 0)
            except Exception:
                return 0
    return 0


def collect_for_case_type(
    case_type: str,
    *,
    rng: np.random.Generator,
    scenarios_per_case: int,
    episodes_per_scenario: int,
    episode_steps: int,
    action_hold_steps: int,
    bias_hold_steps: int,
    max_offset_change_db: float,
    min_demand_load: float,
    max_demand_load: float,
    safe_total_gnb_load: float,
    min_total_gnb_load: float,
    placement_jitter_m: float,
    randomize_traffic_profiles: bool,
    seed: int,
) -> dict:
    """Run all scenarios of one case type and return arrays dict."""

    # Build the scenario list for this case type.
    scenarios = tuple(
        generate_random_demand_scenario(
            rng=rng,
            index=i,
            case_type=case_type,
            min_demand_load=min_demand_load,
            max_demand_load=max_demand_load,
            safe_total_gnb_load=safe_total_gnb_load,
            min_total_gnb_load=min_total_gnb_load,
            placement_jitter_m=placement_jitter_m,
            randomize_traffic_profiles=randomize_traffic_profiles,
        )
        for i in range(scenarios_per_case)
    )

    env = make_lower_env(
        seed=seed,
        controlled_gnb_id=1,
        scenario_mode="random_demand",
        num_random_scenarios=len(scenarios),
        demand_seed=seed,
        min_demand_load=min_demand_load,
        max_demand_load=max_demand_load,
        safe_total_gnb_load=safe_total_gnb_load,
        min_total_gnb_load=min_total_gnb_load,
        episode_steps=episode_steps,
        action_hold_steps=action_hold_steps,
        bias_hold_steps=bias_hold_steps,
        max_offset_change_db=max_offset_change_db,
        randomize_traffic_profiles=randomize_traffic_profiles,
        placement_jitter_m=placement_jitter_m,
        print_scenarios=False,
    )

    # Replace the scenario list with our pre-built case-type-specific list.
    env.upper_env._scenario_list = scenarios
    env.upper_env._active_scenario_idx = 0

    observations: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    demand_matrices: list[np.ndarray] = []
    demand_totals_list: list[list[float]] = []
    sample_gnb_ids: list[int] = []
    episodes_list: list[int] = []
    steps_list: list[int] = []
    rewards: list[float] = []
    ho_counts: list[int] = []
    dones: list[bool] = []
    upper_bias_jsons: list[str] = []
    admission_jsons: list[str] = []

    total_episodes = scenarios_per_case * episodes_per_scenario
    episode_idx = 0
    step_in_ep = 0

    obs, reset_info = env.reset(seed=seed)
    info: dict = dict(reset_info or {})

    while episode_idx < total_episodes:
        # ── upper heuristic: compute bias for all 3 gNBs ──────────────────────
        bias = set_current_upper_bias(env, force=True)
        demand_matrix, _ = extract_demand_load_matrix(env, info)

        # ── admission control state (before step) ─────────────────────────────
        admission_before = (
            env.base_env.get_safe_admission_state()
            if hasattr(env.base_env, "get_safe_admission_state")
            else {}
        )

        # ── lower heuristic: compute action for every gNB ─────────────────────
        # gNBs 0 and 2 are run inside env._run_heuristic_gnbs() on each step.
        # gNB 1 (controlled) action is computed here and passed to env.step().
        ctrl_action, _ctrl_debug = compute_expert_action(
            env,
            gnb_id=1,
            bias=bias,
            demand_load_matrix=demand_matrix,
        )

        # Collect one sample per gNB at the pre-step state.
        metrics = collect_step_metrics(env, info)
        useful_matrix = np.asarray([
            [float(metrics.get(f"useful_g{gid}_{st}", 0.0)) for st in env.slice_types]
            for gid in env.gnb_ids
        ], dtype=np.float32)
        del useful_matrix  # not saved here but available if needed

        for gid in GNB_IDS:
            if int(gid) not in env.local_envs:
                continue
            sample_obs = np.asarray(
                env.local_envs[gid]._build_observation(), dtype=np.float32
            ).reshape(-1)
            sample_action, _ = compute_expert_action(
                env,
                gnb_id=gid,
                bias=bias,
                demand_load_matrix=demand_matrix,
            )
            observations.append(sample_obs)
            actions_list.append(np.asarray(sample_action, dtype=np.float32).reshape(-1))
            demand_matrices.append(np.asarray(demand_matrix, dtype=np.float32))
            demand_totals_list.append([
                float(metrics.get(f"demand_total_g{g}", 0.0)) for g in env.gnb_ids
            ])
            sample_gnb_ids.append(int(gid))
            episodes_list.append(episode_idx)
            steps_list.append(step_in_ep)
            upper_bias_jsons.append(json.dumps(json_safe(bias)))
            admission_jsons.append(json.dumps(json_safe(admission_before)))

        # ── step environment with controlled-gNB heuristic action ─────────────
        obs, reward, terminated, truncated, info = env.step(ctrl_action)

        # back-fill reward / done / HO for all gNBs added this step
        n_added = len(GNB_IDS)
        for _ in range(n_added):
            rewards.append(float(reward))
            dones.append(bool(terminated or truncated))
            ho_counts.append(handover_count_from_info(info))

        step_in_ep += 1
        if terminated or truncated:
            episode_idx += 1
            step_in_ep = 0
            if episode_idx < total_episodes:
                obs, reset_info = env.reset()
                info = dict(reset_info or {})

    env.close()

    N = len(observations)
    return {
        "observations":   np.asarray(observations, dtype=np.float32),
        "actions":        np.asarray(actions_list, dtype=np.float32),
        "demand_matrix":  np.asarray(demand_matrices, dtype=np.float32),
        "demand_totals":  np.asarray(demand_totals_list, dtype=np.float32),
        "sample_gnb_id":  np.asarray(sample_gnb_ids, dtype=np.int32),
        "episode":        np.asarray(episodes_list, dtype=np.int32),
        "step":           np.asarray(steps_list, dtype=np.int32),
        "reward":         np.asarray(rewards, dtype=np.float32),
        "handover_count": np.asarray(ho_counts, dtype=np.int32),
        "done":           np.asarray(dones, dtype=np.bool_),
        "upper_bias_json":  np.asarray(upper_bias_jsons, dtype=object),
        "admission_json":   np.asarray(admission_jsons, dtype=object),
        "_n_samples": N,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_types_to_run = CONTROLLABLE_CASE_TYPES
    if args.case_types:
        requested = [ct.strip() for ct in args.case_types.split(",") if ct.strip()]
        unknown = [ct for ct in requested if ct not in CONTROLLABLE_CASE_TYPES]
        if unknown:
            print(f"ERROR: unknown case types: {unknown}", file=sys.stderr)
            sys.exit(1)
        case_types_to_run = tuple(requested)

    rng = np.random.default_rng(args.demand_seed)
    per_case_summary: dict[str, dict] = {}

    for case_idx, case_type in enumerate(case_types_to_run):
        print(
            f"[{case_idx + 1}/{len(case_types_to_run)}] {case_type} "
            f"({args.scenarios_per_case} scenarios × {args.episodes_per_scenario} eps) ...",
            flush=True,
        )
        arrays = collect_for_case_type(
            case_type,
            rng=rng,
            scenarios_per_case=args.scenarios_per_case,
            episodes_per_scenario=args.episodes_per_scenario,
            episode_steps=args.episode_steps,
            action_hold_steps=args.action_hold_steps,
            bias_hold_steps=args.bias_hold_steps,
            max_offset_change_db=args.max_offset_change_db,
            min_demand_load=args.min_demand_load,
            max_demand_load=args.max_demand_load,
            safe_total_gnb_load=args.safe_total_gnb_load,
            min_total_gnb_load=args.min_total_gnb_load,
            placement_jitter_m=args.placement_jitter_m,
            randomize_traffic_profiles=not args.no_randomize_traffic,
            seed=args.seed + case_idx,
        )

        n = int(arrays.pop("_n_samples"))
        obs_dim = int(arrays["observations"].shape[1]) if n > 0 else 0
        act_dim = int(arrays["actions"].shape[1]) if n > 0 else 0

        meta = {
            "case_type": case_type,
            "n_samples": n,
            "scenarios_per_case": args.scenarios_per_case,
            "episodes_per_scenario": args.episodes_per_scenario,
            "episode_steps": args.episode_steps,
            "gnb_ids": list(GNB_IDS),
            "obs_dim": obs_dim,
            "action_dim": act_dim,
            "load_signal": "demanded PRB (upper_demand_prbs or _last_demand_profile.target_load)",
            "action_units": "dB A3 offsets — lower heuristic for all 3 gNBs",
            "upper_heuristic": "compute_upper_expert_directional_bias",
            "lower_heuristic": "strong_directional_heuristic_local_executor",
            "admission_control": "begin_safe_admission_window per step",
        }
        arrays["metadata_json"] = json.dumps(json_safe(meta))

        out_path = out_dir / f"{case_type}.npz"
        np.savez_compressed(out_path, **arrays)
        per_case_summary[case_type] = {k: v for k, v in meta.items() if k != "metadata_json"}
        print(f"  → {out_path}  ({n} samples, obs_dim={obs_dim})", flush=True)

    global_meta = {
        "case_types": list(case_types_to_run),
        "per_case": per_case_summary,
        "seed": args.seed,
        "demand_seed": args.demand_seed,
        "scenarios_per_case": args.scenarios_per_case,
        "episodes_per_scenario": args.episodes_per_scenario,
        "episode_steps": args.episode_steps,
        "min_demand_load": args.min_demand_load,
        "max_demand_load": args.max_demand_load,
        "safe_total_gnb_load": args.safe_total_gnb_load,
        "min_total_gnb_load": args.min_total_gnb_load,
        "placement_jitter_m": args.placement_jitter_m,
        "randomize_traffic_profiles": not args.no_randomize_traffic,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(json_safe(global_meta), indent=2))
    print(f"\nDone. {len(case_types_to_run)} case types saved to {out_dir}/")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
