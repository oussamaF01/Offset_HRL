#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from local_a3_agent_wrapper import normalize_slice_type, quantize_a3_offset
from local_a3_training_env import UpperScenarioLowerEnv
from strong_heuristic_local_executor import strong_directional_heuristic_local_executor


DEFAULT_SCENARIOS = (
    "jain_balance_controllable,"
    "jain_control_urllc,"
    "jain_control_mmtc,"
    "jain_control_mixed,"
    "jain_control_embb_urllc,"
    "jain_control_embb_mmtc,"
    "jain_control_urllc_mmtc,"
    "jain_control_outer_congested,"
    "jain_control_embb_urllc_impossible"
)
DEFAULT_SLICE_TYPES = ("eMBB", "URLLC", "mMTC")


def json_safe(obj):
    if isinstance(obj, dict):
        return {
            "|".join(map(str, k)) if isinstance(k, tuple) else str(k): json_safe(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def make_lower_env(
    *,
    seed: int = 7,
    controlled_gnb_id: int = 1,
    training_scenarios: str = DEFAULT_SCENARIOS,
    scenario_selection: str = "cycle",
    episode_steps: int = 40,
    action_hold_steps: int = 5,
    bias_hold_steps: int = 20,
    max_offset_change_db: float = 2.0,
    max_handovers_per_local_step: int = 3,
    print_scenarios: bool = False,
    scenario_mode: str = "curriculum",
    num_random_scenarios: int = 0,
    demand_seed: int | None = None,
    min_demand_load: float = 0.05,
    max_demand_load: float = 0.95,
    safe_total_gnb_load: float = 0.65,
    min_total_gnb_load: float = 0.25,
    randomize_traffic_profiles: bool = True,
    traffic_bit_rate_scale_min: float = 0.5,
    traffic_bit_rate_scale_max: float = 2.5,
    traffic_packet_size_scale_min: float = 0.75,
    traffic_packet_size_scale_max: float = 1.5,
    placement_jitter_m: float = 18.0,
) -> UpperScenarioLowerEnv:
    return UpperScenarioLowerEnv(
        seed=seed,
        controlled_gnb_id=controlled_gnb_id,
        training_scenarios=training_scenarios,
        scenario_selection=scenario_selection,
        episode_steps=episode_steps,
        action_hold_steps=action_hold_steps,
        bias_hold_steps=bias_hold_steps,
        max_offset_change_db=max_offset_change_db,
        max_handovers_per_local_step=max_handovers_per_local_step,
        print_scenarios=print_scenarios,
        scenario_mode=scenario_mode,
        num_random_scenarios=num_random_scenarios,
        demand_seed=demand_seed,
        min_demand_load=min_demand_load,
        max_demand_load=max_demand_load,
        safe_total_gnb_load=safe_total_gnb_load,
        min_total_gnb_load=min_total_gnb_load,
        randomize_traffic_profiles=randomize_traffic_profiles,
        traffic_bit_rate_scale_min=traffic_bit_rate_scale_min,
        traffic_bit_rate_scale_max=traffic_bit_rate_scale_max,
        traffic_packet_size_scale_min=traffic_packet_size_scale_min,
        traffic_packet_size_scale_max=traffic_packet_size_scale_max,
        placement_jitter_m=placement_jitter_m,
    )


def extract_demand_load_matrix(env: UpperScenarioLowerEnv, info: Mapping[str, Any] | None = None) -> Tuple[np.ndarray, str]:
    """Demanded PRB load matrix [gNB, slice], preferring live upper_demand_prbs."""
    info = dict(info or {})
    shape = (len(env.gnb_ids), len(env.slice_types))
    upper_env = getattr(env, "upper_env", None)
    for method_name in ("_load_matrix", "_persistent_demand_load_matrix"):
        if upper_env is not None and hasattr(upper_env, method_name):
            try:
                arr = np.asarray(getattr(upper_env, method_name)(), dtype=float)
                if arr.shape == shape:
                    return np.nan_to_num(arr, nan=0.0), f"upper_env.{method_name}"
            except Exception:
                pass
    for key in ("demand_load_matrix_end", "demand_load_matrix_start", "demand_load_matrix", "load_matrix"):
        if key in info:
            arr = np.asarray(info[key], dtype=float)
            if arr.shape == shape:
                return np.nan_to_num(arr, nan=0.0), f"info.{key}"
    profile = getattr(getattr(env, "base_env", None), "_last_demand_profile", None)
    if isinstance(profile, dict):
        arr = np.zeros(shape, dtype=float)
        ok = False
        for gi, gid in enumerate(env.gnb_ids):
            for si, st in enumerate(env.slice_types):
                item = profile.get((int(gid), normalize_slice_type(st)))
                if isinstance(item, dict) and "target_load" in item:
                    arr[gi, si] = float(item["target_load"])
                    ok = True
        if ok:
            return arr, "base_env._last_demand_profile.target_load"
    return useful_load_matrix(env, info), "useful_prb_fallback"


def useful_load_matrix(env: UpperScenarioLowerEnv, info: Mapping[str, Any] | None = None) -> np.ndarray:
    info = dict(info or {})
    upper_env = getattr(env, "upper_env", None)
    if upper_env is not None and hasattr(upper_env, "_current_useful_load_matrix"):
        try:
            return np.asarray(upper_env._current_useful_load_matrix(), dtype=float)
        except Exception:
            pass
    base_env = getattr(env, "base_env", None)
    if base_env is not None and hasattr(base_env, "get_window_average_slice_loads"):
        loads = base_env.get_window_average_slice_loads()
        return np.asarray(
            [[float(loads.get((gid, st), 0.0)) for st in env.slice_types] for gid in env.gnb_ids],
            dtype=float,
        )
    return np.zeros((len(env.gnb_ids), len(env.slice_types)), dtype=float)


def demand_profile_to_matrix(env: UpperScenarioLowerEnv, profile: Mapping | None = None) -> np.ndarray:
    profile = (
        env.base_env.get_demand_prb_loads()
        if profile is None and hasattr(env.base_env, "get_demand_prb_loads")
        else (profile or getattr(env.base_env, "_last_demand_profile", {}))
    )
    matrix = np.zeros((len(env.gnb_ids), len(env.slice_types)), dtype=float)
    for gi, gid in enumerate(env.gnb_ids):
        for si, st in enumerate(env.slice_types):
            item = profile.get((int(gid), normalize_slice_type(st)), {})
            if isinstance(item, Mapping):
                matrix[gi, si] = float(item.get("target_load", 0.0))
    return matrix


def _heuristic_arrays(env: UpperScenarioLowerEnv, demand_load_matrix: np.ndarray):
    all_ues = [
        ue for ue in env.base_env.get_all_ues()
        if getattr(ue, "connected", False) and getattr(ue, "serving_gnb", None) is not None
    ]
    gnb_ids = list(env.gnb_ids)
    gnb_index = {int(gid): i for i, gid in enumerate(gnb_ids)}
    slice_index = {normalize_slice_type(st): i for i, st in enumerate(env.slice_types)}
    ue_slice = np.array([
        slice_index.get(normalize_slice_type(getattr(ue, "slice_type", "eMBB")), 0)
        for ue in all_ues
    ], dtype=int)
    ue_serving = np.array([
        gnb_index.get(int(getattr(ue, "serving_gnb", 0)), 0)
        for ue in all_ues
    ], dtype=int)
    rsrp = np.full((len(all_ues), len(gnb_ids)), -120.0, dtype=float)
    for u_idx, ue in enumerate(all_ues):
        for g_idx, gid in enumerate(gnb_ids):
            try:
                gnb = env.base_env._get_gnb_by_id(int(gid))
                if gnb is not None:
                    rsrp[u_idx, g_idx] = float(
                        env.base_env._compute_link_metrics(gnb, ue).get("rx_power_dbm", -120.0)
                    )
            except Exception:
                pass
    sla_flags = env.base_env.get_slice_sla_flags() if hasattr(env.base_env, "get_slice_sla_flags") else {}
    sla = np.asarray(
        [[float(sla_flags.get((int(gid), st), 0.0)) for st in env.slice_types] for gid in gnb_ids],
        dtype=float,
    )
    zeros = np.zeros((len(gnb_ids), len(env.slice_types)), dtype=float)
    return gnb_ids, gnb_index, ue_slice, ue_serving, rsrp, np.asarray(demand_load_matrix, dtype=float), sla, zeros, zeros


def compute_expert_action(
    env: UpperScenarioLowerEnv,
    *,
    gnb_id: int | None = None,
    bias: Mapping[Tuple[int, int, str], float] | None = None,
    demand_load_matrix: np.ndarray | None = None,
    release_floor_db: float = -2.0,
) -> Tuple[np.ndarray, Dict]:
    gid = int(env.controlled_gnb_id if gnb_id is None else gnb_id)
    local_env = env.local_envs[gid]
    if bias is None:
        bias = dict(local_env.global_bias)
    if demand_load_matrix is None:
        demand_load_matrix, _ = extract_demand_load_matrix(env)

    gnb_ids, gnb_index, ue_slice, ue_serving, rsrp, load, sla, ho_fail, pingpong = _heuristic_arrays(env, demand_load_matrix)
    g_idx = gnb_index[gid]
    max_nb = len(local_env.neighbor_ids)
    n_slices = len(env.slice_types)
    B = np.zeros((len(gnb_ids), max_nb, n_slices), dtype=float)
    for nb_slot, nb_id in enumerate(local_env.neighbor_ids):
        for s_idx, st in enumerate(env.slice_types):
            B[g_idx, nb_slot, s_idx] = float(bias.get((gid, int(nb_id), st), 0.0))

    applied = local_env.get_applied_offsets()
    prev_vec = np.asarray(
        [float(applied.get((int(nb), st), 0.0)) for nb in local_env.neighbor_ids for st in env.slice_types],
        dtype=float,
    )
    prev_full = np.zeros_like(B, dtype=float)
    prev_full[g_idx] = prev_vec.reshape(max_nb, n_slices)
    neighbor_graph = {g_idx: [gnb_index[int(nb)] for nb in local_env.neighbor_ids]}
    offsets_full, debug = strong_directional_heuristic_local_executor(
        B=B,
        prev_offsets=prev_full,
        ue_slice=ue_slice,
        ue_serving_gnb=ue_serving,
        rsrp_matrix=rsrp,
        neighbor_graph=neighbor_graph,
        load=load,
        sla_violation=sla,
        ho_failure_ratio=ho_fail,
        pingpong_ratio=pingpong,
        slice_types=env.slice_types,
        allow_extended_negative_offsets=False,
        return_debug=True,
    )
    raw = offsets_full[g_idx].reshape(-1)
    action = raw.copy()
    flat_idx = 0
    for nb_id in local_env.neighbor_ids:
        nb_idx = gnb_index[int(nb_id)]
        for s_idx, _st in enumerate(env.slice_types):
            dbg = debug.get((g_idx, nb_idx, s_idx), {})
            release_signal = min(float(dbg.get("raw_offset_db", 0.0)), float(dbg.get("proto_offset_db", 0.0)))
            if dbg.get("mode") == "release" and release_signal < -1e-9:
                action[flat_idx] = min(action[flat_idx], float(release_floor_db))
            flat_idx += 1
    return np.asarray([quantize_a3_offset(float(v)) for v in action], dtype=np.float32).reshape(local_env.action_space.shape), debug


def set_current_upper_bias(env: UpperScenarioLowerEnv, force: bool = True) -> Dict:
    bias = env._compute_upper_expert_bias(force=force)
    for local_env in env.local_envs.values():
        local_env.set_global_bias(bias)
    return bias


def collect_step_metrics(env: UpperScenarioLowerEnv, info: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    info = dict(info or {})
    demand, source = extract_demand_load_matrix(env, info)
    useful = useful_load_matrix(env, info)
    row: Dict[str, Any] = {"demand_load_source": source}
    for gi, gid in enumerate(env.gnb_ids):
        row[f"demand_total_g{gid}"] = float(np.sum(demand[gi]))
        row[f"useful_total_g{gid}"] = float(np.sum(useful[gi]))
        for si, st in enumerate(env.slice_types):
            row[f"demand_g{gid}_{st}"] = float(demand[gi, si])
            row[f"useful_g{gid}_{st}"] = float(useful[gi, si])
    return row


def final_total_imbalance(row: Mapping[str, Any], gnb_ids: Sequence[int] = (0, 1, 2)) -> float:
    values = np.asarray([float(row.get(f"demand_total_g{gid}", 0.0)) for gid in gnb_ids], dtype=float)
    return float(np.std(values))


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (128, 128), action_scale: float = 6.0):
        super().__init__()
        layers = []
        last = int(obs_dim)
        for width in hidden_dims:
            layers.append(nn.Linear(last, int(width)))
            layers.append(nn.ReLU())
            last = int(width)
        layers.append(nn.Linear(last, int(action_dim)))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.action_scale = float(action_scale)

    def forward_normalized(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.float())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(obs) * self.action_scale


def load_bc_policy(path: str | Path, map_location: str | torch.device = "cpu") -> Tuple[BCPolicy, Dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    config = dict(payload.get("config", {}))
    policy = BCPolicy(
        obs_dim=int(config["obs_dim"]),
        action_dim=int(config["action_dim"]),
        hidden_dims=tuple(config.get("hidden_dims", (128, 128))),
        action_scale=float(config.get("action_scale", 6.0)),
    )
    policy.load_state_dict(payload["model_state_dict"])
    policy.eval()
    return policy, config


def save_json(path: str | Path, data: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_safe(dict(data)), indent=2), encoding="utf-8")
