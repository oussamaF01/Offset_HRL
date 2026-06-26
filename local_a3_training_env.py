#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import math
from typing import Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

from local_a3_agent_wrapper import LocalA3OffsetEnv, normalize_slice_type, quantize_a3_offset
from local_a3_training_scenarios import (
    DEFAULT_LOCAL_A3_TRAINING_SCENARIOS,
    NEUTRAL_SLICE_SCENARIO,
    EpisodeTrainingScenario,
    choose_training_scenario,
)
from scenario_creator import create_multignb_env
from slice_ran import Packet
from strong_heuristic_local_executor import strong_directional_heuristic_local_executor

# Deferred import to avoid circular dependencies at module load time.
def _import_global_ppo_env():
    from global_ppo_3gnb_env import GlobalPPO3GNBEnv
    return GlobalPPO3GNBEnv


DEFAULT_GNB_CONFIGS = [
    {"id": 0, "x": 0.0, "y": 0.0, "coverage_radius": 500.0, "carrier_id": 0, "n_prbs": 100},
    {"id": 1, "x": 450.0, "y": 0.0, "coverage_radius": 500.0, "carrier_id": 0, "n_prbs": 100},
]

THREE_GNB_LOCAL_CONFIGS = [
    {"id": 0, "x": 0.0, "y": 0.0, "coverage_radius": 500.0, "carrier_id": 0, "n_prbs": 100},
    {"id": 1, "x": -450.0, "y": 0.0, "coverage_radius": 500.0, "carrier_id": 0, "n_prbs": 100},
    {"id": 2, "x": 450.0, "y": 0.0, "coverage_radius": 500.0, "carrier_id": 0, "n_prbs": 100},
]

DEFAULT_BIAS_CASE_PROBS = {
    "offload": 0.30,
    "neutral": 0.25,
    "retain": 0.25,
    "risky_offload": 0.20,
}


class LocalA3RuleBiasTrainingEnv(gym.Env):
    """
    Stage-1 local-agent training environment.

    This wrapper randomizes a small static scenario at every reset and trains one
    LocalA3OffsetEnv with a rule-based fake global bias. It is intentionally
    simple: no SUMO, no global PPO, and usually one slice such as eMBB.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int = 7,
        gnb_id: int = 0,
        neighbor_ids: Sequence[int] = (1,),
        slice_types: Sequence[str] = ("eMBB",),
        scenario_idx: int = 4,
        episode_steps: int = 40,
        local_ues_range: Tuple[int, int] = (2, 3),
        neighbor_ues_range: Tuple[int, int] = (2, 3),
        gnb_configs: Optional[Sequence[Dict]] = None,
        radio_substeps: int = 10,
        steps_per_action: int = 1,
        local_spawn_radius: float = 190.0,
        neighbor_spawn_radius: float = 230.0,
        border_ue_fraction: float = 0.35,
        border_parallel_jitter: float = 70.0,
        border_perp_jitter: float = 160.0,
        force_serving: bool = True,
        balance_bias_cases: bool = True,
        bias_case_probs: Optional[Dict[str, float]] = None,
        max_case_sampling_attempts: int = 20,
        action_hold_steps: int = 5,
        bias_hold_steps: int = 20,
        max_offset_change_db: float = 2.0,
        training_scenarios: Optional[Sequence[EpisodeTrainingScenario]] = None,
        scenario_hold_episodes: int = 1,
        print_scenarios: bool = True,
        heuristic_gnb_ids: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.seed_value = int(seed)
        self.rng = np.random.default_rng(self.seed_value)
        self.gnb_id = int(gnb_id)
        self.neighbor_ids = tuple(int(n) for n in neighbor_ids)
        self.slice_types = tuple(normalize_slice_type(s) for s in slice_types)
        self.scenario_idx = int(scenario_idx)
        self.episode_steps = int(episode_steps)
        self.local_ues_range = tuple(int(v) for v in local_ues_range)
        self.neighbor_ues_range = tuple(int(v) for v in neighbor_ues_range)
        self.gnb_configs = [dict(cfg) for cfg in (gnb_configs or DEFAULT_GNB_CONFIGS)]
        self.radio_substeps = int(radio_substeps)
        self.steps_per_action = int(steps_per_action)
        self.local_spawn_radius = float(local_spawn_radius)
        self.neighbor_spawn_radius = float(neighbor_spawn_radius)
        self.border_ue_fraction = float(np.clip(border_ue_fraction, 0.0, 1.0))
        self.border_parallel_jitter = float(max(border_parallel_jitter, 0.0))
        self.border_perp_jitter = float(max(border_perp_jitter, 0.0))
        self.force_serving = bool(force_serving)
        self.balance_bias_cases = bool(balance_bias_cases)
        self.bias_case_probs = self._normalize_case_probs(bias_case_probs)
        self.training_scenarios = tuple(
            training_scenarios or DEFAULT_LOCAL_A3_TRAINING_SCENARIOS
        )
        self.scenario_hold_episodes = max(1, int(scenario_hold_episodes))
        self.print_scenarios = bool(print_scenarios)
        self.max_case_sampling_attempts = max(1, int(max_case_sampling_attempts))
        self.action_hold_steps = max(1, int(action_hold_steps))
        self.bias_hold_steps = max(1, int(bias_hold_steps))
        self.max_offset_change_db = max(0.0, float(max_offset_change_db))
        self._elapsed_steps = 0
        self._reset_count = 0
        self._bias_case = "unbalanced"
        self._current_scenario = None
        self._held_training_scenario = None
        self._held_scenario_episodes_left = 0
        self._bias_case_matched = False
        self._case_sampling_attempts = 0
        self._action_hold_counter = 0
        self._held_proto_action = None
        self._held_action = None
        self._last_action_debug = self._empty_action_debug()
        self._bias_hold_counter = 0
        self._held_rule_bias = None
        self._last_bias_changed = False
        self._last_demand_profile = {}

        self.base_env = self._make_base_env()
        self.local_env = LocalA3OffsetEnv(
            self.base_env,
            gnb_id=self.gnb_id,
            neighbor_ids=self.neighbor_ids,
            slice_types=self.slice_types,
            steps_per_action=self.steps_per_action,
            ttt=1,
        )

        # Heuristic-controlled gNBs: defaults to all gNBs except the TD3 gNB.
        all_gnb_ids = [int(cfg["id"]) for cfg in self.gnb_configs]
        self.heuristic_gnb_ids = tuple(
            int(g) for g in (heuristic_gnb_ids if heuristic_gnb_ids is not None
                              else [g for g in all_gnb_ids if g != self.gnb_id])
        )
        self._heuristic_prev_offsets: Dict[int, np.ndarray] = {}
        self._heuristic_local_envs: Dict[int, LocalA3OffsetEnv] = {}
        for gid in self.heuristic_gnb_ids:
            h_neighbors = [g for g in all_gnb_ids if g != gid]
            h_env = LocalA3OffsetEnv(
                self.base_env,
                gnb_id=gid,
                neighbor_ids=h_neighbors,
                slice_types=self.slice_types,
                steps_per_action=1,
                ttt=1,
            )
            self._heuristic_local_envs[gid] = h_env
            n_dirs = len(h_neighbors) * len(self.slice_types)
            self._heuristic_prev_offsets[gid] = np.zeros(n_dirs, dtype=float)

        self.action_space = self.local_env.action_space
        self.observation_space = self.local_env.observation_space

    def _make_base_env(self):
        return create_multignb_env(
            rng=self.rng,
            n=self.scenario_idx,
            gnb_configs=self.gnb_configs,
            slots_per_step=5,
            L1_level=False,
            step_dt=1e-3,
            mobility_dt=0.0,
            radio_substeps=self.radio_substeps,
            max_episode_steps=self.episode_steps + 5,
            use_sumo_mobility=False,
        )

    def reset(self, *, seed=None, options=None):
        self._reset_count += 1
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.base_env._rng = np.random.default_rng(seed)

        self._elapsed_steps = 0
        if self.balance_bias_cases:
            if self._held_training_scenario is None or self._held_scenario_episodes_left <= 0:
                self._held_training_scenario = choose_training_scenario(
                    self.rng,
                    self.training_scenarios,
                )
                self._held_scenario_episodes_left = self.scenario_hold_episodes
            self._current_scenario = self._held_training_scenario
            self._held_scenario_episodes_left -= 1
        else:
            self._current_scenario = None
        self._bias_case = (
            f"{self._current_scenario.name}|{self._scenario_case_summary()}"
            if self._current_scenario is not None
            else "unbalanced"
        )
        self._bias_case_matched = False
        self._case_sampling_attempts = 0
        self._action_hold_counter = 0
        self._held_proto_action = None
        self._held_action = None
        self._last_action_debug = self._empty_action_debug()
        self._bias_hold_counter = 0
        self._held_rule_bias = None
        self._last_bias_changed = False

        for gid, h_env in self._heuristic_local_envs.items():
            for key in h_env._offsets:
                h_env._offsets[key] = 0.0
                h_env._prev_proto_offsets[key] = 0.0
            h_env._ttt_counters.clear()
            n_dirs = len(h_env.neighbor_ids) * len(self.slice_types)
            self._heuristic_prev_offsets[gid] = np.zeros(n_dirs, dtype=float)

        last_obs = None
        last_info = {}
        attempts = self.max_case_sampling_attempts if self.balance_bias_cases else 1

        for attempt in range(1, attempts + 1):
            self._case_sampling_attempts = attempt
            self.base_env.clear_ues(reset_ids=True)
            self._spawn_static_ues()
            obs, info = self.local_env.reset(seed=seed, options=options)
            self._reapply_forced_serving()
            self._apply_case_demand_profiles()
            self._seed_case_prbs()
            self.base_env._invalidate_metric_caches()

            self.local_env.set_global_bias(self._held_or_new_rule_bias(force=True))
            obs = self.local_env._build_observation()
            self._bias_case_matched = (
                True
                if not self.balance_bias_cases
                else self._case_matches_all_loads()
            )
            last_obs = obs
            last_info = info
            if self._bias_case_matched:
                break

        info = self._augment_info(last_info)
        if self.print_scenarios:
            self._print_scenario_info(info)
        obs = last_obs if last_obs is not None else self.local_env._build_observation()
        return obs, info

    def _run_heuristic_gnbs(self):
        """Run strong_directional_heuristic_local_executor for every heuristic gNB."""
        if not self._heuristic_local_envs:
            return

        # Collect UE arrays once (shared across all heuristic gNBs).
        all_ues = [
            ue for ue in self.base_env.get_all_ues()
            if ue.connected and ue.serving_gnb is not None
        ]
        if not all_ues:
            return

        all_gnb_ids = [int(cfg["id"]) for cfg in self.gnb_configs]
        n_gnbs = len(all_gnb_ids)
        gnb_index = {gid: i for i, gid in enumerate(all_gnb_ids)}
        slice_index = {st: i for i, st in enumerate(self.slice_types)}

        ue_slice_arr = np.array([
            slice_index.get(normalize_slice_type(getattr(ue, "slice_type", "eMBB")), 0)
            for ue in all_ues
        ], dtype=int)
        ue_serving_arr = np.array([
            gnb_index.get(int(ue.serving_gnb), 0) for ue in all_ues
        ], dtype=int)

        # RSRP matrix [n_ues, n_gnbs]
        rsrp = np.full((len(all_ues), n_gnbs), -120.0, dtype=float)
        for u_idx, ue in enumerate(all_ues):
            for g_idx, gid in enumerate(all_gnb_ids):
                gnb = self.base_env._get_gnb_by_id(gid)
                if gnb is not None:
                    try:
                        rsrp[u_idx, g_idx] = self.base_env._compute_link_metrics(gnb, ue)["rx_power_dbm"]
                    except Exception:
                        pass

        # Load and SLA arrays [n_gnbs, n_slices]
        load = np.array([
            [self.base_env.get_window_average_slice_loads().get((gid, st), 0.0)
             for st in self.slice_types]
            for gid in all_gnb_ids
        ], dtype=float)
        sla_flags = self.base_env.get_slice_sla_flags() if hasattr(self.base_env, "get_slice_sla_flags") else {}
        sla_arr = np.array([
            [float(sla_flags.get((gid, st), 0.0)) for st in self.slice_types]
            for gid in all_gnb_ids
        ], dtype=float)
        ho_fail = np.zeros((n_gnbs, len(self.slice_types)), dtype=float)
        pp_arr = np.zeros((n_gnbs, len(self.slice_types)), dtype=float)

        current_bias = dict(self.local_env.global_bias)

        for gid, h_env in self._heuristic_local_envs.items():
            g_idx = gnb_index[gid]
            h_neighbors = h_env.neighbor_ids
            max_nb = len(h_neighbors)
            n_slices = len(self.slice_types)

            # Build per-direction bias and prev_offsets [1, max_nb, n_slices]
            B = np.zeros((n_gnbs, max_nb, n_slices), dtype=float)
            prev = self._heuristic_prev_offsets[gid].reshape(max_nb, n_slices)
            prev_full = np.zeros((n_gnbs, max_nb, n_slices), dtype=float)
            prev_full[g_idx] = prev

            for nb_slot, nb_id in enumerate(h_neighbors):
                for s_idx, st in enumerate(self.slice_types):
                    B[g_idx, nb_slot, s_idx] = float(
                        current_bias.get((gid, nb_id, st), 0.0)
                    )

            neighbor_graph = {g_idx: [gnb_index[nb] for nb in h_neighbors]}

            offsets_full, _ = strong_directional_heuristic_local_executor(
                B=B,
                prev_offsets=prev_full,
                ue_slice=ue_slice_arr,
                ue_serving_gnb=ue_serving_arr,
                rsrp_matrix=rsrp,
                neighbor_graph=neighbor_graph,
                load=load,
                sla_violation=sla_arr,
                ho_failure_ratio=ho_fail,
                pingpong_ratio=pp_arr,
                slice_types=self.slice_types,
                return_debug=True,
            )

            # offsets_full[g_idx, nb_slot, s_idx] — extract and apply
            offsets_vec = offsets_full[g_idx].reshape(-1)  # [max_nb * n_slices]
            self._heuristic_prev_offsets[gid] = offsets_vec.copy()
            h_env._apply_proto_offsets(offsets_vec)
            h_env._execute_a3_handovers()

            # Push offsets into base env so A3 conditions use the right values
            for nb_slot, nb_id in enumerate(h_neighbors):
                for s_idx, st in enumerate(self.slice_types):
                    self.base_env.set_a3_offset(
                        gid, nb_id, st, float(offsets_full[g_idx, nb_slot, s_idx])
                    )

    def step(self, action):
        self._run_heuristic_gnbs()
        held_action = self._held_or_new_action(action)
        obs, reward, terminated, truncated, info = self.local_env.step(held_action)
        post_action_slice_loads = self.base_env.get_slice_loads()
        post_action_spawn_counts = self._slice_counts()
        self._apply_case_demand_profiles()
        self._seed_case_prbs()
        self.base_env._invalidate_metric_caches()
        self.local_env.set_global_bias(self._held_or_new_rule_bias())
        obs = self.local_env._build_observation()
        self._elapsed_steps += 1
        truncated = bool(truncated or self._elapsed_steps >= self.episode_steps)
        self._bias_case_matched = (
            True
            if not self.balance_bias_cases
            else self._case_matches_all_loads()
        )
        info = self._augment_info(
            info,
            post_action_slice_loads=post_action_slice_loads,
            post_action_spawn_counts=post_action_spawn_counts,
        )
        return obs, reward, terminated, truncated, info

    def close(self):
        self.base_env.close()

    def _normalize_case_probs(self, probs: Optional[Dict[str, float]]) -> Dict[str, float]:
        merged = dict(DEFAULT_BIAS_CASE_PROBS)
        if probs:
            for name, value in dict(probs).items():
                if name in merged:
                    merged[name] = max(float(value), 0.0)

        total = sum(merged.values())
        if total <= 0.0:
            return dict(DEFAULT_BIAS_CASE_PROBS)
        return {name: value / total for name, value in merged.items()}

    def _sample_bias_case(self) -> str:
        names = list(self.bias_case_probs)
        probs = np.asarray([self.bias_case_probs[name] for name in names], dtype=float)
        probs = probs / max(float(probs.sum()), 1e-12)
        return str(self.rng.choice(names, p=probs))

    def _slice_scenario(self, slice_type: str):
        if self._current_scenario is None:
            return NEUTRAL_SLICE_SCENARIO
        return self._current_scenario.for_slice(slice_type)

    def _slice_case_name(self, slice_type: str) -> str:
        return str(self._slice_scenario(slice_type).case)

    def _scenario_case_summary(self) -> str:
        if self._current_scenario is None:
            return "unbalanced"
        return ",".join(
            f"{slice_type}:{self._slice_case_name(slice_type)}"
            for slice_type in self.slice_types
        )

    def _case_ue_ranges(self, case_name: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if not self.balance_bias_cases:
            return self.local_ues_range, self.neighbor_ues_range
        # Keep Stage-1 fake local training intentionally small: the selected
        # case is applied per slice so multi-slice training has signal for each
        # local action dimension.
        if case_name == "offload":
            return (4, 4), (1, 1)
        if case_name == "neutral":
            return (3, 3), (2, 2)
        if case_name == "retain":
            return (1, 1), (4, 4)
        if case_name == "risky_offload":
            return (3, 3), (2, 2)
        return self.local_ues_range, self.neighbor_ues_range

    def _spawn_static_ues(self):
        controlled = self.base_env._get_gnb_by_id(self.gnb_id)
        if controlled is None:
            raise ValueError(f"Unknown controlled gNB id {self.gnb_id}")

        for slice_type in self.slice_types:
            spec = self._slice_scenario(slice_type)
            if self.balance_bias_cases:
                n_local = int(spec.local_ues)
                n_neighbor = int(spec.neighbor_ues)
            else:
                n_local = int(self.rng.integers(self.local_ues_range[0], self.local_ues_range[1] + 1))
                n_neighbor = int(self.rng.integers(self.neighbor_ues_range[0], self.neighbor_ues_range[1] + 1))

            for _ in range(n_local):
                x, y = self._sample_local_point(controlled, spec)
                ue_id = self.base_env.add_ue(x=x, y=y, vx=0.0, vy=0.0, slice_type=slice_type)
                setattr(self.base_env.get_ue(ue_id), "_training_forced_gnb_id", self.gnb_id)
                if self.force_serving:
                    self._force_attach(ue_id, self.gnb_id)

            for neighbor_id in self.neighbor_ids:
                neighbor = self.base_env._get_gnb_by_id(neighbor_id)
                if neighbor is None:
                    continue
                for _ in range(n_neighbor):
                    radius = float(spec.neighbor_radius if self.balance_bias_cases else self.neighbor_spawn_radius)
                    x, y = self._sample_point_around(neighbor, radius)
                    ue_id = self.base_env.add_ue(x=x, y=y, vx=0.0, vy=0.0, slice_type=slice_type)
                    setattr(self.base_env.get_ue(ue_id), "_training_forced_gnb_id", int(neighbor_id))
                    if self.force_serving:
                        self._force_attach(ue_id, neighbor_id)

    def _sample_slice_type(self) -> str:
        return str(self.rng.choice(self.slice_types))

    def _sample_point_around(self, gnb, radius: float):
        r = radius * np.sqrt(float(self.rng.random()))
        theta = 2.0 * np.pi * float(self.rng.random())
        return float(gnb.x + r * np.cos(theta)), float(gnb.y + r * np.sin(theta))

    def _sample_local_point(self, controlled, spec=None):
        border_fraction = self.border_ue_fraction
        local_radius = self.local_spawn_radius
        if self.balance_bias_cases and spec is not None:
            border_fraction = float(np.clip(spec.border_fraction, 0.0, 1.0))
            local_radius = float(spec.local_radius)

        if self.neighbor_ids and float(self.rng.random()) < border_fraction:
            neighbor_id = int(self.rng.choice(self.neighbor_ids))
            neighbor = self.base_env._get_gnb_by_id(neighbor_id)
            if neighbor is not None:
                return self._sample_border_point(controlled, neighbor, spec)
        return self._sample_point_around(controlled, local_radius)

    def _sample_border_point(self, serving_gnb, neighbor_gnb, spec=None):
        sx, sy = float(serving_gnb.x), float(serving_gnb.y)
        nx, ny = float(neighbor_gnb.x), float(neighbor_gnb.y)
        dx, dy = nx - sx, ny - sy
        distance = float(np.hypot(dx, dy))
        if distance <= 1e-9:
            return self._sample_point_around(serving_gnb, self.local_spawn_radius)

        ux, uy = dx / distance, dy / distance
        px, py = -uy, ux
        midpoint_x = sx + 0.5 * dx
        midpoint_y = sy + 0.5 * dy

        # Keep border UEs slightly on the controlled side, but close enough that
        # negative A3 offsets can trigger meaningful handovers.
        parallel_jitter = self.border_parallel_jitter
        perp_jitter = self.border_perp_jitter
        if self.balance_bias_cases and spec is not None:
            parallel_jitter = float(spec.border_parallel_jitter)
            perp_jitter = float(spec.border_perp_jitter)
        along = float(self.rng.normal(loc=-0.08 * distance, scale=parallel_jitter))
        perp = float(self.rng.normal(loc=0.0, scale=perp_jitter))
        x = midpoint_x + along * ux + perp * px
        y = midpoint_y + along * uy + perp * py
        return float(x), float(y)

    def _force_attach(self, ue_id: int, gnb_id: int):
        ue = self.base_env.get_ue(ue_id)
        old_gnb = self.base_env._get_gnb_by_id(ue.serving_gnb)
        new_gnb = self.base_env._get_gnb_by_id(gnb_id)
        if new_gnb is None:
            return
        if old_gnb is not None:
            old_gnb.detach_ue(ue_id)
        attached = new_gnb.attach_ue(ue)
        ue.serving_gnb = int(gnb_id) if attached else None
        ue.connected = bool(attached)
        self.base_env._last_serving_gnb[ue_id] = ue.serving_gnb
        self.base_env._prev_serving_gnb[ue_id] = None
        self.base_env._invalidate_metric_caches()

    def _reapply_forced_serving(self):
        if not self.force_serving:
            return
        for ue in list(getattr(self.base_env, "_ues", {}).values()):
            forced_gnb_id = getattr(ue, "_training_forced_gnb_id", None)
            if forced_gnb_id is not None:
                self._force_attach(int(ue.id), int(forced_gnb_id))

    def _slice_counts(self):
        counts = {
            (int(gnb.id), slice_type): 0
            for gnb in self.base_env.gnbs
            for slice_type in self.slice_types
        }
        for ue in self.base_env.get_all_ues():
            if not ue.connected or ue.serving_gnb is None:
                continue
            slice_type = normalize_slice_type(getattr(ue, "slice_type", "eMBB"))
            if slice_type in self.slice_types:
                counts[(int(ue.serving_gnb), slice_type)] += 1
        return counts

    def _connected_slice_ues(self, gnb_id: int, slice_type: str):
        wanted = normalize_slice_type(slice_type)
        return [
            ue
            for ue in self.base_env.get_all_ues()
            if ue.connected
            and ue.serving_gnb is not None
            and int(ue.serving_gnb) == int(gnb_id)
            and normalize_slice_type(getattr(ue, "slice_type", "eMBB")) == wanted
        ]

    def _set_slice_prb_load(self, gnb_id: int, slice_type: str, target_load: float):
        ues = self._connected_slice_ues(gnb_id, slice_type)
        if not ues:
            return

        budget = int(max(self.base_env.get_slice_prb_budget(gnb_id, slice_type), 0))
        target_used = int(round(float(np.clip(target_load, 0.0, 1.0)) * budget))
        per_ue = target_used // len(ues)
        remainder = target_used % len(ues)

        for idx, ue in enumerate(ues):
            ue.prbs = int(per_ue + (1 if idx < remainder else 0))

    def _set_ue_offered_bit_rate(self, ue, bit_rate: float):
        source = getattr(ue, "traffic_source", None)
        rate = max(float(bit_rate), 0.0)
        if source is None:
            return
        if hasattr(source, "set_bit_rate"):
            source.set_bit_rate(rate)
        else:
            source.bit_rate = rate
            if hasattr(source, "packet_size"):
                source.packet_size = rate * max(float(self.base_env.step_dt), 1e-9)

    def _ensure_ue_queue_floor(self, ue, target_bits: float):
        target_bits = max(float(target_bits), 0.0)
        current_queue = max(float(getattr(ue, "queue", 0.0)), 0.0)
        missing = int(math.ceil(max(target_bits - current_queue, 0.0)))
        if missing <= 0:
            return

        ue.queue = current_queue + missing
        if hasattr(ue, "packet_queue"):
            packet_id = int(getattr(ue, "_next_packet_id", 0))
            arrival_step = int(getattr(ue, "_step_counter", 0))
            arrival_time_s = float(ue.get_current_time_s()) if hasattr(ue, "get_current_time_s") else 0.0
            ue.packet_queue.append(
                Packet(
                    bits=missing,
                    arrival_step=arrival_step,
                    arrival_time_s=arrival_time_s,
                    packet_id=packet_id,
                )
            )
            ue._next_packet_id = packet_id + 1

    def _bits_per_prb(self, sinr_db: float) -> float:
        sinr_linear = max(10.0 ** (float(sinr_db) / 10.0), 1e-6)
        spectral_eff = math.log2(1.0 + sinr_linear)
        spectral_eff = min(max(spectral_eff, 0.0), 8.0)
        return 180e3 * max(float(self.base_env.step_dt), 1e-9) * spectral_eff

    def _set_slice_offered_load(self, gnb_id: int, slice_type: str, target_load: float):
        ues = self._connected_slice_ues(gnb_id, slice_type)
        if not ues:
            return

        budget = int(max(self.base_env.get_slice_prb_budget(gnb_id, slice_type), 0))
        target_used = int(round(float(np.clip(target_load, 0.0, 1.0)) * budget))
        per_ue = target_used // len(ues)
        remainder = target_used % len(ues)

        rates = []
        for idx, ue in enumerate(ues):
            target_prbs = int(per_ue + (1 if idx < remainder else 0))
            metrics = self.base_env.get_ue_radio_metrics(int(ue.id))
            sinr_db = float(metrics.get("sinr_db", self.base_env.disconnect_sinr_db))
            bits_per_prb = self._bits_per_prb(sinr_db)
            bit_rate = target_prbs * bits_per_prb / max(float(self.base_env.step_dt), 1e-9)
            self._set_ue_offered_bit_rate(ue, bit_rate)
            self._ensure_ue_queue_floor(ue, target_prbs * bits_per_prb)
            rates.append(bit_rate)

        key = (int(gnb_id), normalize_slice_type(slice_type))
        self._last_demand_profile[key] = {
            "target_load": float(np.clip(target_load, 0.0, 1.0)),
            "target_prbs": int(target_used),
            "ue_count": int(len(ues)),
            "mean_bit_rate": float(np.mean(rates)) if rates else 0.0,
        }

    def _apply_case_demand_profiles(self):
        if not self.balance_bias_cases or not self.slice_types:
            self._last_demand_profile = {}
            return

        self._last_demand_profile = {}
        for slice_type in self.slice_types:
            spec = self._slice_scenario(slice_type)
            self._set_slice_offered_load(self.gnb_id, slice_type, spec.local_load)
            for neighbor_id in self.neighbor_ids:
                self._set_slice_offered_load(neighbor_id, slice_type, spec.neighbor_load)

    def _seed_case_prbs(self):
        if not self.balance_bias_cases or not self.slice_types:
            return

        for slice_type in self.slice_types:
            spec = self._slice_scenario(slice_type)
            self._set_slice_prb_load(self.gnb_id, slice_type, spec.local_load)
            for neighbor_id in self.neighbor_ids:
                self._set_slice_prb_load(neighbor_id, slice_type, spec.neighbor_load)

    def _case_matches_loads(self, case_name: str, slice_type: str = "eMBB") -> bool:
        L_local = self.base_env.estimate_slice_load(self.gnb_id, slice_type)
        L_neighbors = [
            self.base_env.estimate_slice_load(neighbor_id, slice_type)
            for neighbor_id in self.neighbor_ids
        ]
        min_neighbor = min(L_neighbors) if L_neighbors else 1.0

        if case_name == "offload":
            return L_local > 0.75
        if case_name == "neutral":
            return 0.45 <= L_local <= 0.75
        if case_name == "retain":
            return L_local < 0.45
        if case_name == "risky_offload":
            return L_local > 0.75 and min_neighbor >= 0.70
        return True

    def _case_matches_all_loads(self) -> bool:
        return all(
            self._case_matches_loads(self._slice_case_name(slice_type), slice_type)
            for slice_type in self.slice_types
        )

    # Per-slice parameters for the proportional rule bias.
    # scale:    load-difference that maps to ±1 bias (smaller = more sensitive)
    # deadband: imbalances below this are ignored (outputs 0)
    # min_src:  src load must exceed this before a negative (offload) bias fires
    _RULE_BIAS_PARAMS = {
        "eMBB":  {"scale": 0.35, "deadband": 0.10, "min_src": 0.40},
        "URLLC": {"scale": 0.25, "deadband": 0.07, "min_src": 0.35},
        "mMTC":  {"scale": 0.40, "deadband": 0.12, "min_src": 0.45},
    }
    _RULE_BIAS_DEFAULT_PARAMS = {"scale": 0.35, "deadband": 0.10, "min_src": 0.40}

    def _rule_bias(self):
        # Proportional directional rule bias b_{src,tgt,s}.
        #
        # raw = -(src_load - tgt_load) / scale
        #   → large positive diff (src much more loaded) → strong negative bias (offload)
        #   → large negative diff (tgt much more loaded) → strong positive bias (retain)
        #
        # Gating:
        #   - Deadband suppresses noise from tiny imbalances.
        #   - min_src prevents recommending offload from a lightly loaded cell.
        #
        # Slice-awareness: URLLC reacts faster (smaller scale & deadband),
        # mMTC is more conservative (larger scale & deadband).
        bias = {}
        gnb_ids = [int(gnb.id) for gnb in self.base_env.gnbs]
        for src_id in gnb_ids:
            for tgt_id in gnb_ids:
                if src_id == tgt_id:
                    continue
                for slice_type in self.slice_types:
                    p = self._RULE_BIAS_PARAMS.get(slice_type, self._RULE_BIAS_DEFAULT_PARAMS)
                    src_load = self.base_env.estimate_slice_load(src_id, slice_type)
                    tgt_load = self.base_env.estimate_slice_load(tgt_id, slice_type)
                    diff = src_load - tgt_load

                    if abs(diff) < p["deadband"]:
                        value = 0.0
                    else:
                        raw = -diff / p["scale"]
                        # Veto offload recommendation when src itself is lightly loaded
                        if raw < 0.0 and src_load < p["min_src"]:
                            raw = 0.0
                        value = float(np.clip(raw, -1.0, 1.0))

                    bias[(src_id, tgt_id, slice_type)] = value
        return bias

    def _empty_action_debug(self) -> Dict[str, object]:
        return {
            "raw_action": 0.0,
            "target_offset": 0.0,
            "applied_offset": 0.0,
            "previous_offset": 0.0,
            "offset_delta": 0.0,
            "offset_changed": False,
            "action_hold_counter": 0,
            "action_hold_steps": self.action_hold_steps,
            "raw_actions": {},
            "target_offsets": {},
            "applied_offsets": {},
            "previous_offsets": {},
            "offset_deltas": {},
            "offset_changed_by_key": {},
        }

    def _action_keys(self):
        return [
            (int(neighbor_id), normalize_slice_type(slice_type))
            for neighbor_id in self.neighbor_ids
            for slice_type in self.slice_types
        ]

    def _action_values(self, action) -> np.ndarray:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        expected = int(np.prod(self.action_space.shape))
        if action_arr.size == 0:
            action_arr = np.zeros(expected, dtype=np.float32)
        if action_arr.size == 1 and expected > 1:
            action_arr = np.repeat(action_arr, expected)
        if action_arr.size != expected:
            raise ValueError(f"Expected action size {expected}, got {action_arr.size}")
        return np.clip(action_arr.astype(np.float32), -6.0, 6.0)

    def _format_action_array(self, values) -> np.ndarray:
        return np.asarray(values, dtype=np.float32).reshape(self.action_space.shape)

    def _stringify_action_debug(self, keys, values) -> Dict[str, float]:
        return {
            f"{int(key[0])}:{normalize_slice_type(key[1])}": float(value)
            for key, value in zip(keys, values)
        }

    def _stringify_action_debug_bool(self, keys, values) -> Dict[str, bool]:
        return {
            f"{int(key[0])}:{normalize_slice_type(key[1])}": bool(value)
            for key, value in zip(keys, values)
        }

    def _held_or_new_action(self, action):
        raw_actions = self._action_values(action)
        keys = self._action_keys()
        applied_offsets = dict(self.local_env.get_applied_offsets())
        previous_offsets = np.asarray(
            [float(applied_offsets.get(key, 0.0)) for key in keys],
            dtype=np.float32,
        )

        if self._held_action is None or self._action_hold_counter <= 0:
            target_offsets = np.asarray(
                [quantize_a3_offset(raw) for raw in raw_actions],
                dtype=np.float32,
            )
            limited_deltas = np.clip(
                target_offsets - previous_offsets,
                -self.max_offset_change_db,
                self.max_offset_change_db,
            )
            applied_action = np.asarray(
                [quantize_a3_offset(value) for value in previous_offsets + limited_deltas],
                dtype=np.float32,
            )
            self._held_proto_action = raw_actions.copy()
            self._held_action = applied_action.copy()
            self._action_hold_counter = max(self.action_hold_steps - 1, 0)
            offset_changed_by_key = np.ones(len(keys), dtype=bool)
            offset_changed = True
        else:
            target_offsets = np.asarray(self._held_action, dtype=np.float32)
            applied_action = np.asarray(self._held_action, dtype=np.float32)
            self._action_hold_counter -= 1
            offset_changed_by_key = np.zeros(len(keys), dtype=bool)
            offset_changed = False

        offset_deltas = applied_action - previous_offsets
        first = 0
        self._last_action_debug = {
            "raw_action": float(raw_actions[first]) if len(raw_actions) else 0.0,
            "target_offset": float(target_offsets[first]) if len(target_offsets) else 0.0,
            "applied_offset": float(applied_action[first]) if len(applied_action) else 0.0,
            "previous_offset": float(previous_offsets[first]) if len(previous_offsets) else 0.0,
            "offset_delta": float(offset_deltas[first]) if len(offset_deltas) else 0.0,
            "offset_changed": bool(offset_changed),
            "action_hold_counter": int(self._action_hold_counter),
            "action_hold_steps": int(self.action_hold_steps),
            "raw_actions": self._stringify_action_debug(keys, raw_actions),
            "target_offsets": self._stringify_action_debug(keys, target_offsets),
            "applied_offsets": self._stringify_action_debug(keys, applied_action),
            "previous_offsets": self._stringify_action_debug(keys, previous_offsets),
            "offset_deltas": self._stringify_action_debug(keys, offset_deltas),
            "offset_changed_by_key": self._stringify_action_debug_bool(keys, offset_changed_by_key),
        }
        return self._format_action_array(applied_action)

    def _held_or_new_rule_bias(self, force: bool = False):
        if force or self._held_rule_bias is None or self._bias_hold_counter <= 0:
            self._held_rule_bias = dict(self._rule_bias())
            self._bias_hold_counter = max(self.bias_hold_steps - 1, 0)
            self._last_bias_changed = True
        else:
            self._bias_hold_counter -= 1
            self._last_bias_changed = False
        return dict(self._held_rule_bias)

    def _augment_info(
        self,
        info: Dict,
        post_action_slice_loads: Optional[Dict[Tuple[int, str], float]] = None,
        post_action_spawn_counts: Optional[Dict[Tuple[int, str], int]] = None,
    ) -> Dict:
        info = dict(info or {})
        info["rule_bias"] = dict(self.local_env.global_bias)
        pairwise = self.local_env._pairwise_debug()
        info["serving_bias"] = pairwise["serving_bias"]
        info["target_bias"] = pairwise["target_bias"]
        info["reverse_bias"] = pairwise["reverse_bias"]
        info["desired_offsets"] = pairwise["desired_offsets"]
        info["action_temporal"] = dict(self._last_action_debug)
        info["bias_temporal"] = {
            "bias_hold_counter": int(self._bias_hold_counter),
            "bias_hold_steps": int(self.bias_hold_steps),
            "bias_changed": bool(self._last_bias_changed),
            "held_rule_bias": dict(self._held_rule_bias or {}),
        }
        scenario_slice_loads = self.base_env.get_slice_loads()
        info["slice_loads"] = scenario_slice_loads
        info["scenario_slice_loads"] = dict(scenario_slice_loads)
        info["post_action_slice_loads"] = dict(
            post_action_slice_loads if post_action_slice_loads is not None else scenario_slice_loads
        )
        info["scenario_demand_profile"] = dict(self._last_demand_profile)
        scenario_spawn_counts = self._slice_counts()
        info["spawn_counts"] = scenario_spawn_counts
        info["scenario_spawn_counts"] = dict(scenario_spawn_counts)
        info["post_action_spawn_counts"] = dict(
            post_action_spawn_counts if post_action_spawn_counts is not None else scenario_spawn_counts
        )
        info["bias_case"] = self._bias_case
        info["scenario_name"] = (
            str(self._current_scenario.name)
            if self._current_scenario is not None
            else "unbalanced"
        )
        info["slice_bias_cases"] = {
            slice_type: self._slice_case_name(slice_type)
            for slice_type in self.slice_types
        }
        info["bias_case_matched"] = bool(self._bias_case_matched)
        info["case_sampling_attempts"] = int(self._case_sampling_attempts)
        if self.balance_bias_cases and not self._bias_case_matched:
            info["bias_case_warning"] = (
                f"Accepted closest scenario after {self._case_sampling_attempts} attempts"
            )
        return info

    def _print_scenario_info(self, info: Dict):
        slice_cases = dict(info.get("slice_bias_cases", {}))
        loads = dict(info.get("slice_loads", {}))
        biases = dict(info.get("rule_bias", {}))

        case_text = ", ".join(
            f"{slice_type}={case_name}"
            for slice_type, case_name in slice_cases.items()
        )
        load_text = ", ".join(
            f"g{gnb_id}:{slice_type}={float(value):.2f}"
            for (gnb_id, slice_type), value in sorted(loads.items())
            if slice_type in self.slice_types
        )
        bias_text = ", ".join(
            f"g{src}→g{tgt}:{st}={float(value):+.0f}"
            for (src, tgt, st), value in sorted(biases.items())
            if st in self.slice_types
        )
        print(
            "[LocalA3 scenario] "
            f"reset={self._reset_count} "
            f"name={info.get('scenario_name', 'unknown')} "
            f"cases=[{case_text}] loads=[{load_text}] bias=[{bias_text}]",
            flush=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Upper-scenario lower training env
# ─────────────────────────────────────────────────────────────────────────────

_RULE_BIAS_PARAMS = {
    "eMBB":  {"scale": 0.35, "deadband": 0.10, "min_src": 0.40},
    "URLLC": {"scale": 0.25, "deadband": 0.07, "min_src": 0.35},
    "mMTC":  {"scale": 0.40, "deadband": 0.12, "min_src": 0.45},
}
_RULE_BIAS_DEFAULT_PARAMS = {"scale": 0.35, "deadband": 0.10, "min_src": 0.40}


def _reset_local_env_state(local_env: LocalA3OffsetEnv, base_env) -> None:
    """Reset a LocalA3OffsetEnv's internal state without touching the simulator."""
    for key in local_env._offsets:
        local_env._offsets[key] = 0.0
        local_env._prev_proto_offsets[key] = 0.0
        local_env._mobility_counters[key] = {
            "attempts": 0, "successes": 0, "failures": 0, "ping_pongs": 0,
        }
    local_env._ttt_counters.clear()
    local_env._last_serving = {
        int(ue.id): ue.serving_gnb for ue in base_env.get_all_ues()
    }
    local_env._prev_serving = {ue_id: None for ue_id in local_env._last_serving}
    local_env._last_reward_breakdown = {}


class UpperScenarioLowerEnv(gym.Env):
    """Lower TD3 training env backed by GlobalPPO3GNBEnv upper scenarios.

    One gNB (``controlled_gnb_id``) has its A3 offsets learned by TD3; all
    other gNBs run ``strong_directional_heuristic_local_executor`` every step.
    UE placement is driven by ``UpperTrainingScenario`` objects such as
    ``jain_balance_controllable``, which place UEs at equidistant midpoints
    between the center gNB and the outer gNBs — giving maximum A3 offset
    controllability.

    Defaults: ``controlled_gnb_id=1`` because all controllable scenarios start
    UEs on gNB-1 (the center gNB in medium_270m topology).
    """

    metadata = {"render_modes": []}

    # Shared rule-bias parameters (same formula as LocalA3RuleBiasTrainingEnv)
    _RULE_BIAS_PARAMS = _RULE_BIAS_PARAMS
    _RULE_BIAS_DEFAULT_PARAMS = _RULE_BIAS_DEFAULT_PARAMS

    def __init__(
        self,
        seed: int = 7,
        controlled_gnb_id: int = 1,
        training_scenarios: str = (
            "jain_balance_controllable,"
            "jain_control_urllc,"
            "jain_control_mmtc,"
            "jain_control_mixed"
        ),
        scenario_selection: str = "cycle",
        episode_steps: int = 40,
        slice_types: Sequence[str] = ("eMBB", "URLLC", "mMTC"),
        upper_window_seconds: float = 1.0,
        local_steps_per_global: int = 10,
        radio_substeps: int = 20,
        warmup_steps: int = 1,
        max_handovers_per_local_step: int = 3,
        action_hold_steps: int = 5,
        bias_hold_steps: int = 20,
        max_offset_change_db: float = 2.0,
        print_scenarios: bool = True,
    ):
        super().__init__()
        self.seed_value = int(seed)
        self.controlled_gnb_id = int(controlled_gnb_id)
        self.episode_steps = max(1, int(episode_steps))
        self.slice_types = tuple(normalize_slice_type(s) for s in slice_types)
        self.action_hold_steps = max(1, int(action_hold_steps))
        self.bias_hold_steps = max(1, int(bias_hold_steps))
        self.max_offset_change_db = float(max_offset_change_db)
        self.print_scenarios = bool(print_scenarios)

        GlobalPPO3GNBEnv = _import_global_ppo_env()
        self.upper_env = GlobalPPO3GNBEnv(
            seed=seed,
            slice_types=self.slice_types,
            scenario_mode="curriculum",
            training_scenarios=training_scenarios,
            scenario_selection=scenario_selection,
            upper_window_seconds=upper_window_seconds,
            local_steps_per_global=local_steps_per_global,
            radio_substeps=radio_substeps,
            terminal_reward_only=False,
            warmup_steps=warmup_steps,
            max_handovers_per_local_step=max_handovers_per_local_step,
        )

        self.base_env = self.upper_env.base_env
        self.gnb_ids = tuple(range(self.upper_env.n_gnbs))
        self.neighbors: Dict[int, Tuple[int, ...]] = {
            int(gid): tuple(int(n) for n in self.upper_env.neighbors[int(gid)])
            for gid in self.gnb_ids
        }
        self.heuristic_gnb_ids = tuple(
            g for g in self.gnb_ids if g != self.controlled_gnb_id
        )

        self.local_envs: Dict[int, LocalA3OffsetEnv] = {
            int(gid): LocalA3OffsetEnv(
                self.base_env,
                gnb_id=int(gid),
                neighbor_ids=self.neighbors[int(gid)],
                slice_types=self.slice_types,
                steps_per_action=1,
                ttt=1,
            )
            for gid in self.gnb_ids
        }
        self._ctrl_env = self.local_envs[self.controlled_gnb_id]
        self.action_space = self._ctrl_env.action_space
        self.observation_space = self._ctrl_env.observation_space

        self._heuristic_prev_offsets: Dict[int, np.ndarray] = {
            gid: np.zeros(
                len(self.neighbors[gid]) * len(self.slice_types), dtype=float
            )
            for gid in self.heuristic_gnb_ids
        }

        self._elapsed_steps = 0
        self._reset_count = 0
        self._action_hold_counter = 0
        self._held_action: Optional[np.ndarray] = None
        self._held_rule_bias: Optional[Dict] = None
        self._bias_hold_counter = 0
        self._current_scenario_name = "unknown"

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        self._reset_count += 1
        obs_upper, info = self.upper_env.reset(seed=seed, options=options)
        del obs_upper

        # Re-grab base_env in case it was recreated.
        self.base_env = self.upper_env.base_env
        for local_env in self.local_envs.values():
            local_env.base_env = self.base_env
            _reset_local_env_state(local_env, self.base_env)

        # Reset heuristic prev-offset memory.
        for gid in self.heuristic_gnb_ids:
            n = len(self.neighbors[gid]) * len(self.slice_types)
            self._heuristic_prev_offsets[gid] = np.zeros(n, dtype=float)

        self._elapsed_steps = 0
        self._action_hold_counter = 0
        self._held_action = None
        self._held_rule_bias = None
        self._bias_hold_counter = 0

        # Seed the initial rule bias.
        bias = self._compute_rule_bias(force=True)
        for local_env in self.local_envs.values():
            local_env.set_global_bias(bias)

        self._current_scenario_name = str(
            info.get("scenario_name", getattr(self.upper_env, "_active_scenario", "unknown"))
        )
        if self.print_scenarios:
            print(
                f"[UpperScenarioLowerEnv] reset={self._reset_count} "
                f"controlled_gnb={self.controlled_gnb_id} "
                f"scenario={self._current_scenario_name}",
                flush=True,
            )

        return self._ctrl_env._build_observation(), {}

    def step(self, action: np.ndarray):
        # Heuristic gNBs run first (A3 offsets applied + handovers executed)
        # before radio time advances — same ordering as LocalA3RuleBiasTrainingEnv.
        self._run_heuristic_gnbs()

        # Rate-limit and quantize the TD3 action.
        held = self._apply_action_hold(action)

        # Reset the measurement window so get_window_average_slice_loads()
        # returns a fresh per-step estimate (not cumulative since episode start).
        self.base_env.begin_radio_measurement_window()

        # Step the controlled gNB: applies A3 offsets, executes handovers,
        # advances the radio simulator one step, computes reward.
        obs, reward, terminated, truncated, info = self._ctrl_env.step(held)

        # Update rule bias with hold logic.
        bias = self._compute_rule_bias()
        for local_env in self.local_envs.values():
            local_env.set_global_bias(bias)

        self._elapsed_steps += 1
        truncated = bool(truncated or self._elapsed_steps >= self.episode_steps)

        info["rule_bias"] = bias
        info["scenario_name"] = self._current_scenario_name
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.upper_env.close()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _compute_rule_bias(self, force: bool = False) -> Dict:
        """Load-diff proportional bias; holds for bias_hold_steps steps."""
        if force or self._held_rule_bias is None or self._bias_hold_counter <= 0:
            bias: Dict = {}
            for src_id in self.gnb_ids:
                for tgt_id in self.gnb_ids:
                    if src_id == tgt_id:
                        continue
                    for slice_type in self.slice_types:
                        p = self._RULE_BIAS_PARAMS.get(
                            slice_type, self._RULE_BIAS_DEFAULT_PARAMS
                        )
                        src_load = self.base_env.estimate_slice_load(src_id, slice_type)
                        tgt_load = self.base_env.estimate_slice_load(tgt_id, slice_type)
                        diff = src_load - tgt_load
                        if abs(diff) < p["deadband"]:
                            value = 0.0
                        else:
                            raw = -diff / p["scale"]
                            if raw < 0.0 and src_load < p["min_src"]:
                                raw = 0.0
                            value = float(np.clip(raw, -1.0, 1.0))
                        bias[(src_id, tgt_id, slice_type)] = value
            self._held_rule_bias = bias
            self._bias_hold_counter = max(self.bias_hold_steps - 1, 0)
        else:
            self._bias_hold_counter -= 1
        return dict(self._held_rule_bias)

    def _apply_action_hold(self, action) -> np.ndarray:
        """Hold the current action for action_hold_steps steps; rate-limit deltas."""
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._held_action is None or self._action_hold_counter <= 0:
            applied = self._ctrl_env.get_applied_offsets()
            keys = [
                (nb, st)
                for nb in self._ctrl_env.neighbor_ids
                for st in self.slice_types
            ]
            prev = np.array(
                [float(applied.get(k, 0.0)) for k in keys], dtype=np.float32
            )
            limited = np.clip(
                raw - prev, -self.max_offset_change_db, self.max_offset_change_db
            )
            quantized = np.array(
                [quantize_a3_offset(float(v)) for v in prev + limited],
                dtype=np.float32,
            )
            self._held_action = quantized
            self._action_hold_counter = max(self.action_hold_steps - 1, 0)
        else:
            self._action_hold_counter -= 1
        return self._held_action.reshape(self._ctrl_env.action_space.shape)

    def _run_heuristic_gnbs(self) -> None:
        """Run strong_directional_heuristic_local_executor for all non-controlled gNBs."""
        if not self.heuristic_gnb_ids:
            return

        all_ues = [
            ue for ue in self.base_env.get_all_ues()
            if ue.connected and ue.serving_gnb is not None
        ]
        if not all_ues:
            return

        all_gnb_ids = list(self.gnb_ids)
        n_gnbs = len(all_gnb_ids)
        gnb_index = {gid: i for i, gid in enumerate(all_gnb_ids)}
        slice_index = {st: i for i, st in enumerate(self.slice_types)}

        ue_slice_arr = np.array([
            slice_index.get(normalize_slice_type(getattr(ue, "slice_type", "eMBB")), 0)
            for ue in all_ues
        ], dtype=int)
        ue_serving_arr = np.array([
            gnb_index.get(int(ue.serving_gnb), 0) for ue in all_ues
        ], dtype=int)

        rsrp = np.full((len(all_ues), n_gnbs), -120.0, dtype=float)
        for u_idx, ue in enumerate(all_ues):
            for g_idx, gid in enumerate(all_gnb_ids):
                gnb = self.base_env._get_gnb_by_id(gid)
                if gnb is not None:
                    try:
                        rsrp[u_idx, g_idx] = self.base_env._compute_link_metrics(
                            gnb, ue
                        )["rx_power_dbm"]
                    except Exception:
                        pass

        load = np.array([
            [
                self.base_env.get_window_average_slice_loads().get((gid, st), 0.0)
                for st in self.slice_types
            ]
            for gid in all_gnb_ids
        ], dtype=float)
        sla_flags = (
            self.base_env.get_slice_sla_flags()
            if hasattr(self.base_env, "get_slice_sla_flags")
            else {}
        )
        sla_arr = np.array([
            [float(sla_flags.get((gid, st), 0.0)) for st in self.slice_types]
            for gid in all_gnb_ids
        ], dtype=float)
        ho_fail = np.zeros((n_gnbs, len(self.slice_types)), dtype=float)
        pp_arr = np.zeros((n_gnbs, len(self.slice_types)), dtype=float)
        current_bias = dict(self._ctrl_env.global_bias)

        for gid in self.heuristic_gnb_ids:
            h_env = self.local_envs[gid]
            g_idx = gnb_index[gid]
            h_neighbors = h_env.neighbor_ids
            max_nb = len(h_neighbors)
            n_slices = len(self.slice_types)

            B = np.zeros((n_gnbs, max_nb, n_slices), dtype=float)
            prev = self._heuristic_prev_offsets[gid].reshape(max_nb, n_slices)
            prev_full = np.zeros((n_gnbs, max_nb, n_slices), dtype=float)
            prev_full[g_idx] = prev

            for nb_slot, nb_id in enumerate(h_neighbors):
                for s_idx, st in enumerate(self.slice_types):
                    B[g_idx, nb_slot, s_idx] = float(
                        current_bias.get((gid, nb_id, st), 0.0)
                    )

            neighbor_graph = {g_idx: [gnb_index[nb] for nb in h_neighbors]}

            offsets_full, _ = strong_directional_heuristic_local_executor(
                B=B,
                prev_offsets=prev_full,
                ue_slice=ue_slice_arr,
                ue_serving_gnb=ue_serving_arr,
                rsrp_matrix=rsrp,
                neighbor_graph=neighbor_graph,
                load=load,
                sla_violation=sla_arr,
                ho_failure_ratio=ho_fail,
                pingpong_ratio=pp_arr,
                slice_types=self.slice_types,
                return_debug=True,
            )

            offsets_vec = offsets_full[g_idx].reshape(-1)
            self._heuristic_prev_offsets[gid] = offsets_vec.copy()
            h_env._apply_proto_offsets(offsets_vec)
            h_env._execute_a3_handovers()

            for nb_slot, nb_id in enumerate(h_neighbors):
                for s_idx, st in enumerate(self.slice_types):
                    self.base_env.set_a3_offset(
                        gid, nb_id, st, float(offsets_full[g_idx, nb_slot, s_idx])
                    )
