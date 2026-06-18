#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import gymnasium as gym
import numpy as np

from local_a3_agent_wrapper import normalize_slice_type
from scenario_creator import create_multignb_env
from slice_ran import Packet
from strong_heuristic_local_executor import strong_heuristic_local_executor


SLICE_TYPES = ("eMBB", "URLLC", "mMTC")
SNAPSHOT_DEMAND_SAFETY = 0.90
DEFAULT_GNB_CONFIGS_3 = (
    {"id": 0, "x": 0.0, "y": 0.0, "coverage_radius": 520.0, "carrier_id": 0, "n_prbs": 100},
    {"id": 1, "x": 450.0, "y": 0.0, "coverage_radius": 520.0, "carrier_id": 0, "n_prbs": 100},
    {"id": 2, "x": 225.0, "y": 390.0, "coverage_radius": 520.0, "carrier_id": 0, "n_prbs": 100},
)

GLOBAL_SNAPSHOT_SCENARIOS = {
    "embb_g0_offload": np.asarray(
        [
            [0.88, 0.50, 0.50],
            [0.18, 0.50, 0.50],
            [0.48, 0.50, 0.50],
        ],
        dtype=float,
    ),
    "urllc_g1_offload": np.asarray(
        [
            [0.50, 0.50, 0.50],
            [0.50, 0.88, 0.50],
            [0.50, 0.18, 0.50],
        ],
        dtype=float,
    ),
    "mmtc_g2_offload": np.asarray(
        [
            [0.50, 0.50, 0.18],
            [0.50, 0.50, 0.48],
            [0.50, 0.50, 0.88],
        ],
        dtype=float,
    ),
    "embb_g0_urllc_g1_conflict": np.asarray(
        [
            [0.88, 0.18, 0.50],
            [0.18, 0.88, 0.50],
            [0.50, 0.50, 0.50],
        ],
        dtype=float,
    ),
    "mmtc_g2_embb_g1_conflict": np.asarray(
        [
            [0.50, 0.50, 0.18],
            [0.88, 0.50, 0.50],
            [0.18, 0.50, 0.88],
        ],
        dtype=float,
    ),
    "all_offload_balancing": np.asarray(
        [
            [0.88, 0.18, 0.50],
            [0.50, 0.88, 0.18],
            [0.18, 0.50, 0.88],
        ],
        dtype=float,
    ),
    "multi_slice_multi_gnb_congestion": np.asarray(
        [
            [0.92, 0.86, 0.35],
            [0.32, 0.90, 0.88],
            [0.76, 0.30, 0.28],
        ],
        dtype=float,
    ),
    "all_neutral": np.asarray(
        [
            [0.55, 0.50, 0.50],
            [0.50, 0.55, 0.50],
            [0.50, 0.50, 0.55],
        ],
        dtype=float,
    ),
}


class GlobalPPO3GNBEnv(gym.Env):
    """Upper/global PPO environment for 3-gNB HRL control.

    One upper action is a flattened 3x3 bias matrix B. The matrix is held for
    one upper window. During that window, three heuristic lower agents compute
    neighbor/slice A3 offsets and this wrapper applies the resulting handovers
    simultaneously before advancing the base simulator.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int = 7,
        n_gnbs: int = 3,
        slice_types: Sequence[str] = SLICE_TYPES,
        include_ue_counts: bool = True,
        include_service_metrics: bool = False,
        use_sumo_mobility: bool = False,
        local_steps_per_global: int = 10,
        global_steps_per_episode: int = 12,
        radio_substeps: int = 10,
        gnb_configs: Sequence[Mapping] | None = None,
        scenario_mode: str = "snapshot",
        snapshot_scenario: str = "mixed",
        terminal_reward_only: bool = True,
        use_progress_reward: bool = False,
        max_handovers_per_local_step: int = 1,
        action_direction_reward_weight: float = 2.0,
        snapshot_block_episodes: int = 10,
        light_load_ues: int = 1,
        medium_load_ues: int = 2,
        high_load_ues: int = 3,
        print_scenarios: bool = False,
        slice_prb_budgets: Mapping[str, int] | None = None,
        max_prbs_per_ue: int | None = 20,
        directional_global_action: bool = True,
        global_reward_mu: float = 1.0,
        global_reward_zeta: float = 1.0,
        global_reward_beta: float = 5.0,
        global_action_lambda: float = 0.01,
        global_action_kappa: float = 0.05,
        global_bad_direction_eta: float = 0.5,
        global_unsafe_target_rho: float = 1.0,
        sla_deadband: float = 0.05,
        critical_load_thresholds: Mapping[str, float] | None = None,
    ):
        super().__init__()
        if int(n_gnbs) != 3:
            raise ValueError("GlobalPPO3GNBEnv currently supports exactly n_gnbs=3")

        self.seed_value = int(seed)
        self.rng = np.random.default_rng(self.seed_value)
        self.n_gnbs = 3
        self.slice_types = tuple(normalize_slice_type(s) for s in slice_types)
        self.include_ue_counts = bool(include_ue_counts)
        self.include_service_metrics = bool(include_service_metrics)
        self.use_sumo_mobility = bool(use_sumo_mobility)
        self.local_steps_per_global = max(1, int(local_steps_per_global))
        self.global_steps_per_episode = max(1, int(global_steps_per_episode))
        self.radio_substeps = int(radio_substeps)
        self.gnb_configs = [dict(cfg) for cfg in (gnb_configs or DEFAULT_GNB_CONFIGS_3)]
        self.scenario_mode = str(scenario_mode).strip().lower()
        self.snapshot_scenario = str(snapshot_scenario).strip()
        self.terminal_reward_only = bool(terminal_reward_only)
        self.use_progress_reward = bool(use_progress_reward)
        self.max_handovers_per_local_step = max(1, int(max_handovers_per_local_step))
        self.action_direction_reward_weight = float(action_direction_reward_weight)
        self.snapshot_block_episodes = max(1, int(snapshot_block_episodes))
        self.light_load_ues = max(1, int(light_load_ues))
        self.medium_load_ues = max(self.light_load_ues, int(medium_load_ues))
        self.high_load_ues = max(self.medium_load_ues, int(high_load_ues))
        self.print_scenarios = bool(print_scenarios)
        self.slice_prb_budgets = None if slice_prb_budgets is None else dict(slice_prb_budgets)
        self.max_prbs_per_ue = None if max_prbs_per_ue is None else max(int(max_prbs_per_ue), 1)
        self.directional_global_action = bool(directional_global_action)
        self.global_reward_mu = float(global_reward_mu)
        self.global_reward_zeta = float(global_reward_zeta)
        self.global_reward_beta = float(global_reward_beta)
        self.global_action_lambda = float(global_action_lambda)
        self.global_action_kappa = float(global_action_kappa)
        self.global_bad_direction_eta = float(global_bad_direction_eta)
        self.global_unsafe_target_rho = float(global_unsafe_target_rho)
        self.sla_deadband = max(float(sla_deadband), 0.0)
        default_critical = {"eMBB": 0.95, "URLLC": 0.95, "mMTC": 0.95}
        if critical_load_thresholds is not None:
            default_critical.update({normalize_slice_type(k): float(v) for k, v in critical_load_thresholds.items()})
        self.critical_load_thresholds = default_critical

        self.neighbors = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        self.max_neighbors = max(len(values) for values in self.neighbors.values())
        self.lower_agents = {}

        action_dim = (
            self.n_gnbs * self.max_neighbors * len(self.slice_types)
            if self.directional_global_action
            else self.n_gnbs * len(self.slice_types)
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )
        signals_per_key = 2
        if self.include_service_metrics:
            signals_per_key += 3
        if self.include_ue_counts:
            signals_per_key += 1
        obs_dim = self.n_gnbs * len(self.slice_types) * signals_per_key
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.base_env = self._make_base_env()
        self._global_step = 0
        self._last_directional_bias_tensor = np.zeros(
            (self.n_gnbs, self.max_neighbors, len(self.slice_types)),
            dtype=np.float32,
        )
        self._previous_directional_bias_tensor = self._last_directional_bias_tensor.copy()
        self._last_bias_matrix = np.zeros((self.n_gnbs, len(self.slice_types)), dtype=np.float32)
        self._strong_prev_offsets = np.zeros((self.n_gnbs, len(self.slice_types)), dtype=float)
        self._last_strong_offsets = self._strong_prev_offsets.copy()
        self._last_strong_offset_debug = {}
        self._last_info: Dict = {}
        self._active_scenario = ""
        self._active_target_load_matrix = np.zeros((self.n_gnbs, len(self.slice_types)), dtype=float)
        self._episode_instant_rewards = []
        self._episode_window_rewards = []
        self._episode_handovers = 0
        self._episode_start_imbalance = 0.0
        self._episode_index = 0

    def _make_base_env(self):
        return create_multignb_env(
            rng=self.rng,
            n=4,
            gnb_configs=self.gnb_configs,
            slots_per_step=5,
            L1_level=False,
            step_dt=1e-3,
            mobility_dt=0.0,
            radio_substeps=self.radio_substeps,
            max_episode_steps=self.global_steps_per_episode * self.local_steps_per_global + 5,
            use_sumo_mobility=self.use_sumo_mobility,
            slice_prb_budgets=self.slice_prb_budgets,
            max_prbs_per_ue=self.max_prbs_per_ue,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.base_env._rng = np.random.default_rng(seed)
        self._global_step = 0
        self._last_bias_matrix = np.zeros((self.n_gnbs, len(self.slice_types)), dtype=np.float32)
        self._last_directional_bias_tensor = np.zeros(
            (self.n_gnbs, self.max_neighbors, len(self.slice_types)),
            dtype=np.float32,
        )
        self._previous_directional_bias_tensor = self._last_directional_bias_tensor.copy()
        self._strong_prev_offsets = np.zeros((self.n_gnbs, len(self.slice_types)), dtype=float)
        self._last_strong_offsets = self._strong_prev_offsets.copy()
        self._last_strong_offset_debug = {}
        self._episode_instant_rewards = []
        self._episode_window_rewards = []
        self._episode_handovers = 0
        for agent in self.lower_agents.values():
            agent.reset()

        self.base_env.clear_ues(reset_ids=True)
        self._initialize_load_scenario()
        self.base_env._invalidate_metric_caches()
        self._episode_start_imbalance = self._target_load_error()
        obs = self._get_observation()
        info = self._build_info(
            reward=0.0,
            instant_rewards=[],
            handovers=0,
            start_imbalance=self._target_load_error(),
            end_imbalance=self._target_load_error(),
        )
        self._last_info = info
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        directional_bias = self._action_to_directional_tensor(action)
        previous_directional_bias = self._last_directional_bias_tensor.copy()
        bias_matrix = self._directional_to_summary_bias_matrix(directional_bias)

        self._previous_directional_bias_tensor = previous_directional_bias.astype(np.float32)
        self._last_directional_bias_tensor = directional_bias.astype(np.float32)
        self._last_bias_matrix = bias_matrix.astype(np.float32)
        start_loads = self._load_matrix()
        start_sla = self._sla_matrix()
        start_cost = self._global_network_cost(start_loads, start_sla)
        start_imbalance = self._load_variance(start_loads)
        instant_rewards = []
        start_handover_idx = len(getattr(self.base_env, "handover_events", []))

        offsets, offset_debug = self._compute_strong_local_offsets(bias_matrix)
        self._apply_slice_offsets(offsets)
        self._last_strong_offsets = offsets.copy()
        self._last_strong_offset_debug = offset_debug
        self._strong_prev_offsets = offsets.copy()

        for _ in range(self.local_steps_per_global):
            _obs, _reward, terminated, truncated, _info = self.base_env.step(0)
            if terminated or truncated:
                break

        end_loads = self._load_matrix()
        end_sla = self._sla_matrix()
        end_cost = self._global_network_cost(end_loads, end_sla)
        end_imbalance = self._load_variance(end_loads)
        action_penalty = self._global_action_penalty(directional_bias, previous_directional_bias)
        bad_direction_penalty = self._bad_direction_penalty(directional_bias, start_loads, start_sla)
        window_reward = float(start_cost - end_cost - action_penalty - bad_direction_penalty)
        instant_rewards = [window_reward]
        progress_reward = 2.0 * (start_imbalance - end_imbalance) if self.use_progress_reward else 0.0
        dense_reward = window_reward + progress_reward
        self._episode_instant_rewards.extend(float(value) for value in instant_rewards)
        self._episode_window_rewards.append(float(dense_reward))

        self._global_step += 1
        terminated = False
        truncated = self._global_step >= self.global_steps_per_episode
        total_handovers = len(getattr(self.base_env, "handover_events", [])) - start_handover_idx
        self._episode_handovers += int(total_handovers)
        if self.terminal_reward_only:
            reward = self._episode_terminal_reward() if truncated else 0.0
        else:
            reward = dense_reward
        obs = self._get_observation()
        info = self._build_info(
            reward=reward,
            instant_rewards=instant_rewards,
            handovers=total_handovers,
            start_imbalance=start_imbalance,
            end_imbalance=end_imbalance,
        )
        info["dense_window_reward"] = float(dense_reward)
        info["global_cost_start"] = float(start_cost)
        info["global_cost_end"] = float(end_cost)
        info["global_cost_improvement"] = float(start_cost - end_cost)
        info["global_action_penalty"] = float(action_penalty)
        info["global_bad_direction_penalty"] = float(bad_direction_penalty)
        info["terminal_reward_only"] = bool(self.terminal_reward_only)
        info["use_progress_reward"] = bool(self.use_progress_reward)
        info["episode_terminal_reward"] = (
            float(reward) if truncated and self.terminal_reward_only else 0.0
        )
        self._last_info = info
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self.base_env.close()

    def _action_to_directional_tensor(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        directional_size = self.n_gnbs * self.max_neighbors * len(self.slice_types)
        legacy_size = self.n_gnbs * len(self.slice_types)

        if action.size == directional_size:
            return np.clip(action, -1.0, 1.0).reshape(
                self.n_gnbs,
                self.max_neighbors,
                len(self.slice_types),
            )

        if action.size == legacy_size:
            legacy = np.clip(action, -1.0, 1.0).reshape(self.n_gnbs, len(self.slice_types))
            return np.repeat(legacy[:, None, :], self.max_neighbors, axis=1)

        raise ValueError(
            f"Expected action shape {self.action_space.shape} "
            f"or legacy flat size {legacy_size}, got {action.shape}"
        )

    def _directional_to_summary_bias_matrix(self, directional_bias: np.ndarray) -> np.ndarray:
        directional_bias = np.asarray(directional_bias, dtype=float)
        summary = np.zeros((self.n_gnbs, len(self.slice_types)), dtype=float)
        for gnb_id in range(self.n_gnbs):
            n_neighbors = max(len(self.neighbors.get(gnb_id, [])), 1)
            summary[gnb_id, :] = np.mean(directional_bias[gnb_id, :n_neighbors, :], axis=0)
        return summary

    def _directional_bias_row(self, directional_bias: np.ndarray, gnb_id: int) -> Dict[Tuple[int, str], float]:
        row = {}
        for neighbor_slot, neighbor_id in enumerate(self.neighbors.get(int(gnb_id), [])):
            for s_idx, slice_type in enumerate(self.slice_types):
                row[(int(neighbor_id), slice_type)] = float(directional_bias[int(gnb_id), neighbor_slot, s_idx])
        return row

    def _get_observation(self) -> np.ndarray:
        try:
            obs = self.base_env.get_global_agent_observation(
                include_ue_counts=self.include_ue_counts,
                include_service_metrics=self.include_service_metrics,
            )
        except Exception:
            obs = self._fallback_observation()
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.include_ue_counts:
            n_keys = self.n_gnbs * len(self.slice_types)
            count_block_start = n_keys * (5 if self.include_service_metrics else 2)
            if obs.size >= count_block_start + n_keys:
                obs = obs.copy()
                for idx, slice_type in enumerate(self.slice_types):
                    denom = max(self._kmax_by_slice().get(slice_type, 1.0), 1e-9)
                    for gnb_idx in range(self.n_gnbs):
                        count_idx = count_block_start + gnb_idx * len(self.slice_types) + idx
                        obs[count_idx] = obs[count_idx] / denom
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def _fallback_observation(self) -> np.ndarray:
        loads = self.base_env.get_slice_loads()
        sla = self.base_env.get_slice_sla_flags()
        values = []
        keys = [(i, s) for i in range(self.n_gnbs) for s in self.slice_types]
        values.extend(float(loads.get(key, 0.0)) for key in keys)
        values.extend(float(sla.get(key, 0.0)) for key in keys)
        if self.include_service_metrics:
            kpis = self.base_env._last_info.get("slice_kpis", {})
            values.extend(float(min(kpis.get(key, {}).get("demand_load", 0.0), 1.0)) for key in keys)
            values.extend(float(min(kpis.get(key, {}).get("queue_pressure", 0.0), 1.0)) for key in keys)
            values.extend(
                float(np.clip(1.0 - kpis.get(key, {}).get("served_ratio", 1.0), 0.0, 1.0))
                for key in keys
            )
        if self.include_ue_counts:
            kmax = self._kmax_by_slice()
            values.extend(
                float(self.base_env.get_slice_ue_count(*key)) / max(float(kmax[key[1]]), 1e-9)
                for key in keys
            )
        return np.asarray(values, dtype=np.float32)

    def apply_all_offsets(self, all_offsets):
        """Deprecated compatibility shim; A3 lives in MultiGNBWrapper now."""
        for (serving_id, neighbor_id, slice_type), info in all_offsets.items():
            self.base_env.set_a3_offset(
                serving_id,
                neighbor_id,
                slice_type,
                float(info["applied_offset_db"]),
            )
        return self.base_env._evaluate_a3_handovers()

    def _apply_slice_offsets(self, offsets: np.ndarray) -> None:
        offsets = np.asarray(offsets, dtype=float)
        for serving_id in range(self.n_gnbs):
            for s_idx, slice_type in enumerate(self.slice_types):
                offset_db = float(offsets[serving_id, s_idx])
                for neighbor_id in self.neighbors.get(serving_id, []):
                    self.base_env.set_a3_offset(serving_id, neighbor_id, slice_type, offset_db)

    def _aggregate_mobility_ratio_matrix(self, ratios: Mapping[Tuple[int, int, str], float]) -> np.ndarray:
        matrix = np.zeros((self.n_gnbs, len(self.slice_types)), dtype=float)
        counts = np.zeros_like(matrix)
        slice_index = {normalize_slice_type(s): idx for idx, s in enumerate(self.slice_types)}
        for key, value in (ratios or {}).items():
            if len(key) != 3:
                continue
            serving_id, _neighbor_id, slice_type = key
            serving_id = int(serving_id)
            normalized_slice = normalize_slice_type(slice_type)
            if serving_id < 0 or serving_id >= self.n_gnbs or normalized_slice not in slice_index:
                continue
            s_idx = slice_index[normalized_slice]
            matrix[serving_id, s_idx] += float(max(value, 0.0))
            counts[serving_id, s_idx] += 1.0
        return matrix / np.maximum(counts, 1.0)

    def _strong_executor_arrays(self):
        slice_index = {normalize_slice_type(s): idx for idx, s in enumerate(self.slice_types)}
        connected_ues = [
            ue for ue in self.base_env.get_all_ues()
            if bool(getattr(ue, "connected", False)) and getattr(ue, "serving_gnb", None) is not None
        ]
        ue_slice = []
        ue_serving_gnb = []
        rsrp_rows = []

        for ue in connected_ues:
            normalized_slice = normalize_slice_type(getattr(ue, "slice_type", "eMBB"))
            if normalized_slice not in slice_index:
                continue
            serving_id = int(ue.serving_gnb)
            if serving_id < 0 or serving_id >= self.n_gnbs:
                continue
            ue_slice.append(slice_index[normalized_slice])
            ue_serving_gnb.append(serving_id)
            rsrp_rows.append([
                float(self.base_env._compute_link_metrics(self.base_env._get_gnb_by_id(gnb_id), ue)["rsrp_dbm"])
                for gnb_id in range(self.n_gnbs)
            ])

        if rsrp_rows:
            rsrp_matrix = np.asarray(rsrp_rows, dtype=float)
        else:
            rsrp_matrix = np.zeros((0, self.n_gnbs), dtype=float)

        return (
            np.asarray(ue_slice, dtype=int),
            np.asarray(ue_serving_gnb, dtype=int),
            rsrp_matrix,
            self._load_matrix(),
            np.clip(self._sla_matrix(), 0.0, 1.0),
            self._aggregate_mobility_ratio_matrix(self.base_env.get_handover_failure_ratios()),
            self._aggregate_mobility_ratio_matrix(self.base_env.get_ping_pong_ratios()),
        )

    def _compute_strong_local_offsets(self, bias_matrix: np.ndarray):
        (
            ue_slice,
            ue_serving_gnb,
            rsrp_matrix,
            load,
            sla_violation,
            ho_failure_ratio,
            pingpong_ratio,
        ) = self._strong_executor_arrays()

        return strong_heuristic_local_executor(
            B=np.asarray(bias_matrix, dtype=float),
            prev_offsets=self._strong_prev_offsets,
            ue_slice=ue_slice,
            ue_serving_gnb=ue_serving_gnb,
            rsrp_matrix=rsrp_matrix,
            neighbor_graph=self.neighbors,
            load=load,
            sla_violation=sla_violation,
            ho_failure_ratio=ho_failure_ratio,
            pingpong_ratio=pingpong_ratio,
            hysteresis_db=float(getattr(self.base_env, "a3_hysteresis_db", 1.0)),
            l_safe=0.85,
            slice_types=self.slice_types,
            return_debug=True,
        )

    def _rx_power(self, gnb, ue) -> float:
        return float(self.base_env._compute_link_metrics(gnb, ue)["rx_power_dbm"])

    def _instant_reward(self, bias_matrix: np.ndarray, handovers: int) -> float:
        loads = self._load_matrix()
        sla = self._sla_matrix()
        network_cost = self._global_network_cost(loads, sla)
        n_ue = max(len([ue for ue in self.base_env.get_all_ues() if ue.connected]), 1)
        handover_penalty = float(handovers) / float(n_ue)
        return float(
            -1.0 * network_cost
            -0.05 * handover_penalty
        )

    def _episode_terminal_reward(self) -> float:
        if self._episode_window_rewards:
            return float(np.mean(self._episode_window_rewards))
        return float(self._instant_reward(self._last_bias_matrix, self._episode_handovers))

    def _global_network_cost(self, loads: np.ndarray | None = None, sla: np.ndarray | None = None) -> float:
        if loads is None:
            loads = self._load_matrix()
        if sla is None:
            sla = self._sla_matrix()
        loads = np.asarray(loads, dtype=float)
        sla = np.asarray(sla, dtype=float)

        variance_cost = float(sum(np.var(loads[:, s_idx]) for s_idx in range(len(self.slice_types))))
        saturation_count = self._saturation_count(loads)
        sla_severity = self._sla_severity(sla)
        return float(
            self.global_reward_mu * variance_cost
            + self.global_reward_zeta * float(saturation_count)
            + self.global_reward_beta * sla_severity
        )

    def _sla_severity(self, sla: np.ndarray | None = None) -> float:
        if sla is None:
            sla = self._sla_matrix()
        sla = np.asarray(sla, dtype=float)
        return float(np.sum(np.maximum(sla - self.sla_deadband, 0.0)))

    def _saturation_count(self, loads: np.ndarray | None = None) -> int:
        if loads is None:
            loads = self._load_matrix()
        loads = np.asarray(loads, dtype=float)
        count = 0
        for s_idx, slice_type in enumerate(self.slice_types):
            threshold = float(self.critical_load_thresholds.get(slice_type, 0.95))
            count += int(np.sum(loads[:, s_idx] >= threshold))
        return int(count)

    def _global_action_penalty(
        self,
        directional_bias: np.ndarray,
        previous_directional_bias: np.ndarray,
    ) -> float:
        current = np.asarray(directional_bias, dtype=float)
        previous = np.asarray(previous_directional_bias, dtype=float)
        negative_penalty = float(np.sum(np.maximum(-current, 0.0) ** 2))
        smoothness_penalty = float(np.sum((current - previous) ** 2))
        return float(
            self.global_action_lambda * negative_penalty
            + self.global_action_kappa * smoothness_penalty
        )

    def _bad_direction_penalty(
        self,
        directional_bias: np.ndarray,
        loads: np.ndarray | None = None,
        sla: np.ndarray | None = None,
    ) -> float:
        if loads is None:
            loads = self._load_matrix()
        if sla is None:
            sla = self._sla_matrix()
        loads = np.asarray(loads, dtype=float)
        sla = np.asarray(sla, dtype=float)
        bias = np.asarray(directional_bias, dtype=float)

        load_penalty = 0.0
        unsafe_target_penalty = 0.0
        for src in range(self.n_gnbs):
            for neighbor_slot, dst in enumerate(self.neighbors.get(src, [])):
                for s_idx, _slice_type in enumerate(self.slice_types):
                    offload_strength = max(-float(bias[src, neighbor_slot, s_idx]), 0.0)
                    if offload_strength <= 0.0:
                        continue
                    load_penalty += offload_strength * max(float(loads[dst, s_idx] - loads[src, s_idx]), 0.0)
                    unsafe_target_penalty += offload_strength * max(float(sla[dst, s_idx]), 0.0)

        return float(
            self.global_bad_direction_eta * load_penalty
            + self.global_unsafe_target_rho * unsafe_target_penalty
        )

    def _action_direction_reward(self, bias_matrix: np.ndarray) -> float:
        if self.action_direction_reward_weight <= 0.0:
            return 0.0
        bias = np.asarray(bias_matrix, dtype=float)
        targets = np.asarray(self._active_target_load_matrix, dtype=float)
        if bias.shape != targets.shape:
            return 0.0

        overloaded = targets >= 0.80
        light = targets <= 0.25
        neutral = ~(overloaded | light)
        scores = []
        if overloaded.any():
            scores.append(float(np.mean(-bias[overloaded])))
        if light.any():
            scores.append(float(np.mean(bias[light])))
        if neutral.any():
            scores.append(float(np.mean(1.0 - np.abs(bias[neutral]))))
        if not scores:
            return 0.0
        normalized_score = float(np.clip(np.mean(scores), -1.0, 1.0))
        return float(self.action_direction_reward_weight * normalized_score)

    def _load_imbalance(self) -> float:
        return self._target_load_error()

    def _target_load_error(self, loads: np.ndarray | None = None) -> float:
        if loads is None:
            loads = self._load_matrix()
        targets = self._balance_target_matrix()
        if targets.shape != loads.shape:
            targets = np.full_like(loads, 0.65, dtype=float)
        return float(np.mean((loads - targets) ** 2))

    def _balance_target_matrix(self) -> np.ndarray:
        targets = np.asarray(self._active_target_load_matrix, dtype=float)
        if targets.shape != (self.n_gnbs, len(self.slice_types)):
            return np.full((self.n_gnbs, len(self.slice_types)), 0.65, dtype=float)
        per_slice_target = np.mean(targets, axis=0, keepdims=True)
        return np.repeat(per_slice_target, self.n_gnbs, axis=0)

    def _load_variance(self, loads: np.ndarray | None = None) -> float:
        if loads is None:
            loads = self._load_matrix()
        return float(sum(np.var(loads[:, s_idx]) for s_idx in range(len(self.slice_types))))

    def _load_matrix(self) -> np.ndarray:
        loads = self.base_env.get_slice_loads()
        return np.asarray(
            [
                [float(loads.get((gnb_id, slice_type), 0.0)) for slice_type in self.slice_types]
                for gnb_id in range(self.n_gnbs)
            ],
            dtype=float,
        )

    def _sla_matrix(self) -> np.ndarray:
        flags = self.base_env.get_slice_sla_flags()
        return np.asarray(
            [
                [float(flags.get((gnb_id, slice_type), 0.0)) for slice_type in self.slice_types]
                for gnb_id in range(self.n_gnbs)
            ],
            dtype=float,
        )

    def _ue_count_dict(self) -> Dict[Tuple[int, str], int]:
        return {
            (gnb_id, slice_type): int(self.base_env.get_slice_ue_count(gnb_id, slice_type))
            for gnb_id in range(self.n_gnbs)
            for slice_type in self.slice_types
        }

    def _slice_load_dict(self) -> Dict[Tuple[int, str], float]:
        loads = self._load_matrix()
        return {
            (gnb_id, slice_type): float(loads[gnb_id, s_idx])
            for gnb_id in range(self.n_gnbs)
            for s_idx, slice_type in enumerate(self.slice_types)
        }

    def _kmax_by_slice(self) -> Dict[str, float]:
        kmax = float(max(self.high_load_ues, 1))
        return {"eMBB": kmax, "URLLC": kmax, "mMTC": kmax}

    def _build_info(
        self,
        reward: float,
        instant_rewards: Sequence[float],
        handovers: int,
        start_imbalance: float,
        end_imbalance: float,
    ) -> Dict:
        return {
            "global_step": int(self._global_step),
            "scenario_name": self._active_scenario,
            "reward": float(reward),
            "instant_reward_mean": float(np.mean(instant_rewards)) if instant_rewards else 0.0,
            "load_variance": self._load_variance(),
            "target_load_error": float(end_imbalance),
            "load_imbalance_start": float(start_imbalance),
            "load_imbalance_end": float(end_imbalance),
            "target_load_error_start": float(start_imbalance),
            "target_load_error_end": float(end_imbalance),
            "overload_ratio": float(np.mean(self._load_matrix() > 0.85)),
            "sla_count": float(np.sum(self._sla_matrix() > 0.0)),
            "sla_severity": float(self._sla_severity(self._sla_matrix())),
            "sla_deadband": float(self.sla_deadband),
            "saturation_count": int(self._saturation_count(self._load_matrix())),
            "global_network_cost": float(self._global_network_cost()),
            "handover_count": int(handovers),
            "action_direction_reward": self._action_direction_reward(self._last_bias_matrix),
            "bias_matrix": self._last_bias_matrix.copy(),
            "directional_bias_tensor": self._last_directional_bias_tensor.copy(),
            "strong_local_offsets": self._last_strong_offsets.copy(),
            "strong_local_offset_debug": dict(self._last_strong_offset_debug),
            "target_load_matrix": self._active_target_load_matrix.copy(),
            "balance_target_matrix": self._balance_target_matrix(),
            "load_matrix": self._load_matrix(),
            "sla_matrix": self._sla_matrix(),
            "ue_count_matrix": np.asarray(
                [
                    [self.base_env.get_slice_ue_count(i, s) for s in self.slice_types]
                    for i in range(self.n_gnbs)
                ],
                dtype=float,
            ),
        }

    def _initialize_load_scenario(self):
        if self.scenario_mode == "random":
            scenario_name, targets = self._sample_random_target_loads()
        else:
            scenario_name, targets = self._sample_snapshot_target_loads()
        self._active_scenario = scenario_name
        self._active_target_load_matrix = np.asarray(targets, dtype=float).copy()
        for gnb_id in range(self.n_gnbs):
            for s_idx, slice_type in enumerate(self.slice_types):
                target = float(targets[gnb_id, s_idx])
                n_ues = self._ue_count_for_target_load(target)
                for idx in range(n_ues):
                    x, y = self._sample_ue_position(gnb_id, slice_type, target, idx)
                    ue_id = self.base_env.add_ue(x=x, y=y, vx=0.0, vy=0.0, slice_type=slice_type)
                    self._force_attach(ue_id, gnb_id)
                self._set_slice_prb_load(gnb_id, slice_type, target)
        if self.print_scenarios:
            print(f"[GlobalPPO scenario] {scenario_name} targets={targets.round(2).tolist()}", flush=True)

    def _sample_snapshot_target_loads(self):
        if self.snapshot_scenario and self.snapshot_scenario != "mixed":
            if self.snapshot_scenario not in GLOBAL_SNAPSHOT_SCENARIOS:
                known = ", ".join(sorted(GLOBAL_SNAPSHOT_SCENARIOS))
                raise ValueError(f"Unknown snapshot_scenario={self.snapshot_scenario!r}. Known: {known}")
            return self.snapshot_scenario, GLOBAL_SNAPSHOT_SCENARIOS[self.snapshot_scenario].copy()

        names = tuple(GLOBAL_SNAPSHOT_SCENARIOS)
        block_idx = self._episode_index // self.snapshot_block_episodes
        name = names[block_idx % len(names)]
        self._episode_index += 1
        return name, GLOBAL_SNAPSHOT_SCENARIOS[name].copy()

    def _ue_count_for_target_load(self, target_load: float) -> int:
        if target_load >= 0.8:
            return self.high_load_ues
        if target_load >= 0.5:
            return self.medium_load_ues
        return self.light_load_ues

    def _sample_random_target_loads(self):
        base = np.full((self.n_gnbs, len(self.slice_types)), 0.45, dtype=float)
        choice = int(self.rng.integers(4))
        if choice == 0:
            g = int(self.rng.integers(self.n_gnbs))
            s = int(self.rng.integers(len(self.slice_types)))
            base[:, s] = [0.45, 0.50, 0.55]
            base[g, s] = 0.92
            return "one_overloaded_pair", base
        if choice == 1:
            s = int(self.rng.integers(len(self.slice_types)))
            overloaded = self.rng.choice(self.n_gnbs, size=2, replace=False)
            base[:, s] = 0.45
            for g in overloaded:
                base[int(g), s] = float(self.rng.uniform(0.86, 0.94))
            return "two_overloaded_pairs", base
        if choice == 2:
            base[2, 0] = 0.92
            base[1, 1] = 0.88
            base[0, 2] = 0.25
            return "slice_conflict", base
        return "random_mixed_load", self.rng.uniform(0.25, 0.92, size=base.shape)

    def _sample_ue_position(self, gnb_id: int, slice_type: str, target_load: float, idx: int):
        gnb = self.base_env._get_gnb_by_id(gnb_id)
        if gnb is None:
            return 0.0, 0.0
        if target_load >= 0.8 and idx < 3:
            neighbor_id = int(self.rng.choice(self.neighbors[gnb_id]))
            neighbor = self.base_env._get_gnb_by_id(neighbor_id)
            if neighbor is not None:
                return self._sample_border_point(gnb, neighbor)
        radius = 170.0 if target_load >= 0.8 else 210.0
        r = radius * math.sqrt(float(self.rng.random()))
        theta = 2.0 * math.pi * float(self.rng.random())
        return float(gnb.x + r * math.cos(theta)), float(gnb.y + r * math.sin(theta))

    def _sample_border_point(self, serving_gnb, neighbor_gnb):
        sx, sy = float(serving_gnb.x), float(serving_gnb.y)
        nx, ny = float(neighbor_gnb.x), float(neighbor_gnb.y)
        dx, dy = nx - sx, ny - sy
        dist = max(float(math.hypot(dx, dy)), 1e-9)
        ux, uy = dx / dist, dy / dist
        px, py = -uy, ux
        midpoint_x = sx + 0.5 * dx
        midpoint_y = sy + 0.5 * dy
        along = float(self.rng.normal(loc=-0.04 * dist, scale=35.0))
        perp = float(self.rng.normal(loc=0.0, scale=60.0))
        return float(midpoint_x + along * ux + perp * px), float(midpoint_y + along * uy + perp * py)

    def _force_attach(self, ue_id: int, gnb_id: int):
        ue = self.base_env.get_ue(ue_id)
        old = self.base_env._get_gnb_by_id(ue.serving_gnb)
        new = self.base_env._get_gnb_by_id(gnb_id)
        if new is None:
            return
        if old is not None:
            old.detach_ue(ue_id)
        if new.attach_ue(ue):
            ue.serving_gnb = int(gnb_id)
            ue.connected = True
            self.base_env._last_serving_gnb[ue_id] = ue.serving_gnb
            self.base_env._prev_serving_gnb[ue_id] = None
            self.base_env._invalidate_metric_caches()

    def _set_slice_prb_load(self, gnb_id: int, slice_type: str, target_load: float):
        ues = [
            ue for ue in self.base_env.get_all_ues()
            if ue.connected
            and ue.serving_gnb is not None
            and int(ue.serving_gnb) == int(gnb_id)
            and normalize_slice_type(getattr(ue, "slice_type", "eMBB")) == normalize_slice_type(slice_type)
        ]
        if not ues:
            return
        budget = max(int(self.base_env.get_slice_prb_budget(gnb_id, slice_type)), 1)
        target_used = int(round(float(np.clip(target_load, 0.0, 1.0)) * budget))
        per_ue = target_used // len(ues)
        remainder = target_used % len(ues)
        for idx, ue in enumerate(ues):
            prbs = int(per_ue + (1 if idx < remainder else 0))
            ue.prbs = prbs
            metrics = self.base_env._compute_link_metrics(self.base_env._get_gnb_by_id(gnb_id), ue)
            sinr_db = float(metrics.get("sinr_db", self.base_env.disconnect_sinr_db))
            if hasattr(self.base_env, "_frequency_selective_snr_vector"):
                budget_prbs = max(int(self.base_env.get_slice_prb_budget(gnb_id, slice_type)), 1)
                snr_vector = self.base_env._frequency_selective_snr_vector(
                    ue=ue,
                    nominal_sinr_db=sinr_db,
                    n_prbs=budget_prbs,
                )
                sinr_db = float(np.mean(snr_vector))
            bits_per_prb = self._bits_per_prb(sinr_db)
            source = getattr(ue, "traffic_source", None)
            if source is not None and hasattr(source, "packet_size"):
                source.packet_size = max(1.0, min(float(source.packet_size), bits_per_prb))
            self._set_ue_offered_bit_rate(
                ue,
                SNAPSHOT_DEMAND_SAFETY * prbs * bits_per_prb / max(float(self.base_env.step_dt), 1e-9),
            )

    def _bits_per_prb(self, sinr_db: float) -> float:
        if hasattr(self.base_env, "_ensure_mcs_scheduler") and self.base_env._ensure_mcs_scheduler():
            _mcs, bits_per_sym = self.base_env._mcs_codeset.mcs_rate_vs_error(float(sinr_db), 0.1)
            return max(158.0 * float(bits_per_sym), 1.0)

        sinr_linear = max(10.0 ** (float(sinr_db) / 10.0), 1e-6)
        spectral_eff = math.log2(1.0 + sinr_linear)
        spectral_eff = min(max(spectral_eff, 0.0), 8.0)
        return 180e3 * max(float(self.base_env.step_dt), 1e-9) * spectral_eff

    def _set_ue_offered_bit_rate(self, ue, bit_rate: float):
        source = getattr(ue, "traffic_source", None)
        if source is None:
            return
        rate = max(float(bit_rate), 0.0)
        if hasattr(source, "set_bit_rate"):
            source.set_bit_rate(rate)
        else:
            source.bit_rate = rate
            if hasattr(source, "packet_size"):
                source.packet_size = rate * max(float(self.base_env.step_dt), 1e-9)

    def _ensure_ue_queue_floor(self, ue, target_bits: float):
        bits = int(math.ceil(max(float(target_bits), 0.0)))
        if bits <= 0:
            return
        ue.queue = max(float(getattr(ue, "queue", 0.0)), float(bits))
        if hasattr(ue, "packet_queue") and not ue.packet_queue:
            packet_id = int(getattr(ue, "_next_packet_id", 0))
            arrival_step = int(getattr(ue, "_step_counter", 0))
            arrival_time_s = float(ue.get_current_time_s()) if hasattr(ue, "get_current_time_s") else 0.0
            ue.packet_queue.append(
                Packet(
                    bits=bits,
                    arrival_step=arrival_step,
                    arrival_time_s=arrival_time_s,
                    packet_id=packet_id,
                )
            )
            ue._next_packet_id = packet_id + 1
