from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import gymnasium as gym
import numpy as np

from global_ppo_3gnb_env import GlobalPPO3GNBEnv, SLICE_TYPES
from local_a3_agent_wrapper import LocalA3OffsetEnv, normalize_slice_type
from upper_agent_training_scenarios import CENTER_GAP_GNB_CONFIGS


class OracleDemandLowerA3Env(gym.Env):
    """Lower TD3 env with oracle upper bias and fixed per-UE PRB-demand reward.

    The environment keeps the upper PPO out of the loop. At every lower step it:

    1. computes an oracle directional upper bias from persistent demand load
       (`ue.upper_demand_prbs / gNB PRBs`);
    2. gives the controlled gNB's real lower observation to TD3;
    3. applies TD3 offsets for the controlled gNB;
    4. applies strong heuristic "perfect" offsets for every other gNB;
    5. advances radio time and rewards improvement in fixed-demand balance.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int = 7,
        controlled_gnb_id: int = 1,
        slice_types: Sequence[str] = SLICE_TYPES,
        training_scenarios: str | Sequence[str] | None = "jain_balance_controllable",
        scenario_selection: str = "cycle",
        topology: str = "medium_270m",
        upper_window_seconds: float = 1.0,
        local_steps_per_global: int = 10,
        radio_substeps: int = 20,
        control_interval_steps: int = 3,
        episode_control_intervals: int | None = None,
        warmup_steps: int = 1,
        safe_admission_enabled: bool = True,
        load_reward_weight: float = 1.0,
        offset_imitation_weight: float = 0.25,
        bias_alignment_weight: float = 0.10,
        smoothness_weight: float = 0.05,
        handover_failure_weight: float = 0.25,
        pingpong_weight: float = 0.25,
    ):
        super().__init__()
        self.seed_value = int(seed)
        self.controlled_gnb_id = int(controlled_gnb_id)
        self.slice_types = tuple(normalize_slice_type(s) for s in slice_types)
        self.control_interval_steps = max(1, int(control_interval_steps))
        self.load_reward_weight = float(load_reward_weight)
        self.offset_imitation_weight = float(offset_imitation_weight)
        self.bias_alignment_weight = float(bias_alignment_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.handover_failure_weight = float(handover_failure_weight)
        self.pingpong_weight = float(pingpong_weight)
        self.safe_admission_enabled = bool(safe_admission_enabled)

        self.upper_env = GlobalPPO3GNBEnv(
            seed=seed,
            slice_types=self.slice_types,
            scenario_mode="curriculum",
            training_scenarios=training_scenarios,
            scenario_selection=scenario_selection,
            gnb_configs=CENTER_GAP_GNB_CONFIGS[str(topology)],
            upper_window_seconds=upper_window_seconds,
            local_steps_per_global=local_steps_per_global,
            radio_substeps=radio_substeps,
            terminal_reward_only=False,
            safe_admission_enabled=safe_admission_enabled,
            warmup_steps=warmup_steps,
            print_scenarios=False,
        )
        if episode_control_intervals is None:
            max_duration_s = max(float(s.duration_s) for s in self.upper_env.training_scenarios)
            local_step_s = float(self.upper_env.local_step_seconds)
            interval_s = max(self.control_interval_steps * local_step_s, 1e-9)
            episode_control_intervals = int(math.ceil(max_duration_s / interval_s))
        self.episode_control_intervals = max(1, int(episode_control_intervals))

        self.base_env = self.upper_env.base_env
        self._install_safe_admission_load_provider()
        self.gnb_ids = tuple(range(self.upper_env.n_gnbs))
        self.neighbors = {
            int(gid): tuple(int(n) for n in self.upper_env.neighbors[int(gid)])
            for gid in self.gnb_ids
        }
        self.local_envs = self._make_local_envs()
        self.controlled_env = self.local_envs[self.controlled_gnb_id]
        self.observation_space = self.controlled_env.observation_space
        self.action_space = self.controlled_env.action_space

        self._control_interval_index = 0
        self._last_demand_loads = np.zeros((len(self.gnb_ids), len(self.slice_types)), dtype=float)
        self._last_oracle_bias = np.zeros(
            (len(self.gnb_ids), max(len(v) for v in self.neighbors.values()), len(self.slice_types)),
            dtype=np.float32,
        )
        self._previous_controlled_offsets = np.zeros(self.action_space.shape, dtype=float)
        self._last_info: Dict = {}

    def reset(self, *, seed=None, options=None):
        obs, info = self.upper_env.reset(seed=seed, options=options)
        del obs
        self.base_env = self.upper_env.base_env
        self._install_safe_admission_load_provider()
        for local_env in self.local_envs.values():
            local_env.base_env = self.base_env
            self._reset_local_view(local_env)
        self._control_interval_index = 0
        self._previous_controlled_offsets = np.zeros(self.action_space.shape, dtype=float)
        self._last_demand_loads = self._demand_load_matrix()
        self._set_oracle_bias()
        self._last_info = {
            "upper_reset_info": dict(info),
            "scenario_name": str(info.get("scenario_name", "")),
            "controlled_gnb": int(self.controlled_gnb_id),
            "load_measurement_mode": "persistent_fixed_ue_upper_demand_prbs",
            "demand_load_matrix_start": self._last_demand_loads.copy(),
        }
        return self.controlled_env._build_observation(), dict(self._last_info)

    def step(self, action):
        start_loads = self._demand_load_matrix()
        oracle_bias = self._set_oracle_bias(start_loads)
        perfect_offsets, _perfect_debug = self.upper_env._compute_strong_local_offsets(oracle_bias)

        controlled_proto = self.controlled_env._normalize_action(action)
        controlled_raw_action = np.asarray(action, dtype=float).reshape(-1)
        previous = self._applied_vector(self.controlled_env)
        self.controlled_env._apply_proto_offsets(controlled_proto)
        controlled_applied = self._applied_vector(self.controlled_env)

        handover_stats: Dict[int, Dict[str, int]] = {}
        for gnb_id, local_env in self.local_envs.items():
            if int(gnb_id) != self.controlled_gnb_id:
                local_env._apply_proto_offsets(self._perfect_vector(local_env, perfect_offsets))
            self._apply_offsets_to_base(local_env)
            handover_stats[int(gnb_id)] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "ping_pongs": 0,
            }

        terminated = False
        truncated = False
        base_info = {}
        handovers_before = len(getattr(self.base_env, "handover_events", []))
        if self.safe_admission_enabled and hasattr(self.base_env, "begin_safe_admission_window"):
            self.base_env.begin_safe_admission_window(oracle_bias, self.slice_types)
        self.base_env.begin_radio_measurement_window()
        for _ in range(self.control_interval_steps):
            _obs, _reward, terminated, truncated, base_info = self.base_env.step(0)
            if terminated or truncated:
                break
        new_events = list(getattr(self.base_env, "handover_events", []))[handovers_before:]
        for event in new_events:
            source = int(event.get("from_gnb", -1))
            if source in handover_stats:
                handover_stats[source]["attempts"] += 1
                handover_stats[source]["successes"] += 1

        end_loads = self._demand_load_matrix()
        self._last_demand_loads = end_loads.copy()
        self._control_interval_index += 1
        truncated = bool(truncated or self._control_interval_index >= self.episode_control_intervals)

        reward, reward_parts = self._reward(
            start_loads=start_loads,
            end_loads=end_loads,
            oracle_bias=oracle_bias,
            perfect_offsets=perfect_offsets,
            controlled_applied=controlled_applied,
            previous_applied=previous,
            handover_stats=handover_stats.get(self.controlled_gnb_id, {}),
        )
        self._previous_controlled_offsets = controlled_applied.copy()
        self._set_oracle_bias(end_loads)

        obs = self.controlled_env._build_observation()
        info = self._build_info(
            reward=reward,
            reward_parts=reward_parts,
            start_loads=start_loads,
            end_loads=end_loads,
            oracle_bias=oracle_bias,
            perfect_offsets=perfect_offsets,
            controlled_raw_action=controlled_raw_action,
            controlled_proto=controlled_proto,
            controlled_applied=controlled_applied,
            handover_stats=handover_stats,
            base_info=base_info,
            terminated=terminated,
            truncated=truncated,
        )
        self._last_info = dict(info)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.upper_env.close()

    def _install_safe_admission_load_provider(self) -> None:
        if hasattr(self.base_env, "safe_admission_load_provider"):
            self.base_env.safe_admission_load_provider = self._safe_admission_demand_load

    def _safe_admission_demand_load(self, gnb_id: int, slice_type: str) -> float:
        normalized_slice = normalize_slice_type(slice_type)
        gnb = self.base_env._get_gnb_by_id(int(gnb_id))
        capacity = max(float(getattr(gnb, "n_prbs", 0.0)) if gnb is not None else 0.0, 1.0)
        demand = 0.0
        for ue in self.base_env.get_all_ues():
            if (
                getattr(ue, "connected", False)
                and ue.serving_gnb is not None
                and int(ue.serving_gnb) == int(gnb_id)
                and normalize_slice_type(getattr(ue, "slice_type", "eMBB")) == normalized_slice
            ):
                demand += max(float(getattr(ue, "upper_demand_prbs", 0.0)), 0.0)
        return float(demand / capacity)

    def _make_local_envs(self) -> Dict[int, LocalA3OffsetEnv]:
        return {
            int(gid): LocalA3OffsetEnv(
                self.base_env,
                gnb_id=int(gid),
                neighbor_ids=self.neighbors[int(gid)],
                slice_types=self.slice_types,
                steps_per_action=1,
                ttt=1,
                load_observation_provider=self._demand_load_dict,
            )
            for gid in self.gnb_ids
        }

    def _reset_local_view(self, local_env: LocalA3OffsetEnv) -> None:
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
            for ue in self.base_env.get_all_ues()
        }
        local_env._prev_serving = {ue_id: None for ue_id in local_env._last_serving}
        local_env._last_reward_breakdown = {}

    def _demand_load_matrix(self) -> np.ndarray:
        return np.asarray(self.upper_env._load_matrix(), dtype=float)

    def _demand_load_dict(self) -> Dict[Tuple[int, str], float]:
        loads = self._demand_load_matrix()
        return {
            (int(gnb_id), slice_type): float(loads[int(gnb_id), s_idx])
            for gnb_id in self.gnb_ids
            for s_idx, slice_type in enumerate(self.slice_types)
        }

    def _set_oracle_bias(self, loads: np.ndarray | None = None) -> np.ndarray:
        bias = self._oracle_directional_bias(self._demand_load_matrix() if loads is None else loads)
        self._last_oracle_bias = bias.copy()
        bias_dict = self._bias_dict(bias)
        for local_env in self.local_envs.values():
            local_env.set_global_bias(bias_dict)
        return bias

    def _oracle_directional_bias(self, loads: np.ndarray) -> np.ndarray:
        loads = np.asarray(loads, dtype=float)
        means = loads.mean(axis=0)
        bias = np.zeros_like(self._last_oracle_bias, dtype=np.float32)
        for source in self.gnb_ids:
            for s_idx, slice_type in enumerate(self.slice_types):
                source_excess = float(loads[source, s_idx] - means[s_idx])
                deficits = [
                    max(float(means[s_idx] - loads[target, s_idx]), 0.0)
                    for target in self.neighbors[source]
                ]
                total_deficit = sum(deficits)
                for slot, target in enumerate(self.neighbors[source]):
                    target_excess = float(loads[target, s_idx] - means[s_idx])
                    if source_excess > 1e-9 and total_deficit > 0.0 and deficits[slot] > 0.0:
                        if self._has_radio_feasible_ue(source, target, slice_type):
                            bias[source, slot, s_idx] = -float(deficits[slot] / total_deficit)
                    elif target_excess > 1e-9:
                        bias[source, slot, s_idx] = float(
                            np.clip(target_excess / max(float(means[s_idx]), 0.05), 0.0, 1.0)
                        )
        return bias

    def _has_radio_feasible_ue(self, source: int, target: int, slice_type: str) -> bool:
        source_gnb = self.base_env._get_gnb_by_id(int(source))
        target_gnb = self.base_env._get_gnb_by_id(int(target))
        if source_gnb is None or target_gnb is None:
            return False
        for ue in self.base_env.get_all_ues():
            if (
                not getattr(ue, "connected", False)
                or ue.serving_gnb is None
                or int(ue.serving_gnb) != int(source)
                or normalize_slice_type(getattr(ue, "slice_type", "eMBB")) != slice_type
            ):
                continue
            if not self.base_env._is_in_coverage(target_gnb, ue):
                continue
            margin = (
                self.base_env._measure_rsrp(target_gnb, ue)
                - self.base_env._measure_rsrp(source_gnb, ue)
            )
            if margin > -5.0:
                return True
        return False

    def _bias_dict(self, bias: np.ndarray) -> Dict[Tuple[int, int, str], float]:
        return {
            (int(source), int(target), slice_type): float(bias[int(source), slot, s_idx])
            for source in self.gnb_ids
            for slot, target in enumerate(self.neighbors[int(source)])
            for s_idx, slice_type in enumerate(self.slice_types)
        }

    def _applied_vector(self, local_env: LocalA3OffsetEnv) -> np.ndarray:
        offsets = local_env.get_applied_offsets()
        return np.asarray(
            [
                float(offsets.get((neighbor_id, slice_type), 0.0))
                for neighbor_id in local_env.neighbor_ids
                for slice_type in local_env.slice_types
            ],
            dtype=float,
        )

    def _perfect_vector(self, local_env: LocalA3OffsetEnv, perfect_offsets: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                float(perfect_offsets[int(local_env.gnb_id), slot, s_idx])
                for slot, _neighbor_id in enumerate(local_env.neighbor_ids)
                for s_idx, _slice_type in enumerate(local_env.slice_types)
            ],
            dtype=np.float32,
        )

    def _controlled_vector_labels(self) -> list[str]:
        return [
            f"to_gnb{int(neighbor_id)}_{slice_type}"
            for neighbor_id in self.controlled_env.neighbor_ids
            for slice_type in self.controlled_env.slice_types
        ]

    def _controlled_bias_vector(self, oracle_bias: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                float(oracle_bias[self.controlled_gnb_id, slot, s_idx])
                for slot, _neighbor_id in enumerate(self.controlled_env.neighbor_ids)
                for s_idx, _slice_type in enumerate(self.controlled_env.slice_types)
            ],
            dtype=np.float32,
        )

    def _apply_offsets_to_base(self, local_env: LocalA3OffsetEnv) -> None:
        for (neighbor_id, slice_type), offset in local_env.get_applied_offsets().items():
            self.base_env.set_a3_offset(
                int(local_env.gnb_id),
                int(neighbor_id),
                slice_type,
                float(offset),
            )

    def _load_balance_cost(self, loads: np.ndarray) -> float:
        loads = np.asarray(loads, dtype=float)
        means = loads.mean(axis=0, keepdims=True)
        return float(np.sum((loads - means) ** 2))

    def _demand_std_cost(self, loads: np.ndarray) -> float:
        loads = np.asarray(loads, dtype=float)
        if loads.size == 0:
            return 0.0
        per_gnb_total_demand = np.sum(np.maximum(loads, 0.0), axis=1)
        return float(np.std(per_gnb_total_demand))

    def _controlled_gnb_deviation(self, loads: np.ndarray) -> float:
        """Mean absolute deviation of the controlled gNB's load from the per-slice mean.

        Only measures the controlled gNB's contribution — improvement here is
        directly attributable to the controlled gNB's actions.
        """
        loads = np.asarray(loads, dtype=float)
        if loads.size == 0:
            return 0.0
        controlled = loads[self.controlled_gnb_id]
        mean = loads.mean(axis=0)
        return float(np.mean(np.abs(controlled - mean)))

    def _reward(
        self,
        *,
        start_loads: np.ndarray,
        end_loads: np.ndarray,
        oracle_bias: np.ndarray,
        perfect_offsets: np.ndarray,
        controlled_applied: np.ndarray,
        previous_applied: np.ndarray,
        handover_stats: Mapping[str, int],
    ) -> Tuple[float, Dict[str, float]]:
        del perfect_offsets, handover_stats, oracle_bias

        # Dense direction reward: reward the agent for applying offsets whose sign
        # correctly opposes the observed load imbalance between the controlled gNB
        # and each neighbour.  Computed purely from the loads in the observation —
        # no oracle, no perfect offsets — so the signal is available in both
        # pre-training and joint deployment.
        #
        # Logic: if controlled gNB is overloaded vs neighbour j for slice s
        #   → correct action is negative offset (make A3 easier → push UEs out)
        #   → load_diff > 0, applied < 0 → agreement = -load_diff * applied > 0  ✓
        # If neighbour is overloaded:
        #   → correct action is positive offset (retain UEs)
        #   → load_diff < 0, applied > 0 → agreement = -load_diff * applied > 0  ✓
        direction_signal = 0.0
        n_pairs = 0
        idx = 0
        scale = max(float(np.max(np.abs(start_loads))), 0.05)
        for slot, neighbor_id in enumerate(self.controlled_env.neighbor_ids):
            for s_idx, _slice_type in enumerate(self.slice_types):
                load_diff = float(
                    start_loads[self.controlled_gnb_id, s_idx]
                    - start_loads[int(neighbor_id), s_idx]
                )
                normalized_diff = float(np.clip(load_diff / scale, -1.0, 1.0))
                agreement = -normalized_diff * (float(controlled_applied[idx]) / 6.0)
                direction_signal += float(np.tanh(agreement))
                n_pairs += 1
                idx += 1
        if n_pairs > 1:
            direction_signal /= n_pairs
        direction_reward = self.bias_alignment_weight * direction_signal

        # Sparse outcome reward: did the controlled gNB's own deviation from the
        # mean actually improve?  Non-zero only when handovers occur, but measures
        # the true goal.
        start_demand_std = self._controlled_gnb_deviation(start_loads)
        end_demand_std = self._controlled_gnb_deviation(end_loads)
        demand_std_improvement = start_demand_std - end_demand_std
        demand_std_reward = self.load_reward_weight * demand_std_improvement

        smoothness_penalty = -self.smoothness_weight * float(
            np.mean(((controlled_applied - previous_applied) / 12.0) ** 2)
        )

        reward_terms = {
            "direction_reward": direction_reward,
            "demand_std_reward": demand_std_reward,
            "smoothness_penalty": smoothness_penalty,
        }
        diagnostics = {
            "start_demand_std": float(start_demand_std),
            "end_demand_std": float(end_demand_std),
            "demand_std_improvement": float(demand_std_improvement),
        }
        return float(sum(reward_terms.values())), {**reward_terms, **diagnostics}

    def _build_info(
        self,
        *,
        reward: float,
        reward_parts: Mapping[str, float],
        start_loads: np.ndarray,
        end_loads: np.ndarray,
        oracle_bias: np.ndarray,
        perfect_offsets: np.ndarray,
        controlled_raw_action: np.ndarray,
        controlled_proto: np.ndarray,
        controlled_applied: np.ndarray,
        handover_stats: Mapping[int, Mapping[str, int]],
        base_info: Mapping,
        terminated: bool,
        truncated: bool,
    ) -> Dict:
        controlled_perfect = self._perfect_vector(self.controlled_env, perfect_offsets)
        qos = {
            "slice_sla_metrics": (
                self.base_env.get_slice_sla_metrics()
                if hasattr(self.base_env, "get_slice_sla_metrics")
                else {}
            ),
            "slice_sla_flags": (
                self.base_env.get_slice_sla_flags()
                if hasattr(self.base_env, "get_slice_sla_flags")
                else {}
            ),
            "slice_sla_severity": (
                self.base_env.get_slice_sla_severity()
                if hasattr(self.base_env, "get_slice_sla_severity")
                else {}
            ),
        }
        return {
            "controlled_gnb": int(self.controlled_gnb_id),
            "control_interval_index": int(self._control_interval_index),
            "load_measurement_mode": "persistent_fixed_ue_upper_demand_prbs",
            "demand_load_matrix_start": np.asarray(start_loads, dtype=float).copy(),
            "demand_load_matrix_end": np.asarray(end_loads, dtype=float).copy(),
            "oracle_directional_bias": np.asarray(oracle_bias, dtype=float).copy(),
            "perfect_offsets": np.asarray(perfect_offsets, dtype=float).copy(),
            "controlled_raw_action": np.asarray(controlled_raw_action, dtype=float).copy(),
            "controlled_proto_offsets": np.asarray(controlled_proto, dtype=float).copy(),
            "controlled_applied_offsets": np.asarray(controlled_applied, dtype=float).copy(),
            "controlled_offset_labels": self._controlled_vector_labels(),
            "controlled_oracle_bias": self._controlled_bias_vector(oracle_bias),
            "controlled_perfect_offsets": controlled_perfect,
            "qos": qos,
            "reward": float(reward),
            "reward_parts": dict(reward_parts),
            "handover_stats": {int(k): dict(v) for k, v in handover_stats.items()},
            "safe_admission": (
                self.base_env.get_safe_admission_state()
                if hasattr(self.base_env, "get_safe_admission_state")
                else {"enabled": False}
            ),
            "base_info": dict(base_info or {}),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
