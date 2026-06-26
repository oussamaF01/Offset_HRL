from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import gymnasium as gym
import numpy as np

from global_ppo_3gnb_env import GlobalPPO3GNBEnv, SLICE_TYPES
from local_a3_agent_wrapper import LocalA3OffsetEnv, normalize_slice_type


class SharedLocalA3TD3Env(gym.Env):
    """Turn-based shared-policy lower A3 environment.

    One TD3 policy is reused for every gNB.  The environment presents the local
    observation for one gNB at a time, buffers that gNB's action, and advances
    simulator time only after all gNBs have supplied actions for the same lower
    control instant.  This keeps the lower-agent time span coherent:

        t: observe gNB0, gNB1, gNB2 with the same radio state
        t: apply all offsets together
        t -> t + control interval: advance the simulator
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int = 7,
        slice_types: Sequence[str] = SLICE_TYPES,
        training_scenarios: str | Sequence[str] | None = "jain_control_mixed",
        scenario_selection: str = "cycle",
        upper_window_seconds: float = 1.0,
        local_steps_per_global: int = 10,
        radio_substeps: int = 20,
        control_interval_steps: int = 1,
        episode_control_intervals: int | None = None,
        bias_update_intervals: int = 1,
        safe_admission_enabled: bool = False,
        warmup_steps: int = 1,
        max_handovers_per_local_step: int = 3,
    ):
        super().__init__()
        self.seed_value = int(seed)
        self.slice_types = tuple(normalize_slice_type(s) for s in slice_types)
        self.control_interval_steps = max(1, int(control_interval_steps))
        self.bias_update_intervals = max(1, int(bias_update_intervals))

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
            safe_admission_enabled=safe_admission_enabled,
            warmup_steps=warmup_steps,
            max_handovers_per_local_step=max_handovers_per_local_step,
        )
        if episode_control_intervals is None:
            max_duration_s = max(
                float(scenario.duration_s)
                for scenario in self.upper_env.training_scenarios
            )
            local_step_s = float(self.upper_env.local_step_seconds)
            interval_s = max(self.control_interval_steps * local_step_s, 1e-9)
            episode_control_intervals = int(math.ceil(max_duration_s / interval_s))
        self.episode_control_intervals = max(1, int(episode_control_intervals))
        self.base_env = self.upper_env.base_env
        self.gnb_ids = tuple(range(self.upper_env.n_gnbs))
        self.neighbors = {
            int(gnb_id): tuple(int(n) for n in self.upper_env.neighbors[int(gnb_id)])
            for gnb_id in self.gnb_ids
        }
        self.local_envs = {
            int(gnb_id): LocalA3OffsetEnv(
                self.base_env,
                gnb_id=int(gnb_id),
                neighbor_ids=self.neighbors[int(gnb_id)],
                slice_types=self.slice_types,
                steps_per_action=1,
                ttt=1,
            )
            for gnb_id in self.gnb_ids
        }

        first_env = self.local_envs[self.gnb_ids[0]]
        self.observation_space = first_env.observation_space
        self.action_space = first_env.action_space

        self._turn_index = 0
        self._agent_order = list(self.gnb_ids)
        self._control_interval_index = 0
        self._cycle_start_loads = np.zeros((len(self.gnb_ids), len(self.slice_types)))
        self._pending_smoothness_penalty = 0.0
        self._pending_actions: Dict[int, np.ndarray] = {}
        self._last_info: Dict = {}
        self._rng = np.random.default_rng(self.seed_value)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = self.upper_env.reset(seed=seed, options=options)
        del obs
        self.base_env = self.upper_env.base_env
        for local_env in self.local_envs.values():
            local_env.base_env = self.base_env
            self._reset_local_view(local_env)
        self._turn_index = 0
        self._agent_order = list(self.gnb_ids)
        self._control_interval_index = 0
        self._pending_actions = {}
        self._pending_smoothness_penalty = 0.0
        self._set_fake_directional_bias(force=True)
        self._cycle_start_loads = self._load_matrix()
        self._last_info = {
            "upper_reset_info": dict(info),
            "controlled_gnb": int(self._current_gnb()),
            "turn_index": int(self._turn_index),
            "control_interval_index": int(self._control_interval_index),
            "time_advanced": False,
        }
        return self._current_observation(), dict(self._last_info)

    def step(self, action):
        gnb_id = int(self._current_gnb())
        local_env = self.local_envs[gnb_id]
        proto = local_env._normalize_action(action)

        previous_offsets = local_env.get_applied_offsets()
        previous_vector = np.asarray(
            [
                previous_offsets.get((neighbor_id, slice_type), 0.0)
                for neighbor_id in local_env.neighbor_ids
                for slice_type in local_env.slice_types
            ],
            dtype=float,
        )
        local_env._apply_proto_offsets(proto)
        applied = np.asarray(
            [
                local_env.get_applied_offsets().get((neighbor_id, slice_type), 0.0)
                for neighbor_id in local_env.neighbor_ids
                for slice_type in local_env.slice_types
            ],
            dtype=float,
        )
        self._apply_local_offsets_to_base(gnb_id)
        self._pending_actions[gnb_id] = applied
        self._pending_smoothness_penalty += float(np.mean(((applied - previous_vector) / 12.0) ** 2))

        if self._turn_index < len(self._agent_order) - 1:
            self._turn_index += 1
            info = self._build_turn_info(
                controlled_gnb=gnb_id,
                reward=0.0,
                time_advanced=False,
                handovers=0,
                terminated=False,
                truncated=False,
            )
            return self._current_observation(), 0.0, False, False, info

        reward, terminated, truncated, info = self._advance_control_interval(gnb_id)
        self._control_interval_index += 1
        truncated = bool(truncated or self._control_interval_index >= self.episode_control_intervals)
        self._rotate_agent_order()
        self._turn_index = 0
        self._pending_actions = {}
        self._pending_smoothness_penalty = 0.0
        self._cycle_start_loads = np.asarray(
            info.get("load_matrix_end", self._load_matrix()),
            dtype=float,
        )
        info.update({
            "next_controlled_gnb": int(self._current_gnb()),
            "agent_order": list(map(int, self._agent_order)),
            "turn_index": int(self._turn_index),
            "truncated": bool(truncated),
        })
        self._last_info = dict(info)
        obs = self._current_observation()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.upper_env.close()

    def _current_gnb(self) -> int:
        return int(self._agent_order[self._turn_index])

    def _current_observation(self) -> np.ndarray:
        return self.local_envs[self._current_gnb()]._build_observation()

    def _reset_local_view(self, local_env: LocalA3OffsetEnv) -> None:
        """Clear one local controller without resetting the shared simulator."""
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

    def _rotate_agent_order(self) -> None:
        shift = self._control_interval_index % len(self.gnb_ids)
        ordered = list(self.gnb_ids)
        self._agent_order = ordered[shift:] + ordered[:shift]

    def _load_matrix(self) -> np.ndarray:
        loads = {
            (int(gnb_id), slice_type): self.base_env.estimate_slice_load(
                int(gnb_id), slice_type
            )
            for gnb_id in self.gnb_ids
            for slice_type in self.slice_types
        }
        return np.asarray(
            [
                [
                    float(loads.get((gnb_id, slice_type), 0.0))
                    for slice_type in self.slice_types
                ]
                for gnb_id in self.gnb_ids
            ],
            dtype=float,
        )

    def _window_average_load_matrix(self) -> np.ndarray:
        loads = self.base_env.get_window_average_slice_loads()
        return np.asarray(
            [
                [
                    float(loads.get((gnb_id, slice_type), 0.0))
                    for slice_type in self.slice_types
                ]
                for gnb_id in self.gnb_ids
            ],
            dtype=float,
        )

    def _load_balance_cost(self, loads: np.ndarray) -> float:
        loads = np.asarray(loads, dtype=float)
        means = loads.mean(axis=0, keepdims=True)
        return float(np.sum((loads - means) ** 2))

    def _set_fake_directional_bias(self, force: bool = False) -> None:
        if (not force) and self._control_interval_index % self.bias_update_intervals != 0:
            return
        # Use window-average useful PRBs (demand-proportional) to mirror the upper agent.
        loads = self._window_average_load_matrix()
        bias = {}
        for src_id in self.gnb_ids:
            for tgt_id in self.neighbors[src_id]:
                for s_idx, slice_type in enumerate(self.slice_types):
                    diff = float(loads[int(src_id), s_idx] - loads[int(tgt_id), s_idx])
                    if abs(diff) < 0.05:
                        value = 0.0
                    else:
                        value = float(np.clip(-diff / 0.35, -1.0, 1.0))
                    bias[(int(src_id), int(tgt_id), slice_type)] = value
        for local_env in self.local_envs.values():
            local_env.set_global_bias(bias)

    def _apply_local_offsets_to_base(self, gnb_id: int) -> None:
        local_env = self.local_envs[int(gnb_id)]
        for (neighbor_id, slice_type), offset in local_env.get_applied_offsets().items():
            self.base_env.set_a3_offset(int(gnb_id), int(neighbor_id), slice_type, float(offset))

    def _advance_control_interval(self, controlled_gnb: int):
        handovers_before = len(getattr(self.base_env, "handover_events", []))
        terminated = False
        truncated = False
        base_info = {}
        noop = 0
        self.base_env.begin_radio_measurement_window()
        for _ in range(self.control_interval_steps):
            _obs, _reward, terminated, truncated, base_info = self.base_env.step(noop)
            if terminated or truncated:
                break
        handovers = len(getattr(self.base_env, "handover_events", [])) - handovers_before

        end_loads = self._window_average_load_matrix()
        start_cost = self._load_balance_cost(self._cycle_start_loads)
        end_cost = self._load_balance_cost(end_loads)
        load_reward = float(np.clip((start_cost - end_cost) / max(start_cost, 0.05), -1.0, 1.0))
        sla_penalty = -float(np.sum(self._sla_matrix()))
        smoothness_penalty = -0.05 * float(self._pending_smoothness_penalty)

        # Soft bias-alignment penalty: sum over all gNBs and directions.
        # Penalises applied offsets that contradict the upper bias direction.
        # Averaged per active direction so the scale doesn't grow with topology size.
        bias_align_penalty = 0.0
        n_active = 0
        for gnb_id, local_env in self.local_envs.items():
            for neighbor_id in local_env.neighbor_ids:
                for slice_type in local_env.slice_types:
                    b = local_env._bias_for(int(gnb_id), int(neighbor_id), slice_type)
                    if abs(b) < 0.1:
                        continue
                    applied = local_env._offsets.get((int(neighbor_id), slice_type), 0.0)
                    agreement = b * (applied / 6.0)
                    bias_align_penalty -= local_env.w_bias_align * max(0.0, -agreement)
                    n_active += 1
        if n_active > 1:
            bias_align_penalty /= n_active

        reward = load_reward + sla_penalty + smoothness_penalty + bias_align_penalty

        self._set_fake_directional_bias()
        info = self._build_turn_info(
            controlled_gnb=controlled_gnb,
            reward=reward,
            time_advanced=True,
            handovers=handovers,
            terminated=terminated,
            truncated=truncated,
        )
        info.update({
            "base_info": dict(base_info or {}),
            "load_matrix_start": self._cycle_start_loads.copy(),
            "load_matrix_end": end_loads.copy(),
            "load_measurement_mode": "control_interval_average_useful_prbs",
            "load_balance_cost_start": float(start_cost),
            "load_balance_cost_end": float(end_cost),
            "reward_load_improvement": float(load_reward),
            "reward_sla_penalty": float(sla_penalty),
            "reward_smoothness_penalty": float(smoothness_penalty),
            "reward_bias_align_penalty": float(bias_align_penalty),
            "pending_actions": {
                int(agent_id): action.copy()
                for agent_id, action in self._pending_actions.items()
            },
        })
        self._last_info = dict(info)
        return reward, terminated, truncated, info

    def _sla_matrix(self) -> np.ndarray:
        severities = (
            self.base_env.get_slice_sla_severity()
            if hasattr(self.base_env, "get_slice_sla_severity")
            else {}
        )
        return np.asarray(
            [
                [
                    float(np.clip(severities.get((gnb_id, slice_type), 0.0), 0.0, 1.0))
                    for slice_type in self.slice_types
                ]
                for gnb_id in self.gnb_ids
            ],
            dtype=float,
        )

    def _build_turn_info(
        self,
        controlled_gnb: int,
        reward: float,
        time_advanced: bool,
        handovers: int,
        terminated: bool,
        truncated: bool,
    ) -> Dict:
        current_gnb = int(self._current_gnb())
        return {
            "controlled_gnb": int(controlled_gnb),
            "next_controlled_gnb": current_gnb,
            "agent_order": list(map(int, self._agent_order)),
            "turn_index": int(self._turn_index),
            "control_interval_index": int(self._control_interval_index),
            "control_interval_steps": int(self.control_interval_steps),
            "time_advanced": bool(time_advanced),
            "shared_policy": True,
            "shared_reward": bool(time_advanced),
            "handover_count": int(handovers),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
