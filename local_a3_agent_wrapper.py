#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np


SLICE_TYPES = ("eMBB", "URLLC", "mMTC")
VALID_A3_OFFSETS_DB = np.array([-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0], dtype=float)


def normalize_slice_type(slice_type: str) -> str:
    raw = str(slice_type or "eMBB")
    for known in SLICE_TYPES:
        if raw.upper() == known.upper():
            return known
    return raw


def quantize_a3_offset(offset_db: float) -> float:
    """Project a continuous proto-offset onto the valid A3 offset set."""
    value = float(np.clip(offset_db, VALID_A3_OFFSETS_DB[0], VALID_A3_OFFSETS_DB[-1]))
    distances = np.abs(VALID_A3_OFFSETS_DB - value)
    candidates = VALID_A3_OFFSETS_DB[np.isclose(distances, distances.min())]
    return float(candidates[np.argmin(np.abs(candidates))])


class LocalA3OffsetEnv(gym.Env):
    """
    Local HRL wrapper for one gNB's slice-aware A3 offsets.

    The wrapped base environment remains the radio/mobility simulator. This
    class owns the local-agent interface described in the HRL design:

    observation block for each neighbor j and slice s:
        [b_i,s, b_j,s, tau_i,j,s, K_i,s, K_j,s, v_i,s, v_j,s,
         prev_offset_i,j,s, HF_i,j,s, PP_i,j,s]

    where tau_i,j,s = 0.5 * (b_i,s - b_j,s).
    Negative tau encourages i -> j handover; positive tau discourages it.

    action:
        continuous proto-offsets in [-6, 6] dB, one per neighbor/slice pair.

    The action is quantized to {-6, -4, -2, 0, 2, 4, 6} dB and used to evaluate
    the simplified A3 rule:
        RSRP_target > RSRP_serving + applied_offset

    Notes:
    - The wrapper intentionally calls a few base-env internal radio helpers.
      That keeps learning/control logic outside MultiGNBWrapper while reusing
      its radio model.
    - This wrapper controls outgoing handovers from one serving gNB. Use one
      wrapper per gNB for decentralized execution.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env,
        gnb_id: int,
        neighbor_ids: Optional[Sequence[int]] = None,
        slice_types: Sequence[str] = SLICE_TYPES,
        global_bias: Optional[Dict[Tuple[int, str], float]] = None,
        k_ref: Optional[Dict[str, float]] = None,
        k_target: Optional[Dict[str, float]] = None,
        alpha_k: float = 2.0,
        lambda_track: float = 1.0,
        lambda_delta: float = 0.10,
        w_hf: float = 1.0,
        w_pp: float = 0.3,
        w_sla: float = 1.0,
        alpha_hf: float = 2.0,
        alpha_pp: float = 1.0,
        alpha_sla_target: float = 2.0,
        ttt: int = 1,
        handover_margin_db: float = 0.0,
        steps_per_action: int = 1,
        normalize_observation: bool = True,
    ):
        super().__init__()
        self.base_env = base_env
        self.gnb_id = int(gnb_id)
        self.slice_types = tuple(normalize_slice_type(s) for s in slice_types)
        self.neighbor_ids = self._resolve_neighbor_ids(neighbor_ids)
        self.global_bias = dict(global_bias or {})

        self.k_ref = {
            slice_type: float((k_ref or {}).get(slice_type, 20.0))
            for slice_type in self.slice_types
        }
        self.k_target = {
            slice_type: float((k_target or {}).get(slice_type, 0.5))
            for slice_type in self.slice_types
        }

        self.alpha_k = float(alpha_k)
        self.lambda_track = float(lambda_track)
        self.lambda_delta = float(lambda_delta)
        self.w_hf = float(w_hf)
        self.w_pp = float(w_pp)
        self.w_sla = float(w_sla)
        self.alpha_hf = float(alpha_hf)
        self.alpha_pp = float(alpha_pp)
        self.alpha_sla_target = float(alpha_sla_target)
        self.ttt = max(1, int(ttt))
        self.handover_margin_db = float(handover_margin_db)
        self.steps_per_action = max(1, int(steps_per_action))
        self.normalize_observation = bool(normalize_observation)

        self._offsets: Dict[Tuple[int, str], float] = {
            (neighbor_id, slice_type): 0.0
            for neighbor_id in self.neighbor_ids
            for slice_type in self.slice_types
        }
        self._prev_proto_offsets = dict(self._offsets)
        self._ttt_counters: Dict[int, Dict[Tuple[int, str], int]] = defaultdict(dict)
        self._mobility_counters: Dict[Tuple[int, str], Dict[str, int]] = {
            key: {"attempts": 0, "successes": 0, "failures": 0, "ping_pongs": 0}
            for key in self._offsets
        }
        self._last_serving: Dict[int, Optional[int]] = {}
        self._prev_serving: Dict[int, Optional[int]] = {}
        self._last_reward_breakdown = {}

        n_blocks = len(self.neighbor_ids) * len(self.slice_types)
        self.action_space = gym.spaces.Box(
            low=-6.0,
            high=6.0,
            shape=(n_blocks,),
            dtype=np.float32,
        )
        # 10 values per neighbor-slice block:
        # [b_i, b_j, tau, K_i, K_j, v_i, v_j, prev_offset, HF, PP]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_blocks * 10,),
            dtype=np.float32,
        )

    def _resolve_neighbor_ids(self, neighbor_ids: Optional[Sequence[int]]) -> List[int]:
        valid_ids = [int(gnb.id) for gnb in self.base_env.gnbs if int(gnb.id) != self.gnb_id]
        if neighbor_ids is None:
            return valid_ids

        selected = []
        valid_set = set(valid_ids)
        for neighbor_id in neighbor_ids:
            neighbor_id = int(neighbor_id)
            if neighbor_id == self.gnb_id:
                continue
            if neighbor_id not in valid_set:
                raise ValueError(f"Unknown neighbor gNB id {neighbor_id}")
            selected.append(neighbor_id)
        return selected

    def set_global_bias(self, bias: Dict[Tuple[int, str], float]):
        self.global_bias = {
            (int(gnb_id), normalize_slice_type(slice_type)): float(value)
            for (gnb_id, slice_type), value in dict(bias).items()
        }

    def reset(self, *, seed=None, options=None):
        # The base env reset obs is intentionally discarded: _build_observation()
        # reconstructs the local observation from live base_env state, which is
        # always more up-to-date and slice-aware than the raw base obs.
        _base_obs, _base_info = self.base_env.reset(seed=seed, options=options)
        for key in self._offsets:
            self._offsets[key] = 0.0
            self._prev_proto_offsets[key] = 0.0
            self._mobility_counters[key] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "ping_pongs": 0,
            }
        self._ttt_counters.clear()
        self._last_serving = {
            int(ue.id): ue.serving_gnb
            for ue in self.base_env.get_all_ues()
        }
        self._prev_serving = {ue_id: None for ue_id in self._last_serving}
        return self._build_observation(), self._build_info()

    def step(self, action):
        proto_offsets = self._normalize_action(action)
        self._apply_proto_offsets(proto_offsets)

        handover_stats = self._execute_a3_handovers()

        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        # Build a neutral no-op action appropriate for the base env's action space.
        base_action_space = self.base_env.action_space
        if hasattr(base_action_space, "sample"):
            try:
                _noop = np.zeros(base_action_space.shape, dtype=base_action_space.dtype)
            except Exception:
                _noop = 0
        else:
            _noop = 0
        for _ in range(self.steps_per_action):
            _obs, _reward, terminated, truncated, info = self.base_env.step(_noop)
            reward += float(_reward)
            if terminated or truncated:
                break

        local_reward = self._compute_local_reward(proto_offsets, handover_stats)
        obs = self._build_observation()
        info = self._build_info(base_info=info, handover_stats=handover_stats)
        return obs, float(local_reward + reward), terminated, truncated, info

    def get_applied_offsets(self) -> Dict[Tuple[int, str], float]:
        return dict(self._offsets)

    def get_mobility_counters(self) -> Dict[Tuple[int, str], Dict[str, int]]:
        return {
            key: dict(value)
            for key, value in self._mobility_counters.items()
        }

    def _normalize_action(self, action) -> np.ndarray:
        arr = np.asarray(action, dtype=float).reshape(-1)
        expected = len(self.neighbor_ids) * len(self.slice_types)
        if arr.size == 0:
            arr = np.zeros(expected, dtype=float)
        if arr.size == 1 and expected > 1:
            arr = np.repeat(arr, expected)
        if arr.size != expected:
            raise ValueError(f"Expected action size {expected}, got {arr.size}")
        return np.clip(arr, -6.0, 6.0).astype(float)

    def _iter_keys(self):
        for neighbor_id in self.neighbor_ids:
            for slice_type in self.slice_types:
                yield (neighbor_id, slice_type)

    def _apply_proto_offsets(self, proto_offsets: np.ndarray):
        for key, proto in zip(self._iter_keys(), proto_offsets):
            self._offsets[key] = quantize_a3_offset(float(proto))

    def _execute_a3_handovers(self) -> Dict[str, int]:
        stats = {"attempts": 0, "successes": 0, "failures": 0, "ping_pongs": 0}
        serving_gnb = self.base_env._get_gnb_by_id(self.gnb_id)
        if serving_gnb is None:
            return stats

        for ue in list(self.base_env.get_all_ues()):
            if ue.serving_gnb is None or int(ue.serving_gnb) != self.gnb_id or not ue.connected:
                continue

            slice_type = normalize_slice_type(getattr(ue, "slice_type", "eMBB"))
            if slice_type not in self.slice_types:
                continue

            current_rx = self._rx_power(serving_gnb, ue)
            best_candidate = None
            best_margin = -np.inf

            for neighbor_id in self.neighbor_ids:
                target_gnb = self.base_env._get_gnb_by_id(neighbor_id)
                if target_gnb is None:
                    continue

                offset = self._offsets[(neighbor_id, slice_type)]
                target_rx = self._rx_power(target_gnb, ue)
                # A3 condition: RSRP_target > RSRP_serving + offset
                # positive offset discourages HO (harder to trigger),
                # negative offset encourages HO (easier to trigger).
                margin = target_rx - current_rx - offset
                if margin > best_margin:
                    best_margin = margin
                    best_candidate = target_gnb

            if best_candidate is None or best_margin < self.handover_margin_db:
                self._ttt_counters[int(ue.id)].clear()
                continue

            target_id = int(best_candidate.id)
            key = (target_id, slice_type)
            ue_counters = self._ttt_counters[int(ue.id)]
            ue_counters[key] = ue_counters.get(key, 0) + 1

            if ue_counters[key] < self.ttt:
                continue

            stats["attempts"] += 1
            self._mobility_counters[key]["attempts"] += 1

            if not self._handover_ue(ue, serving_gnb, best_candidate, slice_type):
                stats["failures"] += 1
                self._mobility_counters[key]["failures"] += 1
                ue_counters.clear()
                continue

            stats["successes"] += 1
            self._mobility_counters[key]["successes"] += 1
            if self._is_ping_pong(ue):
                stats["ping_pongs"] += 1
                self._mobility_counters[key]["ping_pongs"] += 1
            ue_counters.clear()

        return stats

    def _handover_ue(self, ue, old_gnb, new_gnb, slice_type: str) -> bool:
        if new_gnb is None or old_gnb is None or int(new_gnb.id) == int(old_gnb.id):
            return False
        if not self.base_env._is_in_coverage(new_gnb, ue):
            return False

        old_id = int(old_gnb.id)
        new_id = int(new_gnb.id)
        ue_id = int(ue.id)

        old_gnb.detach_ue(ue_id)
        attached = new_gnb.attach_ue(ue)
        if not attached:
            old_gnb.attach_ue(ue)
            return False

        self._prev_serving[ue_id] = self._last_serving.get(ue_id)
        self._last_serving[ue_id] = old_id
        ue.serving_gnb = new_id
        ue.connected = True
        ue.target_gnb = new_id
        ue.ho_pending = False
        ue.ho_candidate = None
        ue.ho_counter = 0
        self.base_env.handover_events.append({
            "step": int(self.base_env._step_count),
            "ue_id": ue_id,
            "slice_type": slice_type,
            "from_gnb": old_id,
            "to_gnb": new_id,
            "controller": f"LocalA3OffsetEnv[{self.gnb_id}]",
        })
        self.base_env._invalidate_metric_caches()
        return True

    def _rx_power(self, gnb, ue) -> float:
        metrics = self.base_env._compute_link_metrics(gnb, ue)
        return float(metrics["rx_power_dbm"])

    def _is_ping_pong(self, ue) -> bool:
        ue_id = int(ue.id)
        prev_serving = self._prev_serving.get(ue_id)
        # Ping-pong: UE just returned to the cell it was at two handovers ago.
        # After _handover_ue, ue.serving_gnb == new_id and _last_serving == old_id,
        # so we only need to check whether the new cell matches two steps back.
        return prev_serving is not None and ue.serving_gnb == prev_serving

    def _slice_counts(self) -> Dict[Tuple[int, str], int]:
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
                counts[(int(ue.serving_gnb), slice_type)] = counts.get((int(ue.serving_gnb), slice_type), 0) + 1
        return counts

    def _slice_sla_flags_by_gnb(self) -> Dict[Tuple[int, str], float]:
        """Return SLA violation intensity for every gNB-slice pair.

        This keeps the local observation load-free while allowing the local
        controller to know whether the target neighbor is currently risky for
        that slice. Values are clipped to [0, 1].
        """
        flags = {
            (int(gnb.id), slice_type): 0.0
            for gnb in self.base_env.gnbs
            for slice_type in self.slice_types
        }
        samples = {key: [] for key in flags}

        for ue in self.base_env.get_all_ues():
            if ue.serving_gnb is None or not ue.connected:
                continue
            gnb_id = int(ue.serving_gnb)
            slice_type = normalize_slice_type(getattr(ue, "slice_type", "eMBB"))
            key = (gnb_id, slice_type)
            if key in samples:
                samples[key].append(ue)

        for (gnb_id, slice_type), ues in samples.items():
            if not ues:
                continue
            if slice_type == "eMBB":
                offered_rates = [
                    float(getattr(getattr(ue, "traffic_source", None), "bit_rate", 1_000_000.0))
                    for ue in ues
                ]
                mean_target_bps = 0.80 * max(float(np.mean(offered_rates)), 1.0)
                mean_th = float(np.mean([float(ue.th) for ue in ues]))
                mean_queue_bits = float(np.mean([float(getattr(ue, "queue", 0.0)) for ue in ues]))
                queue_budget_bits = 0.10 * max(float(np.mean(offered_rates)), 1.0)
                throughput_gap = max(0.0, (mean_target_bps - mean_th) / mean_target_bps)
                queue_overage = max(0.0, (mean_queue_bits - queue_budget_bits) / max(queue_budget_bits, 1.0))
                flags[(gnb_id, slice_type)] = float(np.clip(max(throughput_gap, queue_overage), 0.0, 1.0))
            elif slice_type == "URLLC":
                max_delay_s = max(float(getattr(ue, "hol_delay_s", 0.0)) for ue in ues)
                flags[(gnb_id, slice_type)] = float(np.clip((max_delay_s - 0.010) / 0.040, 0.0, 1.0))
            elif slice_type == "mMTC":
                arrived = sum(float(getattr(ue, "new_bits", 0.0)) for ue in ues)
                dropped = sum(float(getattr(ue, "dropped_bits_step", 0.0)) for ue in ues)
                drop_ratio = dropped / max(arrived, 1.0)
                flags[(gnb_id, slice_type)] = float(np.clip((drop_ratio - 0.01) / 0.09, 0.0, 1.0))
        return flags

    def _slice_sla_flags(self) -> Dict[str, float]:
        """Backward-compatible view: serving gNB SLA flags by slice."""
        flags_by_gnb = self._slice_sla_flags_by_gnb()
        return {
            slice_type: float(flags_by_gnb.get((self.gnb_id, slice_type), 0.0))
            for slice_type in self.slice_types
        }

    def _mobility_ratios(self, key: Tuple[int, str]) -> Tuple[float, float]:
        counters = self._mobility_counters[key]
        attempts = max(float(counters["attempts"]), 1.0)
        successes = max(float(counters["successes"]), 1.0)
        hf = float(counters["failures"]) / attempts
        pp = float(counters["ping_pongs"]) / successes
        return hf, pp

    def _bias_for(self, gnb_id: int, slice_type: str) -> float:
        """Return clipped global/upper bias b_{gnb,s}."""
        key = (int(gnb_id), normalize_slice_type(slice_type))
        return float(np.clip(self.global_bias.get(key, 0.0), -1.0, 1.0))

    def _bias(self, slice_type: str) -> float:
        """Backward-compatible serving-gNB bias b_{i,s}."""
        return self._bias_for(self.gnb_id, slice_type)

    def _pairwise_transfer(self, neighbor_id: int, slice_type: str) -> Tuple[float, float, float]:
        """Return (b_i,s, b_j,s, tau_i,j,s).

        tau_i,j,s = 0.5 * (b_i,s - b_j,s).
        tau < 0 encourages i -> j handover.
        tau > 0 discourages i -> j handover.
        """
        b_i = self._bias_for(self.gnb_id, slice_type)
        b_j = self._bias_for(neighbor_id, slice_type)
        tau = float(np.clip(0.5 * (b_i - b_j), -1.0, 1.0))
        return b_i, b_j, tau

    def _desired_offset(self, neighbor_id: int, slice_type: str) -> float:
        """Pairwise desired offset for direction self.gnb_id -> neighbor_id.

        Uses the corrected formulation:
            desired = 6*tau + neighbor pressure + target SLA/mobility safety terms
        where tau = (b_i,s - b_j,s)/2.
        """
        counts = self._slice_counts()
        sla_flags_by_gnb = self._slice_sla_flags_by_gnb()
        _, _, tau = self._pairwise_transfer(neighbor_id, slice_type)

        neighbor_count = counts.get((int(neighbor_id), slice_type), 0)
        kappa_j = float(neighbor_count) / max(float(self.k_ref[slice_type]), 1e-9)
        hf, pp = self._mobility_ratios((int(neighbor_id), slice_type))
        v_j = float(sla_flags_by_gnb.get((int(neighbor_id), slice_type), 0.0))

        desired = (
            6.0 * tau
            + self.alpha_k * (kappa_j - self.k_target[slice_type])
            + self.alpha_hf * hf
            + self.alpha_pp * pp
            + self.alpha_sla_target * v_j
        )
        return float(np.clip(desired, -6.0, 6.0))

    def _build_observation(self) -> np.ndarray:
        counts = self._slice_counts()
        sla_flags_by_gnb = self._slice_sla_flags_by_gnb()
        obs = []

        for neighbor_id, slice_type in self._iter_keys():
            local_count = counts.get((self.gnb_id, slice_type), 0)
            neighbor_count = counts.get((neighbor_id, slice_type), 0)
            offset = self._offsets[(neighbor_id, slice_type)]
            hf, pp = self._mobility_ratios((neighbor_id, slice_type))
            b_i, b_j, tau = self._pairwise_transfer(neighbor_id, slice_type)
            v_i = float(sla_flags_by_gnb.get((self.gnb_id, slice_type), 0.0))
            v_j = float(sla_flags_by_gnb.get((neighbor_id, slice_type), 0.0))

            if self.normalize_observation:
                local_value = local_count / max(self.k_ref[slice_type], 1e-9)
                neighbor_value = neighbor_count / max(self.k_ref[slice_type], 1e-9)
                offset_value = offset / 6.0
            else:
                local_value = float(local_count)
                neighbor_value = float(neighbor_count)
                offset_value = offset

            obs.extend([
                float(b_i),
                float(b_j),
                float(tau),
                float(local_value),
                float(neighbor_value),
                float(v_i),
                float(v_j),
                float(offset_value),
                float(hf),
                float(pp),
            ])

        arr = np.asarray(obs, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

    def _compute_local_reward(self, proto_offsets: np.ndarray, handover_stats: Dict[str, int]) -> float:
        sla_flags = self._slice_sla_flags()
        reward = 0.0
        breakdown = {
            "tracking_penalty": 0.0,
            "sla_penalty": 0.0,
            "mobility_penalty": 0.0,
            "smoothness_penalty": 0.0,
        }

        for (neighbor_id, slice_type), proto in zip(self._iter_keys(), proto_offsets):
            desired = self._desired_offset(neighbor_id, slice_type)
            tracking = ((float(proto) - desired) / 12.0) ** 2
            previous = self._prev_proto_offsets[(neighbor_id, slice_type)]
            smoothness = ((float(proto) - previous) / 12.0) ** 2
            hf, pp = self._mobility_ratios((neighbor_id, slice_type))

            breakdown["tracking_penalty"] -= self.lambda_track * tracking
            breakdown["smoothness_penalty"] -= self.lambda_delta * smoothness
            breakdown["mobility_penalty"] -= self.w_hf * hf + self.w_pp * pp
            self._prev_proto_offsets[(neighbor_id, slice_type)] = float(proto)

        for slice_type, violation in sla_flags.items():
            breakdown["sla_penalty"] -= self.w_sla * float(violation)

        reward = sum(breakdown.values())
        self._last_reward_breakdown = {
            **breakdown,
            "handover_attempts": int(handover_stats.get("attempts", 0)),
            "handover_successes": int(handover_stats.get("successes", 0)),
            "handover_failures": int(handover_stats.get("failures", 0)),
            "handover_ping_pongs": int(handover_stats.get("ping_pongs", 0)),
            "total": float(reward),
        }
        return float(reward)

    def _pairwise_debug(self) -> Dict[str, Dict[Tuple[int, str], float]]:
        serving_bias = {}
        target_bias = {}
        pairwise_tau = {}
        desired_offsets = {}
        for neighbor_id, slice_type in self._iter_keys():
            key = (neighbor_id, slice_type)
            b_i, b_j, tau = self._pairwise_transfer(neighbor_id, slice_type)
            serving_bias[key] = float(b_i)
            target_bias[key] = float(b_j)
            pairwise_tau[key] = float(tau)
            desired_offsets[key] = float(self._desired_offset(neighbor_id, slice_type))
        return {
            "serving_bias": serving_bias,
            "target_bias": target_bias,
            "pairwise_tau": pairwise_tau,
            "desired_offsets": desired_offsets,
        }

    def _build_info(self, base_info=None, handover_stats=None):
        pairwise = self._pairwise_debug()
        return {
            "gnb_id": self.gnb_id,
            "neighbor_ids": list(self.neighbor_ids),
            "slice_types": list(self.slice_types),
            "applied_offsets": self.get_applied_offsets(),
            "mobility_counters": self.get_mobility_counters(),
            "handover_stats": dict(handover_stats or {}),
            "reward_breakdown": dict(self._last_reward_breakdown),
            "serving_bias": pairwise["serving_bias"],
            "target_bias": pairwise["target_bias"],
            "pairwise_tau": pairwise["pairwise_tau"],
            "desired_offsets": pairwise["desired_offsets"],
            "base_info": dict(base_info or {}),
        }
