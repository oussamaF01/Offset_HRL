#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from two_neighbor_offset_heuristic import coordinated_neighbor_offsets


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

    observation block for each neighbor j and slice k:
        [B_i,j,k, B_j,i,k, K_i,k, K_j,k, v_i,k, v_j,k,
         prev_offset_i,j,s, HF_i,j,s, PP_i,j,s]

    where B_i,j,k is the upper agent's directional source-target-slice bias.
    Negative B_i,j,k encourages i -> j handover; positive B_i,j,k retains
    traffic at i. B_j,i,k is included only as reverse-direction context.

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
        global_bias: Optional[Dict[Tuple[int, int, str], float]] = None,
        k_ref: Optional[Dict[str, float]] = None,
        k_target: Optional[Dict[str, float]] = None,
        alpha_k: float = 2.0,
        lambda_delta: float = 0.05,
        w_hf: float = 1.0,
        w_pp: float = 0.5,
        w_sla: float = 1.0,
        w_balance: float = 3.0,
        w_bias_align: float = 0.3,
        alpha_hf: float = 2.0,
        alpha_pp: float = 1.0,
        alpha_sla_target: float = 2.0,
        ttt: int = 1,
        handover_margin_db: float = 0.0,
        steps_per_action: int = 1,
        normalize_observation: bool = True,
        load_observation_provider: Optional[Callable[[], Dict[Tuple[int, str], float]]] = None,
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
        self.lambda_delta = float(lambda_delta)
        self.w_hf = float(w_hf)
        self.w_pp = float(w_pp)
        self.w_sla = float(w_sla)
        self.w_balance = float(w_balance)
        self.w_bias_align = float(w_bias_align)
        self.alpha_hf = float(alpha_hf)
        self.alpha_pp = float(alpha_pp)
        self.alpha_sla_target = float(alpha_sla_target)
        self.ttt = max(1, int(ttt))
        self.handover_margin_db = float(handover_margin_db)
        self.steps_per_action = max(1, int(steps_per_action))
        self.normalize_observation = bool(normalize_observation)
        self.load_observation_provider = load_observation_provider

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
        n_gnbs = len([g for g in self.base_env.gnbs])
        self.action_space = gym.spaces.Box(
            low=-6.0,
            high=6.0,
            shape=(n_blocks,),
            dtype=np.float32,
        )
        # Per neighbor-slice block (9 values each):
        #   [B_{i,j,k}, B_{j,i,k}, K_i, K_j, v_i, v_j, prev_offset, HF, PP]
        # Global context suffix (2 values per gNB per slice):
        #   [load_g, sla_g]  for every gNB g and every slice — coordination signal
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_blocks * 9 + n_gnbs * len(self.slice_types) * 2,),
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

    def set_global_bias(self, bias: Dict[Tuple[int, int, str], float]):
        self.global_bias = {
            (int(src), int(tgt), normalize_slice_type(st)): float(v)
            for (src, tgt, st), v in dict(bias).items()
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
        if hasattr(self.base_env, "get_slice_sla_severity"):
            severity = self.base_env.get_slice_sla_severity()
            return {
                key: float(np.clip(severity.get(key, 0.0), 0.0, 1.0))
                for key in flags
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

    def _bias_for(self, source_id: int, target_id: int, slice_type: str) -> float:
        """Return clipped directional bias b_{source→target,s}."""
        key = (int(source_id), int(target_id), normalize_slice_type(slice_type))
        return float(np.clip(self.global_bias.get(key, 0.0), -1.0, 1.0))

    def _directional_bias_pair(self, neighbor_id: int, slice_type: str) -> Tuple[float, float]:
        """Return (B_{i,j,k}, B_{j,i,k}) for this local-neighbor-slice pair.

        B_{i,j,k} is the active control signal for offset i→j on slice k.
        B_{j,i,k} is reverse-direction context; it does not weaken B_{i,j,k}.
        """
        b_ij = self._bias_for(self.gnb_id, neighbor_id, slice_type)
        b_ji = self._bias_for(neighbor_id, self.gnb_id, slice_type)
        return b_ij, b_ji

    def _best_radio_margin_db(self, neighbor_id: int, slice_type: str) -> float:
        serving_gnb = self.base_env._get_gnb_by_id(self.gnb_id)
        target_gnb = self.base_env._get_gnb_by_id(neighbor_id)
        if serving_gnb is None or target_gnb is None:
            return -np.inf

        margins = []
        wanted_slice = normalize_slice_type(slice_type)
        for ue in self.base_env.get_all_ues():
            if (
                ue.serving_gnb is None
                or int(ue.serving_gnb) != self.gnb_id
                or not ue.connected
                or normalize_slice_type(getattr(ue, "slice_type", "eMBB")) != wanted_slice
            ):
                continue
            if not self.base_env._is_in_coverage(target_gnb, ue):
                continue
            margins.append(self._rx_power(target_gnb, ue) - self._rx_power(serving_gnb, ue))
        return float(max(margins)) if margins else -np.inf

    def _coordinated_desired_offsets(self) -> Dict[Tuple[int, str], float]:
        loads = {
            (int(gnb.id), slice_type): float(
                self.base_env.estimate_slice_load(int(gnb.id), slice_type)
            )
            for gnb in self.base_env.gnbs
            for slice_type in self.slice_types
        }
        sla_flags_by_gnb = self._slice_sla_flags_by_gnb()
        hf = {}
        pp = {}
        radio = {}
        for neighbor_id, slice_type in self._iter_keys():
            key = (int(neighbor_id), slice_type)
            hf[key], pp[key] = self._mobility_ratios(key)
            radio[key] = self._best_radio_margin_db(int(neighbor_id), slice_type)

        return coordinated_neighbor_offsets(
            source_id=self.gnb_id,
            neighbor_ids=self.neighbor_ids,
            slice_types=self.slice_types,
            directional_bias=self.global_bias,
            useful_load=loads,
            sla_severity=sla_flags_by_gnb,
            handover_failure_ratio=hf,
            pingpong_ratio=pp,
            best_radio_margin_db=radio,
        )

    def _desired_offset(self, neighbor_id: int, slice_type: str) -> float:
        """Coordinated desired offset for direction self.gnb_id -> neighbor_id."""
        key = (int(neighbor_id), normalize_slice_type(slice_type))
        return float(self._coordinated_desired_offsets().get(key, 0.0))

    def _build_observation(self) -> np.ndarray:
        counts = self._slice_counts()
        sla_flags_by_gnb = self._slice_sla_flags_by_gnb()
        obs = []

        for neighbor_id, slice_type in self._iter_keys():
            local_count = counts.get((self.gnb_id, slice_type), 0)
            neighbor_count = counts.get((neighbor_id, slice_type), 0)
            offset = self._offsets[(neighbor_id, slice_type)]
            hf, pp = self._mobility_ratios((neighbor_id, slice_type))
            b_ijk, b_jik = self._directional_bias_pair(neighbor_id, slice_type)
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
                float(b_ijk),      # upper tensor B[i,j,k]
                float(b_jik),      # reverse-direction context B[j,i,k]
                float(local_value),
                float(neighbor_value),
                float(v_i),
                float(v_j),
                float(offset_value),
                float(hf),
                float(pp),
            ])

        # Global context: load and SLA severity for every gNB × slice.
        # Gives each agent a network-wide view so it can coordinate with peers.
        # Use window-average useful PRBs (demand-proportional) to match the upper agent.
        if self.load_observation_provider is not None:
            window_loads = dict(self.load_observation_provider())
        else:
            window_loads = self.base_env.get_window_average_slice_loads()
        for gnb in self.base_env.gnbs:
            gid = int(gnb.id)
            for slice_type in self.slice_types:
                load_g = float(window_loads.get((gid, slice_type), 0.0))
                sla_g = float(sla_flags_by_gnb.get((gid, slice_type), 0.0))
                obs.extend([float(load_g), float(sla_g)])

        arr = np.asarray(obs, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

    def _compute_local_reward(self, proto_offsets: np.ndarray, handover_stats: Dict[str, int]) -> float:
        # 1. Global SLA penalty — averaged over all (gNB, slice) pairs so the
        #    scale is independent of topology size. Result is in [-w_sla, 0].
        sla_flags_by_gnb = self._slice_sla_flags_by_gnb()
        sla_values = list(sla_flags_by_gnb.values())
        sla_mean = float(np.mean(sla_values)) if sla_values else 0.0
        sla_penalty = -self.w_sla * sla_mean

        # 2. Network load-balance penalty — per-slice variance across ALL gNBs.
        #    Uses instantaneous allocated loads (estimate_slice_load) so the signal
        #    is available from the very first step, not only after window warm-up.
        gnb_ids = [int(gnb.id) for gnb in self.base_env.gnbs]
        balance_penalty = 0.0
        for slice_type in self.slice_types:
            loads = [self.base_env.estimate_slice_load(g, slice_type) for g in gnb_ids]
            balance_penalty -= self.w_balance * float(np.var(loads))

        # 3. Step-level handover quality (not cumulative episode counters).
        #    Uses the fresh stats from this step's _execute_a3_handovers() call.
        step_attempts = int(handover_stats.get("attempts", 0))
        step_failures = int(handover_stats.get("failures", 0))
        step_successes = int(handover_stats.get("successes", 0))
        step_pingpongs = int(handover_stats.get("ping_pongs", 0))
        hf_rate = step_failures / max(step_attempts, 1)
        pp_rate = step_pingpongs / max(step_successes, 1)
        mobility_penalty = -(self.w_hf * hf_rate + self.w_pp * pp_rate)

        # 4. Smoothness — penalise large per-step offset jumps (action jitter).
        smoothness_penalty = 0.0
        for (neighbor_id, slice_type), proto in zip(self._iter_keys(), proto_offsets):
            previous = self._prev_proto_offsets[(neighbor_id, slice_type)]
            delta = (float(proto) - previous) / 12.0
            smoothness_penalty -= self.lambda_delta * delta ** 2
            self._prev_proto_offsets[(neighbor_id, slice_type)] = float(proto)

        # 5. Bias alignment — soft penalty when offset direction contradicts upper bias.
        #    Only active when |bias| > 0.1 (ignore near-neutral signals).
        #    Penalty = w_bias_align * max(0, -B * offset/6) per direction,
        #    averaged across directions so the scale is independent of topology size.
        bias_align_penalty = 0.0
        n_dirs = 0
        for neighbor_id, slice_type in self._iter_keys():
            b = self._bias_for(self.gnb_id, neighbor_id, slice_type)
            if abs(b) < 0.1:
                continue
            applied = self._offsets[(neighbor_id, slice_type)]
            agreement = b * (applied / 6.0)  # positive = agrees with bias, negative = contradicts
            bias_align_penalty -= self.w_bias_align * max(0.0, -agreement)
            n_dirs += 1
        if n_dirs > 1:
            bias_align_penalty /= n_dirs

        breakdown = {
            "sla_penalty": sla_penalty,
            "balance_penalty": balance_penalty,
            "mobility_penalty": mobility_penalty,
            "smoothness_penalty": smoothness_penalty,
            "bias_align_penalty": bias_align_penalty,
        }
        reward = sum(breakdown.values())
        self._last_reward_breakdown = {
            **breakdown,
            "handover_attempts": step_attempts,
            "handover_successes": step_successes,
            "handover_failures": step_failures,
            "handover_ping_pongs": step_pingpongs,
            "total": float(reward),
        }
        return float(reward)

    def _pairwise_debug(self) -> Dict[str, Dict[Tuple[int, str], float]]:
        serving_bias = {}
        target_bias = {}
        reverse_bias = {}
        desired_offsets = {}
        for neighbor_id, slice_type in self._iter_keys():
            key = (neighbor_id, slice_type)
            b_ijk, b_jik = self._directional_bias_pair(neighbor_id, slice_type)
            serving_bias[key] = float(b_ijk)
            target_bias[key] = float(b_jik)
            reverse_bias[key] = float(b_jik)
            desired_offsets[key] = float(self._desired_offset(neighbor_id, slice_type))
        return {
            "serving_bias": serving_bias,
            "target_bias": target_bias,
            "reverse_bias": reverse_bias,
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
            "reverse_bias": pairwise["reverse_bias"],
            "desired_offsets": pairwise["desired_offsets"],
            "base_info": dict(base_info or {}),
        }
