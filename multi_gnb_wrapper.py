#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import hashlib
import math
from pathlib import Path

import gymnasium as gym
import numpy as np

from slice_ran import UE, CBR
from safe_admission_controller import SafeAdmissionController
from traffic_generators import CbrSource, FixedPacketCbrSource


DEFAULT_UE_TRAFFIC_PROFILES = {
    "eMBB": {
        "traffic_model": "fixed_packet_cbr",
        "packet_size_bits": 3000.0,
        "bit_rate": 200_000.0,
    },
    "URLLC": {
        "traffic_model": "fixed_packet_cbr",
        "packet_size_bits": 800.0,
        "bit_rate": 80_000.0,
    },
    "mMTC": {
        "traffic_model": "fixed_packet_cbr",
        "packet_size_bits": 400.0,
        "bit_rate": 20_000.0,
    },
}

SLICE_TYPE_ORDER = ("eMBB", "URLLC", "mMTC")
DEFAULT_SLICE_PRB_BUDGETS = {}


class MultiGNBWrapper(gym.Env):
    """
    Multi-gNB world wrapper for mobility, radio, traffic, and diagnostics.

    The wrapper owns environment dynamics: mobility, attachment, radio/SINR,
    traffic queues, service, observations, and diagnostics. It no longer
    exposes the old direct RL interface that selected target gNBs for
    individual UEs; future HRL control should act through slice-aware A3
    offsets instead.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        gnb_list: List,
        handover_hysteresis: float = 0.05,
        handover_ttt: int = 3,
        verbose: bool = False,
        step_dt: float = 1e-3,
        mobility_dt: Optional[float] = None,
        radio_substeps: int = 1,
        pf_averaging_window_s: float = 0.25,
        max_episode_steps: int = 100,
        disconnect_sinr_db: float = -100.0,
        degradation_zones=None,
        use_sumo_mobility: bool = False,
        sumo_config_path: str = "scenario/mobility/sim.sumocfg",
        sumo_binary: str = "sumo",
        sumo_port: int = 8813,
        sumo_auto_add_ues: bool = True,
        sumo_vehicle_slice_type: str = "eMBB",
        sumo_person_slice_type: str = "URLLC",
        sumo_vehicle_slice_mix: Optional[Dict[str, float]] = None,
        sumo_person_slice_mix: Optional[Dict[str, float]] = None,
        min_sumo_bootstrap_ues: int = 1,
        ue_traffic_profiles: Optional[Dict] = None,
        default_traffic_model: str = "fixed_packet_cbr",
        slice_prb_budgets: Optional[Dict[str, int]] = None,
        max_prbs_per_ue: Optional[int] = 20,
        a3_hysteresis_db: float = 1.0,
        a3_history_window_s: float = 20.0,
        a3_pingpong_threshold_s: float = 5.0,
        a3_handover_cooldown_s: float = 5.0,
        a3_min_residence_s: float = 15.0,
        a3_pingpong_guard_s: float = 30.0,
        a3_emergency_sinr_db: float = -5.0,
        max_handovers_per_step: int = 1,
        max_handovers_per_ue_episode: int = 2,
        max_handovers_per_episode: int = 20,
        safe_admission_enabled: bool = False,
        safe_admission_load_limits: Optional[Dict[str, float]] = None,
        safe_admission_bias_deadband: float = 0.05,
        safe_admission_max_target_sla_severity: float = 0.50,
        neutralize_offsets_when_quota_exhausted: bool = False,
        embb_min_delivery_ratio: float = 0.80,
        urllc_max_failure_ratio: float = 0.01,
        mmtc_max_failure_ratio: float = 0.01,
    ):
        super().__init__()

        if not gnb_list:
            raise ValueError("gnb_list must contain at least one gNB")

        self.gnbs = list(gnb_list)
        self.n_gnbs = len(self.gnbs)
        self._gnb_by_id = {int(gnb.id): gnb for gnb in self.gnbs}
        self._gnb_index_by_id = {int(gnb.id): idx for idx, gnb in enumerate(self.gnbs)}
        self._same_carrier_interferers = self._build_same_carrier_interferers()
        self.history = []

        # World bounds computed from gNB layout
        xs = [float(gnb.x) for gnb in self.gnbs]
        ys = [float(gnb.y) for gnb in self.gnbs]
        rs = [float(getattr(gnb, "coverage_radius", 0.0)) for gnb in self.gnbs]

        margin = 100.0
        self.x_min = min(x - r for x, r in zip(xs, rs)) - margin
        self.x_max = max(x + r for x, r in zip(xs, rs)) + margin
        self.y_min = min(y - r for y, r in zip(ys, rs)) - margin
        self.y_max = max(y + r for y, r in zip(ys, rs)) + margin

        self.handover_hysteresis = float(handover_hysteresis)
        self.handover_ttt = int(handover_ttt)
        self.verbose = bool(verbose)
        self.step_dt = float(step_dt)
        self.mobility_dt = float(self.step_dt if mobility_dt is None else mobility_dt)
        self.radio_substeps = max(1, int(radio_substeps))
        self.pf_averaging_window_s = max(
            float(pf_averaging_window_s), float(self.step_dt)
        )
        # How many substeps between scheduler re-runs.
        # stride=5 gives packets two scheduling rounds within the 10ms URLLC
        # deadline (stride=10 caused 0.15% SLA failures under queue backlog).
        self._radio_sched_stride = min(5, self.radio_substeps)
        self.max_episode_steps = int(max_episode_steps)
        self.disconnect_sinr_db = float(disconnect_sinr_db)
        self.degradation_zones = list(degradation_zones or [])
        self.use_sumo_mobility = bool(use_sumo_mobility)
        self.sumo_config_path = str(sumo_config_path)
        self.sumo_binary = str(sumo_binary)
        self.sumo_port = int(sumo_port)
        self.sumo_auto_add_ues = bool(sumo_auto_add_ues)
        self.sumo_vehicle_slice_type = str(sumo_vehicle_slice_type)
        self.sumo_person_slice_type = str(sumo_person_slice_type)
        self.sumo_vehicle_slice_mix = self._normalize_slice_mix(
            sumo_vehicle_slice_mix,
            fallback=self.sumo_vehicle_slice_type,
        )
        self.sumo_person_slice_mix = self._normalize_slice_mix(
            sumo_person_slice_mix,
            fallback=self.sumo_person_slice_type,
        )
        self.min_sumo_bootstrap_ues = max(0, int(min_sumo_bootstrap_ues))
        self.default_traffic_model = str(default_traffic_model)
        self.ue_traffic_profiles = self._merge_traffic_profiles(ue_traffic_profiles)
        self.slice_prb_budgets = self._normalize_slice_prb_budgets(slice_prb_budgets)
        self.max_prbs_per_ue = (
            None if max_prbs_per_ue is None else max(int(max_prbs_per_ue), 1)
        )
        self._apply_slice_prb_budgets()
        self.sumo_label = f"multignb_{id(self)}"
        self.sumo = None
        self._sumo_entity_to_ue_id: Dict[str, int] = {}
        self.use_mcs_scheduler = True
        self._pf_scheduler = None
        self._mcs_codeset = None
        self._mcs_codeset_urllc = None
        self._fading_samples = None
        self._fading_users: Dict[int, Dict[str, int]] = {}
        self._last_scheduler_mode = "mcs_pf_csv_fading"

        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=0.0,
            shape=(0,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng()
        self._next_ue_id = 0
        self._ues: Dict[int, UE] = {}
        self._step_count = 0

        self._last_serving_gnb: Dict[int, Optional[int]] = {}
        self._prev_serving_gnb: Dict[int, Optional[int]] = {}

        self._last_info: Dict = {}
        self.handover_events: List[Dict] = []
        self._a3_offsets: Dict[Tuple[int, int, str], float] = {}
        self._ho_attempts: Dict[Tuple[int, int, str], List[int]] = {}
        self._ho_successes: Dict[Tuple[int, int, str], List[int]] = {}
        self._ho_failures: Dict[Tuple[int, int, str], List[int]] = {}
        self._ho_pingpongs: Dict[Tuple[int, int, str], List[int]] = {}
        self._last_ho: Dict[int, Tuple[int, int]] = {}
        self._last_ho_source: Dict[int, int] = {}
        self._ue_episode_handovers: Dict[int, int] = {}
        self._episode_handover_count = 0
        self.a3_hysteresis_db = float(a3_hysteresis_db)
        self.a3_handover_cooldown_s = max(float(a3_handover_cooldown_s), 0.0)
        a3_tick_s = self.mobility_dt if self.mobility_dt > 0.0 else self.step_dt
        self.a3_history_window_s = max(float(a3_history_window_s), 0.0)
        self.a3_window_steps = int(
            math.ceil(self.a3_history_window_s / max(float(a3_tick_s), 1e-9))
        )
        self.a3_pingpong_threshold_s = max(float(a3_pingpong_threshold_s), 0.0)
        self.a3_pingpong_threshold_steps = int(
            math.ceil(self.a3_pingpong_threshold_s / max(float(a3_tick_s), 1e-9))
        )
        self.a3_handover_cooldown_steps = int(
            math.ceil(self.a3_handover_cooldown_s / max(float(a3_tick_s), 1e-9))
        )
        self.a3_min_residence_s = max(float(a3_min_residence_s), self.a3_handover_cooldown_s)
        self.a3_min_residence_steps = int(
            math.ceil(self.a3_min_residence_s / max(float(a3_tick_s), 1e-9))
        )
        self.a3_pingpong_guard_s = max(
            float(a3_pingpong_guard_s),
            self.a3_min_residence_s,
        )
        self.a3_pingpong_guard_steps = int(
            math.ceil(self.a3_pingpong_guard_s / max(float(a3_tick_s), 1e-9))
        )
        self.a3_emergency_sinr_db = float(a3_emergency_sinr_db)
        self.max_handovers_per_step = max(int(max_handovers_per_step), 1)
        self.max_handovers_per_ue_episode = max(int(max_handovers_per_ue_episode), 1)
        self.max_handovers_per_episode = max(int(max_handovers_per_episode), 1)
        self.safe_admission_enabled = bool(safe_admission_enabled)
        self.safe_admission_load_limits = {
            slice_type: 0.80 for slice_type in SLICE_TYPE_ORDER
        }
        for slice_type, value in dict(safe_admission_load_limits or {}).items():
            self.safe_admission_load_limits[self.normalize_slice_type(slice_type)] = float(value)
        self.neutralize_offsets_when_quota_exhausted = bool(
            neutralize_offsets_when_quota_exhausted
        )
        self.safe_admission_controller = SafeAdmissionController(
            bias_deadband=safe_admission_bias_deadband,
            max_target_sla_severity=safe_admission_max_target_sla_severity,
        )
        self.embb_min_delivery_ratio = float(np.clip(embb_min_delivery_ratio, 0.0, 1.0))
        self.urllc_max_failure_ratio = max(float(urllc_max_failure_ratio), 0.0)
        self.mmtc_max_failure_ratio = max(float(mmtc_max_failure_ratio), 0.0)
        self._sla_window_stats: Dict[Tuple[int, str], Dict[str, float]] = {}
        self.begin_sla_window()
        self.begin_radio_measurement_window()
        self._metric_cache_token = 0
        self._link_metrics_cache: Dict[Tuple[int, int], Dict[str, float]] = {}
        self._gnb_load_cache: Dict[int, float] = {}
        self._gnb_load_pressure_cache: Dict[int, float] = {}
        # Composite environments such as GlobalPPO3GNBEnv consume their own
        # diagnostics and otherwise discard this wrapper's large per-step info.
        self.collect_step_diagnostics = True

    @staticmethod
    def normalize_slice_type(slice_type: str) -> str:
        raw = str(slice_type or "eMBB")
        compact = raw.replace("_", "").replace("-", "").replace(" ", "").lower()
        for known in SLICE_TYPE_ORDER:
            if compact == known.lower():
                return known
            if known.lower() in compact:
                return known
        return raw

    def _l1_slice_type(self, l1) -> str:
        explicit = getattr(l1, "type", None)
        if explicit is not None:
            normalized = self.normalize_slice_type(explicit)
            if normalized in SLICE_TYPE_ORDER:
                return normalized
        return self.normalize_slice_type(type(l1).__name__)

    def _configured_slice_types(self) -> Tuple[str, ...]:
        configured = set()
        for gnb in self.gnbs:
            for l1 in getattr(gnb, "slices_l1", []):
                slice_type = self._l1_slice_type(l1)
                if slice_type in SLICE_TYPE_ORDER:
                    configured.add(slice_type)

        if not configured:
            return SLICE_TYPE_ORDER
        return tuple(slice_type for slice_type in SLICE_TYPE_ORDER if slice_type in configured)

    def _global_agent_keys(self) -> List[Tuple[int, str]]:
        slice_types = self._configured_slice_types()
        return [
            (int(gnb.id), slice_type)
            for gnb in self.gnbs
            for slice_type in slice_types
        ]

    def _matching_l1_slices(self, gnb, slice_type: str) -> List:
        wanted = self.normalize_slice_type(slice_type)
        return [
            l1
            for l1 in getattr(gnb, "slices_l1", [])
            if self._l1_slice_type(l1) == wanted
        ]

    def _normalize_slice_prb_budgets(self, budgets: Optional[Dict[str, int]]) -> Dict[str, int]:
        return {}

    def _slice_budget_for_gnb(self, gnb, slice_type: str) -> int:
        wanted = self.normalize_slice_type(slice_type)
        configured = [
            self._l1_slice_type(l1)
            for l1 in getattr(gnb, "slices_l1", [])
            if self._l1_slice_type(l1) in self.slice_prb_budgets
        ]
        budget_total = sum(self.slice_prb_budgets.get(s, 0) for s in configured)
        if budget_total <= 0:
            return 0
        gnb_prbs = max(int(getattr(gnb, "n_prbs", 0)), 0)
        return int(round(gnb_prbs * self.slice_prb_budgets.get(wanted, 0) / float(budget_total)))

    def _apply_slice_prb_budgets(self):
        if not self.slice_prb_budgets:
            return
        for gnb in self.gnbs:
            for l1 in getattr(gnb, "slices_l1", []):
                slice_type = self._l1_slice_type(l1)
                budget = self._slice_budget_for_gnb(gnb, slice_type)
                if budget > 0 and hasattr(l1, "set_prbs"):
                    l1.set_prbs(0, budget)

    def get_slice_prb_budget(self, gnb_id: int, slice_type: str) -> int:
        gnb = self._get_gnb_by_id(gnb_id)
        if gnb is None:
            return 0

        if self.slice_prb_budgets:
            configured_budget = self._slice_budget_for_gnb(gnb, slice_type)
            if configured_budget > 0:
                return configured_budget

        return int(max(getattr(gnb, "n_prbs", 0), 0))

    def get_slice_used_prbs(self, gnb_id: int, slice_type: str) -> int:
        gnb_id = int(gnb_id)
        wanted = self.normalize_slice_type(slice_type)
        used_prbs = 0

        for ue in self._ues.values():
            if not ue.connected or ue.serving_gnb is None:
                continue
            if int(ue.serving_gnb) != gnb_id:
                continue
            if self.normalize_slice_type(getattr(ue, "slice_type", "eMBB")) != wanted:
                continue
            used_prbs += int(max(getattr(ue, "prbs", 0), 0))

        return used_prbs

    def get_slice_ue_count(self, gnb_id: int, slice_type: str) -> int:
        gnb_id = int(gnb_id)
        wanted = self.normalize_slice_type(slice_type)
        count = 0

        for ue in self._ues.values():
            if not ue.connected or ue.serving_gnb is None:
                continue
            if int(ue.serving_gnb) != gnb_id:
                continue
            if self.normalize_slice_type(getattr(ue, "slice_type", "eMBB")) == wanted:
                count += 1

        return count

    def estimate_slice_load(self, gnb_id: int, slice_type: str) -> float:
        """
        Physical allocated-utilization contribution:
            U_{i,s} = useful_allocated_PRBs_{i,s} / PRB_total_i

        Every slice uses the same physical gNB-wide denominator, normally
        100 PRBs. The numerator comes from the radio scheduler's useful PRB
        allocation for connected UEs of the slice. Therefore:

            total_utilization_i = sum_s U_{i,s} <= 1

        Persistent ``upper_demand_prbs`` remains separate and is used only to
        estimate the extra capacity required by a candidate handover.
        """
        gnb = self._get_gnb_by_id(gnb_id)
        total_prbs = int(max(getattr(gnb, "n_prbs", 0), 0))
        if total_prbs <= 0:
            return 0.0
        useful_prbs = self.get_slice_used_prbs(gnb_id, slice_type)
        return self._clip01(useful_prbs / float(total_prbs))

    def _estimate_bits_per_prb_for_ue(self, ue: UE) -> float:
        codeset = self._mcs_codeset_for_slice(getattr(ue, "slice_type", "eMBB"))
        if codeset is not None:
            try:
                _mcs, bits_per_sym = codeset.mcs_rate_vs_error(
                    float(getattr(ue, "e_snr", self.disconnect_sinr_db)),
                    0.1,
                )
                return max(158.0 * float(bits_per_sym), 1.0)
            except Exception:
                pass

        sinr_db = float(getattr(ue, "e_sinr", self.disconnect_sinr_db))
        if not np.isfinite(sinr_db):
            sinr_db = self.disconnect_sinr_db
        sinr_linear = max(10.0 ** (sinr_db / 10.0), 1e-6)
        spectral_eff = min(max(math.log2(1.0 + sinr_linear), 0.0), 8.0)
        return max(180e3 * max(float(self.step_dt), 1e-9) * spectral_eff, 1.0)

    def _mcs_codeset_for_slice(self, slice_type: str):
        normalized = self.normalize_slice_type(slice_type)
        if normalized == "URLLC" and self._mcs_codeset_urllc is not None:
            return self._mcs_codeset_urllc
        return self._mcs_codeset

    def _estimate_queue_demand_prbs(self, ue: UE) -> int:
        queue_bits = max(float(getattr(ue, "queue", 0.0)), 0.0)
        if queue_bits <= 0.0:
            return 0
        bits_per_prb = self._estimate_bits_per_prb_for_ue(ue)
        return int(np.ceil(queue_bits / max(bits_per_prb, 1.0)))

    def get_slice_loads(self) -> Dict[Tuple[int, str], float]:
        return {
            key: self.estimate_slice_load(*key)
            for key in self._global_agent_keys()
        }

    @staticmethod
    def _empty_sla_window_counter() -> Dict[str, float]:
        return {
            "offered_bits": 0.0,
            "delivered_bits": 0.0,
            "generated_packets": 0.0,
            "completed_packets": 0.0,
            "completed_delay_sum_s": 0.0,
            "failed_packets": 0.0,
            "dropped_packets": 0.0,
        }

    def begin_sla_window(self) -> None:
        self._sla_window_stats = {
            key: self._empty_sla_window_counter()
            for key in self._global_agent_keys()
        }

    def begin_radio_measurement_window(self) -> None:
        """Reset physical PRB/service accounting for one upper window."""
        self._radio_window_samples = 0
        self._radio_window_slice_useful_prbs = {
            (int(gnb.id), slice_type): 0.0
            for gnb in self.gnbs
            for slice_type in self._configured_slice_types()
        }
        self._radio_window_ue_stats: Dict[int, Dict[str, float]] = {}

    def _accumulate_radio_measurement_sample(self) -> None:
        self._radio_window_samples += 1
        for ue in self._ues.values():
            if not ue.connected or ue.serving_gnb is None:
                continue
            ue_id = int(ue.id)
            serving_id = int(ue.serving_gnb)
            slice_type = self.normalize_slice_type(
                getattr(ue, "slice_type", "eMBB")
            )
            allocated_prbs = max(float(getattr(ue, "prbs", 0.0)), 0.0)
            useful_prbs = max(float(getattr(ue, "useful_prbs", 0.0)), 0.0)
            scheduled_bits = max(
                float(getattr(ue, "scheduled_bits", 0.0)), 0.0
            )
            served_bits = max(float(getattr(ue, "bits", 0.0)), 0.0)
            key = (serving_id, slice_type)
            self._radio_window_slice_useful_prbs[key] = (
                self._radio_window_slice_useful_prbs.get(key, 0.0)
                + allocated_prbs
            )
            stats = self._radio_window_ue_stats.setdefault(
                ue_id,
                {
                    "useful_prbs": 0.0,
                    "scheduled_bits": 0.0,
                    "served_bits": 0.0,
                    "samples": 0.0,
                },
            )
            stats["useful_prbs"] += useful_prbs
            stats["scheduled_bits"] += scheduled_bits
            stats["served_bits"] += served_bits
            stats["samples"] += 1.0

    def get_window_average_slice_loads(self) -> Dict[Tuple[int, str], float]:
        samples = max(int(self._radio_window_samples), 1)
        loads = {}
        for gnb in self.gnbs:
            gnb_id = int(gnb.id)
            capacity = max(float(getattr(gnb, "n_prbs", 0)), 1.0)
            for slice_type in self._configured_slice_types():
                useful_sum = self._radio_window_slice_useful_prbs.get(
                    (gnb_id, slice_type), 0.0
                )
                loads[(gnb_id, slice_type)] = self._clip01(
                    useful_sum / (samples * capacity)
                )
        return loads

    def get_ue_radio_window_stats(self) -> Dict[int, Dict[str, float]]:
        return {
            int(ue_id): dict(values)
            for ue_id, values in self._radio_window_ue_stats.items()
        }

    def _sla_window_metrics_for_key(self, key: Tuple[int, str]) -> Dict[str, float]:
        values = dict(self._sla_window_stats.get(key, self._empty_sla_window_counter()))
        slice_type = self.normalize_slice_type(key[1])
        active = values["offered_bits"] > 0.0 or values["generated_packets"] > 0.0
        severity = 0.0
        violation = 0.0
        measured_ratio = 1.0
        threshold = 0.0

        if slice_type == "eMBB":
            measured_ratio = (
                values["delivered_bits"] / values["offered_bits"]
                if values["offered_bits"] > 0.0
                else 1.0
            )
            threshold = self.embb_min_delivery_ratio
            severity = (
                max(0.0, threshold - measured_ratio) / max(threshold, 1e-9)
                if active else 0.0
            )
            violation = float(active and measured_ratio < threshold)
        elif slice_type in {"URLLC", "mMTC"}:
            measured_ratio = (
                values["failed_packets"] / values["generated_packets"]
                if values["generated_packets"] > 0.0
                else 0.0
            )
            threshold = (
                self.urllc_max_failure_ratio
                if slice_type == "URLLC"
                else self.mmtc_max_failure_ratio
            )
            severity = float(np.clip(measured_ratio, 0.0, 1.0)) if active else 0.0
            violation = float(active and measured_ratio > threshold)

        return {
            **values,
            "active": float(active),
            "measured_ratio": float(measured_ratio),
            "threshold": float(threshold),
            "severity": float(np.clip(severity, 0.0, 1.0)),
            "violation": float(violation),
        }

    def get_slice_sla_metrics(self) -> Dict[Tuple[int, str], Dict[str, float]]:
        return {
            key: self._sla_window_metrics_for_key(key)
            for key in self._global_agent_keys()
        }

    def get_slice_sla_severity(self) -> Dict[Tuple[int, str], float]:
        return {
            key: values["severity"]
            for key, values in self.get_slice_sla_metrics().items()
        }

    def get_slice_sla_flags(self) -> Dict[Tuple[int, str], float]:
        """Binary per-window SLA violation flags (compatibility API)."""
        return {
            key: values["violation"]
            for key, values in self.get_slice_sla_metrics().items()
        }

    def get_global_agent_observation(
        self,
        include_ue_counts: bool = True,
        include_service_metrics: bool = False,
    ) -> np.ndarray:
        """
        Flat upper/global PPO observation in deterministic order.

        Blocks are grouped by signal type over keys ordered as:
            self.gnbs order x (eMBB, URLLC, mMTC configured slices)

        Base values are:
            1. PRB-based allocated slice loads L_{i,s}
            2. SLA flags v_{i,s}
            3. optional connected UE counts K_{i,s}

        With include_service_metrics=True, the observation also includes:
            - demand_load from queued bits converted to required PRBs
            - queue_pressure against a short queue budget
            - served_deficit = 1 - served_ratio

        This preserves the information-asymmetric HRL split: the upper agent
        observes PRB slice load, while local TD3 agents keep using bias and
        local counters rather than direct PRB load.
        """
        keys = self._global_agent_keys()
        loads = self.get_slice_loads()
        sla_flags = self.get_slice_sla_flags()

        values = [loads.get(key, 0.0) for key in keys]
        values.extend(sla_flags.get(key, 0.0) for key in keys)

        if include_service_metrics:
            kpis = self._build_info(per_gnb_rewards=[0.0] * self.n_gnbs)["slice_kpis"]
            values.extend(float(min(kpis.get(key, {}).get("demand_load", 0.0), 1.0)) for key in keys)
            values.extend(float(min(kpis.get(key, {}).get("queue_pressure", 0.0), 1.0)) for key in keys)
            values.extend(
                float(self._clip01(1.0 - kpis.get(key, {}).get("served_ratio", 1.0)))
                for key in keys
            )

        if include_ue_counts:
            values.extend(float(self.get_slice_ue_count(*key)) for key in keys)

        return np.asarray(values, dtype=np.float32)

    def _build_same_carrier_interferers(self) -> Dict[int, List]:
        interferers = {}
        for serving in self.gnbs:
            peers = []
            for other in self.gnbs:
                if int(other.id) == int(serving.id):
                    continue
                try:
                    same_carrier = serving.uses_same_carrier(other)
                except Exception:
                    same_carrier = False
                if same_carrier:
                    peers.append(other)
            interferers[int(serving.id)] = peers
        return interferers

    def _invalidate_metric_caches(self):
        self._metric_cache_token += 1
        self._link_metrics_cache.clear()
        self._gnb_load_cache.clear()
        self._gnb_load_pressure_cache.clear()

    @staticmethod
    def _normalize_slice_mix(mix: Optional[Dict[str, float]], fallback: str) -> List[Tuple[str, float]]:
        if not mix:
            return [(str(fallback), 1.0)]

        weighted = []
        for slice_type, weight in dict(mix).items():
            weight = max(float(weight), 0.0)
            if weight > 0.0:
                weighted.append((str(slice_type), weight))

        if not weighted:
            return [(str(fallback), 1.0)]

        total = sum(weight for _, weight in weighted)
        cumulative = []
        running = 0.0
        for slice_type, weight in weighted:
            running += weight / total
            cumulative.append((slice_type, running))
        cumulative[-1] = (cumulative[-1][0], 1.0)
        return cumulative

    @staticmethod
    def _stable_unit_interval(value: str) -> float:
        digest = hashlib.sha256(str(value).encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return bucket / float(2**64 - 1)

    def _slice_type_for_sumo_entity(self, sumo_id: str, entity_type: str) -> str:
        mix = (
            self.sumo_vehicle_slice_mix
            if entity_type == "vehicle"
            else self.sumo_person_slice_mix
        )
        sample = self._stable_unit_interval(f"{entity_type}:{sumo_id}")
        for slice_type, cutoff in mix:
            if sample <= cutoff:
                return slice_type
        return mix[-1][0]

    def _environment_loss_db(self, ue) -> float:
        if not self.degradation_zones:
            return 0.0

        loss = 0.0
        ue_slice = str(getattr(ue, "slice_type", "") or "").upper()

        for zone in self.degradation_zones:
            zone_slices = zone.get("slice_types") or zone.get("slices")
            if zone_slices:
                allowed = {str(item).upper() for item in zone_slices}
                if ue_slice not in allowed:
                    continue

            dx = float(ue.x) - float(zone["x"])
            dy = float(ue.y) - float(zone["y"])
            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= float(zone["radius"]):
                loss += float(zone.get("loss_db", 0.0))

        return loss
    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def _apply_world_bounds(self, ue: UE):
        if ue.x < self.x_min:
            ue.x = self.x_min
            ue.vx = abs(ue.vx)
        elif ue.x > self.x_max:
            ue.x = self.x_max
            ue.vx = -abs(ue.vx)

        if ue.y < self.y_min:
            ue.y = self.y_min
            ue.vy = abs(ue.vy)
        elif ue.y > self.y_max:
            ue.y = self.y_max
            ue.vy = -abs(ue.vy)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.history = []
        self._step_count = 0
        self._last_info = {}
        self._sumo_entity_to_ue_id = {}
        self._fading_users = {}
        self.handover_events = []
        self._ho_attempts = {}
        self._ho_successes = {}
        self._ho_failures = {}
        self._ho_pingpongs = {}
        self._last_ho = {}
        self._last_ho_source = {}
        self._ue_episode_handovers = {}
        self._episode_handover_count = 0
        self.safe_admission_controller.reset_window()
        self._invalidate_metric_caches()

        for gnb in self.gnbs:
            if hasattr(gnb, "reset"):
                gnb.reset()
            if hasattr(gnb, "_a3_counters"):
                gnb._a3_counters.clear()

        for ue in self._ues.values():
            if hasattr(ue, "packet_queue"):
                ue.packet_queue.clear()
            ue.queue = 0
            ue.th = 0.0
            ue.bits = 0
            ue.scheduled_bits = 0
            ue.new_bits = 0
            ue.dropped_bits = 0
            ue.dropped_bits_step = 0
            ue.total_bits_arrived = 0
            ue.total_bits_dropped = 0
            ue.total_bits_served = 0
            ue.total_packets_generated = 0
            ue.total_packets_completed = 0
            ue.total_completed_packet_delay_s = 0.0
            ue.total_packets_dropped = 0
            ue.total_packets_deadline_missed = 0
            ue.total_packets_expired = 0
            ue.total_packets_failed_sla = 0
            ue.wait_time = 0
            if hasattr(ue, "set_global_step"):
                ue.set_global_step(0)
            if hasattr(ue, "hol_delay_s"):
                ue.hol_delay_s = 0.0
                ue.max_hol_delay_s = 0.0
                ue.mean_packet_delay_s = 0.0
            if hasattr(ue, "_delay_samples"):
                ue._delay_samples.clear()
            if hasattr(ue, "_next_packet_id"):
                ue._next_packet_id = 0
                ue._last_packet_arrival_step = 0
                ue._time_s = 0.0
                ue._expired_packet_ids.clear()
                ue._failed_sla_packet_ids.clear()
            ue.snr = 0
            ue.e_snr = 0
            ue.sinr = 0
            ue.e_sinr = 0
            ue.prbs = 0
            ue.useful_prbs = 0
            ue.wasted_prbs = 0
            ue.p = 0
            ue.effective_sinr_db = float("nan")
            ue.mcs_codeset_name = "default"
            ue.connected = True
            ue.target_gnb = None
            ue.ho_pending = False
            ue.ho_candidate = None
            ue.ho_counter = 0
            ue.serving_power_dbm = -100.0
            ue.interference_dbm = -100.0
            ue.noise_dbm = -100.0

            best = self._find_best_gnb_for_ue(ue)
            ue.serving_gnb = best.id if best is not None else None
            ue.connected = ue.serving_gnb is not None

            self._ues[ue.id] = ue

            if best is not None:
                best.attach_ue(ue)

            self._last_serving_gnb[ue.id] = ue.serving_gnb
            self._prev_serving_gnb[ue.id] = None

        if self.use_sumo_mobility:
            self._restart_sumo_mobility()
            self._sync_sumo_mobility(advance=False)
            self._ensure_minimum_sumo_ues()

        self._simulate_radio_and_service()
        self._invalidate_metric_caches()
        self.begin_sla_window()

        obs = self._empty_observation()
        if not np.all(np.isfinite(obs)):
            raise ValueError(f"Non-finite observation detected in reset: {obs}")

        info = self._build_info(per_gnb_rewards=[0.0] * self.n_gnbs)
        self._last_info = info
        return obs, info

    def set_a3_offset(self, serving_id: int, neighbor_id: int, slice_type: str, offset_db: float) -> None:
        """Set the slice-aware A3 offset that serving gNB applies toward a neighbor."""
        key = (int(serving_id), int(neighbor_id), self.normalize_slice_type(slice_type).upper())
        self._a3_offsets[key] = float(offset_db)

    def _reset_safe_admission_stats(self) -> None:
        self.safe_admission_controller.stats.clear()

    def begin_safe_admission_window(
        self,
        directional_bias,
        slice_types: Tuple[str, ...] = SLICE_TYPE_ORDER,
    ) -> Dict[Tuple[int, int, str], int]:
        """Reset and freeze directional release quotas for one upper window."""
        bias = np.asarray(directional_bias, dtype=float)
        normalized_slices = tuple(self.normalize_slice_type(s) for s in slice_types)
        neighbor_graph = {
            int(source.id): [
                int(target.id) for target in self.gnbs
                if int(target.id) != int(source.id)
            ]
            for source in self.gnbs
        }
        max_neighbors = max(map(len, neighbor_graph.values()), default=0)
        if bias.shape == (self.n_gnbs, len(normalized_slices)):
            bias = np.repeat(bias[:, None, :], max_neighbors, axis=1)
        if bias.shape != (self.n_gnbs, max_neighbors, len(normalized_slices)):
            raise ValueError(
                "directional_bias must have shape "
                f"{(self.n_gnbs, max_neighbors, len(normalized_slices))}, got {bias.shape}"
            )

        loads = self.get_slice_loads()
        ue_counts = {
            (int(gnb.id), slice_type): self.get_slice_ue_count(
                int(gnb.id), slice_type
            )
            for gnb in self.gnbs
            for slice_type in normalized_slices
        }
        balance_targets = {
            slice_type: float(np.mean([
                loads.get((int(gnb.id), slice_type), 0.0)
                for gnb in self.gnbs
            ]))
            for slice_type in normalized_slices
        }
        self.safe_admission_controller.begin_upper_window(
            directional_bias=bias,
            neighbor_graph=neighbor_graph,
            gnb_ids=[int(gnb.id) for gnb in self.gnbs],
            slice_types=normalized_slices,
            loads=loads,
            ue_counts=ue_counts,
            balance_targets=balance_targets,
            remaining_handover_budget=max(
                self.max_handovers_per_episode - self._episode_handover_count,
                0,
            ),
        )
        return {
            (source_id, target_id, slice_type.upper()):
            self.safe_admission_controller.direction_quota.get(
                (source_id, target_id, slice_type), 0
            )
            for source_id, targets in neighbor_graph.items()
            for target_id in targets
            for slice_type in normalized_slices
        }

    def get_safe_admission_state(self) -> Dict:
        state = self.safe_admission_controller.get_state()
        direction_capacities = {}
        direction_accepted = {}
        for source in self.gnbs:
            for target in self.gnbs:
                if int(source.id) == int(target.id):
                    continue
                for slice_type in SLICE_TYPE_ORDER:
                    source_key = (int(source.id), slice_type)
                    direction_key = (
                        int(source.id),
                        int(target.id),
                        slice_type.upper(),
                    )
                    direction_capacities[direction_key] = state[
                        "direction_quota"
                    ].get(
                        (int(source.id), int(target.id), slice_type), 0
                    )
                    direction_accepted[direction_key] = state[
                        "direction_used"
                    ].get((int(source.id), int(target.id), slice_type), 0)
        return {
            "enabled": bool(self.safe_admission_enabled),
            **state,
            "capacities": direction_capacities,
            "accepted": direction_accepted,
            "source_capacities": dict(state["quota"]),
            "source_accepted": dict(state["used"]),
        }

    def _safe_admission_allows(
        self,
        ue: UE,
        source_id: int,
        target_id: int,
        slice_type: str,
    ) -> bool:
        if not self.safe_admission_enabled:
            return True

        normalized_slice = self.normalize_slice_type(slice_type)
        loads = self.get_slice_loads()
        target_load = float(loads.get((int(target_id), normalized_slice), 0.0))
        target_total_load = float(sum(
            loads.get((int(target_id), slice_name), 0.0)
            for slice_name in SLICE_TYPE_ORDER
        ))
        target_gnb = self._get_gnb_by_id(target_id)
        source_gnb = self._get_gnb_by_id(source_id)
        target_budget = max(
            int(getattr(
                target_gnb,
                "n_prbs",
                self.get_slice_prb_budget(target_id, normalized_slice),
            )),
            1,
        )
        source_budget = max(
            int(getattr(
                source_gnb,
                "n_prbs",
                self.get_slice_prb_budget(source_id, normalized_slice),
            )),
            1,
        )
        persistent_demand = int(getattr(ue, "upper_demand_prbs", 0))
        ue_prbs = (
            max(persistent_demand, 1)
            if persistent_demand > 0
            else max(
                int(getattr(ue, "useful_prbs", 0)),
                int(getattr(ue, "prbs", 0)),
                1,
            )
        )
        candidate = {
            "ue_id": int(getattr(ue, "id", -1)),
            "source_id": int(source_id),
            "target_id": int(target_id),
            "slice_type": normalized_slice,
            "a3_margin": 0.0,
            "target_load": target_load,
            "target_total_load": target_total_load,
            "target_load_increment": ue_prbs / float(target_budget),
            "source_load_contribution": ue_prbs / float(source_budget),
            "target_safe_limit": float(
                self.safe_admission_load_limits.get(normalized_slice, 0.80)
            ),
            "target_total_safe_limit": 1.0,
            "target_sla_severity": self.get_slice_sla_severity().get(
                (int(target_id), normalized_slice), 0.0
            ),
        }
        accepted, _rejected, _debug = (
            self.safe_admission_controller.admit_candidates([candidate])
        )
        return bool(accepted)

    def _commit_safe_admission(
        self,
        source_id: int,
        target_id: int,
        slice_type: str,
        candidate: Optional[Dict] = None,
    ) -> None:
        """Consume one admission slot after a successful handover."""
        if not self.safe_admission_enabled:
            return

        normalized_slice = self.normalize_slice_type(slice_type)
        decision = dict(candidate or {})
        decision.update({
            "source_id": int(source_id),
            "target_id": int(target_id),
            "slice_type": normalized_slice,
        })
        self.safe_admission_controller.commit(decision)
        if (
            self.neutralize_offsets_when_quota_exhausted
            and self.safe_admission_controller.quota_exhausted(
                source_id, normalized_slice
            )
        ):
            for target in self.gnbs:
                if int(target.id) != int(source_id):
                    self.set_a3_offset(
                        source_id, int(target.id), normalized_slice, 0.0
                    )

    def _handover_stability_allows(
        self,
        ue: UE,
        source_id: int,
        target_id: int,
        serving_gnb,
        tick: int,
    ) -> bool:
        """Apply persistent per-UE and per-episode guards before admission."""
        reason = self._handover_stability_rejection_reason(
            ue, source_id, target_id, serving_gnb, tick
        )
        if reason is not None:
            self.safe_admission_controller.stats[f"rejected_{reason}"] += 1
            return False
        return True

    def _handover_stability_rejection_reason(
        self,
        ue: UE,
        source_id: int,
        target_id: int,
        serving_gnb,
        tick: int,
    ) -> Optional[str]:
        """Return a stable machine-readable guard reason, or ``None``."""
        ue_id = int(ue.id)
        if self._episode_handover_count >= self.max_handovers_per_episode:
            return "episode_budget"
        if self._ue_episode_handovers.get(ue_id, 0) >= self.max_handovers_per_ue_episode:
            return "ue_episode_budget"

        last = self._last_ho.get(ue_id)
        previous_source = self._last_ho_source.get(ue_id)
        if last is not None and previous_source is not None:
            _last_target, last_tick = last
            is_direct_return = int(target_id) == int(previous_source)
            inside_guard = int(tick) - int(last_tick) < self.a3_pingpong_guard_steps
            if is_direct_return and inside_guard:
                serving_sinr_db = float(
                    self._compute_link_metrics(serving_gnb, ue).get(
                        "sinr_db",
                        self.disconnect_sinr_db,
                    )
                )
                if serving_sinr_db > self.a3_emergency_sinr_db:
                    return "pingpong_guard"
        return None

    def get_a3_offset(self, serving_id: int, neighbor_id: int, slice_type: str) -> float:
        key = (int(serving_id), int(neighbor_id), self.normalize_slice_type(slice_type).upper())
        return float(self._a3_offsets.get(key, 0.0))

    def get_handover_failure_ratios(self) -> Dict[Tuple[int, int, str], float]:
        out: Dict[Tuple[int, int, str], float] = {}
        t = int(self._step_count)
        w = int(self.a3_window_steps)
        keys = set(self._ho_attempts) | set(self._ho_failures)
        for key in keys:
            attempts = sum(1 for tick in self._ho_attempts.get(key, []) if t - tick <= w)
            failures = sum(1 for tick in self._ho_failures.get(key, []) if t - tick <= w)
            out[key] = float(failures / (attempts + 1e-9))
        return out

    def get_ping_pong_ratios(self) -> Dict[Tuple[int, int, str], float]:
        out: Dict[Tuple[int, int, str], float] = {}
        t = int(self._step_count)
        w = int(self.a3_window_steps)
        keys = set(self._ho_successes) | set(self._ho_pingpongs)
        for key in keys:
            successes = sum(1 for tick in self._ho_successes.get(key, []) if t - tick <= w)
            pingpongs = sum(1 for tick in self._ho_pingpongs.get(key, []) if t - tick <= w)
            out[key] = float(pingpongs / (successes + 1e-9))
        return out

    def close(self):
        if self.sumo is not None:
            self.sumo.close()
            self.sumo = None
        super().close()

    def step(self, action=0):
        if action not in (None, 0):
            raise ValueError(
                "Direct UE-to-target-gNB actions were removed. "
                "Use action=0/None to advance the simulator; HRL control should "
                "be implemented through slice-aware A3 offsets."
            )

        self._step_count += 1

        per_gnb_rewards = self._advance_gnbs()
        mobility_already_advanced = False

        if not self._ues and self.use_sumo_mobility:
            self._sync_sumo_mobility(advance=True)
            self._ensure_minimum_sumo_ues()
            mobility_already_advanced = bool(self._ues)

        if not self._ues:
            obs = self._empty_observation()
            reward = 0.0
            terminated = False
            truncated = self._step_count >= self.max_episode_steps
            info = (
                self._build_info(per_gnb_rewards=per_gnb_rewards)
                if self.collect_step_diagnostics
                else {}
            )
            self._last_info = info
            return obs, reward, terminated, truncated, info

        # --------------------------------------------------------------
        # 1) Advance mobility once per environment/decision step.
        #    Radio traffic/service runs later at the finer step_dt scale.
        # --------------------------------------------------------------
        if not mobility_already_advanced:
            self._advance_mobility()

        self._evaluate_a3_handovers()

        # --------------------------------------------------------------
        # 2) Simulate radio traffic/service.
        #    With radio_substeps > 1, one decision step contains many
        #    radio ticks, e.g. 1000 x 1 ms inside one SUMO second.
        # --------------------------------------------------------------
        self._run_radio_substeps()

        reward = 0.0
        obs = self._empty_observation()

        if not np.all(np.isfinite(obs)):
            raise ValueError(f"Non-finite observation detected in step: {obs}")

        terminated = False
        truncated = self._step_count >= int(self.max_episode_steps)

        info = (
            self._build_info(per_gnb_rewards=per_gnb_rewards)
            if self.collect_step_diagnostics
            else {}
        )
        self._last_info = info
        if self.collect_step_diagnostics:
            self._log_step(float(reward))

        return obs, float(reward), terminated, truncated, info

    def _log_step(self, reward: float):
        row = {
            "step": int(self._step_count),
            "reward": float(reward),
            "n_connected_ues": int(sum(1 for ue in self._ues.values() if ue.connected)),
            "n_disconnected_ues": int(sum(1 for ue in self._ues.values() if not ue.connected)),
        }

        for ue in self._ues.values():
            row[f"ue{ue.id}_x"] = float(ue.x)
            row[f"ue{ue.id}_y"] = float(ue.y)
            row[f"ue{ue.id}_serving_gnb"] = -1 if ue.serving_gnb is None else int(ue.serving_gnb)
            row[f"ue{ue.id}_sinr"] = float(ue.e_sinr if np.isfinite(ue.e_sinr) else self.disconnect_sinr_db)
            row[f"ue{ue.id}_throughput"] = float(ue.th)
            row[f"ue{ue.id}_queue"] = float(ue.queue)
            row[f"ue{ue.id}_connected"] = int(bool(ue.connected))
            row[f"ue{ue.id}_handover_pending"] = int(bool(ue.ho_pending))

        self.history.append(row)

    # ------------------------------------------------------------------
    # Mobility
    # ------------------------------------------------------------------
    def _advance_mobility(self):
        if self.use_sumo_mobility:
            self._sync_sumo_mobility(advance=True)
        else:
            for tracked_ue in self._ues.values():
                tracked_ue.update_position(self.mobility_dt)
                self._apply_world_bounds(tracked_ue)
            self._invalidate_metric_caches()

    def _advance_traffic_one_substep(self):
        for tracked_ue in self._ues.values():
            tracked_ue.traffic_step()

    def _estimate_queue_delay(self, ue: UE) -> float:
        """Estimate queue-drain delay in seconds from backlog and service rate."""
        queue_bits = max(float(getattr(ue, "queue", 0.0)), 0.0)
        if queue_bits <= 0.0:
            return 0.0

        step_dt = max(float(self.step_dt), 1e-9)
        instantaneous_bps = max(float(getattr(ue, "bits", 0.0)), 0.0) / step_dt
        smoothed_bps = max(float(getattr(ue, "th", 0.0)), 0.0)
        service_bps = max(instantaneous_bps, smoothed_bps, 1.0)
        return float(min(queue_bits / service_bps, float(self.max_episode_steps)))

    def _update_delay_after_service(self):
        """Update head-of-line packet delay after service."""
        for tracked_ue in self._ues.values():
            if hasattr(tracked_ue, "_update_hol_delay"):
                tracked_ue.hol_delay_s = tracked_ue._update_hol_delay()
                tracked_ue.wait_time = min(
                    tracked_ue.hol_delay_s / max(float(self.step_dt), 1e-12),
                    1000.0,
                )
            else:
                tracked_ue.wait_time = self._estimate_queue_delay(tracked_ue)

    def _sync_ue_step_counters(self):
        for ue in self._ues.values():
            if hasattr(ue, "set_global_step"):
                ue.set_global_step(self._step_count)

    def _run_radio_substeps(self):
        self._sync_ue_step_counters()
        # UE positions are fixed within a local step (mobility advances once before
        # this call), so SINR and the fading vector do not change between substeps.
        # The scheduler is re-run every _radio_sched_stride ticks instead of every
        # tick, avoiding redundant fading + PF computation while still serving
        # newly-arrived packets within the URLLC SLA deadline window.
        #
        # Handover safety: _evaluate_a3_handovers() always runs before this method.
        # Substep i=0 (which satisfies i % stride == 0) always fires the full
        # scheduler, so any handover that completed before this call is immediately
        # reflected in the cached PRB allocation for the new gNB.
        sched_stride = self._radio_sched_stride
        _sched_capacity: Dict[int, float] = {}
        _rx_probs: Dict[int, float] = {}
        _allocated_prbs: Dict[int, int] = {}

        for i in range(self.radio_substeps):
            before = {
                int(ue.id): (
                    int(ue.serving_gnb) if ue.serving_gnb is not None else None,
                    self.normalize_slice_type(getattr(ue, "slice_type", "eMBB")),
                    self._ue_sla_counters(ue),
                )
                for ue in self._ues.values()
            }
            self._advance_traffic_one_substep()

            if i % sched_stride == 0:
                # Full channel + scheduler: runs on substep 0 and every
                # sched_stride ticks thereafter.  Refreshes the cache so that
                # packets that arrived since the last scheduler run are served.
                self._simulate_radio_and_service()
                for ue in self._ues.values():
                    uid = int(ue.id)
                    _sched_capacity[uid] = float(getattr(ue, "scheduled_bits", 0.0))
                    _rx_probs[uid] = self._scalar_rx_probability(getattr(ue, "p", 0.0))
                    _allocated_prbs[uid] = int(getattr(ue, "prbs", 0))
            else:
                # Replay: restore cached capacity, let transmission_step cap it
                # against the current queue and update the PF history.
                for ue in self._ues.values():
                    if not ue.connected or ue.serving_gnb is None:
                        continue
                    uid = int(ue.id)
                    ue.bits = _sched_capacity.get(uid, 0.0)
                    received = bool(
                        _allocated_prbs.get(uid, 0)
                        and self._rng.random() < _rx_probs.get(uid, 0.0)
                    )
                    ue.transmission_step(received)

            self._update_delay_after_service()
            self._accumulate_radio_measurement_sample()
            for ue in self._ues.values():
                if hasattr(ue, "update_sla_expirations"):
                    ue.update_sla_expirations()
            self._accumulate_sla_window(before)
        self._invalidate_metric_caches()

    @staticmethod
    def _ue_sla_counters(ue: UE) -> Dict[str, float]:
        return {
            "offered_bits": float(getattr(ue, "total_bits_arrived", 0.0)),
            "delivered_bits": float(getattr(ue, "total_bits_served", 0.0)),
            "generated_packets": float(getattr(ue, "total_packets_generated", 0.0)),
            "completed_packets": float(getattr(ue, "total_packets_completed", 0.0)),
            "completed_delay_sum_s": float(
                getattr(ue, "total_completed_packet_delay_s", 0.0)
            ),
            "failed_packets": float(getattr(ue, "total_packets_failed_sla", 0.0)),
            "dropped_packets": float(getattr(ue, "total_packets_dropped", 0.0)),
        }

    def _accumulate_sla_window(self, before: Dict[int, Tuple]) -> None:
        for ue in self._ues.values():
            ue_id = int(ue.id)
            if ue_id not in before:
                continue
            serving_id, slice_type, previous = before[ue_id]
            if serving_id is None:
                continue
            key = (int(serving_id), slice_type)
            if key not in self._sla_window_stats:
                continue
            current = self._ue_sla_counters(ue)
            for field in self._sla_window_stats[key]:
                self._sla_window_stats[key][field] += max(
                    current.get(field, 0.0) - previous.get(field, 0.0),
                    0.0,
                )

    def _evaluate_a3_handovers(self, offsets_table: Optional[Dict] = None) -> int:
        """
        Evaluate A3 handovers for connected UEs.

        A neighbor becomes the target when its RSRP is greater than serving
        RSRP plus the configured slice-aware offset and hysteresis for
        handover_ttt consecutive wrapper ticks.
        """
        if offsets_table:
            for (serving_id, neighbor_id, slice_type), info in offsets_table.items():
                value = info.get("applied_offset_db", info) if isinstance(info, dict) else info
                self.set_a3_offset(serving_id, neighbor_id, slice_type, float(value))

        handovers = 0
        t = int(self._step_count)

        candidates = []
        failure_ratios = self.get_handover_failure_ratios()
        pingpong_ratios = self.get_ping_pong_ratios()
        target_sla = self.get_slice_sla_severity()
        for ue in list(self._ues.values()):
            if not ue.connected or ue.serving_gnb is None:
                continue

            ue_id = int(ue.id)
            serving_id = int(ue.serving_gnb)
            slice_type = self.normalize_slice_type(getattr(ue, "slice_type", "eMBB")).upper()
            serving_gnb = self._gnb_by_id.get(serving_id)
            if serving_gnb is None:
                continue

            last_handover = self._last_ho.get(ue_id)
            if last_handover is not None:
                _last_target, last_tick = last_handover
                time_since_handover = t - int(last_tick)
                if time_since_handover < self.a3_handover_cooldown_steps:
                    if hasattr(serving_gnb, "clear_all_a3_counters_for_ue"):
                        serving_gnb.clear_all_a3_counters_for_ue(ue_id)
                    continue
                if time_since_handover < self.a3_min_residence_steps:
                    serving_sinr_db = float(
                        self._compute_link_metrics(serving_gnb, ue).get(
                            "sinr_db",
                            self.disconnect_sinr_db,
                        )
                    )
                    if serving_sinr_db > self.a3_emergency_sinr_db:
                        if hasattr(serving_gnb, "clear_all_a3_counters_for_ue"):
                            serving_gnb.clear_all_a3_counters_for_ue(ue_id)
                        continue

            rsrp_serving = self._measure_rsrp(serving_gnb, ue)
            for neighbor_id, neighbor_gnb in self._gnb_by_id.items():
                neighbor_id = int(neighbor_id)
                if neighbor_id == serving_id:
                    continue
                if not self._is_in_coverage(neighbor_gnb, ue):
                    if hasattr(serving_gnb, "reset_a3_counter"):
                        serving_gnb.reset_a3_counter(ue_id, neighbor_id)
                    continue

                offset = self.get_a3_offset(serving_id, neighbor_id, slice_type)
                rsrp_neighbor = self._measure_rsrp(neighbor_gnb, ue)
                threshold = rsrp_serving + offset + self.a3_hysteresis_db
                if rsrp_neighbor > threshold:
                    if hasattr(serving_gnb, "tick_a3_counter"):
                        ttt_count = serving_gnb.tick_a3_counter(ue_id, neighbor_id)
                    else:
                        ttt_count = self.handover_ttt
                    if ttt_count >= self.handover_ttt:
                        margin = (
                            float(rsrp_neighbor)
                            - float(rsrp_serving)
                            - float(offset)
                            - self.a3_hysteresis_db
                        )
                        normalized_slice = self.normalize_slice_type(slice_type)
                        source_budget = max(
                            int(getattr(serving_gnb, "n_prbs", 0)),
                            1,
                        )
                        target_budget = max(
                            int(getattr(neighbor_gnb, "n_prbs", 0)),
                            1,
                        )
                        persistent_demand = int(
                            getattr(ue, "upper_demand_prbs", 0)
                        )
                        ue_prbs = (
                            max(persistent_demand, 1)
                            if persistent_demand > 0
                            else max(
                                int(getattr(ue, "useful_prbs", 0)),
                                int(getattr(ue, "prbs", 0)),
                                1,
                            )
                        )
                        direction_key = (
                            serving_id,
                            neighbor_id,
                            normalized_slice.upper(),
                        )
                        candidates.append({
                            "ue_id": ue_id,
                            "source_id": serving_id,
                            "target_id": neighbor_id,
                            "slice_type": normalized_slice,
                            "a3_margin": float(margin),
                            "radio_delta_db": float(rsrp_neighbor)
                            - float(rsrp_serving),
                            "rsrp_serving_dbm": float(rsrp_serving),
                            "rsrp_target_dbm": float(rsrp_neighbor),
                            "handover_failure_ratio": float(
                                failure_ratios.get(direction_key, 0.0)
                            ),
                            "pingpong_ratio": float(
                                pingpong_ratios.get(direction_key, 0.0)
                            ),
                            "target_sla_severity": float(
                                target_sla.get(
                                    (neighbor_id, normalized_slice), 0.0
                                )
                            ),
                            "target_load": float(
                                self.estimate_slice_load(
                                    neighbor_id, normalized_slice
                                )
                            ),
                            "target_total_load": float(sum(
                                self.estimate_slice_load(
                                    neighbor_id, slice_name
                                )
                                for slice_name in SLICE_TYPE_ORDER
                            )),
                            "target_load_increment": ue_prbs
                            / float(target_budget),
                            "source_load_contribution": ue_prbs
                            / float(source_budget),
                            "target_safe_limit": float(
                                self.safe_admission_load_limits.get(
                                    normalized_slice, 0.80
                                )
                            ),
                            "target_total_safe_limit": 1.0,
                            "_ue": ue,
                            "_serving_gnb": serving_gnb,
                            "_target_gnb": neighbor_gnb,
                        })
                elif hasattr(serving_gnb, "reset_a3_counter"):
                    serving_gnb.reset_a3_counter(ue_id, neighbor_id)

        for candidate in candidates:
            candidate["guard_rejection_reason"] = (
                self._handover_stability_rejection_reason(
                    candidate["_ue"],
                    candidate["source_id"],
                    candidate["target_id"],
                    candidate["_serving_gnb"],
                    t,
                )
            )

        if self.safe_admission_enabled:
            selected, _rejected, _admission_debug = (
                self.safe_admission_controller.admit_candidates(
                    candidates,
                    max_acceptances=self.max_handovers_per_step,
                    remaining_handover_budget=max(
                        self.max_handovers_per_episode
                        - self._episode_handover_count,
                        0,
                    ),
                    current_tick=t,
                )
            )
        else:
            selected = sorted(
                (
                    candidate
                    for candidate in candidates
                    if candidate["guard_rejection_reason"] is None
                ),
                key=lambda item: item["a3_margin"],
                reverse=True,
            )[: self.max_handovers_per_step]

        for candidate in selected:
            ue = candidate["_ue"]
            serving_gnb = candidate["_serving_gnb"]
            best_neighbor = candidate["_target_gnb"]
            serving_id = int(candidate["source_id"])
            best_neighbor_id = int(candidate["target_id"])
            slice_type = self.normalize_slice_type(candidate["slice_type"]).upper()
            rsrp_serving = float(candidate["rsrp_serving_dbm"])
            best_neighbor_rsrp = float(candidate["rsrp_target_dbm"])
            if ue.serving_gnb is None or int(ue.serving_gnb) != int(serving_id):
                continue
            ue_id = int(ue.id)

            if handovers >= self.max_handovers_per_step:
                continue

            key = (serving_id, int(best_neighbor_id), slice_type)
            self._ho_attempts.setdefault(key, []).append(t)
            success = self._perform_handover(ue, serving_gnb, best_neighbor)
            if not success:
                self._ho_failures.setdefault(key, []).append(t)
                if hasattr(serving_gnb, "reset_a3_counter"):
                    serving_gnb.reset_a3_counter(ue_id, int(best_neighbor_id))
                continue

            self._commit_safe_admission(
                serving_id,
                best_neighbor_id,
                slice_type,
                candidate=candidate,
            )
            self._ho_successes.setdefault(key, []).append(t)
            last = self._last_ho.get(ue_id)
            previous_source = self._last_ho_source.get(ue_id)
            if last is not None and previous_source is not None:
                _previous_target, previous_tick = last
                if (
                    int(best_neighbor_id) == int(previous_source)
                    and (t - previous_tick) <= self.a3_pingpong_threshold_steps
                ):
                    self._ho_pingpongs.setdefault(key, []).append(t)

            self._last_ho_source[ue_id] = int(serving_id)
            self._last_ho[ue_id] = (int(best_neighbor_id), t)
            self._ue_episode_handovers[ue_id] = (
                self._ue_episode_handovers.get(ue_id, 0) + 1
            )
            self._episode_handover_count += 1
            if hasattr(serving_gnb, "clear_all_a3_counters_for_ue"):
                serving_gnb.clear_all_a3_counters_for_ue(ue_id)
            if hasattr(best_neighbor, "clear_all_a3_counters_for_ue"):
                best_neighbor.clear_all_a3_counters_for_ue(ue_id)

            self.handover_events.append({
                "step": t,
                "ue_id": ue_id,
                "slice_type": slice_type,
                "from_gnb": serving_id,
                "to_gnb": int(best_neighbor_id),
                "rsrp_serving_dbm": float(rsrp_serving),
                "rsrp_target_dbm": float(best_neighbor_rsrp),
                "offset_db": self.get_a3_offset(serving_id, int(best_neighbor_id), slice_type),
                "controller": "MultiGNBWrapper",
                "safe_admission": bool(self.safe_admission_enabled),
            })
            handovers += 1

        if handovers:
            self._invalidate_metric_caches()
        return handovers

    def _perform_handover(self, ue: UE, old_gnb, new_gnb) -> bool:
        ue_id = int(ue.id)
        old_id = int(old_gnb.id)
        new_id = int(new_gnb.id)
        old_gnb.detach_ue(ue_id)
        if not new_gnb.attach_ue(ue):
            old_gnb.attach_ue(ue)
            return False

        self._prev_serving_gnb[ue_id] = self._last_serving_gnb.get(ue_id)
        self._last_serving_gnb[ue_id] = old_id
        ue.serving_gnb = new_id
        ue.connected = True
        ue.target_gnb = new_id
        ue.ho_pending = False
        ue.ho_candidate = None
        ue.ho_counter = 0
        return True

    def _measure_rsrp(self, gnb, ue: UE) -> float:
        return float(self._compute_link_metrics(gnb, ue)["rsrp_dbm"])

    def _advance_mobility_and_traffic(self):
        """Compatibility helper for older callers.

        New code should call _advance_mobility() and _run_radio_substeps()
        separately so handover decisions happen between movement and service.
        This compatibility path runs one complete traffic/service tick so the
        delay counter is still updated after transmission.
        """
        self._sync_ue_step_counters()
        self._advance_mobility()
        self._advance_traffic_one_substep()
        self._simulate_radio_and_service()
        self._update_delay_after_service()

    def _restart_sumo_mobility(self):
        if self.sumo is not None:
            self.sumo.close()

        from sumo_wrapper import SumoMobilityWrapper

        self.sumo = SumoMobilityWrapper(
            config_path=self.sumo_config_path,
            sumo_binary=self.sumo_binary,
            port=self.sumo_port,
            label=self.sumo_label,
            step_length=self.mobility_dt,
        )
        self.sumo.start()

    def _sync_sumo_mobility(self, advance: bool):
        if self.sumo is None or not getattr(self.sumo, "started", False):
            self._restart_sumo_mobility()

        mobility = self.sumo.step() if advance else self.sumo.get_mobility_state()
        entities = []

        for sumo_id, data in mobility.get("vehicles", {}).items():
            sumo_id = str(sumo_id)
            entities.append((
                sumo_id,
                "vehicle",
                self._slice_type_for_sumo_entity(sumo_id, "vehicle"),
                data,
            ))

        for sumo_id, data in mobility.get("persons", {}).items():
            sumo_id = str(sumo_id)
            entities.append((
                sumo_id,
                "person",
                self._slice_type_for_sumo_entity(sumo_id, "person"),
                data,
            ))

        seen_sumo_ids = set()
        for sumo_id, entity_type, slice_type, data in entities:
            seen_sumo_ids.add(sumo_id)
            ue_id = self._ue_id_for_sumo_entity(sumo_id, entity_type, slice_type, data)
            if ue_id is None or ue_id not in self._ues:
                continue

            ue = self._ues[ue_id]
            old_x = float(ue.x)
            old_y = float(ue.y)
            ue.x = float(data["x"])
            ue.y = float(data["y"])

            speed = float(data.get("speed", 0.0) or 0.0)
            angle = data.get("angle")
            if angle is None:
                dt = max(float(self.mobility_dt), 1e-9)
                ue.vx = (ue.x - old_x) / dt
                ue.vy = (ue.y - old_y) / dt
            else:
                theta = math.radians(float(angle))
                ue.vx = speed * math.sin(theta)
                ue.vy = speed * math.cos(theta)

            ue.sumo_id = sumo_id
            ue.sumo_entity_type = entity_type
            ue.sumo_road_id = data.get("road_id")

        stale_sumo_ids = set(self._sumo_entity_to_ue_id) - seen_sumo_ids
        for sumo_id in stale_sumo_ids:
            ue_id = self._sumo_entity_to_ue_id.pop(sumo_id)
            self._fading_users.pop(ue_id, None)
            if ue_id in self._ues:
                ue = self._ues.pop(ue_id)
                old_gnb = self._get_gnb_by_id(ue.serving_gnb)
                if old_gnb is not None:
                    old_gnb.detach_ue(ue.id)
                self._last_serving_gnb.pop(ue_id, None)
                self._prev_serving_gnb.pop(ue_id, None)

        if entities or stale_sumo_ids:
            self._invalidate_metric_caches()

    def _ue_id_for_sumo_entity(self, sumo_id: str, entity_type: str, slice_type: str, data: Dict):
        if sumo_id in self._sumo_entity_to_ue_id:
            return self._sumo_entity_to_ue_id[sumo_id]

        mapped_ue_ids = set(self._sumo_entity_to_ue_id.values())
        available = [
            ue.id for ue in self._ues.values()
            if ue.id not in mapped_ue_ids
        ]

        if available:
            ue_id = int(sorted(available)[0])
            ue = self._ues[ue_id]
            ue.slice_type = slice_type
        elif self.sumo_auto_add_ues:
            ue_id = self.add_ue(
                x=float(data["x"]),
                y=float(data["y"]),
                slice_type=slice_type,
            )
            ue = self._ues[ue_id]
        else:
            return None

        ue.sumo_id = sumo_id
        ue.sumo_entity_type = entity_type
        self._sumo_entity_to_ue_id[sumo_id] = ue_id
        return ue_id

    def _ensure_minimum_sumo_ues(self):
        if (
            not self.use_sumo_mobility
            or not self.sumo_auto_add_ues
            or self.min_sumo_bootstrap_ues <= 0
        ):
            return

        while len(self._ues) < self.min_sumo_bootstrap_ues:
            anchor = self.gnbs[len(self._ues) % len(self.gnbs)]
            ue_id = self.add_ue(
                x=float(getattr(anchor, "x", 0.0)),
                y=float(getattr(anchor, "y", 0.0)),
                vx=0.0,
                vy=0.0,
                slice_type=self.sumo_vehicle_slice_type,
            )
            ue = self._ues[ue_id]
            ue.sumo_bootstrap = True

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def clear_ues(self, reset_ids: bool = True):
        for ue_id in list(self._ues):
            for gnb in self.gnbs:
                if hasattr(gnb, "detach_ue"):
                    gnb.detach_ue(ue_id)

        self._ues.clear()
        self._last_serving_gnb.clear()
        self._prev_serving_gnb.clear()
        self._sumo_entity_to_ue_id.clear()
        self._fading_users.clear()

        if reset_ids:
            self._next_ue_id = 0

        self._invalidate_metric_caches()

    def _merge_traffic_profiles(self, profiles: Optional[Dict]) -> Dict[str, Dict]:
        merged = {
            str(slice_type): dict(profile)
            for slice_type, profile in DEFAULT_UE_TRAFFIC_PROFILES.items()
        }
        for slice_type, profile in dict(profiles or {}).items():
            base = dict(merged.get(str(slice_type), {}))
            base.update(dict(profile or {}))
            merged[str(slice_type)] = base
        return merged

    def _traffic_profile_for_slice(self, slice_type: str) -> Dict:
        profile = dict(self.ue_traffic_profiles.get(str(slice_type), {}))
        if not profile:
            profile = dict(self.ue_traffic_profiles.get("eMBB", {}))
        return profile

    def _make_traffic_source(
        self,
        slice_type: str,
        bit_rate: Optional[float],
        packet_size_bits: Optional[float],
        traffic_model: Optional[str],
        bit_rate_schedule=None,
    ):
        profile = self._traffic_profile_for_slice(slice_type)
        model = str(
            traffic_model
            or profile.get("traffic_model")
            or self.default_traffic_model
        )
        rate = float(bit_rate if bit_rate is not None else profile.get("bit_rate", 1_000_000.0))

        if bit_rate_schedule is None:
            bit_rate_schedule = profile.get("bit_rate_schedule")

        if model in {"cbr", "fluid_cbr", "legacy_cbr"}:
            return CbrSource(bit_rate=rate, step_length=self.step_dt)

        if model in {"fixed_packet_cbr", "packet_cbr", "fixed_packet"}:
            packet_size = float(
                packet_size_bits
                if packet_size_bits is not None
                else profile.get("packet_size_bits", 12000.0)
            )
            return FixedPacketCbrSource(
                packet_size=packet_size,
                bit_rate=rate,
                step_length=self.step_dt,
                bit_rate_schedule=bit_rate_schedule,
            )

        raise ValueError(
            "traffic_model must be one of: fixed_packet_cbr, packet_cbr, "
            "fixed_packet, cbr, fluid_cbr, legacy_cbr"
        )

    def add_ue(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        slice_type: str = "eMBB",
        bit_rate: Optional[float] = None,
        packet_size_bits: Optional[float] = None,
        traffic_model: Optional[str] = None,
        bit_rate_schedule=None,
        buffer_size: float = np.inf,
        **ue_kwargs,
    ) -> int:
        ue_id = self._next_ue_id
        self._next_ue_id += 1

        ue = UE(
            id=ue_id,
            slice_ran_id=0,
            traffic_source=self._make_traffic_source(
                slice_type=slice_type,
                bit_rate=bit_rate,
                packet_size_bits=packet_size_bits,
                traffic_model=traffic_model,
                bit_rate_schedule=bit_rate_schedule,
            ),
            type=CBR,
            x=float(x),
            y=float(y),
            vx=float(vx),
            vy=float(vy),
            slot_length=self.step_dt,
            slice_type=slice_type,
            buffer_size=buffer_size,
            **ue_kwargs,
        )

        best = self._find_best_gnb_for_ue(ue)
        ue.serving_gnb = best.id if best is not None else None
        ue.connected = ue.serving_gnb is not None

        if best is not None:
            best.attach_ue(ue)

        self._ues[ue_id] = ue
        self._last_serving_gnb[ue_id] = ue.serving_gnb
        self._prev_serving_gnb[ue_id] = None

        self._invalidate_metric_caches()
        return ue_id

    def get_ue(self, ue_id: int) -> UE:
        return self._ues[ue_id]

    def get_all_ues(self) -> List[UE]:
        return list(self._ues.values())

    def get_ue_radio_metrics(self, ue_id: int) -> Dict[str, float]:
        ue = self._ues[ue_id]
        serving = self._get_gnb_by_id(ue.serving_gnb) if ue.serving_gnb is not None else None

        metrics = self._compute_link_metrics(serving, ue) if serving is not None else {
            "rx_power_dbm": -100.0,
            "rsrp_dbm": -100.0,
            "rssi_dbm": -100.0,
            "rsrq_db": -100.0,
            "noise_dbm": -100.0,
            "interference_dbm": -100.0,
            "snr_db": -100.0,
            "sinr_db": self.disconnect_sinr_db,
            "environment_loss_db": float(self._environment_loss_db(ue)),
        }

        transmission_attempted = bool(int(getattr(ue, "prbs", 0)) > 0)
        return {
            "ue_id": ue.id,
            "serving_gnb": ue.serving_gnb,
            "connected": bool(ue.connected),
            "x": float(ue.x),
            "y": float(ue.y),
            "vx": float(ue.vx),
            "vy": float(ue.vy),
            "queue": float(ue.queue),
            "throughput": float(ue.th),
            "delay_steps": float(ue.wait_time),
            "new_bits": float(ue.new_bits),
            "served_bits": float(ue.bits),
            "scheduled_bits": float(getattr(ue, "scheduled_bits", ue.bits)),
            "dropped_bits": float(getattr(ue, "dropped_bits_step", 0.0)),
            "total_bits_arrived": float(ue.total_bits_arrived),
            "total_bits_dropped": float(getattr(ue, "total_bits_dropped", ue.dropped_bits)),
            "traffic_packet_size_bits": float(getattr(ue.traffic_source, "packet_size", 0.0)),
            "offered_bit_rate": float(getattr(ue.traffic_source, "bit_rate", 0.0)),
            "allocated_prbs": int(ue.prbs),
            "used_prbs": int(getattr(ue, "useful_prbs", ue.prbs)),
            "useful_prbs": int(getattr(ue, "useful_prbs", ue.prbs)),
            "wasted_prbs": int(getattr(ue, "wasted_prbs", 0)),
            # No PRBs means no decoding attempt. Report NaN rather than
            # conflating "not scheduled" with a radio failure probability of 0.
            "rx_probability": (
                self._scalar_rx_probability(getattr(ue, "p", 0.0))
                if transmission_attempted
                else float("nan")
            ),
            "transmission_attempted": transmission_attempted,
            "mcs": -1 if (not ue.connected or getattr(ue, "mcs", None) is None) else int(ue.mcs),
            "mcs_codeset": str(getattr(ue, "mcs_codeset_name", "default")),
            "spectral_efficiency": float(getattr(ue, "spectral_efficiency", 0.0)),
            "scheduled_sinr_db": float(getattr(ue, "e_sinr", metrics["sinr_db"])),
            "effective_sinr_db": float(
                getattr(ue, "effective_sinr_db", float("nan"))
            ),
            "scheduler_mode": self._last_scheduler_mode,
            "snr_db": float(metrics["snr_db"]),
            "sinr_db": float(metrics["sinr_db"]),
            "rx_power_dbm": float(metrics["rx_power_dbm"]),
            "rsrp_dbm": float(metrics["rsrp_dbm"]),
            "rssi_dbm": float(metrics["rssi_dbm"]),
            "rsrq_db": float(metrics["rsrq_db"]),
            "noise_dbm": float(metrics["noise_dbm"]),
            "interference_dbm": float(metrics["interference_dbm"]),
            "target_gnb": ue.target_gnb,
            "ho_pending": bool(ue.ho_pending),
            "ho_candidate": ue.ho_candidate,
            "ho_counter": int(ue.ho_counter),
            "environment_loss_db": float(metrics["environment_loss_db"]),
        }

    # ------------------------------------------------------------------
    # Neighbor discovery
    # ------------------------------------------------------------------
    def get_candidate_gnbs(self, ue: UE, top_k: int = 3) -> List:
        scored: List[Tuple[float, object]] = []
        serving = self._get_gnb_by_id(ue.serving_gnb) if ue.serving_gnb is not None else None

        for gnb in self.gnbs:
            if not self._is_in_coverage(gnb, ue):
                continue

            sinr_db = self._get_sinr_db(gnb, ue)
            if (not np.isfinite(sinr_db)) or (sinr_db <= self.disconnect_sinr_db):
                continue

            scored.append((sinr_db, gnb))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = []
        seen = set()

        if serving is not None:
            serving_sinr = self._get_sinr_db(serving, ue)
            if self._is_in_coverage(serving, ue) and serving_sinr > self.disconnect_sinr_db:
                selected.append(serving)
                seen.add(serving.id)

        for _, gnb in scored:
            if gnb.id in seen:
                continue
            selected.append(gnb)
            seen.add(gnb.id)
            if len(selected) >= top_k + (1 if serving is not None and serving.id in seen else 0):
                break

        return selected[: top_k + 1]

    # ------------------------------------------------------------------
    # Internal step helpers
    # ------------------------------------------------------------------
    def _advance_gnbs(self) -> List[float]:
        rewards = []
        for gnb in self.gnbs:
            reward = 0.0
            if hasattr(gnb, "n_slices_l1") and hasattr(gnb, "step"):
                n_slices = max(int(gnb.n_slices_l1), 1)
                base = int(gnb.n_prbs // n_slices)
                action = np.full((n_slices,), base, dtype=int)
                rem = int(gnb.n_prbs - action.sum())
                if rem > 0:
                    action[:rem] += 1

                try:
                    _state, info = gnb.step(action.tolist())
                    if isinstance(info, dict):
                        if "reward" in info:
                            reward = float(info.get("reward", 0.0))
                        elif "SLA_labels" in info:
                            reward = float(np.mean(info.get("SLA_labels", [0.0])))
                except Exception:
                    reward = 0.0
            rewards.append(reward)
        return rewards

    def _ensure_mcs_scheduler(self):
        if (
            self._pf_scheduler is not None
            and self._mcs_codeset is not None
            and self._mcs_codeset_urllc is not None
        ):
            return True

        try:
            from channel_models import MCSCodeset
            from schedulers import ProportionalFair
        except Exception:
            self.use_mcs_scheduler = False
            self._last_scheduler_mode = "shannon_fallback"
            return False

        self._mcs_codeset = MCSCodeset()
        self._mcs_codeset_urllc = MCSCodeset(
            Path(__file__).resolve().parent / "datasets" / "mcs_codeset_urllc.csv"
        )
        self._pf_scheduler = ProportionalFair(
            self._mcs_codeset,
            granularity=2,
            slot_length=self.step_dt,
            window_seconds=self.pf_averaging_window_s,
            mcs_codesets_by_slice={"URLLC": self._mcs_codeset_urllc},
        )
        self._last_scheduler_mode = "slice_aware_mcs_pf_csv_fading"
        return True

    def _ensure_fading_samples(self, n_prbs: int):
        if (
            self._fading_samples is not None
            and all(sample.shape[0] >= int(n_prbs) for sample in self._fading_samples)
        ):
            return self._fading_samples

        dataset_dir = Path(__file__).resolve().parent / "datasets"
        filenames = [
            dataset_dir / "fading_trace_EPA_3kmph.csv",
            dataset_dir / "fading_trace_ETU_3kmph.csv",
            dataset_dir / "fading_trace_EVA_60kmph.csv",
        ]

        samples = []
        for filename in filenames:
            matrix = np.genfromtxt(filename, delimiter=",")
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            if matrix.shape[0] < n_prbs:
                reps = int(np.ceil(n_prbs / matrix.shape[0]))
                matrix = np.tile(matrix, (reps, 1))
            samples.append(matrix[:n_prbs, :])

        self._fading_samples = samples
        return self._fading_samples

    def _frequency_selective_snr_vector(self, ue: UE, nominal_sinr_db: float, n_prbs: int):
        samples = self._ensure_fading_samples(n_prbs)

        if ue.id not in self._fading_users:
            trace_idx = int(self._rng.integers(len(samples)))
            n_cols = int(samples[trace_idx].shape[1])
            self._fading_users[ue.id] = {
                "trace_idx": trace_idx,
                "index": int(self._rng.integers(n_cols)),
                "step": int(self._rng.choice([-1, 1])),
                "n_cols": n_cols,
            }

        state = self._fading_users[ue.id]
        trace = samples[state["trace_idx"]]

        for _ in range(8):
            state["index"] += state["step"]
            if state["index"] >= state["n_cols"] or state["index"] < 0:
                state["index"] = int(self._rng.integers(state["n_cols"]))
                state["step"] = int(self._rng.choice([-1, 1]))

            fading = trace[:, state["index"]]
            if not np.isnan(np.sum(fading)):
                return np.asarray(fading, dtype=float) + float(nominal_sinr_db)

        return np.full(n_prbs, float(nominal_sinr_db), dtype=float)

    def _estimate_required_prbs(self, ue, sinr_db):
        sinr_linear = max(10.0 ** (sinr_db / 10.0), 1e-6)
        rb_bw = 180e3

        spectral_eff = math.log2(1.0 + sinr_linear)
        spectral_eff = min(max(spectral_eff, 0.0), 8.0)

        bits_per_prb = rb_bw * self.step_dt * spectral_eff

        if bits_per_prb <= 0:
            return 0

        demand_bits = max(float(ue.queue), 0.0)
        return int(np.ceil(demand_bits / bits_per_prb))

    def _mark_ue_radio_disconnected(self, ue: UE):
        ue.connected = False
        ue.prbs = 0
        ue.useful_prbs = 0
        ue.wasted_prbs = 0
        ue.bits = 0
        ue.scheduled_bits = 0
        ue.p = 0
        ue.mcs = None
        ue.spectral_efficiency = 0.0
        ue.effective_sinr_db = float("nan")
        ue.mcs_codeset_name = "default"
        ue.serving_power_dbm = -100.0
        ue.interference_dbm = -100.0
        ue.noise_dbm = -100.0
        ue.estimate_snr([self.disconnect_sinr_db])
        ue.estimate_sinr(self.disconnect_sinr_db)

    @staticmethod
    def _scalar_rx_probability(value) -> float:
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return 0.0
        scalar = float(np.nanmean(arr))
        if not np.isfinite(scalar):
            return 0.0
        return float(np.clip(scalar, 0.0, 1.0))

    def _simulate_radio_and_service(self):
        attached = self._group_ues_by_serving_gnb()

        for gnb_id, ue_list in attached.items():
            if gnb_id is None:
                for ue in ue_list:
                    self._mark_ue_radio_disconnected(ue)
                    ue.transmission_step(received=False)
                continue

            gnb = self._get_gnb_by_id(gnb_id)

            if gnb is None:
                for ue in ue_list:
                    self._mark_ue_radio_disconnected(ue)
                    ue.transmission_step(received=False)
                continue

            metrics_map = {}
            for ue in ue_list:
                ue.prbs = 0
                ue.useful_prbs = 0
                ue.wasted_prbs = 0
                ue.bits = 0
                ue.scheduled_bits = 0
                ue.p = 0
                ue.mcs = None
                ue.spectral_efficiency = 0.0
                ue.effective_sinr_db = float("nan")
                ue.mcs_codeset_name = "default"
                metrics_map[ue.id] = self._compute_link_metrics(gnb, ue)

            if self.use_mcs_scheduler and self._ensure_mcs_scheduler():
                schedulable_ues = []
                gnb_prbs = max(int(getattr(gnb, "n_prbs", 0)), 1)
                for ue in ue_list:
                    metrics = metrics_map[ue.id]
                    sinr_db = metrics["sinr_db"]

                    ue.serving_power_dbm = metrics["rx_power_dbm"]
                    ue.noise_dbm = metrics["noise_dbm"]
                    ue.interference_dbm = metrics["interference_dbm"]
                    ue.estimate_sinr(float(sinr_db))

                    if (not np.isfinite(sinr_db)) or (sinr_db <= self.disconnect_sinr_db):
                        self._mark_ue_radio_disconnected(ue)
                        ue.transmission_step(received=False)
                        continue

                    ue.connected = True
                    snr_vector = self._frequency_selective_snr_vector(
                        ue=ue,
                        nominal_sinr_db=float(sinr_db),
                        n_prbs=gnb_prbs,
                    )
                    ue.estimate_snr(snr_vector)
                    ue.estimate_sinr(float(np.mean(snr_vector)))
                    schedulable_ues.append(ue)

                if schedulable_ues and gnb_prbs > 0:
                    self._pf_scheduler.allocate(schedulable_ues, gnb_prbs)

                for ue in schedulable_ues:
                    if getattr(ue, "mcs", None) is None and self._mcs_codeset is not None:
                        try:
                            codeset = self._mcs_codeset_for_slice(
                                getattr(ue, "slice_type", "eMBB")
                            )
                            ue.mcs, bits_per_sym = codeset.mcs_rate_vs_error(
                                float(ue.e_snr), 0.1
                            )
                            ue.spectral_efficiency = float(bits_per_sym)
                            ue.mcs_codeset_name = (
                                "URLLC"
                                if self.normalize_slice_type(
                                    getattr(ue, "slice_type", "eMBB")
                                ) == "URLLC"
                                else "default"
                            )
                        except Exception:
                            pass
                    rx_probability = self._scalar_rx_probability(getattr(ue, "p", 0.0))
                    received = bool(ue.prbs and self._rng.random() < rx_probability)
                    ue.transmission_step(received)

                continue

            schedulable_fallback = []
            for ue in ue_list:
                metrics = metrics_map[ue.id]
                rx_dbm = metrics["rx_power_dbm"]
                noise_dbm = metrics["noise_dbm"]
                interf_dbm = metrics["interference_dbm"]
                sinr_db = metrics["sinr_db"]
                snr_db = metrics["snr_db"]

                ue.serving_power_dbm = rx_dbm
                ue.noise_dbm = noise_dbm
                ue.interference_dbm = interf_dbm

                ue.estimate_sinr(float(sinr_db))
                ue.estimate_snr([float(snr_db)] if np.isfinite(snr_db) else [self.disconnect_sinr_db])

                if (not np.isfinite(sinr_db)) or (sinr_db <= self.disconnect_sinr_db):
                    self._mark_ue_radio_disconnected(ue)
                    ue.transmission_step(received=False)
                    continue

                ue.connected = True
                schedulable_fallback.append(ue)

            remaining_prbs = int(max(getattr(gnb, "n_prbs", 0), 0))
            sorted_ues = sorted(
                schedulable_fallback,
                key=lambda u: (
                    u.queue,
                    -(metrics_map[u.id]["sinr_db"] if np.isfinite(metrics_map[u.id]["sinr_db"]) else -1e9)
                ),
                reverse=True
            )

            for ue in sorted_ues:
                sinr_db = metrics_map[ue.id]["sinr_db"]
                required_prbs = self._estimate_required_prbs(ue, sinr_db)
                allocated_prbs = min(required_prbs, remaining_prbs)

                ue.prbs = allocated_prbs
                if allocated_prbs > 0:
                    bits_per_prb = self._estimate_bits_for_ue(
                        ue=ue,
                        sinr_db=sinr_db,
                        prbs=1,
                        gnb=gnb,
                    )
                    ue.useful_prbs = int(
                        min(allocated_prbs, np.ceil(max(float(getattr(ue, "queue", 0.0)), 0.0) / max(bits_per_prb, 1.0)))
                    )
                else:
                    ue.useful_prbs = 0
                ue.wasted_prbs = int(max(allocated_prbs - ue.useful_prbs, 0))
                remaining_prbs -= allocated_prbs

                if allocated_prbs <= 0:
                    ue.bits = 0
                    ue.transmission_step(received=False)
                    continue

                ue.bits = self._estimate_bits_for_ue(
                    ue=ue,
                    sinr_db=sinr_db,
                    prbs=allocated_prbs,
                    gnb=gnb,
                )

                ue.transmission_step(received=True)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def get_history(self):
        return list(self.history)

    def get_handover_events(self):
        return list(self.handover_events)

    def _collect_gnb_slice_info(self) -> List[Dict]:
        def to_builtin(value):
            if isinstance(value, dict):
                return {to_builtin(k): to_builtin(v) for k, v in value.items()}
            if isinstance(value, list):
                return [to_builtin(v) for v in value]
            if isinstance(value, tuple):
                return [to_builtin(v) for v in value]
            if isinstance(value, np.ndarray):
                return to_builtin(value.tolist())
            if isinstance(value, np.generic):
                return value.item()
            return value

        slice_info = []

        for gnb in self.gnbs:
            gnb_row = {
                "gnb_id": int(gnb.id),
                "n_prbs": int(getattr(gnb, "n_prbs", 0)),
                "slices": [],
            }

            for idx, l1 in enumerate(getattr(gnb, "slices_l1", [])):
                l1_type = str(getattr(l1, "type", "unknown"))
                l1_row = {
                    "index": int(idx),
                    "type": l1_type,
                    "n_prbs": int(getattr(l1, "n_prbs", 0)),
                    "wrapper_managed": bool(getattr(l1, "wrapper_managed", False)),
                }

                if hasattr(l1, "get_info"):
                    try:
                        l1_row["info"] = to_builtin(l1.get_info())
                    except Exception:
                        l1_row["info"] = {}

                gnb_row["slices"].append(l1_row)

            slice_info.append(gnb_row)

        return slice_info

    def _build_info(self, per_gnb_rewards: Optional[List[float]] = None) -> Dict:
        ue_per_gnb = [0] * self.n_gnbs
        connected = 0
        disconnected = 0

        for ue in self._ues.values():
            if ue.connected and ue.serving_gnb is not None:
                connected += 1
                gidx = self._gnb_index_from_id(ue.serving_gnb)
                if gidx is not None:
                    ue_per_gnb[gidx] += 1
            else:
                disconnected += 1

        if per_gnb_rewards is None:
            per_gnb_rewards = [0.0] * self.n_gnbs

        slice_kpis = {}
        for gnb in self.gnbs:
            gnb_id = int(gnb.id)
            for slice_type in self._configured_slice_types():
                ues = [
                    ue for ue in self._ues.values()
                    if ue.connected
                    and ue.serving_gnb is not None
                    and int(ue.serving_gnb) == gnb_id
                    and self.normalize_slice_type(getattr(ue, "slice_type", "eMBB")) == slice_type
                ]
                budget_prbs = int(self.get_slice_prb_budget(gnb_id, slice_type))
                allocated_prbs = int(sum(max(int(getattr(ue, "prbs", 0)), 0) for ue in ues))
                used_prbs = int(sum(max(int(getattr(ue, "useful_prbs", getattr(ue, "prbs", 0))), 0) for ue in ues))
                wasted_prbs = int(sum(max(int(getattr(ue, "wasted_prbs", 0)), 0) for ue in ues))
                demand_prbs = int(sum(self._estimate_queue_demand_prbs(ue) for ue in ues))
                queue_bits = float(sum(max(float(getattr(ue, "queue", 0.0)), 0.0) for ue in ues))
                arrived_bits = float(sum(max(float(getattr(ue, "new_bits", 0.0)), 0.0) for ue in ues))
                scheduled_bits = float(sum(max(float(getattr(ue, "scheduled_bits", 0.0)), 0.0) for ue in ues))
                served_bits = float(sum(max(float(getattr(ue, "bits", 0.0)), 0.0) for ue in ues))
                offered_bps = float(sum(
                    max(float(getattr(getattr(ue, "traffic_source", None), "bit_rate", 0.0)), 0.0)
                    for ue in ues
                ))
                queue_budget_bits = max(offered_bps * max(float(self.step_dt), 1e-9) * 10.0, 1.0)
                used_load = self._clip01(used_prbs / float(budget_prbs)) if budget_prbs > 0 else 0.0
                allocated_load = self._clip01(allocated_prbs / float(budget_prbs)) if budget_prbs > 0 else 0.0
                wasted_load = self._clip01(wasted_prbs / float(budget_prbs)) if budget_prbs > 0 else 0.0
                demand_load = demand_prbs / float(budget_prbs) if budget_prbs > 0 else 0.0
                queue_pressure = queue_bits / queue_budget_bits
                slice_kpis[(gnb_id, slice_type)] = {
                    "used_prbs": used_prbs,
                    "useful_prbs": used_prbs,
                    "allocated_prbs": allocated_prbs,
                    "wasted_prbs": wasted_prbs,
                    "budget_prbs": budget_prbs,
                    "load": used_load,
                    "used_load": used_load,
                    "allocated_load": allocated_load,
                    "wasted_load": wasted_load,
                    "demand_prbs": demand_prbs,
                    "demand_load": float(demand_load),
                    "queue_bits": queue_bits,
                    "queue_budget_bits": float(queue_budget_bits),
                    "queue_pressure": float(queue_pressure),
                    "arrived_bits": arrived_bits,
                    "scheduled_bits": scheduled_bits,
                    "served_bits": served_bits,
                    "service_to_arrival_ratio": float(served_bits / max(arrived_bits, 1.0)),
                    "served_ratio": self._clip01(served_bits / max(arrived_bits, 1.0)),
                    "tx_success_ratio": float(served_bits / max(scheduled_bits, 1.0)),
                    "ue_count": int(len(ues)),
                    "throughput_bps": float(sum(float(getattr(ue, "th", 0.0)) for ue in ues)),
                    "delay_s": float(np.mean([float(getattr(ue, "hol_delay_s", 0.0)) for ue in ues])) if ues else 0.0,
                    "dropped_bits": float(sum(float(getattr(ue, "dropped_bits_step", 0.0)) for ue in ues)),
                }

        info = {
            "step_count": self._step_count,
            "n_gnbs": self.n_gnbs,
            "n_tracked_ues": len(self._ues),
            "n_connected_ues": connected,
            "n_disconnected_ues": disconnected,
            "ue_per_gnb": ue_per_gnb,
            "per_gnb_rewards": list(per_gnb_rewards),
            "mean_gnb_reward": float(np.mean(per_gnb_rewards)) if per_gnb_rewards else 0.0,
            "radio_dt": self.step_dt,
            "mobility_dt": self.mobility_dt,
            "radio_substeps": self.radio_substeps,
            "gnb_slice_info": self._collect_gnb_slice_info(),
            "slice_kpis": slice_kpis,
            "slice_loads": self.get_slice_loads(),
            "slice_sla_metrics": self.get_slice_sla_metrics(),
            "slice_sla_flags": self.get_slice_sla_flags(),
            "slice_sla_severity": self.get_slice_sla_severity(),
        }

        if self.use_sumo_mobility:
            info["use_sumo_mobility"] = True
            info["n_sumo_entities"] = len(self._sumo_entity_to_ue_id)
            info["sumo_entity_to_ue_id"] = dict(self._sumo_entity_to_ue_id)

        return info

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _empty_observation(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _find_best_gnb_for_ue(self, ue: UE):
        best_gnb = None
        best_sinr = -np.inf
        for gnb in self.gnbs:
            sinr_db = self._get_sinr_db(gnb, ue)
            if sinr_db > best_sinr:
                best_sinr = sinr_db
                best_gnb = gnb
        return best_gnb

    def _group_ues_by_serving_gnb(self) -> Dict[Optional[int], List[UE]]:
        groups: Dict[Optional[int], List[UE]] = {}
        for ue in self._ues.values():
            groups.setdefault(ue.serving_gnb, []).append(ue)
        return groups

    def _gnb_index_from_id(self, gnb_id: Optional[int]) -> Optional[int]:
        if gnb_id is None:
            return None
        return self._gnb_index_by_id.get(int(gnb_id))

    def _get_gnb_by_id(self, gnb_id: Optional[int]):
        if gnb_id is None:
            return None
        lookup = getattr(self, "_gnb_by_id", None)
        if lookup is not None:
            return lookup.get(int(gnb_id))
        # Keep lightweight diagnostic/test wrappers usable without requiring
        # the full constructor-owned lookup cache.
        return next(
            (
                gnb for gnb in getattr(self, "gnbs", [])
                if int(getattr(gnb, "id", -1)) == int(gnb_id)
            ),
            None,
        )

    def _is_in_coverage(self, gnb, ue: UE) -> bool:
        if hasattr(gnb, "is_point_in_coverage"):
            return bool(gnb.is_point_in_coverage(ue.x, ue.y))
        return True

    def _get_rx_power_dbm(self, gnb, ue: UE) -> float:
        if gnb is None:
            return -100.0
        if hasattr(gnb, "get_received_power_dbm"):
            try:
                value = float(gnb.get_received_power_dbm(ue.x, ue.y))
                return value if np.isfinite(value) else -100.0
            except Exception:
                return -100.0
        return -100.0

    def _get_noise_power_dbm(self, gnb) -> float:
        if gnb is None:
            return -100.0
        if hasattr(gnb, "get_noise_power_dbm"):
            try:
                value = float(gnb.get_noise_power_dbm())
                return value if np.isfinite(value) else -100.0
            except Exception:
                pass
        return -100.0

    def _dbm_to_watts(self, p_dbm: float) -> float:
        if not np.isfinite(p_dbm):
            return 0.0
        return 10.0 ** ((p_dbm - 30.0) / 10.0)

    def _watts_to_dbm(self, p_watts: float) -> float:
        if p_watts <= 0.0:
            return -100.0
        return 10.0 * np.log10(p_watts) + 30.0

    def _rsrq_db(self, rsrp_watts: float, rssi_watts: float, n_prbs: int) -> float:
        if rsrp_watts <= 0.0 or rssi_watts <= 0.0:
            return -100.0
        n_prbs = max(int(n_prbs), 1)
        return float(10.0 * np.log10(n_prbs * rsrp_watts / rssi_watts))

    def _compute_interference_watts(self, serving_gnb, ue: UE) -> float:
        if serving_gnb is None:
            return 0.0

        total_watts = 0.0
        for other in self._same_carrier_interferers.get(int(serving_gnb.id), []):
            if not self._is_in_coverage(other, ue):
                continue

            if hasattr(other, "get_received_power_watts"):
                p_w = float(other.get_received_power_watts(ue.x, ue.y))
            else:
                p_dbm = self._get_rx_power_dbm(other, ue)
                p_w = self._dbm_to_watts(p_dbm)

            total_watts += max(p_w, 0.0)

        return total_watts

    def _compute_link_metrics(self, gnb, ue: UE) -> Dict[str, float]:
        cache_key = (-1 if gnb is None else int(gnb.id), int(ue.id))
        cached = self._link_metrics_cache.get(cache_key)
        if cached is not None:
            return cached

        env_loss_db = self._environment_loss_db(ue)

        if gnb is None or not self._is_in_coverage(gnb, ue):
            noise_dbm = self._get_noise_power_dbm(gnb) if gnb is not None else -100.0
            noise_w = self._dbm_to_watts(noise_dbm)
            metrics = {
                "rx_power_dbm": -100.0,
                "rsrp_dbm": -100.0,
                "rssi_dbm": float(self._watts_to_dbm(noise_w)),
                "rsrq_db": -100.0,
                "noise_dbm": float(noise_dbm),
                "interference_dbm": -100.0,
                "snr_db": -100.0,
                "sinr_db": self.disconnect_sinr_db,
                "environment_loss_db": float(env_loss_db),
            }
            self._link_metrics_cache[cache_key] = metrics
            return metrics

        rx_dbm = self._get_rx_power_dbm(gnb, ue) - env_loss_db
        noise_dbm = self._get_noise_power_dbm(gnb)

        if not np.isfinite(rx_dbm) or not np.isfinite(noise_dbm):
            rsrp_dbm = float(rx_dbm) if np.isfinite(rx_dbm) else -100.0
            noise_dbm = float(noise_dbm) if np.isfinite(noise_dbm) else -100.0
            noise_w = self._dbm_to_watts(noise_dbm)
            metrics = {
                "rx_power_dbm": rsrp_dbm,
                "rsrp_dbm": rsrp_dbm,
                "rssi_dbm": float(self._watts_to_dbm(noise_w)),
                "rsrq_db": -100.0,
                "noise_dbm": noise_dbm,
                "interference_dbm": -100.0,
                "snr_db": -100.0,
                "sinr_db": self.disconnect_sinr_db,
                "environment_loss_db": float(env_loss_db),
            }
            self._link_metrics_cache[cache_key] = metrics
            return metrics

        sig_w = self._dbm_to_watts(rx_dbm)
        noise_w = self._dbm_to_watts(noise_dbm)
        interf_w = self._compute_interference_watts(gnb, ue)
        rssi_w = sig_w + noise_w + interf_w

        snr_db = rx_dbm - noise_dbm
        sinr_lin = sig_w / max(noise_w + interf_w, 1e-15)
        sinr_db = 10.0 * np.log10(max(sinr_lin, 1e-15))

        sinr_db = float(np.clip(sinr_db, -20.0, 40.0))
        snr_db = float(np.clip(snr_db, -20.0, 40.0))

        metrics = {
            "rx_power_dbm": float(rx_dbm),
            "rsrp_dbm": float(rx_dbm),
            "rssi_dbm": float(self._watts_to_dbm(rssi_w)),
            "rsrq_db": self._rsrq_db(sig_w, rssi_w, getattr(gnb, "n_prbs", 1)),
            "noise_dbm": float(noise_dbm),
            "interference_dbm": float(self._watts_to_dbm(interf_w)),
            "snr_db": float(snr_db),
            "sinr_db": float(sinr_db),
            "environment_loss_db": float(env_loss_db),
        }
        self._link_metrics_cache[cache_key] = metrics
        return metrics

    def _get_sinr_db(self, gnb, ue: UE) -> float:
        return float(self._compute_link_metrics(gnb, ue)["sinr_db"])

    def _get_snr_db(self, gnb, ue: UE) -> float:
        if gnb is None:
            return -100.0

        for method_name in ("get_ue_snr", "get_snr_db", "get_snr"):
            if hasattr(gnb, method_name):
                method = getattr(gnb, method_name)
                try:
                    value = float(method(ue.x, ue.y))
                    return value if np.isfinite(value) else -100.0
                except TypeError:
                    try:
                        value = float(method(ue))
                        return value if np.isfinite(value) else -100.0
                    except Exception:
                        pass
                except Exception:
                    pass

        rx = self._get_rx_power_dbm(gnb, ue)
        noise = self._get_noise_power_dbm(gnb)
        return float(rx - noise)

    def _estimate_gnb_load(self, gnb_id: int) -> float:
        gnb_id = int(gnb_id)
        cached = self._gnb_load_cache.get(gnb_id)
        if cached is not None:
            return cached

        gnb = self._get_gnb_by_id(gnb_id)
        if gnb is None:
            return 1.0

        for method_name in ("get_load", "load"):
            if hasattr(gnb, method_name):
                attr = getattr(gnb, method_name)
                try:
                    value = float(attr() if callable(attr) else attr)
                    if np.isfinite(value):
                        load = self._clip01(value)
                        self._gnb_load_cache[gnb_id] = load
                        return load
                except Exception:
                    pass

        total_prbs = max(int(getattr(gnb, "n_prbs", 1)), 1)
        used_prbs = 0
        for ue in self._ues.values():
            if ue.serving_gnb == gnb_id and ue.connected:
                used_prbs += int(max(ue.prbs, 0))

        load = self._clip01(used_prbs / total_prbs)
        self._gnb_load_cache[gnb_id] = load
        return load

    def _estimate_gnb_load_pressure(self, gnb_id: int) -> float:
        """Estimate cell pressure without collapsing every overloaded cell to 1.0."""
        gnb_id = int(gnb_id)
        cached = self._gnb_load_pressure_cache.get(gnb_id)
        if cached is not None:
            return cached

        gnb = self._get_gnb_by_id(gnb_id)
        if gnb is None:
            return 1.0

        total_prbs = max(int(getattr(gnb, "n_prbs", 1)), 1)
        used_prbs = 0
        required_prbs = 0

        for ue in self._ues.values():
            if ue.serving_gnb != gnb_id or not ue.connected:
                continue

            used_prbs += int(max(getattr(ue, "prbs", 0), 0))
            sinr_db = float(self._get_sinr_db(gnb, ue))
            required_prbs += int(max(self._estimate_required_prbs(ue, sinr_db), 0))

        allocated_load = used_prbs / total_prbs
        demand_ratio = required_prbs / total_prbs

        if demand_ratio <= 1.0:
            pressure = float(max(allocated_load, demand_ratio))
        else:
            pressure = float(min(1.0 + math.log1p(demand_ratio - 1.0), 10.0))

        self._gnb_load_pressure_cache[gnb_id] = pressure
        return pressure

    def _estimate_bits_for_ue(self, ue: UE, sinr_db: float, prbs: int, gnb) -> int:
        sinr_linear = max(10.0 ** (sinr_db / 10.0), 1e-6)
        rb_bw = 180e3
        spectral_eff = math.log2(1.0 + sinr_linear)
        spectral_eff = min(max(spectral_eff, 0.0), 8.0)

        ue.spectral_efficiency = float(spectral_eff)

        bits = prbs * rb_bw * self.step_dt * spectral_eff
        return max(int(bits), 0)

    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def _snr_fill_value() -> float:
        return -20.0

    def _get_speed(self, ue):
        return float(np.sqrt(ue.vx ** 2 + ue.vy ** 2))

    def _get_distance(self, gnb, ue):
        if gnb is None:
            return 1e6
        if hasattr(gnb, "distance_to_ue"):
            return float(gnb.distance_to_ue(ue.x, ue.y))
        return float(np.sqrt((gnb.x - ue.x) ** 2 + (gnb.y - ue.y) ** 2))

    def _get_approach_score(self, gnb, ue, eps=1e-9):
        if gnb is None:
            return 0.0

        dx = float(gnb.x - ue.x)
        dy = float(gnb.y - ue.y)
        vx = float(ue.vx)
        vy = float(ue.vy)

        v_norm = np.sqrt(vx * vx + vy * vy)
        d_norm = np.sqrt(dx * dx + dy * dy)

        if v_norm < eps or d_norm < eps:
            return 0.0

        score = (vx * dx + vy * dy) / (v_norm * d_norm + eps)
        return float(np.clip(score, -1.0, 1.0))
