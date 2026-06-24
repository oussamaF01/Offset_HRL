#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import Counter
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


SourceSliceKey = Tuple[int, str]
DirectionKey = Tuple[int, int, str]


class SafeAdmissionController:
    """Window-scoped volume control for A3-eligible handover candidates.

    A3 offsets only determine eligibility. This controller freezes a release
    quota at the start of each upper-agent window, ranks eligible candidates,
    and prevents later local steps from exceeding that quota.
    """

    def __init__(
        self,
        *,
        bias_deadband: float = 0.05,
        max_target_sla_severity: float = 0.50,
        failure_penalty_weight: float = 2.0,
        pingpong_penalty_weight: float = 2.0,
        target_sla_penalty_weight: float = 1.0,
    ) -> None:
        self.bias_deadband = max(float(bias_deadband), 0.0)
        self.max_target_sla_severity = float(
            np.clip(max_target_sla_severity, 0.0, 1.0)
        )
        self.failure_penalty_weight = max(float(failure_penalty_weight), 0.0)
        self.pingpong_penalty_weight = max(float(pingpong_penalty_weight), 0.0)
        self.target_sla_penalty_weight = max(
            float(target_sla_penalty_weight), 0.0
        )
        self.window_id = 0
        self.reset_window()

    @staticmethod
    def _slice_name(slice_type: str) -> str:
        return str(slice_type)

    def reset_window(self) -> None:
        self.bias: Dict[SourceSliceKey, float] = {}
        self.direction_bias: Dict[DirectionKey, float] = {}
        self.balance_target: Dict[SourceSliceKey, float] = {}
        self.source_excess_load: Dict[SourceSliceKey, float] = {}
        self.requested_release_load: Dict[SourceSliceKey, float] = {}
        self.estimated_ue_load: Dict[SourceSliceKey, float] = {}
        self.quota: Dict[SourceSliceKey, int] = {}
        self.direction_quota: Dict[DirectionKey, int] = {}
        self.used: Dict[SourceSliceKey, int] = {}
        self.used_load: Dict[SourceSliceKey, float] = {}
        self.direction_used: Dict[DirectionKey, int] = {}
        self.denied_until: Dict[Tuple[int, int, int, str], int] = {}
        self.stats: Counter = Counter()
        self.last_accepted: List[dict] = []
        self.last_rejected: List[dict] = []

    def begin_upper_window(
        self,
        *,
        bias_matrix: np.ndarray | None = None,
        directional_bias: np.ndarray | None = None,
        neighbor_graph: Mapping[int, Sequence[int]] | None = None,
        gnb_ids: Sequence[int],
        slice_types: Sequence[str],
        loads: Mapping[SourceSliceKey, float],
        ue_counts: Mapping[SourceSliceKey, int],
        balance_targets: Mapping[str, float] | None = None,
        remaining_handover_budget: int | None = None,
    ) -> Dict[SourceSliceKey, int]:
        """Reset state and freeze one release quota per source and slice."""
        if directional_bias is None:
            bias = np.asarray(bias_matrix, dtype=float)
            if bias.shape != (len(gnb_ids), len(slice_types)):
                raise ValueError(
                    "bias_matrix must have shape "
                    f"{(len(gnb_ids), len(slice_types))}, got {bias.shape}"
                )
            graph = {
                int(source_id): [
                    int(target_id) for target_id in gnb_ids
                    if int(target_id) != int(source_id)
                ]
                for source_id in gnb_ids
            }
            directional = np.repeat(bias[:, None, :], max(map(len, graph.values())), axis=1)
        else:
            directional = np.asarray(directional_bias, dtype=float)
            graph = {
                int(source_id): [int(target_id) for target_id in targets]
                for source_id, targets in dict(neighbor_graph or {}).items()
            }
            expected = (
                len(gnb_ids),
                max((len(graph.get(int(source_id), ())) for source_id in gnb_ids), default=0),
                len(slice_types),
            )
            if directional.shape != expected:
                raise ValueError(
                    f"directional_bias must have shape {expected}, got {directional.shape}"
                )

        self.reset_window()
        self.window_id += 1
        normalized_slices = tuple(self._slice_name(s) for s in slice_types)
        if balance_targets is None:
            balance_targets = {
                slice_type: float(
                    np.mean([
                        float(loads.get((int(gnb_id), slice_type), 0.0))
                        for gnb_id in gnb_ids
                    ])
                )
                for slice_type in normalized_slices
            }

        budget_remaining = (
            None
            if remaining_handover_budget is None
            else max(int(remaining_handover_budget), 0)
        )
        for source_idx, gnb_id in enumerate(gnb_ids):
            source_id = int(gnb_id)
            for slice_idx, slice_type in enumerate(normalized_slices):
                key = (source_id, slice_type)
                source_load = max(float(loads.get(key, 0.0)), 0.0)
                target = max(float(balance_targets.get(slice_type, 0.0)), 0.0)
                source_ues = max(int(ue_counts.get(key, 0)), 0)
                targets = graph.get(source_id, ())
                direction_strengths = []
                for target_slot, target_id in enumerate(targets):
                    direction_key = (source_id, int(target_id), slice_type)
                    value = float(np.clip(
                        directional[source_idx, target_slot, slice_idx], -1.0, 1.0
                    ))
                    self.direction_bias[direction_key] = value
                    direction_strengths.append(
                        abs(value) if value < -self.bias_deadband else 0.0
                    )
                release_fraction = min(sum(direction_strengths), 1.0)
                source_bias = -release_fraction if release_fraction > 0.0 else 0.0
                excess = max(source_load - target, 0.0)
                requested_load = release_fraction * excess
                average_ue_load = (
                    source_load / float(source_ues)
                    if source_ues > 0 and source_load > 0.0
                    else 0.0
                )

                if requested_load <= 0.0 or average_ue_load <= 0.0:
                    quota = 0
                else:
                    quota = int(math.ceil(
                        requested_load / average_ue_load - 1e-9
                    ))
                    quota = min(max(quota, 0), source_ues)
                if budget_remaining is not None:
                    quota = min(quota, budget_remaining)

                self.bias[key] = source_bias
                self.balance_target[key] = target
                self.source_excess_load[key] = excess
                self.requested_release_load[key] = requested_load
                self.estimated_ue_load[key] = average_ue_load
                self.quota[key] = quota
                self.used[key] = 0
                self.used_load[key] = 0.0

                if quota > 0 and sum(direction_strengths) > 0.0:
                    exact = [
                        quota * strength / sum(direction_strengths)
                        for strength in direction_strengths
                    ]
                    allocated = [int(math.floor(value)) for value in exact]
                    remaining = quota - sum(allocated)
                    order = sorted(
                        range(len(exact)),
                        key=lambda idx: exact[idx] - allocated[idx],
                        reverse=True,
                    )
                    for idx in order[:remaining]:
                        allocated[idx] += 1
                else:
                    allocated = [0] * len(targets)
                for target_id, direction_quota in zip(targets, allocated):
                    self.direction_quota[
                        (source_id, int(target_id), slice_type)
                    ] = int(direction_quota)

        return dict(self.quota)

    def candidate_score(self, candidate: Mapping) -> float:
        """Rank safer candidates first while retaining A3 margin as the base."""
        return float(
            float(candidate.get(
                "radio_delta_db",
                candidate.get("a3_margin", 0.0),
            ))
            - self.pingpong_penalty_weight
            * float(candidate.get("pingpong_ratio", 0.0))
            - self.failure_penalty_weight
            * float(candidate.get("handover_failure_ratio", 0.0))
            - self.target_sla_penalty_weight
            * float(candidate.get("target_sla_severity", 0.0))
        )

    def admit_candidates(
        self,
        candidates: Iterable[Mapping],
        *,
        max_acceptances: int | None = None,
        remaining_handover_budget: int | None = None,
        current_tick: int | None = None,
    ) -> Tuple[List[dict], List[dict], Dict]:
        """Rank and filter one local-step batch without consuming quota.

        Quota is consumed only by ``commit`` after the simulator confirms that
        the selected handover succeeded.
        """
        ranked = [dict(candidate) for candidate in candidates]
        for candidate in ranked:
            candidate["admission_score"] = self.candidate_score(candidate)
        ranked.sort(key=lambda item: item["admission_score"], reverse=True)

        accepted: List[dict] = []
        rejected: List[dict] = []
        provisional_by_source: Counter = Counter()
        projected_target_load: Dict[Tuple[int, str], float] = {}
        projected_target_total_load: Dict[int, float] = {}
        selected_ues = set()
        step_limit = len(ranked) if max_acceptances is None else max(
            int(max_acceptances), 0
        )
        global_limit = (
            len(ranked)
            if remaining_handover_budget is None
            else max(int(remaining_handover_budget), 0)
        )

        def reject(candidate: dict, reason: str) -> None:
            decision = dict(candidate)
            decision["accepted"] = False
            decision["rejection_reason"] = reason
            rejected.append(decision)
            self.stats[f"rejected_{reason}"] += 1

        for candidate in ranked:
            source_id = int(candidate["source_id"])
            target_id = int(candidate["target_id"])
            slice_type = self._slice_name(candidate["slice_type"])
            source_key = (source_id, slice_type)
            direction_key = (source_id, target_id, slice_type)
            target_key = (target_id, slice_type)
            ue_id = int(candidate["ue_id"])

            self.stats["eligible"] += 1
            guard_reason = candidate.get("guard_rejection_reason")
            if guard_reason:
                reject(candidate, str(guard_reason))
                continue
            denied_key = (ue_id, source_id, target_id, slice_type)
            if (
                current_tick is not None
                and self.denied_until.get(denied_key, -1) > int(current_tick)
            ):
                reject(candidate, "temporary_guard")
                continue
            if ue_id in selected_ues:
                reject(candidate, "duplicate_ue")
                continue
            if len(accepted) >= step_limit:
                reject(candidate, "step_budget")
                continue
            if len(accepted) >= global_limit:
                reject(candidate, "episode_budget")
                continue
            source_excess = self.source_excess_load.get(source_key, 0.0)
            agent_signaled = (
                self.direction_bias.get(direction_key, 0.0)
                < -self.bias_deadband
            )
            if not agent_signaled:
                reject(candidate, "no_directional_release")
                continue
            if source_excess <= 0.0:
                reject(candidate, "no_source_excess")
                continue
            quota = self.quota.get(source_key, 0)
            consumed = self.used.get(source_key, 0) + provisional_by_source[source_key]
            if consumed >= quota:
                reject(candidate, "quota_exhausted")
                continue
            direction_consumed = (
                self.direction_used.get(direction_key, 0)
                + sum(
                    1
                    for item in accepted
                    if (
                        int(item["source_id"]),
                        int(item["target_id"]),
                        self._slice_name(item["slice_type"]),
                    ) == direction_key
                )
            )
            if direction_consumed >= self.direction_quota.get(direction_key, 0):
                reject(candidate, "direction_quota_exhausted")
                continue
            if float(candidate.get("target_sla_severity", 0.0)) > (
                self.max_target_sla_severity
            ):
                reject(candidate, "target_sla")
                continue

            target_load = float(candidate.get("target_load", 0.0))
            load_increment = max(float(candidate.get("target_load_increment", 0.0)), 0.0)
            safe_limit = float(candidate.get("target_safe_limit", 1.0))
            projected_before = projected_target_load.get(target_key, target_load)
            projected_after = projected_before + load_increment
            if projected_after > safe_limit + 1e-12:
                reject(candidate, "target_safety")
                continue
            target_total_load = float(
                candidate.get("target_total_load", target_load)
            )
            target_total_safe_limit = float(
                candidate.get("target_total_safe_limit", 1.0)
            )
            projected_total_before = projected_target_total_load.get(
                target_id, target_total_load
            )
            projected_total_after = projected_total_before + load_increment
            if projected_total_after > target_total_safe_limit + 1e-12:
                reject(candidate, "target_total_safety")
                continue

            decision = dict(candidate)
            decision["accepted"] = True
            decision["projected_target_load_after"] = projected_after
            decision["projected_target_total_load_after"] = (
                projected_total_after
            )
            accepted.append(decision)
            selected_ues.add(ue_id)
            provisional_by_source[source_key] += 1
            projected_target_load[target_key] = projected_after
            projected_target_total_load[target_id] = projected_total_after

        self.last_accepted = [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in accepted
        ]
        self.last_rejected = [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in rejected
        ]
        debug = {
            "window_id": self.window_id,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "rejection_reasons": dict(Counter(
                item["rejection_reason"] for item in rejected
            )),
            "remaining_quota": {
                key: max(self.quota.get(key, 0) - self.used.get(key, 0), 0)
                for key in self.quota
            },
        }
        return accepted, rejected, debug

    def commit(self, candidate: Mapping) -> None:
        """Consume one source quota slot after a successful handover."""
        source_id = int(candidate["source_id"])
        target_id = int(candidate["target_id"])
        slice_type = self._slice_name(candidate["slice_type"])
        source_key = (source_id, slice_type)
        direction_key = (source_id, target_id, slice_type)
        self.used[source_key] = self.used.get(source_key, 0) + 1
        self.used_load[source_key] = (
            self.used_load.get(source_key, 0.0)
            + max(float(candidate.get("source_load_contribution", 0.0)), 0.0)
        )
        self.direction_used[direction_key] = self.direction_used.get(direction_key, 0) + 1
        self.stats["accepted"] += 1

    def quota_exhausted(self, source_id: int, slice_type: str) -> bool:
        key = (int(source_id), self._slice_name(slice_type))
        return self.used.get(key, 0) >= self.quota.get(key, 0)

    def get_state(self) -> Dict:
        stats = {
            "eligible": 0,
            "accepted": 0,
            "rejected_no_release_pressure": 0,
            "rejected_no_directional_release": 0,
            "rejected_no_source_excess": 0,
            "rejected_quota_exhausted": 0,
            "rejected_direction_quota_exhausted": 0,
            "rejected_target_safety": 0,
            "rejected_target_total_safety": 0,
            "rejected_target_sla": 0,
            "rejected_pingpong_guard": 0,
            "rejected_ue_episode_budget": 0,
            "rejected_episode_budget": 0,
            "rejected_step_budget": 0,
            **dict(self.stats),
        }
        return {
            "window_id": int(self.window_id),
            "bias": dict(self.bias),
            "direction_bias": dict(self.direction_bias),
            "quota": dict(self.quota),
            "direction_quota": dict(self.direction_quota),
            "used": dict(self.used),
            "remaining": {
                key: max(self.quota.get(key, 0) - self.used.get(key, 0), 0)
                for key in self.quota
            },
            "requested_release_load": dict(self.requested_release_load),
            "source_excess_load": dict(self.source_excess_load),
            "estimated_ue_load": dict(self.estimated_ue_load),
            "used_load": dict(self.used_load),
            "direction_used": dict(self.direction_used),
            "stats": stats,
            "last_accepted": list(self.last_accepted),
            "last_rejected": list(self.last_rejected),
        }
