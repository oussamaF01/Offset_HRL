#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import Counter
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


SourceSliceKey = Tuple[int, str]
DirectionKey = Tuple[int, int, str]


class DirectionalAdmissionBudgetController:
    """Window-scoped directional handover budget layer.

    A3 offsets create radio eligibility. This controller only limits how many
    eligible handovers may execute for each source -> target -> slice direction
    during one upper-agent window. It deliberately does not veto candidates
    using traffic-safety signals such as source excess, target load, total load,
    or target SLA.
    """

    def __init__(
        self,
        *,
        bias_deadband: float = 0.05,
        **_ignored_legacy_kwargs,
    ) -> None:
        self.bias_deadband = max(float(bias_deadband), 0.0)
        self.window_id = 0
        self.reset_window()

    @staticmethod
    def _slice_name(slice_type: str) -> str:
        return str(slice_type)

    def reset_window(self) -> None:
        self.bias: Dict[SourceSliceKey, float] = {}
        self.direction_bias: Dict[DirectionKey, float] = {}
        self.quota: Dict[SourceSliceKey, int] = {}
        self.direction_quota: Dict[DirectionKey, int] = {}
        self.used: Dict[SourceSliceKey, int] = {}
        self.direction_used: Dict[DirectionKey, int] = {}
        self.denied_until: Dict[Tuple[int, int, int, str], int] = {}
        self.stats: Counter = Counter()
        self.last_accepted: List[dict] = []
        self.last_rejected: List[dict] = []

        # Legacy diagnostics kept so old logging/tests can read state without
        # reintroducing traffic-safety decisions into admission.
        self.balance_target: Dict[SourceSliceKey, float] = {}
        self.source_excess_load: Dict[SourceSliceKey, float] = {}
        self.requested_release_load: Dict[SourceSliceKey, float] = {}
        self.estimated_ue_load: Dict[SourceSliceKey, float] = {}
        self.used_load: Dict[SourceSliceKey, float] = {}

    def begin_upper_window(
        self,
        *,
        bias_matrix: np.ndarray | None = None,
        directional_bias: np.ndarray | None = None,
        neighbor_graph: Mapping[int, Sequence[int]] | None = None,
        gnb_ids: Sequence[int],
        slice_types: Sequence[str],
        loads: Mapping[SourceSliceKey, float] | None = None,
        ue_counts: Mapping[SourceSliceKey, int],
        balance_targets: Mapping[str, float] | None = None,
        remaining_handover_budget: int | None = None,
    ) -> Dict[SourceSliceKey, int]:
        """Reset state and freeze per-direction quotas for one upper window."""
        normalized_slices = tuple(self._slice_name(s) for s in slice_types)
        gnb_ids = tuple(int(gnb_id) for gnb_id in gnb_ids)
        graph = {
            int(source_id): [int(target_id) for target_id in targets]
            for source_id, targets in dict(neighbor_graph or {}).items()
        }
        if not graph:
            graph = {
                int(source_id): [
                    int(target_id) for target_id in gnb_ids
                    if int(target_id) != int(source_id)
                ]
                for source_id in gnb_ids
            }
        max_neighbors = max(
            (len(graph.get(int(source_id), ())) for source_id in gnb_ids),
            default=0,
        )

        if directional_bias is None:
            bias = np.asarray(bias_matrix, dtype=float)
            if bias.shape != (len(gnb_ids), len(normalized_slices)):
                raise ValueError(
                    "bias_matrix must have shape "
                    f"{(len(gnb_ids), len(normalized_slices))}, got {bias.shape}"
                )
            directional = np.repeat(bias[:, None, :], max_neighbors, axis=1)
        else:
            directional = np.asarray(directional_bias, dtype=float)

        expected = (len(gnb_ids), max_neighbors, len(normalized_slices))
        if directional.shape != expected:
            raise ValueError(
                f"directional_bias must have shape {expected}, got {directional.shape}"
            )

        self.reset_window()
        self.window_id += 1
        loads = dict(loads or {})
        balance_targets = dict(balance_targets or {})

        for source_idx, source_id in enumerate(gnb_ids):
            targets = tuple(graph.get(int(source_id), ()))
            for slice_idx, slice_type in enumerate(normalized_slices):
                source_key = (int(source_id), slice_type)
                source_ues = max(int(ue_counts.get(source_key, 0)), 0)
                source_quota = 0
                source_bias_values = []

                source_load = max(float(loads.get(source_key, 0.0)), 0.0)
                target_load = max(float(balance_targets.get(slice_type, 0.0)), 0.0)
                self.balance_target[source_key] = target_load
                self.source_excess_load[source_key] = max(source_load - target_load, 0.0)
                self.estimated_ue_load[source_key] = (
                    source_load / float(source_ues)
                    if source_ues > 0 and source_load > 0.0
                    else 0.0
                )
                self.used[source_key] = 0
                self.used_load[source_key] = 0.0

                for target_slot, target_id in enumerate(targets):
                    direction_key = (int(source_id), int(target_id), slice_type)
                    value = float(
                        np.clip(directional[source_idx, target_slot, slice_idx], -1.0, 1.0)
                    )
                    strength = max(0.0, -value)
                    if strength <= self.bias_deadband:
                        direction_budget = 0
                    else:
                        direction_budget = min(
                            int(math.ceil(strength * float(source_ues) - 1e-6)),
                            source_ues,
                        )
                    self.direction_bias[direction_key] = value
                    self.direction_quota[direction_key] = int(direction_budget)
                    self.direction_used[direction_key] = 0
                    source_quota += int(direction_budget)
                    source_bias_values.append(value)

                self.quota[source_key] = int(source_quota)
                self.requested_release_load[source_key] = (
                    float(source_quota) * self.estimated_ue_load.get(source_key, 0.0)
                )
                negative_strength = sum(max(0.0, -value) for value in source_bias_values)
                self.bias[source_key] = -min(float(negative_strength), 1.0)

        return dict(self.quota)

    def candidate_sort_key(self, candidate: Mapping) -> tuple:
        return (
            float(candidate.get("a3_margin", candidate.get("radio_delta_db", 0.0))),
            -float(candidate.get("pingpong_ratio", 0.0)),
            -float(candidate.get("handover_failure_ratio", 0.0)),
        )

    def admit_candidates(
        self,
        candidates: Iterable[Mapping],
        *,
        max_acceptances: int | None = None,
        remaining_handover_budget: int | None = None,
        current_tick: int | None = None,
    ) -> Tuple[List[dict], List[dict], Dict]:
        """Select eligible candidates without consuming real quota.

        Real budget consumption happens in ``commit`` after the simulator
        confirms that the handover succeeded. This method uses temporary counts
        only to avoid over-selecting a direction in the current batch.
        """
        ranked = [dict(candidate) for candidate in candidates]
        ranked.sort(key=self.candidate_sort_key, reverse=True)

        accepted: List[dict] = []
        rejected: List[dict] = []
        temporary_direction_used: Counter = Counter()
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
                reject(candidate, "global_budget_exhausted")
                continue

            quota = int(self.direction_quota.get(direction_key, 0))
            used = int(self.direction_used.get(direction_key, 0))
            provisional = int(temporary_direction_used[direction_key])
            if quota <= 0:
                reject(candidate, "no_directional_budget")
                continue
            if used + provisional >= quota:
                reject(candidate, "direction_quota_exhausted")
                continue

            decision = dict(candidate)
            decision["accepted"] = True
            accepted.append(decision)
            selected_ues.add(ue_id)
            temporary_direction_used[direction_key] += 1

            # Keep source counters projected only for debug; real consumption is
            # still strictly deferred to commit().
            self.quota.setdefault(source_key, 0)
            self.used.setdefault(source_key, 0)

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
            "direction_remaining": {
                key: max(
                    self.direction_quota.get(key, 0)
                    - self.direction_used.get(key, 0),
                    0,
                )
                for key in self.direction_quota
            },
            "remaining_quota": {
                key: max(self.quota.get(key, 0) - self.used.get(key, 0), 0)
                for key in self.quota
            },
        }
        return accepted, rejected, debug

    def commit(self, candidate: Mapping) -> None:
        """Consume one slot after a successful handover."""
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

    def direction_quota_exhausted(
        self,
        source_id: int,
        target_id: int,
        slice_type: str,
    ) -> bool:
        key = (int(source_id), int(target_id), self._slice_name(slice_type))
        return self.direction_used.get(key, 0) >= self.direction_quota.get(key, 0)

    def get_state(self) -> Dict:
        direction_remaining = {
            key: max(self.direction_quota.get(key, 0) - self.direction_used.get(key, 0), 0)
            for key in self.direction_quota
        }
        stats = {
            "eligible": 0,
            "accepted": 0,
            "rejected_no_directional_budget": 0,
            "rejected_direction_quota_exhausted": 0,
            "rejected_global_budget_exhausted": 0,
            "rejected_pingpong_guard": 0,
            "rejected_ue_episode_budget": 0,
            "rejected_episode_budget": 0,
            "rejected_step_budget": 0,
            "rejected_temporary_guard": 0,
            "rejected_duplicate_ue": 0,
            # Legacy safety-veto counters intentionally remain zero unless
            # external guard code writes them.
            "rejected_no_source_excess": 0,
            "rejected_target_safety": 0,
            "rejected_target_total_safety": 0,
            "rejected_target_sla": 0,
            "rejected_no_directional_release": 0,
            "rejected_quota_exhausted": 0,
            **dict(self.stats),
        }
        return {
            "window_id": int(self.window_id),
            "bias": dict(self.bias),
            "direction_bias": dict(self.direction_bias),
            "quota": dict(self.quota),
            "used": dict(self.used),
            "remaining": {
                key: max(self.quota.get(key, 0) - self.used.get(key, 0), 0)
                for key in self.quota
            },
            "direction_quota": dict(self.direction_quota),
            "direction_used": dict(self.direction_used),
            "direction_remaining": direction_remaining,
            "requested_release_load": dict(self.requested_release_load),
            "source_excess_load": dict(self.source_excess_load),
            "estimated_ue_load": dict(self.estimated_ue_load),
            "used_load": dict(self.used_load),
            "stats": stats,
            "last_accepted": list(self.last_accepted),
            "last_rejected": list(self.last_rejected),
        }


# Compatibility name used by the existing wrapper and tests.
SafeAdmissionController = DirectionalAdmissionBudgetController
