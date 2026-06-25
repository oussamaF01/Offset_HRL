from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np


OffsetKey = Tuple[int, str]
DirectionalKey = Tuple[int, int, str]
LoadKey = Tuple[int, str]


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def coordinated_neighbor_offsets(
    *,
    source_id: int,
    neighbor_ids: Sequence[int],
    slice_types: Sequence[str],
    directional_bias: Mapping[DirectionalKey, float],
    useful_load: Mapping[LoadKey, float],
    sla_severity: Mapping[LoadKey, float] | None = None,
    handover_failure_ratio: Mapping[OffsetKey, float] | None = None,
    pingpong_ratio: Mapping[OffsetKey, float] | None = None,
    best_radio_margin_db: Mapping[OffsetKey, float] | None = None,
    safe_load: float = 0.80,
    one_ue_load: float = 0.15,
    bias_deadband: float = 0.05,
    max_target_rsrp_deficit_db: float = 8.0,
) -> dict[OffsetKey, float]:
    """Return coordinated A3 offsets for one source gNB and its neighbors.

    Negative offsets make source->neighbor handover easier.  Positive offsets
    retain traffic at the source.  For negative upper bias, all neighbors for a
    slice compete through one normalized score, so two good targets split the
    migration pressure instead of both receiving the maximum aggressive offset.
    """
    source_id = int(source_id)
    neighbors = tuple(int(neighbor_id) for neighbor_id in neighbor_ids)
    slices = tuple(str(slice_type) for slice_type in slice_types)
    sla_severity = sla_severity or {}
    handover_failure_ratio = handover_failure_ratio or {}
    pingpong_ratio = pingpong_ratio or {}
    best_radio_margin_db = best_radio_margin_db or {}
    safe_load = max(float(safe_load), 1e-6)
    one_ue_load = max(float(one_ue_load), 0.0)
    max_target_rsrp_deficit_db = max(float(max_target_rsrp_deficit_db), 1e-6)

    offsets: dict[OffsetKey, float] = {}

    for slice_type in slices:
        source_load = max(float(useful_load.get((source_id, slice_type), 0.0)), 0.0)
        slice_loads = [source_load] + [
            max(float(useful_load.get((neighbor_id, slice_type), 0.0)), 0.0)
            for neighbor_id in neighbors
        ]
        balance_target = float(np.mean(slice_loads)) if slice_loads else 0.0
        source_excess = max(source_load - balance_target, 0.0)
        source_pressure = _clip01(source_excess / max(safe_load - balance_target, 0.05))

        negative_scores: dict[int, float] = {}

        for neighbor_id in neighbors:
            key = (neighbor_id, slice_type)
            b_ijk = float(np.clip(
                directional_bias.get((source_id, neighbor_id, slice_type), 0.0),
                -1.0,
                1.0,
            ))
            if b_ijk > bias_deadband:
                offsets[key] = float(np.clip(6.0 * b_ijk, 0.0, 6.0))
                continue
            if b_ijk >= -bias_deadband:
                offsets[key] = 0.0
                continue

            target_load = max(float(useful_load.get(key, 0.0)), 0.0)
            headroom = max(safe_load - target_load, 0.0)
            if source_excess <= 0.0 or headroom <= 0.25 * one_ue_load:
                offsets[key] = 0.0
                continue

            radio_margin = float(best_radio_margin_db.get(key, -np.inf))
            radio_score = _clip01(
                (radio_margin + max_target_rsrp_deficit_db)
                / max_target_rsrp_deficit_db
            )
            if radio_score <= 0.0:
                offsets[key] = 0.0
                continue

            risk = _clip01(
                float(sla_severity.get(key, 0.0))
                + float(handover_failure_ratio.get(key, 0.0))
                + float(pingpong_ratio.get(key, 0.0))
            )
            headroom_score = _clip01(headroom / safe_load)
            negative_scores[neighbor_id] = (
                abs(b_ijk) * headroom_score * radio_score * (1.0 - risk)
            )
            offsets[key] = 0.0

        total_score = sum(negative_scores.values())
        if total_score <= 1e-12 or source_pressure <= 0.0:
            continue

        for neighbor_id, score in negative_scores.items():
            share = float(score / total_score)
            key = (neighbor_id, slice_type)
            offsets[key] = float(np.clip(-6.0 * source_pressure * share, -6.0, 0.0))

    return offsets
