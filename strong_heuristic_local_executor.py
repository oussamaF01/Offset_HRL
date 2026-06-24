#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


OFFSET_SET_DB = np.asarray([-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0], dtype=float)
SAFE_EXTENDED_OFFSET_SET_DB = np.asarray(
    [-12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0],
    dtype=float,
)
EXTENDED_1DB_OFFSET_SET_DB = np.asarray(
    [-12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0,
     -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    dtype=float,
)
DEFAULT_SLICE_TYPES = ("eMBB", "URLLC", "mMTC")
EPS = 1e-9

SLICE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "EMBB": {
        "w_b": 0.5,
        "w_ho": 1.2,
        "w_sla": 1.5,
        "w_risk": 2.0,
        "w_target": 2.0,
        "w_osc": 0.2,
    },
    "URLLC": {
        "w_b": 0.4,
        "w_ho": 0.8,
        "w_sla": 2.5,
        "w_risk": 3.0,
        "w_target": 2.5,
        "w_osc": 0.3,
    },
    "MMTC": {
        "w_b": 0.5,
        "w_ho": 1.0,
        "w_sla": 2.0,
        "w_risk": 2.0,
        "w_target": 2.0,
        "w_osc": 0.2,
    },
}


def _slice_key(slice_type: str | int, slice_types: Sequence[str] = DEFAULT_SLICE_TYPES) -> str:
    if isinstance(slice_type, (int, np.integer)):
        return str(slice_types[int(slice_type)]).replace("_", "").replace("-", "").upper()
    return str(slice_type or "eMBB").replace("_", "").replace("-", "").upper()


def _slice_weights(slice_idx: int, slice_types: Sequence[str]) -> Dict[str, float]:
    key = _slice_key(slice_idx, slice_types)
    return SLICE_WEIGHTS.get(key, SLICE_WEIGHTS["EMBB"])


def _coerce_ue_slice_indices(ue_slice: np.ndarray, slice_types: Sequence[str]) -> np.ndarray:
    arr = np.asarray(ue_slice)
    if arr.size == 0:
        return np.asarray([], dtype=int)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(int).reshape(-1)

    slice_index = {_slice_key(slice_type, slice_types): idx for idx, slice_type in enumerate(slice_types)}
    values = []
    for value in arr.reshape(-1):
        key = _slice_key(str(value), slice_types)
        if key not in slice_index:
            raise ValueError(f"Unknown UE slice label {value!r}; expected one of {tuple(slice_types)}")
        values.append(slice_index[key])
    return np.asarray(values, dtype=int)


def evaluate_candidate_offset(
    *,
    gnb_idx: int,
    slice_idx: int,
    candidate_offset_db: float,
    B: np.ndarray,
    prev_offsets: np.ndarray,
    ue_slice: np.ndarray,
    ue_serving_gnb: np.ndarray,
    rsrp_matrix: np.ndarray,
    neighbor_graph: Mapping[int, Sequence[int]],
    load: np.ndarray,
    sla_violation: np.ndarray,
    ho_failure_ratio: np.ndarray,
    pingpong_ratio: np.ndarray,
    hysteresis_db: float = 1.0,
    l_safe: float = 0.85,
    slice_types: Sequence[str] = DEFAULT_SLICE_TYPES,
) -> Tuple[float, Dict[str, float]]:
    """
    Score one candidate A3 offset for a single (gNB, slice) pair.

    Positive offsets make handover harder; negative offsets make handover easier.
    The score rewards alignment with the upper bias, predicted useful handovers,
    and SLA repair, while penalizing mobility risk, overloaded targets, and
    offset oscillation.
    """
    gnb_idx = int(gnb_idx)
    slice_idx = int(slice_idx)
    o = float(candidate_offset_db)

    b_i_s = float(np.clip(B[gnb_idx, slice_idx], -1.0, 1.0))
    prev = float(prev_offsets[gnb_idx, slice_idx])
    violation = float(np.clip(sla_violation[gnb_idx, slice_idx], 0.0, 1.0))
    rhf = float(max(ho_failure_ratio[gnb_idx, slice_idx], 0.0))
    rpp = float(max(pingpong_ratio[gnb_idx, slice_idx], 0.0))
    weights = _slice_weights(slice_idx, slice_types)

    ue_slice_indices = _coerce_ue_slice_indices(ue_slice, slice_types)
    ue_serving = np.asarray(ue_serving_gnb, dtype=int).reshape(-1)
    if ue_serving.size != ue_slice_indices.size:
        raise ValueError("ue_slice and ue_serving_gnb must have the same length")

    ue_mask = (ue_serving == gnb_idx) & (ue_slice_indices == slice_idx)
    ue_indices = np.flatnonzero(ue_mask)
    k_i_s = int(ue_indices.size)

    predicted_targets = []
    if k_i_s > 0:
        neighbors = [int(j) for j in neighbor_graph.get(gnb_idx, [])]
        for ue_idx in ue_indices:
            serving_rsrp = float(rsrp_matrix[ue_idx, gnb_idx])
            best_target = None
            best_margin = -np.inf
            for neighbor_idx in neighbors:
                neighbor_rsrp = float(rsrp_matrix[ue_idx, neighbor_idx])
                margin = neighbor_rsrp - serving_rsrp - o - float(hysteresis_db)
                if margin > best_margin:
                    best_margin = margin
                    best_target = neighbor_idx
            if best_target is not None and best_margin > 0.0:
                predicted_targets.append(int(best_target))

    n_ho = len(predicted_targets)
    ho_frac = float(n_ho / (k_i_s + EPS)) if k_i_s > 0 else 0.0

    # Bias alignment: negative bias aligns with negative offset, positive with positive.
    a_bias = b_i_s * (o / 12.0)

    # Expected execution: offload bias rewards predicted handovers; retain bias penalizes them.
    a_handover = -b_i_s * ho_frac

    # SLA repair: if this gNB-slice violates SLA, movable UEs are useful.
    a_sla = violation * ho_frac

    # Mobility risk: risky cells should avoid aggressive absolute offsets.
    r_risk = (rhf + rpp) * abs(o / 12.0)

    # Target overload: avoid pushing UEs into already overloaded target slices.
    if n_ho > 0:
        overload = [
            max(0.0, float(load[target_idx, slice_idx]) - float(l_safe))
            for target_idx in predicted_targets
        ]
        r_target = float(np.mean(overload))
    else:
        r_target = 0.0

    # Oscillation: prefer stable offset changes.
    r_osc = float(((o - prev) / 12.0) ** 2)

    score = (
        weights["w_b"] * a_bias
        + weights["w_ho"] * a_handover
        + weights["w_sla"] * a_sla
        - weights["w_risk"] * r_risk
        - weights["w_target"] * r_target
        - weights["w_osc"] * r_osc
    )

    # Empty gNB-slices still receive a bias-aligned offset so the policy is
    # already configured if a UE arrives. Keep the extra magnitude penalty
    # small enough to preserve strong bias, but large enough to avoid ±6 dB.
    if k_i_s == 0:
        score -= 0.2 * abs(o / 12.0)
        if abs(o) > 4.0:
            score -= 1.0

    return float(score), {
        "score": float(score),
        "bias": float(b_i_s),
        "candidate_offset_db": float(o),
        "n_ues": float(k_i_s),
        "n_predicted_handovers": float(n_ho),
        "ho_frac": float(ho_frac),
        "a_bias": float(a_bias),
        "a_handover": float(a_handover),
        "a_sla": float(a_sla),
        "r_risk": float(r_risk),
        "r_target": float(r_target),
        "r_osc": float(r_osc),
    }


def strong_heuristic_local_executor(
    B: np.ndarray,
    prev_offsets: np.ndarray,
    ue_slice: np.ndarray,
    ue_serving_gnb: np.ndarray,
    rsrp_matrix: np.ndarray,
    neighbor_graph: Mapping[int, Sequence[int]],
    load: np.ndarray,
    sla_violation: np.ndarray,
    ho_failure_ratio: np.ndarray,
    pingpong_ratio: np.ndarray,
    hysteresis_db: float = 1.0,
    l_safe: float = 0.85,
    slice_types: Sequence[str] = DEFAULT_SLICE_TYPES,
    return_debug: bool = False,
):
    """
    Deterministically choose one discrete A3 offset per (gNB, slice).

    The function evaluates all seven valid offsets for each gNB-slice pair:
    [-6, -4, -2, 0, +2, +4, +6] dB.
    """
    B = np.asarray(B, dtype=float)
    prev_offsets = np.asarray(prev_offsets, dtype=float)
    load = np.asarray(load, dtype=float)
    sla_violation = np.asarray(sla_violation, dtype=float)
    ho_failure_ratio = np.asarray(ho_failure_ratio, dtype=float)
    pingpong_ratio = np.asarray(pingpong_ratio, dtype=float)
    ue_slice = _coerce_ue_slice_indices(ue_slice, slice_types)
    ue_serving_gnb = np.asarray(ue_serving_gnb, dtype=int).reshape(-1)
    rsrp_matrix = np.asarray(rsrp_matrix, dtype=float)

    if B.ndim != 2:
        raise ValueError("B must have shape [num_gnbs, num_slices]")
    expected_shape = B.shape
    for name, arr in {
        "prev_offsets": prev_offsets,
        "load": load,
        "sla_violation": sla_violation,
        "ho_failure_ratio": ho_failure_ratio,
        "pingpong_ratio": pingpong_ratio,
    }.items():
        if arr.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}")
    if rsrp_matrix.shape != (ue_slice.size, expected_shape[0]):
        raise ValueError(
            "rsrp_matrix must have shape [num_ues, num_gnbs], "
            f"got {rsrp_matrix.shape}, expected {(ue_slice.size, expected_shape[0])}"
        )
    if ue_serving_gnb.size != ue_slice.size:
        raise ValueError("ue_slice and ue_serving_gnb must have the same length")

    num_gnbs, num_slices = expected_shape
    offsets = np.zeros((num_gnbs, num_slices), dtype=float)
    debug: Dict[Tuple[int, int], Dict[str, object]] = {}

    for gnb_idx in range(num_gnbs):
        for slice_idx in range(num_slices):
            best_score = -np.inf
            best_offset = 0.0
            candidates = []

            for candidate_offset in OFFSET_SET_DB:
                score, terms = evaluate_candidate_offset(
                    gnb_idx=gnb_idx,
                    slice_idx=slice_idx,
                    candidate_offset_db=float(candidate_offset),
                    B=B,
                    prev_offsets=prev_offsets,
                    ue_slice=ue_slice,
                    ue_serving_gnb=ue_serving_gnb,
                    rsrp_matrix=rsrp_matrix,
                    neighbor_graph=neighbor_graph,
                    load=load,
                    sla_violation=sla_violation,
                    ho_failure_ratio=ho_failure_ratio,
                    pingpong_ratio=pingpong_ratio,
                    hysteresis_db=hysteresis_db,
                    l_safe=l_safe,
                    slice_types=slice_types,
                )
                candidates.append(terms)
                if score > best_score + EPS:
                    best_score = score
                    best_offset = float(candidate_offset)

            offsets[gnb_idx, slice_idx] = best_offset
            debug[(gnb_idx, slice_idx)] = {
                "selected_offset_db": float(best_offset),
                "selected_score": float(best_score),
                "candidates": candidates,
            }

    if return_debug:
        return offsets, debug
    return offsets


def _snap_to_set(value: float, offset_set: np.ndarray) -> float:
    """Return the element of offset_set nearest to value."""
    distances = np.abs(offset_set - value)
    return float(offset_set[np.argmin(distances)])


def strong_directional_heuristic_local_executor(
    B: np.ndarray,
    prev_offsets: np.ndarray,
    ue_slice: np.ndarray,
    ue_serving_gnb: np.ndarray,
    rsrp_matrix: np.ndarray,
    neighbor_graph: Mapping[int, Sequence[int]],
    load: np.ndarray,
    sla_violation: np.ndarray,
    ho_failure_ratio: np.ndarray,
    pingpong_ratio: np.ndarray,
    hysteresis_db: float = 1.0,
    l_safe: float = 0.80,
    max_target_rsrp_deficit_db: float = 8.0,
    slice_types: Sequence[str] = DEFAULT_SLICE_TYPES,
    allow_extended_negative_offsets: bool = True,
    alpha_load: float = 12.0,
    alpha_mobility: float = 3.0,
    eta: float = 0.0,
    return_debug: bool = False,
):
    """Choose one A3 offset for every source-neighbor-slice direction.

    Uses a direct proportional mapping from bias to offset:
        bias < 0  ->  raw_offset = bias * 6    (e.g. -0.1 -> -0.6, -1.0 -> -6)
        bias > 0  ->  raw_offset = bias * 6    (e.g. +0.1 -> +0.6, +1.0 -> +6)
        bias = 0  ->  raw_offset = 0

    Safety corrections (load overload, mobility instability) add a positive
    offset term that reduces aggressiveness. A hard veto forces the offset to
    zero when the target is overloaded or no UE has a viable radio path.

    The result is snapped to the nearest 1 dB step in EXTENDED_1DB_OFFSET_SET_DB
    (or OFFSET_SET_DB when allow_extended_negative_offsets=False).

    Parameters
    ----------
    alpha_load      : dB push per unit of load overload above l_safe (default 12.0)
    alpha_mobility  : dB push per unit of combined HF+PP ratio (default 3.0)
    eta             : EMA smoothing with previous offset in [0, 1) (default 0 = off)
    """
    B = np.asarray(B, dtype=float)
    load = np.asarray(load, dtype=float)
    sla_violation = np.asarray(sla_violation, dtype=float)
    ho_failure_ratio = np.asarray(ho_failure_ratio, dtype=float)
    pingpong_ratio = np.asarray(pingpong_ratio, dtype=float)
    prev_offsets = np.asarray(prev_offsets, dtype=float)
    ue_slice = _coerce_ue_slice_indices(ue_slice, slice_types)
    ue_serving_gnb = np.asarray(ue_serving_gnb, dtype=int).reshape(-1)
    rsrp_matrix = np.asarray(rsrp_matrix, dtype=float)

    if B.ndim not in (2, 3):
        raise ValueError(
            "B must have shape [num_gnbs, num_slices] or "
            "[num_gnbs, max_neighbors, num_slices]"
        )
    if B.ndim == 2:
        num_gnbs, num_slices = B.shape
    else:
        num_gnbs, _bias_neighbors, num_slices = B.shape
    max_neighbors = max((len(neighbor_graph.get(i, ())) for i in range(num_gnbs)), default=0)
    expected_prev_shape = (num_gnbs, max_neighbors, num_slices)
    if B.ndim == 3 and B.shape != expected_prev_shape:
        raise ValueError(
            f"Directional B must have shape {expected_prev_shape}, got {B.shape}"
        )
    if prev_offsets.shape != expected_prev_shape:
        raise ValueError(
            f"prev_offsets must have shape {expected_prev_shape}, got {prev_offsets.shape}"
        )
    source_slice_shape = (num_gnbs, num_slices)
    for name, arr in {"load": load, "sla_violation": sla_violation}.items():
        if arr.shape != source_slice_shape:
            raise ValueError(
                f"{name} must have shape {source_slice_shape}, got {arr.shape}"
            )
    for name, arr in {"ho_failure_ratio": ho_failure_ratio, "pingpong_ratio": pingpong_ratio}.items():
        if arr.shape not in (source_slice_shape, expected_prev_shape):
            raise ValueError(
                f"{name} must have shape {source_slice_shape} or "
                f"{expected_prev_shape}, got {arr.shape}"
            )
    if rsrp_matrix.shape != (ue_slice.size, num_gnbs):
        raise ValueError(
            f"rsrp_matrix must have shape {(ue_slice.size, num_gnbs)}, got {rsrp_matrix.shape}"
        )

    candidate_set = (
        EXTENDED_1DB_OFFSET_SET_DB
        if allow_extended_negative_offsets
        else OFFSET_SET_DB
    )
    min_offset = float(candidate_set[0])
    max_offset = float(candidate_set[-1])

    offsets = np.zeros(expected_prev_shape, dtype=float)
    debug: Dict[Tuple[int, int, int], Dict[str, object]] = {}

    for gnb_idx in range(num_gnbs):
        for neighbor_slot, neighbor_idx in enumerate(neighbor_graph.get(gnb_idx, ())):
            neighbor_idx = int(neighbor_idx)
            for slice_idx in range(num_slices):
                bias = float(np.clip(
                    B[gnb_idx, neighbor_slot, slice_idx]
                    if B.ndim == 3
                    else B[gnb_idx, slice_idx],
                    -1.0,
                    1.0,
                ))
                prev = float(prev_offsets[gnb_idx, neighbor_slot, slice_idx])

                # Radio feasibility: at least one served UE can reach the neighbor.
                source_ue_indices = np.flatnonzero(
                    (ue_serving_gnb == gnb_idx) & (ue_slice == slice_idx)
                )
                if source_ue_indices.size == 0:
                    radio_feasible = True
                else:
                    rsrp_threshold = float(max_target_rsrp_deficit_db) - float(hysteresis_db)
                    radio_feasible = any(
                        float(rsrp_matrix[ue_idx, neighbor_idx])
                        >= float(rsrp_matrix[ue_idx, gnb_idx]) - rsrp_threshold
                        for ue_idx in source_ue_indices
                    )

                target_load = float(load[neighbor_idx, slice_idx])
                target_is_safe = target_load < l_safe

                # 1. Proportional mapping: bias -> raw offset
                # Symmetric ±6 dB mapping. Safe admission, rather than offset
                # magnitude alone, limits the number of executed handovers.
                if abs(bias) <= EPS:
                    raw_offset = 0.0
                elif bias < 0.0:
                    raw_offset = bias * 6.0
                else:
                    raw_offset = bias * 6.0

                # 2. Safety corrections (push offset positive)
                load_over = max(target_load - l_safe, 0.0)
                load_safety = alpha_load * load_over

                rhf = float(
                    ho_failure_ratio[gnb_idx, neighbor_slot, slice_idx]
                    if ho_failure_ratio.ndim == 3
                    else ho_failure_ratio[gnb_idx, slice_idx]
                )
                rpp = float(
                    pingpong_ratio[gnb_idx, neighbor_slot, slice_idx]
                    if pingpong_ratio.ndim == 3
                    else pingpong_ratio[gnb_idx, slice_idx]
                )
                mobility_safety = alpha_mobility * (max(rhf, 0.0) + max(rpp, 0.0))

                proto_offset = raw_offset + load_safety + mobility_safety

                # 3. Hard veto: never negative if target is unsafe or radio infeasible.
                # Also veto if the target has no headroom for one more UE —
                # prevents over-migration when multiple UEs are already in-flight.
                # 0.15 = 1 typical eMBB UE (15 PRBs / 100 total).
                target_has_headroom = (target_load + 0.15) < l_safe
                if proto_offset < 0.0 and (not target_is_safe or not target_has_headroom or not radio_feasible):
                    proto_offset = 0.0

                # 4. Clip to valid range
                proto_offset = float(np.clip(proto_offset, min_offset, max_offset))

                # 5. Optional EMA smoothing with previous applied offset
                if eta > 0.0:
                    proto_offset = (1.0 - eta) * proto_offset + eta * prev

                # 6. Snap to nearest 1 dB step
                applied_offset = _snap_to_set(proto_offset, candidate_set)

                offsets[gnb_idx, neighbor_slot, slice_idx] = applied_offset
                debug[(gnb_idx, neighbor_idx, slice_idx)] = {
                    "bias": bias,
                    "raw_offset_db": raw_offset,
                    "load_safety_db": load_safety,
                    "mobility_safety_db": mobility_safety,
                    "proto_offset_db": proto_offset,
                    "applied_offset_db": applied_offset,
                    "target_is_safe": target_is_safe,
                    "radio_feasible": radio_feasible,
                }

    if return_debug:
        return offsets, debug
    return offsets
