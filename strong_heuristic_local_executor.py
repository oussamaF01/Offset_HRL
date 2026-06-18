#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


OFFSET_SET_DB = np.asarray([-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0], dtype=float)
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
    a_bias = b_i_s * (o / 6.0)

    # Expected execution: offload bias rewards predicted handovers; retain bias penalizes them.
    a_handover = -b_i_s * ho_frac

    # SLA repair: if this gNB-slice violates SLA, movable UEs are useful.
    a_sla = violation * ho_frac

    # Mobility risk: risky cells should avoid aggressive absolute offsets.
    r_risk = (rhf + rpp) * abs(o / 6.0)

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
    r_osc = float(((o - prev) / 6.0) ** 2)

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
        score -= 0.2 * abs(o / 6.0)
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
