#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from local_a3_agent_wrapper import normalize_slice_type, quantize_a3_offset


class HeuristicLowerAgent:
    """
    Stabilizing rule-based lower A3 controller.

    Meaning:
        positive offset  -> handover harder -> stay / retain
        negative offset  -> handover easier -> go / offload

    The global PPO gives a per-slice bias for this serving gNB:
        b_i_s < 0 : offload this gNB-slice
        b_i_s = 0 : neutral
        b_i_s > 0 : retain users

    This heuristic converts that bias into neighbor-specific A3 offsets.

    Main safety idea:
        Even if serving gNB wants to offload, do not send traffic aggressively
        to a neighbor that is already loaded, crowded, or mobility-unstable.
    """

    def __init__(
        self,
        gnb_id,
        neighbor_ids,
        slice_types=("eMBB", "URLLC", "mMTC"),

        # UE pressure safety
        alpha_k=2.0,
        kappa_target=0.6,

        # Slice-load safety
        alpha_load=8.0,
        load_target=0.60,
        load_soft_margin=0.05,

        # Mobility safety
        alpha_hf=3.0,
        alpha_pp=1.5,

        # Offset smoothing
        eta=0.25,

        # Neutral behavior
        neutral_deadband=0.15,
        neutral_offset_db=0.0,

        # Safety clamp
        min_aggressive_offset=-4.0,
        max_offset_db=6.0,
    ):
        self.gnb_id = int(gnb_id)
        self.neighbor_ids = tuple(int(neighbor_id) for neighbor_id in neighbor_ids)
        self.slice_types = tuple(normalize_slice_type(slice_type) for slice_type in slice_types)

        self.alpha_k = float(alpha_k)
        self.kappa_target = float(kappa_target)

        self.alpha_load = float(alpha_load)
        self.load_target = float(load_target)
        self.load_soft_margin = float(load_soft_margin)

        self.alpha_hf = float(alpha_hf)
        self.alpha_pp = float(alpha_pp)

        self.eta = float(np.clip(eta, 0.0, 1.0))

        self.neutral_deadband = max(0.0, float(neutral_deadband))
        self.neutral_offset_db = float(np.clip(neutral_offset_db, -6.0, 6.0))

        self.min_aggressive_offset = float(np.clip(min_aggressive_offset, -6.0, 0.0))
        self.max_offset_db = float(np.clip(max_offset_db, 0.0, 6.0))

        self.previous_offsets: Dict[Tuple[int, int, str], float] = {
            (self.gnb_id, neighbor_id, slice_type): 0.0
            for neighbor_id in self.neighbor_ids
            for slice_type in self.slice_types
        }

    def reset(self):
        for key in self.previous_offsets:
            self.previous_offsets[key] = 0.0

    def _read_bias(self, bias_row, slice_idx: int, slice_type: str, neighbor_id: int | None = None) -> float:
        if isinstance(bias_row, Mapping):
            if neighbor_id is not None and (int(neighbor_id), slice_type) in bias_row:
                value = float(bias_row.get((int(neighbor_id), slice_type), 0.0))
            elif neighbor_id is not None and str(neighbor_id) in bias_row:
                nested = bias_row.get(str(neighbor_id), {})
                value = float(nested.get(slice_type, 0.0)) if isinstance(nested, Mapping) else 0.0
            elif neighbor_id is not None and int(neighbor_id) in bias_row:
                nested = bias_row.get(int(neighbor_id), {})
                value = float(nested.get(slice_type, 0.0)) if isinstance(nested, Mapping) else 0.0
            else:
                value = float(bias_row.get(slice_type, 0.0))
        else:
            row = np.asarray(bias_row, dtype=float).reshape(-1)
            value = float(row[slice_idx]) if slice_idx < row.size else 0.0

        return float(np.clip(value, -1.0, 1.0))

    def compute_offsets(
        self,
        bias_row: Mapping[str, float] | Sequence[float],
        ue_counts: Mapping[Tuple[int, str], int],
        kmax: Mapping[str, float],
        slice_loads: Mapping[Tuple[int, str], float] | None = None,
        handover_failure_ratios: Mapping[Tuple[int, int, str], float] | None = None,
        ping_pong_ratios: Mapping[Tuple[int, int, str], float] | None = None,
    ) -> Dict[Tuple[int, int, str], Dict[str, float]]:

        outputs: Dict[Tuple[int, int, str], Dict[str, float]] = {}

        slice_loads = slice_loads or {}
        handover_failure_ratios = handover_failure_ratios or {}
        ping_pong_ratios = ping_pong_ratios or {}

        for slice_idx, slice_type in enumerate(self.slice_types):
            source_load = float(slice_loads.get((self.gnb_id, slice_type), 0.0))

            for neighbor_id in self.neighbor_ids:
                neighbor_id = int(neighbor_id)
                key = (self.gnb_id, neighbor_id, slice_type)
                b_i_s = self._read_bias(bias_row, slice_idx, slice_type, neighbor_id)

                prev = float(self.previous_offsets.get(key, 0.0))

                neighbor_count = float(ue_counts.get((neighbor_id, slice_type), 0))
                denom = max(float(kmax.get(slice_type, 1.0)), 1e-9)
                kappa_j_s = neighbor_count / denom

                neighbor_load = float(slice_loads.get((neighbor_id, slice_type), 0.0))

                rhf = float(handover_failure_ratios.get(key, 0.0))
                rpp = float(ping_pong_ratios.get(key, 0.0))

                # 1. UE crowding safety
                crowding = max(kappa_j_s - self.kappa_target, 0.0)
                ue_safety = self.alpha_k * crowding

                # 2. Load safety
                # If neighbor is above target, push offset positive.
                # This prevents sending users to already crowded neighbors.
                neighbor_over_target = max(
                    neighbor_load - (self.load_target - self.load_soft_margin),
                    0.0,
                )
                load_safety = self.alpha_load * neighbor_over_target

                # 3. Mobility safety
                mobility_safety = (
                    self.alpha_hf * max(rhf, 0.0)
                    + self.alpha_pp * max(rpp, 0.0)
                )

                safety_term = ue_safety + load_safety + mobility_safety

                # 4. Main bias-to-offset rule
                if abs(b_i_s) <= self.neutral_deadband:
                    # True neutral: do not force handover, do not force retain.
                    raw_offset = self.neutral_offset_db + safety_term

                elif b_i_s < 0:
                    # Offload instruction.
                    # Negative base offset makes handover easier.
                    base_offset = 6.0 * b_i_s

                    # If the source is not actually above target anymore,
                    # reduce aggressiveness to avoid emptying the serving gNB.
                    source_excess = max(source_load - self.load_target, 0.0)
                    if source_excess <= 0.02:
                        base_offset = max(base_offset, -2.0)

                    raw_offset = base_offset + safety_term

                    # If neighbor is already above target, never use negative offset.
                    if neighbor_load >= self.load_target:
                        raw_offset = max(raw_offset, 0.0)

                    # Avoid too aggressive -6 unless the neighbor is clearly safe.
                    if neighbor_load > self.load_target - 0.15:
                        raw_offset = max(raw_offset, self.min_aggressive_offset)

                else:
                    # Retain instruction.
                    # Positive offset makes handover harder.
                    raw_offset = 6.0 * b_i_s + safety_term

                # 5. Smooth offset change
                proto_offset = (1.0 - self.eta) * raw_offset + self.eta * prev
                proto_offset = float(np.clip(proto_offset, -6.0, self.max_offset_db))

                # 6. Quantize to valid A3 set
                applied_offset = float(quantize_a3_offset(proto_offset))

                self.previous_offsets[key] = applied_offset

                outputs[key] = {
                    "proto_offset_db": proto_offset,
                    "applied_offset_db": applied_offset,
                    "bias": float(b_i_s),
                    "bias_term_db": float(0.0 if abs(b_i_s) <= self.neutral_deadband else 6.0 * b_i_s),
                    "safety_term_db": float(safety_term),
                    "ue_safety_db": float(ue_safety),
                    "load_safety_db": float(load_safety),
                    "mobility_safety_db": float(mobility_safety),
                    "source_load": float(source_load),
                    "neighbor_load": float(neighbor_load),
                    "neighbor_ue_fraction": float(kappa_j_s),
                    "handover_failure_ratio": float(rhf),
                    "ping_pong_ratio": float(rpp),
                }

        return outputs
