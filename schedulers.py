#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 3, 2021

@author: juanjosealcaraz
"""

import numpy as np

'''proportional fair scheduler for average cqi reports'''
class ProportionalFair:
    def __init__(
        self,
        mcs_codeset,
        granularity=2,
        slot_length=1e-3,
        window=50,
        sym_per_prb=158,
        mcs_codesets_by_slice=None,
    ):
        self.granularity = granularity
        self.mcs_codeset = mcs_codeset
        self.mcs_codesets_by_slice = {
            self._slice_key(slice_type): codeset
            for slice_type, codeset in dict(mcs_codesets_by_slice or {}).items()
        }
        self.b = 1/window
        self.a = 1 - self.b
        self.sym_per_prb = sym_per_prb
        self.slot_length = slot_length

    @staticmethod
    def _slice_key(slice_type):
        return str(slice_type or "").replace("_", "").replace("-", "").upper()

    def _codeset_for_ue(self, ue):
        return self.mcs_codesets_by_slice.get(
            self._slice_key(getattr(ue, "slice_type", "")),
            self.mcs_codeset,
        )

    def allocate(self, ues, n_prb, error_bound = 0.1):
        '''
        Updates the following variables of the ues:
        - ue.bits : assigned bits in this subframe
        - ue.prbs : assigned prbs in this subframe
        - ue.p : reception probability

        MCS is first selected on wideband ue.e_snr for the PF metric.
        After allocation, MCS and reception probability are re-selected on
        the SNR vector of the PRBs assigned to each UE so both calculations
        use the same realized channel sample.
        '''
        # create auxiliary data structures
        n_ues = len(ues)
        ue_rbs = np.zeros(n_ues, dtype = int)
        ue_prb_indices = [[] for _ in range(n_ues)]
        ue_mcs = np.zeros(n_ues, dtype = int)
        ue_queue = np.zeros(n_ues, dtype = int)
        ue_rate = np.zeros(n_ues, dtype = int)
        ue_bits = np.zeros(n_ues, dtype = int)
        ue_th = np.zeros(n_ues)
        ue_codesets = []

        # extract ue information
        for i, ue in enumerate(ues):
            codeset = self._codeset_for_ue(ue)
            ue_codesets.append(codeset)
            ue_th[i] = max(ue.th, 1) # to avoid division by zero
            ue_queue[i] = ue.queue
            # determine the mcs given the objective and the estimated snr
            ue_mcs[i], bits_per_sym = codeset.mcs_rate_vs_error(ue.e_snr, error_bound)
            # achievable rate for the ue
            ue_rate[i] = self.sym_per_prb * bits_per_sym
        
        # loop over the resources
        for r in range(0, n_prb, self.granularity):
            # prbs to be allocated in this iteration
            prbs = min(n_prb - r, self.granularity)

            # selected user for this resource (remove users without data)
            scores = ue_rate * (ue_queue > 0) / ue_th
            index = int(np.argmax(scores))
            # Do not burn the rest of the PRB budget after every queue has
            # drained. Besides avoiding wasted allocations, this removes many
            # no-op PF iterations in light-load training scenarios.
            if scores[index] <= 0:
                break
            
            # assign the resource to this ue
            ue_rbs[index] += prbs
            ue_prb_indices[index].extend(range(r, r + prbs))

            # update queue and throughput of this user
            tx_bits = min(prbs * ue_rate[index], ue_queue[index])
            ue_queue[index] -= tx_bits
            ue_bits[index] += tx_bits

            # update the estimated throughput with current allocation
            ue_th[index] = self.a * ue_th[index] + self.b * ue_bits[index] / self.slot_length
    
        # update ues
        for i, ue in enumerate(ues):
            codeset = ue_codesets[i]
            prbs = int(ue_rbs[i])
            ue.prbs = prbs

            if prbs > 0:
                snr_full = np.asarray(ue.snr, dtype=float).reshape(-1)
                assigned_indices = np.asarray(ue_prb_indices[i], dtype=int)
                valid_indices = assigned_indices[
                    (assigned_indices >= 0) & (assigned_indices < snr_full.size)
                ]
                snr_values = snr_full[valid_indices]
                if snr_values.size == 0 and snr_full.size > 0:
                    snr_values = snr_full[:prbs]
                if snr_values.size == 0:
                    snr_values = np.asarray([float(getattr(ue, "e_snr", 0.0))], dtype=float)

                target_rx_prob = 1.0 - float(error_bound)
                mcs_realized = 0
                effective_snr_by_modulation = {}
                for candidate_mcs in range(codeset.n_mcs):
                    modulation = str(codeset.modulation[candidate_mcs])
                    effective_snr = effective_snr_by_modulation.get(modulation)
                    if effective_snr is None:
                        effective_snr = codeset.effective_snr(
                            candidate_mcs,
                            snr_values,
                        )
                        effective_snr_by_modulation[modulation] = effective_snr
                    if (
                        codeset.estimate_rx_prob(candidate_mcs, effective_snr)
                        < target_rx_prob
                    ):
                        break
                    mcs_realized = candidate_mcs
                bits_per_sym = codeset.nominal_rate(mcs_realized)
                realized_rate = int(self.sym_per_prb * bits_per_sym)
                original_queue = int(ue_queue[i]) + int(ue_bits[i])
                realized_bits = min(prbs * realized_rate, original_queue)

                ue.mcs = int(mcs_realized)
                ue.bits = int(realized_bits)
                ue.spectral_efficiency = float(codeset.nominal_rate(mcs_realized))
                realized_modulation = str(codeset.modulation[mcs_realized])
                effective_snr = effective_snr_by_modulation.get(realized_modulation)
                if effective_snr is None:
                    effective_snr = codeset.effective_snr(
                        mcs_realized,
                        snr_values,
                    )
                ue.effective_sinr_db = float(effective_snr)
                ue.p = codeset.estimate_rx_prob(mcs_realized, effective_snr)
                ue.mcs_codeset_name = (
                    "URLLC" if self._slice_key(getattr(ue, "slice_type", "")) == "URLLC"
                    and "URLLC" in self.mcs_codesets_by_slice
                    else "default"
                )
                if realized_bits > 0 and realized_rate > 0:
                    ue.useful_prbs = int(min(prbs, np.ceil(realized_bits / float(realized_rate))))
                else:
                    ue.useful_prbs = 0
            else:
                ue.mcs = int(ue_mcs[i])
                ue.bits = int(ue_bits[i])
                ue.spectral_efficiency = float(codeset.nominal_rate(ue_mcs[i]))
                ue.mcs_codeset_name = (
                    "URLLC" if self._slice_key(getattr(ue, "slice_type", "")) == "URLLC"
                    and "URLLC" in self.mcs_codesets_by_slice
                    else "default"
                )
                ue.p = 0
                ue.effective_sinr_db = float("nan")
                ue.useful_prbs = 0
            ue.wasted_prbs = int(max(prbs - ue.useful_prbs, 0))
