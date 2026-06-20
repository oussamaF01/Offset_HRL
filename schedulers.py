#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 3, 2021

@author: juanjosealcaraz
"""

import numpy as np

'''proportional fair scheduler for average cqi reports'''
class ProportionalFair:
    def __init__(self, mcs_codeset, granularity = 2, slot_length = 1e-3, window = 50, sym_per_prb = 158):
        self.granularity = granularity
        self.mcs_codeset = mcs_codeset
        self.b = 1/window
        self.a = 1 - self.b
        self.sym_per_prb = sym_per_prb
        self.slot_length = slot_length

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
        ue_mcs = np.zeros(n_ues, dtype = int)
        ue_queue = np.zeros(n_ues, dtype = int)
        ue_rate = np.zeros(n_ues, dtype = int)
        ue_bits = np.zeros(n_ues, dtype = int)
        ue_th = np.zeros(n_ues)

        # extract ue information
        for i, ue in enumerate(ues):
            ue_th[i] = max(ue.th, 1) # to avoid division by zero
            ue_queue[i] = ue.queue
            # determine the mcs given the objective and the estimated snr
            ue_mcs[i], bits_per_sym = self.mcs_codeset.mcs_rate_vs_error(ue.e_snr, error_bound)
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

            # update queue and throughput of this user
            tx_bits = min(prbs * ue_rate[index], ue_queue[index])
            ue_queue[index] -= tx_bits
            ue_bits[index] += tx_bits

            # update the estimated throughput with current allocation
            ue_th[index] = self.a * ue_th[index] + self.b * ue_bits[index] / self.slot_length
    
        # update ues
        prb_i = 0
        for i, ue in enumerate(ues):
            prbs = int(ue_rbs[i])
            ue.prbs = prbs

            if prbs > 0:
                snr_full = np.asarray(ue.snr, dtype=float).reshape(-1)
                snr_values = snr_full[prb_i: prb_i + prbs]
                if snr_values.size == 0 and snr_full.size > 0:
                    snr_values = snr_full[:prbs]
                if snr_values.size == 0:
                    snr_values = np.asarray([float(getattr(ue, "e_snr", 0.0))], dtype=float)

                target_rx_prob = 1.0 - float(error_bound)
                mcs_realized = 0
                effective_snr_by_modulation = {}
                for candidate_mcs in range(self.mcs_codeset.n_mcs):
                    modulation = str(self.mcs_codeset.modulation[candidate_mcs])
                    effective_snr = effective_snr_by_modulation.get(modulation)
                    if effective_snr is None:
                        effective_snr = self.mcs_codeset.effective_snr(
                            candidate_mcs,
                            snr_values,
                        )
                        effective_snr_by_modulation[modulation] = effective_snr
                    if (
                        self.mcs_codeset.estimate_rx_prob(candidate_mcs, effective_snr)
                        < target_rx_prob
                    ):
                        break
                    mcs_realized = candidate_mcs
                bits_per_sym = self.mcs_codeset.nominal_rate(mcs_realized)
                realized_rate = int(self.sym_per_prb * bits_per_sym)
                original_queue = int(ue_queue[i]) + int(ue_bits[i])
                realized_bits = min(prbs * realized_rate, original_queue)

                ue.mcs = int(mcs_realized)
                ue.bits = int(realized_bits)
                ue.spectral_efficiency = float(self.mcs_codeset.nominal_rate(mcs_realized))
                realized_modulation = str(self.mcs_codeset.modulation[mcs_realized])
                effective_snr = effective_snr_by_modulation.get(realized_modulation)
                if effective_snr is None:
                    effective_snr = self.mcs_codeset.effective_snr(
                        mcs_realized,
                        snr_values,
                    )
                ue.p = self.mcs_codeset.estimate_rx_prob(mcs_realized, effective_snr)
                if realized_bits > 0 and realized_rate > 0:
                    ue.useful_prbs = int(min(prbs, np.ceil(realized_bits / float(realized_rate))))
                else:
                    ue.useful_prbs = 0
            else:
                ue.mcs = int(ue_mcs[i])
                ue.bits = int(ue_bits[i])
                ue.spectral_efficiency = float(self.mcs_codeset.nominal_rate(ue_mcs[i]))
                ue.p = 0
                ue.useful_prbs = 0
            ue.wasted_prbs = int(max(prbs - ue.useful_prbs, 0))
            prb_i += prbs
