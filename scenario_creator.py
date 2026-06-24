#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import gymnasium as gym
import numpy as np
from itertools import count
from typing import Dict, List, Optional

from node_b import NodeB
from slice_l1 import SliceL1eMBB, SliceL1mMTC, SliceL1URLLC
from slice_ran import SliceRANmMTC, SliceRANeMBB, SliceRANURLC
from schedulers import ProportionalFair
from channel_models import SINRSelectiveFading, MCSCodeset
from kbrl_control import KBRL_Control, Learner
from algorithms.kernel import GaussianKernel
from algorithms.projectron import SVvariable, Projectron

try:
    from multi_gnb_wrapper import MultiGNBWrapper
except ImportError:
    MultiGNBWrapper = None


scenario_1 = {"n_prbs": 200, "n_embb": 5, "n_mmtc": 0}
scenario_2 = {"n_prbs": 150, "n_embb": 3, "n_mmtc": 2}
scenario_3 = {"n_prbs": 100, "n_embb": 1, "n_mmtc": 4}
scenario_4 = {"n_prbs": 70, "n_embb": 1, "n_mmtc": 1}
scenario_5 = {"n_prbs": 100, "n_embb": 1, "n_mmtc": 1, "n_urllc": 1}

scenarios = [scenario_1, scenario_2, scenario_3, scenario_4, scenario_5]


CBR_description = {
    "lambda": 2.0 / 60.0,
    "t_mean": 30.0,
    "bit_rate": 500000,
}

VBR_description = {
    "lambda": 5.0 / 60.0,
    "t_mean": 30.0,
    "p_size": 1000,
    "b_size": 500,
    "b_rate": 1,
}

SLA_embb = {
    "cbr_th": 10e6,
    "cbr_prb": 20,
    "cbr_queue": 10e4,
    "vbr_th": 15e6,
    "vbr_prb": 30,
    "vbr_queue": 15e4,
}

state_variables_embb = [
    "cbr_traffic", "cbr_th", "cbr_prb", "cbr_queue", "cbr_snr",
    "vbr_traffic", "vbr_th", "vbr_prb", "vbr_queue", "vbr_snr",
]


MTC_description = {
    "n_devices": 1000,
    "repetition_set": [2, 4, 8, 16, 32, 64, 128],
    "period_set": [1000, 50000, 10000, 15000, 20000, 25000, 50000, 100000],
}

state_variables_mmtc = ["devices", "avg_rep", "delay"]

SLA_mmtc = {
    "delay": 300,
}


URLLC_CBR_description = {
    "lambda": 2.0 / 60.0,
    "t_mean": 30.0,
    "bit_rate": 500000,
}

URLLC_VBR_description = {
    "lambda": 5.0 / 60.0,
    "t_mean": 30.0,
    "p_size": 1000,
    "b_size": 500,
    "b_rate": 1,
}

SLA_urllc = {
    "cbr_th": 10e6,
    "cbr_prb": 20,
    "cbr_queue": 5e4,
    "vbr_th": 15e6,
    "vbr_prb": 30,
    "vbr_queue": 10e4,
}

state_variables_urllc = [
    "cbr_traffic", "cbr_th", "cbr_prb", "cbr_queue", "cbr_snr",
    "vbr_traffic", "vbr_th", "vbr_prb", "vbr_queue", "vbr_snr",
]


def _get_scenario_config(n: int) -> Dict:
    return scenarios[n]


def _build_norm_constants(slots_per_step: int, slot_length: float):
    time_per_step = slots_per_step * slot_length

    norm_const_embb = {
        "cbr_traffic": 5e6 * time_per_step,
        "cbr_th": 10e6 * time_per_step,
        "cbr_prb": 25 * slots_per_step,
        "cbr_queue": 10e4 * slots_per_step,
        "cbr_snr": 35 * slots_per_step,
        "vbr_traffic": 5e6 * time_per_step,
        "vbr_th": 10e6 * time_per_step,
        "vbr_prb": 35 * slots_per_step,
        "vbr_queue": 10e4 * slots_per_step,
        "vbr_snr": 35 * slots_per_step,
    }

    norm_const_mmtc = {
        "devices": 100 * slots_per_step,
        "avg_rep": 100 * slots_per_step,
        "delay": 100 * slots_per_step,
    }

    norm_const_urllc = {
        "cbr_traffic": 2e6 * time_per_step,
        "cbr_th": 5e6 * time_per_step,
        "cbr_prb": 15 * slots_per_step,
        "cbr_queue": 5e3 * slots_per_step,
        "cbr_snr": 35 * slots_per_step,
        "vbr_traffic": 2e6 * time_per_step,
        "vbr_th": 7e6 * time_per_step,
        "vbr_prb": 20 * slots_per_step,
        "vbr_queue": 5e3 * slots_per_step,
        "vbr_snr": 35 * slots_per_step,
    }

    return norm_const_embb, norm_const_mmtc, norm_const_urllc


def _build_slices_l1(
    rng,
    scenario_idx: int,
    slots_per_step: int,
    propagation_type: str,
    L1_level: bool,
    slot_length: float,
    wrapper_managed_mobile_slices: bool = False,
):
    sc = _get_scenario_config(scenario_idx)
    n_prbs = sc["n_prbs"]
    n_embb = sc["n_embb"]
    n_mmtc = sc["n_mmtc"]
    n_urllc = sc.get("n_urllc", 0)

    norm_const_embb, norm_const_mmtc, norm_const_urllc = _build_norm_constants(
        slots_per_step=slots_per_step,
        slot_length=slot_length,
    )

    def new_slice_mmtc(id_, rng_):
        return SliceRANmMTC(
            rng_,
            id_,
            SLA_mmtc,
            MTC_description,
            state_variables_mmtc,
            norm_const_mmtc,
            slots_per_step,
        )

    def new_slice_embb(id_, rng_, user_counter_):
        return SliceRANeMBB(
            rng_,
            user_counter_,
            id_,
            SLA_embb,
            CBR_description,
            VBR_description,
            state_variables_embb,
            norm_const_embb,
            slots_per_step,
            slot_length=slot_length,
        )

    def new_slice_urllc(id_, rng_, user_counter_):
        return SliceRANURLC(
            rng_,
            user_counter_,
            id_,
            SLA_urllc,
            URLLC_CBR_description,
            URLLC_VBR_description,
            state_variables_urllc,
            norm_const_urllc,
            slots_per_step,
            slot_length=slot_length,
        )

    snr_generator = SINRSelectiveFading(rng, propagation_type, n_prbs=n_prbs)
    mcs_codeset = MCSCodeset()
    scheduler = ProportionalFair(mcs_codeset)
    user_counter = count()

    slices_l1 = []

    def mark_mobile_l1(l1):
        if wrapper_managed_mobile_slices:
            l1.external_ues = True
            l1.wrapper_managed = True
        return l1

    if L1_level:
        for id_ in range(n_embb):
            slices_ran_embb = [new_slice_embb(id_, rng, user_counter)]
            slices_l1.append(mark_mobile_l1(SliceL1eMBB(rng, snr_generator, 20, slices_ran_embb, scheduler)))

        for id_ in range(n_mmtc):
            slices_ran_mmtc = [new_slice_mmtc(id_, rng)]
            slices_l1.append(SliceL1mMTC(5, slices_ran_mmtc))

        for id_ in range(n_urllc):
            slices_ran_urllc = [new_slice_urllc(id_, rng, user_counter)]
            slices_l1.append(mark_mobile_l1(SliceL1URLLC(rng, snr_generator, 15, slices_ran_urllc, scheduler)))
    else:
        if n_embb > 0:
            slices_ran_embb = [new_slice_embb(id_, rng, user_counter) for id_ in range(n_embb)]
            slices_l1.append(mark_mobile_l1(SliceL1eMBB(rng, snr_generator, 20, slices_ran_embb, scheduler)))

        if n_mmtc > 0:
            slices_ran_mmtc = [new_slice_mmtc(id_, rng) for id_ in range(n_mmtc)]
            slices_l1.append(SliceL1mMTC(5, slices_ran_mmtc))

        if n_urllc > 0:
            slices_ran_urllc = [new_slice_urllc(id_, rng, user_counter) for id_ in range(n_urllc)]
            slices_l1.append(mark_mobile_l1(SliceL1URLLC(rng, snr_generator, 15, slices_ran_urllc, scheduler)))

    return slices_l1, n_prbs


def create_nodeb(
    rng,
    n,
    slots_per_step=50,
    propagation_type="macro_cell_urban_2GHz",
    L1_level=True,
    node_id=0,
    node_x=0.0,
    node_y=0.0,
    coverage_radius=500,
    slot_length=1e-3,
    carrier_id=0,
    center_frequency_hz=3.5e9,
    bandwidth_hz=20e6,
    tx_power_dbm=30.0,
    noise_figure_db=7.0,
    n_prbs_override=None,
    wrapper_managed_mobile_slices: bool = False,
):
    slices_l1, n_prbs = _build_slices_l1(
        rng=rng,
        scenario_idx=n,
        slots_per_step=slots_per_step,
        propagation_type=propagation_type,
        L1_level=L1_level,
        slot_length=slot_length,
        wrapper_managed_mobile_slices=wrapper_managed_mobile_slices,
    )

    if n_prbs_override is not None:
        n_prbs = int(n_prbs_override)

    return NodeB(
        id=node_id,
        x=node_x,
        y=node_y,
        slices_l1=slices_l1,
        slots_per_step=slots_per_step,
        n_prbs=n_prbs,
        coverage_radius=coverage_radius,
        slot_length=slot_length,
        carrier_id=carrier_id,
        center_frequency_hz=center_frequency_hz,
        bandwidth_hz=bandwidth_hz,
        tx_power_dbm=tx_power_dbm,
        noise_figure_db=noise_figure_db,
    )


def default_gnb_configs(n_gnbs: int, coverage_radius: float = 500.0, spacing: Optional[float] = None):
    if spacing is None:
        spacing = 1.5 * coverage_radius

    if n_gnbs <= 0:
        raise ValueError("n_gnbs must be >= 1")

    if n_gnbs == 1:
        positions = [(0.0, 0.0)]
    elif n_gnbs == 2:
        positions = [(0.0, 0.0), (spacing, 0.0)]
    elif n_gnbs == 3:
        h = 0.8660254037844386 * spacing
        positions = [(0.0, 0.0), (spacing, 0.0), (0.5 * spacing, h)]
    else:
        positions = [(i * spacing, 0.0) for i in range(n_gnbs)]

    return [
        {
            "id": i,
            "x": float(x),
            "y": float(y),
            "coverage_radius": coverage_radius,
            "carrier_id": 0,
            "center_frequency_hz": 3.5e9,
            "bandwidth_hz": 20e6,
            "tx_power_dbm": 30.0,
            "noise_figure_db": 7.0,
        }
        for i, (x, y) in enumerate(positions)
    ]


def create_multignb_env(
    rng,
    n,
    slots_per_step=50,
    propagation_type="macro_cell_urban_2GHz",
    L1_level=True,
    slot_length=1e-3,
    gnb_configs: Optional[List[Dict]] = None,
    n_gnbs: Optional[int] = None,
    coverage_radius=500,
    handover_hysteresis: float = 0.05,
    handover_ttt: int = 3,
    verbose: bool = False,
    step_dt: float = 1e-3,
    mobility_dt: Optional[float] = None,
    radio_substeps: int = 1,
    pf_averaging_window_s: float = 0.25,
    max_episode_steps: int = 100,
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
    ue_traffic_profiles: Optional[Dict] = None,
    default_traffic_model: str = "fixed_packet_cbr",
    slice_prb_budgets: Optional[Dict[str, int]] = None,
    max_prbs_per_ue: Optional[int] = 20,
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
):
    if MultiGNBWrapper is None:
        raise ImportError(
            "MultiGNBWrapper could not be imported. "
            "Make sure multi_gnb_wrapper.py is available in the project path."
        )

    if gnb_configs is None:
        n_gnbs = 2 if n_gnbs is None else n_gnbs
        gnb_configs = default_gnb_configs(n_gnbs=n_gnbs, coverage_radius=coverage_radius)

    gnb_list = []

    for idx, cfg in enumerate(gnb_configs):
        node = create_nodeb(
            rng=rng,
            n=n,
            slots_per_step=slots_per_step,
            propagation_type=propagation_type,
            L1_level=L1_level,
            node_id=cfg.get("id", idx),
            node_x=cfg.get("x", 0.0),
            node_y=cfg.get("y", 0.0),
            coverage_radius=cfg.get("coverage_radius", coverage_radius),
            slot_length=slot_length,
            carrier_id=cfg.get("carrier_id", 0),
            center_frequency_hz=cfg.get("center_frequency_hz", 3.5e9),
            bandwidth_hz=cfg.get("bandwidth_hz", 20e6),
            tx_power_dbm=cfg.get("tx_power_dbm", 30.0),
            noise_figure_db=cfg.get("noise_figure_db", 7.0),
            n_prbs_override=cfg.get("n_prbs", None),
            wrapper_managed_mobile_slices=True,
        )
        gnb_list.append(node)

    return MultiGNBWrapper(
        gnb_list=gnb_list,
        handover_hysteresis=handover_hysteresis,
        handover_ttt=handover_ttt,
        verbose=verbose,
        step_dt=step_dt,
        mobility_dt=mobility_dt,
        radio_substeps=radio_substeps,
        pf_averaging_window_s=pf_averaging_window_s,
        max_episode_steps=max_episode_steps,
        degradation_zones=degradation_zones,
        use_sumo_mobility=use_sumo_mobility,
        sumo_config_path=sumo_config_path,
        sumo_binary=sumo_binary,
        sumo_port=sumo_port,
        sumo_auto_add_ues=sumo_auto_add_ues,
        sumo_vehicle_slice_type=sumo_vehicle_slice_type,
        sumo_person_slice_type=sumo_person_slice_type,
        sumo_vehicle_slice_mix=sumo_vehicle_slice_mix,
        sumo_person_slice_mix=sumo_person_slice_mix,
        ue_traffic_profiles=ue_traffic_profiles,
        default_traffic_model=default_traffic_model,
        slice_prb_budgets=slice_prb_budgets,
        max_prbs_per_ue=max_prbs_per_ue,
        a3_history_window_s=a3_history_window_s,
        a3_pingpong_threshold_s=a3_pingpong_threshold_s,
        a3_handover_cooldown_s=a3_handover_cooldown_s,
        a3_min_residence_s=a3_min_residence_s,
        a3_pingpong_guard_s=a3_pingpong_guard_s,
        a3_emergency_sinr_db=a3_emergency_sinr_db,
        max_handovers_per_step=max_handovers_per_step,
        max_handovers_per_ue_episode=max_handovers_per_ue_episode,
        max_handovers_per_episode=max_handovers_per_episode,
        safe_admission_enabled=safe_admission_enabled,
        safe_admission_load_limits=safe_admission_load_limits,
    )


def create_env(
    rng,
    n,
    slots_per_step=50,
    propagation_type="macro_cell_urban_2GHz",
    L1_level=True,
    penalty=100,
    node_id=0,
    node_x=0.0,
    node_y=0.0,
    coverage_radius=500,
    slot_length=1e-3,
    multi_gnb: bool = False,
    gnb_configs: Optional[List[Dict]] = None,
    n_gnbs: Optional[int] = None,
    handover_hysteresis: float = 0.05,
    handover_ttt: int = 3,
    verbose: bool = False,
    step_dt: float = 1e-3,
    mobility_dt: Optional[float] = None,
    radio_substeps: int = 1,
    pf_averaging_window_s: float = 0.25,
    max_episode_steps: int = 100,
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
    ue_traffic_profiles: Optional[Dict] = None,
    default_traffic_model: str = "fixed_packet_cbr",
    slice_prb_budgets: Optional[Dict[str, int]] = None,
    max_prbs_per_ue: Optional[int] = 20,
    a3_handover_cooldown_s: float = 5.0,
    a3_min_residence_s: float = 15.0,
    a3_emergency_sinr_db: float = -5.0,
):
    if multi_gnb:
        return create_multignb_env(
            rng=rng,
            n=n,
            slots_per_step=slots_per_step,
            propagation_type=propagation_type,
            L1_level=L1_level,
            slot_length=slot_length,
            gnb_configs=gnb_configs,
            n_gnbs=n_gnbs,
            coverage_radius=coverage_radius,
            handover_hysteresis=handover_hysteresis,
            handover_ttt=handover_ttt,
            verbose=verbose,
            step_dt=step_dt,
            mobility_dt=mobility_dt,
            radio_substeps=radio_substeps,
            pf_averaging_window_s=pf_averaging_window_s,
            max_episode_steps=max_episode_steps,
            degradation_zones=degradation_zones,
            use_sumo_mobility=use_sumo_mobility,
            sumo_config_path=sumo_config_path,
            sumo_binary=sumo_binary,
            sumo_port=sumo_port,
            sumo_auto_add_ues=sumo_auto_add_ues,
            sumo_vehicle_slice_type=sumo_vehicle_slice_type,
            sumo_person_slice_type=sumo_person_slice_type,
            sumo_vehicle_slice_mix=sumo_vehicle_slice_mix,
            sumo_person_slice_mix=sumo_person_slice_mix,
            ue_traffic_profiles=ue_traffic_profiles,
            default_traffic_model=default_traffic_model,
            slice_prb_budgets=slice_prb_budgets,
            max_prbs_per_ue=max_prbs_per_ue,
            a3_handover_cooldown_s=a3_handover_cooldown_s,
            a3_min_residence_s=a3_min_residence_s,
            a3_emergency_sinr_db=a3_emergency_sinr_db,
        )

    node = create_nodeb(
        rng=rng,
        n=n,
        slots_per_step=slots_per_step,
        propagation_type=propagation_type,
        L1_level=L1_level,
        node_id=node_id,
        node_x=node_x,
        node_y=node_y,
        coverage_radius=coverage_radius,
        slot_length=slot_length,
    )

    return gym.make("gym_ran_slice:RanSlice-v1", node_b=node, penalty=penalty)


alfa = 0.05

embb_sec = (2, 8)
embb_a = (4, 20)
mmtc_sec = (1, 4)
mmtc_a = (2, 10)
urllc_sec = (1, 4)
urllc_a = (3, 15)


def create_kbrl_agent(rng, n, accuracy_range=[0.99, 0.999]):
    sc = scenarios[n]
    n_prbs = sc["n_prbs"]
    n_embb = sc["n_embb"]
    n_mmtc = sc["n_mmtc"]
    n_urllc = sc.get("n_urllc", 0)
    embb_dim = len(state_variables_embb)
    mmtc_dim = len(state_variables_mmtc)
    urllc_dim = len(state_variables_urllc)

    learners = []
    i = 0

    for _ in range(n_embb):
        sv = SVvariable()
        kernel = GaussianKernel(sv, 1)
        algorithm = Projectron(kernel)
        initial_action = rng.integers(embb_a[0], embb_a[1])
        sec = rng.integers(embb_sec[0], embb_sec[1])
        learners.append(Learner(algorithm, slice(i, i + embb_dim), initial_action, sec))
        i += embb_dim

    for _ in range(n_mmtc):
        sv = SVvariable()
        kernel = GaussianKernel(sv, 1)
        algorithm = Projectron(kernel)
        initial_action = rng.integers(mmtc_a[0], mmtc_a[1])
        sec = rng.integers(mmtc_sec[0], mmtc_sec[1])
        learners.append(Learner(algorithm, slice(i, i + mmtc_dim), initial_action, sec))
        i += mmtc_dim

    for _ in range(n_urllc):
        sv = SVvariable()
        kernel = GaussianKernel(sv, 1)
        algorithm = Projectron(kernel)
        initial_action = rng.integers(urllc_a[0], urllc_a[1])
        sec = rng.integers(urllc_sec[0], urllc_sec[1])
        learners.append(Learner(algorithm, slice(i, i + urllc_dim), initial_action, sec))
        i += urllc_dim

    return KBRL_Control(learners, n_prbs, alfa=alfa, accuracy_range=accuracy_range)
