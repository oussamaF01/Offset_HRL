import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np

from multi_gnb_wrapper import MultiGNBWrapper
from scenario_creator import create_nodeb


def _make_env():
    rng = np.random.default_rng(1)
    gnbs = [
        create_nodeb(
            rng,
            0,
            slots_per_step=1,
            L1_level=False,
            node_id=0,
            node_x=0.0,
            node_y=0.0,
            coverage_radius=900,
            n_prbs_override=100,
            wrapper_managed_mobile_slices=True,
        ),
        create_nodeb(
            rng,
            0,
            slots_per_step=1,
            L1_level=False,
            node_id=1,
            node_x=500.0,
            node_y=0.0,
            coverage_radius=900,
            n_prbs_override=100,
            wrapper_managed_mobile_slices=True,
        ),
        create_nodeb(
            rng,
            0,
            slots_per_step=1,
            L1_level=False,
            node_id=2,
            node_x=250.0,
            node_y=450.0,
            coverage_radius=900,
            n_prbs_override=100,
            wrapper_managed_mobile_slices=True,
        ),
    ]
    env = MultiGNBWrapper(
        gnbs,
        step_dt=1e-3,
        mobility_dt=1e-3,
        radio_substeps=1,
        max_episode_steps=3000,
        handover_ttt=3,
        a3_hysteresis_db=1.0,
        disconnect_sinr_db=-100.0,
        ue_traffic_profiles={
            "eMBB": {
                "traffic_model": "fixed_packet_cbr",
                "packet_size_bits": 3000.0,
                "bit_rate": 2_000_000.0,
            }
        },
    )
    ue_id = env.add_ue(x=50.0, y=0.0, vx=150.0, vy=0.0, slice_type="eMBB")

    ue = env.get_ue(ue_id)
    for gnb in gnbs:
        gnb.detach_ue(ue_id)
    gnbs[0].attach_ue(ue)
    ue.serving_gnb = 0
    ue.connected = True
    return env, ue_id


def _run_trace():
    env, ue_id = _make_env()
    rows = []
    for _ in range(3000):
        env.step(0)
        rows.append(env.get_ue_radio_metrics(ue_id))
    return env, rows


def test_a3_handover_happens_without_hrl_wrapper():
    env, rows = _run_trace()
    serving = [row["serving_gnb"] for row in rows]
    n_handovers = sum(1 for before, after in zip(serving, serving[1:]) if before != after)

    assert n_handovers == 1
    assert serving[-1] == 1
    assert env.get_handover_failure_ratios()
    assert env.get_ping_pong_ratios()
    assert all(value == 0.0 for value in env.get_handover_failure_ratios().values())
    assert all(value == 0.0 for value in env.get_ping_pong_ratios().values())


def test_rx_probability_stays_high_for_high_sinr():
    _env, rows = _run_trace()
    high_sinr_rxp = [
        row["rx_probability"]
        for row in rows
        if row["sinr_db"] > 15.0
    ]
    assert high_sinr_rxp
    assert float(np.mean(high_sinr_rxp)) >= 0.85


def test_handover_cooldown_blocks_immediate_return():
    env, ue_id = _make_env()
    env.handover_ttt = 1
    env.a3_handover_cooldown_steps = 5
    env.a3_min_residence_steps = 5
    ue = env.get_ue(ue_id)

    old_gnb = env._get_gnb_by_id(0)
    new_gnb = env._get_gnb_by_id(1)
    assert env._perform_handover(ue, old_gnb, new_gnb)
    env._last_ho[ue_id] = (1, 10)
    env._step_count = 10

    env._measure_rsrp = lambda gnb, _ue: -70.0 if int(gnb.id) == 0 else -90.0
    env.set_a3_offset(1, 0, "eMBB", -6.0)

    for tick in range(11, 15):
        env._step_count = tick
        assert env._evaluate_a3_handovers() == 0
        assert ue.serving_gnb == 1

    env._step_count = 15
    assert env._evaluate_a3_handovers() == 1
    assert ue.serving_gnb == 0


def test_minimum_residence_blocks_non_emergency_handover():
    env, ue_id = _make_env()
    env.handover_ttt = 1
    env.a3_handover_cooldown_steps = 5
    env.a3_min_residence_steps = 15
    env.a3_emergency_sinr_db = -5.0
    ue = env.get_ue(ue_id)

    old_gnb = env._get_gnb_by_id(0)
    new_gnb = env._get_gnb_by_id(1)
    assert env._perform_handover(ue, old_gnb, new_gnb)
    env._last_ho[ue_id] = (1, 10)
    env.set_a3_offset(1, 0, "eMBB", -6.0)
    env._measure_rsrp = lambda gnb, _ue: -70.0 if int(gnb.id) == 0 else -90.0

    original_link_metrics = env._compute_link_metrics
    env._compute_link_metrics = lambda gnb, _ue: {
        "sinr_db": 10.0,
        "rsrp_dbm": -70.0 if int(gnb.id) == 0 else -90.0,
    }
    env._step_count = 20
    assert env._evaluate_a3_handovers() == 0
    assert ue.serving_gnb == 1

    env._compute_link_metrics = lambda gnb, _ue: {
        "sinr_db": -10.0 if int(gnb.id) == 1 else 10.0,
        "rsrp_dbm": -70.0 if int(gnb.id) == 0 else -90.0,
    }
    assert env._evaluate_a3_handovers() == 1
    assert ue.serving_gnb == 0
    env._compute_link_metrics = original_link_metrics
