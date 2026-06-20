import numpy as np

from multi_gnb_wrapper import MultiGNBWrapper
from scenario_creator import create_nodeb
from slice_ran import Packet, UE, CBR
from traffic_generators import CbrSource


def _make_env():
    rng = np.random.default_rng(4)
    gnb = create_nodeb(
        rng,
        4,
        slots_per_step=1,
        L1_level=False,
        node_id=0,
        node_x=0.0,
        node_y=0.0,
        coverage_radius=500.0,
        n_prbs_override=100,
        wrapper_managed_mobile_slices=True,
    )
    return MultiGNBWrapper([gnb], radio_substeps=1, max_episode_steps=5)


def test_window_sla_definitions_are_slice_specific():
    env = _make_env()
    try:
        env.begin_sla_window()
        env._sla_window_stats[(0, "eMBB")].update(
            offered_bits=1000.0,
            delivered_bits=700.0,
            generated_packets=10.0,
        )
        env._sla_window_stats[(0, "URLLC")].update(
            generated_packets=100.0,
            failed_packets=2.0,
        )
        env._sla_window_stats[(0, "mMTC")].update(
            generated_packets=100.0,
            failed_packets=1.0,
        )

        metrics = env.get_slice_sla_metrics()
        assert metrics[(0, "eMBB")]["measured_ratio"] == 0.7
        assert np.isclose(metrics[(0, "eMBB")]["severity"], 0.125)
        assert metrics[(0, "eMBB")]["violation"] == 1.0

        assert metrics[(0, "URLLC")]["severity"] == 0.02
        assert metrics[(0, "URLLC")]["violation"] == 1.0

        assert metrics[(0, "mMTC")]["severity"] == 0.01
        assert metrics[(0, "mMTC")]["violation"] == 0.0
    finally:
        env.close()


def test_urllc_packet_is_counted_failed_only_once_after_deadline():
    ue = UE(
        id=0,
        slice_ran_id=0,
        traffic_source=CbrSource(bit_rate=0.0, step_length=1e-3),
        type=CBR,
        slot_length=1e-3,
        slice_type="URLLC",
    )
    ue.packet_queue.append(Packet(bits=100, arrival_step=0, arrival_time_s=0.0, packet_id=7))
    ue.queue = 100
    ue._time_s = 0.011
    ue.update_sla_expirations()
    ue.update_sla_expirations()
    assert ue.total_packets_expired == 1
    assert ue.total_packets_failed_sla == 1


def test_begin_sla_window_clears_previous_measurements():
    env = _make_env()
    try:
        env._sla_window_stats[(0, "eMBB")]["offered_bits"] = 500.0
        env.begin_sla_window()
        assert env._sla_window_stats[(0, "eMBB")]["offered_bits"] == 0.0
        assert env.get_slice_sla_flags()[(0, "eMBB")] == 0.0
    finally:
        env.close()
