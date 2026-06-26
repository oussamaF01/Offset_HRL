import numpy as np

from multi_gnb_wrapper import MultiGNBWrapper
from scenario_creator import create_nodeb


def _make_three_gnb_env():
    rng = np.random.default_rng(12)
    gnbs = [
        create_nodeb(
            rng,
            0,
            slots_per_step=1,
            L1_level=False,
            node_id=0,
            node_x=-270.0,
            node_y=0.0,
            coverage_radius=500.0,
            carrier_id=0,
            n_prbs_override=200,
            wrapper_managed_mobile_slices=True,
        ),
        create_nodeb(
            rng,
            0,
            slots_per_step=1,
            L1_level=False,
            node_id=1,
            node_x=0.0,
            node_y=0.0,
            coverage_radius=500.0,
            carrier_id=0,
            n_prbs_override=200,
            wrapper_managed_mobile_slices=True,
        ),
        create_nodeb(
            rng,
            0,
            slots_per_step=1,
            L1_level=False,
            node_id=2,
            node_x=270.0,
            node_y=0.0,
            coverage_radius=500.0,
            carrier_id=0,
            n_prbs_override=200,
            wrapper_managed_mobile_slices=True,
        ),
    ]
    return MultiGNBWrapper(
        gnbs,
        step_dt=1e-3,
        mobility_dt=1e-3,
        radio_substeps=1,
        max_episode_steps=5,
        disconnect_sinr_db=-100.0,
    )


def _force_center_serving(env, x=135.0, y=0.0):
    ue_id = env.add_ue(x=x, y=y, vx=0.0, vy=0.0, slice_type="eMBB")
    ue = env.get_ue(ue_id)
    for gnb in env.gnbs:
        gnb.detach_ue(ue_id)
    env.gnbs[1].attach_ue(ue)
    ue.serving_gnb = 1
    ue.connected = True
    env._invalidate_metric_caches()
    return ue_id


def test_snr_uses_per_prb_received_power_against_per_prb_noise():
    env = _make_three_gnb_env()
    try:
        ue_id = _force_center_serving(env, x=135.0)
        metrics = env.get_ue_radio_metrics(ue_id)
        power_split_db = 10.0 * np.log10(env.gnbs[1].n_prbs)

        assert np.isclose(
            metrics["rx_power_total_dbm"] - metrics["rx_power_dbm"],
            power_split_db,
        )
        assert np.isclose(metrics["rx_power_total_dbm"], metrics["rsrp_dbm"])
        assert np.isclose(
            metrics["snr_db"],
            metrics["rx_power_dbm"] - metrics["noise_dbm"],
        )
        assert 18.0 <= metrics["snr_db"] <= 24.0
    finally:
        env.close()


def test_idle_same_carrier_neighbor_does_not_create_full_power_interference():
    env = _make_three_gnb_env()
    try:
        ue_id = _force_center_serving(env, x=135.0)

        idle_metrics = env.get_ue_radio_metrics(ue_id)
        assert idle_metrics["interference_dbm"] == -100.0
        assert idle_metrics["sinr_db"] > 18.0

        env._last_gnb_prb_activity[2] = 1.0
        env._invalidate_metric_caches()
        active_metrics = env.get_ue_radio_metrics(ue_id)
        assert active_metrics["interference_dbm"] > -95.0
        assert active_metrics["sinr_db"] < 1.0
    finally:
        env.close()
