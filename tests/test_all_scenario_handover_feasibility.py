import numpy as np

from global_ppo_3gnb_env import GlobalPPO3GNBEnv
from upper_agent_training_scenarios import (
    CENTER_GAP_GNB_CONFIGS,
    get_upper_training_scenarios,
)


def _oracle_directional_action(env, loads):
    means = loads.mean(axis=0)
    action = np.zeros((3, 2, 3), dtype=np.float32)
    for source in range(3):
        for slice_idx in range(3):
            if loads[source, slice_idx] <= means[slice_idx] + 1e-9:
                continue
            deficits = [
                max(means[slice_idx] - loads[target, slice_idx], 0.0)
                for target in env.neighbors[source]
            ]
            total_deficit = sum(deficits)
            if total_deficit <= 0.0:
                continue
            for slot, target in enumerate(env.neighbors[source]):
                radio_feasible = any(
                    int(ue.serving_gnb) == source
                    and env.slice_types.index(str(ue.slice_type)) == slice_idx
                    and env.base_env._is_in_coverage(
                        env.base_env._get_gnb_by_id(target), ue
                    )
                    and (
                        env.base_env._measure_rsrp(
                            env.base_env._get_gnb_by_id(target), ue
                        )
                        - env.base_env._measure_rsrp(
                            env.base_env._get_gnb_by_id(source), ue
                        )
                    ) > -5.0
                    for ue in env.base_env.get_all_ues()
                )
                if radio_feasible and deficits[slot] > 0.0:
                    action[source, slot, slice_idx] = (
                        -deficits[slot] / total_deficit
                    )
    return action


def test_every_retained_scenario_has_bounded_initial_persistent_demand_load():
    for scenario in get_upper_training_scenarios("all"):
        env = GlobalPPO3GNBEnv(
            seed=2,
            scenario_mode="curriculum",
            training_scenarios=scenario.name,
            scenario_selection="cycle",
            gnb_configs=CENTER_GAP_GNB_CONFIGS["medium_270m"],
            upper_window_seconds=1.0,
            local_steps_per_global=10,
            radio_substeps=20,
            terminal_reward_only=False,
            safe_admission_enabled=True,
        )
        try:
            _obs, reset_info = env.reset(seed=2)
            start_loads = np.asarray(reset_info["load_matrix"], dtype=float)
            action = _oracle_directional_action(env, start_loads)

            _obs, reward, _terminated, _truncated, info = env.step(
                action.reshape(-1)
            )
            assert np.any(action < 0.0), scenario.name
            assert any(
                value > 0
                for value in info["safe_admission"]["direction_quota"].values()
            ), scenario.name
            assert np.all(start_loads >= 0.0), scenario.name
            assert np.all(np.isfinite(start_loads)), scenario.name
        finally:
            env.close()


def test_upper_slice_load_is_persistent_demand_prbs_over_physical_gnb_prbs():
    env = GlobalPPO3GNBEnv(
        seed=2,
        scenario_mode="curriculum",
        training_scenarios="jain_control_mixed",
        scenario_selection="cycle",
        gnb_configs=CENTER_GAP_GNB_CONFIGS["medium_270m"],
        slice_prb_budgets={"eMBB": 60, "URLLC": 25, "mMTC": 15},
        upper_window_seconds=1.0,
        local_steps_per_global=10,
        radio_substeps=20,
        terminal_reward_only=False,
        safe_admission_enabled=True,
    )
    try:
        env.reset(seed=2)
        loads = env._load_matrix()

        expected = np.asarray([
            sum(
                max(float(getattr(ue, "upper_demand_prbs", 0.0)), 0.0)
                for ue in env.base_env.get_all_ues()
                if ue.connected
                and ue.serving_gnb is not None
                and int(ue.serving_gnb) == 1
                and str(getattr(ue, "slice_type", "")).upper() == slice_type.upper()
            ) / 100.0
            for slice_type in env.slice_types
        ])
        assert np.allclose(loads[1], expected)
        assert np.all(loads[1] >= 0.0)
    finally:
        env.close()


def test_upper_observation_uses_reward_load_measurement():
    env = GlobalPPO3GNBEnv(
        seed=11,
        scenario_mode="curriculum",
        training_scenarios="jain_control_mixed",
        terminal_reward_only=False,
        warmup_steps=2,
    )
    try:
        observation, info = env.reset(seed=11)
        observed_loads = observation[:9].reshape(3, 3)
        logged_loads = np.asarray(info["demand_load_matrix_start"], dtype=float)

        assert np.allclose(observed_loads, logged_loads)
        assert info["load_measurement_mode"] == "post_settle_window_average_useful_prbs"
        assert (
            info["demand_calibration_mode"]
            == "window_requested_vs_achieved_prbs"
        )
        assert np.asarray(info["demand_load_matrix_start"]).shape == (3, 3)
    finally:
        env.close()


def test_upper_reward_measurement_starts_after_handover_settle_steps():
    env = GlobalPPO3GNBEnv(
        seed=12,
        scenario_mode="curriculum",
        training_scenarios="jain_control_mixed",
        terminal_reward_only=False,
        warmup_steps=0,
        local_steps_per_global=10,
        post_handover_settle_steps=4,
    )
    try:
        env.reset(seed=12)
        step_counter = {"count": 0}
        measurement_opened_after = []
        original_step = env.base_env.step
        original_begin = env.base_env.begin_radio_measurement_window

        def counted_step(action):
            step_counter["count"] += 1
            return original_step(action)

        def counted_begin():
            measurement_opened_after.append(step_counter["count"])
            return original_begin()

        env.base_env.step = counted_step
        env.base_env.begin_radio_measurement_window = counted_begin

        _obs, _reward, _terminated, _truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )

        assert measurement_opened_after == [4]
        assert step_counter["count"] == 10
        assert info["post_handover_settle_steps"] == 4
        assert info["radio_measurement_steps"] == 6
        assert info["load_measurement_mode"] == "post_settle_window_average_useful_prbs"
    finally:
        env.close()


def test_upper_reward_load_uses_post_settle_useful_prbs_after_handover():
    env = GlobalPPO3GNBEnv(
        seed=14,
        scenario_mode="curriculum",
        training_scenarios="jain_balance_controllable",
        scenario_selection="cycle",
        terminal_reward_only=False,
        warmup_steps=0,
        local_steps_per_global=10,
        post_handover_settle_steps=4,
        demand_calibration_alpha=0.0,
        a3_handover_cooldown_s=0.0,
        a3_min_residence_s=0.0,
    )
    try:
        env.reset(seed=14)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action.reshape(3, 2, 3)[1, :, 0] = -1.0

        _obs, _reward, _terminated, _truncated, info = env.step(action)
        start = np.asarray(info["load_matrix_start"], dtype=float)
        end = np.asarray(info["load_matrix_end"], dtype=float)
        useful_end = np.asarray(info["useful_load_matrix_end"], dtype=float)
        demand_start = np.asarray(info["demand_load_matrix_start"], dtype=float)
        demand_end = np.asarray(info["demand_load_matrix_end"], dtype=float)

        assert info["load_measurement_mode"] == "post_settle_window_average_useful_prbs"
        if int(info["handover_count"]) <= 0:
            return
        assert np.allclose(end, useful_end)
        assert np.isclose(np.sum(demand_start[:, 0]), np.sum(demand_end[:, 0]))
        assert np.sum(demand_end[:, 0]) < 1.10
        assert not np.isclose(np.sum(end[:, 0]), np.sum(demand_end[:, 0]))
    finally:
        env.close()
