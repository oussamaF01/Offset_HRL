import numpy as np

from global_ppo_3gnb_env import GlobalPPO3GNBEnv


def _make_env():
    return GlobalPPO3GNBEnv(
        seed=3,
        scenario_mode="curriculum",
        training_scenarios="embb_g0_to_g1_slow,balanced_mixed_hold",
        scenario_selection="cycle",
        upper_window_seconds=2.0,
        local_steps_per_global=2,
        radio_substeps=4,
        terminal_reward_only=False,
    )


def test_curriculum_changes_scenario_every_episode_and_uses_duration():
    env = _make_env()
    try:
        _obs, first = env.reset()
        assert first["scenario_name"] == "embb_g0_to_g1_slow"
        assert first["episode_duration_s"] == 20.0

        done = False
        steps = 0
        while not done:
            _obs, _reward, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            done = terminated or truncated
            steps += 1
        assert steps == 10
        assert info["episode_time_s"] == 20.0

        _obs, second = env.reset()
        assert second["scenario_name"] == "balanced_mixed_hold"
        assert second["episode_duration_s"] == 12.0
        assert env.base_env._step_count == 0
    finally:
        env.close()


def test_time_metadata_distinguishes_mobility_and_radio_service():
    env = _make_env()
    try:
        _obs, info = env.reset()
        assert info["upper_window_seconds"] == 2.0
        assert info["local_step_seconds"] == 1.0
        assert info["radio_tick_seconds"] == 0.25
        assert info["radio_service_seconds_per_upper_window"] == 2.0
        assert info["clock_synchronized"] is True
    finally:
        env.close()


def test_first_populated_sla_window_does_not_create_bootstrap_penalty():
    env = GlobalPPO3GNBEnv(
        seed=23,
        scenario_mode="curriculum",
        training_scenarios="fixed_embb_g0_overlap",
        scenario_selection="cycle",
        terminal_reward_only=False,
        upper_window_seconds=1.0,
        local_steps_per_global=10,
        radio_substeps=10,
        radio_tick_seconds=0.01,
        handover_pingpong_guard_s=15.0,
    )
    try:
        _obs, info = env.reset()
        loads = np.asarray(info["load_matrix"], dtype=float)
        means = loads.mean(axis=0, keepdims=True)
        action = np.zeros_like(loads)
        action[loads > means + 0.04] = -1.0
        action[loads < means - 0.04] = 1.0

        _obs, reward, _terminated, _truncated, info = env.step(
            action.astype(np.float32).reshape(-1)
        )

        assert info["reward_sla_improvement"] == 0.0
        expected = (
            info["load_imbalance_start"]
            - info["load_imbalance_end"]
            - info["global_action_penalty"]
            - info["global_negative_bias_penalty"]
            - env.global_neutral_bias_weight * info["reward_neutral_bias_penalty"]
            - env.wrong_bias_penalty_weight * info["reward_wrong_bias_penalty"]
        )
        assert np.isclose(reward, expected)
        assert np.isclose(
            info["reward_load_improvement"],
            info["load_imbalance_start"] - info["load_imbalance_end"],
        )
        assert np.isfinite(reward)
    finally:
        env.close()


def test_mismatched_explicit_radio_clock_is_rejected():
    try:
        GlobalPPO3GNBEnv(
            upper_window_seconds=1.0,
            local_steps_per_global=10,
            radio_substeps=10,
            radio_tick_seconds=0.001,
        )
    except ValueError as exc:
        assert "Radio and mobility clocks must match" in str(exc)
    else:
        raise AssertionError("Expected a clock mismatch to raise ValueError")


def test_radio_time_and_a3_history_use_seconds():
    env = GlobalPPO3GNBEnv(
        upper_window_seconds=1.0,
        local_steps_per_global=4,
        radio_substeps=5,
        a3_history_window_s=12.0,
        a3_pingpong_threshold_s=3.0,
    )
    try:
        env.reset(seed=7)
        before = env.base_env.get_all_ues()[0].get_current_time_s()
        _obs, _reward, _terminated, _truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        after = env.base_env.get_all_ues()[0].get_current_time_s()

        assert np.isclose(after - before, 1.0)
        assert np.isclose(info["radio_service_seconds_per_upper_window"], 1.0)
        assert env.base_env.a3_window_steps == 48
        assert env.base_env.a3_pingpong_threshold_steps == 12
    finally:
        env.close()
