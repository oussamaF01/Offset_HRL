import numpy as np

from global_ppo_3gnb_env import GlobalPPO3GNBEnv


def _directional_action(source_bias):
    source_bias = np.asarray(source_bias, dtype=np.float32).reshape(3, 3)
    directional = np.repeat(source_bias[:, None, :], 2, axis=1)
    directional[directional < 0.0] /= 2.0
    return directional.reshape(-1)


def _episode_return(action):
    env = GlobalPPO3GNBEnv(
        seed=123,
        scenario_mode="curriculum",
        training_scenarios="fixed_center_embb_left_right",
        scenario_selection="cycle",
        upper_window_seconds=1.0,
        local_steps_per_global=4,
        radio_substeps=2,
        terminal_reward_only=False,
        global_neutral_bias_weight=0.05,
        global_bad_direction_eta=0.05,
        global_unsafe_target_rho=0.1,
        load_balance_level_weight=0.5,
        a3_handover_cooldown_s=5.0,
        a3_min_residence_s=5.0,
    )
    try:
        env.reset(seed=123)
        total = 0.0
        final_info = None
        done = False
        while not done:
            _obs, reward, terminated, truncated, final_info = env.step(action)
            total += reward
            done = terminated or truncated
        return total, final_info
    finally:
        env.close()


def test_useful_release_beats_neutral_over_full_episode():
    neutral = _directional_action(np.zeros((3, 3), dtype=np.float32))
    source_release = np.zeros((3, 3), dtype=np.float32)
    source_release[1, 0] = -0.2
    release = _directional_action(source_release)

    release_return, release_info = _episode_return(release)
    neutral_return, neutral_info = _episode_return(neutral)

    assert release_info["load_variance"] < neutral_info["load_variance"]
    assert release_return > neutral_return


def test_reward_matches_pdf_terms_plus_balanced_nonzero_bias_penalty():
    source_release = np.zeros((3, 3), dtype=np.float32)
    source_release[0, 0] = 0.4
    source_release[1, 0] = -0.4
    source_release[2, 0] = 0.4
    release = _directional_action(source_release)

    env = GlobalPPO3GNBEnv(
        seed=123,
        scenario_mode="curriculum",
        training_scenarios="fixed_center_embb_left_right",
        upper_window_seconds=1.0,
        local_steps_per_global=4,
        radio_substeps=2,
        terminal_reward_only=False,
        load_balance_level_weight=0.5,
    )
    try:
        env.reset(seed=123)
        _obs, reward, _terminated, _truncated, info = env.step(release)

        expected = (
            info["reward_load_improvement"]
            + info["reward_saturation_improvement"]
            + info["reward_excess_load_improvement"]
            + info["reward_served_active_floor"]
            + info["reward_jain_fairness"]
            - info["global_action_penalty"]
            - info["global_negative_bias_penalty"]
        )
        assert info["reward_sla_improvement"] == 0.0
        assert info["global_bad_direction_penalty"] == 0.0
        assert info["reward_neutral_bias_penalty"] == 0.0
        assert info["reward_wrong_bias_penalty"] == 0.0
        assert info["reward_load_balance_level_bonus"] == 0.0
        assert np.isclose(reward, expected)
        assert info["reward_active_slice_count"] == 1
        assert np.isclose(
            info["reward_load_improvement"],
            info["reward_load_improvement_raw"]
            / info["load_imbalance_start"],
        )
    finally:
        env.close()


def test_load_improvement_is_scaled_by_starting_imbalance():
    env = GlobalPPO3GNBEnv(
        seed=123,
        scenario_mode="curriculum",
        training_scenarios="mixed_slices_center_overlap",
        upper_window_seconds=1.0,
        local_steps_per_global=10,
        radio_substeps=2,
        terminal_reward_only=False,
        max_handovers_per_local_step=3,
        a3_handover_cooldown_s=2.0,
        a3_min_residence_s=2.0,
    )
    try:
        _obs, reset_info = env.reset(seed=123)
        loads = np.asarray(reset_info["load_matrix"], dtype=float)
        means = loads.mean(axis=0, keepdims=True)
        source_action = np.zeros_like(loads)
        source_action[loads > means + 0.05] = -0.5
        source_action[loads < means - 0.05] = 0.5

        _obs, reward, _terminated, _truncated, info = env.step(
            _directional_action(source_action)
        )

        raw = (
            info["load_imbalance_start"]
            - info["load_imbalance_end"]
        )
        assert info["reward_active_slice_count"] == 3
        assert np.isclose(info["reward_load_improvement_raw"], raw)
        assert np.isclose(
            info["reward_load_improvement"],
            np.clip(raw / info["load_imbalance_start"], -1.0, 1.0),
        )
        expected = (
            info["reward_load_improvement"]
            + info["reward_saturation_improvement"]
            + info["reward_excess_load_improvement"]
            + info["reward_served_active_floor"]
            + info["reward_jain_fairness"]
            - info["global_action_penalty"]
            - info["global_negative_bias_penalty"]
        )
        assert info["reward_sla_improvement"] == 0.0
        assert np.isclose(reward, expected)
    finally:
        env.close()


def test_sla_is_logged_but_cannot_change_upper_routing_reward():
    env = GlobalPPO3GNBEnv(
        seed=123,
        scenario_mode="curriculum",
        training_scenarios="fixed_center_embb_left_right",
        terminal_reward_only=False,
        global_reward_beta=1000.0,
    )
    try:
        env.reset(seed=123)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _obs, reward, _terminated, _truncated, info = env.step(action)

        expected = (
            info["reward_load_improvement"]
            + info["reward_saturation_improvement"]
            + info["reward_excess_load_improvement"]
            + info["reward_served_active_floor"]
            + info["reward_jain_fairness"]
            - info["global_action_penalty"]
            - info["global_negative_bias_penalty"]
        )
        assert env.global_reward_beta == 0.0
        assert info["reward_sla_improvement"] == 0.0
        assert np.isclose(reward, expected)
        assert "sla_severity" in info
    finally:
        env.close()


def test_gnb_load_target_is_raised_when_point_65_is_infeasible():
    env = GlobalPPO3GNBEnv(
        seed=123,
        scenario_mode="curriculum",
        training_scenarios="mixed_slices_center_overlap",
        terminal_reward_only=False,
        gnb_load_target=0.65,
    )
    try:
        env.reset(seed=123)
        target, feasible, demand_utilization = env._effective_gnb_load_target()

        assert feasible is False
        assert demand_utilization > 0.65
        assert np.isclose(target, demand_utilization)
    finally:
        env.close()


def test_gnb_load_target_stays_point_65_when_demand_is_feasible():
    env = GlobalPPO3GNBEnv(
        seed=123,
        scenario_mode="curriculum",
        training_scenarios="fixed_center_embb_left_right",
        terminal_reward_only=False,
        gnb_load_target=0.65,
    )
    try:
        env.reset(seed=123)
        target, feasible, demand_utilization = env._effective_gnb_load_target()

        assert feasible is True
        assert demand_utilization <= 0.65
        assert target == 0.65
    finally:
        env.close()


def test_inactive_slice_biases_do_not_trigger_neutral_penalty():
    env = GlobalPPO3GNBEnv(
        seed=7,
        scenario_mode="curriculum",
        training_scenarios="fixed_center_embb_left_right",
        upper_window_seconds=1.0,
        local_steps_per_global=4,
        radio_substeps=2,
        terminal_reward_only=False,
    )
    try:
        _obs, info = env.reset(seed=7)
        action = np.zeros((3, 3), dtype=np.float32)
        action[:, 1:] = 1.0

        penalty = env._balanced_slice_neutral_bias_penalty(
            np.asarray(info["load_matrix"], dtype=float),
            action,
            eps=env.neutral_bias_eps,
        )
        assert penalty == 0.0
    finally:
        env.close()


def test_balanced_active_slice_prefers_near_zero_bias():
    env = GlobalPPO3GNBEnv(
        scenario_mode="curriculum",
        training_scenarios="fixed_center_embb_left_right",
    )
    try:
        balanced_loads = np.zeros((3, 3), dtype=np.float32)
        balanced_loads[:, 0] = [0.60, 0.62, 0.58]

        near_zero = np.zeros((3, 3), dtype=np.float32)
        near_zero[:, 0] = [0.01, -0.02, 0.01]
        strong_bias = np.zeros((3, 3), dtype=np.float32)
        strong_bias[:, 0] = [-0.8, 0.7, 0.6]

        near_zero_penalty = env._balanced_slice_neutral_bias_penalty(
            balanced_loads, near_zero, eps=0.05
        )
        strong_bias_penalty = env._balanced_slice_neutral_bias_penalty(
            balanced_loads, strong_bias, eps=0.05
        )

        assert near_zero_penalty < strong_bias_penalty
        assert np.isclose(strong_bias_penalty, 0.7)
    finally:
        env.close()


def test_wrong_bias_direction_penalizes_release_from_light_cells():
    env = GlobalPPO3GNBEnv(
        scenario_mode="curriculum",
        training_scenarios="fixed_center_embb_left_right",
    )
    try:
        loads = np.zeros((3, 3), dtype=np.float32)
        loads[:, 0] = [0.90, 0.20, 0.10]
        selective = np.zeros((3, 3), dtype=np.float32)
        selective[0, 0] = -0.5
        release_everywhere = np.zeros((3, 3), dtype=np.float32)
        release_everywhere[:, 0] = -0.5
        retain_overloaded = np.zeros((3, 3), dtype=np.float32)
        retain_overloaded[0, 0] = 0.5

        assert env._wrong_bias_direction_penalty(loads, selective) == 0.0
        assert env._wrong_bias_direction_penalty(loads, release_everywhere) > 0.0
        assert env._wrong_bias_direction_penalty(loads, retain_overloaded) > 0.0
    finally:
        env.close()


def test_legacy_source_shaping_does_not_change_directional_reward():
    env = GlobalPPO3GNBEnv(
        seed=2,
        scenario_mode="curriculum",
        training_scenarios="high_load_inner_asymmetric",
        scenario_selection="cycle",
        terminal_reward_only=False,
        global_neutral_bias_weight=100.0,
        wrong_bias_penalty_weight=100.0,
    )
    try:
        env.reset(seed=2)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action.reshape(3, 2, 3)[1, 1, 0] = -0.8

        _obs, reward, _terminated, _truncated, info = env.step(action)

        expected = (
            info["reward_load_improvement"]
            + info["reward_saturation_improvement"]
            + info["reward_excess_load_improvement"]
            + info["reward_served_active_floor"]
            + info["reward_jain_fairness"]
            - info["global_action_penalty"]
            - info["global_negative_bias_penalty"]
        )
        assert info["reward_sla_improvement"] == 0.0
        assert np.isclose(reward, expected)
    finally:
        env.close()


def test_negative_bias_magnitude_penalty_is_persistent_and_monotonic():
    env = GlobalPPO3GNBEnv(
        scenario_mode="curriculum",
        training_scenarios="fixed_center_embb_left_right",
        global_action_lambda=0.01,
    )
    try:
        weak = np.zeros((3, 3), dtype=np.float32)
        weak[0, 0] = -0.2
        strong = np.zeros((3, 3), dtype=np.float32)
        strong[0, 0] = -0.8
        release_everywhere = np.full((3, 3), -0.8, dtype=np.float32)

        weak_penalty = env._negative_bias_magnitude_penalty(weak)
        strong_penalty = env._negative_bias_magnitude_penalty(strong)
        everywhere_penalty = env._negative_bias_magnitude_penalty(release_everywhere)

        assert weak_penalty > 0.0
        assert weak_penalty < strong_penalty < everywhere_penalty
        assert np.isclose(weak_penalty, 0.01 * 0.2**2 / 9)
        assert np.isclose(everywhere_penalty, 0.01 * 0.8**2)
    finally:
        env.close()


def test_one_step_episode_has_no_cross_episode_smoothness_penalty():
    env = GlobalPPO3GNBEnv(
        scenario_mode="curriculum",
        training_scenarios="high_load_inner_asymmetric",
        global_action_kappa=0.01,
    )
    try:
        env.reset(seed=7)
        current = np.ones((3, 2, 3), dtype=np.float32)
        previous = np.zeros_like(current)

        assert env._active_episode_steps == 1
        assert env._bias_smoothness_penalty(current, previous) == 0.0
    finally:
        env.close()


def test_negative_penalty_ignores_inactive_slice_dimensions():
    env = GlobalPPO3GNBEnv(
        scenario_mode="curriculum",
        training_scenarios="high_load_inner_asymmetric",
        global_action_lambda=0.01,
    )
    try:
        _obs, info = env.reset(seed=7)
        loads = np.asarray(info["load_matrix"], dtype=float)
        active_only = np.zeros((3, 2, 3), dtype=np.float32)
        active_only[1, 1, 0] = -0.8
        with_inactive_noise = active_only.copy()
        with_inactive_noise[:, :, 1:] = -1.0

        assert np.isclose(
            env._negative_bias_magnitude_penalty(
                active_only, active_loads=loads
            ),
            env._negative_bias_magnitude_penalty(
                with_inactive_noise, active_loads=loads
            ),
        )
    finally:
        env.close()


def test_pdf_bias_smoothing_default_is_in_recommended_range():
    env = GlobalPPO3GNBEnv(
        scenario_mode="curriculum",
        training_scenarios="fixed_center_embb_left_right",
    )
    try:
        assert 0.01 <= env.global_action_kappa <= 0.05
    finally:
        env.close()


def test_served_floor_penalizes_emptying_initially_served_gnb():
    def run(safe_admission_enabled: bool):
        env = GlobalPPO3GNBEnv(
            seed=21,
            scenario_mode="curriculum",
            training_scenarios="mixed_overlap_with_fixed_slice_loads",
            scenario_selection="cycle",
            terminal_reward_only=False,
            safe_admission_enabled=safe_admission_enabled,
            warmup_steps=0,
            local_steps_per_global=10,
            post_handover_settle_steps=4,
            demand_calibration_alpha=0.0,
            max_handovers_per_local_step=10,
            served_share_reward_weight=1.0,
            served_active_floor_reward_weight=1.0,
            served_active_floor=0.20,
        )
        try:
            env.reset(seed=21)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            directional = action.reshape(3, 2, 3)
            directional[1, :, 0] = -1.0
            directional[1, :, 1] = -1.0
            _obs, _reward, _terminated, _truncated, info = env.step(action)
            return info
        finally:
            env.close()

    unsafe = run(False)
    safe = run(True)

    assert (
        safe["reward_served_active_floor"]
        > unsafe["reward_served_active_floor"]
    )
    assert np.sum(unsafe["load_matrix_end"][1, :]) == 0.0
    assert np.sum(safe["load_matrix_end"][1, :]) > 0.0
