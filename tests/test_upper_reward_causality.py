import numpy as np

from global_ppo_3gnb_env import GlobalPPO3GNBEnv


def _episode_return(action):
    env = GlobalPPO3GNBEnv(
        seed=123,
        scenario_mode="curriculum",
        training_scenarios="fixed_embb_g0_overlap",
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
    neutral = np.zeros(9, dtype=np.float32)
    release = neutral.copy()
    release[0] = -0.4
    release[3] = 0.4
    release[6] = 0.4

    release_return, release_info = _episode_return(release)
    neutral_return, neutral_info = _episode_return(neutral)

    assert release_info["load_variance"] < neutral_info["load_variance"]
    assert release_return > neutral_return


def test_reward_matches_pdf_terms_plus_balanced_nonzero_bias_penalty():
    release = np.zeros(9, dtype=np.float32)
    release[0] = -0.4
    release[3] = 0.4
    release[6] = 0.4

    env = GlobalPPO3GNBEnv(
        seed=123,
        scenario_mode="curriculum",
        training_scenarios="fixed_embb_g0_overlap",
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
            info["load_imbalance_start"]
            - info["load_imbalance_end"]
            - info["global_action_penalty"]
            - env.global_neutral_bias_weight * info["reward_neutral_bias_penalty"]
        )
        assert np.isclose(reward, expected)
    finally:
        env.close()


def test_inactive_slice_biases_do_not_trigger_neutral_penalty():
    env = GlobalPPO3GNBEnv(
        seed=7,
        scenario_mode="curriculum",
        training_scenarios="fixed_embb_g0_overlap",
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
        training_scenarios="fixed_embb_g0_overlap",
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


def test_pdf_bias_smoothing_default_is_in_recommended_range():
    env = GlobalPPO3GNBEnv(
        scenario_mode="curriculum",
        training_scenarios="fixed_embb_g0_overlap",
    )
    try:
        assert 0.01 <= env.global_action_kappa <= 0.05
    finally:
        env.close()
