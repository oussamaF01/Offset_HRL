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


def test_persistent_load_bonus_remains_after_progress_window():
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
        _obs, _reward, _terminated, _truncated, first = env.step(release)
        _obs, _reward, _terminated, _truncated, info = env.step(release)

        assert info["reward_load_balance_level_bonus"] > 0.0
        assert first["reward_load_balance_level_bonus"] > 0.0
        assert info["reward_load_balance_level_bonus"] <= 1.0
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


def test_load_terms_dominate_default_reward_weights():
    env = GlobalPPO3GNBEnv(
        scenario_mode="curriculum",
        training_scenarios="fixed_embb_g0_overlap",
    )
    try:
        assert env.global_reward_mu > env.global_reward_beta
        assert env.load_balance_level_weight > env.sla_severity_level_weight
        assert env.global_neutral_bias_weight <= 0.1
    finally:
        env.close()
