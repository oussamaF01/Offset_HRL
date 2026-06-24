import numpy as np

from global_ppo_3gnb_env import GlobalPPO3GNBEnv
from upper_agent_training_scenarios import CENTER_GAP_GNB_CONFIGS


def test_static_asymmetric_training_episode_ends_after_one_directional_action():
    env = GlobalPPO3GNBEnv(
        seed=2,
        scenario_mode="curriculum",
        training_scenarios="high_load_inner_asymmetric",
        scenario_selection="cycle",
        gnb_configs=CENTER_GAP_GNB_CONFIGS["medium_270m"],
        upper_window_seconds=1.0,
        local_steps_per_global=10,
        radio_substeps=20,
        terminal_reward_only=False,
        safe_admission_enabled=True,
    )
    try:
        observation, reset_info = env.reset(seed=2)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action.reshape(3, 2, 3)[1, 1, 0] = -0.8

        _obs, reward, terminated, truncated, info = env.step(action)

        assert observation.shape == (45,)
        assert reset_info["episode_duration_s"] == 1.0
        assert info["episode_time_s"] == 1.0
        assert terminated is False
        assert truncated is True
        assert info["handover_count"] > 0
        assert info["load_imbalance_end"] < info["load_imbalance_start"]
        assert np.isfinite(reward)
        assert np.isclose(
            reward,
            info["reward_load_improvement"]
            + info["reward_saturation_improvement"]
            + info["reward_excess_load_improvement"]
            + info["reward_served_share_improvement"]
            + info["reward_served_active_floor"]
            - info["global_action_penalty"]
            - info["global_negative_bias_penalty"],
        )
    finally:
        env.close()
