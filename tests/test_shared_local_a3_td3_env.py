import numpy as np

from shared_local_a3_td3_env import SharedLocalA3TD3Env


def test_shared_env_buffers_actions_before_advancing_time():
    env = SharedLocalA3TD3Env(
        seed=11,
        episode_control_intervals=1,
        control_interval_steps=2,
        warmup_steps=0,
        radio_substeps=1,
        local_steps_per_global=3,
    )
    try:
        obs, info = env.reset(seed=11)
        start_step = int(env.base_env._step_count)
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated
        assert not truncated
        assert reward == 0.0
        assert info["time_advanced"] is False
        assert int(env.base_env._step_count) == start_step

        obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated
        assert not truncated
        assert reward == 0.0
        assert info["time_advanced"] is False
        assert int(env.base_env._step_count) == start_step

        obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated
        assert truncated
        assert info["time_advanced"] is True
        assert info["shared_policy"] is True
        assert info["shared_reward"] is True
        assert int(env.base_env._step_count) == start_step + 2
        assert set(info["pending_actions"]) == {0, 1, 2}
        assert info["load_measurement_mode"] == "control_interval_average_useful_prbs"
        assert np.asarray(info["load_matrix_start"]).shape == (3, 3)
        assert np.asarray(info["load_matrix_end"]).shape == (3, 3)
        assert np.max(np.sum(info["load_matrix_start"], axis=1)) <= 1.0 + 1e-9
        assert np.max(np.sum(info["load_matrix_end"], axis=1)) <= 1.0 + 1e-9
    finally:
        env.close()


def test_shared_env_observation_shape_is_shared_across_gnb_turns():
    env = SharedLocalA3TD3Env(
        seed=13,
        episode_control_intervals=2,
        control_interval_steps=1,
        warmup_steps=0,
        radio_substeps=1,
        local_steps_per_global=3,
    )
    try:
        obs0, info0 = env.reset(seed=13)
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        obs1, _reward, _terminated, _truncated, info1 = env.step(action)
        obs2, _reward, _terminated, _truncated, info2 = env.step(action)

        assert obs0.shape == env.observation_space.shape
        assert obs1.shape == env.observation_space.shape
        assert obs2.shape == env.observation_space.shape
        assert info0["controlled_gnb"] == 0
        assert info1["controlled_gnb"] == 0
        assert info1["next_controlled_gnb"] == 1
        assert info2["controlled_gnb"] == 1
        assert info2["next_controlled_gnb"] == 2
        assert env.action_space.shape == (6,)
        assert env.observation_space.shape == (72,)
    finally:
        env.close()


def test_shared_env_derives_episode_intervals_from_scenario_time():
    env = SharedLocalA3TD3Env(
        seed=17,
        training_scenarios="jain_control_mixed",
        upper_window_seconds=2.0,
        local_steps_per_global=10,
        control_interval_steps=4,
        episode_control_intervals=None,
        warmup_steps=0,
        radio_substeps=1,
    )
    try:
        # Scenario duration is 20 s.  local_step = 2/10 = 0.2 s, and one
        # lower control interval is 4 local steps = 0.8 s, so 20/0.8 = 25.
        assert env.episode_control_intervals == 25
    finally:
        env.close()
