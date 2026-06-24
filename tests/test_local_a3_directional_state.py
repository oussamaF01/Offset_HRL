import numpy as np

from local_a3_training_env import LocalA3RuleBiasTrainingEnv


def test_local_state_uses_directional_upper_tensor_entries():
    env = LocalA3RuleBiasTrainingEnv(
        seed=3,
        slice_types=("eMBB",),
        episode_steps=2,
        action_hold_steps=1,
        bias_hold_steps=1,
        print_scenarios=False,
    )
    try:
        env.reset(seed=3)
        env.local_env.set_global_bias({
            (0, 1, "eMBB"): -1.0,
            (1, 0, "eMBB"): -1.0,
        })

        obs = env.local_env._build_observation()

        assert obs.shape == env.observation_space.shape
        assert np.isclose(obs[0], -1.0)
        assert np.isclose(obs[1], -1.0)
        assert env.local_env._desired_offset(1, "eMBB") == -6.0
    finally:
        env.close()


def test_reverse_direction_bias_is_context_not_cancellation():
    env = LocalA3RuleBiasTrainingEnv(
        seed=4,
        slice_types=("eMBB",),
        episode_steps=2,
        action_hold_steps=1,
        bias_hold_steps=1,
        print_scenarios=False,
    )
    try:
        env.reset(seed=4)
        env.local_env.set_global_bias({
            (0, 1, "eMBB"): -1.0,
            (1, 0, "eMBB"): 1.0,
        })

        strong_offload = env.local_env._desired_offset(1, "eMBB")
        env.local_env.set_global_bias({
            (0, 1, "eMBB"): -1.0,
            (1, 0, "eMBB"): -1.0,
        })
        same_reverse = env.local_env._desired_offset(1, "eMBB")

        assert strong_offload == -6.0
        assert same_reverse == -6.0
    finally:
        env.close()
