import numpy as np

from global_ppo_3gnb_env import GlobalPPO3GNBEnv
from upper_agent_training_scenarios import CENTER_GAP_GNB_CONFIGS


def _run_direction(target_slot: int):
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
        observation, _info = env.reset(seed=2)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        directional = action.reshape(3, 2, 3)
        directional[1, target_slot, 0] = -1.0
        _obs, reward, _terminated, _truncated, info = env.step(action)
        routes = [
            (event["from_gnb"], event["to_gnb"])
            for event in env.base_env.get_handover_events()
        ]
        return env.action_space.shape, observation.shape, reward, info, routes
    finally:
        env.close()


def test_upper_action_is_directional_source_target_slice_tensor():
    action_shape, observation_shape, _reward, _info, _routes = _run_direction(1)

    assert action_shape == (18,)
    assert observation_shape == (45,)


def test_directional_action_controls_the_handover_target():
    _shape, _obs_shape, _reward, info, routes = _run_direction(1)

    assert info["handover_count"] > 0
    assert set(routes) == {(1, 2)}
    assert info["safe_admission"]["direction_quota"][(1, 2, "eMBB")] > 0
    assert info["safe_admission"]["direction_quota"][(1, 0, "eMBB")] == 0


def test_negative_offset_is_zeroed_only_after_quota_exhaustion():
    env = GlobalPPO3GNBEnv(
        seed=3,
        scenario_mode="curriculum",
        training_scenarios="high_load_inner_asymmetric",
        scenario_selection="cycle",
        gnb_configs=CENTER_GAP_GNB_CONFIGS["medium_270m"],
        safe_admission_enabled=True,
    )
    try:
        offsets = np.zeros((3, 2, 3), dtype=float)
        offsets[1, 1, 0] = -6.0

        env.base_env.get_safe_admission_state = lambda: {
            "remaining": {(1, "eMBB"): 1},
            "direction_quota": {(1, 2, "eMBB"): 1},
            "direction_used": {(1, 2, "eMBB"): 0},
        }
        assert env._zero_quota_exhausted_offsets(offsets)[1, 1, 0] == -6.0

        env.base_env.get_safe_admission_state = lambda: {
            "remaining": {(1, "eMBB"): 0},
            "direction_quota": {(1, 2, "eMBB"): 1},
            "direction_used": {(1, 2, "eMBB"): 1},
        }
        assert env._zero_quota_exhausted_offsets(offsets)[1, 1, 0] == 0.0
    finally:
        env.close()


def test_quota_exhaustion_does_not_zero_positive_retain_offset():
    env = GlobalPPO3GNBEnv(
        seed=4,
        scenario_mode="curriculum",
        training_scenarios="high_load_inner_asymmetric",
        scenario_selection="cycle",
        gnb_configs=CENTER_GAP_GNB_CONFIGS["medium_270m"],
        safe_admission_enabled=True,
    )
    try:
        offsets = np.zeros((3, 2, 3), dtype=float)
        offsets[1, 1, 0] = 6.0
        env.base_env.get_safe_admission_state = lambda: {
            "remaining": {(1, "eMBB"): 0},
            "direction_quota": {(1, 2, "eMBB"): 0},
            "direction_used": {(1, 2, "eMBB"): 0},
        }

        assert env._zero_quota_exhausted_offsets(offsets)[1, 1, 0] == 6.0
    finally:
        env.close()


def test_training_step_logs_zero_offset_when_quota_exhausts_after_settle():
    env = GlobalPPO3GNBEnv(
        seed=7,
        scenario_mode="curriculum",
        training_scenarios="high_load_inner_asymmetric",
        scenario_selection="cycle",
        gnb_configs=CENTER_GAP_GNB_CONFIGS["medium_270m"],
        upper_window_seconds=1.0,
        local_steps_per_global=12,
        radio_substeps=20,
        post_handover_settle_steps=4,
        max_handovers_per_local_step=1,
        safe_admission_enabled=True,
    )
    try:
        env.reset(seed=7)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action.reshape(3, 2, 3)[1, 1, 0] = -1.0

        _obs, _reward, _terminated, _truncated, info = env.step(action)
        safe = info["safe_admission"]

        assert info["handover_count"] == 3
        assert safe["remaining"][(1, "eMBB")] == 0
        assert safe["direction_used"][(1, 2, "eMBB")] == safe["direction_quota"][
            (1, 2, "eMBB")
        ]
        assert info["directional_offset_tensor"][1, 1, 0] == 0.0
    finally:
        env.close()
