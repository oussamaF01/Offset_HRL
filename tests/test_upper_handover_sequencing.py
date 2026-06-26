from collections import Counter

import numpy as np

from global_ppo_3gnb_env import GlobalPPO3GNBEnv
from upper_agent_training_scenarios import CENTER_GAP_GNB_CONFIGS


def test_budgeted_handovers_are_sequential_across_local_steps():
    env = GlobalPPO3GNBEnv(
        seed=2,
        scenario_mode="curriculum",
        training_scenarios="jain_balance_controllable",
        scenario_selection="cycle",
        gnb_configs=CENTER_GAP_GNB_CONFIGS["medium_270m"],
        upper_window_seconds=1.0,
        local_steps_per_global=10,
        radio_substeps=1,
        terminal_reward_only=False,
        safe_admission_enabled=True,
        max_handovers_per_local_step=1,
        a3_handover_cooldown_s=0.0,
        a3_min_residence_s=0.0,
        warmup_steps=0,
        post_handover_settle_steps=0,
    )
    try:
        env.reset(seed=2)
        assert env.base_env.max_handovers_per_step == 1
        assert env.base_env.handover_ttt == 3

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        directional = action.reshape(3, 2, 3)
        directional[1, 0, 0] = -0.5  # budget ceil(0.5 * 6 UEs) = 3
        directional[1, 1, 0] = 1.0   # no budget to g2

        _obs, _reward, _terminated, _truncated, info = env.step(action)
        sequence = info["upper_window_handover_sequence"]

        assert info["safe_admission"]["direction_quota"][(1, 0, "eMBB")] == 3
        assert info["safe_admission"]["direction_used"][(1, 0, "eMBB")] == 3
        assert info["handover_count"] == 3
        assert info["required_handover_settle_steps"] == 6
        assert info["effective_handover_settle_steps"] == 6
        assert info["radio_measurement_steps"] == 4

        assert len(sequence) == 3
        local_steps = [event["local_step_in_upper_window"] for event in sequence]
        assert local_steps == [3, 4, 5]
        assert max(Counter(local_steps).values()) == 1
        assert all(event["source_gnb"] == 1 for event in sequence)
        assert all(event["target_gnb"] == 0 for event in sequence)
        assert all(event["slice"] == "eMBB" for event in sequence)
        assert [event["direction_used_after_commit"] for event in sequence] == [1, 2, 3]
        assert [event["direction_remaining_after_commit"] for event in sequence] == [2, 1, 0]
        assert sequence[0]["active_offset_after_zeroing_db"] < 0.0
        assert sequence[1]["active_offset_after_zeroing_db"] < 0.0
        assert sequence[2]["active_offset_after_zeroing_db"] == 0.0
        assert info["directional_offset_tensor"][1, 0, 0] == 0.0
        assert info["directional_offset_tensor"][1, 1, 0] >= 0.0
    finally:
        env.close()
