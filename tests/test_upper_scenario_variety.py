import numpy as np

from global_ppo_3gnb_env import GlobalPPO3GNBEnv
from local_a3_agent_wrapper import normalize_slice_type
from upper_agent_training_scenarios import (
    UPPER_TRAINING_SCENARIOS,
    get_upper_training_scenarios,
)


def test_scenario_catalog_is_unique_feasible_and_varied():
    names = [scenario.name for scenario in UPPER_TRAINING_SCENARIOS]
    assert len(names) == len(set(names))
    assert len(names) >= 18

    directions = set()
    speeds = set()
    source_slice_pairs = set()
    for scenario in UPPER_TRAINING_SCENARIOS:
        assert 10.0 <= scenario.duration_s <= 40.0
        seen_in_scenario = set()
        for group in scenario.groups:
            assert group.slice_type in {"eMBB", "URLLC", "mMTC"}
            assert group.source_gnb in {0, 1, 2}
            assert group.count > 0
            assert 0.0 < group.total_load <= 0.95
            pair = (group.source_gnb, group.slice_type)
            assert pair not in seen_in_scenario
            seen_in_scenario.add(pair)
            source_slice_pairs.add(pair)
            speeds.add(group.speed_mps)
            if group.target_gnb is not None:
                assert group.target_gnb in {0, 1, 2}
                assert group.target_gnb != group.source_gnb
                directions.add((group.source_gnb, group.target_gnb))

    assert directions == {
        (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    }
    assert 0.0 in speeds
    assert max(speeds) >= 8.0
    assert len(source_slice_pairs) == 9
    tiers = {scenario.tier for scenario in UPPER_TRAINING_SCENARIOS}
    assert tiers == {"fixed", "slow", "fast"}


def test_all_scenarios_reset_with_expected_load_and_finite_step():
    env = GlobalPPO3GNBEnv(
        seed=17,
        scenario_mode="curriculum",
        training_scenarios="all",
        scenario_selection="cycle",
        upper_window_seconds=2.0,
        local_steps_per_global=1,
        radio_substeps=1,
        terminal_reward_only=False,
    )
    try:
        observed = []
        for expected in UPPER_TRAINING_SCENARIOS:
            obs, info = env.reset()
            observed.append(info["scenario_name"])
            assert info["scenario_name"] == expected.name
            assert np.isfinite(obs).all()
            expected_ues = sum(group.count for group in expected.groups)
            assert len(env.base_env.get_all_ues()) == expected_ues
            assert np.allclose(
                np.asarray(info["target_load_matrix"], dtype=float),
                env._active_target_load_matrix,
            )
            obs, reward, terminated, truncated, step_info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
            assert step_info["episode_duration_s"] >= expected.duration_s
        assert observed == [scenario.name for scenario in UPPER_TRAINING_SCENARIOS]
    finally:
        env.close()


def test_named_scenario_subset_preserves_requested_order():
    selected = get_upper_training_scenarios(
        "fast_border_crossing_embb,near_balanced_small_perturbation"
    )
    assert [scenario.name for scenario in selected] == [
        "fast_border_crossing_embb",
        "near_balanced_small_perturbation",
    ]


def test_staged_selection_starts_fixed_then_adds_slow():
    env = GlobalPPO3GNBEnv(
        seed=8,
        scenario_mode="curriculum",
        training_scenarios="all",
        scenario_selection="staged",
        fixed_stage_episodes=4,
        slow_stage_episodes=4,
        upper_window_seconds=2.0,
        local_steps_per_global=1,
        radio_substeps=1,
    )
    try:
        first_tiers = []
        second_tiers = []
        for episode in range(8):
            _obs, info = env.reset()
            tier = next(
                scenario.tier
                for scenario in UPPER_TRAINING_SCENARIOS
                if scenario.name == info["scenario_name"]
            )
            (first_tiers if episode < 4 else second_tiers).append(tier)
        assert set(first_tiers) == {"fixed"}
        assert set(second_tiers).issubset({"fixed", "slow"})
        assert "slow" in second_tiers
    finally:
        env.close()


def test_handover_recalibration_preserves_demand_and_slice_budget():
    env = GlobalPPO3GNBEnv(
        seed=19,
        scenario_mode="curriculum",
        training_scenarios="fixed_embb_g0_overlap",
        scenario_selection="cycle",
        terminal_reward_only=False,
        upper_window_seconds=1.0,
        local_steps_per_global=10,
        radio_substeps=2,
        radio_tick_seconds=0.05,
    )
    try:
        env.reset()
        initial_demand = {
            int(ue.id): int(getattr(ue, "upper_demand_prbs", -1))
            for ue in env.base_env.get_all_ues()
        }
        assert all(value >= 0 for value in initial_demand.values())

        action = np.asarray(
            [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            dtype=np.float32,
        )
        env.step(action)

        for ue in env.base_env.get_all_ues():
            assert int(getattr(ue, "upper_demand_prbs", -1)) == initial_demand[int(ue.id)]
        for gnb_id in range(3):
            attached = [
                ue
                for ue in env.base_env.get_all_ues()
                if ue.connected
                and ue.serving_gnb is not None
                and int(ue.serving_gnb) == gnb_id
                and normalize_slice_type(ue.slice_type) == "eMBB"
            ]
            used = sum(int(getattr(ue, "useful_prbs", 0)) for ue in attached)
            assert used <= env.base_env.get_slice_prb_budget(gnb_id, "eMBB")
            assert all(int(getattr(ue, "useful_prbs", 0)) <= 20 for ue in attached)
    finally:
        env.close()
