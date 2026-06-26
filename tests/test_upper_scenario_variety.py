import math

import numpy as np

from global_ppo_3gnb_env import GlobalPPO3GNBEnv
from local_a3_agent_wrapper import normalize_slice_type
from upper_agent_training_scenarios import (
    CENTER_GAP_GNB_CONFIGS,
    CENTER_LEFT_RIGHT_GNB_CONFIGS,
    UPPER_TRAINING_SCENARIOS,
    get_upper_training_scenarios,
)


def test_scenario_catalog_is_slice_aware_and_uses_explicit_regions():
    names = [scenario.name for scenario in UPPER_TRAINING_SCENARIOS]
    assert names == [
        "jain_balance_controllable",
        "jain_control_urllc",
        "jain_control_mmtc",
        "jain_control_mixed",
    ]
    assert all(scenario.tier == "fixed" for scenario in UPPER_TRAINING_SCENARIOS)
    assert all(scenario.duration_s == 1.0 for scenario in UPPER_TRAINING_SCENARIOS)
    assert any(
        {group.slice_type for group in scenario.groups}
        == {"eMBB", "URLLC", "mMTC"}
        for scenario in UPPER_TRAINING_SCENARIOS
    )
    for scenario in UPPER_TRAINING_SCENARIOS:
        source_slice_pairs = [
            (group.source_gnb, group.slice_type)
            for group in scenario.groups
        ]
        assert len(source_slice_pairs) == len(set(source_slice_pairs))
        for group in scenario.groups:
            assert group.placement_region in {"overlap", "fixed_core"}
            assert group.speed_mps == 0.0
            assert len(group.fixed_source_offsets_m) == group.count
            assert 0.0 < group.total_load <= 0.95
            assert group.total_load <= group.count * 20 / 100.0 + 1e-12
    assert tuple(CENTER_GAP_GNB_CONFIGS) == (
        "tight_220m",
        "medium_270m",
        "wide_320m",
    )


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
        "jain_control_mmtc,jain_balance_controllable"
    )
    assert [scenario.name for scenario in selected] == [
        "jain_control_mmtc",
        "jain_balance_controllable",
    ]


def test_controllable_embb_scenario_is_stationary_and_bidirectional():
    env = GlobalPPO3GNBEnv(
        seed=123,
        gnb_configs=CENTER_LEFT_RIGHT_GNB_CONFIGS,
        scenario_mode="curriculum",
        training_scenarios="jain_balance_controllable",
        upper_window_seconds=1.0,
        local_steps_per_global=10,
        radio_substeps=2,
        terminal_reward_only=False,
        max_handovers_per_local_step=3,
        a3_handover_cooldown_s=2.0,
        a3_min_residence_s=2.0,
    )
    try:
        env.reset(seed=123)
        ues = list(env.base_env.get_all_ues())
        assert len(ues) == 6
        assert all(int(ue.serving_gnb) == 1 for ue in ues)
        assert all(float(ue.vx) == 0.0 and float(ue.vy) == 0.0 for ue in ues)
        assert sum(float(ue.x) < 0.0 for ue in ues) == 3
        assert sum(float(ue.x) > 0.0 for ue in ues) == 3
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action.reshape(3, 2, 3)[1, :, 0] = -0.4
        _obs, _reward, _terminated, _truncated, info = env.step(action)
        events = list(env.base_env.handover_events)
        state = info["safe_admission"]

        assert state["quota"][(1, "eMBB")] == 6
        assert state["used"][(1, "eMBB")] <= 6
        assert info["handover_count"] == state["used"][(1, "eMBB")]
        assert state["direction_quota"][(1, 0, "eMBB")] == 3
        assert state["direction_quota"][(1, 2, "eMBB")] == 3
        assert {event["to_gnb"] for event in events}.issubset({0, 2})
        assert sum(int(ue.serving_gnb) == 1 for ue in ues) == 6 - info["handover_count"]
    finally:
        env.close()


def test_controllable_training_ues_use_learnable_midpoint_placement():
    env = GlobalPPO3GNBEnv(
        seed=123,
        gnb_configs=CENTER_LEFT_RIGHT_GNB_CONFIGS,
        scenario_mode="curriculum",
        training_scenarios="jain_balance_controllable",
        upper_window_seconds=1.0,
        local_steps_per_global=10,
        radio_substeps=2,
        terminal_reward_only=False,
    )
    try:
        env.reset(seed=123)
        ues = list(env.base_env.get_all_ues())

        assert len(ues) == 6
        assert {abs(float(ue.x)) for ue in ues} == {135.0}
        assert all(
            all(
                env.base_env._is_in_coverage(
                    env.base_env._get_gnb_by_id(gnb_id), ue
                )
                for gnb_id in range(3)
            )
            for ue in ues
        )
    finally:
        env.close()


def test_controlled_jain_scenarios_build_symmetric_directional_budgets():
    slice_names = ("eMBB", "URLLC", "mMTC")
    for scenario_name, active_slice_indices in (
        ("jain_balance_controllable", (0,)),
        ("jain_control_urllc", (1,)),
        ("jain_control_mmtc", (2,)),
        ("jain_control_mixed", (0, 1, 2)),
    ):
        env = GlobalPPO3GNBEnv(
            seed=123,
            gnb_configs=CENTER_LEFT_RIGHT_GNB_CONFIGS,
            scenario_mode="curriculum",
            training_scenarios=scenario_name,
            upper_window_seconds=1.0,
            local_steps_per_global=10,
            radio_substeps=2,
            terminal_reward_only=False,
            max_handovers_per_local_step=3,
            safe_admission_enabled=True,
            a3_handover_cooldown_s=0.0,
            a3_min_residence_s=0.0,
        )
        try:
            env.reset(seed=123)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action_3d = action.reshape(3, 2, 3)
            for slice_idx in active_slice_indices:
                action_3d[1, :, slice_idx] = -1.0

            _obs, _reward, _terminated, _truncated, info = env.step(action)
            state = info["safe_admission"]
            for slice_idx in active_slice_indices:
                slice_type = slice_names[slice_idx]
                assert state["direction_quota"][(1, 0, slice_type)] == 6
                assert state["direction_quota"][(1, 2, slice_type)] == 6
            assert state["stats"]["rejected_target_safety"] == 0
            assert state["stats"]["rejected_no_source_excess"] == 0
        finally:
            env.close()


def test_center_gap_catalog_keeps_controllable_placement_and_three_cell_coverage():
    reference_positions = None
    for topology_name, gnb_configs in CENTER_GAP_GNB_CONFIGS.items():
        env = GlobalPPO3GNBEnv(
            seed=123,
            gnb_configs=gnb_configs,
            scenario_mode="curriculum",
            training_scenarios="jain_balance_controllable",
            upper_window_seconds=1.0,
            local_steps_per_global=10,
            radio_substeps=2,
            terminal_reward_only=False,
            max_handovers_per_local_step=3,
        )
        try:
            env.reset(seed=123)
            ues = list(env.base_env.get_all_ues())
            assert len(ues) == 6
            assert all(int(ue.serving_gnb) == 1 for ue in ues)
            assert all(float(ue.vx) == 0.0 and float(ue.vy) == 0.0 for ue in ues)
            positions = tuple(
                (round(float(ue.x), 6), round(float(ue.y), 6))
                for ue in ues
            )
            if reference_positions is None:
                reference_positions = positions
            assert positions == reference_positions, topology_name
            for ue in ues:
                assert all(
                    math.hypot(
                        float(ue.x) - float(gnb.x),
                        float(ue.y) - float(gnb.y),
                    )
                    <= float(gnb.coverage_radius)
                    for gnb in env.base_env.gnbs
                )
        finally:
            env.close()


def test_staged_selection_cycles_retained_fixed_scenarios():
    env = GlobalPPO3GNBEnv(
        seed=8,
        scenario_mode="curriculum",
        training_scenarios="all",
        scenario_selection="staged",
        fixed_stage_episodes=20,
        slow_stage_episodes=4,
        upper_window_seconds=2.0,
        local_steps_per_global=1,
        radio_substeps=1,
    )
    try:
        observed = []
        for episode in range(len(UPPER_TRAINING_SCENARIOS)):
            _obs, info = env.reset()
            observed.append(info["scenario_name"])
        assert observed == [
            scenario.name for scenario in UPPER_TRAINING_SCENARIOS
        ]
    finally:
        env.close()


def test_block_selection_repeats_one_scenario_before_switching():
    env = GlobalPPO3GNBEnv(
        seed=8,
        scenario_mode="curriculum",
        training_scenarios=(
            "jain_balance_controllable,"
            "jain_control_urllc,"
            "jain_control_mmtc"
        ),
        scenario_selection="block",
        curriculum_block_episodes=3,
        upper_window_seconds=1.0,
        local_steps_per_global=1,
        radio_substeps=1,
    )
    try:
        observed = []
        block_indices = []
        for _episode in range(9):
            _obs, info = env.reset()
            observed.append(info["scenario_name"])
            block_indices.append(info["curriculum_block_index"])

        assert observed == [
            "jain_balance_controllable",
            "jain_balance_controllable",
            "jain_balance_controllable",
            "jain_control_urllc",
            "jain_control_urllc",
            "jain_control_urllc",
            "jain_control_mmtc",
            "jain_control_mmtc",
            "jain_control_mmtc",
        ]
        assert block_indices == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    finally:
        env.close()


def test_controllable_overlap_coverage_semantics():
    env = GlobalPPO3GNBEnv(
        seed=31,
        gnb_configs=CENTER_LEFT_RIGHT_GNB_CONFIGS,
        scenario_mode="curriculum",
        training_scenarios="jain_control_mixed",
        upper_window_seconds=1.0,
        local_steps_per_global=1,
        radio_substeps=1,
    )
    try:
        env.reset(seed=31)
        scenario = env._active_training_scenario
        ues = list(env.base_env.get_all_ues())
        cursor = 0
        for group in scenario.groups:
            group_ues = ues[cursor:cursor + group.count]
            cursor += group.count
            for ue in group_ues:
                coverage = [
                    math.hypot(
                        float(ue.x) - float(gnb.x),
                        float(ue.y) - float(gnb.y),
                    )
                    <= float(gnb.coverage_radius)
                    for gnb in env.base_env.gnbs
                ]
                assert group.placement_region == "overlap"
                assert sum(coverage) == 3
    finally:
        env.close()


def test_all_region_semantics_hold_across_all_three_topologies():
    for topology_name, configs in CENTER_GAP_GNB_CONFIGS.items():
        for scenario in UPPER_TRAINING_SCENARIOS:
            for group in scenario.groups:
                source = configs[group.source_gnb]
                for dx, dy in group.fixed_source_offsets_m:
                    x = float(source["x"]) + float(dx)
                    y = float(source["y"]) + float(dy)
                    coverage = [
                        math.hypot(x - float(cfg["x"]), y - float(cfg["y"]))
                        <= float(cfg["coverage_radius"])
                        for cfg in configs
                    ]
                    assert group.placement_region == "overlap"
                    assert sum(coverage) == 3, (
                        topology_name,
                        scenario.name,
                        group.slice_type,
                        (x, y),
                    )


def test_handover_recalibration_preserves_demand_and_slice_budget():
    env = GlobalPPO3GNBEnv(
        seed=19,
        scenario_mode="curriculum",
        training_scenarios="jain_balance_controllable",
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

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action.reshape(3, 2, 3)[1, :, 0] = -0.4
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
    finally:
        env.close()
