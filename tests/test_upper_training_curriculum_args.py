from argparse import Namespace

import numpy as np

from global_ppo_3gnb_env import GlobalPPO3GNBEnv
from train_upper_ppo_3gnb import resolve_upper_training_curriculum_args
from upper_agent_training_scenarios import get_upper_training_scenarios


def test_default_upper_training_uses_one_coherent_scenario():
    args = Namespace(
        block_curriculum_training=False,
        curriculum_training=False,
        single_training_scenario="jain_balance_controllable",
        training_scenarios="jain_control_urllc,jain_control_mmtc",
        scenario_selection="random",
    )

    resolved = resolve_upper_training_curriculum_args(args)

    assert resolved.training_scenarios == "jain_balance_controllable"
    assert resolved.scenario_selection == "cycle"


def test_explicit_curriculum_training_preserves_pool_and_selection():
    args = Namespace(
        block_curriculum_training=False,
        curriculum_training=True,
        single_training_scenario="jain_balance_controllable",
        training_scenarios="jain_control_urllc,jain_control_mmtc",
        scenario_selection="staged",
    )

    resolved = resolve_upper_training_curriculum_args(args)

    assert resolved.training_scenarios == "jain_control_urllc,jain_control_mmtc"
    assert resolved.scenario_selection == "staged"


def test_block_curriculum_computes_episodes_for_multiple_ppo_updates():
    args = Namespace(
        block_curriculum_training=True,
        curriculum_training=False,
        single_training_scenario="jain_balance_controllable",
        training_scenarios="jain_balance_controllable",
        scenario_selection="random",
        curriculum_block_episodes=0,
        ppo_updates_per_scenario=2,
        ppo_n_steps=40,
        upper_window_seconds=1.0,
    )

    resolved = resolve_upper_training_curriculum_args(args)

    assert resolved.curriculum_training is True
    assert resolved.scenario_selection == "block"
    assert resolved.training_scenarios == (
        "jain_balance_controllable,"
        "jain_control_urllc,"
        "jain_control_mmtc,"
        "jain_control_mixed"
    )
    assert resolved.curriculum_block_episodes == 80


def test_controllable_type1_training_sets_exact_phase_schedule():
    args = Namespace(
        block_curriculum_training=False,
        curriculum_training=False,
        controllable_type1_training=True,
        single_training_scenario="jain_balance_controllable",
        training_scenarios="jain_balance_controllable",
        scenario_selection="random",
        curriculum_block_episodes=0,
        total_timesteps=50_000,
        type1_phase_timesteps=15_000,
        mixed_phase_timesteps=30_000,
    )

    resolved = resolve_upper_training_curriculum_args(args)

    assert resolved.curriculum_training is True
    assert resolved.block_curriculum_training is False
    assert resolved.scenario_selection == "controllable_type1_then_mixed"
    assert resolved.curriculum_block_episodes == 15_000
    assert resolved.total_timesteps == 75_000
    assert resolved.training_scenarios == (
        "jain_balance_controllable,"
        "jain_control_urllc,"
        "jain_control_mmtc,"
        "jain_control_mixed"
    )


def test_controllable_type1_selector_keeps_mixed_after_single_slice_phases():
    env = GlobalPPO3GNBEnv.__new__(GlobalPPO3GNBEnv)
    env.scenario_selection = "controllable_type1_then_mixed"
    env.training_scenarios = get_upper_training_scenarios(
        "jain_balance_controllable,jain_control_urllc,jain_control_mmtc,jain_control_mixed"
    )
    env.curriculum_block_episodes = 2
    env._episode_index = 0
    env.rng = np.random.default_rng(0)

    chosen = [GlobalPPO3GNBEnv._choose_training_scenario(env).name for _ in range(9)]

    assert chosen == [
        "jain_balance_controllable",
        "jain_balance_controllable",
        "jain_control_urllc",
        "jain_control_urllc",
        "jain_control_mmtc",
        "jain_control_mmtc",
        "jain_control_mixed",
        "jain_control_mixed",
        "jain_control_mixed",
    ]
