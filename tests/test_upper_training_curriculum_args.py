from argparse import Namespace

from train_upper_ppo_3gnb import resolve_upper_training_curriculum_args


def test_default_upper_training_uses_one_coherent_scenario():
    args = Namespace(
        block_curriculum_training=False,
        curriculum_training=False,
        single_training_scenario="high_load_inner_asymmetric",
        training_scenarios="high_load_inner_embb,high_load_inner_mixed",
        scenario_selection="random",
    )

    resolved = resolve_upper_training_curriculum_args(args)

    assert resolved.training_scenarios == "high_load_inner_asymmetric"
    assert resolved.scenario_selection == "cycle"


def test_explicit_curriculum_training_preserves_pool_and_selection():
    args = Namespace(
        block_curriculum_training=False,
        curriculum_training=True,
        single_training_scenario="high_load_inner_asymmetric",
        training_scenarios="high_load_inner_embb,high_load_inner_mixed",
        scenario_selection="staged",
    )

    resolved = resolve_upper_training_curriculum_args(args)

    assert resolved.training_scenarios == "high_load_inner_embb,high_load_inner_mixed"
    assert resolved.scenario_selection == "staged"


def test_block_curriculum_computes_episodes_for_multiple_ppo_updates():
    args = Namespace(
        block_curriculum_training=True,
        curriculum_training=False,
        single_training_scenario="high_load_inner_asymmetric",
        training_scenarios="high_load_inner_asymmetric",
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
        "high_load_inner_embb,"
        "high_load_inner_mixed,"
        "high_load_inner_asymmetric"
    )
    assert resolved.curriculum_block_episodes == 80
