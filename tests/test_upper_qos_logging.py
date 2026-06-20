import argparse
import csv
import tempfile
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from global_ppo_3gnb_env import GlobalPPO3GNBEnv
from train_upper_ppo_3gnb import (
    QOS_MATRIX_KEYS,
    QOS_SCALAR_FIELDS,
    UpperTrainingCsvCallback,
    make_env,
)


def test_upper_info_exposes_window_qos_matrices():
    env = GlobalPPO3GNBEnv(
        seed=9,
        scenario_mode="curriculum",
        training_scenarios="mixed_g0_release",
        upper_window_seconds=1.0,
        local_steps_per_global=1,
        radio_substeps=2,
        terminal_reward_only=False,
    )
    try:
        obs, _info = env.reset()
        _obs, _reward, _terminated, _truncated, info = env.step(
            np.zeros_like(obs[:9], dtype=np.float32)
        )
        qos = info["qos"]
        for field in QOS_SCALAR_FIELDS:
            assert np.isfinite(qos[field])
        for key in QOS_MATRIX_KEYS:
            assert np.asarray(qos[key]).shape == (3, 3)
        assert qos["network_throughput_mbps"] >= 0.0
        assert qos["network_queue_kbits"] >= 0.0
    finally:
        env.close()


def test_training_csv_contains_qos_columns():
    args = argparse.Namespace(
        seed=2,
        n_gnbs=3,
        include_ue_counts=True,
        include_service_metrics=False,
        use_sumo_mobility=False,
        radio_substeps=1,
        local_steps_per_global=1,
        global_steps_per_episode=2,
        scenario_mode="curriculum",
        snapshot_scenario="mixed",
        dense_window_reward=True,
        use_progress_reward=False,
        max_handovers_per_local_step=1,
        action_direction_reward_weight=0.0,
        snapshot_block_episodes=1,
        light_load_ues=1,
        medium_load_ues=2,
        high_load_ues=3,
        debug=False,
        slice_prb_budgets=None,
        max_prbs_per_ue=20,
        directional_global_action=False,
        sla_deadband=0.05,
        upper_window_seconds=2.0,
        training_scenarios="balanced_mixed_hold",
        scenario_selection="cycle",
    )
    env = make_env(args)
    with tempfile.TemporaryDirectory() as directory:
        csv_path = Path(directory) / "qos_training.csv"
        try:
            model = PPO(
                "MlpPolicy",
                env,
                n_steps=4,
                batch_size=4,
                n_epochs=1,
                verbose=0,
                seed=2,
            )
            model.learn(
                total_timesteps=4,
                callback=UpperTrainingCsvCallback(csv_path),
                progress_bar=False,
            )
        finally:
            env.close()

        with csv_path.open(newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
    assert "network_throughput_mbps" in row
    assert "qos_completed_delay_ms_matrix" in row
    assert "qos_throughput_mbps_g0_eMBB" in row
    assert float(row["network_queue_kbits"]) >= 0.0
