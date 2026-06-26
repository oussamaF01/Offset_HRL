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
    save_learning_curve,
)


def test_upper_info_exposes_window_qos_matrices():
    env = GlobalPPO3GNBEnv(
        seed=9,
        scenario_mode="curriculum",
        training_scenarios="jain_balance_controllable",
        upper_window_seconds=1.0,
        local_steps_per_global=10,
        radio_substeps=2,
        terminal_reward_only=False,
    )
    try:
        obs, _info = env.reset()
        _obs, _reward, _terminated, _truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        qos = info["qos"]
        for field in QOS_SCALAR_FIELDS:
            assert np.isfinite(qos[field])
        for key in QOS_MATRIX_KEYS:
            assert np.asarray(qos[key]).shape == (3, 3)
        assert qos["network_throughput_mbps"] >= 0.0
        assert qos["network_queue_kbits"] >= 0.0
        assert np.asarray(info["load_matrix_start"]).shape == (3, 3)
        assert np.asarray(info["load_matrix_end"]).shape == (3, 3)
        assert np.isclose(
            info["network_total_load_start"],
            np.sum(info["load_matrix_start"]),
        )
        assert np.isclose(
            info["network_total_load_end"],
            np.sum(info["load_matrix_end"]),
        )
    finally:
        env.close()


def test_training_csv_is_compact_and_contains_before_after_loads():
    args = argparse.Namespace(
        seed=2,
        n_gnbs=3,
        include_ue_counts=True,
        include_service_metrics=False,
        use_sumo_mobility=False,
        radio_substeps=1,
        local_steps_per_global=10,
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
        post_handover_settle_steps=4,
        training_scenarios="jain_balance_controllable",
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
    assert row["ppo_update_index"] == "0"
    assert row["rollout_step_in_update"] == "1"
    assert row["policy_has_updated"] == "False"
    assert "qos_completed_delay_ms_matrix" not in row
    assert "obs_0" not in row
    assert "action_0" not in row
    assert row["post_handover_settle_steps"] == "4"
    assert row["radio_measurement_steps"] == "6"
    assert row["prb_measurement_mode"] == "post_settle_window_average_useful_prbs"
    assert "load_start_g0_eMBB" not in row
    assert "load_end_g0_eMBB" not in row
    assert "demand_load_start_g0_eMBB" not in row
    assert "demand_load_end_g0_eMBB" not in row
    assert "useful_load_start_g0_eMBB" not in row
    assert "useful_load_end_g0_eMBB" not in row
    assert "gnb_demand_load_start_g0" not in row
    assert "gnb_demand_load_end_g0" not in row
    assert "gnb_useful_load_start_g0" not in row
    assert "gnb_useful_load_end_g0" not in row
    assert "slice_demand_load_start_eMBB" not in row
    assert "slice_demand_load_end_eMBB" not in row
    assert "slice_useful_load_start_eMBB" not in row
    assert "slice_useful_load_end_eMBB" not in row
    assert "network_total_load_start" not in row
    assert "network_total_load_end" not in row
    assert "ppo_network_demand_load_start" not in row
    assert "ppo_network_demand_load_end" not in row
    assert "radio_network_useful_load_end" not in row
    assert "radio_network_total_useful_load_start" not in row
    assert "radio_network_total_useful_load_end" not in row
    assert "radio_mean_gnb_useful_load_start" not in row
    assert "radio_mean_gnb_useful_load_end" not in row
    assert "radio_max_gnb_useful_load_start" not in row
    assert "radio_max_gnb_useful_load_end" not in row
    assert "used_prb_start_g0_eMBB" in row
    assert "used_prb_end_g0_eMBB" in row
    assert "network_used_prb_start" in row
    assert "network_used_prb_end" in row
    assert "mean_gnb_used_prb_start" in row
    assert "mean_gnb_used_prb_end" in row
    assert "max_gnb_used_prb_start" in row
    assert "max_gnb_used_prb_end" in row
    assert "gnb_used_prb_start_g0" in row
    assert "gnb_used_prb_end_g0" in row
    assert "slice_used_prb_start_eMBB" in row
    assert "slice_used_prb_end_eMBB" in row
    assert "used_prb_balance_cost_start" in row
    assert "used_prb_balance_cost_end" in row
    assert "reward_used_prb_balance_improvement" in row
    assert "reward_used_prb_balance_improvement_raw" in row
    assert "reward_load_improvement" not in row
    assert "reward_served_share_improvement" in row
    assert "reward_served_share_improvement_raw" in row
    assert "served_share_cost_start" in row
    assert "served_share_cost_end" in row
    assert "reward_served_active_floor" in row
    assert "reward_served_active_floor_raw" in row
    assert "served_active_floor_cost_start" in row
    assert "served_active_floor_cost_end" in row
    assert "served_active_floor" in row
    assert "served_active_floor_reference_g0" in row
    assert "served_active_floor_reference_g1" in row
    assert "served_active_floor_reference_g2" in row
    assert float(row["max_gnb_used_prb_start"]) <= 100.0 + 1e-9
    assert float(row["max_gnb_used_prb_end"]) <= 100.0 + 1e-9
    assert float(row["network_queue_kbits"]) >= 0.0


def test_training_csv_generates_learning_curve():
    with tempfile.TemporaryDirectory() as directory:
        csv_path = Path(directory) / "training.csv"
        graph_path = Path(directory) / "learning_curve.png"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "step", "reward", "used_prb_balance_cost_end",
                    "handover_count", "bias_g1_to_g0_eMBB",
                    "bias_g1_to_g2_eMBB",
                ],
            )
            writer.writeheader()
            for step in range(1, 21):
                writer.writerow({
                    "step": step,
                    "reward": step / 20.0,
                    "used_prb_balance_cost_end": 1.0 - step / 20.0,
                    "handover_count": min(step / 10.0, 3.0),
                    "bias_g1_to_g0_eMBB": 0.2,
                    "bias_g1_to_g2_eMBB": -step / 20.0,
                })

        result = save_learning_curve(csv_path, graph_path, rolling_window=5)

        assert result == graph_path
        assert graph_path.exists()
        assert graph_path.stat().st_size > 0
