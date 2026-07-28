from types import SimpleNamespace

import numpy as np

from global_ppo_3gnb_env import GlobalPPO3GNBEnv
from local_a3_training_env import compute_upper_expert_directional_bias


class DummyUpperExpertEnv:
    def __init__(self, loads, margins=None, sla=None):
        self.loads = dict(loads)
        self.margins = dict(margins or {})
        self.sla = dict(sla or {})

    def estimate_slice_load(self, gnb_id, slice_type):
        return float(self.loads.get((int(gnb_id), str(slice_type)), 0.0))

    def get_slice_sla_severity(self):
        return dict(self.sla)

    def _get_gnb_by_id(self, gnb_id):
        return SimpleNamespace(id=int(gnb_id))

    def get_all_ues(self):
        return [SimpleNamespace(id=1, connected=True, serving_gnb=0, slice_type="eMBB")]

    def _compute_link_metrics(self, gnb, ue):
        del ue
        margin = float(self.margins.get(int(gnb.id), 0.0))
        return {"rx_power_dbm": -80.0 + margin}


def test_upper_expert_splits_release_across_safe_targets():
    env = DummyUpperExpertEnv(
        loads={(0, "eMBB"): 0.95, (1, "eMBB"): 0.20, (2, "eMBB"): 0.20},
        margins={0: 0.0, 1: 1.0, 2: 1.0},
    )

    bias = compute_upper_expert_directional_bias(
        env,
        gnb_ids=(0, 1, 2),
        neighbor_graph={0: (1, 2), 1: (0, 2), 2: (0, 1)},
        slice_types=("eMBB",),
    )

    assert bias[(0, 1, "eMBB")] < 0.0
    assert bias[(0, 2, "eMBB")] < 0.0
    assert abs(bias[(0, 1, "eMBB")] - bias[(0, 2, "eMBB")]) < 1e-9


def test_upper_expert_does_not_release_toward_unsafe_target():
    env = DummyUpperExpertEnv(
        loads={(0, "eMBB"): 0.95, (1, "eMBB"): 0.90, (2, "eMBB"): 0.20},
        margins={0: 0.0, 1: 1.0, 2: 1.0},
    )

    bias = compute_upper_expert_directional_bias(
        env,
        gnb_ids=(0, 1, 2),
        neighbor_graph={0: (1, 2), 1: (0, 2), 2: (0, 1)},
        slice_types=("eMBB",),
    )

    assert bias[(0, 1, "eMBB")] >= 0.0
    assert bias[(0, 2, "eMBB")] < 0.0


def test_upper_expert_releases_present_slices_when_source_total_is_high():
    env = DummyUpperExpertEnv(
        loads={
            (0, "eMBB"): 0.30,
            (0, "URLLC"): 0.25,
            (0, "mMTC"): 0.20,
            (1, "eMBB"): 0.10,
            (1, "URLLC"): 0.10,
            (1, "mMTC"): 0.10,
            (2, "eMBB"): 0.10,
            (2, "URLLC"): 0.10,
            (2, "mMTC"): 0.10,
        },
        margins={0: 0.0, 1: 1.0, 2: 1.0},
    )

    bias = compute_upper_expert_directional_bias(
        env,
        gnb_ids=(0, 1, 2),
        neighbor_graph={0: (1, 2), 1: (0, 2), 2: (0, 1)},
        slice_types=("eMBB", "URLLC", "mMTC"),
    )

    assert bias[(0, 1, "mMTC")] < 0.0
    assert bias[(0, 2, "mMTC")] < 0.0


def test_upper_expert_blocks_target_with_high_total_load_even_if_slice_has_room():
    env = DummyUpperExpertEnv(
        loads={
            (0, "eMBB"): 0.10,
            (0, "URLLC"): 0.10,
            (0, "mMTC"): 0.30,
            (1, "eMBB"): 0.35,
            (1, "URLLC"): 0.28,
            (1, "mMTC"): 0.00,
            (2, "eMBB"): 0.05,
            (2, "URLLC"): 0.05,
            (2, "mMTC"): 0.00,
        },
        margins={0: 0.0, 1: 1.0, 2: 1.0},
    )

    bias = compute_upper_expert_directional_bias(
        env,
        gnb_ids=(0, 1, 2),
        neighbor_graph={0: (1, 2), 1: (0, 2), 2: (0, 1)},
        slice_types=("eMBB", "URLLC", "mMTC"),
    )

    assert bias[(0, 1, "mMTC")] >= 0.0
    assert bias[(0, 2, "mMTC")] < 0.0
    debug = getattr(env, "_last_upper_expert_debug", {})
    rejected = debug.get("rejected_target_total_high", [])
    assert any(
        item["source_id"] == 0
        and item["target_id"] == 1
        and item["slice_type"] == "mMTC"
        for item in rejected
    )


def test_expert_bias_reward_scores_only_non_neutral_expert_entries():
    env = GlobalPPO3GNBEnv.__new__(GlobalPPO3GNBEnv)
    env.expert_bias_reward_weight = 0.5
    env.neutral_bias_eps = 0.05
    env._active_scenario = "masked_expert"
    expert = np.zeros((3, 2, 3), dtype=float)
    expert[1, 0, 0] = -1.0
    expert[1, 1, 0] = -1.0
    env._expert_bias_by_scenario = {"masked_expert": expert}

    matching_bias = np.zeros_like(expert)
    matching_bias[1, 0, 0] = -1.0
    matching_bias[1, 1, 0] = -1.0
    reward, mse, closeness, found, active_entries, active_fraction = (
        env._expert_bias_guidance_reward(matching_bias)
    )

    assert found is True
    assert active_entries == 2
    assert active_fraction == 2 / expert.size
    assert mse == 0.0
    assert closeness == 1.0
    assert reward == 0.5

    wrong_active_bias = np.zeros_like(expert)
    reward, mse, closeness, found, active_entries, _active_fraction = (
        env._expert_bias_guidance_reward(wrong_active_bias)
    )

    assert found is True
    assert active_entries == 2
    assert mse == 1.0
    assert closeness == 0.0
    assert reward == 0.0

    opposite_bias = np.zeros_like(expert)
    opposite_bias[1, 0, 0] = 1.0
    opposite_bias[1, 1, 0] = 1.0
    reward, mse, closeness, found, active_entries, _active_fraction = (
        env._expert_bias_guidance_reward(opposite_bias)
    )

    assert found is True
    assert active_entries == 2
    assert mse == 4.0
    assert closeness == -1.0
    assert reward == -0.5
