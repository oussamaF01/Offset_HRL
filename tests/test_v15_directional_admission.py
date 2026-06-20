from types import SimpleNamespace

import numpy as np

from multi_gnb_wrapper import MultiGNBWrapper
from strong_heuristic_local_executor import (
    SAFE_EXTENDED_OFFSET_SET_DB,
    strong_directional_heuristic_local_executor,
)


def _directional_inputs():
    return {
        "B": np.asarray([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "prev_offsets": np.zeros((3, 2, 3), dtype=float),
        "ue_slice": np.asarray([0], dtype=int),
        "ue_serving_gnb": np.asarray([0], dtype=int),
        "rsrp_matrix": np.asarray([[-80.0, -79.0, -100.0]], dtype=float),
        "neighbor_graph": {0: [1, 2], 1: [0, 2], 2: [0, 1]},
        "load": np.asarray(
            [[0.90, 0.0, 0.0], [0.30, 0.0, 0.0], [0.85, 0.0, 0.0]],
            dtype=float,
        ),
        "sla_violation": np.zeros((3, 3), dtype=float),
        "ho_failure_ratio": np.zeros((3, 3), dtype=float),
        "pingpong_ratio": np.zeros((3, 3), dtype=float),
    }


def test_directional_executor_can_treat_neighbors_differently():
    offsets = strong_directional_heuristic_local_executor(**_directional_inputs())

    assert offsets.shape == (3, 2, 3)
    assert offsets[0, 0, 0] < offsets[0, 1, 0]
    assert set(np.unique(offsets)).issubset(set(SAFE_EXTENDED_OFFSET_SET_DB))


def test_directional_executor_preserves_upper_bias_meaning():
    inputs = _directional_inputs()

    inputs["B"][0, 0] = -1.0
    release = strong_directional_heuristic_local_executor(**inputs)
    assert np.all(release[0, :, 0] <= 0.0)

    inputs["B"][0, 0] = 1.0
    retain = strong_directional_heuristic_local_executor(**inputs)
    assert np.all(retain[0, :, 0] >= 0.0)

    inputs["B"][0, 0] = 0.0
    neutral = strong_directional_heuristic_local_executor(**inputs)
    assert np.all(neutral[0, :, 0] == 0.0)


def test_unsafe_target_vetoes_release_without_reversing_other_directions():
    inputs = _directional_inputs()
    inputs["B"][0, 0] = -1.0
    inputs["load"][1, 0] = 0.90
    offsets = strong_directional_heuristic_local_executor(**inputs)

    assert offsets[0, 0, 0] >= 0.0
    assert offsets[0, 1, 0] <= 0.0


def _admission_stub():
    env = object.__new__(MultiGNBWrapper)
    env.n_gnbs = 3
    env.gnbs = [SimpleNamespace(id=i) for i in range(3)]
    env.safe_admission_enabled = True
    env.safe_admission_load_limits = {"eMBB": 0.80, "URLLC": 0.80, "mMTC": 0.80}
    env._safe_admission_bias = {}
    env._safe_admission_source_excess = {}
    env._safe_admission_target_headroom = {}
    env._safe_admission_capacities = {}
    env._safe_admission_accepted = {}
    env._safe_admission_source_capacities = {}
    env._safe_admission_source_accepted = {}
    env._safe_admission_stats = {}
    env._reset_safe_admission_stats()

    loads = {
        (0, "eMBB"): 0.90,
        (1, "eMBB"): 0.40,
        (2, "eMBB"): 0.50,
        **{(g, s): 0.0 for g in range(3) for s in ("URLLC", "mMTC")},
    }
    env.get_slice_loads = lambda: dict(loads)
    env.get_slice_ue_count = lambda gnb_id, slice_type: 10 if (gnb_id, slice_type) == (0, "eMBB") else 0
    env.get_slice_prb_budget = lambda _gnb_id, _slice_type: 100
    return env, loads


def test_admission_capacity_uses_absolute_bias_fraction_and_is_enforced():
    env, _loads = _admission_stub()
    bias = np.zeros((3, 3), dtype=float)
    bias[0, 0] = -0.8

    capacities = env.begin_safe_admission_window(bias)
    assert capacities[(0, 1, "EMBB")] == 8
    assert env.get_safe_admission_state()["source_capacities"][(0, "eMBB")] == 8

    ue = SimpleNamespace(prbs=5, useful_prbs=5)
    for _ in range(8):
        assert env._safe_admission_allows(ue, 0, 1, "eMBB")
    assert not env._safe_admission_allows(ue, 0, 1, "eMBB")
    assert env.get_safe_admission_state()["stats"]["rejected_capacity"] == 1


def test_half_release_bias_shares_one_quota_across_both_neighbors():
    env, _loads = _admission_stub()
    bias = np.zeros((3, 3), dtype=float)
    bias[0, 0] = -0.5
    env.begin_safe_admission_window(bias)

    ue = SimpleNamespace(prbs=1, useful_prbs=1)
    for target in (1, 2, 1, 2, 1):
        assert env._safe_admission_allows(ue, 0, target, "eMBB")
    assert not env._safe_admission_allows(ue, 0, 2, "eMBB")
    state = env.get_safe_admission_state()
    assert state["source_capacities"][(0, "eMBB")] == 5
    assert state["source_accepted"][(0, "eMBB")] == 5


def test_float32_bias_does_not_round_admission_quota_up():
    env, _loads = _admission_stub()
    bias = np.zeros((3, 3), dtype=np.float32)
    bias[0, 0] = -0.2

    env.begin_safe_admission_window(bias)

    assert env.get_safe_admission_state()["source_capacities"][(0, "eMBB")] == 2


def test_hard_target_limit_rejects_candidate():
    env, loads = _admission_stub()
    loads[(0, "eMBB")] = 1.0
    loads[(1, "eMBB")] = 0.79
    loads[(2, "eMBB")] = 1.0
    bias = np.zeros((3, 3), dtype=float)
    bias[0, 0] = -1.0
    env.begin_safe_admission_window(bias)

    ue = SimpleNamespace(prbs=5, useful_prbs=5)
    assert not env._safe_admission_allows(ue, 0, 1, "eMBB")
    assert env.get_safe_admission_state()["stats"]["rejected_target_safety"] == 1


def _stability_stub():
    env, _loads = _admission_stub()
    env.max_handovers_per_episode = 20
    env.max_handovers_per_ue_episode = 2
    env.a3_pingpong_guard_steps = 300
    env.a3_emergency_sinr_db = -5.0
    env.disconnect_sinr_db = -100.0
    env._last_ho = {}
    env._last_ho_source = {}
    env._ue_episode_handovers = {}
    env._episode_handover_count = 0
    env._compute_link_metrics = lambda _gnb, _ue: {"sinr_db": 10.0}
    return env


def test_direct_return_is_blocked_inside_pingpong_guard():
    env = _stability_stub()
    ue = SimpleNamespace(id=4)
    serving = SimpleNamespace(id=1)
    env._last_ho[4] = (1, 100)
    env._last_ho_source[4] = 0

    assert not env._handover_stability_allows(ue, 1, 0, serving, 200)
    assert env.get_safe_admission_state()["stats"]["rejected_pingpong_guard"] == 1
    assert env._handover_stability_allows(ue, 1, 2, serving, 200)
    assert env._handover_stability_allows(ue, 1, 0, serving, 401)


def test_per_ue_and_episode_handover_budgets_are_enforced():
    env = _stability_stub()
    ue = SimpleNamespace(id=7)
    serving = SimpleNamespace(id=0)
    env._ue_episode_handovers[7] = 2

    assert not env._handover_stability_allows(ue, 0, 1, serving, 500)
    assert env.get_safe_admission_state()["stats"]["rejected_ue_episode_budget"] == 1

    env._ue_episode_handovers[7] = 0
    env._episode_handover_count = 20
    assert not env._handover_stability_allows(ue, 0, 1, serving, 500)
    assert env.get_safe_admission_state()["stats"]["rejected_episode_budget"] == 1
