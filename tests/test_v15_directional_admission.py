from types import SimpleNamespace

import numpy as np

from multi_gnb_wrapper import MultiGNBWrapper
from safe_admission_controller import SafeAdmissionController
from strong_heuristic_local_executor import (
    EXTENDED_1DB_OFFSET_SET_DB,
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
    assert set(np.unique(offsets)).issubset(set(EXTENDED_1DB_OFFSET_SET_DB))


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
    assert np.all(neutral[0, :, 0] >= 0.0)


def test_directional_executor_negative_bias_is_capped_at_minus_6_db():
    inputs = _directional_inputs()
    inputs["B"] = np.zeros((3, 2, 3), dtype=float)
    inputs["B"][0, 0, 0] = -1.0
    inputs["B"][0, 1, 0] = 1.0

    offsets = strong_directional_heuristic_local_executor(**inputs)

    assert offsets[0, 0, 0] == -6.0
    assert offsets[0, 1, 0] >= 0.0
    assert np.min(offsets) >= -6.0


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
    env.safe_admission_controller = SafeAdmissionController(bias_deadband=0.05)
    env.safe_admission_load_provider = None
    env.neutralize_offsets_when_quota_exhausted = False
    env.max_handovers_per_episode = 20
    env._episode_handover_count = 0

    loads = {
        (0, "eMBB"): 0.90,
        (1, "eMBB"): 0.40,
        (2, "eMBB"): 0.50,
        **{(g, s): 0.0 for g in range(3) for s in ("URLLC", "mMTC")},
    }
    env.get_slice_loads = lambda: dict(loads)
    env.get_slice_ue_count = lambda gnb_id, slice_type: 10 if (gnb_id, slice_type) == (0, "eMBB") else 0
    env.get_slice_prb_budget = lambda _gnb_id, _slice_type: 100
    env.get_slice_sla_severity = lambda: {
        (g, s): 0.0 for g in range(3) for s in ("eMBB", "URLLC", "mMTC")
    }
    return env, loads


def _directional_bias(
    value: float,
    *,
    source: int = 0,
    target_slot: int = 0,
    slice_idx: int = 0,
    dtype=float,
):
    bias = np.zeros((3, 2, 3), dtype=dtype)
    bias[source, target_slot, slice_idx] = value
    return bias


def test_admission_quota_uses_bias_times_source_ues_and_is_enforced():
    env, _loads = _admission_stub()
    bias = _directional_bias(-0.8)

    capacities = env.begin_safe_admission_window(bias)
    assert capacities[(0, 1, "EMBB")] == 8
    assert env.get_safe_admission_state()["source_capacities"][(0, "eMBB")] == 8

    ue = SimpleNamespace(id=1, prbs=5, useful_prbs=5)
    for _ in range(8):
        assert env._safe_admission_allows(ue, 0, 1, "eMBB")
        env._commit_safe_admission(0, 1, "eMBB")
    assert not env._safe_admission_allows(ue, 0, 1, "eMBB")
    assert env.get_safe_admission_state()["stats"]["rejected_direction_quota_exhausted"] == 1


def test_half_release_bias_builds_independent_directional_quotas():
    env, _loads = _admission_stub()
    bias = np.zeros((3, 2, 3), dtype=float)
    bias[0, :, 0] = -0.25
    env.begin_safe_admission_window(bias)

    ue = SimpleNamespace(id=2, prbs=1, useful_prbs=1)
    for _ in range(3):
        assert env._safe_admission_allows(ue, 0, 1, "eMBB")
        env._commit_safe_admission(0, 1, "eMBB")
    assert not env._safe_admission_allows(ue, 0, 1, "eMBB")
    for _ in range(3):
        assert env._safe_admission_allows(ue, 0, 2, "eMBB")
        env._commit_safe_admission(0, 2, "eMBB")
    assert not env._safe_admission_allows(ue, 0, 2, "eMBB")
    state = env.get_safe_admission_state()
    assert state["capacities"][(0, 1, "EMBB")] == 3
    assert state["capacities"][(0, 2, "EMBB")] == 3
    assert state["source_capacities"][(0, "eMBB")] == 6
    assert state["source_accepted"][(0, "eMBB")] == 6


def test_float32_bias_does_not_round_admission_quota_up():
    env, _loads = _admission_stub()
    bias = _directional_bias(-0.2, dtype=np.float32)

    env.begin_safe_admission_window(bias)

    assert env.get_safe_admission_state()["source_capacities"][(0, "eMBB")] == 2


def test_target_load_does_not_veto_budgeted_candidate():
    env, loads = _admission_stub()
    loads[(0, "eMBB")] = 1.0
    loads[(1, "eMBB")] = 0.79
    loads[(2, "eMBB")] = 1.0
    bias = _directional_bias(-1.0)
    env.begin_safe_admission_window(bias)

    ue = SimpleNamespace(id=3, prbs=5, useful_prbs=5)
    assert env._safe_admission_allows(ue, 0, 1, "eMBB")
    assert env.get_safe_admission_state()["stats"]["rejected_target_safety"] == 0


def test_negative_bias_has_budget_even_without_source_excess():
    env, loads = _admission_stub()
    loads[(0, "eMBB")] = 0.30
    loads[(1, "eMBB")] = 0.10
    loads[(2, "eMBB")] = 0.90
    bias = _directional_bias(-0.5)
    env.begin_safe_admission_window(bias)
    ue = SimpleNamespace(id=4, prbs=1, useful_prbs=1)

    assert env.get_safe_admission_state()["source_capacities"][(0, "eMBB")] == 5
    assert env._safe_admission_allows(ue, 0, 1, "eMBB")
    assert env.get_safe_admission_state()["stats"]["rejected_no_source_excess"] == 0


def test_overloaded_source_cannot_release_without_negative_upper_bias():
    env, _loads = _admission_stub()
    ue = SimpleNamespace(id=40, prbs=1, useful_prbs=1)

    for source_bias in (0.0, 0.8):
        bias = _directional_bias(source_bias)
        env.begin_safe_admission_window(bias)

        state = env.get_safe_admission_state()
        assert state["requested_release_load"][(0, "eMBB")] == 0.0
        assert state["source_capacities"][(0, "eMBB")] == 0
        assert not env._safe_admission_allows(ue, 0, 1, "eMBB")
        assert (
            env.get_safe_admission_state()["stats"][
                "rejected_no_directional_budget"
            ]
            == 1
        )


def test_failed_handover_does_not_consume_admission_quota():
    env, _loads = _admission_stub()
    bias = _directional_bias(-0.1)
    env.begin_safe_admission_window(bias)
    ue = SimpleNamespace(id=5, prbs=1, useful_prbs=1)

    assert env._safe_admission_allows(ue, 0, 1, "eMBB")
    # Simulate _perform_handover() returning False: no commit occurs.
    state = env.get_safe_admission_state()
    assert state["source_capacities"][(0, "eMBB")] == 1
    assert state["source_accepted"][(0, "eMBB")] == 0
    assert state["stats"]["accepted"] == 0
    assert env._safe_admission_allows(ue, 0, 1, "eMBB")


def test_successful_handover_commit_consumes_exactly_one_slot():
    env, _loads = _admission_stub()
    bias = _directional_bias(-0.2)
    env.begin_safe_admission_window(bias)
    ue = SimpleNamespace(id=6, prbs=1, useful_prbs=1)

    assert env._safe_admission_allows(ue, 0, 1, "eMBB")
    env._commit_safe_admission(0, 1, "eMBB")

    state = env.get_safe_admission_state()
    assert state["accepted"][(0, 1, "EMBB")] == 1
    assert state["source_accepted"][(0, "eMBB")] == 1
    assert state["stats"]["accepted"] == 1


def test_new_upper_window_recomputes_and_resets_used_quota():
    env, loads = _admission_stub()
    bias = _directional_bias(-1.0)
    env.begin_safe_admission_window(bias)
    first_quota = env.get_safe_admission_state()["quota"][(0, "eMBB")]
    env._commit_safe_admission(0, 1, "eMBB")
    assert env.get_safe_admission_state()["used"][(0, "eMBB")] == 1

    loads[(0, "eMBB")] = 0.60
    loads[(1, "eMBB")] = 0.60
    loads[(2, "eMBB")] = 0.60
    env.begin_safe_admission_window(bias)
    state = env.get_safe_admission_state()

    assert first_quota > 0
    assert state["used"][(0, "eMBB")] == 0
    assert state["quota"][(0, "eMBB")] == first_quota
    assert state["window_id"] == 2


def test_controller_ranks_candidates_and_clamps_to_remaining_quota():
    controller = SafeAdmissionController(bias_deadband=0.05)
    controller.begin_upper_window(
        directional_bias=np.asarray([[[-0.5]]], dtype=float),
        neighbor_graph={0: [1]},
        gnb_ids=[0],
        slice_types=["eMBB"],
        loads={(0, "eMBB"): 0.90},
        ue_counts={(0, "eMBB"): 4},
        balance_targets={"eMBB": 0.60},
    )
    candidates = [
        {
            "ue_id": 10,
            "source_id": 0,
            "target_id": 1,
            "slice_type": "eMBB",
            "a3_margin": 2.0,
            "target_load": 0.30,
            "target_load_increment": 0.05,
            "target_safe_limit": 0.80,
        },
        {
            "ue_id": 11,
            "source_id": 0,
            "target_id": 1,
            "slice_type": "eMBB",
            "a3_margin": 8.0,
            "target_load": 0.30,
            "target_load_increment": 0.05,
            "target_safe_limit": 0.80,
        },
        {
            "ue_id": 12,
            "source_id": 0,
            "target_id": 1,
            "slice_type": "eMBB",
            "a3_margin": 5.0,
            "target_load": 0.30,
            "target_load_increment": 0.05,
            "target_safe_limit": 0.80,
        },
    ]

    accepted, rejected, debug = controller.admit_candidates(candidates)

    assert [item["ue_id"] for item in accepted] == [11, 12]
    assert rejected[0]["ue_id"] == 10
    assert rejected[0]["rejection_reason"] == "direction_quota_exhausted"
    assert debug["rejection_reasons"] == {"direction_quota_exhausted": 1}


def test_controller_ignores_target_slice_safety_for_budgeted_candidate():
    controller = SafeAdmissionController()
    controller.begin_upper_window(
        directional_bias=np.asarray([[[-1.0]]], dtype=float),
        neighbor_graph={0: [1]},
        gnb_ids=[0],
        slice_types=["eMBB"],
        loads={(0, "eMBB"): 0.90},
        ue_counts={(0, "eMBB"): 10},
        balance_targets={"eMBB": 0.60},
    )
    candidate = {
        "ue_id": 20,
        "source_id": 0,
        "target_id": 1,
        "slice_type": "eMBB",
        "a3_margin": 10.0,
        "target_load": 0.79,
        "target_load_increment": 0.05,
        "target_safe_limit": 0.80,
    }

    accepted, rejected, debug = controller.admit_candidates([candidate])

    assert [item["ue_id"] for item in accepted] == [20]
    assert rejected == []
    assert debug["rejection_reasons"] == {}


def test_controller_ignores_target_total_safety_for_budgeted_candidate():
    controller = SafeAdmissionController()
    controller.begin_upper_window(
        directional_bias=np.asarray([[[-1.0]]], dtype=float),
        neighbor_graph={0: [1]},
        gnb_ids=[0],
        slice_types=["eMBB"],
        loads={(0, "eMBB"): 0.90},
        ue_counts={(0, "eMBB"): 6},
        balance_targets={"eMBB": 0.30},
    )
    candidate = {
        "ue_id": 21,
        "source_id": 0,
        "target_id": 1,
        "slice_type": "eMBB",
        "target_load": 0.20,
        "target_total_load": 0.95,
        "target_load_increment": 0.15,
        "target_safe_limit": 0.80,
        "target_total_safe_limit": 1.0,
    }

    accepted, rejected, debug = controller.admit_candidates([candidate])

    assert [item["ue_id"] for item in accepted] == [21]
    assert rejected == []
    assert debug["rejection_reasons"] == {}


def test_directional_quota_only_allows_the_requested_target():
    controller = SafeAdmissionController()
    directional_bias = np.asarray([[[-1.0], [1.0]]], dtype=float)
    controller.begin_upper_window(
        directional_bias=directional_bias,
        neighbor_graph={0: [1, 2]},
        gnb_ids=[0],
        slice_types=["eMBB"],
        loads={(0, "eMBB"): 0.90},
        ue_counts={(0, "eMBB"): 6},
        balance_targets={"eMBB": 0.30},
    )
    base = {
        "source_id": 0,
        "slice_type": "eMBB",
        "target_load": 0.0,
        "target_load_increment": 0.15,
        "target_safe_limit": 0.80,
    }
    candidates = [
        {**base, "ue_id": 1, "target_id": 1, "a3_margin": 2.0},
        {**base, "ue_id": 2, "target_id": 2, "a3_margin": 10.0},
    ]

    accepted, rejected, _debug = controller.admit_candidates(candidates)

    assert [item["target_id"] for item in accepted] == [1]
    assert rejected[0]["target_id"] == 2
    assert rejected[0]["rejection_reason"] == "no_directional_budget"
    state = controller.get_state()
    assert state["direction_quota"][(0, 1, "eMBB")] > 0
    assert state["direction_quota"][(0, 2, "eMBB")] == 0


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
