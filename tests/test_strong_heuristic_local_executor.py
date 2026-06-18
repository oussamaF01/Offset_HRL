import numpy as np

from strong_heuristic_local_executor import (
    OFFSET_SET_DB,
    evaluate_candidate_offset,
    strong_heuristic_local_executor,
)


def _base_inputs():
    return {
        "B": np.zeros((3, 3), dtype=float),
        "prev_offsets": np.zeros((3, 3), dtype=float),
        "ue_slice": np.asarray([0, 0], dtype=int),
        "ue_serving_gnb": np.asarray([0, 0], dtype=int),
        "rsrp_matrix": np.asarray(
            [
                [-80.0, -79.0, -95.0],
                [-82.0, -80.5, -96.0],
            ],
            dtype=float,
        ),
        "neighbor_graph": {0: [1, 2], 1: [0, 2], 2: [0, 1]},
        "load": np.zeros((3, 3), dtype=float),
        "sla_violation": np.zeros((3, 3), dtype=float),
        "ho_failure_ratio": np.zeros((3, 3), dtype=float),
        "pingpong_ratio": np.zeros((3, 3), dtype=float),
    }


def test_outputs_are_valid_discrete_offsets():
    kwargs = _base_inputs()
    kwargs["B"][0, 0] = -1.0
    offsets = strong_heuristic_local_executor(**kwargs)

    assert offsets.shape == (3, 3)
    assert set(np.unique(offsets)).issubset(set(OFFSET_SET_DB))


def test_accepts_string_slice_labels():
    kwargs = _base_inputs()
    kwargs["B"][0, 0] = -1.0
    kwargs["ue_slice"] = np.asarray(["eMBB", "eMBB"])
    offsets = strong_heuristic_local_executor(**kwargs)

    assert offsets[0, 0] < 0.0


def test_negative_bias_rewards_feasible_handover():
    kwargs = _base_inputs()
    kwargs["B"][0, 0] = -1.0
    offsets, debug = strong_heuristic_local_executor(**kwargs, return_debug=True)

    assert offsets[0, 0] < 0.0
    selected = debug[(0, 0)]["candidates"][int(np.where(OFFSET_SET_DB == offsets[0, 0])[0][0])]
    assert selected["n_predicted_handovers"] > 0


def test_positive_bias_prefers_retain_offset():
    kwargs = _base_inputs()
    kwargs["B"][0, 0] = 1.0
    offsets = strong_heuristic_local_executor(**kwargs)

    assert offsets[0, 0] >= 0.0


def test_target_overload_penalizes_predicted_handover_score():
    safe = _base_inputs()
    safe["B"][0, 0] = -1.0
    overloaded = _base_inputs()
    overloaded["B"][0, 0] = -1.0
    overloaded["load"][1, 0] = 1.0

    safe_score, safe_terms = evaluate_candidate_offset(
        gnb_idx=0,
        slice_idx=0,
        candidate_offset_db=-6.0,
        **safe,
    )
    overloaded_score, overloaded_terms = evaluate_candidate_offset(
        gnb_idx=0,
        slice_idx=0,
        candidate_offset_db=-6.0,
        **overloaded,
    )

    assert overloaded_terms["r_target"] > safe_terms["r_target"]
    assert overloaded_score < safe_score


def test_no_ues_preserves_strong_bias_conservatively():
    kwargs = _base_inputs()
    kwargs["B"][0, 0] = -1.0
    kwargs["ue_slice"] = np.asarray([], dtype=int)
    kwargs["ue_serving_gnb"] = np.asarray([], dtype=int)
    kwargs["rsrp_matrix"] = np.zeros((0, 3), dtype=float)
    offsets = strong_heuristic_local_executor(**kwargs)

    assert offsets[0, 0] < 0.0
    assert abs(offsets[0, 0]) <= 4.0


def test_no_ues_with_neutral_bias_stays_neutral():
    kwargs = _base_inputs()
    kwargs["ue_slice"] = np.asarray([], dtype=int)
    kwargs["ue_serving_gnb"] = np.asarray([], dtype=int)
    kwargs["rsrp_matrix"] = np.zeros((0, 3), dtype=float)
    offsets = strong_heuristic_local_executor(**kwargs)

    assert np.all(offsets == 0.0)


def test_candidate_helper_exposes_score_terms():
    kwargs = _base_inputs()
    kwargs["B"][0, 0] = -1.0
    score, terms = evaluate_candidate_offset(
        gnb_idx=0,
        slice_idx=0,
        candidate_offset_db=-2.0,
        **kwargs,
    )

    assert isinstance(score, float)
    assert terms["ho_frac"] > 0.0
    assert {"a_bias", "a_handover", "a_sla", "r_risk", "r_target", "r_osc"}.issubset(terms)
