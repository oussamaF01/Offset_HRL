import numpy as np

from two_neighbor_offset_heuristic import coordinated_neighbor_offsets


def test_negative_bias_splits_pressure_across_two_safe_neighbors():
    offsets = coordinated_neighbor_offsets(
        source_id=0,
        neighbor_ids=(1, 2),
        slice_types=("eMBB",),
        directional_bias={
            (0, 1, "eMBB"): -1.0,
            (0, 2, "eMBB"): -1.0,
        },
        useful_load={
            (0, "eMBB"): 0.90,
            (1, "eMBB"): 0.20,
            (2, "eMBB"): 0.20,
        },
        best_radio_margin_db={
            (1, "eMBB"): 0.0,
            (2, "eMBB"): 0.0,
        },
    )

    assert np.isclose(offsets[(1, "eMBB")], -3.0)
    assert np.isclose(offsets[(2, "eMBB")], -3.0)


def test_negative_bias_prefers_neighbor_with_headroom():
    offsets = coordinated_neighbor_offsets(
        source_id=0,
        neighbor_ids=(1, 2),
        slice_types=("eMBB",),
        directional_bias={
            (0, 1, "eMBB"): -1.0,
            (0, 2, "eMBB"): -1.0,
        },
        useful_load={
            (0, "eMBB"): 0.90,
            (1, "eMBB"): 0.79,
            (2, "eMBB"): 0.20,
        },
        best_radio_margin_db={
            (1, "eMBB"): 0.0,
            (2, "eMBB"): 0.0,
        },
    )

    assert offsets[(1, "eMBB")] == 0.0
    assert offsets[(2, "eMBB")] < 0.0


def test_positive_bias_maps_to_retain_offset():
    offsets = coordinated_neighbor_offsets(
        source_id=0,
        neighbor_ids=(1, 2),
        slice_types=("URLLC",),
        directional_bias={
            (0, 1, "URLLC"): 0.5,
            (0, 2, "URLLC"): 0.0,
        },
        useful_load={
            (0, "URLLC"): 0.30,
            (1, "URLLC"): 0.20,
            (2, "URLLC"): 0.20,
        },
    )

    assert np.isclose(offsets[(1, "URLLC")], 3.0)
    assert offsets[(2, "URLLC")] == 0.0
