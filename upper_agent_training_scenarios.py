#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


def center_left_right_gnb_configs(
    center_gap_m: float,
    coverage_radius_m: float = 500.0,
) -> Tuple[dict, ...]:
    """Return a collinear left-center-right topology."""
    gap = float(center_gap_m)
    radius = float(coverage_radius_m)
    return tuple(
        {
            "id": gnb_id,
            "x": x,
            "y": 0.0,
            "coverage_radius": radius,
            "carrier_id": 0,
            "center_frequency_hz": 3.5e9,
            "bandwidth_hz": 20e6,
            "tx_power_dbm": 30.0,
            "noise_figure_db": 7.0,
        }
        for gnb_id, x in enumerate((-gap, 0.0, gap))
    )


# The only supported upper-training topologies. UE coordinates remain fixed;
# only the center-to-outer-gNB gap changes.
CENTER_GAP_GNB_CONFIGS = {
    "tight_220m": center_left_right_gnb_configs(220.0),
    "medium_270m": center_left_right_gnb_configs(270.0),
    "wide_320m": center_left_right_gnb_configs(320.0),
}
CENTER_LEFT_RIGHT_GNB_CONFIGS = CENTER_GAP_GNB_CONFIGS["medium_270m"]


@dataclass(frozen=True)
class UpperUEGroup:
    slice_type: str
    source_gnb: int
    count: int
    total_load: float
    target_gnb: int | None = None
    speed_mps: float = 0.0
    path_progress: float = 0.22
    lateral_offset_m: float = 0.0
    placement_target_gnbs: Tuple[int, ...] = ()
    fixed_source_offsets_m: Tuple[Tuple[float, float], ...] = ()
    placement_region: str = "overlap"


@dataclass(frozen=True)
class UpperTrainingScenario:
    name: str
    duration_s: float
    groups: Tuple[UpperUEGroup, ...]
    description: str
    tier: str = "fixed"


OVERLAP_LEFT_RIGHT_6 = (
    (-165.0, -30.0),
    (165.0, -30.0),
    (-35.0, -35.0),
    (35.0, 35.0),
    (-165.0, 30.0),
    (165.0, 30.0),
)
OVERLAP_LEFT_RIGHT_4 = (
    (-165.0, -30.0),
    (165.0, -30.0),
    (-165.0, 30.0),
    (165.0, 30.0),
)
LEFT_FIXED_CORE_2 = ((-300.0, -35.0), (-300.0, 35.0))
RIGHT_FIXED_CORE_2 = ((300.0, -35.0), (300.0, 35.0))

# UEs at ±132 m from center gNB on the x-axis. In medium_270m, neutral A3
# remains inactive, while the symmetric ±6 dB directional mapping begins
# producing candidates around bias -0.3. This placement gives PPO a useful
# progression: neutral/small bias -> 0 HOs, moderate bias -> partial release,
# stronger bias -> the full safe-admission quota.
CENTER_INNER_6 = (
    (-132.0, -30.0),
    (132.0, -30.0),
    (-132.0, 0.0),
    (132.0, 0.0),
    (-132.0, 30.0),
    (132.0, 30.0),
)
CENTER_INNER_4 = (
    (-132.0, -30.0),
    (132.0, -30.0),
    (-132.0, 30.0),
    (132.0, 30.0),
)


# Slice-aware traffic scenarios. The three gap topologies remain independent:
# selecting a topology changes gNB overlap only, not these UE coordinates.
UPPER_TRAINING_SCENARIOS = (
    UpperTrainingScenario(
        "paper_six_ue_slice_aware",
        20.0,
        (
            UpperUEGroup(
                "eMBB", 1, 2, 0.40,
                fixed_source_offsets_m=((-165.0, -45.0), (165.0, -45.0)),
                placement_region="overlap",
            ),
            UpperUEGroup(
                "URLLC", 1, 2, 0.40,
                fixed_source_offsets_m=((-165.0, 0.0), (165.0, 0.0)),
                placement_region="overlap",
            ),
            UpperUEGroup(
                "mMTC", 1, 2, 0.40,
                fixed_source_offsets_m=((-165.0, 45.0), (165.0, 45.0)),
                placement_region="overlap",
            ),
        ),
        (
            "Paper-style six-UE topology made slice-aware: one eMBB, one "
            "URLLC, and one mMTC UE in each left/right overlap cluster."
        ),
    ),
    UpperTrainingScenario(
        "fixed_center_embb_left_right",
        20.0,
        (
            UpperUEGroup(
                "eMBB",
                1,
                6,
                0.90,
                speed_mps=0.0,
                fixed_source_offsets_m=OVERLAP_LEFT_RIGHT_6,
                placement_region="overlap",
            ),
        ),
        (
            "Six fixed eMBB UEs remain attached initially to center gNB1; "
            "the UE placement and load stay identical while the selected "
            "left-center-right topology changes the overlap gap."
        ),
    ),
    UpperTrainingScenario(
        "mixed_slices_center_overlap",
        20.0,
        (
            UpperUEGroup(
                "eMBB", 1, 4, 0.72,
                fixed_source_offsets_m=OVERLAP_LEFT_RIGHT_4,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "URLLC", 1, 4, 0.68,
                fixed_source_offsets_m=tuple(
                    (x, y + 12.0) for x, y in OVERLAP_LEFT_RIGHT_4
                ),
                placement_region="overlap",
            ),
            UpperUEGroup(
                "mMTC", 1, 6, 0.60,
                fixed_source_offsets_m=tuple(
                    (x, y - 12.0) for x, y in OVERLAP_LEFT_RIGHT_6
                ),
                placement_region="overlap",
            ),
        ),
        "All three slices are mixed in the center-cell overlap regions.",
    ),
    UpperTrainingScenario(
        "embb_overlap_preloaded_targets",
        20.0,
        (
            UpperUEGroup(
                "eMBB", 1, 6, 0.90,
                fixed_source_offsets_m=OVERLAP_LEFT_RIGHT_6,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "eMBB", 0, 2, 0.30,
                fixed_source_offsets_m=LEFT_FIXED_CORE_2,
                placement_region="fixed_core",
            ),
            UpperUEGroup(
                "eMBB", 2, 2, 0.30,
                fixed_source_offsets_m=RIGHT_FIXED_CORE_2,
                placement_region="fixed_core",
            ),
        ),
        "Center eMBB overlap traffic sees persistent eMBB load in both outer-cell cores.",
    ),
    UpperTrainingScenario(
        "asymmetric_embb_target_loads",
        20.0,
        (
            UpperUEGroup(
                "eMBB", 1, 6, 0.90,
                fixed_source_offsets_m=OVERLAP_LEFT_RIGHT_6,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "eMBB", 0, 3, 0.54,
                fixed_source_offsets_m=(
                    (-300.0, -55.0),
                    (-300.0, 0.0),
                    (-300.0, 55.0),
                ),
                placement_region="fixed_core",
            ),
            UpperUEGroup(
                "eMBB", 2, 1, 0.12,
                fixed_source_offsets_m=((300.0, 0.0),),
                placement_region="fixed_core",
            ),
        ),
        "Asymmetric fixed eMBB target loads teach directional neighbor preference.",
    ),
    UpperTrainingScenario(
        "urllc_mmtc_overlap_fixed_embb",
        20.0,
        (
            UpperUEGroup(
                "URLLC", 1, 4, 0.76,
                fixed_source_offsets_m=OVERLAP_LEFT_RIGHT_4,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "mMTC", 1, 6, 0.66,
                fixed_source_offsets_m=tuple(
                    (x, y + 15.0) for x, y in OVERLAP_LEFT_RIGHT_6
                ),
                placement_region="overlap",
            ),
            UpperUEGroup(
                "eMBB", 0, 2, 0.40,
                fixed_source_offsets_m=LEFT_FIXED_CORE_2,
                placement_region="fixed_core",
            ),
            UpperUEGroup(
                "eMBB", 2, 2, 0.36,
                fixed_source_offsets_m=RIGHT_FIXED_CORE_2,
                placement_region="fixed_core",
            ),
        ),
        "URLLC and mMTC overlap traffic coexist with fixed eMBB background load.",
    ),
    UpperTrainingScenario(
        "mixed_overlap_with_fixed_slice_loads",
        20.0,
        (
            UpperUEGroup(
                "eMBB", 1, 4, 0.72,
                fixed_source_offsets_m=OVERLAP_LEFT_RIGHT_4,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "URLLC", 1, 4, 0.70,
                fixed_source_offsets_m=tuple(
                    (x, y + 15.0) for x, y in OVERLAP_LEFT_RIGHT_4
                ),
                placement_region="overlap",
            ),
            UpperUEGroup(
                "mMTC", 0, 2, 0.36,
                fixed_source_offsets_m=LEFT_FIXED_CORE_2,
                placement_region="fixed_core",
            ),
            UpperUEGroup(
                "URLLC", 2, 2, 0.34,
                fixed_source_offsets_m=RIGHT_FIXED_CORE_2,
                placement_region="fixed_core",
            ),
        ),
        "Mixed overlap UEs operate beside fixed non-overlap loads of other slices.",
    ),
    # ── Inner-zone scenarios ──────────────────────────────────────────────────
    # UEs placed at ±132 m from center gNB (medium_270m), just inside the
    # natural-A3 threshold. Neutral bias triggers no HO; moderate directional
    # bias around -0.3 exposes candidates and safe admission controls volume.
    UpperTrainingScenario(
        "high_load_inner_embb",
        1.0,
        (
            UpperUEGroup(
                "eMBB",
                1,
                6,
                0.90,
                speed_mps=0.0,
                fixed_source_offsets_m=CENTER_INNER_6,
                placement_region="overlap",
            ),
        ),
        (
            "Six eMBB UEs at ±132 m from center gNB. Neutral A3 stays inactive, "
            "while moderate negative directional bias creates handover "
            "candidates and safe admission controls the released volume in one "
            "upper decision."
        ),
    ),
    UpperTrainingScenario(
        "high_load_inner_mixed",
        1.0,
        (
            UpperUEGroup(
                "eMBB", 1, 4, 0.72,
                fixed_source_offsets_m=CENTER_INNER_4,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "URLLC", 1, 4, 0.68,
                fixed_source_offsets_m=tuple(
                    (x, y + 12.0) for x, y in CENTER_INNER_4
                ),
                placement_region="overlap",
            ),
            UpperUEGroup(
                "mMTC", 1, 6, 0.60,
                fixed_source_offsets_m=tuple(
                    (x, y - 12.0) for x, y in CENTER_INNER_6
                ),
                placement_region="overlap",
            ),
        ),
        (
            "All three slices heavily loaded at center gNB, UEs at ±132 m "
            "inner zone. Agent must apply negative biases across all slices "
            "to reduce multi-slice congestion in one upper decision."
        ),
    ),
    UpperTrainingScenario(
        "high_load_inner_asymmetric",
        1.0,
        (
            UpperUEGroup(
                "eMBB", 1, 6, 0.90,
                fixed_source_offsets_m=CENTER_INNER_6,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "eMBB", 0, 3, 0.54,
                fixed_source_offsets_m=(
                    (-300.0, -55.0),
                    (-300.0, 0.0),
                    (-300.0, 55.0),
                ),
                placement_region="fixed_core",
            ),
            UpperUEGroup(
                "eMBB", 2, 1, 0.12,
                fixed_source_offsets_m=((300.0, 0.0),),
                placement_region="fixed_core",
            ),
        ),
        (
            "Asymmetric neighbor loads with center UEs in inner zone.  Left "
            "gNB is pre-loaded to 0.54, right is near-empty at 0.12 — agent "
            "must learn directional preference toward gNB-2 in one upper step."
        ),
    ),
)

UPPER_TRAINING_SCENARIO_BY_NAME = {
    scenario.name: scenario for scenario in UPPER_TRAINING_SCENARIOS
}


def get_upper_training_scenarios(
    names=None,
) -> tuple[UpperTrainingScenario, ...]:
    if names is None or names == "all":
        return UPPER_TRAINING_SCENARIOS
    requested = (
        tuple(part.strip() for part in names.split(",") if part.strip())
        if isinstance(names, str)
        else tuple(names)
    )
    unknown = [
        name
        for name in requested
        if name not in UPPER_TRAINING_SCENARIO_BY_NAME
    ]
    if unknown:
        known = ", ".join(UPPER_TRAINING_SCENARIO_BY_NAME)
        raise ValueError(
            f"Unknown upper scenarios {unknown}. Known: {known}"
        )
    return tuple(
        UPPER_TRAINING_SCENARIO_BY_NAME[name]
        for name in requested
    )
