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

# UEs placed at the equidistant midpoint between gNB1 and gNB0/gNB2 (medium_270m).
# At x=±135m (with y=±30) each UE is exactly 138.3m from both its serving gNB1
# and the neighbouring outer gNB, giving a 0 dB signal difference.
# A3 condition: RSRP_target > RSRP_serving + offset + hysteresis (1 dB)
#   offset <  −1 dB  (bias < −0.167) → A3 fires
#   offset >= −1 dB                  → A3 blocked
#   offset =  +6 dB                  → A3 completely blocked
# This gives the maximum ±6 dB controllability range of any fixed UE placement.
JAIN_CONTROL_6 = (
    (-135.0, -30.0),
    (135.0, -30.0),
    (-135.0, 0.0),
    (135.0, 0.0),
    (-135.0, 30.0),
    (135.0, 30.0),
)
JAIN_CONTROL_6_UPPER = tuple((x, y + 12.0) for x, y in JAIN_CONTROL_6)
JAIN_CONTROL_6_LOWER = tuple((x, y - 12.0) for x, y in JAIN_CONTROL_6)


# Controllable upper-agent scenarios. The three gap topologies remain independent:
# selecting a topology changes gNB overlap only, not these UE coordinates.
UPPER_TRAINING_SCENARIOS = (
    # Jain-fairness controllable family:
    # UEs sit at the equidistant midpoint between gNB1 and the outer gNBs
    # (x = ±135 m in medium_270m → both distances = 138.3 m, Δ RSRP = 0 dB).
    # Full ±6 dB offset range controls the A3 outcome:
    #   bias > −0.167  (offset > −1 dB) → A3 blocked, UEs stay on gNB1
    #   bias < −0.167  (offset < −1 dB) → A3 fires,   UEs move to outer gNBs
    # All demand starts on gNB1 (Jain = 1/3). A small negative bias produces
    # three symmetric handovers that drive Jain to ≈ 1.0 in a single upper step.
    UpperTrainingScenario(
        "jain_balance_controllable",
        1.0,
        (
            UpperUEGroup(
                "eMBB", 1, 6, 0.90,
                speed_mps=0.0,
                fixed_source_offsets_m=JAIN_CONTROL_6,
                placement_region="overlap",
            ),
        ),
        (
            "Six eMBB UEs at the gNB1/outer equidistant midpoint (±135 m). "
            "Signal difference = 0 dB so A3 fires with any bias below −0.17 and "
            "is fully blocked at +1.0. All demand on gNB1 (Jain = 1/3); the agent "
            "must apply a small negative bias to trigger three outward handovers "
            "and reach Jain ≈ 1.0 — maximum ±6 dB offset controllability."
        ),
    ),
    UpperTrainingScenario(
        "jain_control_urllc",
        1.0,
        (
            UpperUEGroup(
                "URLLC", 1, 6, 0.72,
                speed_mps=0.0,
                fixed_source_offsets_m=JAIN_CONTROL_6_UPPER,
                placement_region="overlap",
            ),
        ),
        (
            "URLLC-only controlled Jain scenario: six midpoint UEs start on "
            "gNB1 with the same A3 controllability as jain_balance_controllable, "
            "but shifted slightly upward so slice placement is distinct."
        ),
    ),
    UpperTrainingScenario(
        "jain_control_mmtc",
        1.0,
        (
            UpperUEGroup(
                "mMTC", 1, 6, 0.60,
                speed_mps=0.0,
                fixed_source_offsets_m=JAIN_CONTROL_6_LOWER,
                placement_region="overlap",
            ),
        ),
        (
            "mMTC-only controlled Jain scenario: six midpoint UEs start on "
            "gNB1 with symmetric left/right handover controllability and a "
            "slightly lower placement band."
        ),
    ),
    UpperTrainingScenario(
        "jain_control_mixed",
        1.0,
        (
            UpperUEGroup(
                "eMBB", 1, 6, 0.36,
                speed_mps=0.0,
                fixed_source_offsets_m=JAIN_CONTROL_6,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "URLLC", 1, 6, 0.30,
                speed_mps=0.0,
                fixed_source_offsets_m=JAIN_CONTROL_6_UPPER,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "mMTC", 1, 6, 0.24,
                speed_mps=0.0,
                fixed_source_offsets_m=JAIN_CONTROL_6_LOWER,
                placement_region="overlap",
            ),
        ),
        (
            "Mixed controlled Jain scenario: eMBB, URLLC, and mMTC each have "
            "six controllable midpoint UEs starting on gNB1. A correct policy "
            "must open both outer directions for all active slices while safe "
            "admission limits the released volume; per-slice loads are kept "
            "below saturation so this scenario tests controllability rather "
            "than raw target capacity."
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
